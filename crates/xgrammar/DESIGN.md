<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Design note — the Tier-3 constrained-decoding synthesis

This crate is a pure-Rust port of XGrammar's grammar-constrained decoding
engine. Beyond a faithful port it carries a stack of performance tiers; the
**novel contribution** is not any single tier but the **combination of all of
them in one engine**. This note records that synthesis and, in particular, how
the three Tier-3 techniques compose.

No prior constrained-decoding engine — C++ XGrammar-2 included — has all of:

* a **JIT-compiled, CSR-backed Earley engine** (lazy per-state mask compilation
  over a compact CSR FSM),
* **dynamic dead-state pruning** of the per-token search (ZapFormat — Tier 3a),
* **forced-token elision** on the hot path (Coalescence — Tier 3b),
* **compile-time static/dynamic decomposition** of the grammar
  (WGRAMMAR — Tier 3c).

Each tier is individually known in the literature. Putting all four in a single
`unsafe`-free engine, each composing cleanly with the next rather than
duplicating its work, is the contribution.

---

## The tier stack

| Tier | Technique | Where | What it does |
|------|-----------|-------|--------------|
| 1 | CSR FSM + Earley port | `fsm/`, `earley/` | Compact-array FSM-accelerated Earley recognizer. |
| 2 | JIT cache + cross-grammar `RuleLevelCache` | `compiler/compile.rs`, `compiler/rule_cache.rs` | Per-state `AdaptiveTokenMask` computed lazily on first touch and memoized; structurally-identical rules reuse masks across grammars. |
| 3a | ZapFormat dynamic dead-state pruning | `earley/`, `matcher/` | During the per-token scan, parser states with no surviving continuation are pruned so they cost nothing on later steps. |
| 3b | Coalescence forced-token elision | `compiler/coalesce.rs`, `matcher/coalesce.rs` | When the final next-token bitmask has exactly one set bit the token is *forced*; the model sample is skipped. `accept_forced_chain` walks a whole forced run at once. |
| 3c | WGRAMMAR static/dynamic decomposition | `compiler/decompose.rs` | At **compile time**, classifies every rule body into fixed-literal *scaffolding* segments (bytes precomputed) and dynamic *value-slot* segments. |

Tiers 1, 2, 3a, 3b predate this note; Tier 3c (`decompose.rs`) is added here.

---

## How the three Tier-3 techniques compose

The Tier-3 techniques attack the same observation — *a tool-call JSON-schema
grammar is overwhelmingly fixed structure* — from three different angles, at
three different times, and they are deliberately built so none repeats another's
work.

### 3a — ZapFormat: dynamic pruning (per-token, decode)

ZapFormat operates **inside one decode step**. As the matcher computes the
next-token mask it walks live Earley/FSM states; states that can lead nowhere
are dropped. This shrinks the *search*, not the *grammar*. It is purely
dynamic — it reacts to the token actually sampled — and it owns no precomputed
state.

### 3b — Coalescence: forced-token elision (per-token, decode)

Coalescence operates **between decode steps**. After the mask is computed
(`compute_partitions` → `set_token_bitmask`), `analyze_bitmask` asks one
question: does the mask have exactly one bit? If so the token is determined and
the GPU sample is skipped. `next_forced_tokens` / `accept_forced_chain` chase a
*chain* of such forced positions.

Crucially, 3b **discovers** forcedness lazily — one state at a time, on the hot
path, the first time the matcher reaches that state. It does not know in advance
which spans of the output are forced; it finds out as it goes.

### 3c — WGRAMMAR: static/dynamic decomposition (compile time)

WGRAMMAR operates **once, at compile time, over the whole grammar**. The
`decompose_static_regions` pass walks the optimized AST and splits every rule
body into:

* `Segment::Static { bytes }` — a run of consecutive `ByteString` literals: the
  fixed scaffolding (`{`, the property keys, `:`, `,`, `}`). Its bytes are
  **precomputed and stored** on the `CompiledGrammar`.
* `Segment::Dynamic` — a genuine value slot (`CharacterClass`,
  `CharacterClassStar`, number/string sub-grammar, `Repeat`, `TagDispatch`).
* `Segment::Choice { branches }` — a divergence point; each branch is itself
  decomposed recursively, so per-branch scaffolding (the fixed `{"key":` prefix
  of a JSON object's non-empty branch) is still surfaced as `Static` spans.

The result, `GrammarDecomposition`, is the compile-time **index of the
scaffolding**: before the first decode step the engine already knows which byte
spans of the output are fixed and what they are.

### The composition — what is genuinely new in 3c

Here is the honest accounting, because 3b and 3c are closely related and it
would be easy to write a redundant second mechanism.

* The **per-token masks** are owned by Tier 2 (the JIT cache). 3c does **not**
  recompute or cache masks — `decompose.rs` produces *byte sequences*, never
  `AdaptiveTokenMask`s.
* The **forced-token decisions at decode** are owned by Tier 3b
  (`analyze_bitmask` over the authoritative mask). 3c does **not** make
  forced-token decisions at decode — the matcher's hot path is untouched.
* A `Segment::Static` span **is exactly** a forced-token chain — the same chain
  Tier 3b's `accept_forced_chain` would walk. The crate's tests assert this
  byte-for-byte (`precomputed_static_bytes_match_byte_forced_path`): the
  compile-time precompute equals the lazy byte-forced path.

So what does 3c actually add? **Time of knowledge, not a new mechanism.** Tier
3b answers "is *this state* forced?" lazily, one state at a time, on the hot
path. WGRAMMAR's genuine delta is to answer, **at compile time and for the whole
grammar at once**, "which spans of which rule bodies are nothing but a fixed
literal?" — and to precompute their literal bytes then. The static structure of
a schema is identical for every request against that schema; discovering it once
at compile time, rather than rediscovering it token-by-token on every request,
is the WGRAMMAR insight (arXiv:2507.16768). The decomposition is stored on the
`CompiledGrammar` and exposed via `CompiledGrammar::decomposition()`, so a
scheduler can consult the static/dynamic split *before* decoding begins.

In short: **3a shrinks the search, 3b skips the sample, 3c moves the
classification of "what is scaffolding" from decode time to compile time.** They
stack — a static segment (known by 3c) is elided by 3b while 3a keeps the
remaining dynamic search tight — and none of them duplicates another's state.

### What was deliberately *not* built

An earlier draft of 3c cached per-segment token masks. That was discarded as a
redundant parallel mechanism: Tier 2 already owns mask caching, and a static
segment's mask is, by Tier 3b's contract, just a single-bit mask the matcher
will compute anyway. Re-deriving and re-storing it under a second key would
violate SSOT. `decompose.rs` therefore stops at the byte sequence — the one
piece of information Tiers 2 and 3b genuinely do not have ahead of time.

---

## Honest limits of 3c

* The decomposition is a **grammar property**; it is computed even on the
  empty-vocabulary degenerate path, and it is deterministic across compiles.
* Optional whitespace (`any_whitespace = true`) inserts `CharacterClassStar`
  elements *between* literals, which break literal runs. With `any_whitespace =
  false` (fixed JSON separators — the realistic tool-call setup) far more of the
  body coalesces into long static segments. The decomposition reports whatever
  the optimized grammar actually contains; it does not rewrite the grammar.
* 3c does not change decode-time behavior at all. It is additive and
  observational: 791 pre-existing tests are unchanged, and the new tests assert
  the precompute equals the lazy path. Its payoff is realized by a *consumer*
  (a scheduler) that uses `decomposition()` to plan ahead — e.g. to emit a whole
  static segment without entering the per-token loop.

---

## File map (Tier-3)

```
src/compiler/
  coalesce.rs    Tier 3b — forced-token bitmask analysis (analyze_bitmask)
  decompose.rs   Tier 3c — WGRAMMAR static/dynamic decomposition  ← new
  compile.rs     runs decompose_static_regions() once per compile
  compiled_grammar.rs  stores GrammarDecomposition; decomposition() accessor
src/matcher/
  coalesce.rs    Tier 3b — forced_token / next_forced_tokens / accept_forced_chain
```
