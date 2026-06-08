# SOTA Constrained Decoding for `minLength≥1` Enforcement
### Research brief for Atlas (XGrammar) — 2026-05-25

---

## Scope and concrete problem

Atlas compiles tool-call grammars in `crates/spark-server/src/grammar/compile_tools.rs`. The current
helper `enforce_min_length_on_required_strings` injects `"minLength": 1` on every required string
field, then for **Qwen3-Coder XML** it falls back to a hand-written regex
`(<parameter=NAME>[ \t\r\n]*[^ \t\r\n<][^<]*</parameter>\s*)+` because XGrammar's `\S`-quantifier
sandwich in a `[\s\S]*` Kleene closure permits ε-transitions that the FSM crosses without
consuming a non-whitespace byte (see code comment lines 247-257). The MoE FP8 dequant drift
(MASTER_DRIFT_TABLE: `ssm.moe_out @ L20 = 0.91983`) routinely samples `</parameter>` immediately
after the opener, emitting `<parameter=command></parameter>`.

Below: what every other 2024-2026 SOTA stack does, and 5 ranked recommendations.

---

## 1. The landscape (2024-2026)

### 1.1 FSM family — Outlines, lm-format-enforcer

**Outlines** (Willard & Louf 2023; dottxt-ai). Pioneered FSM-based decoding by compiling regex →
DFA, then projecting onto the BPE vocabulary as an `(state, token) → next_state` table.
**Critical finding**: per github.com/dottxt-ai/outlines#215, **`minLength` was not implemented**
as of mid-2025 — only `maxLength`. Outlines' canonical path is to translate JSON-schema string →
regex; when `minLength: N` is requested the project's TODO is to emit `.{N,}` as the body. The
compile-time cost is the bottleneck (40s–10min on deep recursive schemas; #658).

**lm-format-enforcer** (noamgat). Imperative character-level approach via
`StringParsingState.get_allowed_characters()`. Tracks a counter of bytes emitted and removes the
JSON-string closing quote `"` from the allowed-character set until the counter reaches
`minLength`. This is the **only mainstream OSS engine that enforces `minLength` correctly today
without relying on a regex quantifier** — because it tracks state imperatively rather than
through FSM transitions. Cost: ~50-100μs/token due to the trie intersection at every step.

### 1.2 CFG family — XGrammar, llguidance, SynCode

**XGrammar** (Dong et al., MLSys 2025; arXiv:2411.15100). Byte-level pushdown automaton with
context-independent token mask caching: precomputes ≥99% of the next-token mask offline.
Compiles JSON-schema → EBNF where `minLength: 1` becomes a `{1,}` repetition in the string-body
rule. Achieves up to 100× speedup vs Outlines, ~5μs/token on H100. **Known limitations**:
(a) `minItems` rejected outright (vllm#16880); (b) regex `\S` inside `[\s\S]*` Kleene closure
allows ε-transitions to bypass the quantifier — Atlas hit exactly this (compile_tools.rs:247);
(c) `TagDispatch` is non-anchored, so partial-trigger byte streams aren't masked (Atlas F70/F72).

**llguidance** (Moskal et al., arXiv:2502.05111; Microsoft/guidance-ai). Rust implementation,
Earley parser + derivative-based lazy lexer + memory-optimized tokenizer trie. ~50μs CPU/token
on a 128k vocab. Supports Lark-format CFG with embedded JSON-schemas and regexes. **Critically:
llguidance's lexeme model treats `minLength` as an Earley-parser cardinality constraint, not a
regex quantifier**, so it is ε-immune — the parser counts accepted lexemes against the minimum
and rejects EOF until satisfied. This is the cleanest SOTA solution. Now backs llama.cpp
(PR#10224), TensorRT-LLM (llgtrt), and SGLang (PR#3298).

**SynCode** (arXiv:2403.01632, ICML 2024). DFA mask store over EBNF terminals. Treats
quantifiers via separate DFA states per cardinality bucket (`{0,1}`, `{1,}`, `{n,m}` each become
their own DFA), guaranteeing minimum-occurrence at the FSM level. 96% syntax-error reduction on
code. Not adopted in production engines but the technique generalizes.

### 1.3 Hybrid sampler + grammar — DOMINO, AICI, OpenAI Structured Outputs

**DOMINO** (Beurer-Kellner et al., ICML 2024; arXiv:2403.06988). Sub-word-aligned constraint
enforcement combined with speculative decoding inside the constraint. Pre-computes legal
sub-word continuations, then speculatively executes the most-likely path. **~2× speedup vs
unconstrained** in some configurations. Handles `minLength` via constraint-aware speculation —
it cannot speculate past the minimum-cardinality boundary without first consuming the required
content tokens.

**AICI** (Moskal 2024; Microsoft). WASM-based controller VM running custom Rust constraint
programs in-loop with the LLM. The Controller is free to track any state (including
`bytes_emitted_in_field`) and emit token-level masks. This is the **most general approach** but
requires writing imperative controllers per use-case.

**OpenAI Structured Outputs** (2024 launch). Converts JSON schema → CFG → cached
context-independent mask table (same family as XGrammar). The OpenAI docs explicitly state
which JSON-schema keywords are supported — `minLength` is **NOT in the supported list** as of
their 2025 published spec. They handle the empty-string problem by training (their gpt-4o-mini
finetune for structured output) rather than by hard grammar enforcement.

### 1.4 Sampler-layer overlays — PICARD (foundational), token-level rejection

**PICARD** (Scholak et al., EMNLP 2021). Incremental parser integrated with beam search; rejects
inadmissible tokens by re-parsing after each step. Pure validate-and-resample; high overhead but
flexible. Foundational. Modern engines (XGrammar / llguidance) precompute the rejection mask
instead of re-parsing each step.

**Token-level rejection sampling** (Hewitt et al. 2024-2025): bias-then-resample, where the
constraint is a soft penalty rather than a hard mask. Better for fuzzy constraints but does NOT
guarantee `minLength`.

### 1.5 Speculative decoding + grammar (EAGLE/Medusa/MTP)

XGrammar 2025 roadmap explicitly addresses this: their tree-scoring API for speculative tokens
calls the same `accept_token` path as standard decoding, with the matcher state rolled back if
the verify step rejects the draft. llguidance has equivalent rollback. **The hard problem with
speculative decoding + minLength** is that a draft sequence may straddle the `minLength`
boundary — both engines handle this by tracking the boundary in matcher state, so rollback
restores the byte counter.

---

## 2. Specific answers to the 5 questions

### Q1. How do SOTA systems enforce `minLength: 1` on JSON-string fields?

| System | Mechanism | Status |
|---|---|---|
| Outlines | regex `.{1,}` quantifier | **NOT implemented** (#215) |
| lm-format-enforcer | byte-counter state in `StringParsingState`, masks closing `"` until ≥N | **Works correctly** |
| XGrammar | EBNF `{1,}` repetition in string body | **Works for plain JSON**, breaks with `\S` in `[\s\S]*` Kleene closure |
| llguidance | Earley parser cardinality constraint on lexeme stream | **Works correctly** (ε-immune) |
| SynCode | per-cardinality DFA state machines | Works correctly |
| OpenAI SO | not supported; relies on model finetuning | Soft enforcement only |
| DOMINO | speculation-aware regex quantifier | Works correctly |

### Q2. Hybrid sampler + grammar — SOTA combination, when each enforces?

The SOTA pattern (2025) is **two-layer**:
1. **Grammar layer** (XGrammar / llguidance) enforces structural correctness — bracket balance,
   field names, JSON syntax. Hard mask: invalid tokens → `-inf`.
2. **Sampler layer** enforces semantic / cardinality constraints the grammar engine can't
   express cleanly — byte counters, context-sensitive masks (e.g. "no `</tool_call>` until
   `minLength` reached"), temperature / penalty scoping (Atlas already does this — see
   `decode_logits_seq.rs:395-427` `in_tool` gating of DRY/presence/frequency).

Atlas is **already** doing this hybrid correctly for thinking/`</think>`/`tool_call_start_token`.
The missing piece: a byte-counter sampler stage that masks the parameter-close token
(`</parameter>` / `"`) until N≥1 non-whitespace bytes of parameter VALUE have been emitted.

### Q3. Context-sensitive constraints ("non-empty for THIS tool, optional for THAT tool")

- XGrammar: **Yes**, via per-tool structural-tag with per-tool JSON schema (Atlas already does
  this — each `tag_entries` carries the tool-specific schema with `enforce_min_length_on_required_strings`).
- llguidance: **Yes**, native (per-grammar-rule cardinality).
- Outlines: **No** (one FSM per request).
- lm-format-enforcer: **Yes**, via per-field parsing-state composition.
- AICI: **Yes**, by construction (arbitrary controller logic).

### Q4. Performance overhead

| System | Per-token cost | % of decode |
|---|---|---|
| XGrammar | 5-30 μs | <1% on H100 |
| llguidance | ~50 μs / 128k vocab | 1-2% |
| Outlines | 50-200 μs (slower compile) | 2-5% |
| lm-format-enforcer | 50-100 μs | 2-3% |
| DOMINO | ~negative (speedup via spec) | -50% to +5% |
| AICI | depends on controller | 100μs-1ms |

Atlas's `forced_token_fastpath` already skips both bitmask fill and sampling when the grammar
admits one legal next token — this is XGrammar's "Coalescence / Tier 3b" optimization, the
fastest possible path. Adding a sampler-layer byte counter costs O(1) and runs only inside the
tool body — negligible.

### Q5. Compatibility with speculative decoding (MTP / EAGLE / Medusa)

XGrammar 2025: tree-scoring API; matcher state advances per accepted draft token, rolls back per
rejected. llguidance: equivalent rollback model. **For Atlas's MTP K=2..K=4 verify steps**: any
sampler-layer byte counter for `minLength` MUST be checkpointed before draft and rolled back
on reject — same rollback discipline as `grammar_state` and the existing Mamba SSM rollback
(see project_qwen36_drift_gdn_clean / phase16 wy_investigation memory notes for the SSM
rollback precedent).

---

## 3. Top 5 ranked recommendations for Atlas

### #1 — Sampler-layer byte counter for parameter VALUE bodies (highest ROI)
Track `bytes_emitted_in_current_value: u32` per active sequence. When inside a parameter VALUE
body (between `<parameter=NAME>` and `</parameter>`), mask the closing-tag opener `<` until
`bytes_emitted_in_current_value ≥ 1` and at least one non-whitespace byte has been seen. This is
the **lm-format-enforcer pattern** transplanted to Atlas's existing sampler — close cousin of
the `inside_tool_body` / `in_tool` gating already in `decode_logits_seq.rs:395`. **ε-immune**
(it's a counter, not a regex transition). Citation: noamgat/lm-format-enforcer
`StringParsingState.get_allowed_characters()`.

### #2 — Migrate Qwen3-Coder XML grammar to llguidance for that wire format only
llguidance's Lark CFG + cardinality at the lexeme level natively handles `minLength`. Pin only
the Qwen3-Coder XML path to llguidance (keep XGrammar everywhere else). Avoids the
ε-transition footgun. Cost: add `llguidance` Rust dep alongside `xgrammar`. Citation: Moskal
et al., arXiv:2502.05111; github.com/guidance-ai/llguidance.

### #3 — File XGrammar bug + PR for `\S` in Kleene closure
The XGrammar regex-to-CFG converter at `regex/converter.rs:154` drops the non-greedy `?`
and lets ε-transitions cross `\S` in `[\s\S]*\S[\s\S]*`. The fix is to compile the regex to a
**deterministic** NFA (not allowing ε across consume nodes). Upstream this with a minimal
repro. Reference vllm/vllm#16880 and mlc-ai/xgrammar#175.

### #4 — Adopt the DOMINO speculative-constraint pattern under MTP
At the MTP draft-verify boundary, pre-compute the grammar-legal continuation tree at the draft
horizon and reject drafts that would skip the `minLength` boundary. This composes with Atlas's
K=2..K=4 verify pipeline (`verify_k2_step.rs` … `verify_k4_step.rs`). Citation:
Beurer-Kellner et al., ICML 2024 (arXiv:2403.06988).

### #5 — Add TAFC `_think` field interaction guard
The `augment_schema_with_tafc_think` helper already injects a `_think` field at schema head
(schema.rs:29). With `minLength: 1` enforcement added, ensure the `_think` field is **excluded**
from required (which it already is) AND that the sampler-layer counter resets cleanly at
field boundaries — otherwise a long `_think` body would satisfy the counter for the *next* field
spuriously. Citation: TAFC arXiv:2601.18282, CRANE ICML 2025.

---

## 4. Cross-references in Atlas code

- `crates/spark-server/src/grammar/compile_tools.rs:60` — current `enforce_min_length_on_required_strings`
- `crates/spark-server/src/grammar/compile_tools.rs:247-257` — Tier-0 regex Pattern C workaround
  for `\S` ε-transition bug
- `crates/spark-server/src/scheduler/decode_logits_seq.rs:395-427` — existing in-tool penalty gating
  (where the byte-counter would slot in)
- `crates/spark-server/src/scheduler/decode_logits_seq.rs:307-317` — forced-token fastpath
  (compatible with sampler-layer counter; counter check is O(1))
- `crates/spark-server/src/scheduler/verify_k{2,3,4}_step.rs` — rollback sites that must also
  rewind the byte counter

---

**Sources**
- Dong et al., XGrammar, MLSys 2025 — arXiv:2411.15100
- Moskal et al., llguidance, arXiv:2502.05111
- Beurer-Kellner et al., DOMINO, ICML 2024 — arXiv:2403.06988
- Ugare et al., SynCode, arXiv:2403.01632
- Scholak et al., PICARD, EMNLP 2021 — arXiv:2109.05093
- Microsoft AICI — github.com/microsoft/aici
- noamgat/lm-format-enforcer
- dottxt-ai/outlines#215 (minLength not implemented)
- vllm-project/vllm#16880 (XGrammar `minItems` rejection)
- OpenAI Structured Outputs technical blog 2024
