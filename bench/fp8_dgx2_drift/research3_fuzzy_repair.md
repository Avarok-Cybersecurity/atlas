# Research 3: Token-Level Fuzzy Repair / Neural Error Correction for Tool Args

**Date**: 2026-05-26
**Context**: Atlas FP8 dgx2 drift produces single- to Levenshtein-3 corruptions in tool arguments
(`axum`→`axut`, `axum-v3`→`axuma-aadac`, `/test-rust-axum-v3`→`/test-tu-au-u8`). The corruptions
sit on top of a "known-good" vocabulary visible in the user prompt + prior tool responses.
This document maps the literature, production state, and ranks shippable repair tactics.

---

## 1. Academic Landscape (arXiv 2024–2026)

### 1.1 Direct precedent: "fuzzy repair of LLM tool args"

There is **no** paper that names this exact problem. The closest cluster is **tool-call error
recovery via reflection / self-correction**, which is a *post-execution* loop, not a
*pre-execution* repair:

- *Failure Makes the Agent Stronger* (arXiv 2509.18847, 2025) — trains LLMs to reflect on tool
  errors and regenerate. Per-error retry cost: 1 extra forward pass (hundreds of ms minimum).
- *Robust Tool Use via Fission-GRPO* (arXiv 2601.15625) — RL-trained recovery from execution
  errors. Training-time, not inference-time.
- *Tool-Reflection-Bench* (cited in 2509.18847) — benchmark showing Claude Sonnet 4 recovers
  ~50% of broken calls, Qwen3-8B only ~20%.

**Takeaway**: nobody in the literature does the *lightweight, deterministic, prompt-vocabulary*
repair Atlas needs. This is an open lane.

### 1.2 Adjacent: copy mechanisms & context anchoring

These are the **most relevant** academic ideas for Atlas:

- **CopySpec (EMNLP 2025, arXiv 2502.08923)** — speculative-decoding draft from a "copy buffer"
  populated by recent context. Up to +49% throughput; treats verbatim repetition as a fast path.
  Applies to *generation*, not post-hoc repair, but the data structure (rolling context trie)
  is reusable.
- **Copy-as-Decode (arXiv 2604.18170, 2026)** — grammar-FSM that emits "copy span(i..j) from
  context" as a primitive. 6.8–303× speed-up on Qwen2.5 edits. Production-relevant: shows that
  forcing verbatim copies at decode time is *cheaper* than free generation.
- **Selective Prompt Anchoring (arXiv 2408.09121, multiple revs through 2025)** — amplifies
  attention to prompt spans during decode. Reduces drift from named entities in the prompt.
- **LOGIC (arXiv 2601.15397)** — constant-time logit-space contextual biasing for ASR LLMs.
  The technique generalises: at decode time, boost logits of tokens that belong to
  context-derived n-grams. Zero retraining, ~1µs per step.
- **Trie-constrained decoding (arXiv 2602.22647 STATIC, 2502.00085 trie-beam)** — vectorised
  trie masking at 47–1033× over naive. The exact substrate for "constrain tool-arg-string to
  prompt-derived trie".

### 1.3 BERT / neural post-correction

- *Streaming BERT for ASR correction* (2205.00620) — sub-100ms post-edits.
- *Punctuation BERT (2412.02698)* — 95% quality at 10× smaller.
- *Typo neurons (2502.19669)* — LLMs already have internal "typo-fix" heads.

Realism: BERT-base on a T4 ≈ 130µs–10ms/arg. A tiny distilled BERT post-corrector at <5ms/arg
is feasible but adds a second model — heavy for the size of fix.

### 1.4 Constrained decoding

- *Don't Fine-Tune, Decode (2310.07075)* — FSM-constrained tool syntax.
- vLLM (#27766) — constrained tool-call decoding is open gap.
- TRT-LLM — strict guided decoding "best-in-class".
- XGrammar (Atlas already uses) — natural place to plug a prompt-derived trie.

---

## 2. Production State

### 2.1 OpenAI

- **Structured Outputs (Jun 2024)**: `strict: true` JSON-Schema enforcement at decode time.
  Type-correct but **no string-value repair**. A bad string passes if it parses.
- No documented server-side fuzzy correction for tool args. Recommendation in docs: client
  validates with Pydantic, retries.

### 2.2 Anthropic Claude

- **Strict tool use**: same as OpenAI — compiles `input_schema` to grammar. Type-safe, not
  value-safe.
- No public server-side fuzzy repair.
- Claude Code's `text_editor` tool uses `str_replace` with **exact** matching; failures are
  retried by the model (no built-in fuzzing on the server side).

### 2.3 vLLM / TRT-LLM / SGLang / TGI

- vLLM: `tool_choice="auto"` extracts args from raw text via parser, **no schema constraint
  applied during decode**. Open feature request to integrate parser grammar into guided decoding.
- TRT-LLM: strict guided decoding works, but again — types, not value fuzz.
- None of the inference engines surveyed do "string repair against known-good corpus".

### 2.4 opencode (the consumer Atlas talks to)

This is the most actionable finding:

- **`edit` tool runs a 9-strategy fuzzy-replacer chain** (exact → line-trim → whitespace
  normalised → indent-flexible → escape-normalised → block-anchor → **Levenshtein-distance**).
  Atlas's `oldString` corruption is already partially tolerated.
- **`read` / `bash` / file-path tools do NOT fuzz-match.** A bad path returns ENOENT and the
  model retries — often into an **infinite retry loop** (issues #14729, #17169, #735). No
  "did you mean" suggestion.
- Bash tool on Windows has known unfixed path-mangling (issue #15810), suggesting opencode
  considers path normalisation in scope but unfinished.

**Implication for Atlas**: when Atlas emits `axut` instead of `axum`, opencode passes it
verbatim, server returns ENOENT, and the model burns context retrying. Atlas can dramatically
improve user experience by repairing **before sending to opencode**.

### 2.5 Aider, Cursor, Copilot, Morph

Aider: polyglot diff, exact-match, no arg fuzz. Cursor: auto-imports missing symbols (NER-ish),
no arg fuzz. Copilot: no public fuzz docs. Morph: 98% structural-accuracy edit-repair, but
diffs-only — too narrow for Atlas's general case.

---

## 3. Libraries: Fast Fuzzy Match in Rust

Atlas server is Rust. Realistic options for <10ms per arg:

| Library                | Algorithm              | Throughput (single core) | Suitability |
|------------------------|------------------------|--------------------------|-------------|
| **rapidfuzz-rs**       | Optimised Levenshtein, OSA, Jaro-Winkler | ~5–20 µs / pair @ ≤32 char | Best general-purpose |
| **strsim-rs**          | Levenshtein, DL, JW    | similar, smaller binary  | Drop-in if size matters |
| **symspellpy / wolfgarbe/symspell_rs** | Symmetric-delete index | ~5500 lookups/sec @ edit-distance 3, sub-µs at d≤1 | Best for "lookup against fixed corpus" — **direct fit for prompt-vocab** |
| **fst (BurntSushi)**   | Finite-state transducer | ~1µs/lookup for prefix/Levenshtein-d≤2 | Compact, immutable corpus |
| **bk-tree**            | Burkhard–Keller tree   | ~100× slower than SymSpell | Skip — SymSpell dominates |

**Best fit**: SymSpell + rapidfuzz hybrid. SymSpell handles "is candidate within edit-distance
d of any known-good token", rapidfuzz handles tie-breaks and longer substring scoring.

---

## 4. Atlas-Specific Synthesis

### 4.1 The vocabulary Atlas already has

At tool-call emit time the Atlas server has, **for free**:

1. **User prompt text** (already tokenised, sitting in the request).
2. **Prior tool responses** (file listings, `ls` output, `grep` results — already in context).
3. **Prior assistant tool calls** (the model's own previous arg strings — already in context).
4. **CWD-relative path scaffolding** (if Atlas opt-in connects to a workspace).

Source (1)+(2)+(3) is available with zero extra I/O. Extracting strings is regex over the
prompt buffer — sub-ms.

### 4.2 The corruption pattern observed

- `axum`→`axut`: 1-substitution. Levenshtein 1. SymSpell with d=1 returns `axum` in <1µs.
- `axum-v3`→`axuma-aadac`: edit-3, length-perturbed. SymSpell d=2 misses; rapidfuzz partial
  ratio against extracted prompt vocab catches it (`axum-v3` will score 0.7+).
- `/test-rust-axum-v3`→`/test-tu-au-u8`: heavy corruption but **prefix `/test-` preserved and
  `axum-v3` root substring is in the prompt**. A regex extracting `/[a-zA-Z0-9_./-]+` tokens
  from the prompt + scoring candidates with rapidfuzz partial-ratio recovers it.

### 4.3 Where to insert the repair

Three feasible insertion points, in increasing invasiveness:

**A. Post-parse, pre-respond (cheapest)**:
After the tool-call parser extracts `{name, args}`, before serialising to OpenAI/Anthropic
response, run repair on string-typed args. Pure CPU. No decode-loop changes.

**B. Per-step logit shaping (decode-time)**:
During tool-arg generation, boost logits of tokens that extend a known-good prompt n-gram
(prompt-anchored decoding à la LOGIC). Catches errors *before* they happen.

**C. Constrained decoding to prompt-trie (most invasive)**:
While emitting a tool-arg string, mask the vocabulary to only tokens that continue a
prompt-derived trie. Strongest correctness guarantee, biggest risk of mis-masking when the
correct value *isn't* in the prompt.

(A) ships in a day. (B) ships in a week. (C) is a 1-month project.

### 4.4 Risk: false repairs

The catastrophic failure mode is "the model meant something the user *didn't* say". E.g., the
model emits `axum-v4` (a new crate name not in the prompt) and Atlas "repairs" it to
`axum-v3` because that's what the prompt mentioned. Mitigations:

1. **Only repair when corruption is detected**: require the emitted string to *not* be a
   schema-validated value (e.g., path doesn't exist as a literal) before fuzzing.
2. **Conservative thresholds**: only auto-replace when edit distance ≤ 2 *and* there's
   exactly one prompt-vocab candidate within that distance.
3. **Surface the repair in the response**: emit metadata `{repaired_from: "axut", to: "axum"}`
   so the consumer can audit / opt out.
4. **Per-tool, per-arg whitelist**: only fuzz args declared as `path` / `string` matching a
   regex; never fuzz freeform `query` / `description` args.

---

## 5. Ranked Top-5 Implementations for Atlas

### #1. Tier-2 prompt-vocab fuzzy repair for path/identifier args (ship-in-1-day)

- **What**: Extend the existing tier-2 validator. After validating "starts with /", run
  SymSpell d≤2 lookup against tokens extracted from `{user_prompt + prior_tool_outputs}`. On
  unique match, auto-replace. Log both old + new.
- **Cost**: <100µs per arg. CPU only. ~300 LoC Rust + symspell_rs crate.
- **Risk**: Bounded — only fires when arg fails validation. Worst case = current behavior.
- **Gain**: Recovers `axum`→`axut` and similar single-char drifts immediately. ~70% of
  observed drift cases.

### #2. RapidFuzz partial-ratio second-pass for multi-char corruption (ship-in-2-days)

- **What**: When SymSpell finds nothing, slide a length-tolerant rapidfuzz partial-ratio across
  prompt n-grams (n=4..32). Threshold ≥0.65, single best match wins.
- **Cost**: ~1–5ms per arg with proper pruning (length filter first).
- **Gain**: Recovers `axuma-aadac`→`axum-v3` and `/test-tu-au-u8`→`/test-rust-axum-v3` class.
  Closes ~95% of observed drift cases.

### #3. Repair metadata + opencode-facing hint protocol (ship-in-1-week)

- **What**: When repair fires, attach a non-standard tool_call extension
  `_atlas_repaired_from` so consumers (and the user) can see what was changed. Coupled with a
  conservative "do not auto-repair, suggest only" mode behind an env var.
- **Cost**: Trivial code, mostly protocol work.
- **Gain**: Auditability + safety. Required before #1/#2 ship to production paths where false
  repair is dangerous (e.g., bash arg).

### #4. Prompt-derived logit boost during tool-arg generation (ship-in-2-weeks)

- **What**: Extract literal strings and identifier n-grams from prompt. Build a small bloom
  filter / FST. During tool-arg decode, apply +0.5 logit boost to tokens that extend a
  prompt-derived span. (LOGIC-style.)
- **Cost**: ~20µs per decode step, one extra GPU buffer.
- **Gain**: Prevents corruption rather than repairing it. Stacks with #1/#2.
- **Risk**: Tuning the boost magnitude — too high → over-copy, too low → no effect.

### #5. XGrammar prompt-trie tool-arg constraint (ship-in-1-month)

- **What**: When tool-arg is typed `path` or `enum-derived-from-prompt`, build an XGrammar
  trie of prompt-extracted strings and constrain decode to it. Fall back to free decode if
  the model wants to emit something off-trie (e.g., a new identifier).
- **Cost**: ~10µs/step trie lookup using vectorised STATIC-style mask. Engineering: significant
  — needs new arg-type annotation in tool schema, new XGrammar codepath.
- **Gain**: Strongest guarantee. The model literally cannot emit `axut` because no prompt token
  starts that prefix.
- **Risk**: Mis-classification of args that look path-like but legitimately need free text.
  Requires per-tool opt-in metadata.

---

## 6. Recommendation

Ship **#1 + #2 + #3 as a single PR** behind `ATLAS_TOOL_ARG_FUZZ_REPAIR=1` (default off,
opt-in for FP8 paths first). This is ~1 week of work, gets ~95% of observed FP8 drift cases,
and adds zero risk to NVFP4 paths. Defer #4/#5 until #1–#3 are validated in production for
30+ days and the false-positive rate is measured.

The literature gap (no academic paper exists for this) means Atlas could *publish* the
mechanism as a "lightweight tool-arg drift mitigation" technique — fits with the
"Pure Rust LLM Inference" positioning and is genuinely novel.

---

## Sources

- [Failure Makes the Agent Stronger (arXiv 2509.18847)](https://www.arxiv.org/pdf/2509.18847)
- [CopySpec (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1337.pdf)
- [Copy-as-Decode (arXiv 2604.18170)](https://arxiv.org/abs/2604.18170)
- [Selective Prompt Anchoring (arXiv 2408.09121)](https://arxiv.org/pdf/2408.09121)
- [LOGIC logit-space biasing (arXiv 2601.15397)](https://arxiv.org/pdf/2601.15397)
- [STATIC trie decoding (arXiv 2602.22647)](https://arxiv.org/pdf/2602.22647)
- [Don't Fine-Tune, Decode (arXiv 2310.07075)](https://arxiv.org/pdf/2310.07075)
- [Tool-call error recovery survey (arXiv 2601.22352)](https://arxiv.org/pdf/2601.22352)
- [Robust Tool Use Fission-GRPO (arXiv 2601.15625)](https://arxiv.org/pdf/2601.15625)
- [Typo neurons in transformers (arXiv 2502.19669)](https://arxiv.org/pdf/2502.19669)
- [opencode edit tool fuzzy chain (rmk40 gist)](https://gist.github.com/rmk40/cde7a98c1c90614a27478216cc01551f)
- [opencode infinite retry loops (issue #14729)](https://github.com/openclaw/openclaw/issues/14729)
- [opencode subagent retry loop (issue #17169)](https://github.com/anomalyco/opencode/issues/17169)
- [opencode Windows path mangling (issue #15810)](https://github.com/anomalyco/opencode/issues/15810)
- [OpenAI Structured Outputs (function-calling docs)](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)
- [vLLM constrained tool-call gap (issue #27766)](https://github.com/vllm-project/vllm/issues/27766)
- [rapidfuzz-rs](https://github.com/rapidfuzz/rapidfuzz-rs)
- [strsim-rs](https://github.com/rapidfuzz/strsim-rs)
- [SymSpell (wolfgarbe)](https://github.com/wolfgarbe/SymSpell)
- [SymSpell vs BK-tree benchmark](https://medium.com/data-science/symspell-vs-bk-tree-100x-faster-fuzzy-string-search-spell-checking-c4f10d80a078)
- [Morph code-edit repair](https://www.morphllm.com/common-errors/error-editing-file)
