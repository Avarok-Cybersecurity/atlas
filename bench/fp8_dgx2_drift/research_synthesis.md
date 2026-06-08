# Research Synthesis — 11-Agent Sweep on Multi-Turn Agentic Coherence

**Date**: 2026-05-25 evening
**Scope**: 11 parallel agents covering production inference engines (vLLM v1,
SGLang, TGI, TensorRT-LLM, llama.cpp, MLC-LLM/XGrammar), arXiv 2024-2026 SOTA
(constrained decoding, multi-turn coherence, long-context attention), agentic
frontends (opencode, Cline, Aider, etc.), and the Inference Bible (Kiely
2026).

---

## 1. Consensus root-cause (corroborated by 7+ agents)

Multi-turn coherence loss in Qwen3.6-35B-A3B-FP8 is **NOT a single bug** —
it's the compounding of three independently-documented mechanisms that
single-pass cosine (0.994) cannot detect:

1. **Quantization-induced MoE routing instability at deep layers** (B7, B9,
   B11). FP8 forward selects slightly different experts than BF16,
   concentrated in outlier dimensions that encode agent-critical signal.
   Atlas's measured 8/8 → 3/8 expert agreement drift at L38 matches exactly
   (EAQuant arXiv:2506.13329, long-context quant arXiv:2505.20276).
2. **Positional/recency attention bias** pushes system prompt + tool schema
   out of effective attention by turn ~10 (B7, B9; LongFuncEval
   arXiv:2505.10570, NoLiMa arXiv:2502.05167).
3. **Autoregressive feedback into attractor cycles** — model's own earlier
   malformed tool calls become templates (B7; Attractor Cycles
   arXiv:2502.15208).

The Inference Bible (B11, Ch. 5.1.2) explicitly frames this as
"compounding-sensitive paths" — Atlas's single-pass 0.994 cosine is
*expected* to coexist with multi-turn failure because cosine averages over
bulk hidden dimensions while agentic failure lives in the tail.

---

## 2. The xgrammar `\S` bug — finally explained (B5, B6, B8 converged)

My earlier regex `[\s\S]*\S[\s\S]*?` failed to enforce non-empty because:
- xgrammar's regex compiler lowers `*` and `+` as quantifier edges in an
  FSM, where `\S` sandwiched between two Kleene closures becomes
  ε-transitive (the FSM walks past `\S` without consuming a non-WS byte)
- Worse, there's a **literal bug** in
  `crates/xgrammar/src/regex/escape_handlers.rs:129` — top-level `\S`
  emits `[^[\f\n\r\t\v  ]` (with stray `[`), a malformed class.
  Atlas's tests at `regex/tests.rs:79-82` codify the bug as expected.
- Atlas's CURRENT regex `[^ \t\r\n<][^<]*` (Tier-0 shipped today) is the
  RIGHT workaround — structurally enforces non-empty.

**The remaining empty-parameter failures are at a different layer**: the
OUTER `+` quantifier in `(<parameter=...>...)+` allows `</tool_call>` to
fire BEFORE any `<parameter=` block opens. The inner enforcement works;
the outer envelope doesn't require at least one parameter block.

---

## 3. Master ranked recommendation matrix

### Tier 0 — Smallest grammar tightening (1 day effort, high impact)

**Item 0.1**: Add `at_least_one: true` semantic to the OUTER `+` quantifier
of qwen3_coder grammar so `</tool_call>` is unreachable until ≥1
`<parameter=` block has opened. Per B6, the canonical xgrammar idiom is
`json_schema` content type with `style: "qwen_xml"` and `minLength: 1` on
required strings — this compiles to `[^]{1,}` EBNF (deterministic,
ε-immune). Atlas already has `enforce_min_length_on_required_strings`
helper at `grammar/schema.rs` — just unblock that path.
**Files**: `crates/spark-server/src/grammar/compile_tools.rs:250-266` and
`:325-345`. Switch content type from `regex` → `json_schema` style
`qwen_xml`.

**Item 0.2**: Replace string triggers with `token_triggered_tags` keyed on
the `<tool_call>` token id 248058 (Atlas's resolved value). Eliminates
BPE-boundary-spans-trigger class of failures.
**Files**: same `compile_tools.rs:264-271` (trigger construction).

**Item 0.3**: Fix the `\S` bug at `crates/xgrammar/src/regex/escape_handlers.rs:129`
and update test at `regex/tests.rs:79-82`. PR upstream to mlc-ai/xgrammar.
**Effort**: 30 min + test update.

### Tier 1 — Sampler-layer byte counter (CONSENSUS PICK across 4 agents: A10, B1, B5, B8)

The cleanest ε-immune enforcement for `minLength≥1` is a **byte counter
state machine** at the sampler. Pattern from lm-format-enforcer:

```
Track per-sequence: inside_parameter_body: bool, param_body_chars_emitted: u32
Flip on/off in emit_step.rs by inspecting recent emitted bytes for
  <parameter=NAME> opener and </parameter> closer
In decode_logits_seq.rs:395-427, when inside_parameter_body && chars_emitted==0,
  append (parameter_close_token_id, -8.0) to logit_bias
```

**Files**: `crates/spark-server/src/scheduler/types.rs` (new ActiveSeq
fields), `emit_step.rs:90-121` (flag flip), `decode_logits_seq.rs:395-427`
(bias injection), `main_modules/serve_phases/tokenizer_runtime.rs:226+`
(close-token resolution at boot).

This complements Tier 0 (grammar enforces *structure*, sampler enforces
*content cardinality*) and is what TRT-LLM, vLLM, and lm-format-enforcer
all do.

### Tier 2 — vLLM-style parser strengthening (B1)

Three concrete gaps Atlas has vs vLLM's qwen3_coder parser:

**2a**: Add **closer-suffix holdback** to `safe_emit_len` at
`tool_parser/streaming_impl.rs:368-397`. Today it holds OPENER prefixes
only (`<tool_call`, `<param=` etc.). Add `</parameter`, `</function`,
`</tool_call` closer prefixes and apply when `self.inside_tag=true`.
**Direct fix** for the `axum-v51 → axums_v51` mutation class.

**2b**: **Per-`<parameter>` JSON mini-delta** with prefix-invariant check
— vLLM's `compute_tool_delta` asserts `new_args.starts_with(prev_args)`.
Atlas emits args in one chunk at `</tool_call>` boundary. Replace with
incremental loop in `streaming_impl.rs:64-83` that emits at each
`</parameter>`.

**2c**: Schema-aware **typed-argument coercion** (just shipped today as
`wants_typed_arguments=true` on Qwen3CoderParser; uncommitted). Make
unconditional during streaming. Bind to declared schema (B10) — `bash`'s
`timeout: "30"` → `30` (number) before opencode sees it.

### Tier 3 — Sampler/penalty quality (B7, A4, A7)

**3a**: **LZ Penalty** (arXiv:2504.20131) — information-theoretic
Lempel-Ziv residual penalty breaks attractor cycles without the
quality degradation that vanilla rep/freq penalties cause. Atlas
already has LZ penalty infrastructure (`lz_penalty` field in
SamplingParams, currently zeroed inside `inside_tool_body`). Tune
outside tool body for the meta-narration loop class.

**3b**: **min-p floor inside tool body** (Nguyen et al. ICLR 2025 oral,
arXiv:2407.01082) replacing top_p when distribution flattens under FP8
drift. Atlas currently zeros sampler penalties inside body but doesn't
substitute min-p; the distribution flattens and empty-attractor tokens
win.

**3c**: **`presence_penalty=1.5` outside tool body** per Qwen team's
official model card (A4). MODEL.toml `[sampling.tools]` currently sets
0.0. This affects only the OUT-OF-tool prose between calls — exactly
where the meta-narration ("I'm going to stop this loop...") leaks.

### Tier 4 — Multi-turn architectural interventions (B7, B9, B11)

**4a**: **Periodic system-prompt re-injection every K turns** (B7;
arXiv:2510.07777 "Drift No More?"). Drift settles to a bounded
equilibrium that reminder interventions provably lower. Pure
serving-side context surgery. Atlas has prompt-injection infrastructure
already (Phase A removed *unconditional* injections; this would be a
gated *turn-counter* injection).

**4b**: **FlowKV per-turn KV isolation** (arXiv:2505.15347) — showed
10.9% → 75.4% retention in later turns. Zero training, zero kernel
changes. Wraps any compression strategy.

**4c**: **NVFP4 KV for deep attention layers (L31-L39)** via Atlas's
existing `--kv-high-precision-layers` knob, but flipping direction:
NVFP4 at deep, FP8 at early (Atlas's own data shows NVFP4 wins at deep).

### Tier 5 — opencode protocol compatibility (B10)

**5a**: opencode's bash tool has a REQUIRED but undocumented `description`
field (issue #1388). Atlas should auto-emit it from a heuristic OR
update qwen3_coder system prompt to teach the model.

**5b**: **Never emit empty `tool_calls: []` or empty `parameters: {}`**
(opencode #4255 hangs forever, SGLang #8184). Audit
`crates/spark-server/src/openai/chat_response.rs`.

**5c**: **In-server schema-validation re-roll** — when tool args fail
declared JSON-schema, do one XGrammar-constrained re-sample before
returning. Closes opencode's missing-retry-path invisibly.

### Tier 6 — Deferred / multi-day kernel work

- Native FP8 MMA (`mma.sync.m16n8k32.f32.e4m3.e4m3.f32`) on SM121
  — TRT-LLM confirmed using it; Atlas's two-level accumulator
  workaround is good enough for now.
- MXFP8 block-scaled KV cache (Inference Bible Ch. 5.1.1) — bigger
  refactor; defer until multi-turn behavior stabilizes.
- GPU-resident grammar bitmask — vLLM uses, Atlas forces to host;
  removes FP32 round-trip artifact biasing low-margin token choices.
- Cap MTP to K=1 across turn boundaries (B11) — already at K=1
  default; the issue is rollback state at `</think>` boundary
  (A6 — verify_pipeline_helper.rs:173).

### Tier 7 — Critical NON-recommendations

- **DO NOT** deploy self-correction loops (B7; Self-Correction Bench
  arXiv:2507.02778: 64.5% blind-spot rate across 14 open models).
- **DO NOT** replace XGrammar with GBNF wholesale (B5) — bug surface
  larger, llama.cpp has its own active issues (#20164, #20198, #20260).
- **DO NOT** match SGLang's permissive empty-param behavior — Atlas
  should be STRICTER (B2).

---

## 4. Top 3 highest-EV NEW interventions Atlas should ship next

Given Atlas's current state (Tier-0 regex `+` shipped, byte-exact paths,
0.994 model-wide cosine), the **3 highest-expected-value next steps**:

### A. Switch grammar to `json_schema` style `qwen_xml` with `minLength:1` (Tier 0)
**Why**: deterministic, ε-immune, uses existing `enforce_min_length_on_required_strings` helper, fixes the OUTER `+` quantifier hole that lets `</tool_call>` close with zero parameters.
**Effort**: ~10 LoC in `compile_tools.rs`, half-day.
**Risk**: low — canonical xgrammar idiom; MiniMax already uses similar.

### B. Sampler-layer byte counter for `</parameter>` masking (Tier 1)
**Why**: defense-in-depth on Tier 0. Belt + suspenders pattern from lm-format-enforcer. Closes any sampling-time ε-edge xgrammar might still have. ~50 LoC across types.rs, emit_step.rs, decode_logits_seq.rs.
**Effort**: 1 day.
**Risk**: medium — requires correct close-token resolution at boot.

### C. Closer-suffix holdback + per-parameter mini-delta in streaming_impl.rs (Tier 2)
**Why**: directly addresses the `axum-v51 → axums_v51` value mutation class. vLLM pattern, well-tested upstream. Catches malformed mutations BEFORE they reach opencode.
**Effort**: 1-2 days.
**Risk**: medium — touches the hot path of qwen3_coder streaming parser.

These three combined are **half the SOTA stack** and should be testable
end-to-end against opencode within 2 days of focused work. Tier 3 (LZ
Penalty + min-p + presence_penalty) is the next stack to add if Tier 0+1+2
proves insufficient.

---

## 5. Critical files for next implementation cycle

- `crates/spark-server/src/grammar/compile_tools.rs` (Tier 0 grammar switch)
- `crates/spark-server/src/grammar/schema.rs::enforce_min_length_on_required_strings` (reuse)
- `crates/spark-server/src/scheduler/types.rs` (Tier 1 new ActiveSeq fields)
- `crates/spark-server/src/scheduler/emit_step.rs:90-121` (Tier 1 flag flip)
- `crates/spark-server/src/scheduler/decode_logits_seq.rs:395-427` (Tier 1 logit bias)
- `crates/spark-server/src/main_modules/serve_phases/tokenizer_runtime.rs:226+` (Tier 1 close-token resolution)
- `crates/spark-server/src/tool_parser/streaming_impl.rs:64-83, 368-397` (Tier 2 closer holdback + per-param mini-delta)
- `crates/spark-server/src/tool_parser/qwen3_coder.rs:25` (Tier 2c: `wants_typed_arguments=true` — already done, uncommitted)
- `crates/xgrammar/src/regex/escape_handlers.rs:129` (Tier 0.3 `\S` bug, upstream PR)

---

## 6. Verification plan

Test grid after each Tier:
- **v53**: Tier 0 only (grammar switch). Expect: ≥3 files, axum dep added,
  no empty `<parameter=>`. Pass criterion: no fewer files than v51 (3).
- **v54**: Tier 0 + Tier 1. Pass criterion: no SchemaError(Missing key) on
  any tool call; empty filePath impossible at sampling level.
- **v55**: Tier 0 + 1 + 2. Pass criterion: ≥5 files, main.rs has `async fn`
  + Router + at least one route handler. Stretch: tests file present.
- **v56**: Tier 0 + 1 + 2 + 3a (LZ penalty outside body). Pass criterion:
  no meta-narration in tool args, no `which cargo` repetition loops.

Each tier is independently committable. Each is ~1 day of work + 30 min
of opencode test.

---

## 7. References (all 11 research files)

- `research_vllm_pipeline.md` (B1)
- `research_sglang_pipeline.md` (B2)
- `research_tgi_pipeline.md` (B3)
- `research_trtllm_pipeline.md` (B4)
- `research_llamacpp_grammar.md` (B5)
- `research_xgrammar_features.md` (B6)
- `research_multi_turn_arxiv.md` (B7)
- `research_constrained_decoding_sota.md` (B8)
- `research_long_context_arxiv.md` (B9)
- `research_agentic_frontends.md` (B10)
- `research_inference_bible.md` (B11)
