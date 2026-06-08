# Multi-turn Opencode Faithfulness — Synthesis of 10-agent research cycle

**Date**: 2026-05-26
**Goal**: Identify the actual root cause(s) of Qwen3.6-35B-A3B-FP8 multi-turn drift in opencode flows and produce a tiered fix plan.

---

## Top-line conclusion

Multi-turn drift is **NOT primarily** a precision/cosine problem. It is the compounding of ~9 independent Atlas server-layer departures from upstream Qwen guidance and from what vLLM/SGLang/MLC/TRT-LLM ship. The 0.04 BF16-LUT-to-BF16-unquant cosine gap is real but small; the dominant problems are **prompt fidelity** (what the template feeds the model) and **logit-bias reliability** (which biases actually fire end-to-end).

Six fixes can ship in **under a day total** with high confidence. The remaining items are precision work (multi-day) and SOTA novelty (multi-week).

---

## Master finding table

| # | Finding | Source | File:line | Class | Scope |
|---|---|---|---|---|---|
| **F1** | `IncomingMessage` has no `reasoning_content` field — Atlas silently drops it on deserialization. Multi-turn replays render historical `<think></think>` empty, triggering empty-think poisoning → premature `<|im_end|>`. | Agent 6 internal | `openai/chat_message.rs:6-19` | **BUG** | 1 hr |
| **F2** | EBNF tool body `first_char ::= [^ \t\r\n<]`, `rest ::= [^<]*` forbids `<` anywhere in tool values. Rust generics (`Vec<String>`), shell redirects, HTML — all rejected. `accept_token` returns false → sequence ends prematurely with garbage args. **Direct cause of "1-char garbage" Epoch-3 behavior.** | Agent 6 internal | `grammar/compile_tools.rs:248-254` (dup 333-339) | **BUG (our regression)** | 30 min |
| **F3** | `forced_token_fastpath` bypasses all logit biases except the Tier-1 empty-param gate. `suppress_tool_call=-12`, client bias, DRY, rep penalty all become no-ops at grammar-narrowed positions. | Agent 6 internal | `decode_logits_seq.rs:280-327` | **BUG** | 1 hr |
| **F4** | MTP verify path (`verify_pick_with_pipeline`) returns post-mask argmax directly — Tier-1 `</`+whitespace bias is bypassed for K=2 verify tokens. Real tool-call corruption candidate. | Agent 7 MTP | `decode_logits_seq.rs:441` + verify path | **BUG** | 1 hr |
| **F5** | FP8 KV EMA recalibration in `fp8_calibration.rs:179-194` still fires every 128 tokens after `frozen=true`, silently corrupting already-written cache. The comment "responds faster to multi-turn topic switches" describes exactly the path that breaks multi-turn coherence. | Agent 7 model | `fp8_calibration.rs:179-194` | **BUG** | 30 min |
| **F6** | Empty `<think>\n\n</think>\n\n` wrappers emitted in two places (template + chat/msg_entry.rs) for the same historic turns even when reasoning_content is absent. Off-distribution from Qwen training corpus; MLC just shipped a strip fix (commit `d75d64e`, Apr 2026) targeting this exact symptom. | Agents 3+8+vLLM | `jinja-templates/openai/qwen3_5_moe.jinja:100-104` + `chat/msg_entry.rs:73-80` | **BUG** | 30 min |
| **F7** | vLLM #39055 / qwen3_reasoning_parser pattern: tool_calls emitted inside `<think>` block silently vanish because the reasoning parser extracts everything before `</think>` and the downstream tool parser only inspects `content`. Atlas exposed if the same shadowing exists in our reasoning extraction. | Agent 4 community | `tool_parser/qwen3_coder.rs` + reasoning extractor | **AUDIT** | 1 hr audit + 1 hr fix if confirmed |
| **F8** | Rep-penalty exemption is inverted: zeroed inside tool body (no attractor protection — allows `Cargo.toml.new.tmp.bak1...`), active outside (penalizes natural keyword recurrences). 256-token window hardcoded in 4 sites. | Agent 9 sampler + Agent 6 | `decode_logits_seq.rs:405,449-466` + `prefill_a_step.rs:234,310,390` + `lifecycle.rs:180,265` | **BUG** | 1 hr |
| **F9** | Atlas `rep_pen=1.10` and `dry_mult=0.5` outside Qwen3.6 card recommendations (1.0 / 1.05 for rep, 0.8 or off for DRY). `thinking_coding` preset is dead code (compiled but never dispatched in `sampling_setup.rs::build_sampling()`). | Agent 9 sampler | `kernels/gb10/qwen3.6-35b-a3b/MODEL.toml` + `sampling_setup.rs` | **TUNING** | half day |
| **F10** | `<tool_call>` exponential bias schedule (+3/+3/0/-5/-10) stacks with rep_pen also penalizing `<tool_call>` once it appears. At parallel-tool-call repeat=3, model is double-penalized away from legitimate sibling calls. | Agent 9 sampler | `decode_logits_seq.rs` bias schedule | **BUG** | 1 hr |
| **F11** | TRT-LLM PR #12061: their tool parser was silently dropping tool calls whose `function_name` wasn't in `_tool_indices`. Atlas almost certainly has equivalent silent drops in `tool_parser/qwen3_coder.rs`. | Agent 8 TRT-LLM | `tool_parser/qwen3_coder.rs` | **AUDIT** | 30 min audit |
| **F12** | `w8a16_gemm` / `w8a16_gemm_t` kernels never got the DeepGEMM two-level FP32 accumulator that fixed MoE. Shared-expert gate/up/down + Qwen3-attention QKV/O-proj + SSM in_proj/out_proj all still cast `LUT × scale → BF16` before MMA. **This IS the residual L31-L39 cosine gap.** | Agent 7 model | `kernels/gb10/common/w8a16_gemm.cu:213-217` + `w8a16_gemm_t.cu:214-217` | **PRECISION** | 2-3 days |
| **F13** | MLC-LLM commit `d75d64e` (Apr 2026) added `strip_reasoning_in_history` for qwen3 — PR matches our drift verbatim: echoing prior-turn `<think>` causes `<|im_end|>` premature abort. qwen-agent ships `preserve_thinking=true` (opposite). Tension is **prefix-cache reuse vs output coherence**; our failure mode matches MLC's better. | Agents 3+8 | New flag in template + `chat_impl.rs` | **DECISION** | 1 hr + A/B test |
| **F14** | XGrammar 0.1.34 (Apr 2026) ships built-in `qwen_3_5` / `qwen_3_coder` structural-tag builder using `TriggeredTagsFormat + JSONSchemaFormat(style="qwen_xml")`. XGrammar 0.2.0 (May 2026) cut compile cost 21.7s → 2.1s at 1000 tools. Replaces our hand-rolled EBNF entirely. | Agent 8 TRT-LLM | `crates/xgrammar` (vendored) | **UPGRADE** | 1-2 days |
| **F15** | Qwen3-Coder official sampler: `temp=0.7, top_p=0.8, top_k=20, rep_pen=1.05`. Both vLLM and SGLang auto-load from `generation_config.json`; Atlas should verify it does too. | Agent 4 vLLM/SGLang | `MODEL.toml` / `sampling_setup.rs` | **TUNING** | 1 hr |
| **F16** | Probe forensics: 30-message opencode probe shows (a) **reasoning↔action disconnect at turn 24** — reasoning correct, tool call uses phantom path from turn 18; (b) **reasoning channel collapse** — `reasoning_content` length decays 233→0 char across 14 assistant turns; (c) one-byte path drift (hyphen dropped); (d) `(no output)` tool returns trigger paraphrase loop-attractor. | Agent 10 forensics | (observed behavior) | **DIAGNOSTIC** | — |
| **F17** | MTP on Qwen3.6-35B-A3B-FP8 amplifies MoE expert-routing drift in agentic flows. On 27B dense sibling it demonstrably **causes** 30k-token CSS attractor loops. Recommendation: disable MTP for tool-active turns. | Agent 7 MTP | MTP scheduler | **REC** | 1 hr |
| **F18** | `enable_thinking: false` silently ignored from request body. `thinking_in_tools=true` MODEL.toml setting overrides per-request `enable_thinking=false`. | Agent 6 internal | `chat_request.rs:404-406` | **BUG** | 1 hr |
| **F19** | AdaDec (arXiv:2506.08980): pause-then-rerank at high-entropy positions inside argument values. +20.9pp Pass@1 code gen. **Direct fix for the one-byte path drift symptom.** Inference-time, no training. | Agent 1 arXiv | new sampler hook | **NOVEL** | 2-3 days |
| **F20** | TACT (arXiv:2605.05980, May 2026): residual-stream activation steering vs "overthinking/overacting drift" axes. +5.8pp SWE-bench on Qwen3.5-27B (same model family). | Agent 1 arXiv | new model hook | **NOVEL** | 3-5 days |
| **F21** | Hallucination-gate classifier (arXiv:2601.05214): 86.4% real-time detection of parameter-level hallucinations via last-layer hidden-state classifier. Direct fit for our wrong-but-valid symptom. | Agent 1 arXiv | new model hook + tiny classifier | **NOVEL** | 3-5 days |

---

## Conflict resolution: strip vs preserve historical `<think>`

- **Agent 3** (Qwen official): qwen-agent ships `preserve_thinking=true`. Optimizes for prefix-cache reuse.
- **Agent 8** (MLC-LLM): commit `d75d64e` Apr 2026 ships `strip_reasoning_in_history` for qwen3. Optimizes for output coherence; PR cites the exact failure mode we observe (premature `<|im_end|>` inside `<think>`).

**Resolution**: F1 (add `reasoning_content` to `IncomingMessage`) is the prerequisite — until that's fixed, both strategies render empty `<think></think>` and we can't even A/B test. Ship F1 first, then A/B test strip vs preserve under the same probe. The MLC failure mode is closer to ours (Agent 10's reasoning-channel-collapse finding), so my prior is **strip is the right call**, but A/B will be definitive.

---

## Ranked tier plan

### Wave 1 — "Bug fixes" (≤ 1 day total wall time)

Six concrete bugs, high confidence, small surface, all ship together as one image.

1. **F1**: Add `reasoning_content: Option<String>` to `IncomingMessage`, forward to chat template. **Most-likely-biggest fix.**
2. **F2**: Change EBNF tool body to allow `<` in values. Either drop the constraint entirely or restrict only to forbidding the literal `</parameter` close-tag prefix. (Use a negative-lookahead via the grammar combinator rather than a blanket `[^<]` exclusion.)
3. **F3**: Apply ALL logit biases in `forced_token_fastpath`, not just the Tier-1 empty-param gate.
4. **F4**: Apply Tier-1 `</`+whitespace bias on the MTP verify path (currently bypassed).
5. **F5**: Gate FP8 KV EMA recalibration on `!frozen`. After freezing the cache, do not retroactively modify it.
6. **F6**: Skip empty `<think></think>` wrapper emission when `reasoning_content` is empty after trim (both in template and `chat/msg_entry.rs`).

Plus 1-hr audits for **F7** (reasoning parser shadowing tool calls) and **F11** (silent tool-name drop in `tool_parser/qwen3_coder.rs`).

**Expected impact**: empty-think poisoning gone (F1+F6), path-drift "garbage value" mechanism gone (F2), MTP corruption gone (F4), cache-corruption gone (F5). This alone may resolve the bulk of opencode multi-turn failures.

### Wave 2 — "Sampler polish" (½ day)

7. **F8**: Invert rep-penalty exemption — penalize inside tool body (attractor protection), exempt the structural markers (`<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<|im_end|>`).
8. **F9**: Drop `rep_pen` 1.10 → 1.05 for tools/coding, → 1.0 for prose. Drop `dry_mult` 0.5 → 0.0. Wire `thinking_coding` preset into `build_sampling()` dispatch.
9. **F10**: Decouple `<tool_call>` exponential bias schedule from rep_pen by adding tool tokens to the exempt list while the bias schedule is active.
10. **F15**: Verify Atlas auto-loads `generation_config.json` sampling defaults; if not, port the values to MODEL.toml.
11. **F17**: Add `disable_mtp_for_tools` MODEL.toml flag, default on for Qwen3.6-35B-A3B-FP8.
12. **F18**: Honor per-request `enable_thinking: false` (currently silently ignored).

### Wave 3 — "Decide strip vs preserve" (½ day)

13. **F13**: Implement `strip_reasoning_in_history` MODEL.toml flag (MLC-style). Run opencode probe twice — once strip, once preserve. Adopt whichever wins on (a) reasoning_content length stability, (b) no premature EOS, (c) probe outcome (does the model finish the axum task?).

### Wave 4 — "Precision parity" (multi-day, gates on Waves 1-3 showing residual gap)

14. **F12**: Port DeepGEMM two-level FP32 accumulator to `w8a16_gemm`, `w8a16_gemm_t`, shared-expert kernels, Qwen3-attention QKV/O-proj, SSM in_proj/out_proj. This closes the L31-L39 cosine cliff for the deep attention layers where multi-turn collapse manifests.

### Wave 5 — "Engine upgrade" (multi-day, optional)

15. **F14**: Port XGrammar 0.1.34's built-in `qwen_3_5` structural tag. Replaces our hand-rolled EBNF entirely; gets 10× compile speedup at 1000 tools.

### Wave 6 — "SOTA novelty" (multi-week, only if Waves 1-5 leave a measurable gap)

16. **F19** AdaDec — pause-and-rerank inside tool argument values. Targets one-byte token drift directly.
17. **F20** TACT — activation steering against overthinking/overacting drift.
18. **F21** Hallucination-gate classifier — last-layer parameter-level hallucination detector.

### Opencode-side (out of Atlas scope; hand to user if desired)

- Echo the canonical file path in tool result responses (closes one-byte drift loop).
- Replace `(no output)` with `(command completed successfully, no stdout)` to break paraphrase loop-attractor.
- Auto-emit `description` field for the bash tool.

---

## Recommended starting point

**Ship all of Wave 1 (six fixes) as a single image, A/B against the BF16-LUT v2 baseline using the canonical 30-message opencode probe + a small live opencode session.**

Why bundle: each is small, each touches a different layer (template / grammar / sampler / MTP / cache), and they share build/deploy cycle. Cosine + per-turn `reasoning_content` length curves are clean A/B metrics; opencode probe outcome is the headline.

Then choose:
- If Wave 1 closes the gap → ship Wave 2 (polish), then stop.
- If Wave 1 helps but a precision residual remains → Wave 4 (port two-level FP32 acc to remaining kernels).
- If Wave 1 reveals deeper structural issue → Wave 6 SOTA exploration.

---

## Open questions for the user

1. Wave 1 bundle (6 fixes) is my recommendation. Approve, or want to split / re-order?
2. For F13 (strip vs preserve), are you OK with empirical A/B, or do you have a prior preference?
3. Out-of-scope opencode-side fixes — are you OK forwarding those to the opencode maintainers, or should we stay strictly Atlas-side?
