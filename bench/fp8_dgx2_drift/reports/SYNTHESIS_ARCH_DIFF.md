# Synthesis: vLLM vs Atlas architectural comparison (Qwen3.6-A3B-FP8)

**Date**: 2026-05-27
**Source**: 16 background research agents (10 kernel-level, 6 phase-enumeration); all returned.
**Question**: vLLM = 10/10 cargo_valid; Atlas = ~30%. Same FP8 checkpoint. Where is the architectural bug?

---

## TL;DR

The gap is **not** a single kernel rounding bug. It's three stacks of architectural divergences:

1. **Tool-call grammar** — Atlas grammar-constrains every `tool_choice="auto"` request; vLLM doesn't. Opencode default is `auto`. **Probability this is the dominant lever for the opencode harness specifically: 8/10.** Testable in one config flip.
2. **FP8 quantization plumbing** — Atlas takes systematic precision losses on every block-FP8 GEMM that vLLM doesn't (BF16 scales, BF16-narrowed dequant in smem, per-tensor KV descales, MTP head with scale=1.0).
3. **Prompt-mutation + adaptive-sampling feedback loop** — Atlas re-injects degraded `<think>` traces, serves a different sampling preset than vLLM for the same client request, and an adaptive-greedy gate locks in low-margin drift.

Atlas's sampler is doing the kernel's job, masking and amplifying upstream drift. The grammar pipeline is doing more than vLLM's. Even with bit-exact kernels the prompt+grammar pipeline alone would diverge.

---

## Class C — Tool-call grammar pipeline (phase #5)

**Smoking gun for the opencode harness specifically.** Opencode sends `tool_choice="auto"` by default.

| | vLLM | Atlas |
|---|---|---|
| Grammar in `tool_choice="auto"` | **None** — free-text + regex tool_parser | **Always-on structural-tag XGrammar** (`sampling_setup.rs:172-210`) |
| Effect | Model emits whatever, parser figures it out | Every token must satisfy EBNF grammar; FP8 noise can hit a forbidden state |
| Rep-penalty target | Prompt ∪ output (`utils.py:78-94`) | Output only |
| `stop_token_ids` field | Used | **Dead code** (`runtime/src/sampler.rs:96` declared, never read) |

**Three amplifying Atlas-only constraints in the grammar body:**

1. **EBNF rejects values starting with whitespace OR containing `<`** (`grammar/compile_tools.rs:257-263`):
   ```
   value ::= first_char rest
   first_char ::= [^ \t\r\n<]
   rest ::= [^<]*
   ```
   `Cargo.toml` dependency `Vec<T>`, `Vec<u8>`, shell `>` redirects, Rust generics, HTML all contain `<` → **grammar rejects mid-generation**. F2-revert restored strict form because relaxation caused worse drift, comment that `<` recovery should fall to parse-side instead.
2. **Implicit `minLength: 1` auto-injected** on all required strings (`grammar/schema.rs::enforce_min_length_on_required_strings`) — sharpens first-byte distribution.
3. **Forced-token fast-path** (`grammar/state.rs:163-168` + `decode_logits_seq.rs:344-356`) bypasses logit_bias / rep_penalty / sampler when grammar deterministic; WS1/AM1 masks become dead at the very positions xgrammar already pins.

**Verification path (one test):** run opencode harness with `[behavior].disable_tool_grammar=true`. If cargo_valid hits 100%, Class C is dominant; ship a behavior-aware grammar disable for `tool_choice="auto"` requests and re-evaluate Class A/B priority.

---

## Class A — FP8 numerical levers (all confirmed by ≥2 agents)

| # | Lever | vLLM | Atlas | Where | Conviction |
|---|---|---|---|---|---|
| A1 | **Activation quant** | Dynamic per-token-group FP8 (`per_token_group_quant_fp8`) on every block-FP8 GEMM | None on decode (W8A16), implicit scale=1.0 on prefill (`fp8_fp8_gemm_t`) | `loaders_fp8.rs:143`; agents #2, #9, phase #6 | **HIGH** — primary driver |
| A2 | **Block-scale dtype** | FP32 (`fp8_utils.py:1085`) | BF16 — `scale_row_bytes = scale_cols * 2` hardcoded | `linear_attn_arms.rs:111`; **6 `Fp8Weight` sites, no SSOT** | **HIGH** — ~9 mantissa bits lost on every block scale |
| A3 | **FP8 MMA path** | FP8×FP8 → FP32 native MMA, scale applied in epilogue (CUTLASS scaled-mm) | FP8 → FP32 dequant → BF16 narrow in smem → BF16×BF16 MMA | `w8a16_gemm.cu:215`, `w8a16_gemm_t.cu:217` (agent #2) | **HIGH** — per-weight 7-bit round across 40 layers |
| A4 | **FP8 KV cache descale (attn read)** | Per-(seq, kv_head) descale via `_k_scale.expand((seqs, num_kv_heads))` | Single scalar `k_scale`/`v_scale` per layer | `inferspark_prefill_paged_fp8.cu:47-50,74`; `decode.rs:19` (agent #6 F1, agent #3 F-DEC-2) | **HIGH** — deep-layer outlier heads miscalibrated 10-50% |
| A5 | **MTP head FP8 calibration** | Inherits parent layer scales | **Hardcoded `k_scale=v_scale=1.0`** | `mtp_head/forward.rs:264-265,291-292` (agent #3 F-DEC-8) | **HIGH** — one-line fixable; MTP+FP8 is uncalibrated |
| A6 | **MoE topk tie-break** | Lower-index-wins deterministic (`topk_softmax_kernels.cu:440-445`) | No tie-break — undefined on near-ties | `moe_topk.cu:78,95`; `moe_topk_sigmoid.cu:74,91` (agent #8) | **MED** — directly matches L0→L24→L38 routing flips memory |
| A7 | **QKV GEMM fusion** | One fused GEMM Q+K+V+gate (`QKVParallelLinear(attn_output_gate=True)`) | 3 separate GEMMs | `paged_qkv.rs:42,68,80` (agent #2 D4, phase #2) | LOW — perf + tiny numerics from independent vs unified accumulator |
| A8 | **`gate_up_proj` fusion** | One `MergedColumnParallelLinear` | 2 GEMMs | phase #2 | LOW |
| A9 | **MoE SSOT violation** | Single block-FP8 engine | Two engines disagree: `moe_fp8_grouped_gemm.cu` uses K_PROMOTE=64 two-level promotion; `moe_shared_expert_fused_fp8*.cu` folds scale eagerly per-element | agent #9 finding 1 | LOW today, brittle tomorrow |
| A10 | **GDN Frobenius clamp** | None | `\|\|H\|\|_F > 1000` clamp (asymmetric: chunk1 yes, chunk2/3 missing) | `gated_delta_rule.cu:43-46, 161-202, 333, 618` (agent #10) | LOW for fix #2, audit-only for clamp itself |
| A11 | **FP8 cache scale dispatch** | Clean per-tensor | Runtime pointer-equality K_cache==V_cache (F3) | `inferspark_prefill_paged_fp8.cu:72-74` (agent #6 F3) | LOW — robustness only |

**Cleanly ruled out (no bug):**
- Embedding + first RMSNorm (agent #1)
- K/V/Q RMSNorm — both engines round to BF16 between norm and RoPE (agent #3)
- RoPE / MRoPE — Atlas slightly more precise (agent #4)
- BF16 KV cache write — pure memcpy (agent #5)
- O-proj GEMM — W8A8 vs W8A16 is intentional; Atlas more precise (agent #7)
- SSM state precision — Atlas FP32 vs vLLM BF16 default; Atlas safer (agent #10)
- Marconi SSM snapshots — FP32 byte-exact

---

## Class B — Architectural pipeline levers

These don't touch kernels but produce different outputs on the same input. They explain why even with kernel fixes Atlas wouldn't match vLLM bit-for-bit, and why opencode multi-turn collapses faster than the single-turn drift would predict.

### B1 — Atlas-only prompt mutations (phase #1)

| Mutation | File | Effect |
|---|---|---|
| **Historical-reasoning rehydration** | `msg_entry.rs:148-156`, `template.rs:62-71` | Prior `<think>` traces re-injected into next prompt. Under FP8, degraded traces feed back → multi-turn collapse mechanism |
| Loop-detector + `<tool_call>` logit-bias decay (conv-history-keyed) | `sampling_setup.rs:96-108` | Modifies logits based on prior turns |
| Tail-8 unclosed-`<think>` force-enable | `template.rs:142-163` | Toggles thinking mode based on prior output |
| Tool-error hint injection | `msg_entry.rs:108-114` | Injects assistant guidance into prompt |
| CWD `<environment>` injection | `msg_entry.rs:167-195` | Adds context block |
| Vacuous-system-prompt drop | `msg_entry.rs:207-216` | Removes empty system prompts vLLM keeps |
| **3-way sampling preset on `(tools_active, enable_thinking)`** | `sampling_setup.rs:53-72` | Atlas serves a **different** `(T, top_p, top_k)` than vLLM for the same client request — comparison is apples-to-oranges |

### B2 — Sampling-pipeline divergence (phase #4)

| | vLLM | Atlas |
|---|---|---|
| Transform count | 11 | 33 |
| Resident | GPU batched | CPU f32 |
| Logit-bias placement | Before penalty | **After penalty** (re-ranks low margin) |
| Grammar bitmask | AFTER penalty+bias | **BEFORE penalty+bias** |
| Min-p basis | Full-vocab softmax | Top-k-truncated probs |

**Six Atlas patches are explicit drift-MASKERS** (in-code comments say so):
- WS1 (440-WS-token -8.0 bias)
- AM1 (`lean://` attractor mask)
- WS2 (digit-ending WS gate)
- Mid-word `</think>` mask
- Top-n-sigma (doc says "NVFP4 quantization noise")
- C4v1 top-2 lift (disabled — caused its own drift)

**One Atlas patch is a drift AMPLIFIER:**
- `--adaptive-sampling` greedy-gate forces argmax at top-1 ≥ 0.9 — exactly the regime where FP8 noise pushed the wrong token up. Going greedy at low-margin **locks in** the bad selection.

### B3 — Decode-graph divergence (phase #3)

| | vLLM | Atlas |
|---|---|---|
| Decode graphs/total | ~67 (1 per batch size) | N × K (slot_idx × verify_shape) |
| Reason | `ssm_state_indices` is data-dependent kernel arg | SSM h_state/conv_state pointers baked into args |
| SSM rollback | In-kernel `num_accepted_tokens` arg | Rust-side per-layer snapshot/restore |

Perf only — explains MTP=0.1 tok/s in v3-v21 (now resolved differently).

---

## The failure mechanism, end-to-end

```
FP8 quant plumbing (A1+A2+A3+A4)
  → mid/late-layer probability distributions drift
  → MoE topk tie-break (A6) flips expert routing at deep layers (L20+)
  → MTP head reads with k_scale=1.0 (A5) → spec-draft drifts further
  → Adaptive-sampling greedy-gate (B2) locks in wrong token at low margin
  → Output degrades partway through turn
  → Atlas re-injects the degraded <think> trace into next prompt (B1)
  → Sampling preset (B1) compounds with different (T, top_p, top_k) vs vLLM
  → Multi-turn collapse → opencode harness 30% vs 100%
```

Every prior fix attempted (B1 fused-kernel K rounding, B2 v_memset, B5 BC=32→64) sits *downstream* of this chain. That's why none moved the needle.

---

## Recommended fix order (highest expected impact first)

**Phase 0 — verify the dominant lever (1 hour):**
- **C — Test with `disable_tool_grammar=true`** on the opencode harness. If cargo_valid hits 100%, ship a behavior-aware grammar-disable for `tool_choice="auto"` and skip 90% of the work below.

**Phase 1 — cheap, no-kernel-rewrite (1 day):**
1. **A5 — MTP head FP8 calibration** (one-line). Set MTP head k_scale/v_scale from parent layer at load.
2. **A6 — MoE topk deterministic tie-break** (~10 lines, 2 files). Add `|| (val == max && idx < winning_idx)` to argmax reductions.
3. **A2 — FP32 block-scale storage** (multi-site Rust refactor, no kernel change). Single SSOT `Fp8Weight` constructor stores FP32 scales. Update all 6 sites. Existing kernels already do `__bfloat162float(scale)` — making it pass-through.
4. **B2 — Add `--no-drift-patches` mode** to the harness. Bypass all 7 Atlas-only sampling patches + adaptive-greedy gate for clean A/B with vLLM.
5. **B1 — Match vLLM sampling preset and disable historical-reasoning rehydration** via env var for measurement runs.

**Phase 2 — kernel work (1-2 weeks):**
6. **A1+A3 — Native FP8 GEMM with per-token activation quant + FP32 epilogue scale**. Extend `fp8_gemm_t` (NVFP4-predequanted path) at `kernels/gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:436` to accept block scales in the epilogue and a per-token activation quant pre-pass. Replace `w8a16_gemm.cu` callsites across QKV, o_proj, in_proj_qkvz, MoE shared, MoE routed gate/up/down.
7. **A4 — Per-(seq, kv_head) FP8 KV descales**. Wire writer-side calibration through and update reader kernels. Should resolve the deep-layer FP8 KV cliff documented in `project_qwen36_phase2b_softmax_expf`.

**Stop spending on:** K-side BF16 rounding fusion (B1 fused kernel), v_memset (B2), BC=32→64 (B5). The just-finished 10-run b1_fused harness confirms 0/10 webserver_ok — these are at the wrong layer of the chain.

---

## Open questions for the next session

- Quantify what each lever buys in isolation. Phase 0 (C, grammar-disable) is testable in one harness run. Class A is testable per-fix. Class B needs the `--no-drift-patches` mode first.
- Decide whether to keep the `--adaptive-sampling` greedy gate at all once A1-A4 land — it may stop being needed.
- If C is dominant, the structural-tag grammar default needs to change to opt-in (`tool_choice="required"` only). Current always-on default is forcing every opencode/Claude-Code request through grammar walls vLLM never builds.
