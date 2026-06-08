# Causal-Pathway Audit: FP8 → NVFP4 Double-Quantization in Atlas

**Subject:** `Qwen/Qwen3.6-35B-A3B-FP8` on Atlas (branch `fix/in-think-tool-call-leak`)
**Date:** 2026-05-24
**Mode:** read-only forensic audit

---

## 1. TL;DR

Atlas DOES dispatch canonical FP8 kernels for routed MoE experts and full-attention
QKVO of `Qwen3.6-35B-A3B-FP8` via `set_fp8_experts` / `set_fp8_weights`. The
"nvfp4" tag in `Selected kernel target: (sm_121, qwen3.6-35b-a3b, nvfp4)` is the
**kernel-bundle name**, not a runtime selector — FP8 ptr-table kernels live
inside that bundle (`kernels/gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu:1182`).

The `quantize_to_nvfp4` lines in the boot log come from the subset of weights
that DO take the bad path. Findings, ranked by severity:

| # | Severity | Site | Impact |
|---|----------|------|--------|
| 1 | **HIGH** | SSM `in_proj_qkv` + `out_proj` decode: FP8 → BF16 → **NVFP4** | every linear-attn layer × every decode token; aligns with `project_qwen36_drift_gdn_clean.md` post-norm cliff and "L31-L39 deep-layer regression" memory entries |
| 2 | **MED** | MoE shared-expert weights duplicated as NVFP4 even when native-FP8 shared expert is wired | wasted memory + risk of wrong pointer in fallback paths (forward_batched k=1) |
| 3 | **MED** | MoE `gate` (router) projection: FP8 → BF16 → **NVFP4** | routing-decision noise; matches `project_qwen36_drift_moe_smoking_gun.md` "MoE expert routing diverges 8/8→3/8" |
| 4 | **LOW** | LM head: passed BF16-shaped pointer through `quantize_to_nvfp4` without FP8 dequant if checkpoint stores `lm_head` as FP8 (latent — most FP8 checkpoints leave lm_head as BF16) | catastrophic if triggered; usually dormant |
| 5 | **LOW** | MTP head `quantize_to_nvfp4` chain (BF16→NVFP4) for all projections regardless of FP8 native availability | only relevant with `--mtp-quantization nvfp4` |

Routed MoE experts and full-attention QKVO are NOT in the wrong path.

**Single highest-leverage fix:** Bypass the SSM NVFP4 round-trip in
`weight_loader/qwen35/load_layers/linear_attn_arms.rs` for `Fp8Dequanted`
(NVFP4 is built unconditionally at lines 176-192; the parallel FP8 prefill copy
at 213-235 is decode-blind).

---

## 2. Chain of decisions for one Qwen3.6-35B-A3B-FP8 request

```
[boot]
  spark-server/main_modules/serve.rs:68    ptx_for_config("qwen3_6_moe", 2048) → nvfp4 bundle
  spark-server/main_modules/serve.rs:83    "Selected kernel target ... nvfp4 (90 modules)"  ← kernel-bundle name, NOT runtime quant
  factory/build.rs:98                      loader.load_layers(...)   → qwen35::load_layers
[weight_loader/qwen35/load_layers.rs]
  :70   detect_nvfp4_variant            → returns Fp8Dequanted        (nvfp4_detect.rs:51)
  :76   quant_format = QuantFormat::Fp8
  :81   native_fp8 = true
  :128  ATLAS_FORCE_NVFP4_MOE? → false   so skip_nvfp4_experts = true
  :139  load_moe_qwen35(..., skip_routed_experts=true)
       └─ ssm_qwen35.rs:75 load_moe_qwen35:
           :89   gate = dense(...)                                    ← FP8 byte ptr survives untouched here
           :151  gate_nvfp4 = quantize_to_nvfp4(&moe_weights.gate, ...) ← BUG #3: lm_head router BF16→NVFP4, but gate ptr is FP8 bytes
           :184  shared_expert = load_expert(...) → variant==Fp8Dequanted →
                  quantized_from_fp8 → BF16 → NVFP4               ← BUG #2 (NVFP4 shared_expert built, but unused)
           :188  for e in routed:  experts.push(NULL)                  ← OK
  :160  MoeLayer::new(...)
  :183  load_moe_qwen35_fp8_experts                                    ← OK, routed experts FP8
  :197  load_fp8_block_scaled_as_fp8weight(shared_expert/{gate,up,down}_proj) ← OK, FP8 shared expert
  :215  moe_layer.set_fp8_experts(&fp8_experts, shared_fp8, gpu)       ← FP8 path enabled
  ─ FullAttention layers ─
  :226  LayerType::FullAttention if native_fp8 =>                      ← FP8 attention arm taken
  :255  load_qkvo_tp(load_fp8_proj)                                    ← FP8 QKVO loaded zero-copy
  :298  layer.set_fp8_weights(...)                                     ← FP8 path enabled for full attn
  ─ LinearAttention (SSM) layers — 30 of 40 layers ─
  :344  build_linear_attention_nvfp4                                   ← BUG #1: name says nvfp4 unconditionally
       └─ linear_attn_arms.rs:147 load_ssm_qwen35 (Fp8Dequanted) → dense_auto → BF16 (good so far)
        :176-190 quantize_to_nvfp4(qkvz_dense | out_proj)              ← BUG #1 fires (decode path will be NVFP4)
        :203-235 if Fp8Dequanted { bf16_to_fp8(...) for prefill_only }  ← FP8 prefill path built in PARALLEL,
                                                                          but only `set_fp8_prefill_only_weights`
                                                                          installs it — decode still NVFP4
[factory/build.rs]
  :101  lm_head = loader.load_lm_head(store, &config) → qwen35.rs:56 dense("lm_head.weight")
                                                                       ← BUG #4: dense, not dense_auto.
                                                                          If checkpoint stores FP8 lm_head, raw FP8 bytes are passed downstream as BF16
  :144  skip_lm_head_quantization() = false for qwen3.6
  :148  quantize_to_nvfp4(&lm_head, ...)                                ← treats whatever the pointer is as BF16
  :102  loader.load_mtp_weights_multi
  :120  effective_mtp_quant = checkpoint ignores mtp.*  ? Bf16 : mtp_quant  ← OK-ish gate
[per-decode-step runtime]
  layers/moe/forward_prefill.rs:27  self.fp8_gate_weight_ptrs.is_some() ? FP8 path : NVFP4 path
                                                                       ← chooses FP8 for routed experts (correct)
  layers/moe/forward.rs:217          (same)                            ← decode FP8 path taken (correct)
  Qwen3SsmLayer::forward_decode      uses qkvz_nvfp4 / out_proj_nvfp4   ← BUG #1: SSM decode is always NVFP4
```

---

## 3. Per-bug table

| Bug | File:line | Currently dispatched | Canonical kernel that exists | Difficulty | Expected impact |
|-----|-----------|---------------------|------------------------------|-----------|-----------------|
| **#1 SSM decode NVFP4** | `weight_loader/qwen35/load_layers/linear_attn_arms.rs:176-190` | `quantize_to_nvfp4(qkvz_dense, ...)` and `out_proj` → NVFP4 used for ALL paths except a parallel FP8-prefill-only override | `load_fp8_block_scaled_as_fp8weight` + `w8a16_gemv` / `w8a16_gemm` already used by `build_linear_attention_fp8` (file marks itself "currently unused", `load_layers.rs:343` always takes `_nvfp4` arm) | **Medium**: build_linear_attention_fp8 already exists at line 24 of same file. Need to: (a) flip dispatch to fp8 arm when `variant==Fp8Dequanted`, (b) verify `set_fp8_weights` on `Qwen3SsmLayer` reaches the SSM decode/verify GEMV path. | High. Memory `project_qwen36_phase2b_softmax_expf.md` already attributes deep-layer regression to FP8-KV+NVFP4-weight noise on out_proj. Eliminating the BF16→NVFP4 step on `out_proj` should match the existing `ATLAS_GDN_BF16_WEIGHTS=1` benefit but with FP8 precision (lower memory). |
| **#2 dead NVFP4 shared-expert** | `weight_map/ssm_qwen35.rs:184` + `loaders_moe.rs:32-60` | Loaded via `quantized_from_fp8` → BF16 → NVFP4 then never consumed (forward_prefill_fp8 + forward.rs use `fp8_shared_expert`) | n/a — fix is to elide the load when `native_fp8 && !force_nvfp4_moe` | **Trivial**: thread a `skip_shared_expert: bool` like `skip_routed_experts` already does. | Saves a few hundred MB and one source of quant noise if any fallback path ever consults the NVFP4 shared expert. |
| **#3 router gate FP8→NVFP4** | `weight_loader/qwen35/load_layers.rs:151-159` + `ssm_qwen35.rs:89` | `dense(...)` returns raw FP8 bytes; `quantize_to_nvfp4` then treats them as BF16 if the checkpoint stores `gate.weight` as FP8. For Qwen3.6 FP8 the gate is typically **BF16** in checkpoint (small enough to leave alone) — verify with `WeightDtype` of `mlp.gate.weight`. If it IS FP8 in checkpoint, this is silently miscomputing the entire routing distribution. | `dense_auto` (already exists) — would correctly dequant FP8 to BF16 before NVFP4 quantization. | **Trivial**: swap `dense` → `dense_auto`. | If the gate is FP8 in checkpoint, this is the root of the routing divergence reported in `project_qwen36_drift_moe_smoking_gun.md` (8/8→3/8 expert overlap collapse). Even if gate is BF16 in this checkpoint, applying `dense_auto` everywhere is the defensive fix. |
| **#4 lm_head FP8 latent corruption** | `weight_loader/qwen35.rs:60-68` + `factory/build.rs:148` | `dense(store, "lm_head.weight")` returns raw pointer regardless of dtype; if the checkpoint's `lm_head` is FP8E4M3 + has `weight_scale_inv`, the FP8 bytes are pumped into `quantize_to_nvfp4` as if BF16 — catastrophic top-token corruption. | `dense_auto` (already exists). | **Trivial**: swap `dense` → `dense_auto` in `load_lm_head` for all FP8-capable loaders (qwen35, qwen3, qwen35_dense). | Latent. Qwen3.6 FP8 typically ships lm_head as BF16; if it does, no current impact. But it's a silent footgun for the next FP8 checkpoint that quantizes lm_head. |
| **#5 MTP head BF16→NVFP4** | `layers/mtp_head/new.rs:44-53` + `:81-118` | Every MTP projection goes through `quantize_to_nvfp4` when `quant==Nvfp4`. For FP8 source weights this is the same FP8→BF16→NVFP4 chain as the main model. | An `MtpQuantization::Fp8` variant exists (mtp_head.rs:29, line 180 in new.rs) but the user must opt in via `--mtp-quantization fp8`. | **Low**: default `--mtp-quantization` based on `native_fp8`. | Only matters for users who turn on `--speculative`. Spec drafts produced from a doubly-quantized MTP head would amplify Bug #1's drift into wholesale rejection. |
| **Dead code** | `weight_map/ssm_qwen35.rs:262-279` | `_shared_fp8 = ...` is computed and discarded (sigil `_`). The caller at `qwen35/load_layers.rs:197-214` re-loads the same tensors. | n/a — duplicate I/O. | Trivial: return shared_fp8 from `load_moe_qwen35_fp8_experts` or delete the bind. | None on correctness; ~2× the FP8 shared-expert load time. |

---

## 4. Recommended fix order

1. **Bug #1** (SSM decode NVFP4). Highest expected quality lift. The plumbing
   (`build_linear_attention_fp8`, `Qwen3SsmLayer::set_fp8_weights`,
   `qkvz_fp8 + out_fp8` SSM GEMV) is **already implemented**; it's behind a
   dead-coded gate (`load_layers.rs:334-342` comment: "permanently
   short-circuited"). Re-enabling it is a one-line dispatch flip plus removal
   of the parallel `_nvfp4` build in `build_linear_attention_nvfp4`. Verify
   against `ATLAS_GDN_BF16_WEIGHTS=1` numerics — they should match or beat the
   BF16 dense path while saving ~2× weight memory.
2. **Bug #3** (`dense` → `dense_auto` on router gate). Even if the gate is
   BF16 in *this* checkpoint, this defensive fix prevents the silent FP8-as-BF16
   misread that has already been observed bites elsewhere
   (cf. `project_qwen36_numerical_drift_2026_05_23.md`).
3. **Bug #4** (LM head `dense_auto`). Same one-line fix, eliminates a class of
   latent corruption for any future FP8 checkpoint that quantizes the head.
4. **Bug #2** (dead NVFP4 shared-expert). Memory savings, cleanliness.
5. **Bug #5** (default `--mtp-quantization` to `fp8` when `native_fp8`). Only
   after Bug #1 — otherwise the MTP head still feeds an NVFP4-corrupted target.

---

## 5. Non-quant red flags noticed in passing

- `weight_map/ssm_qwen35.rs:263` — `_shared_fp8` discarded (Dead code).
- `linear_attn_arms.rs:267-269` — installs FP8 prefill weights via
  `set_fp8_prefill_only_weights`. Decode is never routed to FP8 SSM weights
  even though the FP8 buffers exist (lines 215-232).
- `attention_arms.rs:83` collapses `Standard | Fp8Dequanted | Bf16Raw` to one
  arm. Unreachable for Qwen3.5 (peeled off at `load_layers.rs:226`), so it's
  load-bearing only for non-FP8 checkpoints. Brittle: a future caller that
  enters with `Fp8Dequanted && !native_fp8` (e.g. an `ATLAS_FORCE_NVFP4_MOE`
  variant that spilled into attention) would read FP8 bytes as BF16. Add a
  `debug_assert!`.
- KV cache: `attn_layer_dtypes` is independent of weight quant. The FP8-KV
  cliff at L35–L39 in `project_qwen36_phase2b_softmax_expf.md` interacts
  with Bug #1: SSM-decode NVFP4 noise compounds with FP8-KV rounding at deep
  layers, so fixing Bug #1 should reduce but not eliminate that regression.
- `factory/build.rs:144-160` always BF16→NVFP4 the LM head when
  `!skip_lm_head_quantization()`. Atlas has `gemv_fp8w`; an FP8 lm_head
  could stay FP8.

---

## Appendix: Canonical FP8 kernels (cross-reference)

| Kernel | File | Used by |
|--------|------|---------|
| `moe_fp8_grouped_gemm_ptrtable_t/k/v2_k` | `kernels/gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu:1182` | `forward_prefill_fp8.rs`, `forward_batched.rs`, `forward_k2/k3.rs` |
| `moe_expert_gate_up_shared_fp8` / `moe_expert_silu_down_shared_fp8` | `kernels/gb10/common/` | `forward.rs:217-260` |
| `w8a16_gemv` / `w8a16_gemm` (FP8w × BF16a) | `kernels/gb10/common/gemv_fp8w` | `Qwen3AttentionLayer::set_fp8_weights`, FP8 shared expert |
| `bf16_to_fp8` | `kernels/gb10/common/w4a16` | `linear_attn_arms.rs:213` — SSM prefill-only |
| `fp8_gemm_n128` | `kernels/gb10/common/` | SSM prefill (Bug #1 partial mitigation) |

Asymmetry: every full-attention layer and every routed MoE expert already runs
on canonical FP8. Only SSM layers (30/40), the router gate, the dead NVFP4
shared expert, and any FP8 lm_head sit wrong. Fix Bug #1 and the runtime
profile changes from "10 FP8 + 30 NVFP4-from-FP8 layers" to "40 FP8 layers".
