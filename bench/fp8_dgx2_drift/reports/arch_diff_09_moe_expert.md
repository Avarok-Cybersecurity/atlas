# Arch-Diff #09 — MoE Expert FFN (FP8): vLLM vs Atlas

Compares `gate_proj → up_proj → silu+mul → down_proj → weighted topK reduce`
on FP8 block-scaled experts between vLLM (Triton/DeepGEMM/CUTLASS SM100) and
Atlas (sm_121 hand-rolled m16n8k16 BF16-MMA after smem dequant).

## 1. GEMM tile shape & promotion structure

| Path | Tile (M×N×K) | K_PROMOTE | Accum |
|---|---|---|---|
| vLLM Triton `fused_moe_kernel` block-FP8 (`fused_moe.py:316-516`) | `BLOCK_SIZE_{M,N,K}` autotuned (typ. 64×64×128) | scale folded inside K-loop per tile: `tl.dot(a,b) * a_s * b_s` (line 510) | FP32, cast at store (line 533) |
| vLLM DeepGEMM `m_grouped_fp8_gemm_nt_contiguous` (`deep_gemm_moe.py:283`) | 128×N×128 grouped, Hopper WGMMA-FP8 | promote-every-4-WGMMA | two-level FP32 |
| vLLM CUTLASS SM100 `blockwise_scaled_group_mm_sm100.cu:253-265` | `MmaTileShape=128×128×128`, `ClusterShape=1×1×1`, UMMA-FP8 native | `Sm100BlockwiseScaleConfig<1,128,128,K,K>` | FP32 in pipeline |
| Atlas `moe_fp8_grouped_gemm.cu:26-36, 211-219, 273-286` | **64×64×16** m16n8k16 BF16-MMA | **K_PROMOTE=64** two-level (line 275) | `inner_acc` FP32 → `outer_acc` FP32 |
| Atlas `moe_shared_expert_fused_fp8.cu:179-234` (gate_up & silu_down) | 1×2×16 GEMV | **none** — scale folded per-element: `wf1_0 = s_lut[byte] * sc1` (line 214) | single FP32 acc |

**Key divergence — Atlas's two engines disagree internally.** The grouped
GEMM uses true two-level promotion (DeepGEMM-style). The shared-expert
kernels (`moe_shared_expert_fused_fp8.cu:184-214`,
`..._fp8_batch2.cu:197-225`) apply scale eagerly per element. Mathematically
equivalent (`Σ s·a·w == s·Σ a·w` in FP32), but only because all scaled-load
math is FP32 and the BF16→FP32 scale cast is exact. **If K_PROMOTE is ever
lowered or scales move to FP32 storage with rounding, the shared-expert
kernel will not track the grouped-GEMM's numerics.** SSOT violation.

## 2. Dequant point

- **vLLM Triton block-FP8**: FP8 stays as input to `tl.dot`. On Hopper this
  is native `mma.sync.…e4m3`; scale applied to FP32 acc after each tile dot
  (`fused_moe.py:510`).
- **vLLM CUTLASS SM100 / DeepGEMM**: native UMMA-FP8 / WGMMA-FP8.
- **Atlas**: sm_121 has no FP8 MMA. FP8 byte → constant LUT
  `E4M3_LUT_GMOE[256]` (`moe_fp8_grouped_gemm.cu:38-103`) → BF16 in smem →
  `mma.sync.m16n8k16.f32.bf16.bf16.f32` (line 138-150). LUT→BF16 is
  **lossless** (FP8-E4M3 has 3 mantissa bits; BF16 has 7), comment
  line 204-210.

## 3. Block-scale storage

- vLLM block-FP8 scales are **FP32** (Triton signature treats `Bs` as
  float; DeepGEMM/CUTLASS scale tensors are FP32).
- Atlas stores `block_scale` as **BF16** in HBM (`moe_fp8_grouped_gemm.cu:163`
  `const __nv_bfloat16* S_exp`, cast to FP32 once per K-block at line 277).

This loses ~9 bits of mantissa vs vLLM. For Qwen3.6 expert scales (typically
O(1)–O(10)), BF16's ~3-decimal-digit precision sets a rounding floor on
every expert-output element. **Flag**: plausibly the residual drift
contributor at deep layers (cf. memory note `project_qwen36_drift_moe_smoking_gun`).
Worth A/B-testing FP32 scale storage (weight-loader change only).

## 4. Activation quantization

- **vLLM block-FP8 MoE**: per-token-per-128 FP8 activation quant
  (`per_token_group_quant_fp8` at `deep_gemm_moe.py:290-292`); inner kernel
  multiplies `tl.dot(a,b) * a_scale[:,None] * b_scale[None,:]`
  (`fp8_utils.py:769`, `fused_moe.py:510`).
- **Atlas**: activations stay **BF16**. No `a_scale` anywhere in the
  grouped-GEMM or shared-expert kernels.

Divergence by design (sm_121 has no FP8 MMA, no reason to quantize
activations), but it means **Atlas accumulates BF16·BF16, vLLM accumulates
FP8·FP8**. Atlas should be *more* accurate on the activation side, *less*
accurate on the scale side. Net direction: ambiguous.

## 5. SiLU + gating

vLLM `csrc/activation_kernels.cu:14-38`: `silu(x) = x/(1+expf(-x))` in
FP32, then `silu_and_mul` multiplies by `y`. Identical to Atlas
`moe_silu_mul.cu:23-27`. No divergence.

vLLM DeepGEMM path **re-quantizes** the SiLU output to FP8 before the
down-proj (`deep_gemm_moe.py:290`). Atlas leaves it BF16. So vLLM
down-proj input precision is FP8; Atlas's is BF16. Atlas advantage here.

## 6. Down-proj weighted reduce

vLLM `fused_moe_kernel` writes `acc * moe_weight` into expanded `[M,topk,N]`
(line 524-526); a follow-up `moe_sum` or `deepgemm_unpermute_and_reduce`
(deep_gemm_moe.py:301) sums across topk. Atlas
`moe_unpermute_reduce_indexed` (`moe_permute.cu:94-116`) does the fused
weighted sum in **FP32** (`acc += w * val`), final cast to BF16. No
divergence.

## 7. Shared expert

Both treat the shared FFN as a parallel dense path. Atlas runs it via two
`w8a16_gemm` calls plus `silu_mul`
(`crates/spark-model/src/layers/moe/forward_prefill_fp8.rs:48-98`) — i.e.
**the shared expert uses per-channel FP8 dense GEMM, NOT the block-scaled
MoE FP8 kernel.** vLLM uses whatever quant the shared layer is stored as.
Functionally identical.

## Bugs / flags

1. **SSOT violation**: `moe_fp8_grouped_gemm.cu` uses two-level promotion;
   `moe_shared_expert_fused_fp8*.cu` use per-element eager scale-folding.
   Equivalent today, but they are not the same numerical pipeline and any
   future K_PROMOTE / scale-dtype change must touch both.
2. **BF16 block scale storage** (`moe_fp8_grouped_gemm.cu:163, 277`): a
   real precision regression vs all three vLLM backends. Uncommented.
   Suggest one-shot A/B with FP32 scale storage to quantify deep-layer
   drift contribution.
3. **No activation quant**: by design — flags only because it makes the
   FP8 drift comparison vs vLLM not apples-to-apples. Atlas should be
   *more* accurate per-layer if scale-storage is fixed (#2).
4. **K_PROMOTE=64 vs DeepGEMM 128** (line 31-36 comment): identical in
   FP32; one extra scale-FMA per 64K. No action.
5. **`max_m_tiles` heuristic** (`forward_prefill_fp8.rs:202-203`):
   worst-case `(n*top_k).div_ceil(64)` correctly sized. The historical
   `avg*2` bug (line 192-202) silently dropped rows from heavy experts.
   Validated.
6. **Buffer-zero pre-GEMM** (`forward_prefill_fp8.rs:218-236`): required
   — kernel only writes `m_idx < M_expert` rows; unzeroed leftover rows
   would contaminate `moe_unpermute_reduce_indexed`. Validated.
