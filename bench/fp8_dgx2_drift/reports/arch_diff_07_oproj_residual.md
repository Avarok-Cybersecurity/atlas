# arch_diff_07 — o_proj + post-attention residual addition

Compares the attention-output projection and the residual/norm fusion that
immediately follows it for Qwen3-Next FP8 (compressed-tensors W8A8) between
vLLM `main` and Atlas `atlas-mtp` on GB10 (SM121).

## 1. Pipeline overview (Qwen3-Next FP8)

| Stage | vLLM | Atlas |
|------|------|------|
| sigmoid gate × attn_out | `qwen3_next.py:941-943` (`attn_output * sigmoid(gate)`) | `prefill/paged.rs:496-512` (`sigmoid_gate_mul_batched`) |
| o_proj GEMM | `qwen3_next.py:945` → `compressed_tensors_w8a8_fp8.py:191-206` → `cutlass_scaled_mm` | `prefill/paged_oproj.rs:41-56` → `ops::w8a16_gemm` |
| TP all-reduce | inside `RowParallelLinear` (NCCL) | `trait_impl/prefill_inner.rs:157-177` |
| residual add + post-attn norm | `qwen3_next.py:1070` → `GemmaRMSNorm.forward_native` (torch.compile) | `trait_impl/prefill_inner.rs:258-271` → `ops::residual_add_rms_norm` |

## 2. o_proj GEMM precision

vLLM `cutlass_scaled_mm` path (compressed-tensors dynamic FP8 activations):
- Per-token dynamic FP8 quantization of `attn_output` (`scaled_fp8_quant`),
- FP8×FP8 GEMM with FP32 accumulator → BF16 output
  (`scaled_mm/cutlass.py:170-172`, `out_dtype=x.dtype`).
- DevQuasar config: `quantization_config.config_groups.group_0` has
  `input_activations.dynamic=true` (HF config.json) — confirmed runtime
  per-token FP8 quant.

Atlas `w8a16_gemm` (`kernels/gb10/common/w8a16_gemm.cu:5-13`):
- Activations stay **BF16** (no FP8 quantization of input).
- Weight FP8 dequant via 256-entry LUT × per-`[128,128]` block scale
  (`w8a16_gemm.cu:27-148`).
- BF16×BF16 MMA with FP32 accumulator
  (`mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32`, line 132), final cast
  to BF16 (`__float2bfloat16`, lines 236-239).

**Divergence (intentional)**: Atlas avoids the FP8 activation quantization
step, trading bandwidth for precision. This is identical in arithmetic
shape (FP32 accum → BF16 output) but **different rounding**: vLLM rounds
activations to FP8 (E4M3 has 3-bit mantissa) per token before the dot
product, Atlas keeps the 8-bit BF16 mantissa. Both produce BF16 `attn_out`.

## 3. Residual stream dtype

Atlas exposes `config.use_fp32_residual()` (e.g. `prefill_inner.rs:427`,
`init.rs:350-355,365-374`). Kernel selection branches in
`init.rs:350-374`: when enabled, `f32_residual_add` /
`residual_add_rms_norm_f32` replace the BF16 versions. The FP32 residual
is gated on `model_type == "gemma4"` for the `_abs` variant.

For **Qwen3-Next FP8** the config does **not** enable
`use_fp32_residual()` (Gemma-4-specific gate in current source, see
`model/trait_impl/prefill_b/embed_chunk.rs:40` and friends), so Atlas
runs the **BF16 residual stream** kernels — matching vLLM, which keeps
`residual` BF16 throughout (`models/qwen3_next.py:1037-1041`).

## 4. Fused residual-add + RMSNorm

vLLM `GemmaRMSNorm.forward_cuda` (`layernorm.py:473-489`) dispatches to
`torch.compile`-wrapped `_forward_static_with_residual`
(`layernorm.py:433-456`). For BF16 inputs (Qwen3-Next) the residual add
is `x + residual` in BF16 (line 445, `else` branch), then the variance
path upcasts via `x.float()`. **`GemmaRMSNorm` does NOT call
`vllm._custom_ops.fused_add_rms_norm`** (lines 84-90) — that fused
kernel is only used by the base `RMSNorm` class (`layernorm.py:362-364`).

Atlas calls `ops::residual_add_rms_norm`
(`kernels/gb10/common/rms_norm.cu:261-344`). The kernel does:
- Pass 1 (lines 281-299): `h[i] = bf16(fp32(h[i]) + fp32(s[i]))`,
  accumulates `sum_sq` in FP32.
- Pass 2 (lines 330-343): writes `out = h * rms * (1+w)` in FP32 then
  truncates to BF16, **and** copies the post-add `h_packed` (BF16) into
  `residual`.

**Divergence (subtle)**: vLLM does the residual add in pure BF16
arithmetic (BF16 hardware add, single rounding to BF16). Atlas
explicitly upcasts each operand to FP32 with `unpack_bf16x2`, adds in
FP32, and rounds once to BF16 (line 290). The two are nearly identical
because BF16 has FP32's exponent range; the only theoretical difference
is the rounding mode of the BF16 hardware add (RNE on Blackwell) vs the
FP32→BF16 cast (`__float2bfloat16` uses RNE on SM12x). **Net result:
1 ulp of BF16 at most, and equivalent for non-cancelling adds.**

## 5. post_attention_layernorm weight semantics

Both engines use `(1 + weight)` scaling — Qwen3-Next inherits Gemma's
RMSNorm convention. vLLM `GemmaRMSNorm._forward_static_with_residual`
line 454: `x * (1.0 + weight.float())`. Atlas `residual_add_rms_norm`
line 335: `pack_bf16x2(xv0 * rms * (1.0f + wv0), …)`. Match.

`eps` and reduction order: vLLM uses `mean(dim=-1)` then `rsqrt`.
Atlas computes `sum_sq / hidden_size + eps` then `rsqrtf`
(line 322). Match.

## 6. TP all-reduce ordering

Both engines all-reduce **after** o_proj GEMM but **before** the
residual add (vLLM via `RowParallelLinear` post-comm,
`prefill_inner.rs:157-177` for Atlas). Match.

## 7. Findings / risk

1. **W8A8 vs W8A16 o_proj** (Section 2) is the largest arithmetic
   divergence. Per-token FP8 activation quantization in vLLM rounds the
   gated attention output before the dot-product; Atlas does not. On the
   FP8 DGX-2 drift mission this is a **suspect contributor** to
   late-layer drift — Atlas computes a *more accurate* o_proj than
   vLLM. **Not a bug**; flag for investigation if reference is vLLM.

2. **Residual-add precision** (Section 4) is essentially equivalent;
   no action.

3. **Atlas op_dump hook** (`paged_oproj.rs:135-151`) dumps the *last*
   token's o_proj output; comparing against HF
   `full_attention.o_proj.forward` is correct for the last token only.

4. **No fused-add-norm in vLLM Qwen3-Next path**: confirmed —
   `GemmaRMSNorm.forward_cuda` is `torch.compile(forward_native)`, NOT
   `ops.fused_add_rms_norm`. Both engines materialise the post-add
   BF16 residual in memory; neither holds an FP32 residual stream for
   Qwen3-Next. Earlier optimisation lore ("vLLM keeps FP32 throughout
   the fused norm") **does not apply to Qwen3-Next / Gemma-style RMS**.

No bugs detected in this stage.
