# Arch-diff #2 — QKV Projection (Qwen3.6-A3B-FP8, block-scaled FP8)

## 1. Fusion topology

| Engine | Construction | GEMM |
|--------|--------------|------|
| vLLM   | `QKVParallelLinear(... attn_output_gate=True)` at `qwen3_next.py:731` builds a **single fused QKV+Gate** projection (`total_num_heads*(1+attn_output_gate)` for Q/Gate, plus K, V). Output then `qkv.split([q_size*2, kv_size, kv_size], dim=-1)` (`qwen3_next.py:787`). | One block-FP8 GEMM. |
| Atlas  | Q-gate, K, V are **three independent GEMMs** dispatched sequentially in `Qwen3AttentionLayer::prefill_attention_paged_qkv` (`crates/spark-model/src/layers/qwen3_attention/prefill/paged_qkv.rs:42-89`). Each calls `prefill_one_proj` → `ops::w8a16_gemm_t` (`paged_qkv.rs:130`). | Three separate block-FP8 GEMMs. |

Math equivalent; Atlas pays 3× GEMM launch + activation re-read. Perf only.

## 2. Activation handling (CRITICAL DIVERGENCE)

- **vLLM (block-FP8 path):** `Fp8LinearMethod.apply()` at `fp8.py:673-682` dispatches to `W8A8BlockFp8LinearOp.apply` (`fp8_utils.py:305`). That op **always quantizes activations to FP8 with per-token-1×128 group scales** before the GEMM:
  - `act_q_group_shape = GroupShape(1, weight_block_size[0])` (`fp8.py:379`, == `GroupShape(1,128)`).
  - In `_run_cutlass`/`_run_triton` paths (`fp8_utils.py:359, 406`): `q_input, input_scale = self.input_quant_op(input_2d)` → `QuantFP8 / per_token_group_quant_fp8` (`input_quant_fp8.py:73`) producing FP8 activations + FP32 per-token×128-group scales.
  - The CUTLASS/DeepGEMM kernel consumes `(A_fp8, a_scale[M, K/128], B_fp8, b_scale[N/128, K/128])` and accumulates in FP32, applying both scales per K-block.
- **Atlas:** activations remain **BF16** through the GEMM. `w8a16_gemm_t.cu:151-157` declares `A` as `const __nv_bfloat16*` and the MMA shape is `m16n8k16.row.col.f32.bf16.bf16.f32` (`w8a16_gemm_t.cu:128`). The dequanted FP8 weights are upcast to BF16 in smem (`w8a16_gemm_t.cu:217`: `smem_B[k][n] = __float2bfloat16(dequant_val)`), then MMA runs **BF16 × BF16 → FP32**, never FP8 × FP8.

Atlas does `bf16 = fp8_lut * scale; bf16 → mma` — a 7-bit-mantissa round on every dequanted weight before MMA. vLLM keeps E4M3 magnitude and applies block scale in FP32 at accumulation. On large block-scale magnitudes (K/V in late layers) this is the largest weight-drift source on QKV; matches the L31–L39 attn-regression fingerprint (`project_qwen36_phase2b_softmax_expf.md`).

## 3. Weight-scale handling

- **vLLM:** `weight_scale` is FP32, shape `[N/128, K/128]` or its transpose (`fp8.py:601-621`); consumed in FP32 in the CUTLASS scaled-mm.
- **Atlas:** `block_scale_t` is **BF16**, shape `[K/128, N/128]` (transposed) (`w8a16_gemm_t.cu:153`, dequant at `:214` `__bfloat162float(block_scale_t[...])`). Loaded into FP32 register before multiply — fine. **But the checkpoint scale itself is downcast to BF16 at load time**, losing ~16 bits of mantissa vs the FP32 scale vLLM keeps. Second source of drift.

## 4. Accumulator / output dtype

Both: FP32 accumulator, BF16 store with single round-down. `w8a16_gemm_t.cu:238-241` matches vLLM CUTLASS epilogue. No divergence.

## 5. K/V output buffer aliasing (`ssm_qkvz`)

`paged_qkv.rs:67-80`:
- K written at `ssm_qkvz + 0`.
- V written at `ssm_qkvz + num_tokens * kv_dim * 2`.

Sizing in `spark-runtime/src/buffers/sizes.rs:166-170` takes `max(... , m * 2 * kv_heads * hd * bf16, ...)`. K+V capacity guaranteed, ranges non-overlapping. **Aliasing safe.** `ssm_qkvz` is shared scratch with SSM in-proj and MoE shared-up; uses sequential within a layer, so reuse is sound (hot-spot, not a bug).

## 6. Q-gate output buffer

Atlas writes `q_proj_dim = 2*q_dim` (Q‖gate interleaved per-head) to `qkv_output` (`paged_qkv.rs:41-65`). vLLM's `QKVParallelLinear` produces `[q_size*2, kv_size, kv_size]` concat (`qwen3_next.py:734, 787-794`) then `view + chunk(2, dim=-1)` to split Q and gate. Both flavours interleave Q and gate per-head before applying `q_norm` only to Q half. Equivalent.

---

## Atlas-side divergences flagged

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| **D1** | **HIGH (drift)** | `kernels/gb10/common/w8a16_gemm_t.cu:217` and `w8a16_gemm.cu:215` | Dequanted FP8 weight rounded to BF16 (`__float2bfloat16`) before MMA. vLLM keeps E4M3 magnitude and applies block-scale in FP32 inside CUTLASS scaled-mm. Per-element 7-bit-mantissa round on every weight. |
| **D2** | MED (drift) | `kernels/gb10/common/w8a16_gemm_t.cu:153, 214` | `block_scale_t` is stored as BF16 (Atlas load-time downcast). vLLM keeps FP32 weight_scale. |
| **D3** | MED (no a_scale) | `paged_qkv.rs:130`, `w8a16_gemm_t.cu:151` | No per-token activation FP8 quantization. vLLM does per-token×128 quantization (`fp8.py:379`, `fp8_utils.py:359`) and uses FP8×FP8 → FP32 MMA. Atlas BF16×BF16→FP32. Different MMA target precision — typically Atlas would be *more* precise on activations, but combined with D1/D2 the kernel-level numerics differ from vLLM by more than rounding noise. |
| **D4** | LOW (perf) | `paged_qkv.rs:42, 68, 80` | Three independent GEMMs instead of one fused QKV+Gate. Math equivalent, latency only. |

**Smoking gun for native-FP8 drift vs vLLM:** D1 + D2. Atlas's `w8a16_gemm_t` (and non-transposed sibling) dequants FP8→FP32 then narrows to BF16 **inside the inner loop** before MMA. vLLM never narrows: FP32 scale, FP8 E4M3 weight, FP32 accumulate inside CUTLASS scaled-mm. To match vLLM, Atlas needs an `m16n8k32.e4m3.e4m3.f32` MMA (already used in `fp8_gemm_t` at `kernels/gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:436`) plus activation pre-quant and per-K-block scale fusion in epilogue. `prefill_weights.rs:99-103` documents the gap: no FP8-MMA-with-block-scales kernel exists, so Atlas deliberately falls through to BF16 dequant. **This is the architectural bug.**
