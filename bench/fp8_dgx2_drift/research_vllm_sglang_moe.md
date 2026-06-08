# Research: vLLM / SGLang MoE FP8 Prefill Precision

**Date**: 2026-05-25
**Target bug**: Atlas `moe_fp8_grouped_gemm` produces `ssm.moe_out` cosine **0.91983** at L20 vs HF[BF16-unquant] when prefill token count > 64. Reference (HF, FP32 accum) is ~1.0; vLLM/SGLang are visibly cleaner.

---

## 1. Atlas's current kernel (what we do today)

`kernels/gb10/common/moe_fp8_grouped_gemm.cu` (v1 + v2 coalesced):

- **Tile**: 64 × 64 × 16 (M × N × K), 4 warps, 1 CTA per (expert, m_tile, n_tile).
- **MMA**: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` — BF16 inputs, **FP32 accumulator** (good — matches everyone else).
- **Dequant path**: at every K-step (every 16 K elements), each thread reads one FP8 byte → LUT[256] in constant memory → multiply by BF16 block scale (`__bfloat162float`) → **immediately round to BF16** (`__float2bfloat16`) → write to smem_B. Then BF16×BF16→FP32 MMA.
- **Block scale layout**: `[N/128, K/128]` 2D BF16 (matches Qwen `weight_block_size=(128,128)`).
- **Accumulator**: FP32 the whole K-loop, single-level (no two-stage promotion). Stored back to BF16 once per output element.

The MMA accumulator is fine. **The losses are upstream of the MMA**:

1. **Dequant rounded to BF16 before MMA** — `dequant_val(f32) → __float2bfloat16 → smem_B(bf16)`. Every weight element loses ~8 bits of mantissa *before* the dot product even runs. Across K=2048 with topk=8 routed experts, the rounding error in B compounds linearly (random walk: ~√K ≈ 45× per-element BF16 ULP → cos drops ~1e-3 to ~1e-2 per layer; across 38 hybrid layers, that integrates to exactly the 0.92 we see).
2. **Block scale stored in BF16** — Qwen ships the scales in BF16 on disk, but they could be kept in FP32 inside the kernel. Each scale touches a 128×128 weight tile; BF16 rounding of the scale itself is ~3e-3 relative per scale, which then multiplies every dequant value in the block.
3. **No two-level accumulation** — FP8 mantissa is only 3 bits (E4M3). When summing K/128 = 16 partial sums with very different magnitudes per block, cancellation amplifies the BF16-rounding noise. DeepSeek showed this gives ~14 bits of effective precision on Hopper FP8 tensor cores; we are doing the equivalent on Ampere/Blackwell-compatible m16n8k16 by collapsing to BF16 before the dot, then accumulating in FP32 — but the *input* to the FP32 accum is already lossy.

---

## 2. vLLM Triton fused_moe (`vllm/model_executor/layers/fused_moe/fused_moe.py`)

The default Qwen3-MoE-FP8 path. Confirmed by reading the kernel:

```python
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
...
for k in range(0, K, BLOCK_SIZE_K):
    a = tl.load(a_ptr)          # bf16 activations
    b = tl.load(b_ptr)           # fp8 weights, no dequant yet
    accumulator = tl.dot(a, b, acc=accumulator)   # bf16 × fp8 → f32 ACCUM-IN-LOOP
# After full K loop:
if use_fp8_w8a8:
    accumulator = accumulator * a_scale * b_scale   # scale applied ONCE at the end
```

**Key differences from Atlas**:

- **`tl.dot` does FP8×BF16 (or FP8×FP8) → FP32 directly inside the tensor core.** Triton lowers to `mma.sync ... f32.e4m3.bf16.f32` on Hopper / `wmma.mma.f32.e4m3.f32` on Ada. The weight is **never rounded to BF16** — it stays FP8 going into the MMA, and the FP32 accumulator absorbs the dequant after the dot.
- **Block scale is applied once at the end**, not per-K-step. For block-128 quantization, the loop integrates `Σ a · b_fp8` over K=128 into one FP32 partial sum, then multiplies by `a_scale[…] * b_scale[n_block, k_block]`. The partial sum is full FP32 precision the whole way.
- **Per-token activation scale** (`a_scale`) is computed dynamically on the activation chunk. This is the "G(128)" mode in `moe_kernel_features.md`.

vLLM also has a CUTLASS path (PR #13972) that does the same thing but with TMA + warp specialization on SM90. Same numerical semantics.

---

## 3. SGLang + DeepGEMM (`sgl-kernel/csrc/moe/`)

SGLang uses **DeepGEMM** for FP8 grouped GEMM (DeepSeek's library, also adopted by vLLM with `VLLM_USE_DEEP_GEMM=1`).

DeepGEMM does **two-level accumulation** to work around a known Hopper FP8 tensor-core flaw:

> "FP8 Tensor Cores used a fixed-point accumulation strategy that effectively used only 14 bits of precision (not 32). To address this, DeepGEMM employs CUDA-core two-level accumulation: 4 consecutive WGMMA ops accumulate in lower precision in tensor-core registers, then are added into a separate FP32 register-backed accumulator after each WGMMA group."

Other DeepGEMM properties:

- Activation scale: **1D blockwise** per-token, shape `(M, K/128)`, dequantized inside the K-loop.
- Weight scale: **2D blockwise** `(K/128, N/128)`, dequantized inside the K-loop.
- TMA loads scales coalesced; CUDA-core promotion is overlapped with the next WGMMA via warp specialization.
- Both inputs stay FP8 going into the MMA. Scales applied **per K-block** (every 128 elements), promoted to FP32 register.

**Known accuracy bug worth noting**: vLLM issue #37804 — DeepGEMM on Blackwell uses E8M0 scale format which loses ~0.4–0.5 bits/layer, compounding across 80 layers in Qwen3.5. Workaround: `VLLM_USE_DEEP_GEMM=0` (falls back to Triton). On Hopper / non-E8M0 path, DeepGEMM is at least as good as Triton fused_moe.

---

## 4. HF transformers reference (`finegrained_fp8.md`)

Ground truth path. Pseudocode:

```python
# Weights stored as fp8 in (128,128) blocks, scale in fp32
def dequant(w_fp8, scale_fp32):
    w_f32 = w_fp8.to(torch.float32)          # FP8 → FP32 (exact, no rounding)
    return w_f32 * scale_fp32[block_idx]     # scale in FP32

def forward(x_bf16, w_fp8, scale):
    w_bf16_or_f32 = dequant(w_fp8, scale)    # full FP32 internally
    return x_bf16.to(f32) @ w_f32 + ...      # FP32 matmul, BF16 cast only on output
```

This is what Atlas's `hf_forward_bf16_unquant.py` is doing for the cosine reference. The kernel above rounds to BF16 *after* the FP32 matmul (or never, depending on the activation upcast).

---

## 5. Marlin MoE (vLLM W8A16 path)

Marlin is the W8A16 path: weights FP8, activations BF16/FP16, no FP8 tensor cores used.

- Packs 4 FP8 bytes into int32, unpacks via SIMT bit-arithmetic + LUT on the fly inside the K-loop **just before** feeding into a BF16 tensor-core MMA.
- Accumulator is FP32.
- **Critical**: Marlin's unpack uses FP16-via-bit-cast or BF16-via-LUT — and that round-to-BF16 step is the same loss Atlas has. But Marlin pays for it with W4 (4-bit) workloads where bandwidth dominates and the rounding noise is dwarfed by quant noise. For W8 FP8 with 128-block scales, Marlin and Atlas are numerically equivalent.

**Takeaway**: Marlin is NOT a precision improvement over Atlas's current kernel. It's a layout/throughput improvement on Ampere/Ada. Reference: vLLM PR #5975.

---

## 6. The exact bug pattern

Atlas does this:

```
fp8_byte → LUT[byte] (f32) → × scale_bf16 (cast to f32) → ROUND to bf16 → smem_B → bf16×bf16 → f32 accum
```

vLLM Triton / DeepGEMM does this:

```
fp8_byte stays as fp8 → fp8 × bf16 → f32 accum → (after K loop) × a_scale × b_scale (f32)
```

Or equivalently, the "dequant inside loop, but keep in f32" variant:

```
fp8_byte → LUT (f32) → × scale (f32) → KEEP IN F32 → f32 × bf16 (via tcgen05.mma or fma) → f32 accum
```

The difference between Atlas and reference is exactly **one BF16 rounding per weight element**, which at K=2048 produces a random walk of ~√K ≈ 45 ULPs in the accumulator, which is exactly the ~1e-2 cosine drop per high-magnitude layer.

---

## 7. Recommendation for Atlas

**Highest-leverage, lowest-risk fix: keep B in FP32 in smem (or use FP32-friendly MMA path).**

GB10 (sm_121) does not have FP8 tensor cores accessible from PTX (confirmed in Atlas memory `project_fp4_mma_gb10.md`). So the FP8-direct-into-MMA path that vLLM uses on H100 is not available. But we can still recover the BF16 rounding loss:

### Option A (recommended, minimal change): FP32 smem for B, then FMA in registers

In `moe_fp8_grouped_gemm.cu`, replace:
```cpp
__shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];
// ...
smem_B[k][n] = __float2bfloat16(dequant_val);   // ← lossy
```
with FP32 smem:
```cpp
__shared__ float smem_B_f32[K_STEP][N_TILE + PAD];
// ...
smem_B_f32[k][n] = dequant_val;                  // ← exact
```

Then do the MMA as **BF16 × BF16 → FP32 with B rounded only at the MMA boundary** (`__float2bfloat16` per fragment register, not per smem write). This shifts the rounding from "every K element across the whole tile" to "once per MMA fragment load" — same number of roundings but they happen *after* the per-block scale is fully applied in FP32, so the error is no longer correlated with the scale magnitude.

Cost: 2× smem on B (16 KB → 32 KB per CTA), still well under GB10's 100 KB/SM limit.

### Option B (better precision, larger change): two-level accumulation with scale-in-loop

Mimic DeepGEMM. Keep the FP32 accumulator, but apply the block scale **inside the K loop, once per 128-element block**, in FP32:

```cpp
float acc_block[8][4] = {0};        // partial sum for current K-block
float acc_total[8][4] = {0};        // FP32 promotion accumulator

for (k_base = 0; k_base < K; k_base += FP8_BLOCK) {  // step by 128
    // run 8 K_STEP=16 inner iterations using BF16 MMA into acc_block
    for (inner = 0; inner < FP8_BLOCK/K_STEP; inner++) {
        // existing K_STEP=16 dequant + MMA, but DO NOT apply scale
        // (load LUT[byte] only, no × scale)
    }
    // promote: multiply by FP32 scale, add to total
    float s = __bfloat162float(S_exp[n_block * k_blocks + k_block]);
    for (i,j) acc_total[i][j] += acc_block[i][j] * s;
    for (i,j) acc_block[i][j] = 0;
}
```

This recovers two things:
- Scale is applied in FP32 to a FP32 partial sum (no BF16 round between dequant and scale).
- FP32 promotion accumulator absorbs ULP cancellation across blocks.

Cost: a few extra FP32 FMAs per K-block (negligible), and we save the `× scale` from the per-element dequant inner loop (small win).

### Option C (full DeepGEMM port): use `mma.sync ... f32.e4m3.bf16.f32` if sm_121 supports it

Worth checking — PTX 8.3+ on Blackwell SM100 has `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.bf16.f32` (FP8 × BF16 → FP32 in one instruction). If GB10 (sm_121, derivative of SM120) supports this, we get true Triton-style "FP8 stays FP8 into the tensor core." Atlas's prior FP4 MMA investigation said FP4 wasn't available; FP8 mma support on sm_121 needs a separate ptxas check (`nvcc --ptxas-options=-v` + try the asm).

---

## 8. Recommendation summary

| Option | Effort | Precision gain | Throughput cost | When to pick |
|--------|--------|----------------|------------------|---------------|
| **A: FP32 smem for B** | ~30 LoC | Recovers most of 0.92→0.99 | 2× B smem, neutral on math | **Start here.** Minimal blast radius, easy to validate with the existing cosine harness. |
| **B: Two-level accum with scale-in-loop** | ~80 LoC | Recovers nearly all, matches DeepGEMM semantics | ~2% extra FMAs | If A doesn't close the gap below cos 0.99. |
| **C: Native FP8 MMA** | 1–2 weeks (needs PTX validation + new kernel) | Full vLLM parity | Likely faster (skip dequant) | Long-term, after A/B is shipped and validated. |

**Concrete first step**: implement Option A as a `_v3` variant gated behind `ATLAS_FP8_MOE_FP32_SMEM=1`, run the existing `bench/fp8_dgx2_drift/cosine_run.py` harness against L20 ssm.moe_out at 64 / 128 / 256 / 1024 tokens, expect cos to climb from 0.92 to >0.99. Then promote as default once decode/prefill throughput is verified non-regressed on bench/moe_decode_vs_prefill.

**Do not** chase Marlin or W4A16 paths — they don't help this specific drift (the rounding loss is in the dequant step, not the storage format).

---

## Sources

- vLLM fused_moe kernel: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
- vLLM CUTLASS grouped GEMM MoE PR #13972: https://github.com/vllm-project/vllm/pull/13972
- vLLM Marlin FP8 PR #5975: https://github.com/vllm-project/vllm/pull/5975
- vLLM MoE kernel features design doc: https://docs.vllm.ai/en/latest/design/moe_kernel_features/
- vLLM DeepGEMM E8M0 accuracy bug on Qwen3.5: https://github.com/vllm-project/vllm/issues/37804
- DeepGEMM repo: https://github.com/deepseek-ai/DeepGEMM
- Colfax: DeepSeek FP8 14-bit accumulator analysis: https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/
- Kingsley Kim DeepGEMM tear-down: https://kingsleykim.dev/blog/deepgemm/
- HF fine-grained FP8 docs: https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/finegrained_fp8.md
- Qwen3-Next FP8 model card: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
