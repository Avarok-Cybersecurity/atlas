# GB10 / Qwen3.5-35B-A3B / NVFP4 — Kernel Context

> AI instruction context for optimizing kernels in this (H, M, Q) target.

## Hardware: GB10 (SM121)

- **Architecture**: SM121 (Blackwell, compute capability 12.1)
- **Memory**: 120 GB LPDDR5X @ 273 GB/s peak bandwidth
- **Practical bandwidth**: ~65-70% of peak (178-191 GB/s) due to memory controller overhead
- **No multi-CTA clusters**: ClusterShape forced to 1x1x1
- **No HBM**: LPDDR5X has higher latency, lower per-pin bandwidth than HBM3e
- **FP4 tensor cores**: SM120_16x8x64_TN_VS (native E2M1 support)
- **Missing PTX**: `cvt.rn.satfinite.e2m1x2.f32` not available on SM121

## Compilation

```bash
nvcc --ptx -arch=sm_121f -O3 --use_fast_math <file>.cu
```

## Model: Qwen3.5-35B-A3B-NVFP4

- **Parameters**: 35B total, ~3B active per token (MoE)
- **Architecture**: Hybrid — 10 full attention + 30 linear attention (Gated DeltaNet)
- **Layer pattern**: 3:1 (linear, linear, linear, full) × 10 = 40 layers
- **Full Attention**: GQA 16:2 (16 query heads, 2 KV heads), head_dim=256
- **Linear Attention**: 16 key heads (dim 128), 32 value heads (dim 128)
- **DeltaNet**: Gated delta rule with causal Conv1d (kernel=4)
- **MoE**: 256 experts, top-8 routing per token + 1 shared expert
- **Hidden dim**: 2048
- **Vocab**: 248,320 tokens
- **Vision**: Qwen2VL encoder (27-layer ViT, text-only for now)
- **MTP**: 1 layer with **full attention** (not SSM like 80B model)
- **Q/Gate interleaving**: Same as 80B — HF q_proj output is per-head interleaved
  `[Q_h0, G_h0, Q_h1, G_h1, ...]`. Deinterleave before RoPE/norm.

## Quantization: Mixed NVFP4 + BF16

**Critical**: Not all weights are NVFP4 quantized.

| Component | Format | Reason |
|-----------|--------|--------|
| Full attention (Q/K/V/O) | NVFP4 | Standard compressed-tensors |
| MoE experts (gate/up/down) | NVFP4 | Standard compressed-tensors |
| **Linear attention projections** | **BF16** | Quantizer skipped these |
| **LM head** | **BF16** | Too large to quantize |
| **Conv1d, norms, gates** | **BF16** | Small tensors |

**Optimization opportunity**: Self-quantize BF16 linear_attn weights to NVFP4 at load time.
This saves ~1,447 MB per decode token (37% reduction in bandwidth).

## Key Differences from Qwen3-Next-80B Target

1. **Separate linear_attn projections**: `in_proj_qkv` + `in_proj_z` + `in_proj_a` + `in_proj_b` (80B has fused `in_proj_qkvz` + `in_proj_ba`)
2. **Fewer experts**: 256 (80B: 512), top-8 (80B: top-10)
3. **Fewer layers**: 40 (80B: 48)
4. **Larger vocab**: 248K (80B: 152K) — bigger embedding/LM head
5. **MTP uses full attention** (80B uses SSM)
6. **Vision encoder** present (skip for text-only)

## Weight Layout Per Token

| Component | Weight Read | Notes |
|-----------|-----------|-------|
| Linear attn projections (BF16) | 2,023 MB | **52% — dominant** |
| LM head (BF16, 248K vocab) | 1,017 MB | 26% |
| MoE (8 experts × gate_up + down, NVFP4) | 679 MB | 18% |
| Full attn (Q/K/V/O, NVFP4) | 153 MB | 4% |
| Norms, conv, gates | ~0.3 MB | <1% |
| **Total** | **3,871 MB** | |

### Theoretical Performance

| Scenario | Reads/token | Time @ 273 GB/s | tok/s |
|----------|-------------|-----------------|-------|
| As-is (BF16 linear_attn) | 3,871 MB | 14.2ms (100%) | 70.5 |
| As-is (65% BW) | 3,871 MB | 21.8ms | 45.9 |
| Self-quantized (all NVFP4) | ~2,424 MB | 8.9ms (100%) | 112.6 |
| Self-quantized (65% BW) | ~2,424 MB | 13.7ms | 73.0 |

## MoE Dispatch (256 experts, top-8)

- Same 3D grid approach as 80B: `(expert_idx, row_chunk, col_chunk)`
- `MAX_EXPERTS=512` compile-time constant covers both 256 and 512
- Shared expert with sigmoid gate runs in parallel
- Expert GEMV fuses gate_up + SiLU + down projection

## DeltaNet Linear Attention

- 30 layers, same inner dims as 80B (16K heads, 32V heads, 128 dim)
- Conv1d: depthwise [8192, 1, 4], same as 80B
- Recurrent state: 32 heads × 128 × 128 = 2.10 MB per layer (FP32)
- Total state: 30 layers × 2.15 MB ≈ 64.4 MB
- State is 10.9x larger than 80B SSM state (5.9 MB) due to matrix vs vector

## Full Attention (10 layers)

- Same as 80B: GQA 16:2, head_dim=256, partial_rotary_factor=0.25
- Paged KV cache (block_size=16), FP8 with cache_stride
- Q/Gate interleaving requires deinterleave_qg kernel
