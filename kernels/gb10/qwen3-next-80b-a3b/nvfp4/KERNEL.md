# GB10 / Qwen3-Next-80B-A3B / NVFP4 — Kernel Context

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

## Model: Qwen3-Next-80B-A3B-Instruct

- **Parameters**: 80B total, ~3B active per token (MoE)
- **Architecture**: Hybrid — 12 attention layers + 36 SSM (Mamba) layers
- **Attention**: GQA 16:2 (16 query heads, 2 KV heads), head_dim=256
- **SSM layers**: Gated Delta Rule with causal Conv1d
- **MoE**: 512 experts, top-10 routing per token
- **Hidden dim**: 2048
- **Q/Gate interleaving**: HF q_proj output is per-head interleaved `[Q_h0, G_h0, Q_h1, G_h1, ...]`
  Must deinterleave before RoPE/norm (see `ssm_preprocess.cu:deinterleave_qg`)

## Quantization: NVFP4

- **Format**: E2M1 (4-bit) weights with FP8 block scales
- **E2M1 dequant**: Shared memory LUT of 16 possible values. Load LUT once per block, index with 4-bit nibble. This single optimization was +71% throughput.
- **GEMV pattern**: W4A16 — FP4 weights dequantized to BF16 on-the-fly, multiplied against BF16 activations
- **M=1 specialization**: Decode is always M=1 (single token). GEMV kernels must be optimized for this case, not general GEMM.

## Key Optimization Patterns

### GEMV (Memory-bound at M=1)
- Single token decode reads ~1,517 MB of weights per forward pass
- At 273 GB/s peak: 5.6ms theoretical minimum per token
- Actual: ~8.5ms (GEMV) + ~3.6ms (non-GEMV) = ~12.2ms → 82 tok/s
- Bandwidth utilization: 45-68% per GEMV kernel

### MoE Dispatch
- 512 experts, top-10 selected per token
- 3D grid approach: `(expert_idx, row_chunk, col_chunk)`
- Shared expert runs in parallel with routed experts
- Expert GEMV kernels fuse gate_up + SiLU + down projection

### SSM (Mamba) Layers
- 36 layers, each with causal Conv1d (d_conv=4) + gated delta rule
- Conv1d decode: single-step update from sliding window state
- FLA (Fused Linear Attention): recurrent mode for decode, chunk mode for prefill

### Attention Layers
- 12 layers with paged KV cache (block_size=16)
- FP8 KV cache with explicit cache_stride parameter
- Decode attention: single-query against paged KV blocks
- Prefill: tiled attention with causal masking

## Weight Layout Per Token

| Component | Weight Read | Count |
|-----------|-----------|-------|
| Attention (Q/K/V/O proj) | ~67 MB | ×12 layers |
| SSM (in/out proj, conv, gate) | ~20 MB | ×36 layers |
| MoE (10 experts × gate_up + down) | ~234 MB | ×48 layers |
| Shared expert | ~12 MB | ×48 layers |
| Embeddings + final norm | ~16 MB | ×1 |
| **Total** | **~1,517 MB** | |
