# Atlas Progress Report — February 2026

## Executive Summary

Atlas is a custom SM121 (GB10 Blackwell) CUDA attention backend for vLLM, providing hand-tuned paged decode and Flash Attention v2 prefill kernels for the Qwen3-Next-80B hybrid model. Current sustained throughput is **40.5 tok/s** with FP8 KV cache, up from ~32 tok/s at initial deploy. The v22 FlashInfer baseline is **42 tok/s** (no MTP) — a gap of only **~4%**.

| Milestone | Throughput | Delta | Date |
|-----------|-----------|-------|------|
| Initial Atlas deploy | ~32 tok/s | — | Feb 24 |
| CUDA graph FULL mode | ~36.4 tok/s | +14% | Feb 25 |
| FP8 KV cache fix | 36.5 tok/s | +0.3% | Feb 25 |
| **Attention-only patching (P0)** | **40.5 tok/s** | **+11%** | **Feb 26** |
| v22 FlashInfer target | 42 tok/s | ~4% gap | — |
| v22 Marlin+MTP ceiling | 59.9 tok/s | +48% gap | — |

---

## P0 Results: Selective Patching

**The single biggest optimization: stop patching ops that torch.compile can fuse.**

Previously, Atlas patched 7 vLLM operations with individual CUDA kernels. Each patch inserted a kernel launch boundary, preventing torch.compile's inductor backend from fusing operations (e.g., `residual_add + rms_norm`, `rms_norm + rope`, `silu * gate + down_proj`).

By reducing to **attention-only patching** (1 patch instead of 7), torch.compile recovers full fusion for RMSNorm, SiLU, RoPE, RMSNormGated, SSM, and MoE. Result: **+11% throughput** (36.5 → 40.5 tok/s).

| Config | Patches | Throughput | torch.compile |
|--------|---------|-----------|---------------|
| All-patched | 7 (attn, norm, silu, rope, ssm, moe) | 36.5 tok/s | Fusion broken |
| **Attention-only** | **1 (attn)** | **40.5 tok/s** | **Full fusion** |
| v22 FlashInfer | 0 (native) | 42 tok/s | Full fusion |

---

## Current Architecture

### What Atlas Handles

| Operation | Atlas Kernel | Notes |
|-----------|-------------|-------|
| **Attention (decode)** | `paged_decode_attn` / `paged_decode_attn_splitk` | BF16 + FP8 variants, split-K for low batch |
| **Attention (prefill)** | `inferspark_prefill_v47` | Flash Attention v2, BR=64, tiled online softmax |
| **KV cache write** | `reshape_and_cache_flash` / `_fp8` | Zero-copy non-contiguous QKV, FP8 quantize |

### What Uses vLLM Defaults (torch.compile fusible)

| Operation | vLLM Kernel | Notes |
|-----------|------------|-------|
| **RMSNorm** | `torch.ops._C.rms_norm` | Fused with residual add by inductor |
| **SiLU*Mul** | `torch.ops._C.silu_and_mul` | Fused with surrounding ops |
| **RoPE** | `torch.ops._C.rotary_embedding` | Pre-computed cos/sin cache |
| **RMSNormGated** | `layernorm_guard.rmsnorm_fn` | Fused norm+gate in one kernel |
| **Conv1d** | Triton | Complex varlen interface |
| **GDN** | Triton/FLA | Recurrent + chunked |
| **MoE GEMM** | Marlin W4A16 | Same as v22 baseline |
| **Linear projections** | PyTorch/torch.compile | QKV, output, gate/up/down |

---

## Remaining Gap: 40.5 → 42 tok/s (~4%)

The remaining gap is entirely in **attention kernel quality**:

### FlashInfer Attention vs Atlas Attention

FlashInfer's paged decode attention is extensively optimized:
- **Software pipelining**: Overlaps global memory loads with computation via `cp.async`
- **Warp specialization**: Producer/consumer warps for better SM utilization
- **Page table indirection**: Optimized with async copy
- **Split-K heuristics**: Tuned per-architecture

Atlas decode attention uses:
- 8 warps splitting KV sequence (basic parallelism)
- BC=4 batched loading within blocks
- Online softmax with tree-based inter-warp reduction
- No software pipelining or warp specialization

### Optimization Roadmap (40.5 → 42+ tok/s)

| Priority | Optimization | Expected Gain | Effort |
|----------|-------------|---------------|--------|
| **P1** | Decode attention: `cp.async` memory pipelining | +1-2% | Medium |
| **P2** | Decode attention: warp specialization | +1-2% | High |
| **P3** | Prefill batching (single kernel launch for multi-request prefill) | +0-1% | Low |

### Beyond 42 tok/s (requires MTP)

To reach 59.9 tok/s, MTP speculative decoding is required. This is orthogonal to kernel optimization — it requires vLLM's built-in MTP support and is compatible with Atlas attention.

---

## Architecture Reference

### Model: Qwen3-Next-80B-A3B-Instruct (Hybrid)

- 12 Attention layers (standard transformer)
- 36 GDN layers (Gated DeltaNet / Mamba SSM)
- MoE: 128 experts, top-8 routing, ~3B active params/token
- Quantization: NVFP4 (4-bit E2M1 weights, FP8 block scales)

### Per-Decode-Token Kernel Breakdown

| Component | Count | Kernel | Framework |
|-----------|-------|--------|-----------|
| QKV projection | 12 attn + 36 SSM | Linear (GEMM) | torch.compile |
| RMSNorm | ~96 | `rms_norm` | vLLM (torch.compile fused) |
| RoPE | 12 | rotary_embedding | vLLM (torch.compile fused) |
| **Attention decode** | 12 | `paged_decode_attn` | **Atlas CUDA** |
| **KV cache write** | 12 | `reshape_and_cache_flash` | **Atlas CUDA** |
| SiLU*Mul | ~48 | `silu_and_mul` | vLLM (torch.compile fused) |
| RMSNormGated | 36 | `rmsnorm_fn` | vLLM (torch.compile fused) |
| Conv1d update | 36 | `causal_conv1d_update` | vLLM Triton |
| GDN decode | 36 | `fused_recurrent_gdr` | vLLM Triton/FLA |
| MoE GEMM | 48 | Marlin W4A16 | Marlin |
| Output proj | 48 | Linear (GEMM) | torch.compile |

MoE GEMM dominates compute time (~60-70%). Attention + KV cache is ~10-15%. Norms + activations are ~5-10%.

---

## Key Files

| File | Purpose |
|------|---------|
| `atlas/backend.py` | vLLM V1 attention backend (zero-copy, FP8, CUDA graphs) |
| `atlas/ops.py` | Python wrappers (stride + FP8 + cache_stride) |
| `atlas/patch_vllm_atlas.py` | Master patcher — attention-only (runs at Docker build time) |
| `atlas/patch_ops.py` | Patch functions (attention backend registration) |
| `crates/atlas-py/src/attention.rs` | Rust kernel dispatch (BF16/FP8 routing) |
| `cuda_kernels/reshape_and_cache.cu` | KV cache write (BF16 + FP8) |
| `cuda_kernels/paged_decode_attn.cu` | Paged decode attention (BF16) |
| `cuda_kernels/paged_decode_attn_fp8.cu` | Paged decode attention (FP8 + cache_stride) |
| `cuda_kernels/inferspark_prefill_v47.cu` | Flash Attention v2 prefill |
| `Dockerfile` | Docker image build (extends v22 base) |
| `start-atlas.sh` | Launch script (Marlin env vars) |

---

**Last Updated**: 2026-02-26
**Current Throughput**: 40.5 tok/s (attention-only patching, FP8 KV cache, CUDA graph FULL mode)
**Target**: 42+ tok/s (match v22 FlashInfer baseline without MTP)
**Primary Bottleneck**: Atlas decode attention lacks software pipelining and warp specialization
