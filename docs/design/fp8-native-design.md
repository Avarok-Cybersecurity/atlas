# Native FP8 Serving Design (Workstream 4)

Date: 2026-03-16
Updated: 2026-03-18
Status: **IMPLEMENTED** — 35B single-GPU + 122B EP=2 both working with MTP

## What Was Implemented (2026-03-18)

### 4A: Native FP8 Checkpoint Loading ✅
- FP32→BF16 SSM param auto-conversion (A_log, norm.weight)
- Unindexed shard support (model.safetensors-NNNNN-of-NNNNN format)
- Pre-flight OOM estimate with header-only scan (no mmap on GB10)
- OOM watchdog async task (2s poll, exit before system freeze)
- QuantFormat enum (Nvfp4, Fp8) replaces native_fp8 bool

### 4B: FP8 GEMM Dispatch ✅
- w8a16_gemv: FP8 E4M3 LUT decode GEMV (batch1)
- w8a16_gemm: FP8 block-scaled prefill GEMM (non-transposed)
- w8a16_gemm_t: Transposed FP8 prefill GEMM (coalesced reads)
- FP8 MoE fused kernels: batch1/2/3 for MTP verify
- QuantWeight enum + quant_gemv/quant_gemm dispatch wrappers
- FP8 dispatch in decode_multi_seq for MTP verify attention

### 4C: FP8 KV Cache ✅
- FP8 KV cache with online calibration (--fp8-kv-calibration-tokens)
- BF16 high-precision KV on attention layers (--kv-high-precision-layers max)

## Performance Results

| Config | tok/s | TTFT (warm) | Coherence |
|--------|-------|-------------|-----------|
| 35B FP8 + MTP | 95 | 100ms | 10/10 |
| 35B NVFP4 + MTP | 53.6 | 100ms | 10/10 |
| 122B FP8 EP=2 + MTP | 34 | 282ms | 10/10 |
| 122B NVFP4 EP=2 + MTP | 50 | 200ms | 10/10 |

## Remaining Work

- **FP8 MoE grouped GEMM**: Needed for ISL>4k prefill (forward_prefill falls back to per-token GEMV)
- **CUDA graph investigation**: Graphs work for decode but may have issues with specific kernel patterns
- **QuantWeight full integration**: Replace all if/else chains with quant_gemv dispatch (Phase 2 partial)
- **122B FP8 single-spark**: Doesn't fit (119 GB weights > 121.7 GB GPU)

## Key Commits
- `4d8d310` FP32→BF16 SSM param conversion
- `8709ecf` Zero-copy FP8 attention
- `bde51ce` FP8-only MoE dispatch
- `5f4effe` FP8 MoE batch2/3 fused kernels
- `fdc46b9` FP8 dispatch in decode_multi_seq
- `2621977` Re-enable CUDA graphs for FP8
- `d915da6` w8a16_gemm_t transposed kernel
- `f1aca43` QuantWeight enum (Phase 1)
- `2fee170` QuantFormat enum (Phase 4)
