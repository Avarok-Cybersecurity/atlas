# NVFP4 Coherence: Selective FP8 Precision via `--nvfp4-increase-coherence`

## Problem
NVFP4 (4-bit E2M1) quantization degrades attention Q/K/V/O precision, causing hallucinations
and poor instruction following. SSM layers are less sensitive (recurrent state acts as low-pass filter).

## Solution
CLI flag `--nvfp4-increase-coherence` that dequants attention Q/K/V/O weights to FP8 at load time,
serving them through W8A16 GEMV/GEMM kernels. SSM + MoE stay NVFP4.

**Memory cost: +277 MB** (0.2% of 120 GB — trivial). MoE at FP8 would cost +14.6 GB (too much).

## Architecture
- **Attention layers** (10 on 35B): NVFP4 kept for MTP batch2/3 paths, FP8 copy for decode+prefill
- **MoE layers**: stay NVFP4 (memory prohibitive for 256 experts)
- **SSM layers**: stay NVFP4 (sufficient precision)

## CLI
```bash
spark serve model --nvfp4-increase-coherence   # opt-in, default off
```
When enabled, logs: "Mixed precision: attention Q/K/V/O at FP8 (+277 MB)"

## Files to Modify
1. `cli.rs` — new `--nvfp4-increase-coherence` bool flag
2. `weight_loader.rs` — after NVFP4 load for FullAttention, dequant to FP8 via nvfp4→BF16→FP8
3. `weight_map.rs` — new `nvfp4_to_fp8weight()`: NVFP4→BF16 (CPU) → upload → FP8 blockscaled (GPU)
4. `quantize_bf16_to_fp8_blockscaled.cu` — new kernel: per-128x128 block max + E4M3 quantize
5. `ops.rs` — kernel launch wrapper

## Load-time Pipeline
```
NVFP4 (disk) → dequant_nvfp4_to_bf16 (CPU) → upload BF16 (GPU) → quantize_bf16_to_fp8_blockscaled (GPU) → Fp8Weight
```
One-time cost: ~0.5s for 273M attention params. Negligible vs 20s total load.

## Runtime Dispatch (already exists)
- Decode: `q_fp8w.is_some()` → W8A16 GEMV (FP8), else NVFP4 GEMV
- Prefill: `q_fp8w.is_some() && w8a16_gemm_k` → W8A16 GEMM, else NVFP4
- MTP batch2/3: always NVFP4 (kept alongside FP8)

## Risk
- Low: +277 MB memory, opt-in flag, existing dispatch handles FP8 automatically
- Medium: NVFP4→BF16→FP8 double-quantization — quality ceiling is BF16 dequant of NVFP4, strictly better than raw NVFP4 but worse than native FP8 checkpoint
- Mitigation: quality A/B test before shipping as default

## Future Phases
- Phase 2: Shared expert at FP8 (+57 MB, always-active expert)
- Phase 3: Routed experts at FP8 (EP=2 only, +7.3 GB per GPU)
