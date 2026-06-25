# Strix Halo (gfx1151) W4A8 integer-DP4A decode path

Native-HIP ROCm port of Atlas's NVFP4 decode GEMV, accelerated with the
integer-DP4A (`v_dot4` / `__builtin_amdgcn_sudot4`) W4A8 technique grabbed and
adapted from `charlie12345/rocmfp4-llama`. This is the `cuda → rocm →
rocm-nvfp4` line: Atlas's own CUDA NVFP4 kernels, hipified to native ROCm, then
extended with the rocmfp4 int8 codebook/v_perm technique so we maintain our own
port rather than depending on the llama.cpp fork.

## What this adds

Additive, strix-hip-only, **OFF by default** (`ATLAS_W4A16_DP4A=1` to enable;
the float E2M1-LUT path is untouched and remains the default on every target,
and the NVIDIA/gb10 path is bit-identical). On gfx1151 builds the kernels are
present; on any other target the handles miss and callers keep the float path.

Kernels (`kernels/strix-hip/common/w4a16_gemv_dp4a.cu`):
- `quantize_act_int8_g16` — symmetric block-q8_1 activation quant (d = amax/127,
  per-16-group), **hoisted** to run once per distinct activation.
- `w4a16_gemv_dp4a` — W4A8 GEMV from a pre-quantized int8 activation. Weights are
  EXACT in int8 (NVFP4 codebook `{0,.5,1,1.5,2,3,4,6}` × 2 → integer grid, ×0.5
  folded into the scale); the only new error vs W4A16 is the int8 act quant.
  Uses the branchless 2× `__builtin_amdgcn_perm` codebook expansion grabbed from
  rocmfp4-llama (`rocmfp4_hip_codebook.cuh`), re-derived for Atlas's
  consecutive-pair nibble layout and proven byte-exact on gfx1151.
- `silu_mul_quant_int8_g16` — **new**: fuses `silu(gate)*up` (bit-identical math
  to the float `w4a16_gemv_silu_input`) + the int8 group-quant, so the down-proj
  activation is hoisted out of the GEMV too.

Wiring (`crates/spark-model/src/layers/dense_ffn.rs`, decode `forward`): the
flag-gated path quantizes the post-norm input **once** (shared by gate+up),
runs three `w4a16_gemv_dp4a` GEMVs, and fuses `silu(gate)*up`+quant for the
down-proj input. Per-call quant is break-even; the hoist is what makes it a win.
Dispatch helpers live in `crates/spark-model/src/layers/ops/dp4a.rs`.

## Measured on gfx1151 (Radeon 8060S, 60 GB GTT)

**Per-GEMV microtest** (`w4a16_gemv_dp4a_microtest`, N=2048 K=4096): cosine
**0.999991** vs the float oracle; GEMV-only **20.9 µs vs 23.4 µs (1.12×)**.

**End-to-end A/B**, identical binary / model (`unsloth/Qwen3.6-27B-NVFP4`, 29 GB
mixed-precision VL, dense FFN h=5120 inter=17408 × 64 layers) / env / prompts —
only `ATLAS_W4A16_DP4A` changes:

| Path | 200-tok decode | Output |
|------|----------------|--------|
| `ATLAS_W4A16_DP4A=0` (float fused) | **9.83 tok/s** | coherent |
| `ATLAS_W4A16_DP4A=1` (int8-DP4A)   | **12.35 tok/s** | coherent, byte-identical |

**+25.6 % decode**, coherence-neutral (greedy hashmap output byte-identical;
Paris / first-8-primes / 7! all correct on both).

## vs llama.cpp ROCmFP4 (same box, BFCL-v4 single-turn, 1004 samples)

| Engine | Model | BPW / size | BFCL-v4 ST | Decode |
|--------|-------|-----------|-----------|--------|
| **Atlas NVFP4** (this port) | Qwen3.6-27B-NVFP4 | 29 GB (VL, 4-bit text) | **88.82** | 12.35 tok/s (DP4A) |
| llama.cpp ROCmFP4 | Qwen3.6-27B-ROCmFP4-MIX | 16.4 GB | 86.65 | — |
| llama.cpp ROCmFP4 | STRIX_LEAN | 14.6 GB | 86.06 | 12.5 tok/s |
| llama.cpp ROCmFP4 | COHERENT | 19.8 GB | — | 9.2 tok/s |

Atlas wins **coherence** (+2.17 BFCL over the accuracy-comparable MIX variant)
and, with DP4A, matches/beats llama.cpp ROCmFP4 on **decode** despite serving a
2× larger (29 GB VL) checkpoint on the bandwidth-bound LPDDR5X part. A
size-matched (~14 GB text-only) Atlas NVFP4 checkpoint is the remaining lever to
make the speed win unambiguous.

## Reproduce

```bash
# build (box /workspace/atlas, native-HIP): ~/build-hip-27b.sh  + SKIP_ATLAS_BUILD=1
# A/B smoke:        bash /workspace/atlas/dp4a_smoke.sh {0,1}
# ST accuracy gate: bash /workspace/atlas/dp4a_st_run.sh {0,1}
```
