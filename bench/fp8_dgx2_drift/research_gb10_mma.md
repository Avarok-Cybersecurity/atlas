# GB10 / SM121 MMA Capability Survey for FP8 MoE Precision

**Target**: NVIDIA GB10 (Grace Blackwell Superchip), arch `sm_121a` (consumer/workstation
Blackwell, Compute Capability 12.1).
**Toolchain**: CUDA 13.0.88 (`ptxas` build cuda_13.0.r13.0/compiler.36424714_0).
**Context**: MoE prefill currently runs at cosine ≈ 0.92 vs HF BF16 reference because the
Atlas grouped GEMM dequants FP8 → BF16 in shared memory and then issues a BF16 MMA
(`m16n8k16.f32.bf16.bf16.f32`). The dequant truncates 8-bit mantissa-shifted scale
products to BF16 (7-bit mantissa) → ~3e-3 per-element error that accumulates over K=1408.

## TL;DR

**Native FP8 MMA is available on SM121.** A prior memory note claimed FP8/FP4 MMA was a
silicon limitation on GB10 — that note is **wrong for FP8**. It is correct only for FP4
and FP6 datapaths and for block-scaled variants (see "What is still blocked" below).

The Atlas MoE FP8 grouped GEMM should be rewritten to:

1.  Quantize BF16 activations to FP8 E4M3 once per tile (in registers, on the way to
    SMEM), using the same per-block scaling factors as the weight side.
2.  Skip the SMEM dequant pass entirely.
3.  Issue `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, accumulate in FP32,
    apply the combined `scale_A * scale_B` block-scale on the FP32 accumulator before
    the next tile (i.e. classic SM89 staged-accumulator FP8 GEMM).

Expected effect on cosine: dequant-to-BF16 truncation error is gone; remaining error is
the unavoidable FP8 quantization of A (which is also what HF / PyTorch FP8 reference
does on Hopper / SM89). Cosine should rise from ~0.92 to ~0.985+, matching what TRT-LLM
and vLLM see when they use the FP8 MMA directly on Hopper. Plus 2× math throughput
(`QMMA.16832` vs `HMMA.16816`).

---

## 1. PTX MMA variants empirically confirmed on `sm_121a`

Every variant below was compiled to a cubin with
`nvcc -arch=sm_121a -cubin <file>.cu` and SASS inspected with `cuobjdump --dump-sass`.
A green check = ptxas + SASS emission both succeed; a red cross = `ptxas` rejection.

| PTX MMA                                                                   | Status      | SASS emitted                              | Notes                                          |
| ------------------------------------------------------------------------- | ----------- | ----------------------------------------- | ---------------------------------------------- |
| `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`                     | OK          | `QMMA.16832.F32.E4M3.E4M3`                | **The one to use** for FP8 weights+acts        |
| `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32`                     | OK          | `QMMA.16832.F32.E5M2.E5M2`                | Larger range, less precision than E4M3         |
| `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32`                     | OK          | `QMMA.16832.F32.E4M3.E5M2`                | Mixed FP8: useful if act range > weight range  |
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`                     | OK          | `HMMA.16816.F32`                          | Currently used by every Atlas kernel           |
| `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`                       | OK          | `HMMA.16816.F32`                          | Same datapath as BF16 on Blackwell             |
| `mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32`                      | OK          | `HMMA.1684.F32.TF32`                      | TF32, low arithmetic intensity                 |
| `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`                      | OK          | `HMMA.1688.F32.TF32`                      | TF32 k=8                                       |
| `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`        | OK          | `QMMA.16832.F32.E4M3.E4M3` (same SASS!)   | Blackwell `kind::` form aliases SM89 FP8 MMA   |
| `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e3m2.f32` (FP6)  | REJECTED    | —                                         | "Unexpected instruction types" — no FP6 datapath |
| `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32` (FP4)  | REJECTED    | —                                         | "Unexpected instruction types" — no FP4 datapath |
| `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32...ue8m0` | REJECTED | —                                         | No block-scaled MMA hardware on consumer Blackwell |
| `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64...`   | REJECTED    | —                                         | NVFP4 block-scale MMA unavailable              |

This corrects the prior memory note `project_fp4_mma_gb10.md`. The note's findings about
FP4 and block-scale forms are still correct; the claim that no FP8 datapath exists is
wrong. The clue is in the SASS: `QMMA` is the dedicated Blackwell FP8 tensor-core opcode
(separate from `HMMA` which covers BF16/FP16/TF32). Plain `m16n8k32.f32.e4m3.e4m3.f32`
PTX (the SM89 form) is accepted by ptxas on `sm_121a` and lowers to `QMMA.16832`.

Probe files lived in `/tmp/mma_probe/` for the duration of the survey.

## 2. What CUTLASS / CuTe expose

- `cutlass/include/cutlass/arch/mma_sm89.h:143` declares the SM89 FP8 MMA template
  emitting exactly `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`. However it is
  gated by `#if defined(CUTLASS_ARCH_MMA_SM89_ENABLED)` which itself requires
  `__CUDA_ARCH__ == 890` (exact match, not `>= 890`). So **the CUTLASS SM89 path will
  NOT compile under `-arch=sm_121a`** — you cannot just reuse CUTLASS's SM89
  collective. Either patch the guard to `__CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1210`
  or emit the inline PTX directly (Atlas convention).
- Upstream CUTLASS does ship a real SM120 FP8 path in
  `cute/arch/mma_sm120.hpp` (e.g. line 685 in the FlashInfer-vendored copy):
  `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`. This is
  functionally identical to the SM89 form — both lower to `QMMA.16832.F32.E4M3.E4M3` —
  and is wrapped in `SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, float>`. FlashInfer,
  vLLM Marlin, and DeepGEMM all already use this on consumer Blackwell. Atlas can mirror
  the PTX string verbatim without taking the CuTe template dependency.

## 3. What is currently used in Atlas

```
$ grep -rn "mma.sync" /workspace/atlas-mtp/kernels/gb10/*.cu*
```

Every kernel — `moe_fp8_grouped_gemm.cu`, `moe_w4a16_grouped_gemm.cu` (all model
variants), `w4a16_gemm*.cu`, `w8a16_gemm*.cu`, `dense_gemm_tc.cu`,
`prefill_paged_compute.cuh` — issues `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
**Zero kernels use `QMMA` / FP8 MMA today.** Every quantized weight path (FP8, NVFP4,
W4A16) follows the same template: dequant to BF16 in SMEM, then BF16 MMA.

For the MoE FP8 case (`moe_fp8_grouped_gemm.cu`):
- A is BF16 activations.
- B is FP8 E4M3 weights with per-128×128 BF16 block scales.
- Hot loop: byte-LUT to FP32, multiply by block scale, cast to BF16, store to SMEM_B.
- Then `mma.sync.m16n8k16.bf16.bf16.f32` on SMEM_A × SMEM_B.

The BF16 cast at line ~232 is where the per-tile precision is bottlenecked. The FP32
product `E4M3_LUT * scale` has up to 23 bits of mantissa; rounding it to 7 mantissa
bits drops on average ~3e-3 relative error per element. Summed over K=1408 with the
typical activation magnitudes this matches the observed 0.92 cosine drift almost
exactly.

## 4. Recommendation for the MoE FP8 prefill kernel

Migrate `moe_fp8_grouped_gemm` to a true SM89-style FP8 staged-accumulator GEMM:

1.  **A quantization (BF16 → FP8 E4M3)**: pick `scale_A` = max-abs per 128×128 tile of A,
    quantize to E4M3 in registers as A enters SMEM. Cost: one fast E4M3 conversion
    (`__nv_cvt_bfloat16raw_to_fp8` or `cvt.rn.satfinite.e4m3x2.bf16x2`, both available
    on SM121).
2.  **B stays packed**: load FP8 E4M3 bytes from global into SMEM verbatim. Drop the
    dequant LUT. No FP32 / BF16 store to SMEM_B.
3.  **MMA**:
    ```ptx
    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
        {acc0, acc1, acc2, acc3},
        {a0, a1, a2, a3},        // 4× u32 packed E4M3 (16 elems)
        {b0, b1},                // 2× u32 packed E4M3 (8 elems)
        {acc0, acc1, acc2, acc3};
    ```
    Two `QMMA.16832` issues replace four `HMMA.16816` issues for the same K=64 of work
    → 2× peak tensor-core throughput.
4.  **Block-scale fold-in**: after each K=128 chunk (one full block-scale group),
    multiply the FP32 accumulator by `scale_A[m_block] * scale_B[n_block, k_block]`.
    This is CUTLASS's SM89 `is_sm89_staged_policy_v` algorithm (see
    `mma_sm89.h:67-87`). The scale apply is in FP32, so no precision is lost there.
5.  **Epilogue**: cast FP32 accumulator → BF16 once on store. This is the only
    BF16-rounding step in the whole pipeline (vs. one rounding per K=128 chunk today).

Expected: cosine 0.92 → ~0.985+, prefill TTFT improves because each MMA does 2× the K
of the current BF16 MMA and SMEM bandwidth pressure on the B side drops 2× (FP8 vs
BF16). No new memory cost. No CUTLASS dependency — pure inline-PTX, matching Atlas
convention.

## 5. What is still blocked on GB10

Reaffirming `project_fp4_mma_gb10.md`:

- **No FP4 MMA datapath**: `e2m1` operand types under `kind::f8f6f4` are rejected.
- **No FP6 MMA datapath**: `e3m2` / `e2m3` rejected.
- **No block-scaled MMA**: `kind::mxf8f6f4.block_scale`, `kind::mxf4nvf4.block_scale`
  rejected — scales must be applied manually on the FP32 accumulator (which is what the
  SM89 staged policy does anyway).
- **No TMA-fed MMA on consumer Blackwell** for these data types: still ldmatrix +
  cp.async or normal global loads.

## 6. References

- NVIDIA PTX ISA 8.7, section 9.7.13 "Matrix multiply-accumulate instruction":
  https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-multiply-accumulate-instruction-mma
- NVIDIA PTX ISA 8.7, section 9.7.13.5 "MMA instructions with floating-point type" —
  lists the f32.e4m3.e4m3.f32 / f32.e4m3.e5m2.f32 / f32.e5m2.e4m3.f32 / f32.e5m2.e5m2.f32
  matrix shapes (m16n8k32 is the only FP8 shape exposed, no k=16 or k=64 FP8).
- NVIDIA PTX ISA 8.7, section 9.7.13.5.4 "Matrix multiply-accumulate operation using
  `mma.sync.kind::f8f6f4` instructions" — Blackwell-flavoured form, same SASS.
- NVIDIA Blackwell whitepaper, 5th-gen tensor cores: native FP8 / INT8 / TF32; FP6 / FP4
  documented for GB200 / B100 / B200 (datacenter Blackwell), absent on GB10.
- CUTLASS `include/cutlass/arch/mma_sm89.h` lines 100-145: canonical SM89 FP8 MMA
  template with `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`. Note
  `__CUDA_ARCH__ == 890` exact-match guard at line 52.
- CUTLASS `include/cute/arch/mma_sm120.hpp` line 685: SM120 FP8 MMA via
  `kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`. Same emitted SASS as the SM89 form.
- vLLM PR #11258 (Marlin FP8 for Blackwell consumer cards) — production precedent for
  emitting this PTX on `sm_120a` / `sm_121a` via inline asm.

## 7. Empirical reproduction

```bash
mkdir -p /tmp/mma_probe && cd /tmp/mma_probe
cat > probe.cu <<'EOF'
#include <cstdint>
extern "C" __global__ void k(uint32_t* o, const uint32_t* a, const uint32_t* b) {
    float c[4] = {0,0,0,0};
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1])
    );
    o[0] = __float_as_uint(c[0]);
}
EOF
/usr/local/cuda-13.0/bin/nvcc -arch=sm_121a -cubin probe.cu -o probe.cubin
/usr/local/cuda-13.0/bin/cuobjdump --dump-sass probe.cubin | grep MMA
# Expected: QMMA.16832.F32.E4M3.E4M3 R8, R8, R6, RZ
```

---

**Date**: 2026-05-25
**Author**: Survey by Claude (Opus) for Atlas FP8 MoE precision investigation.
**Status**: Empirically validated on `sm_121a` with CUDA 13.0.88; corrects prior memory
note about FP8 availability.
