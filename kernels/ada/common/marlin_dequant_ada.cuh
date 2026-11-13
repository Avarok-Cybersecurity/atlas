// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0
// Portions copied from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08),
// csrc/quantization/marlin/dequant.h:1-216 and 473-474, Copyright contributors to the vLLM
// project, Apache-2.0.
// Copyright (C) Marlin.2024 Elias Frantar, Apache-2.0, adapted from
// https://github.com/IST-DASLab/marlin.
// Changes (by Atlas contributors), listed in VLLM-MARLIN-ATTRIBUTION.md:
//   * pruned to the 4-bit integer weight types Atlas instantiates (kU4B8 = GPTQ, kU4 = AWQ
//     with zero points): the int8 (kU8B128, kU8), FP8 (kFE4M3fn), FP4 (kFE2M1f)
//     specializations and the FP8 / E8M0 scale specializations of upstream lines 218-518 are
//     not copied. The primary templates stay declared, so a new weight type is a new
//     specialization here plus a new instantiation in marlin_ada.cu;
//   * specializations are keyed by the scalar `k*Id` constants;
//   * the two FP16 `ADD` constants above INT_MAX are written as explicit casts of unsigned
//     literals (same bits), so the strict kernel build raises no sign-change diagnostic;
//   * includes point at the Atlas file names; the include guard is `#pragma once`.
//
// NUMERICS NOTE. Upstream's explanation of the bit tricks is kept below verbatim. The
// `lop3` and the `__hsub2` / `__hfma2` steps are explicit instructions, so the Ada tree's
// `--fmad=false` (kernels/ada/common/KERNEL.toml) does not change them: an FP16 or BF16
// dequantized weight here has the same bits as in vLLM's build.

/*
Fast Dequantization (Converting INT4/INT8/FP4/FP8 to FP16/BF16)

The process of fast dequantization can be summarized as a combination
of bitwise operations and floating-point computations:

weight =>(bit_op / bitwise operations)=>
f16_value =>(flop / floating-point computation)=>
dequantized_weight

Since the dequantized weights typically require subtracting the zero point and
applying a scale factor, the floating-point computation step can be fused with
the zero-point subtraction and scaling operations.

The following are the parts that need to be modified for the fused operation
of zero-point subtraction and scaling.

## INT4 => FP16/BF16 or INT8 => FP16

The floating-point computation is `__hsub2`

If has zero points:

    flop(bit_op(weight)) - flop(bit_op(zp))
  = sub(bit_op(weight), bias) - sub(bit_op(zp), bias)
  = bit_op(weight) - bit_op(zp)

so we don't need additional modification.

If has float zero points:

    flop(bit_op(weight)) - fzp
  = sub(bit_op(weight), bias) - fzp
  = bit_op(weight) - (fzp + bias)

where the `fzp + bias` can be computed at weight loading. But this
may have accuracy issue, so we should not use this in most cases.

If has not zero points:

    scale(flop(bit_op(weight)))
  = scale(sub(bit_op(weight), bias))
  = scale(bit_op(weight)) - scale(bias)
  = fma(bit_op(weight), scale_factor, scale(bias))

where the `scale(bias)` can be cached. But this may have accuracy issue,
so we should not use this in most cases.


## INT8 => BF16

INT8 => BF16 is a special case, it use byte_perm instead of flop.
We cannot fused byte_perm with scaling.


## FP4/FP8 => FP16/BF16

    scale(flop(bit_op(weight)))
  = scale(mul(bit_op(weight), multiplier))
  = mul(bit_op(weight), scale_factor * multiplier)

where `scale_factor * multiplier` can be computed at weight loading.

*/
#pragma once

#include "marlin_dtypes_ada.cuh"

namespace MARLIN_NAMESPACE_NAME {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// Constructs destination register by taking bytes from 2 sources (based on
// mask)
template <int start_byte, int mask>
__device__ inline uint32_t prmt(uint32_t a) {
  uint32_t res;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n"
               : "=r"(res)
               : "r"(a), "n"(start_byte), "n"(mask));
  return res;
}

template <typename scalar_t2, vllm::ScalarTypeId w_type_id,
          bool skip_flop = false>
__device__ inline void dequant(int q, scalar_t2* frag_b);

//
// Efficiently dequantize 4bit values packed in an int32 value into a full
// B-fragment of 4 fp16 values. We mostly follow the strategy in the link below,
// with some small changes:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L215-L287
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L327-L385
//
template <>
__device__ inline void dequant<half2, vllm::kU4B8Id, true>(int q,
                                                           half2* frag_b) {
  const int MASK = 0x000f000f;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  frag_b[0] = *reinterpret_cast<half2*>(&lo);
  frag_b[1] = *reinterpret_cast<half2*>(&hi);
}

template <>
__device__ inline void dequant<half2, vllm::kU4B8Id, false>(int q,
                                                            half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // clang-format on
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  // Atlas: spelled as an explicit cast of the unsigned literal (same bits); a plain
  // `const int ADD = 0xd480d480;` is EDG diagnostic 68 (change of sign), which the
  // Atlas kernel build's `--Werror all-warnings` turns into an error.
  const int ADD = static_cast<int>(0xd480d480u);
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
}

template <>
__device__ inline void dequant<half2, vllm::kU4Id, true>(int q,
                                                         half2* frag_b) {
  dequant<half2, vllm::kU4B8Id, true>(q, frag_b);
}

template <>
__device__ inline void dequant<half2, vllm::kU4Id, false>(int q,
                                                          half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // clang-format on
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64006400;
  const int MUL = 0x2c002c00;
  const int ADD = static_cast<int>(0xd400d400u);  // Atlas: as above, same bits
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU4B8Id, true>(
    int q, nv_bfloat162* frag_b) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  // Guarantee that the `(a & b) | c` operations are LOP3s.
  // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  // clang-format on

  frag_b[0] = *reinterpret_cast<nv_bfloat162*>(&lo);
  frag_b[1] = *reinterpret_cast<nv_bfloat162*>(&hi);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU4B8Id, false>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, vllm::kU4B8Id, true>(q, frag_b);

  static constexpr uint32_t SUB = 0x43084308;

  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const nv_bfloat162*>(&SUB));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const nv_bfloat162*>(&SUB));
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU4Id, true>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, vllm::kU4B8Id, true>(q, frag_b);
}

template <>
__device__ inline void dequant<nv_bfloat162, vllm::kU4Id, false>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, vllm::kU4Id, true>(q, frag_b);

  static constexpr uint32_t SUB = 0x43004300;

  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const nv_bfloat162*>(&SUB));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const nv_bfloat162*>(&SUB));
}

// FP8 / E8M0 scale decode. Declared (the kernel template names it in a branch that only
// FP8-scale weight types compile), never specialized here: Atlas instantiates no type with
// 8-bit scales.
template <typename scalar_t2, vllm::ScalarTypeId s_type_id>
__device__ inline void dequant_fp8_scales(int q, scalar_t2* frag_b);

#endif

}  // namespace MARLIN_NAMESPACE_NAME
