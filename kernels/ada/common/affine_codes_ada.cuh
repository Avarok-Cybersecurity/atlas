// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0
// Portions copied and adapted from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/kernels/cuda/affine_codes.cuh:1-45 (code_at, byte_pair, code_pair, pair_at), and the device
// primitives this lane's GEMV uses from zig/kernels/cuda/common/qmm_frag.cuh:14-174 (ldmatrix4, mma,
// mma0, swz) and zig/kernels/cuda/common/cp_async.cuh:13-46 (smem, cp16, cp4, commit, wait).
// Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written
// before TensorFold 0.6.0 is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: one Atlas-owned namespace `avarok_affine_n` (so this header never collides with
// the 4-bit lane's `affine_frag_ada.cuh` in one translation unit); only the primitives the narrow and
// 8-bit tiles use are kept; the PDL `grid_wait`/`grid_launch` pair is dropped (sm_90 only, a no-op on
// sm_89); comments rewritten.
//
// MLX affine codes (2, 3, 5, 6 or 8 bits a weight) read for the bf16 tensor-core MMA of
// `affine_n_ada.cu`. Every code is a small unsigned integer, so it is EXACT as a bf16 operand: the
// GEMV multiplies inputs by raw codes and applies the group's scale and bias in fp32 afterwards
// (value = scale * code + bias, so x . w = scale * (x . codes) + bias * sum(x)).
//
// Include-only header of lane L17c (port sprint `port/consumer-opt`); `affine_n_ada.cu` is its one
// includer. Built with Ada's `--fmad=false` (kernels/ada/common/KERNEL.toml): every multiply-add
// below is an explicit instruction or an exact integer operation, so the flag changes nothing here.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace avarok_affine_n {

// ── Code extraction ──────────────────────────────────────────────────────

// Code i of an MLX row packed `bits` wide, low code first, in little-endian words (a code may cross
// a word boundary).
__device__ __forceinline__ uint32_t code_at(const uint32_t* __restrict__ row, long long i, int bits) {
    const long long bit = i * bits;
    const int sh = static_cast<int>(bit & 31);
    const long long wd = bit >> 5;
    uint32_t v = row[wd] >> sh;
    if (sh + bits > 32) v |= row[wd + 1] << (32 - sh);
    return v & ((1u << bits) - 1u);
}

// Bytes at bit s and s + 8 of w as a bf16 pair (the first in the low half): exact, as every byte is
// a bf16 integer (2^23 + b - 2^23 in fp32, then a lossless bf16 conversion).
__device__ __forceinline__ uint32_t byte_pair(uint32_t w, int s) {
    const float a = __uint_as_float(0x4B000000u | ((w >> s) & 0xFFu)) - 8388608.0f;
    const float b = __uint_as_float(0x4B000000u | ((w >> (s + 8)) & 0xFFu)) - 8388608.0f;
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Codes c0 (low half) and c1 (high half) of up to 6 bits as a bf16 pair: bf16 0x4300 | c is 128 + c
// exactly, and subtracting 128 is exact.
__device__ __forceinline__ uint32_t code_pair(uint32_t r) {
    r |= 0x43004300u;
    const __nv_bfloat162 h =
        __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&r), __floats2bfloat162_rn(128.0f, 128.0f));
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Pair i of a lane's BITS words: the pairs' first codes stream through the words' low halves, the
// second codes through the high halves, code i at bits [i * BITS, i * BITS + BITS) of each stream.
template <int BITS>
__device__ __forceinline__ uint32_t pair_at(const uint32_t (&w)[BITS], int i) {
    constexpr uint32_t M = (1u << BITS) - 1u;
    const int o = i * BITS, h = o >> 4, sh = o & 15;
    if (sh + BITS <= 16) return code_pair((w[h] >> sh) & (M | M << 16));
    const uint32_t lo = __byte_perm(w[h], w[h + 1], 0x5410), hi = __byte_perm(w[h], w[h + 1], 0x7632);
    return code_pair(((lo >> sh) & M) | ((hi >> sh) & M) << 16);
}

// ── Device primitives (cp.async, ldmatrix, bf16 mma.sync, swizzle) ───────

__device__ __forceinline__ uint32_t smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem(dst)), "l"(src));
}

__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem(dst)), "l"(src));
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem(p)));
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma0(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    const float z = 0.0f;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
}

// XOR swizzle of a row's 16-byte chunk, so ldmatrix's eight row reads hit eight distinct banks.
template <int CHUNKS>
__device__ __forceinline__ int swz(int r, int c) {
    return c ^ (r % CHUNKS);
}

}  // namespace avarok_affine_n
