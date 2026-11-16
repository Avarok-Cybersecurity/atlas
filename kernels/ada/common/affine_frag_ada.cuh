// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0
// Portions copied and adapted from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/kernels/cuda/common/qmm_frag.cuh:1-174 and zig/kernels/cuda/common/cp_async.cuh:1-67.
// Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written
// before TensorFold 0.6.0 is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: namespaces qmm_frag and tf_common merged into avarok_affine; only the primitives the
// Ada affine lane GEMV uses are kept (LaneTile, smem, cp16, cp4, commit, wait, ldmatrix4, mma, mma0, pairm,
// swz); the sm_90 `sub.rn.bf16x2` branch of pair/pairm is dropped, so every build takes the sm_80-class
// `fma.rn.bf16x2` form (bit-identical by the upstream argument below); the PDL grid_wait/grid_launch
// helpers (sm_90) and the fp_pairs.cuh include (e2m1 pairs, unused here) are dropped.

// Shared device primitives of the affine 4-bit lane GEMV (affine4_lane_ada.cu): cp.async, ldmatrix x4,
// the BF16 m16n8k16 MMA, the int4 code-pair decode and the shared-memory swizzle.
//
// INCLUDE-ONLY. No entry points; the lane GEMV and the pack/quantize module include it.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace avarok_affine {

// One CTA's tile of the lane GEMV: BM rows by BN columns, WM x WN warps, a pipeline of STAGES groups.
// Each stage holds one GS-input group: the BM input rows (bf16), the BN columns' packed codes, the
// BN scales and biases (bf16) and the BM rows' group sums (fp32).
template <int GS, int BM, int BN, int WM, int WN, int STAGES>
struct LaneTile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;               // m16 tiles a warp
    static constexpr int NT = BN / WN / 8;                // n8 tiles a warp
    static constexpr int ROW = GS * 2;                    // bytes of one input row a group
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;                    // stage bytes: inputs,
    static constexpr int W = BN * GS / 2;                 // weights,
    static constexpr int S = BN * 2;                      // scales, biases (bf16),
    static constexpr int XS = BM * 4;                     // and input sums (fp32)
    static constexpr int STAGE = X + W + 2 * S + XS;
    static constexpr int PARTIALS = MT * NT * 4 * THREADS * 4;
    static constexpr int SMEM = STAGES * STAGE > PARTIALS ? STAGES * STAGE : PARTIALS;
};

__device__ __forceinline__ uint32_t smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem(dst)), "l"(src));
}

__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem(dst)), "l"(src));
}

// Commit the copies issued so far as one group.
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }

// Wait until at most N committed groups are still in flight.
template <int N>
__device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem(p)));
}

// d += a * b (bf16 operands, fp32 accumulators).
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// d = a * b + 0.
__device__ __forceinline__ void mma0(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    const float z = 0.0f;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
}

// Two 4-bit codes of `w` (the nibbles at bit s and at bit s + 16) as two EXACT bf16 integers 0..15.
// OR-ing 0x4300 makes the bf16 value 128 + q; one fma.rn.bf16x2 computes t * 1 - 128, which is exact for
// every t = 128 + q, so the result has the same bits as upstream's sm_90 `sub.rn.bf16x2` form.
// `mask` is 0x000F000F, passed in a register so the compiler keeps one copy.
__device__ __forceinline__ uint32_t pairm(uint32_t w, int s, uint32_t mask) {
    const uint32_t t = ((w >> s) & mask) | 0x43004300u;
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x3F803F80u), "r"(0xC300C300u));
    return r;
}

// The 16-byte chunk a row's chunk c lands on in shared memory (XOR swizzle over CHUNKS chunks).
template <int CHUNKS>
__device__ __forceinline__ int swz(int r, int c) { return c ^ (r % CHUNKS); }

}  // namespace avarok_affine
