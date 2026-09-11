// SPDX-License-Identifier: AGPL-3.0-only
//
// Hopper twin of `per_token_group_quant_fp8` — the per-token / per-128-K-group
// FP8 E4M3 activation quantizer that feeds every W8A8 projection (#928, #927).
//
// WHY A TWIN. The shared kernel (`kernels/gb10/common/per_token_group_quant_fp8.cu`,
// left untouched — gb10, b200, strix and strix-hip all compile it) spends ONE
// 128-thread CTA on ONE 128-element group, i.e. one bf16 element per thread.
// Per CTA that is a single 256-byte load, and 256 bytes is all the memory-level
// parallelism an SM can hold per resident CTA. nsys on 1xH100 80GB HBM3
// (Qwen/Qwen3.8-27B-FP8 @ `3c0379030`, round 13 cell T1N, grids `(4576,136)`,
// `(4576,40)`, `(4576,48)` = M x K/128) prices the result at 633 / 641 / 636 GB/s
// for K = 5120 / 17408 / 6144 — 18.9-19.1% of this part's 3350 GB/s — and
// 47 659.5 us, 10.36% of a 4593-token prefill. `rms_norm_residual` in the SAME
// trace reaches 2 644 GB/s (78.9%) and 3 298 GB/s (98.4% at M=1168) on the same
// compulsory-traffic model, so the headroom is the kernel's, not the machine's.
//
// WHAT CHANGES. Sixteen threads cover a group, eight groups share a 128-thread
// CTA, and each thread moves one `uint4` (8 bf16). The CTA's load is 2 KB
// instead of 256 B, which is the whole fix. Two further consequences fall out:
// the eight values stay in registers, so A is read ONCE (the shared kernel reads
// it twice — once for the amax, once to quantize), and the group max reduces
// through a 16-lane `__shfl_xor_sync` butterfly, so there is no shared memory
// and no `__syncthreads` in the kernel at all.
//
// WHAT DOES NOT CHANGE — this is the gate. The FP8 bytes and the FP32 scales are
// BIT-IDENTICAL to the shared kernel's, at every (M, K) the engine launches.
// Same `amax / 448.0f`, same `1e-12f` floor, same per-element `div.rn.f32` by
// that scale (NOT a reciprocal multiply), same saturating clamp, same
// `__nv_cvt_float_to_fp8(..., __NV_SATFINITE, __NV_E4M3)`. The reduction TREE
// differs and may: `fmaxf` is exact, associative and commutative over the reals,
// and both kernels seed the reduction with `0.0f`, which is what makes the NaN
// case agree too — PTX `max.f32` returns the non-NaN operand, so a NaN input is
// dropped by both regardless of where it sits in the tree. See
// `crates/spark-model/examples/native_fp8_act_quant_hopper_microtest.rs`, which
// asserts byte equality against the shared kernel on device and carries a
// KNOWN_BAD control that must fail.
//
// Grid: (M, ceil(K/128 / 8), 1)  Block: (128, 1, 1)
// The CTA's group span is derived from `gridDim.y` rather than hard-coded, so
// the launcher (`ops::per_token_group_quant_fp8`) may pick any Y extent and the
// kernel still covers `K/128` groups exactly once. M on grid X (limit 2^31-1)
// supports MoE `total_expanded` > 65535, as in the shared kernel.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define FP8_GROUP_K 128
#define FP8_E4M3_MAX 448.0f

// 128 elements / 8 per thread = 16 threads per group; 128 threads / 16 = 8
// groups per CTA. Mirrored by `ops::FP8_QUANT_HOPPER_GROUPS_PER_CTA`.
#define HQ_LANES_PER_GROUP 16
#define HQ_GROUPS_PER_CTA 8
#define HQ_ELEMS_PER_THREAD 8

// One `uint4` (8 bf16) if the address allows it, else eight scalar loads.
//
// The guard is a CTA-uniform branch on a base pointer, not a per-lane test: the
// arena buffers this kernel reads are 256-byte-aligned allocations and `K` is a
// multiple of 128, so the fast path is the one that runs. The scalar arm exists
// so a future caller handing in an odd sub-slice gets the same NUMBERS rather
// than a misaligned-access fault — the two arms load the same values in the same
// order and are bit-identical by construction.
__device__ __forceinline__ void hq_load8(const __nv_bfloat16* __restrict__ p, float* v) {
    if ((reinterpret_cast<uintptr_t>(p) & 15u) == 0) {
        const uint4 w = *reinterpret_cast<const uint4*>(p);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&w);
        #pragma unroll
        for (int i = 0; i < HQ_ELEMS_PER_THREAD; i++) v[i] = __bfloat162float(b[i]);
    } else {
        #pragma unroll
        for (int i = 0; i < HQ_ELEMS_PER_THREAD; i++) v[i] = __bfloat162float(p[i]);
    }
}

// The store side of the same bargain: one 8-byte `uint2` when aligned.
__device__ __forceinline__ void hq_store8(unsigned char* __restrict__ p, const unsigned char* b) {
    if ((reinterpret_cast<uintptr_t>(p) & 7u) == 0) {
        uint2 w;
        w.x = (unsigned int)b[0] | ((unsigned int)b[1] << 8) | ((unsigned int)b[2] << 16)
              | ((unsigned int)b[3] << 24);
        w.y = (unsigned int)b[4] | ((unsigned int)b[5] << 8) | ((unsigned int)b[6] << 16)
              | ((unsigned int)b[7] << 24);
        *reinterpret_cast<uint2*>(p) = w;
    } else {
        #pragma unroll
        for (int i = 0; i < HQ_ELEMS_PER_THREAD; i++) p[i] = b[i];
    }
}

extern "C" __global__ void per_token_group_quant_fp8_hopper(
    const __nv_bfloat16* __restrict__ A,   // [M, K] BF16 activations
    unsigned char* __restrict__ A_fp8,     // [M, K] FP8 E4M3
    float* __restrict__ a_scale,           // [M, K/128] FP32 scale, row-major
    unsigned int M,
    unsigned int K
) {
    const unsigned int m = blockIdx.x;
    if (m >= M) return;

    // `K / 128`, exactly the count the shared kernel's grid.y carries. A K that
    // is not a multiple of 128 drops the same partial tail there as here.
    const unsigned int L = K / FP8_GROUP_K;
    if (L == 0) return;

    const unsigned int gpc = (L + gridDim.y - 1u) / gridDim.y;
    const unsigned int g0 = blockIdx.y * gpc;
    if (g0 >= L) return;
    const unsigned int g_end = (g0 + gpc < L) ? (g0 + gpc) : L;

    const unsigned int tid = threadIdx.x;
    const unsigned int sub = tid / HQ_LANES_PER_GROUP;   // which of the 8 groups
    const unsigned int lane = tid % HQ_LANES_PER_GROUP;  // which 8-element chunk

    const size_t row = (size_t)m * (size_t)K;
    // Uniform across the CTA, so every lane runs the same trip count and the
    // butterfly below is never entered by a divergent subset of the warp.
    const unsigned int iters = (gpc + HQ_GROUPS_PER_CTA - 1u) / HQ_GROUPS_PER_CTA;

    for (unsigned int it = 0; it < iters; ++it) {
        const unsigned int g = g0 + it * HQ_GROUPS_PER_CTA + sub;
        const bool live = (g < g_end);
        const size_t base =
            row + (size_t)g * FP8_GROUP_K + (size_t)lane * HQ_ELEMS_PER_THREAD;

        // 1. Load 8 elements per thread and take their abs-max.
        //
        // Seeded with 0.0f, exactly as the shared kernel seeds its cross-warp
        // combine. Every input is an absolute value, so the seed is a no-op on
        // the real domain; on NaN it is what makes the two trees agree.
        float v[HQ_ELEMS_PER_THREAD];
        float amax = 0.0f;
        if (live) {
            hq_load8(A + base, v);
            #pragma unroll
            for (int i = 0; i < HQ_ELEMS_PER_THREAD; i++) amax = fmaxf(amax, fabsf(v[i]));
        }

        // 2. Reduce the max across the group's 16 lanes. XOR offsets 1, 2, 4, 8
        //    never cross a 16-lane boundary, so two groups share a warp without
        //    mixing and every lane ends holding its own group's amax.
        #pragma unroll
        for (int off = HQ_LANES_PER_GROUP / 2; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        }

        // 3. The scale, byte-for-byte the shared kernel's.
        float scale = amax / FP8_E4M3_MAX;
        if (scale < 1e-12f) scale = 1e-12f;
        if (live && lane == 0) a_scale[(size_t)m * (size_t)L + (size_t)g] = scale;
        if (!live) continue;

        // 4. Quantize from registers — A is never re-read.
        unsigned char out[HQ_ELEMS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < HQ_ELEMS_PER_THREAD; i++) {
            float q = v[i] / scale;
            q = fmaxf(fminf(q, FP8_E4M3_MAX), -FP8_E4M3_MAX);
            out[i] = (unsigned char)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
        }
        hq_store8(A_fp8 + base, out);
    }
}
