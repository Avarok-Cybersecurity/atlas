// SPDX-License-Identifier: MIT
//
// Vendored from turboderp's ExLlamaV3 (https://github.com/turboderp-org/exllamav3)
// Copyright (c) 2025 turboderp — MIT license.
// Snapshot original: .research/exllamav3_ref/exl3_gemm_kernel.cuh.
// Adaptations (bodies verbatim otherwise):
//   * the two `__global__` template kernels became `inline __device__ void
//     exl3_gemm_kernel_body / exl3_mgemm_kernel_body`: Atlas needs plain
//     extern "C" __global__ entry points selectable by name from the PTX
//     module, and a __global__ cannot call another __global__, so the
//     __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16) moves to
//     the wrappers in exl3_matmul.cu (cg::this_grid().sync() is legal in
//     __device__ functions; validity still requires cooperative launch)
//   * include paths made local; cooperative_groups included here (upstream
//     got it from the host .cu)
//   * the dead commented-out post-pass output rotation block was dropped
//   * upstream's commented-out `// if (suh)` guard is preserved as-is:
//     suh / A_had / svh are all unconditionally dereferenced and therefore
//     MANDATORY (the host must never pass null)
//
// The module-scope __device__ globals v_indices/v_weights/bszm_sync are the
// mgemm index-compaction scratch: ONE in-flight filtered mgemm per PTX module
// — the host must serialize filtered mgemm launches on a device.

#pragma once

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "exl3_kernel_map.cuh"
#include "exl3_compat.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner.cuh"
#include "exl3_devctx.cuh"

template<EXL3_GEMM_T_ARGS>
inline __device__
void exl3_gemm_kernel_body(EXL3_GEMM_ARGS)
{
    auto grid = cg::this_grid();

    // if (suh)
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner<true, false>
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                0.088388347648f  // 1/sqrt(128)
            );

        grid.sync();
        A = A_had;
    }

    int size_m_ = size_m;
    const half* A_ = A;
    void* C_ = C;

    while (size_m_ > 0)
    {
        exl3_gemm_kernel_inner
        <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, true>
        (A_, B, C_, MIN(size_m_, 16), size_k, size_n, locks, svh);

        A_ += 16 * size_k;
        if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * size_n);
        else                  C_ = (void*) (((half*) C_) + 16 * size_n);
        size_m_ -= 16;

        if (size_m_ > 0 || svh)
            grid.sync();
    }
}

#define MAX_INDICES 128

__device__ int64_t v_indices[128];
__device__ half v_weights[128];
__device__ int bszm_sync;

template<EXL3_GEMM_T_ARGS>
inline __device__
void exl3_mgemm_kernel_body(EXL3_MGEMM_ARGS)
{
    int bszm = MAX(bszm_in, bszm_out);
    auto grid = cg::this_grid();

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 890)
        int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;
    #endif

    // Pack indices within min_index <= idx < max_index

    if (min_index >= 0)
    {
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        {
            if (num_tokens > 1)
            {
                // Position-preserving mask: the grouped reduction below sums each token's
                // fixed run of (bszm / num_tokens) slots, and with bszm_in > 1 slot j also
                // addresses input row j, so out-of-range picks are marked inactive in place
                // (skipped by the compute stages and the reduction) instead of compacted away
                for (int i = 0; i < bszm; ++i)
                {
                    int idx = B_indices[i];
                    bool keep = idx >= min_index && idx < max_index;
                    v_indices[i] = keep ? idx - min_index : -1;
                    if (B_weights) v_weights[i] = keep ? B_weights[i] : __float2half(0.0f);
                }
                bszm_sync = bszm;
            }
            else
            {
                int j = 0;
                for (int i = 0; i < bszm; ++i)
                {
                    int idx = B_indices[i];
                    if (idx >= min_index && idx < max_index)
                    {
                        v_indices[j] = idx - min_index;
                        if (B_weights) v_weights[j] = B_weights[i];
                        j++;
                    }
                }
                bszm_sync = j;
                for (; j < bszm; ++j)
                {
                    v_indices[j] = -1;
                }
            }
        }
        __threadfence();
        grid.sync();
        B_indices = v_indices;
        if (B_weights) B_weights = v_weights;
        bszm = bszm_sync;
    }

    for (int i = 0; i < bszm; i += gridDim.z)
    {
        int j = i + blockIdx.z;
        int mat_index = -1;
        const uint16_t* B = nullptr;
        if (j >= bszm) j = -1;
        else
        {
            mat_index = B_indices ? (int) B_indices[j] : j;
            if (mat_index >= 0)
            {
                B = B_list[mat_index];
            }
        }

        // Had and input scales

        if (B)
        {
            int total_warps = size_m * size_k / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

            const half* suh = suh_list[mat_index];
            const half* A_ = bszm_in == 1 ? A : A + j * size_m * size_k;
            half* A_had_ = A_had + j * size_m * size_k;

            for(; this_warp < total_warps; this_warp += warps_grid)
                had_hf_r_128_inner<true, false>
                (
                    A_ + this_warp * 128,
                    A_had_ + this_warp * 128,
                    suh + (this_warp * 128) % size_k,
                    0.088388347648f  // 1/sqrt(128)
                );
        }

        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 890)
            group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);
        #else
            grid.sync();
        #endif

        // Matmul. Per-matrix output width/pointer when the caller supplies the lists
        // (size_n then only sizes the per-z-slice lock ranges and must be the max width);
        // resolved once per matrix, outside all inner loops

        int n_j = (size_n_list && mat_index >= 0) ? size_n_list[mat_index] : size_n;
        int size_m_ = size_m;
        half* A_ = A_had + j * size_m * size_k;
        void* C_;
        if (C_list && mat_index >= 0) C_ = C_list[mat_index];
        else if constexpr (c_fp32) C_ = (void*) (((float*) C) + j * size_m * size_n);
        else                       C_ = (void*) (((half*) C) + j * size_m * size_n);
        void* C_base = C_;

        while (size_m_ > 0)
        {
            if (B)
            {
                int lock_offs = blockIdx.z * size_n / 128;

                exl3_gemm_kernel_inner
                <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, false>
                (A_, B, C_, MIN(size_m_, 16), size_k, n_j, locks + lock_offs, nullptr);
            }

            A_ += 16 * size_k;
            if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * n_j);
            else                  C_ = (void*) (((half*) C_) + 16 * n_j);
            size_m_ -= 16;

            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 890)
                group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);
            #else
                grid.sync();
            #endif
        }

        // Had and output scales

        if (B)
        {
            int total_warps = size_m * n_j / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

            const half* svh = svh_list[mat_index];
            float scale = 0.088388347648f;  // 1/sqrt(128)
            if (B_weights) scale *= __half2float(B_weights[j]);

            C_ = C_base;

            for(; this_warp < total_warps; this_warp += warps_grid)
            {
                if constexpr (c_fp32)
                    had_ff_r_128_inner<false, true>
                    (
                        ((const float*) C_) + this_warp * 128,
                        ((float*) C_) + this_warp * 128,
                        svh + (this_warp * 128) % n_j,
                        scale
                    );
                else
                    had_hf_r_128_inner<false, true>
                    (
                        ((const half*) C_) + this_warp * 128,
                        ((half*) C_) + this_warp * 128,
                        svh + (this_warp * 128) % n_j,
                        scale
                    );
            }
        }
    }

    if (B_weights)
        grid.sync();

    // Final reduction: each of the num_tokens groups of (bszm / num_tokens) contiguous slots is
    // summed into its own output row (row t for group t), instead of always collapsing into row
    // 0. num_tokens == 1 (the legacy single-token case) reduces to exactly the original
    // single-row behavior. Groups MUST be processed in increasing t order per column: row t is
    // only ever read by group floor(t / stride), which is <= t, so it has already been fully
    // read (and, if that group's index equals t, is only then correctly overwritten) by the time
    // group t's own write happens.
    if (B_weights && blockIdx.z == 0)
    {
        int total_warps = size_m * size_n / 32;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        int this_lane = threadIdx.x % 32;
        int stride = bszm / num_tokens;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            for (int t = 0; t < num_tokens; ++t)
            {
                int col = this_warp * 32 + this_lane;
                if constexpr (c_fp32)
                {
                    float* C___ = ((float*) C) + t * stride * size_m * size_n + col;
                    float sum = 0.0f;
                    for (int j = 0; j < stride; ++j)
                    {
                        // Inactive slots (masked by range filtering, or -1 selections) were
                        // never written by the compute stages: their scratch is stale
                        if (!B_indices || B_indices[t * stride + j] >= 0)
                            sum += *C___;
                        C___ += size_m * size_n;
                    }
                    ((float*) C)[t * size_m * size_n + col] = sum;
                }
                else
                {
                    half* C___ = ((half*) C) + t * stride * size_m * size_n + col;
                    half sum = {};
                    for (int j = 0; j < stride; ++j)
                    {
                        if (!B_indices || B_indices[t * stride + j] >= 0)
                            sum = __hadd(sum, *C___);
                        C___ += size_m * size_n;
                    }
                    ((half*) C)[t * size_m * size_n + col] = sum;
                }
            }
        }
    }
}
