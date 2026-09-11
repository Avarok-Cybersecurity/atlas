// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 GEMV Fused, HOPPER-TUNED — dual projection + SiLU-input
// variants (FP8 E4M3), for M=1 decode (#928).
//
// The H100/H200 override of `kernels/gb10/common/w8a16_gemv_fused.cu`. Both
// entry points keep their names, their argument lists and their launch
// geometry, so `layers::dense_ffn` dispatches them unchanged and the gb10 file
// is untouched.
//
// `w8a16_gemv_dual` is the second-largest kernel of the C=1 decode step on an
// H100 (nsys, 1xH100, Qwen/Qwen3.8-27B-FP8, 2026-09-11 round 10: 64 launches,
// 5.79 ms of an 18.5 ms step = 31%, moving 178 MB of gate+up FP8 weights per
// token). It shares every instruction of its inner loop with `w8a16_gemv`, so
// it shares the same two problems and the same two fixes: the shared-memory
// E4M3 LUT gather that saturates the SM load/store unit, and the single
// outstanding weight load per warp. Both are diagnosed with their arithmetic
// in `w8a16_gemv_hopper.cuh`, which also states why the result is bit-identical
// to the gb10 kernels for every byte a block-scaled FP8 checkpoint can hold.
//
// w8a16_gemv_dual: blockIdx.z selects projection 0 (gate) vs 1 (up). Both
//   projections share the same BF16 input A[1, K].
//   Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
//
// w8a16_gemv_silu_input: reads gate_out + up_out BF16 vectors, computes
//   silu(gate)*up inline as the activation, then GEMV with FP8 down weights.
//   Eliminates the separate silu_mul kernel.
//   Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
//
// FP8-E4M3 weight format (per projection, unchanged from gb10):
//   B:           [N, K]          uint8 — one byte per weight (FP8 E4M3)
//   block_scale: [N/128, K/128]  FP32  — per-128x128-block scale (scale_inv
//                widened to FP32 at load; applied in full FP32 precision,
//                matching vLLM / DeepGEMM / HF block-FP8 numerics)

#include "w8a16_gemv_hopper.cuh"

/// Activation source for `w8a16_gemv_silu_input`: `silu(gate[k]) * up[k]`,
/// computed in the gb10 kernel's order and with its exact spelling —
/// `(g / (1 + __expf(-g))) * u` on the FP32 widenings of the BF16 inputs, the
/// approximate `__expf` intrinsic included.
struct HopperActSilu {
    const __nv_bfloat16* __restrict__ gate;
    const __nv_bfloat16* __restrict__ up;

    __device__ __forceinline__ void half8(const uint4 g4, const uint4 u4, float* out) const {
        float g[8];
        float u[8];
        hopper_unpack_bf16x8(g4, g);
        hopper_unpack_bf16x8(u4, u);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            out[i] = (g[i] / (1.0f + __expf(-g[i]))) * u[i];
        }
    }

    __device__ __forceinline__ void chunk(unsigned int k16, HopperActChunk& out) const {
        const uint4* g4 = (const uint4*)gate;
        const uint4* u4 = (const uint4*)up;
        half8(g4[k16 * 2], u4[k16 * 2], &out.v[0]);
        half8(g4[k16 * 2 + 1], u4[k16 * 2 + 1], &out.v[8]);
    }
};

// ── W8A16 GEMV Dual Projection (FP8 E4M3) ──
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 4) void w8a16_gemv_dual(
    const __nv_bfloat16* __restrict__ A,            // [1, K] shared input
    const unsigned char* __restrict__ B1,            // [N, K] proj 0 FP8 E4M3
    const float* __restrict__ B1_scale,              // [N/128, K/128] proj 0 FP32
    __nv_bfloat16* __restrict__ C1,                  // [1, N] proj 0 output
    const unsigned char* __restrict__ B2,            // [N, K] proj 1 FP8 E4M3
    const float* __restrict__ B2_scale,              // [N/128, K/128] proj 1 FP32
    __nv_bfloat16* __restrict__ C2,                  // [1, N] proj 1 output
    unsigned int N,
    unsigned int K
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B = proj == 0 ? B1 : B2;
    const float* block_scale = proj == 0 ? B1_scale : B2_scale;
    __nv_bfloat16* C = proj == 0 ? C1 : C2;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int K16 = K / K_PER_CHUNK;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;  // ceil(K/128)
    const unsigned int n_block = n / FP8_BLOCK;

    // The 1 KB E4M3 table is gone with the LUT gather; only the two-warp
    // reduction's 8 floats remain.
    __shared__ float smem[N_PER_BLOCK * 2];

    const HopperActRow act{A};
    const float acc = hopper_gemv_row<HOPPER_GEMV_UNROLL>(
        B + (unsigned long long)n * K,
        block_scale + (unsigned long long)n_block * k_blocks,
        act,
        K16,
        lane
    );

    hopper_gemv_reduce_store(acc, smem, local_out, lane, C, n);
}

// ── W8A16 GEMV with SiLU-fused Input (FP8 E4M3) ──
//
// UNROLL is 2, not `HOPPER_GEMV_UNROLL`: this kernel's activation decode holds
// two 8-float staging arrays on top of the chunk, and ptxas already needed 53
// registers (4 CTAs/SM) for the gb10 version. Two chunks in flight doubles the
// bytes in flight without pushing the working set past the 64-register ceiling
// `__launch_bounds__` sets here.
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 4) void w8a16_gemv_silu_input(
    const __nv_bfloat16* __restrict__ gate_out,     // [1, K] gate proj output
    const __nv_bfloat16* __restrict__ up_out,       // [1, K] up proj output
    const unsigned char* __restrict__ B,             // [N, K] down FP8 E4M3
    const float* __restrict__ block_scale,           // [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,                   // [1, N] output
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int K16 = K / K_PER_CHUNK;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;  // ceil(K/128)
    const unsigned int n_block = n / FP8_BLOCK;

    __shared__ float smem[N_PER_BLOCK * 2];

    const HopperActSilu act{gate_out, up_out};
    const float acc = hopper_gemv_row<2>(
        B + (unsigned long long)n * K,
        block_scale + (unsigned long long)n_block * k_blocks,
        act,
        K16,
        lane
    );

    hopper_gemv_reduce_store(acc, smem, local_out, lane, C, n);
}
