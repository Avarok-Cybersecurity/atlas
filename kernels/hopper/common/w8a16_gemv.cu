// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 GEMV, HOPPER-TUNED — fused FP8-E4M3 weight dequant + BF16 GEMV
// for M=1 decode (#928).
//
//   out[n] = sum_k A[0,k] * E4M3[B[n,k]] * block_scale[n/128, k/128]
//
// This is the H100/H200 override of `kernels/gb10/common/w8a16_gemv.cu`. Same
// entry point, same six arguments, same `ceil(N/4)` x 256 launch geometry from
// `ops::w8a16_gemv`, and the same arithmetic in the same order — so it is a
// drop-in the Rust side never learns about, and the gb10 kernel is untouched.
//
// It is 40% of the C=1 decode step on an H100 (nsys, 1xH100, Qwen/Qwen3.8-27B
// -FP8, 2026-09-11 round 10: 224 launches, 7.39 ms of an 18.5 ms step) at an
// aggregate 1.84 TB/s against HBM3's 3.35 TB/s peak. The two reasons it is not
// bandwidth-bound on this hardware — a shared-memory LUT gather that saturates
// the SM load/store unit, and too few weight bytes in flight per SM at the
// small-N shapes the host grid cannot fill — are diagnosed with their
// arithmetic, and the fixes and the bit-identity argument are stated, in
// `w8a16_gemv_hopper.cuh`. Read that file first; this one is the entry point.
//
// FP8-E4M3 weight format (unchanged from gb10):
//   B:           [N, K]          uint8 — one byte per weight (FP8 E4M3)
//   block_scale: [N/128, K/128]  FP32  — per-128x128-block scale (scale_inv
//                from the checkpoint, widened to FP32 at load so the scale is
//                applied in full FP32 precision, matching vLLM / DeepGEMM / HF
//                block-FP8 numerics)
//
// 4 outputs per block, 64 lanes per output, 16-byte weight reads.
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)

#include "w8a16_gemv_hopper.cuh"

// `__launch_bounds__` states the geometry the host launcher guarantees. The
// second argument is a FLOOR on CTAs per SM, not a wish: at 256 threads it
// caps ptxas at 64 registers, which is what keeps the UNROLL=4 pipeline from
// trading its bytes-in-flight win back for an occupancy loss. Register and
// spill counts are pinned by `scripts/hopper_ptx_gate.sh` and by the
// `nvcc -cubin -Xptxas -v -arch=sm_90a` receipt in the commit message.
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 4) void w8a16_gemv(
    const __nv_bfloat16* __restrict__ A,            // [1, K]
    const unsigned char* __restrict__ B,             // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,           // [N/BS, K/BS] FP32
    __nv_bfloat16* __restrict__ C,                   // [1, N]
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

    // No LUT staging: `cvt.rn.f16x2.e4m3x2` decodes on the math pipe, so the
    // only shared memory left is the 8 floats the two-warp reduction needs
    // (1,056 B -> 32 B). The first `__syncthreads` of the gb10 kernel went
    // with the table; the one in the reduction is unchanged.
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
