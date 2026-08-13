// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Dense BF16 batched GEMV (M rows) for SM121 (GB10).
//
// The M-row generalisation of dense_gemv_bf16_batch2: computes M output rows
// from ONE pass over the BF16 weight matrix, so weight bandwidth is paid once
// instead of M times. Bit-identical to running dense_gemv_bf16 M times — each
// row's accumulator follows the exact same K-iteration order and reduction
// tree; the extra rows only add independent accumulators over the same loop.
// (KERNEL.toml builds this dir with --fmad=false, which is what makes that
// identity hold rather than merely "close".)
//
//   C[t, n] = dot(A[t, :], B[n, :])   for t in [0, M)
//
//   A: [M, K] BF16 (activation rows, contiguous — this is exactly the layout
//      the multi-seq decode path already has: `normed.offset(i * h * bf16)`)
//   B: [N, K] BF16 (weights, row-major)
//   C: M rows at C + t * out_stride (BF16 elements)
//
// `out_stride` decouples the output row stride from N so callers can write
// straight into per-token strided layouts (e.g. the multi-seq qkv buffer,
// whose rows are `per_seq_qkv` apart, not `N` apart).
//
// WHY THIS EXISTS: at decode, Laguna's q/k/v/o and shared-expert projections
// are BF16 (the checkpoint ships them unquantized and they stay that way), and
// the BF16 path had no batched tier — only the quantized paths did
// (w4a16_gemv_batch2/3/4, w8a16_gemv_batch2/4). So every sequence in a decode
// batch re-read the whole weight matrix, making 54% of the decode step scale
// linearly with concurrency.
//
// A tile-based GEMM is the wrong tool here: at M<=4 an M64-tile GEMM is ~94%
// padding, and was measured 3.6x SLOWER than the batched GEMV on this exact
// workload (see the note in multi_seq/qkv.rs::wide_verify_gemm).
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define VEC_SIZE 8   // BF16 values per vectorized load (uint4 = 16 bytes)
#define MAX_M 8      // compile-time cap on batched rows; callers must pass M <= MAX_M

extern "C" __global__ void dense_gemv_bf16_batchm(
    const __nv_bfloat16* __restrict__ A,  // [M, K]
    const __nv_bfloat16* __restrict__ B,  // [N, K]
    __nv_bfloat16* __restrict__ C,        // rows at C + t*out_stride
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride                // BF16 elements between output rows
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int m = (M > MAX_M) ? MAX_M : M;

    float acc[MAX_M];
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        // ONE weight load feeds every row — this is the whole point.
        uint4 b_data = B_vec[kv];
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        float bf[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 b_lo, b_hi;
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            bf[2 * i] = __bfloat162float(b_lo);
            bf[2 * i + 1] = __bfloat162float(b_hi);
        }

        for (unsigned int t = 0; t < m; t++) {
            const uint4* At_vec = (const uint4*)(A + (unsigned long long)t * K);
            uint4 a_data = At_vec[kv];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            float a = acc[t];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat16 a_lo, a_hi;
                *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                // Same add order as dense_gemv_bf16: lo then hi, per vector slot.
                a += __bfloat162float(a_lo) * bf[2 * i];
                a += __bfloat162float(a_hi) * bf[2 * i + 1];
            }
            acc[t] = a;
        }
    }

    // Scalar tail for K not divisible by VEC_SIZE (never hits for model dims).
    {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            const float bfv = __bfloat162float(B_row[k]);
            for (unsigned int t = 0; t < m; t++) {
                acc[t] += __bfloat162float(A[(unsigned long long)t * K + k]) * bfv;
            }
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    for (unsigned int t = 0; t < m; t++) {
        float a = acc[t];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        acc[t] = a;
    }

    // 2 warps per output: cross-warp reduce via shared memory, per row.
    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];

    if (warp_lane == 0) {
        const unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        for (unsigned int t = 0; t < m; t++) smem[t][smem_idx] = acc[t];
    }
    __syncthreads();

    if (lane == 0) {
        for (unsigned int t = 0; t < m; t++) {
            const float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * out_stride + n] = __float2bfloat16(r);
        }
    }
}
