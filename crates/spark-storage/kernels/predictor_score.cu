// SPDX-License-Identifier: AGPL-3.0-only
//
// predictor_score: per-block attention-importance score for `--high-speed-swap`.
//
// score[blk] = max over (q_head, tok in blk) of  ⟨q_proj[q_head], K_lr[blk, kv_head(qh), tok]⟩
//
// where kv_head(qh) = q_head / gqa_ratio. The token max replaces the previous
// block-mean anchor: softmax weight is dominated by the single highest-
// alignment token, which a mean reduction systematically misses.
//
// Shapes (all BF16 unless noted):
//   q_proj   : [num_q_heads, r]
//   K_lr_seq : [num_active_blocks, num_kv_heads, block_size, r]   layout
//              MUST match `kv_lowrank_project` writer.
//   scores   : [num_active_blocks]                                f32 output
//
// Launch:
//   grid  = (num_active_blocks, 1, 1)
//   block = (BLOCK_THREADS,     1, 1)   power-of-two ≤ 256

#include <cuda_bf16.h>

constexpr int BLOCK_THREADS = 128;

static __device__ inline float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    }
    return v;
}

extern "C" __global__ void predictor_score(
    const __nv_bfloat16* __restrict__ q_proj,    // [num_q_heads, r]
    const __nv_bfloat16* __restrict__ K_lr_seq,  // [num_active_blocks, num_kv_heads, block_size, r]
    float*               __restrict__ scores,    // [num_active_blocks]
    int num_q_heads,
    int num_kv_heads,
    int block_size,
    int r,
    int gqa_ratio
) {
    const int block_idx  = blockIdx.x;
    const int tid        = threadIdx.x;
    const size_t per_block_floats = (size_t)num_kv_heads * block_size * r;
    const __nv_bfloat16* k_lr_block = K_lr_seq + (size_t)block_idx * per_block_floats;

    float my_max = -INFINITY;

    // Stride: each thread sweeps a subset of (q_head, tok) pairs.
    const int total_pairs = num_q_heads * block_size;
    for (int p = tid; p < total_pairs; p += blockDim.x) {
        const int q_head = p / block_size;
        const int tok    = p % block_size;
        const int kv_head = q_head / gqa_ratio;
        const __nv_bfloat16* q_row = q_proj + (size_t)q_head * r;
        const __nv_bfloat16* k_row = k_lr_block
            + ((size_t)kv_head * block_size + tok) * r;
        float dot = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < r; ++i) {
            dot = fmaf(__bfloat162float(q_row[i]),
                       __bfloat162float(k_row[i]),
                       dot);
        }
        if (dot > my_max) my_max = dot;
    }

    // Block-wide max reduction.
    __shared__ float warp_max[BLOCK_THREADS / 32];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    my_max = warp_reduce_max(my_max);
    if (lane == 0) warp_max[warp] = my_max;
    __syncthreads();

    if (warp == 0) {
        const int n_warps = blockDim.x / 32;
        my_max = (lane < n_warps) ? warp_max[lane] : -INFINITY;
        my_max = warp_reduce_max(my_max);
        if (lane == 0) scores[block_idx] = my_max;
    }
}
