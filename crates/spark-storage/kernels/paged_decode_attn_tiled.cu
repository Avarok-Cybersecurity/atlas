// SPDX-License-Identifier: AGPL-3.0-only
//
// paged_decode_attn_tiled: online-softmax decode attention over a TILE of
// blocks. Repeated launches of this kernel — one per tile — accumulate into
// a per-(seq, q_head) running state (m, l, o) so that the host can stream
// blocks from NVMe → HBM scratch and feed them through attention without
// ever materialising the full sequence in HBM.
//
// State semantics (initial values for the first tile of a step):
//   m_state = -INF    (BF16 -inf is fine but we store FP32 for accumulation)
//   l_state = 0
//   o_state = 0
// On each tile:
//   for (block, token) in tile: update (m, l, o) using the FlashAttention-2
//   online-softmax recurrence (tested against a single-tile reference).
// After the last tile: caller invokes `attention_finalize` to write
// `output = o / l` as BF16.
//
// Layout assumptions match the production paged_decode_attn (kernels/gb10/
// nvfp4/paged_decode_attn.cu): K_pool, V_pool are NHD
// `[num_blocks, block_size, num_kv_heads, head_dim]` BF16. The tile is
// described by an *int32* block-id list per sequence and a per-sequence
// count of valid entries.

#include <cuda_bf16.h>

constexpr int MAX_HEAD_DIM = 256;
constexpr int MAX_WARPS    = MAX_HEAD_DIM / 32;

static __device__ inline float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

// blk_stride / tok_stride / kvh_stride are in BF16 elements. The kernel
// addresses K[blk, tok, kv_head, dim] = K_pool + blk*blk_stride +
// tok*tok_stride + kv_head*kvh_stride + dim. V_pool uses the same strides.
// Default (kernel-native paged) layout:
//   blk_stride = block_size * num_kv_heads * head_dim
//   tok_stride =              num_kv_heads * head_dim
//   kvh_stride =                             head_dim
// Scratch-pool layout `[slot][kv_head][tok][dim]`:
//   blk_stride = 2 * num_kv_heads * block_size * head_dim   (full slot stride; K and V interleaved per slot)
//   tok_stride =                                  head_dim
//   kvh_stride =                  block_size *    head_dim
// And V_pool = K_pool + (num_kv_heads * block_size * head_dim) elements.
extern "C" __global__ void paged_decode_attn_tiled(
    const __nv_bfloat16* __restrict__ Q,                // [num_seqs, num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ K_pool,
    const __nv_bfloat16* __restrict__ V_pool,
    const int*           __restrict__ tile_blocks,      // [num_seqs, tile_capacity]
    const int*           __restrict__ tile_block_counts,// [num_seqs]
    float*               __restrict__ m_state,          // [num_seqs, num_q_heads]
    float*               __restrict__ l_state,          // [num_seqs, num_q_heads]
    float*               __restrict__ o_state,          // [num_seqs, num_q_heads, head_dim]
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int tile_capacity,
    int gqa_ratio,
    long long blk_stride,
    long long tok_stride,
    long long kvh_stride,
    // Causal mask parameters. When > 0 and >= 1, the kernel only consumes
    // the FIRST `last_block_valid_slots` slots of the LAST block in the
    // tile (i.e., the block containing the query's own position). All
    // other blocks are processed in full. Set to `block_size` for unmasked
    // (decode) attention; set to `(q_pos % block_size) + 1` for prefill
    // per-query causal attention. Required to prevent future-token
    // attention leakage in prefill HSS streaming attention.
    int last_block_valid_slots
) {
    const int seq = blockIdx.x;
    const int qh  = blockIdx.y;
    const int tid = threadIdx.x;
    const int kh  = qh / gqa_ratio;

    __shared__ float Q_sm[MAX_HEAD_DIM];
    __shared__ float O_sm[MAX_HEAD_DIM];
    __shared__ float warp_buf[MAX_WARPS];
    __shared__ float logit_sh;

    // Load Q row + O running state into shared mem.
    if (tid < head_dim) {
        Q_sm[tid] = __bfloat162float(Q[((size_t)seq * num_q_heads + qh) * head_dim + tid]);
        O_sm[tid] = o_state[((size_t)seq * num_q_heads + qh) * head_dim + tid];
    }
    float m_run = m_state[seq * num_q_heads + qh];
    float l_run = l_state[seq * num_q_heads + qh];
    __syncthreads();

    const int n_blocks = tile_block_counts[seq];
    const float inv_sqrt_d = rsqrtf((float)head_dim);
    const int n_warps = (blockDim.x + 31) / 32;

    // Causal mask: in the LAST block of the tile, process only the first
    // `last_block_valid_slots` slots. All other blocks are processed in
    // full. This prevents prefill queries from attending to future tokens
    // in their own block. Pass `block_size` (or any value >= block_size)
    // to disable masking (decode case).
    const int t_max_last = last_block_valid_slots < block_size
        ? last_block_valid_slots : block_size;
    for (int b = 0; b < n_blocks; ++b) {
        const int blk_id = tile_blocks[(size_t)seq * tile_capacity + b];
        const size_t blk_base = (size_t)blk_id * (size_t)blk_stride;
        const int t_lim = (b == n_blocks - 1) ? t_max_last : block_size;

        for (int t = 0; t < t_lim; ++t) {
            const size_t kv_base = blk_base
                + (size_t)t * (size_t)tok_stride
                + (size_t)kh * (size_t)kvh_stride;

            // Compute partial Q · K_t (each thread holds one element).
            float partial = 0.0f;
            if (tid < head_dim) {
                partial = Q_sm[tid] * __bfloat162float(K_pool[kv_base + tid]);
            }
            // Block-wide sum: warp shuffle, write per-warp partial to shared,
            // then reduce in the first warp.
            float w = warp_sum(partial);
            const int lane = tid & 31;
            const int warp = tid >> 5;
            if (lane == 0) warp_buf[warp] = w;
            __syncthreads();
            if (warp == 0) {
                float v = (lane < n_warps) ? warp_buf[lane] : 0.0f;
                v = warp_sum(v);
                if (lane == 0) logit_sh = v * inv_sqrt_d;
            }
            __syncthreads();
            const float logit = logit_sh;

            // FA-2 online softmax recurrence (every thread computes the
            // same scalars — branchless via fmaxf/expf).
            const float m_new = fmaxf(m_run, logit);
            const float scale_old = __expf(m_run - m_new);
            const float scale_new = __expf(logit - m_new);
            const float l_new = l_run * scale_old + scale_new;

            // Update O.
            if (tid < head_dim) {
                const float v = __bfloat162float(V_pool[kv_base + tid]);
                O_sm[tid] = O_sm[tid] * scale_old + v * scale_new;
            }
            __syncthreads();

            m_run = m_new;
            l_run = l_new;
        }
    }

    // Persist updated state.
    if (tid == 0) {
        m_state[seq * num_q_heads + qh] = m_run;
        l_state[seq * num_q_heads + qh] = l_run;
    }
    if (tid < head_dim) {
        o_state[((size_t)seq * num_q_heads + qh) * head_dim + tid] = O_sm[tid];
    }
}
