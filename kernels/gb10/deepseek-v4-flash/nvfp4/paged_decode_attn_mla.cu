// SPDX-License-Identifier: AGPL-3.0-only

// MLA Absorbed Paged Decode — HDIM=576 for compressed latent KV cache.
// Override the compile-time HDIM to match kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576.
#undef HDIM
#define HDIM 576

// Paged Decode Attention — decode attention reading K/V from paged cache via block table.
//
// Compatible with vLLM's paged attention interface.
// Uses NHD cache layout: [num_blocks, block_size, num_kv_heads, head_dim] BF16.
//
// Key design:
//   - One CTA per (q_head, seq) pair
//   - 8 warps split the KV sequence, each thread covers head_dim/32 = 8 BF16 elements
//   - Block table lookup to find physical blocks for each logical position
//   - Within-block positions are contiguous → good memory coalescing
//   - Batched KV loading (BC=4) within blocks, single-load at block boundaries
//   - Online softmax with tree-based inter-warp reduction
//
// Grid: (num_q_heads, num_seqs, 1)   [or with split-K: (num_q_heads, num_splits, num_seqs)]
// Block: (256, 1, 1)

#include <cuda_bf16.h>

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 256
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32  (HDIM / (WARP_SIZE * 2))
#define NUM_WARPS 8
#define BC 4            // KV positions batched per loop iteration

__device__ __forceinline__ void unpack2_pd(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

// Helper: compute pointer to K or V for a given position in paged cache
__device__ __forceinline__ const __nv_bfloat16* paged_kv_ptr(
    const __nv_bfloat16* __restrict__ cache,   // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_table,       // [max_blocks_per_seq]
    unsigned int pos,
    unsigned int block_size,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int kv_head
) {
    unsigned int logical_block = pos / block_size;
    unsigned int block_offset = pos % block_size;
    unsigned int physical_block = (unsigned int)block_table[logical_block];
    unsigned long long page_stride = (unsigned long long)block_size * num_kv_heads * head_dim;
    return cache + (unsigned long long)physical_block * page_stride
                 + (unsigned long long)block_offset * num_kv_heads * head_dim
                 + (unsigned long long)kv_head * head_dim;
}

extern "C" __global__ void paged_decode_attn(
    const __nv_bfloat16* __restrict__ Q,          // [num_seqs, num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ K_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ V_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ O,                // [num_seqs, num_q_heads, head_dim]
    const int* __restrict__ block_tables,         // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,             // [num_seqs]
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride               // query.stride(0) in elements
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset = lane_id * VEC_BF16;

    // Block table for this sequence
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q into registers (strided: Q may be a non-contiguous QKV split view)
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_pd(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    // Each warp handles a chunk of the KV sequence
    unsigned int chunk_size = (seq_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    // Online softmax state
    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    // === Main loop: process positions with batched KV loading ===
    // We can batch BC=4 positions when they're in the same physical block.
    // At block boundaries, fall back to single-position processing.
    unsigned int pos = my_start;

    while (pos < my_end) {
        // Check how many consecutive positions share the same physical block
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

        // Get physical block pointer base
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned long long page_stride = (unsigned long long)block_size * num_kv_heads * head_dim;
        unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;
        const __nv_bfloat16* k_block_base = K_cache + (unsigned long long)physical_block * page_stride
                                                     + (unsigned long long)block_offset * head_stride_kv
                                                     + (unsigned long long)kv_head * head_dim;
        const __nv_bfloat16* v_block_base = V_cache + (unsigned long long)physical_block * page_stride
                                                     + (unsigned long long)block_offset * head_stride_kv
                                                     + (unsigned long long)kv_head * head_dim;

        // Process in batches of BC within this physical block
        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;

        // Batched path: BC=4 positions at a time (contiguous in memory)
        for (; processed < aligned_count; processed += BC) {
            // Load BC K vectors
            unsigned int k_packed[BC][VEC_U32];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* k32 = (const unsigned int*)(k_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++)
                    k_packed[b][i] = k32[i];
            }

            // Compute BC dot products
            float scores[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    float k0, k1;
                    unpack2_pd(k_packed[b][i], k0, k1);
                    dot += q_reg[2*i] * k0 + q_reg[2*i+1] * k1;
                }
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                scores[b] = dot * inv_sqrt_d;
            }

            // Prefetch V
            unsigned int v_packed[BC][VEC_U32];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* v32 = (const unsigned int*)(v_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++)
                    v_packed[b][i] = v32[i];
            }

            // Batched softmax
            float m_new = m;
            #pragma unroll
            for (int b = 0; b < BC; b++)
                m_new = fmaxf(m_new, scores[b]);

            float exp_old = __expf(m - m_new);
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                o_reg[i] *= exp_old;
            l *= exp_old;

            float exp_factors[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                exp_factors[b] = __expf(scores[b] - m_new);
                l += exp_factors[b];
            }
            m = m_new;

            // V accumulate
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float ef = exp_factors[b];
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    float v0, v1;
                    unpack2_pd(v_packed[b][i], v0, v1);
                    o_reg[2*i]   += ef * v0;
                    o_reg[2*i+1] += ef * v1;
                }
            }
        }

        // Remainder: single positions
        for (; processed < batch_count; processed++) {
            const unsigned int* k32 = (const unsigned int*)(k_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float k0, k1;
                unpack2_pd(k32[i], k0, k1);
                dot += q_reg[2*i] * k0 + q_reg[2*i+1] * k1;
            }
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot * inv_sqrt_d;
            float m_new = fmaxf(m, score);
            float exp_old = __expf(m - m_new);
            float exp_new = __expf(score - m_new);
            l = l * exp_old + exp_new;

            const unsigned int* v32 = (const unsigned int*)(v_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float v0, v1;
                unpack2_pd(v32[i], v0, v1);
                o_reg[2*i]   = o_reg[2*i]   * exp_old + exp_new * v0;
                o_reg[2*i+1] = o_reg[2*i+1] * exp_old + exp_new * v1;
            }
            m = m_new;
        }

        pos += batch_count;
    }

    // === Tree-based inter-warp reduction ===
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset + i] = o_reg[i];
    }
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.0f) {
                float mw = smem_m[other];
                float my_m = smem_m[warp_id];
                float my_l = smem_l[warp_id];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new);
                float scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    smem_o[warp_id][vec_offset + i] =
                        smem_o[warp_id][vec_offset + i] * scale_me +
                        smem_o[other][vec_offset + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    // Warp 0 writes final output
    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim + vec_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0 = smem_o[0][vec_offset + 2*i]     * inv_l;
            float v1 = smem_o[0][vec_offset + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}

// ============================================================================
// Split-K variant for long sequences with few heads (low SM utilization)
// Grid: (num_q_heads, num_splits, num_seqs)
// ============================================================================

extern "C" __global__ void paged_decode_attn_splitk(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    float* __restrict__ workspace,               // [num_seqs, num_q_heads, num_splits, head_dim+2]
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride               // query.stride(0) in elements
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split_id = blockIdx.y;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Compute this split's KV range
    unsigned int split_size = (seq_len + num_splits - 1) / num_splits;
    unsigned int kv_start = split_id * split_size;
    unsigned int kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_start >= seq_len) kv_start = kv_end;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset = lane_id * VEC_BF16;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q (strided: Q may be a non-contiguous QKV split view)
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_pd(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    // Each warp handles a chunk of this split's range
    unsigned int local_len = kv_end - kv_start;
    unsigned int chunk_size = (local_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = kv_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;
    unsigned long long page_stride = (unsigned long long)block_size * head_stride_kv;

    // Process positions
    for (unsigned int pos = my_start; pos < my_end; pos++) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const unsigned int* k32 = (const unsigned int*)(K_cache
            + (unsigned long long)physical_block * page_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float k0, k1;
            unpack2_pd(k32[i], k0, k1);
            dot += q_reg[2*i] * k0 + q_reg[2*i+1] * k1;
        }
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, offset);

        float score = dot * inv_sqrt_d;
        float m_new = fmaxf(m, score);
        float exp_old = __expf(m - m_new);
        float exp_new = __expf(score - m_new);
        l = l * exp_old + exp_new;

        const unsigned int* v32 = (const unsigned int*)(V_cache
            + (unsigned long long)physical_block * page_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim + vec_offset);

        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0, v1;
            unpack2_pd(v32[i], v0, v1);
            o_reg[2*i]   = o_reg[2*i]   * exp_old + exp_new * v0;
            o_reg[2*i+1] = o_reg[2*i+1] * exp_old + exp_new * v1;
        }
        m = m_new;
    }

    // Tree merge within CTA
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset + i] = o_reg[i];
    }
    __syncthreads();

    #pragma unroll
    for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.0f) {
                float mw = smem_m[other];
                float my_m = smem_m[warp_id];
                float my_l = smem_l[warp_id];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new);
                float scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    smem_o[warp_id][vec_offset + i] =
                        smem_o[warp_id][vec_offset + i] * scale_me +
                        smem_o[other][vec_offset + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    // Write partial to workspace
    unsigned int ws_stride = (head_dim + 2);
    float* ws_base = workspace + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride
                   + split_id * ws_stride;

    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) {
            ws_base[vec_offset + i] = smem_o[0][vec_offset + i];
        }
        if (lane_id == 0) {
            ws_base[head_dim] = smem_m[0];
            ws_base[head_dim + 1] = smem_l[0];
        }
    }
}

// Reduction kernel — identical to inferspark_decode_reduce (reuse is fine)
extern "C" __global__ void paged_decode_attn_reduce(
    const float* __restrict__ workspace,
    __nv_bfloat16* __restrict__ O,
    const unsigned int num_q_heads,
    const unsigned int head_dim,
    const unsigned int num_splits
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid % WARP_SIZE;
    const unsigned int vec_offset = lane_id * VEC_BF16;

    if (q_head >= num_q_heads) return;

    unsigned int ws_stride = (head_dim + 2);
    const float* ws_head = workspace + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride;

    float m = ws_head[head_dim];
    float l = ws_head[head_dim + 1];
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        o_reg[i] = ws_head[vec_offset + i];
    }

    for (unsigned int s = 1; s < num_splits; s++) {
        const float* ws_s = ws_head + s * ws_stride;
        float ms = ws_s[head_dim];
        float ls = ws_s[head_dim + 1];
        if (ls > 0.0f) {
            float m_new = fmaxf(m, ms);
            float scale_me = __expf(m - m_new);
            float scale_s = __expf(ms - m_new);
            l = l * scale_me + ls * scale_s;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) {
                o_reg[i] = o_reg[i] * scale_me + ws_s[vec_offset + i] * scale_s;
            }
            m = m_new;
        }
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                          + (unsigned long long)q_head * head_dim + vec_offset);
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        float v0 = o_reg[2*i]     * inv_l;
        float v1 = o_reg[2*i + 1] * inv_l;
        unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
        unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
        o32[i] = lo | (hi << 16);
    }
}
