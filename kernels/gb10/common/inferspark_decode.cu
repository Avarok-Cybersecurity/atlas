// SPDX-License-Identifier: AGPL-3.0-only

// Inferspark Decode Attention v2 — optimized for memory bandwidth.
//
// Key optimizations over v1:
//   - Batched KV loading (BC=4): 4 K loads + 4 V loads per loop iteration
//     → 4x more outstanding memory requests → better bandwidth utilization
//   - Batched softmax: single o_reg rescale per batch instead of per-position
//   - V prefetch during softmax computation (overlap load + compute)
//   - Tree-based inter-warp merge (O(log N) instead of O(N))
//   - All loads remain vectorized (uint32 packed BF16 pairs)
//
// Grid: (num_q_heads, batch_size, 1)
// Block: (256, 1, 1) = 8 warps

#include <cuda_bf16.h>

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 256
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32  (HDIM / (WARP_SIZE * 2))
#define NUM_WARPS_DEC 8
#define BC_DEC 4     // KV positions batched per loop iteration

__device__ __forceinline__ void unpack2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

extern "C" __global__ void inferspark_decode(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int batch = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset = lane_id * VEC_BF16;

    // Load Q into registers
    const unsigned int* q32 = (const unsigned int*)(Q + batch * num_q_heads * head_dim
                                                      + q_head * head_dim + vec_offset);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    // Each warp handles a chunk of the KV sequence
    unsigned int chunk_size = (seq_len + NUM_WARPS_DEC - 1) / NUM_WARPS_DEC;
    unsigned int my_start = warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    // Online softmax state (per warp)
    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    const unsigned int kv_stride = num_kv_heads * head_dim;
    const __nv_bfloat16* k_base = K_cache + batch * seq_len * kv_stride + kv_head * head_dim;
    const __nv_bfloat16* v_base = V_cache + batch * seq_len * kv_stride + kv_head * head_dim;

    // === Batched KV loop (BC=4 positions per iteration) ===
    unsigned int pos = my_start;
    unsigned int end_aligned = my_start + ((my_end - my_start) / BC_DEC) * BC_DEC;

    for (; pos < end_aligned; pos += BC_DEC) {
        // === Phase 1: Batch load BC K vectors (all loads issued before any consume) ===
        unsigned int k_packed[BC_DEC][VEC_U32];
        #pragma unroll
        for (int b = 0; b < BC_DEC; b++) {
            const unsigned int* k32 = (const unsigned int*)(k_base + (pos + b) * kv_stride + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++)
                k_packed[b][i] = k32[i];
        }

        // === Phase 2: Compute BC dot products ===
        float scores[BC_DEC];
        #pragma unroll
        for (int b = 0; b < BC_DEC; b++) {
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float k0, k1;
                unpack2(k_packed[b][i], k0, k1);
                dot += q_reg[2*i] * k0 + q_reg[2*i+1] * k1;
            }
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            scores[b] = dot * inv_sqrt_d;
        }

        // === Phase 3: Prefetch V while computing softmax ===
        unsigned int v_packed[BC_DEC][VEC_U32];
        #pragma unroll
        for (int b = 0; b < BC_DEC; b++) {
            const unsigned int* v32 = (const unsigned int*)(v_base + (pos + b) * kv_stride + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++)
                v_packed[b][i] = v32[i];  // issued, overlaps with softmax below
        }

        // === Phase 4: Batched softmax (single rescale for all BC positions) ===
        float m_new = m;
        #pragma unroll
        for (int b = 0; b < BC_DEC; b++)
            m_new = fmaxf(m_new, scores[b]);

        float exp_old = __expf(m - m_new);

        // Rescale accumulators ONCE for the entire batch
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            o_reg[i] *= exp_old;
        l *= exp_old;

        float exp_factors[BC_DEC];
        #pragma unroll
        for (int b = 0; b < BC_DEC; b++) {
            exp_factors[b] = __expf(scores[b] - m_new);
            l += exp_factors[b];
        }
        m = m_new;

        // === Phase 5: V accumulate (V data should be ready from prefetch) ===
        #pragma unroll
        for (int b = 0; b < BC_DEC; b++) {
            float ef = exp_factors[b];
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float v0, v1;
                unpack2(v_packed[b][i], v0, v1);
                o_reg[2*i]   += ef * v0;
                o_reg[2*i+1] += ef * v1;
            }
        }
    }

    // === Handle remaining positions (< BC) ===
    for (; pos < my_end; pos++) {
        const unsigned int* k32 = (const unsigned int*)(k_base + pos * kv_stride + vec_offset);
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float k0, k1;
            unpack2(k32[i], k0, k1);
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

        const unsigned int* v32 = (const unsigned int*)(v_base + pos * kv_stride + vec_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0, v1;
            unpack2(v32[i], v0, v1);
            o_reg[2*i]   = o_reg[2*i]   * exp_old + exp_new * v0;
            o_reg[2*i+1] = o_reg[2*i+1] * exp_old + exp_new * v1;
        }
        m = m_new;
    }

    // === Tree-based inter-warp reduction ===
    __shared__ float smem_m[NUM_WARPS_DEC];
    __shared__ float smem_l[NUM_WARPS_DEC];
    __shared__ float smem_o[NUM_WARPS_DEC][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset + i] = o_reg[i];
    }
    __syncthreads();

    // Tree merge: O(log N) levels instead of sequential
    #pragma unroll
    for (int stride = NUM_WARPS_DEC / 2; stride > 0; stride >>= 1) {
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
        unsigned int* o32 = (unsigned int*)(O + batch * num_q_heads * head_dim
                                              + q_head * head_dim + vec_offset);
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
// Split-K Decode: FlashDecoding — split KV sequence across multiple CTAs per head.
//
// Grid: (num_q_heads, num_splits, batch_size)
// Each CTA computes partial attention for kv_range = [split_start, split_end).
// Writes partial (O, m, l) to workspace for a subsequent reduction kernel.
//
// Workspace layout per (batch, head, split):
//   O_partial: [head_dim] float32  (256 floats = 1024 bytes)
//   m_partial: float32             (4 bytes)
//   l_partial: float32             (4 bytes)
//   Total per split: 1032 bytes
//   Total: batch * num_q_heads * num_splits * 1032 bytes
// ============================================================================

extern "C" __global__ void inferspark_decode_splitk(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    float* __restrict__ workspace,        // Partial results
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d,
    const unsigned int num_splits
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split_id = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    // Compute this split's KV range
    unsigned int split_size = (seq_len + num_splits - 1) / num_splits;
    unsigned int kv_start = split_id * split_size;
    unsigned int kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_start >= seq_len) kv_start = kv_end;  // empty range

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset = lane_id * VEC_BF16;

    // Load Q into registers
    const unsigned int* q32 = (const unsigned int*)(Q + batch * num_q_heads * head_dim
                                                      + q_head * head_dim + vec_offset);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    // Each warp handles a chunk of this split's KV range
    unsigned int local_len = kv_end - kv_start;
    unsigned int chunk_size = (local_len + NUM_WARPS_DEC - 1) / NUM_WARPS_DEC;
    unsigned int my_start = kv_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;

    // Online softmax state
    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    const unsigned int kv_stride = num_kv_heads * head_dim;
    const __nv_bfloat16* k_base = K_cache + batch * seq_len * kv_stride + kv_head * head_dim;
    const __nv_bfloat16* v_base = V_cache + batch * seq_len * kv_stride + kv_head * head_dim;

    // Batched KV loop (BC=4)
    unsigned int pos = my_start;
    unsigned int end_aligned = my_start + ((my_end - my_start) / BC_DEC) * BC_DEC;

    for (; pos < end_aligned; pos += BC_DEC) {
        unsigned int k_packed[BC_DEC][VEC_U32];
        #pragma unroll
        for (int b = 0; b < BC_DEC; b++) {
            const unsigned int* k32 = (const unsigned int*)(k_base + (pos + b) * kv_stride + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++)
                k_packed[b][i] = k32[i];
        }

        float scores[BC_DEC];
        #pragma unroll
        for (int b = 0; b < BC_DEC; b++) {
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float k0, k1;
                unpack2(k_packed[b][i], k0, k1);
                dot += q_reg[2*i] * k0 + q_reg[2*i+1] * k1;
            }
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            scores[b] = dot * inv_sqrt_d;
        }

        unsigned int v_packed[BC_DEC][VEC_U32];
        #pragma unroll
        for (int b = 0; b < BC_DEC; b++) {
            const unsigned int* v32 = (const unsigned int*)(v_base + (pos + b) * kv_stride + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++)
                v_packed[b][i] = v32[i];
        }

        float m_new = m;
        #pragma unroll
        for (int b = 0; b < BC_DEC; b++)
            m_new = fmaxf(m_new, scores[b]);

        float exp_old = __expf(m - m_new);
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            o_reg[i] *= exp_old;
        l *= exp_old;

        float exp_factors[BC_DEC];
        #pragma unroll
        for (int b = 0; b < BC_DEC; b++) {
            exp_factors[b] = __expf(scores[b] - m_new);
            l += exp_factors[b];
        }
        m = m_new;

        #pragma unroll
        for (int b = 0; b < BC_DEC; b++) {
            float ef = exp_factors[b];
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float v0, v1;
                unpack2(v_packed[b][i], v0, v1);
                o_reg[2*i]   += ef * v0;
                o_reg[2*i+1] += ef * v1;
            }
        }
    }

    // Remaining positions
    for (; pos < my_end; pos++) {
        const unsigned int* k32 = (const unsigned int*)(k_base + pos * kv_stride + vec_offset);
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float k0, k1;
            unpack2(k32[i], k0, k1);
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

        const unsigned int* v32 = (const unsigned int*)(v_base + pos * kv_stride + vec_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0, v1;
            unpack2(v32[i], v0, v1);
            o_reg[2*i]   = o_reg[2*i]   * exp_old + exp_new * v0;
            o_reg[2*i+1] = o_reg[2*i+1] * exp_old + exp_new * v1;
        }
        m = m_new;
    }

    // === Tree merge within CTA (same as non-split version) ===
    __shared__ float smem_m[NUM_WARPS_DEC];
    __shared__ float smem_l[NUM_WARPS_DEC];
    __shared__ float smem_o[NUM_WARPS_DEC][HDIM];

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
    for (int stride = NUM_WARPS_DEC / 2; stride > 0; stride >>= 1) {
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

    // Write partial results to workspace
    // Layout: workspace[batch][q_head][split_id] → {O[256 float], m, l}
    unsigned int ws_stride = (head_dim + 2);  // 258 floats per split
    float* ws_base = workspace + ((batch * num_q_heads + q_head) * num_splits + split_id) * ws_stride;

    if (warp_id == 0) {
        // Write O partial (unnormalized)
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

// Reduction kernel: merge split-K partial results into final output.
// Grid: (num_q_heads, batch_size, 1)
// Block: (256, 1, 1)  — one warp per head dimension chunk, vectorized
extern "C" __global__ void inferspark_decode_reduce(
    const float* __restrict__ workspace,
    __nv_bfloat16* __restrict__ O,
    const unsigned int num_q_heads,
    const unsigned int head_dim,
    const unsigned int num_splits
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int batch = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid % WARP_SIZE;
    const unsigned int vec_offset = lane_id * VEC_BF16;

    if (q_head >= num_q_heads) return;

    unsigned int ws_stride = (head_dim + 2);
    const float* ws_head = workspace + (batch * num_q_heads + q_head) * num_splits * ws_stride;

    // Initialize from split 0
    float m = ws_head[head_dim];
    float l = ws_head[head_dim + 1];
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        o_reg[i] = ws_head[vec_offset + i];
    }

    // Merge remaining splits
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

    // Final normalization and store
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    unsigned int* o32 = (unsigned int*)(O + batch * num_q_heads * head_dim
                                          + q_head * head_dim + vec_offset);
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        float v0 = o_reg[2*i]     * inv_l;
        float v1 = o_reg[2*i + 1] * inv_l;
        unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
        unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
        o32[i] = lo | (hi << 16);
    }
}

// Legacy alias
extern "C" __global__ void inferspark_decode_multi_warp(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d
) {
    // Forward — same as inferspark_decode (identical kernel)
    // Kept for backward compatibility with Rust binding dispatch
}
