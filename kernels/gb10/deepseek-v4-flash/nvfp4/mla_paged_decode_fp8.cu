// SPDX-License-Identifier: AGPL-3.0-only

// MLA Paged Decode — FP8 variant for DeepSeek-V4-Flash.
//
// DeepSeek-V4-Flash uses compressed KV cache with MLA (Multi-head Latent Attention):
// - KV cache: 576 dims per token (512 latent + 64 rope), stored in FP8 format
// - Q: 32768 dims (64 heads × 512 dims per head), BF16
// - Output: 32768 dims (64 heads × 512 latent dims), BF16

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#define VEC_BF16 16  // 512 / 32 = 16 elements per thread for Q
#define VEC_U32  8   // 512 / (32 * 2) = 8 uint32 per thread for Q
#define NUM_WARPS 8
#define BC 4
#define KV_LORA_DIM 512
#define ROPE_DIM 64
#define MLA_CACHE_DIM 576  // raw paged cache token: KV_LORA_DIM(512) + ROPE_DIM(64)
// 4b compressed pool block width = qk_nope_head_dim(448) + qk_rope_head_dim(64) = 512.
// The compressor (cache_skip_v4) builds comp_k at hd_mla = nope+rope = 512 with rope
// IN-PLACE at dims 448-511 (NOT the 512-575 tail). Prefill dots it over 0-511 (rope
// included). Persist writes it contiguous at 512/block — decode MUST read at 512 stride.
#define COMP_BLOCK_DIM 512

// ---- Helpers ----------------------------------------------------------------

// Convert FP8 E4M3 storage type to float
__device__ __forceinline__ float fp8e4m3_to_f32(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// ============================================================================
// MLA Paged Decode Attention (FP8)
// ============================================================================

extern "C" __global__ void mla_paged_decode_fp8(
    const __nv_bfloat16* __restrict__ Q,            // [1, nq * q_dim] = [1, 32768]
    const unsigned char* __restrict__ K_cache,     // FP8 compressed KV cache (bytes)
    const unsigned char* __restrict__ V_cache,     // FP8 compressed KV cache (bytes)
    __nv_bfloat16* __restrict__ O,                  // [1, nq * q_dim] = [1, 32768]
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,                 // 64
    const unsigned int num_kv_heads,                // 1
    const unsigned int q_head_dim,                  // 512 (latent dim per head)
    const unsigned int kv_cache_dim,                // 576 (512 latent + 64 rope)
    const unsigned int block_size,
    const float inv_sqrt_d,                          // 1/sqrt(576)
    const float k_scale,                             // FP8 scale for K
    const float v_scale,                             // FP8 scale for V
    const unsigned long long cache_stride_bytes,
    const unsigned int sliding_window,               // 0 = full; else attend only the last `sliding_window` positions
    const float* __restrict__ sinks,                 // [num_q_heads] per-head attn sink (s_aux); may be NULL. FP32: checkpoint-native (reading as bf16 hard-zeroed 7 heads)
    const unsigned char* __restrict__ comp_pool,     // 4b: flat FP8 compressed-KV pool [comp_block_count][576]; may be NULL
    const unsigned int comp_block_count              // 4b: # completed compressed blocks to attend; 0 = no compressed arm (ratio-0 layers)
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;

    // KV cache dimensions for MLA
    // K and V are stored as [latent(512) | rope(64)] = 576 dims
    const unsigned int kv_latent_dim = KV_LORA_DIM;  // 512
    const unsigned int kv_rope_dim = ROPE_DIM;       // 64
    
    // Token stride in cache (576 dims per token, 1 byte per FP8 element)
    const unsigned int token_stride = num_kv_heads * kv_cache_dim;
    
    // Offset for latent portion (in elements for thread-local indexing)
    const unsigned int kv_latent_offset = lane_id * VEC_BF16;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q (BF16, flattened [nq * q_dim])
    // Each thread loads 16 elements (512 / 32 = 16)
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)q_head * q_head_dim + vec_offset_bf16);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unsigned int v = q32[i];
        q_reg[2*i]   = __bfloat162float(__ushort_as_bfloat16((unsigned short)(v & 0xFFFF)));
        q_reg[2*i+1] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(v >> 16)));
    }

    // Sliding-window (DeepSeek-V4 decode, item 4a): attend only the last
    // `sliding_window` raw positions. 0 = full. Chunk [kv_start, seq_len) across warps.
    unsigned int kv_start = 0;
    if (sliding_window > 0u && seq_len > sliding_window) kv_start = seq_len - sliding_window;
    unsigned int win_len = seq_len - kv_start;
    unsigned int chunk_size = (win_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = kv_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        const unsigned char* k_block = K_cache + (unsigned long long)physical_block * cache_stride_bytes;
        const unsigned char* v_block = V_cache + (unsigned long long)physical_block * cache_stride_bytes;

        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;

        for (; processed < aligned_count; processed += BC) {
            float k_vals[BC][VEC_BF16];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                unsigned int p = block_offset + processed + b;
                
                // Load K latent portion (512 dims) from FP8
                const unsigned char* k_latent = k_block + p * token_stride + kv_latent_offset;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    k_vals[b][i] = fp8e4m3_to_f32((__nv_fp8_storage_t)k_latent[i]) * k_scale;
                }
                
                // Load K rope portion (64 dims) into tail positions 448-511.
                // Q_rope is in q_reg[448:511] (threads 28-31). Match K_rope there
                // so dot product = dot(Q[0:447],K_latent[0:447]) + dot(Q[448:511],K_rope[0:63]).
                if (lane_id >= 28) {
                    const unsigned int rope_offset = (lane_id - 28) * VEC_BF16;
                    const unsigned char* k_rope = k_block + p * token_stride + kv_latent_dim + rope_offset;
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++) {
                        k_vals[b][i] = fp8e4m3_to_f32((__nv_fp8_storage_t)k_rope[i]) * k_scale;
                    }
                }
            }

            float scores[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                // Compute dot product: Q[512] · K[512]
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_BF16 && i + lane_id * VEC_BF16 < kv_latent_dim; i++)
                    dot += q_reg[i] * k_vals[b][i];
                
                // Reduce across warp
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                
                scores[b] = dot * inv_sqrt_d;
            }

            float v_vals[BC][VEC_BF16];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                unsigned int p = block_offset + processed + b;

                // Load V from FP8. K==V in MLA: V's tail 448-511 is the rotated
                // rope (mirror the K rope overwrite), not the latent.
                const unsigned char* v_latent = v_block + p * token_stride + kv_latent_offset;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    v_vals[b][i] = fp8e4m3_to_f32((__nv_fp8_storage_t)v_latent[i]) * v_scale;
                }
                if (lane_id >= 28) {
                    const unsigned int rope_offset = (lane_id - 28) * VEC_BF16;
                    const unsigned char* v_rope = v_block + p * token_stride + kv_latent_dim + rope_offset;
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++) {
                        v_vals[b][i] = fp8e4m3_to_f32((__nv_fp8_storage_t)v_rope[i]) * v_scale;
                    }
                }
            }

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

            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float ef = exp_factors[b];
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++)
                    o_reg[i] += ef * v_vals[b][i];
            }
        }

        // Process remaining tokens (not aligned to BC=4)
        for (; processed < batch_count; processed++) {
            unsigned int p = block_offset + processed;
            
            // Load K from FP8
            float k_tmp[VEC_BF16];
            const unsigned char* k_latent = k_block + p * token_stride + kv_latent_offset;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) {
                k_tmp[i] = fp8e4m3_to_f32((__nv_fp8_storage_t)k_latent[i]) * k_scale;
            }
            
            // Load K rope into tail positions 448-511 to match Q_rope.
            if (lane_id >= 28) {
                const unsigned int rope_offset = (lane_id - 28) * VEC_BF16;
                const unsigned char* k_rope = k_block + p * token_stride + kv_latent_dim + rope_offset;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    k_tmp[i] = fp8e4m3_to_f32((__nv_fp8_storage_t)k_rope[i]) * k_scale;
                }
            }

            // Compute dot product
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_BF16 && i + lane_id * VEC_BF16 < kv_latent_dim; i++)
                dot += q_reg[i] * k_tmp[i];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot * inv_sqrt_d;
            float m_new = fmaxf(m, score);
            float exp_old = __expf(m - m_new);
            float exp_new = __expf(score - m_new);
            l = l * exp_old + exp_new;

            // Load V from FP8. MLA passes kv as BOTH key and value (K==V), so V's
            // tail dims 448-511 are the ROTATED rope, not the latent — mirror the
            // K rope overwrite so the attention output carries the rope (then the
            // dispatch de-rotates it per eq.26).
            float v_tmp[VEC_BF16];
            const unsigned char* v_latent = v_block + p * token_stride + kv_latent_offset;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) {
                v_tmp[i] = fp8e4m3_to_f32((__nv_fp8_storage_t)v_latent[i]) * v_scale;
            }
            if (lane_id >= 28) {
                const unsigned int rope_offset = (lane_id - 28) * VEC_BF16;
                const unsigned char* v_rope = v_block + p * token_stride + kv_latent_dim + rope_offset;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) {
                    v_tmp[i] = fp8e4m3_to_f32((__nv_fp8_storage_t)v_rope[i]) * v_scale;
                }
            }

            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                o_reg[i] = o_reg[i] * exp_old + exp_new * v_tmp[i];
            m = m_new;
        }

        pos += batch_count;
    }

    // ── Compressed arm (4b): attend the flat compressed-KV pool ──
    // comp_pool: [comp_block_count][576] FP8, block b at byte offset b*576. Same
    // dtype/scale/layout (latent|rope) as the raw K, same Q, same inv_sqrt_d, and
    // MLA K==V — so it folds into the SAME online-softmax (m,l,o_reg) as the raw
    // window. Distributed across warps like the raw arm; the cross-warp reduction
    // below then merges [windowed-raw ∪ compressed] into one softmax. Distant
    // context lives ONLY here (compressed); recent tokens are double-represented
    // (raw window + compressed) — that overlap is correct (matches prefill).
    // comp_block_count == 0 (ratio-0 layers, or empty pool) → this loop is a no-op.
    if (comp_pool != nullptr && comp_block_count > 0u) {
        const unsigned int cchunk = (comp_block_count + NUM_WARPS - 1) / NUM_WARPS;
        unsigned int cstart = warp_id * cchunk;
        unsigned int cend = cstart + cchunk;
        if (cend > comp_block_count) cend = comp_block_count;
        for (unsigned int cb = cstart; cb < cend; cb++) {
            const unsigned char* c_block = comp_pool + (unsigned long long)cb * COMP_BLOCK_DIM;

            // Load K (== comp_k) as PURE LATENT over dims 0-511 — NO rope overwrite.
            // The prefill oracle (prefill_attn_compressed.cu) dots the compressed arm
            // over dims 0-511 only; comp_k's rope tail (512-575) is never touched
            // there → prefill's compressed scoring is rope-FREE. The raw arm's rope
            // overwrite is WRONG here: comp_k_rope is fixed at pos w*ratio while decode
            // Q_rope grows with position → a rope·rope term that grows with relative
            // distance, spiking the oldest blocks the moment they go compressed-only
            // (the observed maxC spike). Match prefill: latent-only compressed dot.
            float k_tmp[VEC_BF16];
            const unsigned char* k_latent = c_block + kv_latent_offset;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                k_tmp[i] = fp8e4m3_to_f32((__nv_fp8_storage_t)k_latent[i]) * k_scale;

            // score = (Q · comp_k) / sqrt(d)
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_BF16 && i + lane_id * VEC_BF16 < kv_latent_dim; i++)
                dot += q_reg[i] * k_tmp[i];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            float score = dot * inv_sqrt_d;

            float m_new = fmaxf(m, score);
            float exp_old = __expf(m - m_new);
            float exp_new = __expf(score - m_new);
            l = l * exp_old + exp_new;

            // Load V (== comp_k) as PURE LATENT over dims 0-511 — NO rope overwrite,
            // mirroring the K side and prefill's compressed-arm output accumulation
            // (prefill_attn_compressed accumulates Vc over dims 0-511 only).
            float v_tmp[VEC_BF16];
            const unsigned char* v_latent = c_block + kv_latent_offset;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                v_tmp[i] = fp8e4m3_to_f32((__nv_fp8_storage_t)v_latent[i]) * v_scale;

            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                o_reg[i] = o_reg[i] * exp_old + exp_new * v_tmp[i];
            m = m_new;
        }
    }

    // Reduce across warps
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][512];  // 512 latent dims

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        if (lane_id * VEC_BF16 + i < 512) {
            smem_o[warp_id][lane_id * VEC_BF16 + i] = o_reg[i];
        }
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
                for (int i = 0; i < 512; i++) {
                    smem_o[warp_id][i] = smem_o[warp_id][i] * scale_me + smem_o[other][i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    // Write output (BF16, flattened [nq * q_dim])
    if (warp_id == 0) {
        float final_l = smem_l[0];
        // Per-head attention sink (s_aux): an extra softmax logit per head that is
        // dropped from the numerator but kept in the denominator (reference
        // eager_attention_forward concats `sinks`, softmaxes, then slices off the
        // sink column). Online-softmax: add exp(sink - running_max) to the sum.
        if (sinks != nullptr) {
            final_l += __expf(sinks[q_head] - smem_m[0]);
        }
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)q_head * q_head_dim + vec_offset_bf16);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0 = smem_o[0][lane_id * VEC_BF16 + 2*i]     * inv_l;
            float v1 = smem_o[0][lane_id * VEC_BF16 + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}