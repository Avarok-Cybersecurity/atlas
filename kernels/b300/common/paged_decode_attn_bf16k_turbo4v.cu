// SPDX-License-Identifier: AGPL-3.0-only

// Paged Decode Attention — Asymmetric K=bf16, V=turbo4 (HDIM=256).
//
// TurboQuant+ safer-asym variant: K kept at full BF16 precision (preserves
// attention-score fidelity), V compressed to 4-bit Lloyd-Max + per-group FP8
// scale (saves the V-side bandwidth which dominates KV traffic at long
// context). One CTA per (q_head, seq) pair, 8 warps, BC=4 batched positions,
// sparse V dequant on both batched and remainder paths.
//
// Cache pools have DIFFERENT byte layouts and therefore DIFFERENT block strides:
//   K pool: [num_blocks, block_size, num_kv_heads, head_dim]  BF16 contiguous
//   V pool: [num_blocks, block_size_v_layout]
//           where each block holds:
//             [data: block_size * num_kv_heads * head_dim / 2 bytes  (4-bit nibbles)]
//             [scales: block_size * num_kv_heads * head_dim/16 bytes (FP8)]
//
// Grid: (num_q_heads, num_seqs, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>

// Sparse V dequant threshold — skip V load + dequant when attn weight is tiny.
#ifndef TQ_PLUS_SPARSE_V_THRESHOLD
#define TQ_PLUS_SPARSE_V_THRESHOLD 1e-3f
#endif

#include <cuda_fp8.h>

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 256
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32  (HDIM / (WARP_SIZE * 2))
#define NUM_WARPS 8
#define BC 4
#define NVFP4_GROUP_SIZE 16

// ---- Helpers ----------------------------------------------------------------

__device__ __forceinline__ void unpack2_bf16_asym(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ float fp8e4m3_to_f32_asym(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// Turbo4 V dequant — 4-bit nibbles, 2 values per byte.
// HDIM=256: each thread reads VEC_BF16=4 values from 2 bytes (uint16).
// HDIM=256: each thread reads VEC_BF16=8 values from 4 bytes (uint32).
__device__ __forceinline__ void turbo4_dequant_v(
    const unsigned char* data_ptr,
    const unsigned char* scale_ptr,
    const __half* lut,
    float* out
) {
    float gs = fp8e4m3_to_f32_asym((__nv_fp8_storage_t)*scale_ptr);
#if VEC_BF16 == 8
    unsigned int pk = *(const unsigned int*)data_ptr;
    out[0] = __half2float(lut[(pk)       & 0xF]) * gs;
    out[1] = __half2float(lut[(pk >> 4)  & 0xF]) * gs;
    out[2] = __half2float(lut[(pk >> 8)  & 0xF]) * gs;
    out[3] = __half2float(lut[(pk >> 12) & 0xF]) * gs;
    out[4] = __half2float(lut[(pk >> 16) & 0xF]) * gs;
    out[5] = __half2float(lut[(pk >> 20) & 0xF]) * gs;
    out[6] = __half2float(lut[(pk >> 24) & 0xF]) * gs;
    out[7] = __half2float(lut[pk >> 28])         * gs;
#elif VEC_BF16 == 4
    unsigned short pk = *(const unsigned short*)data_ptr;
    out[0] = __half2float(lut[(pk)       & 0xF]) * gs;
    out[1] = __half2float(lut[(pk >> 4)  & 0xF]) * gs;
    out[2] = __half2float(lut[(pk >> 8)  & 0xF]) * gs;
    out[3] = __half2float(lut[pk >> 12])         * gs;
#else
    #error "Unsupported VEC_BF16 (need 4 or 8)"
#endif
}

// ============================================================================
// Asymmetric Bf16K + Turbo4V paged decode attention
// ============================================================================
extern "C" __global__ void paged_decode_attn_bf16k_turbo4v(
    const __nv_bfloat16* __restrict__ Q,          // [num_seqs, num_q_heads, head_dim] BF16
    const __nv_bfloat16* __restrict__ K_cache,    // K pool: NHD layout BF16
    const unsigned char* __restrict__ V_cache,    // V pool: byte-addressed turbo4
    __nv_bfloat16* __restrict__ O,                // [num_seqs, num_q_heads, head_dim] BF16
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned long long v_block_stride_bytes,    // V-pool block stride
    const unsigned long long v_data_section_bytes,    // V-pool data section size
    const unsigned int sliding_window
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // 16-level Lloyd-Max codebook (turbo4) in shared memory as fp16 LUT.
    __shared__ __half e2m1_lut[16];
    if (tid < 16) {
        const float lut_init[16] = {
            -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
             0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
        };
        e2m1_lut[tid] = __float2half(lut_init[tid]);
    }
    __syncthreads();

    // Sliding-window start (Gemma-4 hybrid attn). 0 = full attention.
    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;

    // V-pool addressing (turbo4 layout: 4-bit nibbles, 2 values per byte)
    const unsigned int v_head_data_bytes  = head_dim / 2;
    const unsigned int v_head_scale_bytes = head_dim / NVFP4_GROUP_SIZE;
    const unsigned int v_token_data_stride  = num_kv_heads * v_head_data_bytes;
    const unsigned int v_token_scale_stride = num_kv_heads * v_head_scale_bytes;
    // Each thread covers VEC_BF16=4 values = 2 bytes (HDIM=256) or 8 values = 4 bytes (HDIM=256).
    // Direct per-lane stride: lane_id * (VEC_BF16/2) bytes.
    const unsigned int v_data_offset_per_thread  = kv_head * v_head_data_bytes  + lane_id * (VEC_BF16 / 2);
    const unsigned int v_scale_offset_per_thread = kv_head * v_head_scale_bytes + (lane_id * VEC_BF16 / NVFP4_GROUP_SIZE);

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q (BF16, strided)
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset_bf16);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_bf16_asym(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    // Split attended range [window_start, seq_len) across NUM_WARPS warps.
    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
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

        // K-side pointer base (BF16, NHD layout, head_dim per-thread vec offset).
        const unsigned long long k_page_stride =
            (unsigned long long)block_size * num_kv_heads * head_dim;
        const unsigned long long k_head_stride_kv =
            (unsigned long long)num_kv_heads * head_dim;
        const __nv_bfloat16* k_block_base = K_cache
            + (unsigned long long)physical_block * k_page_stride
            + (unsigned long long)block_offset * k_head_stride_kv
            + (unsigned long long)kv_head * head_dim;

        // V-side pointer base (byte-addressed turbo4).
        const unsigned char* v_block = V_cache
            + (unsigned long long)physical_block * v_block_stride_bytes;

        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;

        // ── Batched path: BC=4 positions ──
        for (; processed < aligned_count; processed += BC) {
            // Load BC K vectors (BF16 contiguous read, vec_offset_bf16 stride).
            unsigned int k_packed[BC][VEC_U32];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* k32 = (const unsigned int*)(k_block_base
                    + (unsigned long long)(processed + b) * k_head_stride_kv + vec_offset_bf16);
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
                    unpack2_bf16_asym(k_packed[b][i], k0, k1);
                    dot += q_reg[2*i] * k0 + q_reg[2*i+1] * k1;
                }
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                scores[b] = dot * inv_sqrt_d;
            }

            // Batched softmax update
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

            // ── Sparse V dequant on batched path ──
            // Skip V load + dequant when attn weight is below threshold.
            float v_vals[BC][VEC_BF16];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                if (exp_factors[b] > TQ_PLUS_SPARSE_V_THRESHOLD) {
                    unsigned int p = block_offset + processed + b;
                    const unsigned char* vd = v_block + p * v_token_data_stride  + v_data_offset_per_thread;
                    const unsigned char* vs = v_block + v_data_section_bytes
                                            + p * v_token_scale_stride + v_scale_offset_per_thread;
                    turbo4_dequant_v(vd, vs, e2m1_lut, v_vals[b]);
                } else {
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++) v_vals[b][i] = 0.0f;
                }
            }

            // V accumulate
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float ef = exp_factors[b];
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++)
                    o_reg[i] += ef * v_vals[b][i];
            }
        }

        // ── Remainder: single positions ──
        for (; processed < batch_count; processed++) {
            unsigned int p = block_offset + processed;

            // K (bf16 vector load)
            const unsigned int* k32 = (const unsigned int*)(k_block_base
                + (unsigned long long)processed * k_head_stride_kv + vec_offset_bf16);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float k0, k1;
                unpack2_bf16_asym(k32[i], k0, k1);
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

            // Sparse V (turbo4 dequant only when exp_new > threshold)
            float v_tmp[VEC_BF16] = {0};
            if (exp_new > TQ_PLUS_SPARSE_V_THRESHOLD) {
                const unsigned char* vd = v_block + p * v_token_data_stride  + v_data_offset_per_thread;
                const unsigned char* vs = v_block + v_data_section_bytes
                                        + p * v_token_scale_stride + v_scale_offset_per_thread;
                turbo4_dequant_v(vd, vs, e2m1_lut, v_tmp);
            }

            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                o_reg[i] = o_reg[i] * exp_old + exp_new * v_tmp[i];
            m = m_new;
        }

        pos += batch_count;
    }

    // ── Tree-based inter-warp reduction (identical to turbo3/bf16) ──
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) {
        smem_o[warp_id][vec_offset_bf16 + i] = o_reg[i];
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
                    smem_o[warp_id][vec_offset_bf16 + i] =
                        smem_o[warp_id][vec_offset_bf16 + i] * scale_me +
                        smem_o[other][vec_offset_bf16 + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim + vec_offset_bf16);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0 = smem_o[0][vec_offset_bf16 + 2*i]     * inv_l;
            float v1 = smem_o[0][vec_offset_bf16 + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}
