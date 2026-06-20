// SPDX-License-Identifier: AGPL-3.0-only

// Paged Decode Attention — Turbo3 (WHT + Lloyd-Max 8-level + per-group FP8 scales).
// 3-bit packed data (8 values in 3 bytes) with 8-level Lloyd-Max codebook.
//
// Memory layout per KV cache block (K or V separately):
//   [data section: block_size * num_kv_heads * head_dim/2 bytes (packed E2M1 nibble pairs)]
//   [scale section: block_size * num_kv_heads * head_dim/GROUP_SIZE bytes (FP8 E4M3 scales)]
//
// GROUP_SIZE = 16 elements share one FP8 E4M3 scale.
// Dequant: f32_val = E2M1_LUT[nibble] * fp8_to_f32(scale_byte)
//
// Same algorithmic structure as paged_decode_attn_fp8.cu:
//   - One CTA per (q_head, seq) pair
//   - 8 warps split KV sequence
//   - Batched loading (BC=4) within physical blocks
//   - Online softmax with tree-based inter-warp reduction
//
// Key differences from FP8:
//   - Cache is byte-addressed with appended per-group FP8 scales
//   - No global k_scale/v_scale params — scales embedded per-group in cache
//   - LUT-based dequant (16-entry E2M1 table in shared memory)
//   - HDIM=256: 1 uint32 load per thread (8 nibbles = 8 elements)
//   - HDIM=128: 1 uint16 load per thread (4 nibbles = 4 elements)
//
// Grid: (num_q_heads, num_seqs, 1)  [splitk: (num_q_heads, num_splits, num_seqs)]
// Block: (256, 1, 1)

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

__device__ __forceinline__ void unpack2_bf16(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ float fp8e4m3_to_f32(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// Load packed NVFP4 data + FP8 group scale → dequantized floats.
// HDIM=256: data_ptr has 4 bytes (uint32, 8 nibbles), out[0..7]
// HDIM=128: data_ptr has 2 bytes (uint16, 4 nibbles), out[0..3]
// Turbo3 dequant: unpack 3-bit values from packed bytes + scale.
// 8 values packed in 3 bytes: b0 = v0|(v1<<3)|(v2<<6), b1 = (v2>>2)|(v3<<1)|(v4<<4)|(v5<<7), ...
__device__ __forceinline__ void nvfp4_dequant(
    const unsigned char* data_ptr,
    const unsigned char* scale_ptr,
    const __half* lut,
    float* out
) {
    float gs = fp8e4m3_to_f32((__nv_fp8_storage_t)*scale_ptr);
#if VEC_BF16 == 8
    // Unpack 8 × 3-bit from 3 bytes
    unsigned char b0 = data_ptr[0], b1 = data_ptr[1], b2 = data_ptr[2];
    out[0] = __half2float(lut[(b0) & 0x7]) * gs;
    out[1] = __half2float(lut[(b0 >> 3) & 0x7]) * gs;
    out[2] = __half2float(lut[((b0 >> 6) | (b1 << 2)) & 0x7]) * gs;
    out[3] = __half2float(lut[(b1 >> 1) & 0x7]) * gs;
    out[4] = __half2float(lut[(b1 >> 4) & 0x7]) * gs;
    out[5] = __half2float(lut[((b1 >> 7) | (b2 << 1)) & 0x7]) * gs;
    out[6] = __half2float(lut[(b2 >> 2) & 0x7]) * gs;
    out[7] = __half2float(lut[(b2 >> 5) & 0x7]) * gs;
#elif VEC_BF16 == 4
    // Unpack 4 × 3-bit from shared 3-byte group.
    // Even threads (lane%2==0): values 0-3, Odd threads: values 4-7
    unsigned char b0 = data_ptr[0], b1 = data_ptr[1], b2 = data_ptr[2];
    // Use __shfl to get lane parity without divergent reads
    // Both threads in a pair read the same 3 bytes (overlapping access is fine)
    if ((threadIdx.x % 2) == 0) {
        // Values 0-3
        out[0] = __half2float(lut[(b0) & 0x7]) * gs;
        out[1] = __half2float(lut[(b0 >> 3) & 0x7]) * gs;
        out[2] = __half2float(lut[((b0 >> 6) | (b1 << 2)) & 0x7]) * gs;
        out[3] = __half2float(lut[(b1 >> 1) & 0x7]) * gs;
    } else {
        // Values 4-7
        out[0] = __half2float(lut[(b1 >> 4) & 0x7]) * gs;
        out[1] = __half2float(lut[((b1 >> 7) | (b2 << 1)) & 0x7]) * gs;
        out[2] = __half2float(lut[(b2 >> 2) & 0x7]) * gs;
        out[3] = __half2float(lut[(b2 >> 5) & 0x7]) * gs;
    }
#else
    #error "Unsupported VEC_BF16 (need 4 or 8)"
#endif
}

// ============================================================================
// Basic NVFP4 paged decode attention
// ============================================================================

extern "C" __global__ void paged_decode_attn_turbo3(
    const __nv_bfloat16* __restrict__ Q,          // [num_seqs, num_q_heads, head_dim] BF16
    const unsigned char* __restrict__ K_cache,     // [num_blocks, block_stride_bytes] raw bytes
    const unsigned char* __restrict__ V_cache,     // [num_blocks, block_stride_bytes] raw bytes
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
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Turbo4 Lloyd-Max 16-level codebook in shared memory
    // Turbo3 Lloyd-Max 8-level codebook
    __shared__ __half e2m1_lut[8];
    if (tid < 8) {
        const float lut_init[8] = {
            -2.1520f, -1.3440f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f
        };
        e2m1_lut[tid] = __float2half(lut_init[tid]);
    }
    __syncthreads();

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;

    // NVFP4 cache addressing constants
    const unsigned int head_data_bytes = head_dim * 3 / 8;  // Turbo3: 3 bits per element
    const unsigned int head_scale_bytes = head_dim / NVFP4_GROUP_SIZE;
    const unsigned int token_data_stride = num_kv_heads * head_data_bytes;
    const unsigned int token_scale_stride = num_kv_heads * head_scale_bytes;
#if VEC_BF16 == 8
    const unsigned int kv_data_offset = kv_head * head_data_bytes + (lane_id * 3);
    const unsigned int kv_scale_offset = kv_head * head_scale_bytes + (lane_id / 2);
#else
    const unsigned int kv_data_offset = kv_head * head_data_bytes + ((lane_id / 2) * 3);
    const unsigned int kv_scale_offset = kv_head * head_scale_bytes + (lane_id / 4);
#endif

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q (BF16, strided)
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset_bf16);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_bf16(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    unsigned int chunk_size = (seq_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = warp_id * chunk_size;
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
        const unsigned char* k_block = K_cache + (unsigned long long)physical_block * block_stride_bytes;
        const unsigned char* v_block = V_cache + (unsigned long long)physical_block * block_stride_bytes;

        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;

        // Batched path: BC=4 positions at a time
        for (; processed < aligned_count; processed += BC) {
            // Load BC K vectors (NVFP4 data + FP8 group scale)
            float k_vals[BC][VEC_BF16];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                unsigned int p = block_offset + processed + b;
                const unsigned char* kd = k_block + p * token_data_stride + kv_data_offset;
                const unsigned char* ks = k_block + data_section_bytes + p * token_scale_stride + kv_scale_offset;
                nvfp4_dequant(kd, ks, e2m1_lut, k_vals[b]);
            }

            // Compute BC dot products
            float scores[BC];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++)
                    dot += q_reg[i] * k_vals[b][i];
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

            // Sparse V load — gate per-row on exp_factor magnitude.
            float v_vals[BC][VEC_BF16];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                if (exp_factors[b] > TQ_PLUS_SPARSE_V_THRESHOLD) {
                    unsigned int p = block_offset + processed + b;
                    const unsigned char* vd = v_block + p * token_data_stride + kv_data_offset;
                    const unsigned char* vs = v_block + data_section_bytes + p * token_scale_stride + kv_scale_offset;
                    nvfp4_dequant(vd, vs, e2m1_lut, v_vals[b]);
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

        // Remainder: single positions
        for (; processed < batch_count; processed++) {
            unsigned int p = block_offset + processed;
            const unsigned char* kd = k_block + p * token_data_stride + kv_data_offset;
            const unsigned char* ks = k_block + data_section_bytes + p * token_scale_stride + kv_scale_offset;
            float k_tmp[VEC_BF16];
            nvfp4_dequant(kd, ks, e2m1_lut, k_tmp);

            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                dot += q_reg[i] * k_tmp[i];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot * inv_sqrt_d;
            float m_new = fmaxf(m, score);
            float exp_old = __expf(m - m_new);
            float exp_new = __expf(score - m_new);
            l = l * exp_old + exp_new;

            float v_tmp[VEC_BF16] = {0};
            if (exp_new > TQ_PLUS_SPARSE_V_THRESHOLD) {
                const unsigned char* vd = v_block + p * token_data_stride + kv_data_offset;
                const unsigned char* vs = v_block + data_section_bytes + p * token_scale_stride + kv_scale_offset;
                nvfp4_dequant(vd, vs, e2m1_lut, v_tmp);
            }

            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                o_reg[i] = o_reg[i] * exp_old + exp_new * v_tmp[i];
            m = m_new;
        }

        pos += batch_count;
    }

    // Tree-based inter-warp reduction (identical to FP8/BF16)
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

// ============================================================================
// Split-K NVFP4 variant for long sequences with few heads
// Grid: (num_q_heads, num_splits, num_seqs)
// ============================================================================

extern "C" __global__ void paged_decode_attn_splitk_nvfp4(
    const __nv_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,
    const unsigned char* __restrict__ V_cache,
    float* __restrict__ workspace,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes
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

    // Turbo4 Lloyd-Max codebook
    __shared__ __half e2m1_lut[16];
    if (tid < 16) {
        const float lut_init[16] = {
            -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
             0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
        };
        e2m1_lut[tid] = __float2half(lut_init[tid]);
    }
    __syncthreads();

    unsigned int split_size = (seq_len + num_splits - 1) / num_splits;
    unsigned int kv_start = split_id * split_size;
    unsigned int kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_start >= seq_len) kv_start = kv_end;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;

    const unsigned int head_data_bytes = head_dim * 3 / 8;  // Turbo3: 3 bits per element
    const unsigned int head_scale_bytes = head_dim / NVFP4_GROUP_SIZE;
    const unsigned int token_data_stride = num_kv_heads * head_data_bytes;
    const unsigned int token_scale_stride = num_kv_heads * head_scale_bytes;
#if VEC_BF16 == 8
    const unsigned int kv_data_offset = kv_head * head_data_bytes + (lane_id * 3);
    const unsigned int kv_scale_offset = kv_head * head_scale_bytes + (lane_id / 2);
#else
    const unsigned int kv_data_offset = kv_head * head_data_bytes + ((lane_id / 2) * 3);
    const unsigned int kv_scale_offset = kv_head * head_scale_bytes + (lane_id / 4);
#endif

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Load Q (BF16, strided)
    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset_bf16);
    float q_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        unpack2_bf16(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    unsigned int local_len = kv_end - kv_start;
    unsigned int chunk_size = (local_len + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = kv_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;

    float m_val = -1e30f;
    float l_val = 0.0f;
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    for (unsigned int pos = my_start; pos < my_end; pos++) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const unsigned char* k_block = K_cache + (unsigned long long)physical_block * block_stride_bytes;
        const unsigned char* v_block = V_cache + (unsigned long long)physical_block * block_stride_bytes;

        const unsigned char* kd = k_block + block_offset * token_data_stride + kv_data_offset;
        const unsigned char* ks = k_block + data_section_bytes + block_offset * token_scale_stride + kv_scale_offset;
        float k_tmp[VEC_BF16];
        nvfp4_dequant(kd, ks, e2m1_lut, k_tmp);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            dot += q_reg[i] * k_tmp[i];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, offset);

        float score = dot * inv_sqrt_d;
        float m_new = fmaxf(m_val, score);
        float exp_old = __expf(m_val - m_new);
        float exp_new = __expf(score - m_new);
        l_val = l_val * exp_old + exp_new;

        const unsigned char* vd = v_block + block_offset * token_data_stride + kv_data_offset;
        const unsigned char* vs = v_block + data_section_bytes + block_offset * token_scale_stride + kv_scale_offset;
        float v_tmp[VEC_BF16];
        nvfp4_dequant(vd, vs, e2m1_lut, v_tmp);

        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            o_reg[i] = o_reg[i] * exp_old + exp_new * v_tmp[i];
        m_val = m_new;
    }

    // Tree merge within CTA
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m_val;
        smem_l[warp_id] = l_val;
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

    // Write partial to workspace (F32)
    unsigned int ws_stride = (head_dim + 2);
    float* ws_base = workspace + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride
                   + split_id * ws_stride;

    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) {
            ws_base[vec_offset_bf16 + i] = smem_o[0][vec_offset_bf16 + i];
        }
        if (lane_id == 0) {
            ws_base[head_dim] = smem_m[0];
            ws_base[head_dim + 1] = smem_l[0];
        }
    }
}

// ============================================================================
// Reduction kernel: merge split-K partials → final BF16 output
// Grid: (num_q_heads, num_seqs, 1)  Block: (32, 1, 1)
// ============================================================================

extern "C" __global__ void paged_decode_attn_reduce_nvfp4(
    const float* __restrict__ workspace,    // [num_seqs, num_q_heads, num_splits, (head_dim+2)] F32
    __nv_bfloat16* __restrict__ O,          // [num_seqs, num_q_heads, head_dim] BF16
    const int* __restrict__ seq_lens,
    const unsigned int num_q_heads,
    const unsigned int head_dim,
    const unsigned int num_splits
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int lane_id = threadIdx.x;  // 0..31

    if (q_head >= num_q_heads) return;
    if (seq_lens[seq_idx] == 0) return;

    const unsigned int vec_off = lane_id * VEC_BF16;
    const unsigned int ws_stride = head_dim + 2;
    const float* ws_base = workspace
        + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride;

    // Load split 0
    float m = ws_base[head_dim];
    float l = ws_base[head_dim + 1];
    float o_reg[VEC_BF16];
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++)
        o_reg[i] = ws_base[vec_off + i];

    // Merge splits 1..num_splits-1
    for (unsigned int s = 1; s < num_splits; s++) {
        const float* ws = ws_base + s * ws_stride;
        float ms = ws[head_dim];
        float ls = ws[head_dim + 1];

        if (ls <= 0.0f) continue;

        float m_new = fmaxf(m, ms);
        float scale_me = __expf(m - m_new);
        float scale_s = __expf(ms - m_new);

        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            o_reg[i] = o_reg[i] * scale_me + ws[vec_off + i] * scale_s;

        l = l * scale_me + ls * scale_s;
        m = m_new;
    }

    // Normalize and write BF16 output
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                          + (unsigned long long)q_head * head_dim + vec_off);
    #pragma unroll
    for (int i = 0; i < VEC_U32; i++) {
        float v0 = o_reg[2*i] * inv_l;
        float v1 = o_reg[2*i + 1] * inv_l;
        unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
        unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
        o32[i] = lo | (hi << 16);
    }
}
