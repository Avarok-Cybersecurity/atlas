// SPDX-License-Identifier: AGPL-3.0-only

// MLA Paged Decode — NVFP4 variant for DeepSeek-V4-Flash.
//
// DeepSeek-V4-Flash uses compressed KV cache with MLA (Multi-head Latent Attention):
// - KV cache: 576 dims per token (512 latent + 64 rope), stored in NVFP4 format
// - Q: 32768 dims (64 heads × 512 dims per head), BF16
// - Output: 32768 dims (64 heads × 512 latent dims), BF16
//
// Memory layout per KV cache block (K or V separately):
//   [data section: block_size * num_kv_heads * (kv_lora+rope)/2 bytes (packed E2M1 nibble pairs)]
//   [scale section: block_size * num_kv_heads * (kv_lora+rope)/GROUP_SIZE bytes (FP8 E4M3 scales)]
//
// GROUP_SIZE = 16 elements share one FP8 E4M3 scale.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#define VEC_BF16 16  // 512 / 32 = 16 elements per thread for Q
#define VEC_U32  8   // 512 / (32 * 2) = 8 uint32 per thread for Q
#define NUM_WARPS 8
#define BC 4
#define NVFP4_GROUP_SIZE 16
#define KV_LORA_DIM 512
#define ROPE_DIM 64
#define MLA_CACHE_DIM 576  // KV_LORA_DIM + ROPE_DIM

// ---- Helpers ----------------------------------------------------------------

__device__ __forceinline__ float fp8e4m3_to_f32(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

__device__ __forceinline__ void nvfp4_dequant_mla(
    const unsigned char* data_ptr,
    const unsigned char* scale_ptr,
    const float* lut,
    float* out,
    unsigned int dims  // 512 for latent, 64 for rope
) {
    float gs = fp8e4m3_to_f32((__nv_fp8_storage_t)*scale_ptr);
    
    // Handle 512-dim latent portion (16 elements per thread)
    if (dims == KV_LORA_DIM) {
        unsigned long long pk64 = *(const unsigned long long*)data_ptr;
        out[0]  = lut[(pk64)       & 0xF] * gs;
        out[1]  = lut[(pk64 >> 4)  & 0xF] * gs;
        out[2]  = lut[(pk64 >> 8)  & 0xF] * gs;
        out[3]  = lut[(pk64 >> 12) & 0xF] * gs;
        out[4]  = lut[(pk64 >> 16) & 0xF] * gs;
        out[5]  = lut[(pk64 >> 20) & 0xF] * gs;
        out[6]  = lut[(pk64 >> 24) & 0xF] * gs;
        out[7]  = lut[(pk64 >> 28) & 0xF] * gs;
        out[8]  = lut[(pk64 >> 32) & 0xF] * gs;
        out[9]  = lut[(pk64 >> 36) & 0xF] * gs;
        out[10] = lut[(pk64 >> 40) & 0xF] * gs;
        out[11] = lut[(pk64 >> 44) & 0xF] * gs;
        out[12] = lut[(pk64 >> 48) & 0xF] * gs;
        out[13] = lut[(pk64 >> 52) & 0xF] * gs;
        out[14] = lut[(pk64 >> 56) & 0xF] * gs;
        out[15] = lut[pk64 >> 60]         * gs;
    } 
    // Handle 64-dim rope portion (4 elements per thread for first 16 threads)
    else if (dims == ROPE_DIM) {
        unsigned int pk = *(const unsigned int*)data_ptr;
        out[0] = lut[(pk)       & 0xF] * gs;
        out[1] = lut[(pk >> 4)  & 0xF] * gs;
        out[2] = lut[(pk >> 8)  & 0xF] * gs;
        out[3] = lut[pk >> 12]         * gs;
    }
}

// ============================================================================
// MLA Paged Decode Attention
// ============================================================================

extern "C" __global__ void mla_paged_decode_nvfp4(
    const __nv_bfloat16* __restrict__ Q,            // [1, nq * q_dim] = [1, 32768]
    const unsigned char* __restrict__ K_cache,      // NVFP4 compressed KV cache
    const unsigned char* __restrict__ V_cache,      // NVFP4 compressed KV cache
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

    // E2M1 dequant LUT in shared memory
    __shared__ float e2m1_lut[16];
    if (tid < 16) {
        const float lut_init[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
        };
        e2m1_lut[tid] = lut_init[tid];
    }
    __syncthreads();

    const unsigned int vec_offset_bf16 = lane_id * VEC_BF16;

    // KV cache dimensions for MLA
    // K and V are stored as [latent(512) | rope(64)] = 576 dims
    const unsigned int kv_latent_dim = KV_LORA_DIM;  // 512
    const unsigned int kv_rope_dim = ROPE_DIM;       // 64
    
    // Byte offsets for NVFP4 format
    const unsigned int latent_data_bytes = kv_latent_dim / 2;
    const unsigned int latent_scale_bytes = kv_latent_dim / NVFP4_GROUP_SIZE;
    const unsigned int rope_data_bytes = kv_rope_dim / 2;
    const unsigned int rope_scale_bytes = kv_rope_dim / NVFP4_GROUP_SIZE;
    
    const unsigned int token_data_stride = num_kv_heads * (latent_data_bytes + rope_data_bytes);
    const unsigned int token_scale_stride = num_kv_heads * (latent_scale_bytes + rope_scale_bytes);
    
    // Offsets for latent and rope portions
    const unsigned int kv_latent_data_offset = lane_id * (VEC_BF16 / 2);
    const unsigned int kv_latent_scale_offset = lane_id * VEC_BF16 / NVFP4_GROUP_SIZE;
    const unsigned int kv_rope_data_offset = lane_id * (4 / 2);  // 4 bytes for rope
    const unsigned int kv_rope_scale_offset = lane_id * 4 / NVFP4_GROUP_SIZE;

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

        for (; processed < aligned_count; processed += BC) {
            float k_vals[BC][VEC_BF16];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                unsigned int p = block_offset + processed + b;
                
                // Dequantize latent portion (512 dims)
                const unsigned char* kd_latent = k_block + p * token_data_stride + kv_latent_data_offset;
                const unsigned char* ks_latent = k_block + data_section_bytes + p * token_scale_stride + kv_latent_scale_offset;
                nvfp4_dequant_mla(kd_latent, ks_latent, e2m1_lut, k_vals[b], kv_latent_dim);
                
                // Dequantize rope portion (64 dims) - only first 16 threads participate
                if (lane_id < 16) {
                    const unsigned char* kd_rope = k_block + p * token_data_stride + latent_data_bytes + kv_rope_data_offset;
                    const unsigned char* ks_rope = k_block + data_section_bytes + p * token_scale_stride + latent_scale_bytes + kv_rope_scale_offset;
                    nvfp4_dequant_mla(kd_rope, ks_rope, e2m1_lut, k_vals[b], kv_rope_dim);
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
                
                // Dequantize V (only latent portion, no rope)
                const unsigned char* vd_latent = v_block + p * token_data_stride + kv_latent_data_offset;
                const unsigned char* vs_latent = v_block + data_section_bytes + p * token_scale_stride + kv_latent_scale_offset;
                nvfp4_dequant_mla(vd_latent, vs_latent, e2m1_lut, v_vals[b], kv_latent_dim);
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
            
            // Dequantize K
            float k_tmp[VEC_BF16];
            const unsigned char* kd_latent = k_block + p * token_data_stride + kv_latent_data_offset;
            const unsigned char* ks_latent = k_block + data_section_bytes + p * token_scale_stride + kv_latent_scale_offset;
            nvfp4_dequant_mla(kd_latent, ks_latent, e2m1_lut, k_tmp, kv_latent_dim);
            
            if (lane_id < 16) {
                const unsigned char* kd_rope = k_block + p * token_data_stride + latent_data_bytes + kv_rope_data_offset;
                const unsigned char* ks_rope = k_block + data_section_bytes + p * token_scale_stride + latent_scale_bytes + kv_rope_scale_offset;
                nvfp4_dequant_mla(kd_rope, ks_rope, e2m1_lut, k_tmp, kv_rope_dim);
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

            // Dequantize V
            float v_tmp[VEC_BF16];
            const unsigned char* vd_latent = v_block + p * token_data_stride + kv_latent_data_offset;
            const unsigned char* vs_latent = v_block + data_section_bytes + p * token_scale_stride + kv_latent_scale_offset;
            nvfp4_dequant_mla(vd_latent, vs_latent, e2m1_lut, v_tmp, kv_latent_dim);

            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                o_reg[i] = o_reg[i] * exp_old + exp_new * v_tmp[i];
            m = m_new;
        }

        pos += batch_count;
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