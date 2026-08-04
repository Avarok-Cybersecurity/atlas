// SPDX-License-Identifier: AGPL-3.0-only

// MLA Cache Assembly (FP8) — DeepSeek-V4-Flash prefill path.
//
// Assembles compressed MLA cache entries for N tokens from BF16 K/V and writes FP8.
// K_cache = [kv_latent(kv_lora) | k_rope(rope)] per token
// V_cache = [kv_latent(kv_lora) | zeros(rope)] per token
//
// Input: K and V in BF16 format (already RoPE'd)
// Output: Compressed cache in FP8-E4M3 format

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 256


// Convert BF16 to FP8 E4M3
__device__ __forceinline__ unsigned char bf16_to_fp8(__nv_bfloat16 b) {
    return __nv_cvt_float_to_fp8(__bfloat162float(b), __NV_SATFINITE, __NV_E4M3);
}

// Batched MLA cache assembly + FP8 quantization for N tokens.
// Grid: (num_tokens, 1, 1)  Block: (BLOCK_SIZE, 1, 1)
//
// k_bf16: [N, nkv * (kv_lora + rope)] - K values in BF16 (already RoPE'd)
// v_bf16: [N, nkv * kv_lora] - V values in BF16 (no rope padding)
// k_cache_fp8: [N, nkv * (kv_lora + rope)] - FP8 encoded K cache
// v_cache_fp8: [N, nkv * (kv_lora + rope)] - FP8 encoded V cache
//
// For DeepSeek-V4-Flash: nkv=1, kv_lora=512, rope=64
extern "C" __global__ void mla_cache_assemble_fp8_batched(
    const __nv_bfloat16* __restrict__ k_bf16,
    const __nv_bfloat16* __restrict__ v_bf16,
    unsigned char* __restrict__ k_cache_fp8,
    unsigned char* __restrict__ v_cache_fp8,
    unsigned int num_tokens,
    unsigned int nkv,
    unsigned int kv_lora,
    unsigned int rope,
    unsigned int mla_cache_dim,  // kv_lora + rope
    float k_scale,
    float v_scale
) {
    unsigned int t = blockIdx.x;  // token index
    unsigned int idx = threadIdx.x;

    const unsigned long long k_bf16_offset = (unsigned long long)t * nkv * mla_cache_dim;
    const unsigned long long v_bf16_offset = (unsigned long long)t * nkv * kv_lora;
    const unsigned long long k_cache_offset = (unsigned long long)t * nkv * mla_cache_dim;
    const unsigned long long v_cache_offset = (unsigned long long)t * nkv * mla_cache_dim;

    // Each thread handles one dimension of the cache
    for (unsigned int d = idx; d < mla_cache_dim; d += BLOCK_SIZE) {
        if (d < kv_lora) {
            // Latent portion: copy from K and V
            for (unsigned int head = 0; head < nkv; head++) {
                unsigned long long k_idx = k_bf16_offset + head * mla_cache_dim + d;
                unsigned long long v_idx = v_bf16_offset + head * kv_lora + d;
                unsigned long long k_cache_idx = k_cache_offset + head * mla_cache_dim + d;
                unsigned long long v_cache_idx = v_cache_offset + head * mla_cache_dim + d;

                // Load BF16, scale, convert to FP8
                float k_val = __bfloat162float(k_bf16[k_idx]);
                float v_val = __bfloat162float(v_bf16[v_idx]);

                k_cache_fp8[k_cache_idx] = __nv_cvt_float_to_fp8(k_val * k_scale, __NV_SATFINITE, __NV_E4M3);
                v_cache_fp8[v_cache_idx] = __nv_cvt_float_to_fp8(v_val * v_scale, __NV_SATFINITE, __NV_E4M3);
            }
        } else if (d < kv_lora + rope) {
            // RoPE portion. DeepSeek-V4 MLA: V == K (the kv latent is passed as
            // BOTH key and value), so V carries the SAME rotated rope in its tail
            // as K. The decode kernel reconstructs V's rope tail from this cache;
            // writing zeros here (the old behaviour) made decode's V differ from
            // the prefill inline V (which uses k_out with rope), so decode
            // attention diverged from prefill and generation derailed.
            for (unsigned int head = 0; head < nkv; head++) {
                unsigned long long k_idx = k_bf16_offset + head * mla_cache_dim + d;
                unsigned long long k_cache_idx = k_cache_offset + head * mla_cache_dim + d;
                unsigned long long v_cache_idx = v_cache_offset + head * mla_cache_dim + d;

                // Load K RoPE value, quantize to FP8 for both K and V caches (V==K).
                float k_val = __bfloat162float(k_bf16[k_idx]);
                k_cache_fp8[k_cache_idx] = __nv_cvt_float_to_fp8(k_val * k_scale, __NV_SATFINITE, __NV_E4M3);
                v_cache_fp8[v_cache_idx] = __nv_cvt_float_to_fp8(k_val * v_scale, __NV_SATFINITE, __NV_E4M3);
            }
        }
    }
}