// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3-Flash — NoPE MLA latent cache write, FP8.
//
// RMSNorm(kv_a_proj output) -> FP8 -> paged slot. That is the whole decode-side KV write
// for a NoPE MLA layer: no rope, no per-head expansion, one latent row per token.
//
// 🪤 Why not common/fused_k_norm_rope_cache_write_fp8 — it is wrong for GLM TWICE:
//
//   1. It applies `x * rms * (1.0f + w)`, the Gemma **+1 offset** RMSNorm convention.
//      GLM's kv_a_layernorm is a plain `x * rms * w`. The +1 is invisible in the shapes
//      and shifts every cached latent.
//   2. It carries a rope arm keyed on `rotary_dim`, and expands per KV head. GLM is NoPE
//      with a single latent head.
//
// Addressing matches glm5next_dsa_mla_decode_fp8 exactly: the reader computes
// `physical_block * (block_size * kv_lora_dim) + p * kv_lora_dim`, which is
// `slot * kv_lora_dim` for `slot = physical_block * block_size + p`. The slot mapping
// therefore carries absolute slots and this kernel never sees a block table.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ float warp_reduce_sum_glm(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

extern "C" __global__ void glm5next_mla_latent_write_fp8(
    const __nv_bfloat16* __restrict__ kv_a,       // [num_tokens, kv_lora_dim] pre-norm
    const __nv_bfloat16* __restrict__ norm_w,     // [kv_lora_dim] kv_a_layernorm.weight
    __nv_fp8_storage_t* __restrict__ cache,       // paged FP8 latent cache
    const long long* __restrict__ slot_mapping,   // [num_tokens], -1 = skip this token
    const unsigned int kv_lora_dim,
    const float rms_eps,
    const float inv_scale                         // 1 / k_scale, matching the decode read
) {
    const unsigned int token = blockIdx.x;
    const unsigned int t = threadIdx.x;
    if (t >= kv_lora_dim) return;

    // A -1 slot is a real case (prefix cache hit, padded row), not an error.
    const long long slot = slot_mapping[token];
    if (slot < 0) return;

    const __nv_bfloat16* row = kv_a + (unsigned long long)token * kv_lora_dim;
    const float x = __bfloat162float(row[t]);

    // RMS over the full latent. Block is exactly kv_lora_dim wide, so the two-stage
    // reduction covers every element with no tail loop.
    float ss = warp_reduce_sum_glm(x * x);
    __shared__ float warp_sums[32];
    const unsigned int warp_id = t / 32;
    const unsigned int lane = t % 32;
    if (lane == 0) warp_sums[warp_id] = ss;
    __syncthreads();
    if (warp_id == 0) {
        const unsigned int n_warps = (kv_lora_dim + 31) / 32;
        float v = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        v = warp_reduce_sum_glm(v);
        if (lane == 0) warp_sums[0] = v;
    }
    __syncthreads();

    const float rms = rsqrtf(warp_sums[0] / (float)kv_lora_dim + rms_eps);
    // 🪤 PLAIN RMSNorm — no `1.0f +` on the weight. See the header.
    const float normed = x * rms * __bfloat162float(norm_w[t]);

    cache[(unsigned long long)slot * kv_lora_dim + t] =
        __nv_cvt_float_to_fp8(normed * inv_scale, __NV_SATFINITE, __NV_E4M3);
}
