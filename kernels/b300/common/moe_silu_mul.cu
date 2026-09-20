// SPDX-License-Identifier: AGPL-3.0-only

// Atlas MoE element-wise SiLU activation + multiply.
//
// output[i] = silu(gate[i]) * up[i]
// where silu(x) = x * sigmoid(x)
//
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)
//
// Used after grouped gate+up GEMMs to fuse activation before down GEMM.
//
// NO SWIGLU CLAMP HERE, deliberately. One lived here from #186 until wave 55: a
// `fminf(g, 10.0f)` / `clamp(u, ±10.0f)` labelled "DeepSeek-V4 routed-expert
// swiglu clamp (swiglu_limit = 10.0, config)". The label was right about where
// the number came from and wrong about who it reached. This kernel is not the
// DeepSeek routed path — it is the SiLU activation for every dense model's
// decode and K-verify FFN, every MoE model's grouped prefill, and the MTP and
// DFlash draft heads. `swiglu_limit` is a per-checkpoint config value
// (DeepSeek-V4 10.0, GPT-OSS 7.0, every Qwen/Gemma/Nemotron/Mistral checkpoint
// on the fleet: absent), and the references for the models that do not declare
// one do not clamp: `Qwen3_5MLP.forward` is a bare `act_fn(gate) * up`.
//
// It was not dormant. Instrumented on Qwen3.6-27B over a 20-sample BFCL draw it
// bound over 100,000 times in this kernel alone, with `up` reaching -21.78 and
// `gate` reaching 17.38 — truncations of more than 2x, on a checkpoint whose
// config declares no limit.
//
// The models that DO declare a limit shadow this file:
// `deepseek-v4-flash/nvfp4/moe_silu_mul.cu` and
// `step3p7-flash/nvfp4/moe_silu_mul.cu`. That is a holding pattern, not the
// destination — Step-3.7's limits are PER LAYER, so they cannot be a compile-
// time constant and eventually want a kernel argument fed from `ModelConfig`.
// Anything added here reaches the whole fleet; add it to a shadow instead.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

extern "C" __global__ void moe_silu_mul(
    const __nv_bfloat16* __restrict__ gate,   // [total_expanded, inter_size]
    const __nv_bfloat16* __restrict__ up,     // [total_expanded, inter_size]
    __nv_bfloat16* __restrict__ output,        // [total_expanded, inter_size]
    unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    output[idx] = __float2bfloat16(result);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused SiLU·mul + per-token-group(128) FP8-E4M3 quantization.
//
// Replaces the `moe_silu_mul` → `per_token_group_quant_fp8` pair on the W8A8
// prefill down-path without ever materializing / re-reading the BF16
// intermediate (36.9 MB per MoE layer at N=4510): 9 B/elem of traffic → 5 B
// (7 B with `out_bf16`). Measured 1.83× over the pair at the production
// expanded shape [36080, 512] on GB10, bit-identical outputs.
//
// BIT-EXACT CONTRACT with the unfused pair:
//   • the product is rounded through BF16 (`__float2bfloat16` round-trip)
//     BEFORE the group max / encode, so the quantizer input is exactly the
//     BF16 value `per_token_group_quant_fp8` would have re-read from memory;
//   • the max reduction replicates that kernel's structure verbatim (per-warp
//     `__shfl_down_sync` cascade → 4-slot smem → thread-0 sequential fmaxf),
//     so even NaN propagation is order-identical;
//   • scale floor (1e-12), saturation clamp and `__NV_SATFINITE` E4M3 encode
//     are copied unchanged.
//
// `out_bf16` (nullable): when the caller also needs the post-SiLU BF16 rows
// (expert down_proj LoRA fold consumes them), pass a [M, K] buffer and the
// kernel additionally writes the exact `moe_silu_mul` output. Pass NULL
// otherwise (fast path).
//
// Grid: (M, 1, 1)  Block: (128, 1, 1) — one block per row, K/128 groups
// per row. Requires K % 128 == 0 and K/128 <= SILU_QUANT_MAX_GROUPS (the
// launch site falls back to the unfused pair otherwise).
//
// Same no-clamp scope note as `moe_silu_mul` above: models that declare a
// swiglu_limit shadow this file and do not get this entry point (their
// kernel handle lookup returns null → unfused fallback).

#define FP8_GROUP_K 128
#define FP8_E4M3_MAX 448.0f
#define SILU_QUANT_MAX_GROUPS 16

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// gfx1151/SCALE: software SATFINITE E4M3 encode — byte-identical duplicate of
// `scl_enc_fp8` in per_token_group_quant_fp8.cu (which itself mirrors scl_fp8
// in w4a16_gemm.cu / fp8_gemm_t_blockscaled.cu). Must stay in sync with the
// decode those kernels use.
__device__ __forceinline__ unsigned char silu_quant_enc_fp8(float v) {
    if (v != v) return 0x7F;
    unsigned int bb = __float_as_uint(v); unsigned int sign = (bb >> 31) & 1u;
    int e = (int)((bb >> 23) & 0xFF) - 127; unsigned int man = bb & 0x7FFFFFu;
    int ee = e + 7; unsigned int em;
    if (ee < 1) { ee = 0; em = 0; if (e >= -10) { float a = v < 0 ? -v : v; em = (unsigned int)(a / 0.001953125f + 0.5f); if (em > 7u) em = 7u; } }
    else if (ee > 15) { ee = 15; em = 6; }
    else { em = (man + (1u << 19)) >> 20; if (em > 7u) { em = 0; ee++; if (ee > 15) { ee = 15; em = 6; } } }
    return (unsigned char)((sign << 7) | ((unsigned)ee << 3) | em);
}
#endif

extern "C" __global__ void silu_mul_quant_fp8(
    const __nv_bfloat16* __restrict__ gate,   // [M, K] gate GEMM output
    const __nv_bfloat16* __restrict__ up,     // [M, K] up GEMM output
    unsigned char* __restrict__ out_fp8,      // [M, K] FP8 E4M3
    float* __restrict__ a_scale,              // [M, K/128] FP32 group scale
    __nv_bfloat16* __restrict__ out_bf16,     // [M, K] post-SiLU BF16, or NULL
    unsigned int M,
    unsigned int K
) {
    const unsigned int m = blockIdx.x;
    if (m >= M) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int ngroups = K / FP8_GROUP_K;
    const __nv_bfloat16* grow = gate + (unsigned long long)m * K;
    const __nv_bfloat16* urow = up + (unsigned long long)m * K;
    unsigned char* orow = out_fp8 + (unsigned long long)m * K;

    __shared__ float smem_warp_max[4];
    __shared__ float smem_scale;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31;

    float vals[SILU_QUANT_MAX_GROUPS];
    for (unsigned int kg = 0; kg < ngroups; kg++) {
        const unsigned int k = kg * FP8_GROUP_K + tid;
        float g = __bfloat162float(grow[k]);
        float u = __bfloat162float(urow[k]);
        float sigmoid_g = 1.0f / (1.0f + __expf(-g));
        float result = g * sigmoid_g * u;
        // Round through BF16 FIRST — the quantizer input must be exactly the
        // BF16 value the unfused pair writes to / re-reads from memory.
        __nv_bfloat16 r16 = __float2bfloat16(result);
        if (out_bf16 != nullptr) out_bf16[(unsigned long long)m * K + k] = r16;
        float r = __bfloat162float(r16);
        vals[kg] = r;

        float warp_max = fabsf(r);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, off));
        }
        if (lane == 0) smem_warp_max[warp_id] = warp_max;
        __syncthreads();
        if (tid == 0) {
            float global_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) global_max = fmaxf(global_max, smem_warp_max[i]);
            float scale = global_max / FP8_E4M3_MAX;
            if (scale < 1e-12f) scale = 1e-12f;
            a_scale[m * ngroups + kg] = scale;
            smem_scale = scale;
        }
        __syncthreads();
        float v = vals[kg] / smem_scale;
        v = fmaxf(fminf(v, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        orow[k] = silu_quant_enc_fp8(v);
#else
        orow[k] = (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
#endif
        __syncthreads();  // smem_warp_max / smem_scale reused next group
    }
}
