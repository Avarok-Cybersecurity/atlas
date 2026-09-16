// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//
// DeepSeek-V4.1 Flash engram, the gate stage on the mHC highway.
//
// Engram (layers 1 and 14) looks up `n_hash_cols` rows of a hashed n-gram table
// per token, projects the concatenation through `wkv` into one key per
// hyper-connection stream plus one shared value, and adds `gate * value` to
// each stream, where the gate is a signed-sqrt sigmoid of the normalised dot
// product of the stream against its key. The lookup is `pread` + the Q2_K row
// dequant (common/dequant_gguf_bf16.cu), the projection is the dense bf16 GEMM
// (common/dense_gemm_bf16.cu); this file is the gate, written straight onto the
// FP32 highway the mHC kernels keep ([T, hc, H], stream-major per token).
//
// Reference: deepseek-ai/DeepSeek-V4.1-Flash inference/model.py Engram.forward,
// pinned by the CPU reference `deepseek_v41_ref::engram::gate_and_add`:
//   rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)
//   dot  = sum_d h[d] * (q_w[c,d] * k_w[c,d]) * key[d] * rstd * dim^-0.5
//   g    = copysign(sqrt(max(|dot|, 1e-6)), dot);  gate = sigmoid(g)
//   h   += gate * value
// `qk` below is the elementwise product q_w * k_w, precomputed at load: the two
// weights are only ever used as that product.
//
// Grid: (T, hc). Block: 256. One block per (token, stream).

#include <cuda_bf16.h>

#define ENGRAM_BLOCK 256

__device__ __forceinline__ float engram_block_sum(float v, float* red) {
    const unsigned int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (unsigned int s = ENGRAM_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

extern "C" __global__ void engram_v41_gate(
    float* __restrict__ streams,             // [T, hc, H] FP32 highway, updated in place
    const __nv_bfloat16* __restrict__ kv,    // [T, H * (hc + 1)] bf16: hc keys then the value
    const float* __restrict__ qk,            // [hc, H] q_w * k_w
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const float norm_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int c = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    __shared__ float red[ENGRAM_BLOCK];

    float* h = streams + ((size_t)t * hc_mult + c) * H;
    const __nv_bfloat16* key = kv + (size_t)t * H * (hc_mult + 1) + (size_t)c * H;
    const __nv_bfloat16* value = kv + (size_t)t * H * (hc_mult + 1) + (size_t)hc_mult * H;
    const float* w = qk + (size_t)c * H;

    float hh = 0.0f, kk = 0.0f, dot = 0.0f;
    for (unsigned int d = tid; d < H; d += ENGRAM_BLOCK) {
        const float hv = h[d];
        const float kvv = __bfloat162float(key[d]);
        hh += hv * hv;
        kk += kvv * kvv;
        dot += hv * w[d] * kvv;
    }
    hh = engram_block_sum(hh, red);
    kk = engram_block_sum(kk, red);
    dot = engram_block_sum(dot, red);

    const float inv_h = rsqrtf(hh / (float)H + norm_eps);
    const float inv_k = rsqrtf(kk / (float)H + norm_eps);
    const float scaled = dot * inv_h * inv_k * rsqrtf((float)H);
    const float g = copysignf(sqrtf(fmaxf(fabsf(scaled), 1e-6f)), scaled);
    const float gate = 1.0f / (1.0f + expf(-g));

    for (unsigned int d = tid; d < H; d += ENGRAM_BLOCK) {
        h[d] += gate * __bfloat162float(value[d]);
    }
}
