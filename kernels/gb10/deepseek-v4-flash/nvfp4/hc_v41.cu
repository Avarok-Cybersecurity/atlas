// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//
// DeepSeek-V4.1 Flash hyper-connections, the DELAYED-mix form.
//
// V4 Flash's hc_pre (hyper_connection.cu) derives `pre` from the streams it
// collapses. V4.1's Block.forward does not: a block's attention collapses with
// the `pre` its predecessor's FFN mixes produced, and its FFN collapses with the
// `pre` its own attention mixes produced (deepseek_v41_ref::hc, checked against
// DeepSeek's golden). So the mixes and the collapse are two kernels here:
//
//   hc_v41_mixes    streams [T, hc, H] f32 + site fn/scale/base -> pre [T, hc],
//                   post [T, hc], comb [T, hc, hc] (row-major [j][k]) f32.
//                   One RMS over the flattened hc*H stream, mix_hc = (2+hc)*hc
//                   projections scaled by it, then hc_split_sinkhorn:
//                   pre = sigmoid(m*s0 + b) + eps, post = 2 sigmoid(m*s1 + b),
//                   comb = row softmax + eps, then col/row normalisations.
//   hc_v41_collapse streams [T, hc, H] f32 x pre [T, hc] -> y [T, H] bf16.
//
// hc_post is V4's kernel unchanged: y[k][d] = post[k] x[d] + sum_j comb[j][k] r[j][d].
// hc_mult <= 4.

#include <cuda_bf16.h>

#define HCV_BLOCK 256
#define HCV_MAX_HC 4
#define HCV_MAX_MIX 24

__device__ __forceinline__ float hcv_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float hcv_block_sum(float v, float* red) {
    const unsigned int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (unsigned int s = HCV_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

// Grid: (T). Block: 256.
extern "C" __global__ void hc_v41_mixes(
    const float* __restrict__ streams, // [T, hc, H]
    const float* __restrict__ hc_fn,   // [mix_hc, hc*H]
    const float* __restrict__ hc_scale,// [3]
    const float* __restrict__ hc_base, // [mix_hc]
    float* __restrict__ pre,           // [T, hc]
    float* __restrict__ post,          // [T, hc]
    float* __restrict__ comb,          // [T, hc, hc]
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int sinkhorn_iters,
    const float norm_eps,
    const float hc_eps) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int hc = hc_mult;
    const unsigned int n = hc * hidden_size;
    const unsigned int mix_hc = (2 + hc) * hc;
    __shared__ float red[HCV_BLOCK];
    __shared__ float mixes[HCV_MAX_MIX];
    __shared__ float c[HCV_MAX_HC * HCV_MAX_HC];
    const float* x = streams + (size_t)t * n;

    float ss = 0.0f;
    for (unsigned int i = tid; i < n; i += HCV_BLOCK) ss += x[i] * x[i];
    ss = hcv_block_sum(ss, red);
    const float rsq = rsqrtf(ss / (float)n + norm_eps);
    for (unsigned int m = 0; m < mix_hc; ++m) {
        const float* w = hc_fn + (size_t)m * n;
        float acc = 0.0f;
        for (unsigned int i = tid; i < n; i += HCV_BLOCK) acc += w[i] * x[i];
        acc = hcv_block_sum(acc, red);
        if (tid == 0) mixes[m] = acc * rsq;
    }
    __syncthreads();
    if (tid == 0) {
        for (unsigned int j = 0; j < hc; ++j) {
            pre[(size_t)t * hc + j] = hcv_sigmoid(mixes[j] * hc_scale[0] + hc_base[j]) + hc_eps;
            post[(size_t)t * hc + j] = 2.0f * hcv_sigmoid(mixes[hc + j] * hc_scale[1] + hc_base[hc + j]);
        }
        for (unsigned int i = 0; i < hc * hc; ++i) c[i] = mixes[2 * hc + i] * hc_scale[2] + hc_base[2 * hc + i];
        // row softmax + eps
        for (unsigned int j = 0; j < hc; ++j) {
            float m = -INFINITY;
            for (unsigned int k = 0; k < hc; ++k) m = fmaxf(m, c[j * hc + k]);
            float sum = 0.0f;
            for (unsigned int k = 0; k < hc; ++k) { c[j * hc + k] = expf(c[j * hc + k] - m); sum += c[j * hc + k]; }
            for (unsigned int k = 0; k < hc; ++k) c[j * hc + k] = c[j * hc + k] / sum + hc_eps;
        }
        // col norm, then (row norm, col norm) x (iters - 1)
        for (unsigned int it = 0; it < sinkhorn_iters; ++it) {
            if (it > 0) {
                for (unsigned int j = 0; j < hc; ++j) {
                    float s = 0.0f;
                    for (unsigned int k = 0; k < hc; ++k) s += c[j * hc + k];
                    for (unsigned int k = 0; k < hc; ++k) c[j * hc + k] /= s + hc_eps;
                }
            }
            for (unsigned int k = 0; k < hc; ++k) {
                float s = 0.0f;
                for (unsigned int j = 0; j < hc; ++j) s += c[j * hc + k];
                for (unsigned int j = 0; j < hc; ++j) c[j * hc + k] /= s + hc_eps;
            }
        }
        for (unsigned int i = 0; i < hc * hc; ++i) comb[(size_t)t * hc * hc + i] = c[i];
    }
}

// Grid: (T). Block: 256.
extern "C" __global__ void hc_v41_collapse(
    const float* __restrict__ streams, // [T, hc, H]
    const float* __restrict__ pre,     // [T, hc]
    __nv_bfloat16* __restrict__ y,     // [T, H]
    const unsigned int hidden_size,
    const unsigned int hc_mult) {
    const unsigned int t = blockIdx.x;
    const float* x = streams + (size_t)t * hc_mult * hidden_size;
    const float* p = pre + (size_t)t * hc_mult;
    for (unsigned int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int c = 0; c < hc_mult; ++c) acc += p[c] * x[(size_t)c * hidden_size + d];
        y[(size_t)t * hidden_size + d] = __float2bfloat16(acc);
    }
}
