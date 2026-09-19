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

#include <math_constants.h>
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
            float m = -CUDART_INF_F;
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

// The same mixes, spread over the GPU: one block per (token, mix). Each block
// recomputes the stream rms with the identical strided loop and block
// reduction (so the bits match hc_v41_mixes), takes its one dot product, and
// writes mixes[t][m] = acc * rsq. Grid: (T, mix_hc). Block: 256.
extern "C" __global__ void hc_v41_mixes_dot(
    const float* __restrict__ streams, // [T, hc, H]
    const float* __restrict__ hc_fn,   // [mix_hc, hc*H]
    float* __restrict__ mixes_out,     // [T, mix_hc]
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const float norm_eps) {
    const unsigned int t = blockIdx.x;
    const unsigned int m = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int hc = hc_mult;
    const unsigned int n = hc * hidden_size;
    const unsigned int mix_hc = (2 + hc) * hc;
    __shared__ float red[HCV_BLOCK];
    const float* x = streams + (size_t)t * n;
    float ss = 0.0f;
    for (unsigned int i = tid; i < n; i += HCV_BLOCK) ss += x[i] * x[i];
    ss = hcv_block_sum(ss, red);
    const float rsq = rsqrtf(ss / (float)n + norm_eps);
    const float* w = hc_fn + (size_t)m * n;
    float acc = 0.0f;
    for (unsigned int i = tid; i < n; i += HCV_BLOCK) acc += w[i] * x[i];
    acc = hcv_block_sum(acc, red);
    if (tid == 0) mixes_out[(size_t)t * mix_hc + m] = acc * rsq;
}

// The epilogue of hc_v41_mixes on the precomputed mixes: sigmoids, the row
// softmax, the sinkhorn passes. One thread's serial work; with HC a compile
// time constant the loops unroll and c[HC*HC] lives in registers instead of
// local memory (the runtime-hc form spent 52 us a call on a 16-element
// array). Same expressions in the same order, so the same bits.
template <int HC>
__device__ __forceinline__ void hcv_finish_t(
    const float* __restrict__ mixes,    // [mix_hc] of this token
    const float* __restrict__ hc_scale, // [3]
    const float* __restrict__ hc_base,  // [mix_hc]
    float* __restrict__ pre,            // [HC] of this token
    float* __restrict__ post,           // [HC]
    float* __restrict__ comb,           // [HC, HC]
    const unsigned int sinkhorn_iters,
    const float hc_eps) {
    constexpr unsigned int hc = HC;
    float c[HC * HC];
#pragma unroll
    for (unsigned int j = 0; j < hc; ++j) {
        pre[j] = hcv_sigmoid(mixes[j] * hc_scale[0] + hc_base[j]) + hc_eps;
        post[j] = 2.0f * hcv_sigmoid(mixes[hc + j] * hc_scale[1] + hc_base[hc + j]);
    }
#pragma unroll
    for (unsigned int i = 0; i < hc * hc; ++i) c[i] = mixes[2 * hc + i] * hc_scale[2] + hc_base[2 * hc + i];
#pragma unroll
    for (unsigned int j = 0; j < hc; ++j) {
        float m = -CUDART_INF_F;
#pragma unroll
        for (unsigned int k = 0; k < hc; ++k) m = fmaxf(m, c[j * hc + k]);
        float sum = 0.0f;
#pragma unroll
        for (unsigned int k = 0; k < hc; ++k) { c[j * hc + k] = expf(c[j * hc + k] - m); sum += c[j * hc + k]; }
#pragma unroll
        for (unsigned int k = 0; k < hc; ++k) c[j * hc + k] = c[j * hc + k] / sum + hc_eps;
    }
    for (unsigned int it = 0; it < sinkhorn_iters; ++it) {
        if (it > 0) {
#pragma unroll
            for (unsigned int j = 0; j < hc; ++j) {
                float s = 0.0f;
#pragma unroll
                for (unsigned int k = 0; k < hc; ++k) s += c[j * hc + k];
#pragma unroll
                for (unsigned int k = 0; k < hc; ++k) c[j * hc + k] /= s + hc_eps;
            }
        }
#pragma unroll
        for (unsigned int k = 0; k < hc; ++k) {
            float s = 0.0f;
#pragma unroll
            for (unsigned int j = 0; j < hc; ++j) s += c[j * hc + k];
#pragma unroll
            for (unsigned int j = 0; j < hc; ++j) c[j * hc + k] /= s + hc_eps;
        }
    }
#pragma unroll
    for (unsigned int i = 0; i < hc * hc; ++i) comb[i] = c[i];
}
// hc_mult <= 4 (HCV_MAX_HC); dispatch to the constant-HC form.
__device__ __forceinline__ void hcv_finish(
    const unsigned int hc, const float* mixes, const float* hc_scale, const float* hc_base,
    float* pre, float* post, float* comb, const unsigned int sinkhorn_iters, const float hc_eps) {
    switch (hc) {
        case 4: hcv_finish_t<4>(mixes, hc_scale, hc_base, pre, post, comb, sinkhorn_iters, hc_eps); break;
        case 3: hcv_finish_t<3>(mixes, hc_scale, hc_base, pre, post, comb, sinkhorn_iters, hc_eps); break;
        case 2: hcv_finish_t<2>(mixes, hc_scale, hc_base, pre, post, comb, sinkhorn_iters, hc_eps); break;
        default: hcv_finish_t<1>(mixes, hc_scale, hc_base, pre, post, comb, sinkhorn_iters, hc_eps); break;
    }
}
// Grid: (T). Block: 32 (thread 0 works).
extern "C" __global__ void hc_v41_mixes_finish(
    const float* __restrict__ mixes_in, // [T, mix_hc]
    const float* __restrict__ hc_scale, // [3]
    const float* __restrict__ hc_base,  // [mix_hc]
    float* __restrict__ pre,            // [T, hc]
    float* __restrict__ post,           // [T, hc]
    float* __restrict__ comb,           // [T, hc, hc]
    const unsigned int hc_mult,
    const unsigned int sinkhorn_iters,
    const float hc_eps) {
    const unsigned int t = blockIdx.x;
    if (threadIdx.x != 0) return;
    const unsigned int hc = hc_mult;
    const unsigned int mix_hc = (2 + hc) * hc;
    hcv_finish(hc, mixes_in + (size_t)t * mix_hc, hc_scale, hc_base, pre + (size_t)t * hc,
               post + (size_t)t * hc, comb + (size_t)t * hc * hc, sinkhorn_iters, hc_eps);
}
// The collapse of one column: y[t][d] = sum_c pre[c] * streams[t][c][d].
__device__ __forceinline__ void hcv_collapse_col(
    const float* __restrict__ x, const float* __restrict__ p, __nv_bfloat16* __restrict__ y,
    const unsigned int hidden_size, const unsigned int hc_mult, const unsigned int d) {
    float acc = 0.0f;
    for (unsigned int c = 0; c < hc_mult; ++c) acc += p[c] * x[(size_t)c * hidden_size + d];
    y[d] = __float2bfloat16(acc);
}
// The site's finish and the block's collapse in one launch. They are
// independent: the collapse reads the PREVIOUS site's pre (pre_in) while the
// finish writes this site's (pre_out), never the same buffer. Block (0, t)
// thread 0 runs the finish; every thread owns one column d of the collapse.
// Grid: (ceil(H/256), T). Block: 256.
extern "C" __global__ void hc_v41_finish_collapse(
    const float* __restrict__ mixes_in, // [T, mix_hc]
    const float* __restrict__ hc_scale, // [3]
    const float* __restrict__ hc_base,  // [mix_hc]
    float* __restrict__ pre_out,        // [T, hc]
    float* __restrict__ post,           // [T, hc]
    float* __restrict__ comb,           // [T, hc, hc]
    const float* __restrict__ streams,  // [T, hc, H]
    const float* __restrict__ pre_in,   // [T, hc]
    __nv_bfloat16* __restrict__ y,      // [T, H]
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int sinkhorn_iters,
    const float hc_eps) {
    const unsigned int t = blockIdx.y;
    const unsigned int hc = hc_mult;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const unsigned int mix_hc = (2 + hc) * hc;
        hcv_finish(hc, mixes_in + (size_t)t * mix_hc, hc_scale, hc_base, pre_out + (size_t)t * hc,
                   post + (size_t)t * hc, comb + (size_t)t * hc * hc, sinkhorn_iters, hc_eps);
    }
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden_size) return;
    hcv_collapse_col(streams + (size_t)t * hc * hidden_size, pre_in + (size_t)t * hc,
                     y + (size_t)t * hidden_size, hidden_size, hc, d);
}
// hc_v41_collapse spread over ceil(H/256) blocks a token, one column a thread.
// Grid: (ceil(H/256), T). Block: 256.
extern "C" __global__ void hc_v41_collapse_wide(
    const float* __restrict__ streams, // [T, hc, H]
    const float* __restrict__ pre,     // [T, hc]
    __nv_bfloat16* __restrict__ y,     // [T, H]
    const unsigned int hidden_size,
    const unsigned int hc_mult) {
    const unsigned int t = blockIdx.y;
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden_size) return;
    hcv_collapse_col(streams + (size_t)t * hc_mult * hidden_size, pre + (size_t)t * hc_mult,
                     y + (size_t)t * hidden_size, hidden_size, hc_mult, d);
}
// hc_post (hyper_connection.cu) spread the same way: one column a thread, the
// per-column body unchanged (all hc residual values of the column are read
// before the column is written, so `out` may still alias `residual`).
// out[t,j,d] = post[t,j]*block_out[t,d] + sum_i comb[t,i,j]*residual[t,i,d].
// Grid: (ceil(H/256), T). Block: 256.
extern "C" __global__ void hc_v41_post_wide(
    const __nv_bfloat16* __restrict__ block_out, // [T, H]
    const float* __restrict__ residual,          // [T, hc, H]
    const float* __restrict__ post,              // [T, hc]
    const float* __restrict__ comb,              // [T, hc, hc]
    float* __restrict__ out,                     // [T, hc, H]
    const unsigned int hidden_size,
    const unsigned int hc_mult) {
    const unsigned int t = blockIdx.y;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= H) return;
    const __nv_bfloat16* x = block_out + (size_t)t * H;
    const float* res = residual + (size_t)t * hc * H;
    const float* p = post + (size_t)t * hc;
    const float* c = comb + (size_t)t * hc * hc;
    float* o = out + (size_t)t * hc * H;
    float xd = (float)x[d];
    float rv[HCV_MAX_HC];
    for (unsigned int i = 0; i < hc; ++i) rv[i] = res[i * H + d];
    for (unsigned int j = 0; j < hc; ++j) {
        float acc = p[j] * xd;
        for (unsigned int i = 0; i < hc; ++i) acc += c[i * hc + j] * rv[i];
        o[j * H + d] = acc;
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
