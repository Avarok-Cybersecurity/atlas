// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//
// DeepSeek-V4.1 Flash attention, the pieces the dense bf16 GEMM does not cover.
// One forward serves every layer: ratio 0 (sliding window only), ratio 1 and
// ratio 2 (window + the shared compressed latent). The GEMMs (wq_a, wq_b, wkv,
// wo_b, the indexer projections, the grouped wo_a) run on common/dense_gemm_bf16;
// everything here is the elementwise and reduction work between them, written
// to the CPU reference's precision (deepseek_v41_ref::{attn,compress}): f32
// math, bf16 storage, the quantisers' exact rounding rules.
//
// Kernels:
//   attn_v41_rmsnorm_bf16 / _f32   RMSNorm with an f32 weight, bf16 output
//   attn_v41_rope                  partial RoPE on the last rope_dim of a row,
//                                  adjacent pairs, f32 complex multiply, bf16
//   attn_v41_act_quant_fp8         act_quant: block 32, e4m3 with a ue8m0 scale
//   attn_v41_fp4_quant             fp4_act_quant: e2m1 with an e8m0 (mode 0) or
//                                  e4m3 (mode 1) scale, any block size
//   attn_v41_gemm_f32              C[M,N] = A[M,K](bf16) * B[N,K](f32)^T in f32 (compressor)
//   attn_v41_scale_bf16            out = bf16(bf16(in) * s) (the indexer's head weights)
//   attn_v41_pool                  the ratio-2 compressor's per-dim softmax pooling
//   attn_v41_index_score           the indexer's rectified, weighted head scores
//   attn_v41_sparse_attn           softmax over gathered rows plus the sink
//   attn_v41_slice_cols / _scatter_cols   column slices for the grouped wo_a
//   attn_v41_ring_put              one row into the window ring at pos % win,
//                                  the position read from the device (CUDA graph)

#include <cuda_bf16.h>
#include <math_constants.h>
#include <cuda_fp8.h>

#define AV_BLOCK 256

__device__ __forceinline__ float av_bf16r(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

__device__ __forceinline__ float av_block_sum(float v, float* red) {
    const unsigned int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float av_block_max(float v, float* red) {
    const unsigned int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

// `2 ** ceil(log2(v))` via the IEEE fields (kernel.py fast_log2_ceil / fast_pow2).
__device__ __forceinline__ float av_pow2_ceil(float v) {
    const unsigned int bits = __float_as_uint(v);
    const int e = (int)((bits >> 23) & 0xFF) - 127;
    const unsigned int man = bits & 0x7FFFFF;
    const int n = e + (man != 0 ? 1 : 0);
    return __uint_as_float((unsigned int)(n + 127) << 23);
}

// e4m3 round trip, RNE, saturating (the reference clamps to +-448 first, then RNE).
__device__ __forceinline__ float av_e4m3_round_trip(float x) {
    const __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    return __half2float(__nv_cvt_fp8_to_halfraw(q, __NV_E4M3));
}

// Round |y| onto the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to the even
// code, sign kept (`_to_e2m1_rne`).
__device__ __forceinline__ float av_e2m1_rne(float y) {
    const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float a = fabsf(y);
    int lo = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (grid[i] <= a) lo = i;
    }
    const int hi = lo + 1 < 7 ? lo + 1 : 7;
    const float dlo = a - grid[lo];
    const float dhi = grid[hi] - a;
    const bool pick_hi = dhi < dlo || (dhi == dlo && (hi % 2) == 0 && (lo % 2) == 1);
    const float v = pick_hi ? grid[hi] : grid[lo];
    return copysignf(v, y);
}

// ── RMSNorm: `w * (x * rsqrt(mean(x^2) + eps))` in f32, out bf16 ──────────────
// Grid: (rows). Block: AV_BLOCK.
extern "C" __global__ void attn_v41_rmsnorm_bf16(
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ w,
    __nv_bfloat16* __restrict__ out, const unsigned int dim, const float eps) {
    __shared__ float red[AV_BLOCK];
    const size_t base = (size_t)blockIdx.x * dim;
    float ss = 0.0f;
    for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float v = __bfloat162float(x[base + d]);
        ss += v * v;
    }
    ss = av_block_sum(ss, red);
    const float r = rsqrtf(ss / (float)dim + eps);
    for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) {
        out[base + d] = __float2bfloat16(w[d] * (__bfloat162float(x[base + d]) * r));
    }
}

extern "C" __global__ void attn_v41_rmsnorm_f32(
    const float* __restrict__ x, const float* __restrict__ w,
    __nv_bfloat16* __restrict__ out, const unsigned int dim, const float eps) {
    __shared__ float red[AV_BLOCK];
    const size_t base = (size_t)blockIdx.x * dim;
    float ss = 0.0f;
    for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float v = x[base + d];
        ss += v * v;
    }
    ss = av_block_sum(ss, red);
    const float r = rsqrtf(ss / (float)dim + eps);
    for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) {
        out[base + d] = __float2bfloat16(w[d] * (x[base + d] * r));
    }
}

// ── RoPE on the last `rope_dim` of each `row_len`-wide row ───────────────────
// `fc` is [max_pos][rope_dim/2] of (cos, sin); `pos[r]` per row; `inverse`
// negates sin. Grid: (rows). Block: >= rope_dim/2.
extern "C" __global__ void attn_v41_rope(
    __nv_bfloat16* __restrict__ x, const int* __restrict__ pos,
    const float2* __restrict__ fc, const unsigned int row_len,
    const unsigned int rope_dim, const unsigned int inverse) {
    const unsigned int r = blockIdx.x;
    const unsigned int half = rope_dim / 2;
    const unsigned int k = threadIdx.x;
    if (k >= half) return;
    __nv_bfloat16* row = x + (size_t)r * row_len + (row_len - rope_dim);
    const float2 cs = fc[(size_t)pos[r] * half + k];
    const float c = cs.x;
    const float s = inverse ? -cs.y : cs.y;
    const float a = __bfloat162float(row[2 * k]);
    const float b = __bfloat162float(row[2 * k + 1]);
    row[2 * k] = __float2bfloat16(a * c - b * s);
    row[2 * k + 1] = __float2bfloat16(a * s + b * c);
}

// ── act_quant: block 32, scale = pow2_ceil(amax / 448) with amax >= 1e-4 ──────
// In place on bf16 rows; `n_blocks` blocks of 32. Grid: ceil(n_blocks / 8).
// Block: 256 = 8 blocks x 32 lanes, one warp per block.
extern "C" __global__ void attn_v41_act_quant_fp8(
    __nv_bfloat16* __restrict__ x, const unsigned int n_blocks) {
    const unsigned int gb = blockIdx.x * 8 + threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;
    if (gb >= n_blocks) return;
    __nv_bfloat16* p = x + (size_t)gb * 32 + lane;
    const float v = __bfloat162float(*p);
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    amax = fmaxf(amax, 1e-4f);
    const float s = av_pow2_ceil(amax / 448.0f);
    const float q = av_e4m3_round_trip(fminf(fmaxf(v / s, -448.0f), 448.0f));
    *p = __float2bfloat16(q * s);
}

// ── fp4_act_quant: e2m1 values, scale mode 0 = e8m0 (pow2 ceil of amax/6, amax
// >= 6*2^-126), mode 1 = e4m3 (amax/6 through e4m3, amax >= 6*2^-9) ───────────
// In place on bf16; `block` values per scale; one warp per block, lanes stride.
extern "C" __global__ void attn_v41_fp4_quant(
    __nv_bfloat16* __restrict__ x, const unsigned int n_blocks,
    const unsigned int block, const unsigned int mode) {
    const unsigned int gb = blockIdx.x * 8 + threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;
    if (gb >= n_blocks) return;
    __nv_bfloat16* p = x + (size_t)gb * block;
    float amax = 0.0f;
    for (unsigned int i = lane; i < block; i += 32) amax = fmaxf(amax, fabsf(__bfloat162float(p[i])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float s;
    if (mode == 0) {
        amax = fmaxf(amax, 6.0f * 1.1754943508222875e-38f);
        s = av_pow2_ceil(amax / 6.0f);
    } else {
        amax = fmaxf(amax, 6.0f * 0.001953125f);
        s = av_e4m3_round_trip(amax / 6.0f);
    }
    for (unsigned int i = lane; i < block; i += 32) {
        const float v = __bfloat162float(p[i]);
        const float q = av_e2m1_rne(fminf(fmaxf(v / s, -6.0f), 6.0f));
        p[i] = __float2bfloat16(q * s);
    }
}

// ── f32 GEMM: C[M,N] = A[M,K] * B[N,K]^T, 16x16 tiles: the compressor's fp32
// weights on the bf16-valued input, accumulated in f32 as the reference does ──
#define AV_TILE 16
extern "C" __global__ void attn_v41_gemm_f32(
    const __nv_bfloat16* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    const unsigned int M, const unsigned int N, const unsigned int K) {
    const unsigned int row = blockIdx.y * AV_TILE + threadIdx.y;
    const unsigned int col = blockIdx.x * AV_TILE + threadIdx.x;
    __shared__ float sa[AV_TILE][AV_TILE];
    __shared__ float sb[AV_TILE][AV_TILE];
    float acc = 0.0f;
    for (unsigned int k0 = 0; k0 < K; k0 += AV_TILE) {
        const unsigned int ka = k0 + threadIdx.x;
        const unsigned int kb = k0 + threadIdx.y;
        sa[threadIdx.y][threadIdx.x] = (row < M && ka < K) ? __bfloat162float(A[(size_t)row * K + ka]) : 0.0f;
        sb[threadIdx.y][threadIdx.x] = (col < N && kb < K) ? B[(size_t)col * K + kb] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < AV_TILE; ++k) acc += sa[threadIdx.y][k] * sb[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// ── compressor pooling: per group g and dim d, softmax over `ratio` members of
// score, weighted sum of kv; result rounded to bf16 and stored as f32 for the
// f32 RMSNorm that follows ─────────────────────────────────────────────────────
// kv, score: [groups * ratio, hd] f32. out: [groups, hd] f32 (bf16-valued).
extern "C" __global__ void attn_v41_pool(
    const float* __restrict__ kv, const float* __restrict__ score, float* __restrict__ out,
    const unsigned int ratio, const unsigned int hd) {
    const unsigned int g = blockIdx.x;
    for (unsigned int d = threadIdx.x; d < hd; d += blockDim.x) {
        float m = -CUDART_INF_F;
        for (unsigned int r = 0; r < ratio; ++r) m = fmaxf(m, score[((size_t)g * ratio + r) * hd + d]);
        float sum = 0.0f;
        for (unsigned int r = 0; r < ratio; ++r) sum += expf(score[((size_t)g * ratio + r) * hd + d] - m);
        float acc = 0.0f;
        for (unsigned int r = 0; r < ratio; ++r) {
            const float e = expf(score[((size_t)g * ratio + r) * hd + d] - m);
            acc += kv[((size_t)g * ratio + r) * hd + d] * (e / sum);
        }
        out[(size_t)g * hd + d] = av_bf16r(acc);
    }
}

// ── indexer scores: score[t, p] = bf16(sum_h bf16(relu(bf16(q[t,h] . k[p])) * w[t,h])) ──
// q: [T, nh, ihd] bf16, k: [width, ihd] bf16, w: [T, nh] bf16 (already scaled),
// out: [T, width] f32. Grid: (ceil(width / 256), T). Block: 256, one (t, p) per thread.
extern "C" __global__ void attn_v41_index_score(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ w, float* __restrict__ out,
    const unsigned int width, const unsigned int nh, const unsigned int ihd) {
    const unsigned int t = blockIdx.y;
    const unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= width) return;
    const __nv_bfloat16* kr = k + (size_t)p * ihd;
    float acc = 0.0f;
    for (unsigned int h = 0; h < nh; ++h) {
        const __nv_bfloat16* qv = q + ((size_t)t * nh + h) * ihd;
        float dot = 0.0f;
        for (unsigned int d = 0; d < ihd; ++d) dot += __bfloat162float(qv[d]) * __bfloat162float(kr[d]);
        dot = av_bf16r(dot);
        acc += av_bf16r(fmaxf(dot, 0.0f) * __bfloat162float(w[(size_t)t * nh + h]));
    }
    out[(size_t)t * width + p] = av_bf16r(acc);
}

// ── sparse attention with the sink ───────────────────────────────────────────
// q: [T, nh, hd] bf16. Rows come from two sources: index i < split reads
// rows_a[i], else rows_b[i - split] (window rows, then the compressed cache).
// idx: [T, topk] i32, -1 = absent. sink: [nh] f32. o: [T, nh, hd] bf16.
// Grid: (T, nh). Block: 256. Scores in shared memory (topk <= AV_MAX_TOPK).
#define AV_MAX_TOPK 2048
extern "C" __global__ void attn_v41_sparse_attn(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ rows_a,
    const __nv_bfloat16* __restrict__ rows_b, const unsigned int split,
    const int* __restrict__ idx, const float* __restrict__ sink,
    __nv_bfloat16* __restrict__ o, const unsigned int nh, const unsigned int hd,
    const unsigned int topk, const float scale) {
    const unsigned int t = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % 32;
    const unsigned int warp = tid / 32;
    const unsigned int nwarps = blockDim.x / 32;
    __shared__ float sc[AV_MAX_TOPK];
    __shared__ float red[AV_BLOCK];
    const __nv_bfloat16* qv = q + ((size_t)t * nh + h) * hd;
    const int* ids = idx + (size_t)t * topk;

    // one warp per candidate row: the dot product over hd
    for (unsigned int j = warp; j < topk; j += nwarps) {
        const int i = ids[j];
        float s = -CUDART_INF_F;
        if (i >= 0) {
            const __nv_bfloat16* kr = (unsigned int)i < split ? rows_a + (size_t)i * hd : rows_b + (size_t)((unsigned int)i - split) * hd;
            float acc = 0.0f;
            for (unsigned int d = lane; d < hd; d += 32) acc += __bfloat162float(qv[d]) * __bfloat162float(kr[d]);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
            s = acc * scale;
        }
        if (lane == 0) sc[j] = s;
    }
    __syncthreads();
    // softmax over the scores plus the sink (denominator only)
    float m = sink[h];
    for (unsigned int j = tid; j < topk; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = av_block_max(m, red);
    float den = 0.0f;
    for (unsigned int j = tid; j < topk; j += blockDim.x) {
        const float e = sc[j] == -CUDART_INF_F ? 0.0f : expf(sc[j] - m);
        sc[j] = e;
        den += e;
    }
    den = av_block_sum(den, red) + expf(sink[h] - m);
    __syncthreads();
    // o[d] = sum_j p_j * row_j[d]
    __nv_bfloat16* ov = o + ((size_t)t * nh + h) * hd;
    for (unsigned int d = tid; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int j = 0; j < topk; ++j) {
            const int i = ids[j];
            if (i < 0) continue;
            const __nv_bfloat16* kr = (unsigned int)i < split ? rows_a + (size_t)i * hd : rows_b + (size_t)((unsigned int)i - split) * hd;
            acc += (sc[j] / den) * __bfloat162float(kr[d]);
        }
        ov[d] = __float2bfloat16(acc);
    }
}

// ── column slices for the grouped wo_a: out[t, 0..w) = in[t, col0 + 0..w) ────
// in rows are `in_stride` wide, out rows `w` wide. Grid: (T). Block: 256.
extern "C" __global__ void attn_v41_slice_cols(
    const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
    const unsigned int in_stride, const unsigned int col0, const unsigned int w) {
    const unsigned int t = blockIdx.x;
    for (unsigned int c = threadIdx.x; c < w; c += blockDim.x)
        out[(size_t)t * w + c] = in[(size_t)t * in_stride + col0 + c];
}

// out[t, col0 + 0..w) = in[t, 0..w), out rows `out_stride` wide.
extern "C" __global__ void attn_v41_scatter_cols(
    const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
    const unsigned int out_stride, const unsigned int col0, const unsigned int w) {
    const unsigned int t = blockIdx.x;
    for (unsigned int c = threadIdx.x; c < w; c += blockDim.x)
        out[(size_t)t * out_stride + col0 + c] = in[(size_t)t * w + c];
}

// out[i] = bf16(in[i] * s); `in` is bf16 so the product is what the reference's
// `to_bf16_rne(v * wscale)` computes. Grid: ceil(n / 256). Block: 256.
extern "C" __global__ void attn_v41_scale_bf16(
    const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
    const unsigned int n, const float s) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(in[i]) * s);
}

// ── the window ring write for the graph-captured decode step ─────────────────
// ring[(pos[0] % win) * row + c] = src[c]. The eager step does this as a
// device copy at a host-computed slot; a captured graph bakes its pointers,
// so the slot has to come from device memory. Bit-identical to the copy.
// Grid: (1). Block: 256.
extern "C" __global__ void attn_v41_ring_put(
    const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ ring,
    const int* __restrict__ pos, const unsigned int win, const unsigned int row) {
    const unsigned int slot = ((unsigned int)pos[0]) % win;
    __nv_bfloat16* dst = ring + (size_t)slot * row;
    for (unsigned int c = threadIdx.x; c < row; c += blockDim.x) dst[c] = src[c];
}
