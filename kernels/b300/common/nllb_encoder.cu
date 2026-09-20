// SPDX-License-Identifier: AGPL-3.0-only
//
// Self-contained fp32 CUDA kernels for the NLLB-200 / M2M-100 encoder forward
// pass (milestone-1 GPU PoC). Deliberately naive (correctness-first, tiny
// sequence lengths) and kept in `common/` so the module `"nllb_encoder"` is
// present in every model's PTX set. All math is fp32 to match the
// bit-faithful CPU reference (spark-nllb) so the encoder checksum validates
// tightly rather than "approximately".
//
// Weight layout is HuggingFace `nn.Linear`: weight is [out=N, in=K] row-major,
// consumed transposed (C[m,n] = bias[n] + Σ_k A[m,k]·W[n,k]).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

// out[tok, :] = table[ids[tok], :]   (embedding gather, fp32)
extern "C" __global__ void nllb_embed(
    const unsigned int* __restrict__ ids,
    const float* __restrict__ table,
    float* __restrict__ out,
    unsigned int d) {
    unsigned int tok = blockIdx.x;
    unsigned long long id = ids[tok];
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) {
        out[(unsigned long long)tok * d + i] = table[id * d + i];
    }
}

// x[i] *= s
extern "C" __global__ void nllb_scale_inplace(float* __restrict__ x, unsigned int n, float s) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// dst[i] += src[i]
extern "C" __global__ void nllb_add_inplace(
    float* __restrict__ dst, const float* __restrict__ src, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// x[i] = max(x[i], 0)
extern "C" __global__ void nllb_relu_inplace(float* __restrict__ x, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(x[i], 0.0f);
}

// LayerNorm over the last dim, affine (weight/bias), in place.
// One block per row; blockDim.x MUST be a power of two.
extern "C" __global__ void nllb_layernorm(
    float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    unsigned int rows, unsigned int dim, float eps) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sm[];
    unsigned int tid = threadIdx.x;
    float* rowp = x + (unsigned long long)row * dim;

    float local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) local += rowp[i];
    sm[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float mean = sm[0] / dim;
    __syncthreads();

    local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        float dv = rowp[i] - mean;
        local += dv * dv;
    }
    sm[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sm[0] / dim + eps);
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        rowp[i] = (rowp[i] - mean) * inv * w[i] + b[i];
    }
}

// C[M,N] = A[M,K] @ W[N,K]^T + bias[N]   (bias may be null)
extern "C" __global__ void nllb_linear(
    const float* __restrict__ a, const float* __restrict__ w, const float* __restrict__ bias,
    float* __restrict__ c, unsigned int M, unsigned int N, unsigned int K) {
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    const float* arow = a + (unsigned long long)m * K;
    const float* wrow = w + (unsigned long long)n * K;
    float acc = bias ? bias[n] : 0.0f;
    for (unsigned int k = 0; k < K; k++) acc += arow[k] * wrow[k];
    c[(unsigned long long)m * N + n] = acc;
}

// General multi-head SDPA with separate query/key lengths and optional causal
// masking. q:[tq, H*D], k/v:[tk, H*D], out:[tq, H*D] (heads interleaved on the
// feature axis). One block per (query, head); blockDim.x == D (power of two).
// causal!=0 → query i attends keys 0..=i (decoder self-attn); causal==0 →
// query attends all tk keys (encoder self-attn / decoder cross-attn).
// shared mem = (tk + D) floats.
extern "C" __global__ void nllb_attn_kv(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, unsigned int tq, unsigned int tk, unsigned int H, unsigned int D,
    float scale, unsigned int causal) {
    unsigned int qh = blockIdx.x;
    unsigned int query = qh / H;
    unsigned int head = qh % H;
    unsigned int tid = threadIdx.x;
    extern __shared__ float sh2[];
    float* scores = sh2;
    float* red = sh2 + tk;
    unsigned int dmodel = H * D;
    unsigned long long bq = (unsigned long long)query * dmodel + (unsigned long long)head * D;
    unsigned int kmax = causal ? (query + 1) : tk;

    for (unsigned int j = 0; j < kmax; j++) {
        unsigned long long bk = (unsigned long long)j * dmodel + (unsigned long long)head * D;
        red[tid] = q[bq + tid] * k[bk + tid];
        __syncthreads();
        for (unsigned int s = D / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) scores[j] = red[0] * scale;
        __syncthreads();
    }
    if (tid == 0) {
        float m = -1e30f;
        for (unsigned int j = 0; j < kmax; j++) m = fmaxf(m, scores[j]);
        float su = 0.0f;
        for (unsigned int j = 0; j < kmax; j++) {
            scores[j] = expf(scores[j] - m);
            su += scores[j];
        }
        for (unsigned int j = 0; j < kmax; j++) scores[j] /= su;
    }
    __syncthreads();
    float acc = 0.0f;
    for (unsigned int j = 0; j < kmax; j++) {
        acc += scores[j] * v[(unsigned long long)j * dmodel + (unsigned long long)head * D + tid];
    }
    out[bq + tid] = acc;
}

// Dense non-causal multi-head SDPA. q/k/v/out are [seq, H*D] row-major
// (heads interleaved on the feature axis). One block per (query, head);
// blockDim.x == D (power of two). scale applied to the logits.
extern "C" __global__ void nllb_attention(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, unsigned int seq, unsigned int H, unsigned int D, float scale) {
    unsigned int qh = blockIdx.x;
    unsigned int query = qh / H;
    unsigned int head = qh % H;
    unsigned int tid = threadIdx.x; // 0..D-1
    extern __shared__ float sh[];   // scores[seq] then red[D]
    float* scores = sh;
    float* red = sh + seq;
    unsigned int dmodel = H * D;
    unsigned long long bq = (unsigned long long)query * dmodel + (unsigned long long)head * D;

    for (unsigned int j = 0; j < seq; j++) {
        unsigned long long bk = (unsigned long long)j * dmodel + (unsigned long long)head * D;
        red[tid] = q[bq + tid] * k[bk + tid];
        __syncthreads();
        for (unsigned int s = D / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) scores[j] = red[0] * scale;
        __syncthreads();
    }
    if (tid == 0) {
        float m = -1e30f;
        for (unsigned int j = 0; j < seq; j++) m = fmaxf(m, scores[j]);
        float su = 0.0f;
        for (unsigned int j = 0; j < seq; j++) {
            scores[j] = expf(scores[j] - m);
            su += scores[j];
        }
        for (unsigned int j = 0; j < seq; j++) scores[j] /= su;
    }
    __syncthreads();
    float acc = 0.0f;
    for (unsigned int j = 0; j < seq; j++) {
        acc += scores[j] * v[(unsigned long long)j * dmodel + (unsigned long long)head * D + tid];
    }
    out[bq + tid] = acc;
}

// ─────────────────────────────────────────────────────────────────────────
// bf16 variants (milestone-4 tensor-core pipeline). Activations + weights are
// bf16; the heavy GEMMs use the shared tensor-core `gemm` module
// (dense_gemm_bf16_pipelined). These ops store bf16 but accumulate in f32.
// ─────────────────────────────────────────────────────────────────────────

extern "C" __global__ void nllb_embed_bf16(
    const unsigned int* __restrict__ ids, const __nv_bfloat16* __restrict__ table,
    __nv_bfloat16* __restrict__ out, unsigned int d) {
    unsigned int tok = blockIdx.x;
    unsigned long long id = ids[tok];
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x)
        out[(unsigned long long)tok * d + i] = table[id * d + i];
}

extern "C" __global__ void nllb_scale_bf16(__nv_bfloat16* __restrict__ x, unsigned int n, float s) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16(__bfloat162float(x[i]) * s);
}

extern "C" __global__ void nllb_add_bf16(
    __nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ src, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + __bfloat162float(src[i]));
}

// c[m,n] += bias[n]  (row-broadcast, after a bias-less GEMM)
extern "C" __global__ void nllb_bias_bf16(
    __nv_bfloat16* __restrict__ c, const __nv_bfloat16* __restrict__ bias,
    unsigned int M, unsigned int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    unsigned int n = idx % N;
    c[idx] = __float2bfloat16(__bfloat162float(c[idx]) + __bfloat162float(bias[n]));
}

extern "C" __global__ void nllb_relu_bf16(__nv_bfloat16* __restrict__ x, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16(fmaxf(__bfloat162float(x[i]), 0.0f));
}

extern "C" __global__ void nllb_layernorm_bf16(
    __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ b, unsigned int rows, unsigned int dim, float eps) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sm[];
    unsigned int tid = threadIdx.x;
    __nv_bfloat16* rowp = x + (unsigned long long)row * dim;
    float local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) local += __bfloat162float(rowp[i]);
    sm[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float mean = sm[0] / dim;
    __syncthreads();
    local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        float dv = __bfloat162float(rowp[i]) - mean;
        local += dv * dv;
    }
    sm[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sm[0] / dim + eps);
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        float v = (__bfloat162float(rowp[i]) - mean) * inv * __bfloat162float(w[i]) + __bfloat162float(b[i]);
        rowp[i] = __float2bfloat16(v);
    }
}

// Out-of-place layer norm: identical arithmetic to nllb_layernorm_bf16 (same
// tree reduction, mean, rsqrt, and per-element transform) but reads `in` and
// writes `out`. Lets the beam decode fuse the pre-LN `dh->normed` copy away by
// normalizing dh directly into the LN scratch. Requires in != out (callers only
// ever pass distinct buffers).
extern "C" __global__ void nllb_layernorm_oop_bf16(
    const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ b,
    unsigned int rows, unsigned int dim, float eps) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sm[];
    unsigned int tid = threadIdx.x;
    const __nv_bfloat16* inp = in + (unsigned long long)row * dim;
    __nv_bfloat16* outp = out + (unsigned long long)row * dim;
    float local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) local += __bfloat162float(inp[i]);
    sm[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float mean = sm[0] / dim;
    __syncthreads();
    local = 0.0f;
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        float dv = __bfloat162float(inp[i]) - mean;
        local += dv * dv;
    }
    sm[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sm[0] / dim + eps);
    for (unsigned int i = tid; i < dim; i += blockDim.x) {
        float v = (__bfloat162float(inp[i]) - mean) * inv * __bfloat162float(w[i]) + __bfloat162float(b[i]);
        outp[i] = __float2bfloat16(v);
    }
}

extern "C" __global__ void nllb_attn_kv_bf16(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ out,
    unsigned int tq, unsigned int tk, unsigned int H, unsigned int D, float scale, unsigned int causal) {
    unsigned int qh = blockIdx.x;
    unsigned int query = qh / H;
    unsigned int head = qh % H;
    unsigned int tid = threadIdx.x;
    extern __shared__ float sh2b[];
    float* scores = sh2b;
    float* red = sh2b + tk;
    unsigned int dmodel = H * D;
    unsigned long long bq = (unsigned long long)query * dmodel + (unsigned long long)head * D;
    unsigned int kmax = causal ? (query + 1) : tk;
    for (unsigned int j = 0; j < kmax; j++) {
        unsigned long long bk = (unsigned long long)j * dmodel + (unsigned long long)head * D;
        red[tid] = __bfloat162float(q[bq + tid]) * __bfloat162float(k[bk + tid]);
        __syncthreads();
        for (unsigned int s = D / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) scores[j] = red[0] * scale;
        __syncthreads();
    }
    if (tid == 0) {
        float m = -1e30f;
        for (unsigned int j = 0; j < kmax; j++) m = fmaxf(m, scores[j]);
        float su = 0.0f;
        for (unsigned int j = 0; j < kmax; j++) {
            scores[j] = expf(scores[j] - m);
            su += scores[j];
        }
        for (unsigned int j = 0; j < kmax; j++) scores[j] /= su;
    }
    __syncthreads();
    float acc = 0.0f;
    for (unsigned int j = 0; j < kmax; j++)
        acc += scores[j] * __bfloat162float(v[(unsigned long long)j * dmodel + (unsigned long long)head * D + tid]);
    out[bq + tid] = __float2bfloat16(acc);
}

// bf16 GEMV for single-token decode: y[N] = W[N,K] @ x[K] + bias[N] (bias may
// be null). One warp per output row; f32 accumulation → bit-compatible with the
// tensor-core GEMM's f32 accumulate. Right-sized for M=1 (the pipelined GEMM
// wastes 127/128 of its 128-row M-tile on a single token) and fuses the bias.
extern "C" __global__ void nllb_gemv_bf16(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ y,
    unsigned int N, unsigned int K) {
    unsigned int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned int lane = threadIdx.x & 31u;
    if (warp >= N) return;
    const __nv_bfloat16* wrow = W + (unsigned long long)warp * K;
    float acc = 0.0f;
    for (unsigned int k = lane; k < K; k += 32u)
        acc += __bfloat162float(x[k]) * __bfloat162float(wrow[k]);
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        if (bias) acc += __bfloat162float(bias[warp]);
        y[warp] = __float2bfloat16(acc);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Batched decode kernels (milestone-6: request-batching / beam-batching).
// Batch B sequences decode in lockstep; each has its own [B, stride, H*D]
// batch-major K/V cache. Elementwise/LayerNorm/embed kernels already batch
// (per-row / flat), and projections use the M=B tensor-core GEMM — only
// attention, argmax, cache-scatter and the broadcast pos-add are new.
// ─────────────────────────────────────────────────────────────────────────

// Batched single-query decode attention. q:[B,H*D], kc/vc:[B,stride,H*D]
// (batch-major), out:[B,H*D]. Batch element b attends its rows 0..tk[b]-1.
// grid=[B*H], block=[D]. shared = (D + max_tk) floats (red[D] then scores).
extern "C" __global__ void nllb_attn_bdecode(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out,
    unsigned int B, unsigned int stride, unsigned int group,
    const unsigned int* __restrict__ tk, unsigned int H, unsigned int D, float scale) {
    unsigned int bh = blockIdx.x;
    unsigned int b = bh / H;
    unsigned int head = bh % H;
    unsigned int tid = threadIdx.x;
    if (b >= B) return;
    unsigned int t = tk[b];
    unsigned int dmodel = H * D;
    extern __shared__ float shd[];
    float* red = shd;
    float* scores = shd + D;
    // `group` maps rows to the K/V cache slab: self-attn passes group=1 (slab=b,
    // per-row cache); grouped cross-attn passes group=B_per_request so rows of the
    // same request (b/group) share one padded [C,max_enc,d] cross slab.
    unsigned long long qbase = (unsigned long long)b * dmodel + (unsigned long long)head * D;
    unsigned long long cbase =
        (unsigned long long)(b / group) * stride * dmodel + (unsigned long long)head * D;
    for (unsigned int j = 0; j < t; j++) {
        red[tid] = __bfloat162float(q[qbase + tid]) *
                   __bfloat162float(kc[cbase + (unsigned long long)j * dmodel + tid]);
        __syncthreads();
        for (unsigned int s = D / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) scores[j] = red[0] * scale;
        __syncthreads();
    }
    if (tid == 0) {
        float m = -1e30f;
        for (unsigned int j = 0; j < t; j++) m = fmaxf(m, scores[j]);
        float su = 0.0f;
        for (unsigned int j = 0; j < t; j++) {
            scores[j] = expf(scores[j] - m);
            su += scores[j];
        }
        for (unsigned int j = 0; j < t; j++) scores[j] /= su;
    }
    __syncthreads();
    float acc = 0.0f;
    for (unsigned int j = 0; j < t; j++)
        acc += scores[j] * __bfloat162float(vc[cbase + (unsigned long long)j * dmodel + tid]);
    out[qbase + tid] = __float2bfloat16(acc);
}

// Write src[B,d] into a batch-major cache [B,stride,d] at row `pos`.
extern "C" __global__ void nllb_scatter_batched(
    const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
    unsigned int pos, unsigned int B, unsigned int stride, unsigned int d) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * d) return;
    unsigned int b = idx / d;
    unsigned int i = idx % d;
    dst[(unsigned long long)b * stride * d + (unsigned long long)pos * d + i] = src[idx];
}

// dst[B*d] += row[d] broadcast to every one of the B rows (positional add).
extern "C" __global__ void nllb_add_row_bf16(
    __nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ row,
    unsigned int n, unsigned int d) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        dst[idx] = __float2bfloat16(__bfloat162float(dst[idx]) + __bfloat162float(row[idx % d]));
}

// Batched argmax: out[b] = argmax_v logits[b, v]. grid=[B], block=[256],
// shared = block*(4+4) bytes.
extern "C" __global__ void nllb_argmax_batched(
    const __nv_bfloat16* __restrict__ logits, unsigned int* __restrict__ out,
    unsigned int B, unsigned int vocab) {
    unsigned int b = blockIdx.x;
    unsigned int tid = threadIdx.x;
    extern __shared__ char smem[];
    float* sval = (float*)smem;
    unsigned int* sidx = (unsigned int*)(sval + blockDim.x);
    const __nv_bfloat16* row = logits + (unsigned long long)b * vocab;
    float best = -1e30f;
    unsigned int bi = 0;
    for (unsigned int v = tid; v < vocab; v += blockDim.x) {
        float x = __bfloat162float(row[v]);
        if (x > best) {
            best = x;
            bi = v;
        }
    }
    sval[tid] = best;
    sidx[tid] = bi;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sval[tid + s] > sval[tid]) {
            sval[tid] = sval[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out[b] = sidx[0];
}

// Reorder batch-major caches by a beam permutation: dst[i] = src[perm[i]] for
// rows 0..used (all layers call this per step). Used by beam-batching to
// materialise each child beam's cache from its parent slot (HF _reorder_cache).
extern "C" __global__ void nllb_gather_batched(
    const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
    const unsigned int* __restrict__ perm, unsigned int B, unsigned int used,
    unsigned int stride, unsigned int d) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)B * used * d;
    if (idx >= total) return;
    unsigned int i = (unsigned int)(idx / ((unsigned long long)used * d)); // dst batch slot
    unsigned long long rem = idx % ((unsigned long long)used * d);         // row*d + elem
    unsigned int src_b = perm[i];
    dst[(unsigned long long)i * stride * d + rem] =
        src[(unsigned long long)src_b * stride * d + rem];
}

// Batched beam top-k + logsumexp (Phase-d on-device candidate reduction). For
// each row b of the [B,vocab] bf16 logits, compute the log-sum-exp over the FULL
// vocab (lse_out[b]) and the K = 2*num_beams largest (value, token) pairs
// (val_out[b*K+j], idx_out[b*K+j], descending; ties broken by lower token id to
// match the host `top_k`). Replaces the per-step full-[B,vocab] D2H + host
// logsumexp/top_k in the beam decode loop: the D2H shrinks from B*vocab*2 bytes
// to B*K*(4+4) bytes. grid=[B], block=[128], dynamic shared = 128*K*(4+4).
#define NLLB_TOPK_MAX 32
extern "C" __global__ void nllb_beam_topk(
    const __nv_bfloat16* __restrict__ logits, float* __restrict__ lse_out,
    float* __restrict__ val_out, unsigned int* __restrict__ idx_out,
    unsigned int B, unsigned int vocab, unsigned int K) {
    unsigned int b = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int nt = blockDim.x; // == 128
    const __nv_bfloat16* row = logits + (unsigned long long)b * vocab;

    // Per-thread local top-K (descending) + streaming logsumexp state (m, s).
    float lv[NLLB_TOPK_MAX];
    unsigned int li[NLLB_TOPK_MAX];
    for (unsigned int j = 0; j < K; ++j) {
        lv[j] = -1e30f;
        li[j] = 0u;
    }
    float m = -1e30f, s = 0.0f;
    for (unsigned int v = tid; v < vocab; v += nt) {
        float x = __bfloat162float(row[v]);
        if (x > m) {
            s = s * __expf(m - x) + 1.0f;
            m = x;
        } else {
            s += __expf(x - m);
        }
        // Insert into the local top-K (strict >, so equal values keep the
        // lower token id already held; scan is ascending in v within a thread).
        if (x > lv[K - 1]) {
            int j = (int)K - 1;
            while (j > 0 && lv[j - 1] < x) {
                lv[j] = lv[j - 1];
                li[j] = li[j - 1];
                --j;
            }
            lv[j] = x;
            li[j] = v;
        }
    }

    extern __shared__ char smem[];
    float* cval = (float*)smem;                          // nt*K candidate values
    unsigned int* cidx = (unsigned int*)(cval + nt * K); // nt*K candidate ids
    for (unsigned int j = 0; j < K; ++j) {
        cval[tid * K + j] = lv[j];
        cidx[tid * K + j] = li[j];
    }
    __shared__ float sm[128];
    __shared__ float ss[128];
    sm[tid] = m;
    ss[tid] = s;
    __syncthreads();

    if (tid == 0) {
        // Combine per-thread logsumexp partials: lse = M + log(Σ s_t·exp(m_t−M)).
        float M = -1e30f;
        for (unsigned int t = 0; t < nt; ++t)
            if (sm[t] > M) M = sm[t];
        float S = 0.0f;
        for (unsigned int t = 0; t < nt; ++t) S += ss[t] * __expf(sm[t] - M);
        lse_out[b] = M + logf(S);
        // Extract the global top-K from the nt*K candidate pool (small, serial).
        unsigned int cand = nt * K;
        for (unsigned int r = 0; r < K; ++r) {
            float best = -1e30f;
            unsigned int bpos = 0, bid = 0xFFFFFFFFu;
            for (unsigned int c = 0; c < cand; ++c) {
                float cv = cval[c];
                if (cv > best || (cv == best && cidx[c] < bid)) {
                    best = cv;
                    bid = cidx[c];
                    bpos = c;
                }
            }
            val_out[b * K + r] = best;
            idx_out[b * K + r] = cidx[bpos];
            cval[bpos] = -1e30f; // mask the extracted candidate
        }
    }
}
