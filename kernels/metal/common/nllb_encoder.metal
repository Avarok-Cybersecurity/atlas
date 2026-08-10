// SPDX-License-Identifier: AGPL-3.0-only
//
// Self-contained fp32 Metal kernels for the NLLB-200 / M2M-100 GPU PoC.
// Mirrors the CUDA `nllb_encoder` module used by PR #51's examples: naive,
// correctness-first fp32 embedding, LayerNorm, Linear, ReLU, and dense SDPA.
//
// Weight layout is HuggingFace `nn.Linear`: weight is [out=N, in=K] row-major,
// consumed transposed (C[m,n] = bias[n] + sum_k A[m,k] * W[n,k]).

#include <metal_stdlib>
using namespace metal;

kernel void nllb_embed(
    device const uint *ids [[buffer(0)]],
    device const float *table [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &d [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint tok = tgid.x;
    uint tid = tid3.x;
    ulong id = ids[tok];
    for (uint i = tid; i < d; i += 256) {
        out[(ulong)tok * d + i] = table[id * d + i];
    }
}

kernel void nllb_scale_inplace(
    device float *x [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    constant float &s [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        x[gid] *= s;
    }
}

kernel void nllb_add_inplace(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        dst[gid] += src[gid];
    }
}

kernel void nllb_relu_inplace(
    device float *x [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        x[gid] = fmax(x[gid], 0.0f);
    }
}

kernel void nllb_layernorm(
    device float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *b [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint row = tgid.x;
    if (row >= rows) {
        return;
    }
    uint tid = tid3.x;
    threadgroup float sm[256];
    device float *rowp = x + (ulong)row * dim;

    float local = 0.0f;
    for (uint i = tid; i < dim; i += 256) {
        local += rowp[i];
    }
    sm[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sm[tid] += sm[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sm[0] / float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    local = 0.0f;
    for (uint i = tid; i < dim; i += 256) {
        float dv = rowp[i] - mean;
        local += dv * dv;
    }
    sm[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sm[tid] += sm[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(sm[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += 256) {
        rowp[i] = (rowp[i] - mean) * inv * w[i] + b[i];
    }
}

kernel void nllb_linear(
    device const float *a [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *c [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint n = gid.x;
    uint m = gid.y;
    if (m >= M || n >= N) {
        return;
    }
    device const float *arow = a + (ulong)m * K;
    device const float *wrow = w + (ulong)n * K;
    float acc = bias[n];
    for (uint k = 0; k < K; k++) {
        acc += arow[k] * wrow[k];
    }
    c[(ulong)m * N + n] = acc;
}

kernel void nllb_linear_no_bias(
    device const float *a [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint n = gid.x;
    uint m = gid.y;
    if (m >= M || n >= N) {
        return;
    }
    device const float *arow = a + (ulong)m * K;
    device const float *wrow = w + (ulong)n * K;
    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += arow[k] * wrow[k];
    }
    c[(ulong)m * N + n] = acc;
}

kernel void nllb_attn_kv(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &tq [[buffer(4)]],
    constant uint &tk [[buffer(5)]],
    constant uint &H [[buffer(6)]],
    constant uint &D [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    constant uint &causal [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint qh = tgid.x;
    uint query = qh / H;
    uint head = qh % H;
    uint tid = tid3.x;
    threadgroup float scratch[256];
    threadgroup float *scores = scratch;
    threadgroup float *red = scratch + tk;
    uint dmodel = H * D;
    ulong bq = (ulong)query * dmodel + (ulong)head * D;
    uint kmax = causal != 0 ? (query + 1) : tk;

    for (uint j = 0; j < kmax; j++) {
        ulong bk = (ulong)j * dmodel + (ulong)head * D;
        red[tid] = q[bq + tid] * k[bk + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                red[tid] += red[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            scores[j] = red[0] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float m = -1.0e30f;
        for (uint j = 0; j < kmax; j++) {
            m = fmax(m, scores[j]);
        }
        float sum = 0.0f;
        for (uint j = 0; j < kmax; j++) {
            scores[j] = exp(scores[j] - m);
            sum += scores[j];
        }
        for (uint j = 0; j < kmax; j++) {
            scores[j] /= sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint j = 0; j < kmax; j++) {
        acc += scores[j] * v[(ulong)j * dmodel + (ulong)head * D + tid];
    }
    out[bq + tid] = acc;
}

kernel void nllb_attention(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &seq [[buffer(4)]],
    constant uint &H [[buffer(5)]],
    constant uint &D [[buffer(6)]],
    constant float &scale [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint qh = tgid.x;
    uint query = qh / H;
    uint head = qh % H;
    uint tid = tid3.x;
    threadgroup float scratch[256];
    threadgroup float *scores = scratch;
    threadgroup float *red = scratch + seq;
    uint dmodel = H * D;
    ulong bq = (ulong)query * dmodel + (ulong)head * D;

    for (uint j = 0; j < seq; j++) {
        ulong bk = (ulong)j * dmodel + (ulong)head * D;
        red[tid] = q[bq + tid] * k[bk + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                red[tid] += red[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            scores[j] = red[0] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float m = -1.0e30f;
        for (uint j = 0; j < seq; j++) {
            m = fmax(m, scores[j]);
        }
        float sum = 0.0f;
        for (uint j = 0; j < seq; j++) {
            scores[j] = exp(scores[j] - m);
            sum += scores[j];
        }
        for (uint j = 0; j < seq; j++) {
            scores[j] /= sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint j = 0; j < seq; j++) {
        acc += scores[j] * v[(ulong)j * dmodel + (ulong)head * D + tid];
    }
    out[bq + tid] = acc;
}

kernel void nllb_embed_bf16(
    device const uint *ids [[buffer(0)]],
    device const bfloat *table [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint &d [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint tok = tgid.x;
    uint tid = tid3.x;
    ulong id = ids[tok];
    for (uint i = tid; i < d; i += 256) {
        out[(ulong)tok * d + i] = table[id * d + i];
    }
}

kernel void nllb_scale_bf16(
    device bfloat *x [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    constant float &s [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        x[gid] = bfloat(float(x[gid]) * s);
    }
}

kernel void nllb_add_bf16(
    device bfloat *dst [[buffer(0)]],
    device const bfloat *src [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        dst[gid] = bfloat(float(dst[gid]) + float(src[gid]));
    }
}

kernel void nllb_bias_bf16(
    device bfloat *c [[buffer(0)]],
    device const bfloat *bias [[buffer(1)]],
    constant uint &M [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)M * N;
    if (gid < total) {
        c[gid] = bfloat(float(c[gid]) + float(bias[gid % N]));
    }
}

kernel void nllb_relu_bf16(
    device bfloat *x [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        x[gid] = bfloat(fmax(float(x[gid]), 0.0f));
    }
}

kernel void nllb_layernorm_bf16(
    device bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device const bfloat *b [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint row = tgid.x;
    if (row >= rows) {
        return;
    }
    uint tid = tid3.x;
    threadgroup float sm[256];
    device bfloat *rowp = x + (ulong)row * dim;

    float local = 0.0f;
    for (uint i = tid; i < dim; i += 256) {
        local += float(rowp[i]);
    }
    sm[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sm[tid] += sm[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sm[0] / float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    local = 0.0f;
    for (uint i = tid; i < dim; i += 256) {
        float dv = float(rowp[i]) - mean;
        local += dv * dv;
    }
    sm[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sm[tid] += sm[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(sm[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += 256) {
        float v = (float(rowp[i]) - mean) * inv * float(w[i]) + float(b[i]);
        rowp[i] = bfloat(v);
    }
}

kernel void nllb_linear_bf16(
    device const bfloat *a [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device bfloat *c [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint n = gid.x;
    uint m = gid.y;
    if (m >= M || n >= N) {
        return;
    }
    device const bfloat *arow = a + (ulong)m * K;
    device const bfloat *wrow = w + (ulong)n * K;
    float acc = float(bias[n]);
    for (uint k = 0; k < K; k++) {
        acc += float(arow[k]) * float(wrow[k]);
    }
    c[(ulong)m * N + n] = bfloat(acc);
}

kernel void nllb_linear_no_bias_bf16(
    device const bfloat *a [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device bfloat *c [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint n = gid.x;
    uint m = gid.y;
    if (m >= M || n >= N) {
        return;
    }
    device const bfloat *arow = a + (ulong)m * K;
    device const bfloat *wrow = w + (ulong)n * K;
    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += float(arow[k]) * float(wrow[k]);
    }
    c[(ulong)m * N + n] = bfloat(acc);
}

kernel void nllb_add_position_bf16(
    device bfloat *dst [[buffer(0)]],
    device const bfloat *pos [[buffer(1)]],
    constant uint &batch [[buffer(2)]],
    constant uint &D [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)batch * D;
    if (gid < total) {
        dst[gid] = bfloat(float(dst[gid]) + float(pos[gid % D]));
    }
}

kernel void nllb_add_row_bf16(
    device bfloat *dst [[buffer(0)]],
    device const bfloat *row [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &d [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        dst[gid] = bfloat(float(dst[gid]) + float(row[gid % d]));
    }
}

kernel void nllb_cache_write_bf16(
    device const bfloat *src [[buffer(0)]],
    device bfloat *cache [[buffer(1)]],
    constant uint &pos [[buffer(2)]],
    constant uint &batch [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &D [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)batch * D;
    if (gid >= total) {
        return;
    }
    uint b = gid / D;
    uint d = gid % D;
    cache[((ulong)b * rows + pos) * D + d] = src[gid];
}

kernel void nllb_scatter_batched(
    device const bfloat *src [[buffer(0)]],
    device bfloat *dst [[buffer(1)]],
    constant uint &pos [[buffer(2)]],
    constant uint &B [[buffer(3)]],
    constant uint &stride [[buffer(4)]],
    constant uint &d [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)B * d;
    if (gid >= total) {
        return;
    }
    uint b = gid / d;
    uint i = gid % d;
    dst[((ulong)b * stride + pos) * d + i] = src[gid];
}

kernel void nllb_gather_batched(
    device const bfloat *src [[buffer(0)]],
    device bfloat *dst [[buffer(1)]],
    device const uint *perm [[buffer(2)]],
    constant uint &B [[buffer(3)]],
    constant uint &used [[buffer(4)]],
    constant uint &stride [[buffer(5)]],
    constant uint &d [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)B * used * d;
    if (gid >= total) {
        return;
    }
    uint i = gid % d;
    uint t = (gid / d) % used;
    uint b = gid / (used * d);
    uint parent = perm[b];
    dst[((ulong)b * stride + t) * d + i] = src[((ulong)parent * stride + t) * d + i];
}

kernel void nllb_gemv_bf16(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device bfloat *y [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= N) {
        return;
    }
    threadgroup float partial[32];
    float acc = 0.0f;
    for (uint k = tid; k < K; k += tg_size) {
        acc += float(x[k]) * float(w[(ulong)row * K + k]);
    }
    float simd_acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) {
            y[row] = bfloat(v + float(bias[row]));
        }
    }
}

kernel void nllb_gemv_bf16_no_bias(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device bfloat *y [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= N) {
        return;
    }
    threadgroup float partial[32];
    float acc = 0.0f;
    for (uint k = tid; k < K; k += tg_size) {
        acc += float(x[k]) * float(w[(ulong)row * K + k]);
    }
    float simd_acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) {
            y[row] = bfloat(v);
        }
    }
}

kernel void nllb_gemv_batched_bf16(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device bfloat *y [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tg_size3 [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint n = group.x;
    uint m = group.y;
    if (m >= M || n >= N) {
        return;
    }
    threadgroup float partial[32];
    uint tid = tid3.x;
    uint tg_size = tg_size3.x;
    device const bfloat *xrow = x + (ulong)m * K;
    device const bfloat *wrow = w + (ulong)n * K;
    float acc = 0.0f;
    for (uint k = tid; k < K; k += tg_size) {
        acc += float(xrow[k]) * float(wrow[k]);
    }
    float simd_acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) {
            y[(ulong)m * N + n] = bfloat(v + float(bias[n]));
        }
    }
}

kernel void nllb_attn_kv_bf16(
    device const bfloat *q [[buffer(0)]],
    device const bfloat *k [[buffer(1)]],
    device const bfloat *v [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant uint &tq [[buffer(4)]],
    constant uint &tk [[buffer(5)]],
    constant uint &H [[buffer(6)]],
    constant uint &D [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    constant uint &causal [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint qh = tgid.x;
    uint query = qh / H;
    uint head = qh % H;
    uint tid = tid3.x;
    threadgroup float scratch[256];
    threadgroup float *scores = scratch;
    threadgroup float *red = scratch + tk;
    uint dmodel = H * D;
    ulong bq = (ulong)query * dmodel + (ulong)head * D;
    uint kmax = causal != 0 ? (query + 1) : tk;

    for (uint j = 0; j < kmax; j++) {
        ulong bk = (ulong)j * dmodel + (ulong)head * D;
        red[tid] = float(q[bq + tid]) * float(k[bk + tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                red[tid] += red[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            scores[j] = red[0] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float m = -1.0e30f;
        for (uint j = 0; j < kmax; j++) {
            m = fmax(m, scores[j]);
        }
        float sum = 0.0f;
        for (uint j = 0; j < kmax; j++) {
            scores[j] = exp(scores[j] - m);
            sum += scores[j];
        }
        for (uint j = 0; j < kmax; j++) {
            scores[j] /= sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint j = 0; j < kmax; j++) {
        acc += scores[j] * float(v[(ulong)j * dmodel + (ulong)head * D + tid]);
    }
    out[bq + tid] = bfloat(acc);
}

kernel void nllb_attn_kv_batched_bf16(
    device const bfloat *q [[buffer(0)]],
    device const bfloat *k [[buffer(1)]],
    device const bfloat *v [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant uint &batch [[buffer(4)]],
    constant uint &tq [[buffer(5)]],
    constant uint &tk [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    constant uint &D [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    constant uint &causal [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint idx = tgid.x;
    uint head = idx % H;
    uint query = (idx / H) % tq;
    uint b = idx / (H * tq);
    if (b >= batch) {
        return;
    }
    uint tid = tid3.x;
    threadgroup float scratch[256];
    threadgroup float *scores = scratch;
    threadgroup float *red = scratch + tk;
    uint dmodel = H * D;
    ulong bq = ((ulong)b * tq + query) * dmodel + (ulong)head * D;
    uint kmax = causal != 0 ? (query + 1) : tk;

    for (uint j = 0; j < kmax; j++) {
        ulong bk = ((ulong)b * tk + j) * dmodel + (ulong)head * D;
        red[tid] = float(q[bq + tid]) * float(k[bk + tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                red[tid] += red[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            scores[j] = red[0] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float m = -1.0e30f;
        for (uint j = 0; j < kmax; j++) {
            m = fmax(m, scores[j]);
        }
        float sum = 0.0f;
        for (uint j = 0; j < kmax; j++) {
            scores[j] = exp(scores[j] - m);
            sum += scores[j];
        }
        for (uint j = 0; j < kmax; j++) {
            scores[j] /= sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint j = 0; j < kmax; j++) {
        acc += scores[j] * float(v[((ulong)b * tk + j) * dmodel + (ulong)head * D + tid]);
    }
    out[bq + tid] = bfloat(acc);
}

kernel void nllb_attn_bdecode(
    device const bfloat *q [[buffer(0)]],
    device const bfloat *kc [[buffer(1)]],
    device const bfloat *vc [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant uint &B [[buffer(4)]],
    constant uint &stride [[buffer(5)]],
    device const uint *tk [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    constant uint &D [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint bh = tgid.x;
    uint b = bh / H;
    uint head = bh % H;
    if (b >= B) {
        return;
    }
    uint tid = tid3.x;
    uint t = tk[b];
    threadgroup float scratch[256];
    threadgroup float *scores = scratch;
    threadgroup float *red = scratch + t;
    uint dmodel = H * D;
    ulong qbase = (ulong)b * dmodel + (ulong)head * D;
    ulong cbase = (ulong)b * stride * dmodel + (ulong)head * D;

    for (uint j = 0; j < t; j++) {
        red[tid] = float(q[qbase + tid]) * float(kc[cbase + (ulong)j * dmodel + tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = D >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                red[tid] += red[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            scores[j] = red[0] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float m = -1.0e30f;
        for (uint j = 0; j < t; j++) {
            m = fmax(m, scores[j]);
        }
        float sum = 0.0f;
        for (uint j = 0; j < t; j++) {
            scores[j] = exp(scores[j] - m);
            sum += scores[j];
        }
        for (uint j = 0; j < t; j++) {
            scores[j] /= sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint j = 0; j < t; j++) {
        acc += scores[j] * float(vc[cbase + (ulong)j * dmodel + tid]);
    }
    out[qbase + tid] = bfloat(acc);
}

kernel void nllb_argmax_batched(
    device const bfloat *logits [[buffer(0)]],
    device uint *out [[buffer(1)]],
    constant uint &B [[buffer(2)]],
    constant uint &vocab [[buffer(3)]],
    uint b [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    if (b >= B) {
        return;
    }
    threadgroup float partial_val[32];
    threadgroup uint partial_idx[32];
    device const bfloat *row = logits + (ulong)b * vocab;
    float best_val = -INFINITY;
    uint best_idx = 0;
    for (uint i = tid; i < vocab; i += tg_size) {
        float v = float(row[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other_val = simd_shuffle_xor(best_val, offset);
        uint other_idx = simd_shuffle_xor(best_idx, offset);
        bool other_wins = other_val > best_val ||
                          (other_val == best_val && other_idx < best_idx);
        best_val = other_wins ? other_val : best_val;
        best_idx = other_wins ? other_idx : best_idx;
    }
    if (simd_lane_id == 0) {
        partial_val[simd_group_id] = best_val;
        partial_idx[simd_group_id] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial_val[tid] : -INFINITY;
        uint i = (tid < num_simds) ? partial_idx[tid] : 0u;
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            float other_v = simd_shuffle_xor(v, offset);
            uint other_i = simd_shuffle_xor(i, offset);
            bool other_wins = other_v > v || (other_v == v && other_i < i);
            v = other_wins ? other_v : v;
            i = other_wins ? other_i : i;
        }
        if (tid == 0) {
            out[b] = i;
        }
    }
}

kernel void nllb_topk_lse_bf16(
    device const bfloat *logits [[buffer(0)]],
    device float *top_vals [[buffer(1)]],
    device uint *top_ids [[buffer(2)]],
    device float *lse_out [[buffer(3)]],
    constant uint &B [[buffer(4)]],
    constant uint &vocab [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    constexpr uint K = 10;
    if (row >= B) {
        return;
    }
    threadgroup float partial_max[256];
    threadgroup float partial_sum[256];
    threadgroup float local_vals[2560];
    threadgroup uint local_ids[2560];
    device const bfloat *base = logits + (ulong)row * vocab;

    float vals[K];
    uint ids[K];
    for (uint j = 0; j < K; j++) {
        vals[j] = -INFINITY;
        ids[j] = 0;
    }
    float local_max = -INFINITY;
    for (uint i = tid; i < vocab; i += tg_size) {
        float v = float(base[i]);
        local_max = fmax(local_max, v);
        if (v > vals[K - 1]) {
            vals[K - 1] = v;
            ids[K - 1] = i;
            for (uint j = K - 1; j > 0 && vals[j] > vals[j - 1]; j--) {
                float tv = vals[j - 1];
                uint ti = ids[j - 1];
                vals[j - 1] = vals[j];
                ids[j - 1] = ids[j];
                vals[j] = tv;
                ids[j] = ti;
            }
        }
    }
    partial_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial_max[tid] = fmax(partial_max[tid], partial_max[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = partial_max[0];
    float local_sum = 0.0f;
    for (uint i = tid; i < vocab; i += tg_size) {
        local_sum += exp(float(base[i]) - m);
    }
    partial_sum[tid] = local_sum;
    for (uint j = 0; j < K; j++) {
        local_vals[tid * K + j] = vals[j];
        local_ids[tid * K + j] = ids[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sum[tid] += partial_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float best_vals[K];
        uint best_ids[K];
        for (uint j = 0; j < K; j++) {
            best_vals[j] = -INFINITY;
            best_ids[j] = 0;
        }
        for (uint t = 0; t < tg_size; t++) {
            for (uint j = 0; j < K; j++) {
                float v = local_vals[t * K + j];
                uint id = local_ids[t * K + j];
                if (v > best_vals[K - 1]) {
                    best_vals[K - 1] = v;
                    best_ids[K - 1] = id;
                    for (uint p = K - 1; p > 0 && best_vals[p] > best_vals[p - 1]; p--) {
                        float tv = best_vals[p - 1];
                        uint ti = best_ids[p - 1];
                        best_vals[p - 1] = best_vals[p];
                        best_ids[p - 1] = best_ids[p];
                        best_vals[p] = tv;
                        best_ids[p] = ti;
                    }
                }
            }
        }
        for (uint j = 0; j < K; j++) {
            top_vals[row * K + j] = best_vals[j];
            top_ids[row * K + j] = best_ids[j];
        }
        lse_out[row] = m + log(partial_sum[0]);
    }
}

kernel void nllb_argmax_bf16_rows(
    device const bfloat *logits [[buffer(0)]],
    device uint *result [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= rows) {
        return;
    }
    threadgroup float partial_val[32];
    threadgroup uint partial_idx[32];
    device const bfloat *base = logits + (ulong)row * N;
    float best_val = -INFINITY;
    uint best_idx = 0;
    for (uint i = tid; i < N; i += tg_size) {
        float v = float(base[i]);
        if (v > best_val || (v == best_val && i < best_idx)) {
            best_val = v;
            best_idx = i;
        }
    }
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other_val = simd_shuffle_xor(best_val, offset);
        uint other_idx = simd_shuffle_xor(best_idx, offset);
        bool other_wins = other_val > best_val ||
                          (other_val == best_val && other_idx < best_idx);
        best_val = other_wins ? other_val : best_val;
        best_idx = other_wins ? other_idx : best_idx;
    }
    if (simd_lane_id == 0) {
        partial_val[simd_group_id] = best_val;
        partial_idx[simd_group_id] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial_val[tid] : -INFINITY;
        uint i = (tid < num_simds) ? partial_idx[tid] : 0u;
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            float other_v = simd_shuffle_xor(v, offset);
            uint other_i = simd_shuffle_xor(i, offset);
            bool other_wins = other_v > v || (other_v == v && other_i < i);
            v = other_wins ? other_v : v;
            i = other_wins ? other_i : i;
        }
        if (tid == 0) {
            result[row] = i;
        }
    }
}
