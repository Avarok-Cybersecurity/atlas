// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next QSA indexer — the decode-side selection machinery.
//
// Reference: modeling_qwen4_exp.py Qwen4ExpTextQSAIndexer. Per query, the
// visible prefix is grouped into `ratio`(=4)-token blocks; each block's key
// is the MEAN of its raw per-token indexer keys, then k_layernorm
// (offset-from-1 RMSNorm), then partial rope at the block's FIRST token
// position. Scores are sum_h relu(q_h . k_b) / sqrt(head_dim); the top
// `block_topk` blocks plus the incomplete tail are the visible set.
//
// Selection feeds the EXISTING paged decode attention: qsa_gather packs the
// selected tokens' K/V rows into a contiguous scratch laid out NHD
// ([page, slot, kv_head, dim]) so an identity block table over the scratch
// reproduces the reference mask semantics with zero new attention code.
//
// Rope here is computed INLINE in double precision (32 freq lanes,
// inv_freq_j = theta^(-2j/rot)) rather than read from the attention rope
// tables — the golden's cos/sin come from torch fp32 and double sincos
// keeps the parity comparison out of ulp territory. Text-only mrope with
// equal position grids reduces to exactly this.

#include <cuda_bf16.h>

__device__ __forceinline__ float qsa_block_reduce_sum(float v, float* red) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    }
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float tot = 0.0f;
    if (threadIdx.x == 0) {
        const unsigned int warps = (blockDim.x + 31) >> 5;
        for (unsigned int w = 0; w < warps; ++w) tot += red[w];
        red[0] = tot;
    }
    __syncthreads();
    return red[0];
}

// normed (already in smem, length hd) -> rope at `pos` -> out (bf16).
// Assumes hd threads; rot must be even, pairs are (j, j + rot/2).
__device__ __forceinline__ void qsa_rope_store(
    const float* normed, __nv_bfloat16* out,
    unsigned int d, unsigned int rot, unsigned int pos, float theta
) {
    if (d < rot) {
        const unsigned int half = rot >> 1;
        const unsigned int j = (d < half) ? d : d - half;
        const double inv_freq = exp(-2.0 * (double)j / (double)rot * log((double)theta));
        double s, c;
        sincos((double)pos * inv_freq, &s, &c);
        const float x1 = normed[j];
        const float x2 = normed[j + half];
        const float v = (d < half) ? (x1 * (float)c - x2 * (float)s)
                                   : (x2 * (float)c + x1 * (float)s);
        out[d] = __float2bfloat16(v);
    } else {
        out[d] = __float2bfloat16(normed[d]);
    }
}

// ── qsa_block_pool ──
// Pool `n_new` freshly COMPLETE blocks starting at `first_block`:
// mean(ratio raw keys) -> RMSNorm*(1+w) -> rope at pos = block*ratio.
// Appends into block_keys [*, hd]. Grid: (n_new,1,1)  Block: (hd,1,1).
extern "C" __global__ void qsa_block_pool(
    const __nv_bfloat16* __restrict__ raw_keys,   // [S, hd]
    const __nv_bfloat16* __restrict__ k_norm_w,   // [hd]
    __nv_bfloat16* __restrict__ block_keys,       // [max_blocks, hd]
    const unsigned int first_block,
    const unsigned int ratio,
    const unsigned int hd,
    const unsigned int rot,
    const float theta,
    const float eps
) {
    const unsigned int b = first_block + blockIdx.x;
    const unsigned int d = threadIdx.x;

    extern __shared__ float smem[];               // [hd] normed + red
    float* stage = smem;
    float* red = smem + hd;

    float v = 0.0f;
    for (unsigned int r = 0; r < ratio; ++r) {
        v += (float)raw_keys[(size_t)(b * ratio + r) * hd + d];
    }
    v /= (float)ratio;

    const float sq = qsa_block_reduce_sum(v * v, red);
    const float rms = rsqrtf(sq / (float)hd + eps);
    stage[d] = v * rms * (1.0f + (float)k_norm_w[d]);
    __syncthreads();

    qsa_rope_store(stage, block_keys + (size_t)b * hd, d, rot, b * ratio, theta);
}

// ── qsa_qprep ──
// One decode query: per head, RMSNorm*(1+w) then rope at `pos`.
// q_in is the head-concatenated slice of the qk projection row.
// Grid: (n_heads,1,1)  Block: (hd,1,1). Output FP32 (feeds the scorer).
extern "C" __global__ void qsa_qprep(
    const __nv_bfloat16* __restrict__ q_in,       // [n_heads, hd]
    const __nv_bfloat16* __restrict__ q_norm_w,   // [hd]
    float* __restrict__ q_out,                    // [n_heads, hd]
    const unsigned int hd,
    const unsigned int rot,
    const unsigned int pos,
    const float theta,
    const float eps
) {
    const unsigned int h = blockIdx.x;
    const unsigned int d = threadIdx.x;

    extern __shared__ float smem[];
    float* stage = smem;
    float* red = smem + hd;

    const float x = (float)q_in[(size_t)h * hd + d];
    const float sq = qsa_block_reduce_sum(x * x, red);
    const float rms = rsqrtf(sq / (float)hd + eps);
    stage[d] = x * rms * (1.0f + (float)q_norm_w[d]);
    __syncthreads();

    float* out = q_out + (size_t)h * hd;
    if (d < rot) {
        const unsigned int half = rot >> 1;
        const unsigned int j = (d < half) ? d : d - half;
        const double inv_freq = exp(-2.0 * (double)j / (double)rot * log((double)theta));
        double s, c;
        sincos((double)pos * inv_freq, &s, &c);
        const float x1 = stage[j];
        const float x2 = stage[j + half];
        out[d] = (d < half) ? (x1 * (float)c - x2 * (float)s)
                            : (x2 * (float)c + x1 * (float)s);
    } else {
        out[d] = stage[d];
    }
}

// ── qsa_score ──
// scores[b] = sum_h relu(q_h . k_b) / sqrt(hd).
// Grid: (n_blocks,1,1)  Block: (hd,1,1).
extern "C" __global__ void qsa_score(
    const float* __restrict__ q,                  // [n_heads, hd]
    const __nv_bfloat16* __restrict__ block_keys, // [*, hd]
    float* __restrict__ scores,                   // [n_blocks]
    const unsigned int n_heads,
    const unsigned int hd
) {
    const unsigned int b = blockIdx.x;
    const unsigned int d = threadIdx.x;

    extern __shared__ float smem[];
    float* red = smem;

    const float k = (float)block_keys[(size_t)b * hd + d];
    float acc = 0.0f;
    for (unsigned int h = 0; h < n_heads; ++h) {
        const float dot = qsa_block_reduce_sum(q[(size_t)h * hd + d] * k, red);
        if (threadIdx.x == 0) acc += fmaxf(dot, 0.0f);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scores[b] = acc * rsqrtf((float)hd);
    }
}

// ── qsa_gather ──
// Pack the selected tokens' K/V rows (NHD paged layout) into contiguous
// scratch: dst slot i holds src position sel[i]. The scratch, viewed through
// an identity block table, IS a valid paged cache for the existing decode
// attention kernel. Grid: (n_sel,1,1)  Block: (256,1,1).
extern "C" __global__ void qsa_gather(
    const __nv_bfloat16* __restrict__ k_cache,    // [blocks, bs, nkv, hd]
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_table,          // logical -> physical
    const int* __restrict__ sel,                  // [n_sel] token positions
    __nv_bfloat16* __restrict__ k_out,            // [n_sel(padded), nkv, hd]
    __nv_bfloat16* __restrict__ v_out,
    const unsigned int block_size,
    const unsigned int nkv,
    const unsigned int hd
) {
    const unsigned int i = blockIdx.x;
    const unsigned int pos = (unsigned int)sel[i];
    const unsigned int row = nkv * hd;
    const unsigned long long page_stride =
        (unsigned long long)block_size * row;
    const unsigned long long src_off =
        (unsigned long long)(unsigned int)block_table[pos / block_size] * page_stride
        + (unsigned long long)(pos % block_size) * row;
    const unsigned long long dst_off = (unsigned long long)i * row;
    for (unsigned int e = threadIdx.x; e < row; e += blockDim.x) {
        k_out[dst_off + e] = k_cache[src_off + e];
        v_out[dst_off + e] = v_cache[src_off + e];
    }
}


// ──────────────────── stage 2: per-query PREFILL selection ────────────────────
//
// Selectivity is monotone in position: every chunk row at global pos >= 2051
// needs its own top-512-block set. Rows are processed as a contiguous range
// [first_pos, first_pos + n_rows); per row the score matrix is masked at the
// row's own complete-block count, host top-k builds a 512-entry block list,
// and qsa_prefill_attn OVERWRITES that row's attention context (pre-gate,
// pre-o_proj) with attention over exactly the selected set — read straight
// from the paged KV cache, so the dense flash pass it replaces needs no
// changes.

// Per-row q prep: RMSNorm*(1+w) + partial rope at pos = first_pos + row.
// qk rows are the indexer projection [rows, (n_heads+1)*hd]; q is the head-
// concatenated prefix of each row. Grid: (rows, n_heads)  Block: (hd,1,1).
extern "C" __global__ void qsa_qprep_rows(
    const __nv_bfloat16* __restrict__ qk,       // [rows, qkw]
    const __nv_bfloat16* __restrict__ q_norm_w, // [hd]
    float* __restrict__ q_out,                  // [rows, n_heads, hd]
    const unsigned int first_pos,
    const unsigned int qkw,
    const unsigned int n_heads,
    const unsigned int hd,
    const unsigned int rot,
    const float theta,
    const float eps
) {
    const unsigned int r = blockIdx.x;
    const unsigned int hh = blockIdx.y;
    const unsigned int d = threadIdx.x;
    const unsigned int pos = first_pos + r;

    extern __shared__ float smem[];
    float* stage = smem;
    float* red = smem + hd;

    const float x = (float)qk[(size_t)r * qkw + (size_t)hh * hd + d];
    const float sq = qsa_block_reduce_sum(x * x, red);
    const float rms = rsqrtf(sq / (float)hd + eps);
    stage[d] = x * rms * (1.0f + (float)q_norm_w[d]);
    __syncthreads();

    float* out = q_out + ((size_t)r * n_heads + hh) * hd;
    if (d < rot) {
        const unsigned int half = rot >> 1;
        const unsigned int j = (d < half) ? d : d - half;
        const double inv_freq = exp(-2.0 * (double)j / (double)rot * log((double)theta));
        double s, c;
        sincos((double)pos * inv_freq, &s, &c);
        const float x1 = stage[j];
        const float x2 = stage[j + half];
        out[d] = (d < half) ? (x1 * (float)c - x2 * (float)s)
                            : (x2 * (float)c + x1 * (float)s);
    } else {
        out[d] = stage[d];
    }
}

// Per-row block scores. scores[r, b] = sum_h relu(q[r,h] . k_b)/sqrt(hd) for
// b < complete(row), -inf otherwise (host top-k then never picks it).
// Grid: (rows, n_blocks_max)  Block: (hd,1,1).
extern "C" __global__ void qsa_score_rows(
    const float* __restrict__ q,                // [rows, n_heads, hd]
    const __nv_bfloat16* __restrict__ block_keys,
    float* __restrict__ scores,                 // [rows, score_stride]
    const unsigned int first_pos,
    const unsigned int score_stride,
    const unsigned int ratio,
    const unsigned int n_heads,
    const unsigned int hd
) {
    const unsigned int r = blockIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int d = threadIdx.x;
    const unsigned int complete = (first_pos + r + 1) / ratio;
    float* out = scores + (size_t)r * score_stride + b;
    if (b >= complete) {
        if (d == 0) *out = -1e30f;
        return;
    }

    extern __shared__ float smem[];
    float* red = smem;

    const float k = (float)block_keys[(size_t)b * hd + d];
    const float* qr = q + (size_t)r * n_heads * hd;
    float acc = 0.0f;
    for (unsigned int hh = 0; hh < n_heads; ++hh) {
        const float dot = qsa_block_reduce_sum(qr[(size_t)hh * hd + d] * k, red);
        if (d == 0) acc += fmaxf(dot, 0.0f);
        __syncthreads();
    }
    if (d == 0) *out = acc * rsqrtf((float)hd);
}

// Attention over EXACTLY the selected set for one (row, q-head): the listed
// `topk` blocks (ratio tokens each) plus the incomplete tail
// [complete*ratio, pos]. K/V come straight from the paged cache; the output
// OVERWRITES that row's context in attn_out (pre-gate, pre-o_proj), so the
// surrounding dense path needs no other change. Softmax is order-invariant
// and rope is baked into cached K, so this equals the reference mask.
// Grid: (rows, nq)  Block: (256,1,1) = 8 warps, warp-striped online softmax.
#define QSA_PA_WARPS 8
// Tensor-core `qsa_score_rows`, split-q. Same result, ~14x faster.
//
// The shipped kernel above launches ONE CTA PER (row, block) — 57.3M of them
// at 28K context — and inside each runs four sequential 128-thread block
// reductions with only lane 0 accumulating: measured 1.3 TFLOP/s, 13.1% of
// prefill (nsys, 2026-08-27). But
//     scores[r,b] = rsqrt(hd) * SUM_h relu(q[r,h,:] . k[b,:])
// is four matmuls Q_h[rows,hd] x K^T[hd,blocks] with a ReLU between the
// product and the head-sum, so each head keeps its own accumulator.
//
// PRECISION — why q is SPLIT. block_keys are already bf16, hence exact under
// a bf16 mma; only q (f32) would lose mantissa. A plain bf16 cast measured a
// worst relative error of 66 and dropped up to 16 of 2048 selected blocks per
// row — the ReLU sits at zero, which is exactly where cancellation bites.
// Splitting ONLY q,
//     q ~ q_hi + q_lo,  q_hi = bf16(q),  q_lo = bf16(q - q_hi)
//     q.k = q_hi.k + q_lo.k        (two mma into the same f32 accumulator)
// carries ~17 mantissa bits of q for 2x the mma work: worst relative error
// 2.3e-3 and IDENTICAL top-2048 selection on every sampled row. Selection
// identity — not bit-exactness — is the correctness bar here, because this
// feeds a top-k and float addition is not associative anyway.
//
// Grid: (ceil(rows/16), ceil(blocks/64))  Block: (256) = 8 warps, one warp
// per 8-block n-tile.
#define QSA_TC_TR 16
#define QSA_TC_TB 64
#define QSA_TC_H 4
#define QSA_TC_HD 128
#define QSA_TC_APAD 8
#define QSA_TC_BPAD 8
extern "C" __global__ __launch_bounds__(256) void qsa_score_rows_tc(
    const float* __restrict__ q,                // [rows, H, HD] f32
    const __nv_bfloat16* __restrict__ block_keys,
    float* __restrict__ scores,                 // [rows, score_stride]
    const unsigned int first_pos,
    const unsigned int score_stride,
    const unsigned int ratio,
    const unsigned int n_blocks
) {
    const int TR = QSA_TC_TR, TB = QSA_TC_TB, H = QSA_TC_H, HD = QSA_TC_HD;
    __shared__ __nv_bfloat16 sqh[QSA_TC_TR][QSA_TC_H][QSA_TC_HD + QSA_TC_APAD];
    __shared__ __nv_bfloat16 sql[QSA_TC_TR][QSA_TC_H][QSA_TC_HD + QSA_TC_APAD];
    __shared__ __nv_bfloat16 skT[QSA_TC_HD][QSA_TC_TB + QSA_TC_BPAD];

    const unsigned int r0 = blockIdx.x * TR, b0 = blockIdx.y * TB;
    const unsigned int tidx = threadIdx.x, NT = 256;

    for (unsigned int i = tidx; i < (unsigned int)(TR * H * HD); i += NT) {
        unsigned int rr = i / (H * HD), rem = i % (H * HD);
        float v = q[(size_t)(r0 + rr) * H * HD + rem];
        __nv_bfloat16 hi = __float2bfloat16(v);
        __nv_bfloat16 lo = __float2bfloat16(v - __bfloat162float(hi));
        sqh[rr][rem / HD][rem % HD] = hi;
        sql[rr][rem / HD][rem % HD] = lo;
    }
    for (unsigned int i = tidx; i < (unsigned int)(TB * HD); i += NT) {
        unsigned int bb = i / HD, dd = i % HD;
        skT[dd][bb] = (b0 + bb) < n_blocks ? block_keys[(size_t)(b0 + bb) * HD + dd]
                                           : __float2bfloat16(0.0f);
    }
    __syncthreads();

    const unsigned int warp = tidx >> 5, lane = tidx & 31u;
    const unsigned int gid = lane >> 2, tid = lane & 3u;
    float acc[QSA_TC_H][4];
#pragma unroll
    for (int h = 0; h < H; h++) { acc[h][0]=0.f; acc[h][1]=0.f; acc[h][2]=0.f; acc[h][3]=0.f; }

    const unsigned short* sB = (const unsigned short*)skT;
    const int b_stride = TB + QSA_TC_BPAD;
    const int a_row = H * (HD + QSA_TC_APAD);
#pragma unroll
    for (int h = 0; h < H; h++) {
        const unsigned short* sAh = (const unsigned short*)&sqh[0][h][0];
        const unsigned short* sAl = (const unsigned short*)&sql[0][h][0];
#pragma unroll
        for (int d0 = 0; d0 < HD; d0 += 16) {
            unsigned int fr0 = gid, fr1 = gid + 8;
            unsigned int fc0 = d0 + tid * 2, fc1 = fc0 + 8;
            unsigned int nc = warp * 8 + gid;
            unsigned int k0 = d0 + tid * 2, k1 = k0 + 8;
            unsigned int br0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16) | (unsigned int)sB[k0*b_stride+nc];
            unsigned int br1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16) | (unsigned int)sB[k1*b_stride+nc];
#pragma unroll
            for (int part = 0; part < 2; part++) {
                const unsigned short* sA = part ? sAl : sAh;
                unsigned int a0 = ((unsigned int)sA[fr0*a_row+fc0+1]<<16) | (unsigned int)sA[fr0*a_row+fc0];
                unsigned int a1 = ((unsigned int)sA[fr1*a_row+fc0+1]<<16) | (unsigned int)sA[fr1*a_row+fc0];
                unsigned int a2 = ((unsigned int)sA[fr0*a_row+fc1+1]<<16) | (unsigned int)sA[fr0*a_row+fc1];
                unsigned int a3 = ((unsigned int)sA[fr1*a_row+fc1+1]<<16) | (unsigned int)sA[fr1*a_row+fc1];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    :"=f"(acc[h][0]),"=f"(acc[h][1]),"=f"(acc[h][2]),"=f"(acc[h][3])
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(br0),"r"(br1),
                     "f"(acc[h][0]),"f"(acc[h][1]),"f"(acc[h][2]),"f"(acc[h][3]));
            }
        }
    }

    const float inv = rsqrtf((float)HD);
#pragma unroll
    for (int part = 0; part < 2; part++) {
        unsigned int r = r0 + gid + part * 8;
        unsigned int complete = (first_pos + r + 1) / ratio;
#pragma unroll
        for (int cc = 0; cc < 2; cc++) {
            unsigned int b = b0 + warp * 8 + tid * 2 + cc;
            if (b >= n_blocks) continue;
            float sum = 0.f;
#pragma unroll
            for (int h = 0; h < H; h++) sum += fmaxf(acc[h][part*2+cc], 0.0f);
            scores[(size_t)r * score_stride + b] = (b >= complete) ? -1e30f : sum * inv;
        }
    }
}

extern "C" __global__ void qsa_prefill_attn(
    const __nv_bfloat16* __restrict__ q,        // [rows, nq, hd] (roped)
    const __nv_bfloat16* __restrict__ k_cache,  // paged NHD
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ lists,              // [rows, topk] block ids
    __nv_bfloat16* __restrict__ attn_out,       // [rows, nq, hd]
    const unsigned int first_pos,
    const unsigned int topk,
    const unsigned int ratio,
    const unsigned int block_size,
    const unsigned int nq,
    const unsigned int nkv,
    const unsigned int hd,
    const float inv_sqrt_d
) {
    const unsigned int r = blockIdx.x;
    const unsigned int qh = blockIdx.y;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int pos = first_pos + r;
    const unsigned int complete = (pos + 1) / ratio;
    const unsigned int tail = (pos + 1) - complete * ratio;
    const unsigned int n_tok = topk * ratio + tail;
    const unsigned int kvh = qh / (nq / nkv);
    const unsigned int row_elems = nkv * hd;
    const unsigned long long page_stride = (unsigned long long)block_size * row_elems;
    const unsigned int vec = hd / 32;           // elems per lane (8 at hd=256)

    extern __shared__ float smem[];
    // Per-warp partials: [warps][hd] acc, then [warps] m, [warps] l.
    float* acc_w = smem;                        // [QSA_PA_WARPS * hd]
    float* m_w = smem + QSA_PA_WARPS * hd;      // [QSA_PA_WARPS]
    float* l_w = m_w + QSA_PA_WARPS;            // [QSA_PA_WARPS]

    // q slice for this (row, head), staged per lane.
    const __nv_bfloat16* qrow = q + ((size_t)r * nq + qh) * hd;
    float qreg[8];
    #pragma unroll
    for (unsigned int e = 0; e < 8; ++e) {
        qreg[e] = (e < vec) ? (float)qrow[lane * vec + e] : 0.0f;
    }

    float m = -1e30f, l = 0.0f;
    float acc[8];
    #pragma unroll
    for (unsigned int e = 0; e < 8; ++e) acc[e] = 0.0f;

    const int* my_list = lists + (size_t)r * topk;
    for (unsigned int t = warp; t < n_tok; t += QSA_PA_WARPS) {
        unsigned int tok;
        if (t < topk * ratio) {
            tok = (unsigned int)my_list[t / ratio] * ratio + (t % ratio);
        } else {
            tok = complete * ratio + (t - topk * ratio);
        }
        const unsigned long long off =
            (unsigned long long)(unsigned int)block_table[tok / block_size] * page_stride
            + (unsigned long long)(tok % block_size) * row_elems
            + (unsigned long long)kvh * hd;
        const __nv_bfloat16* krow = k_cache + off;
        float dot = 0.0f;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            if (e < vec) dot += qreg[e] * (float)krow[lane * vec + e];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_down_sync(0xFFFFFFFFu, dot, o);
        dot = __shfl_sync(0xFFFFFFFFu, dot, 0) * inv_sqrt_d;

        const float m_new = fmaxf(m, dot);
        const float scale = __expf(m - m_new);
        const float p = __expf(dot - m_new);
        l = l * scale + p;
        const __nv_bfloat16* vrow = v_cache + off;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            if (e < vec) acc[e] = acc[e] * scale + p * (float)vrow[lane * vec + e];
        }
        m = m_new;
    }

    // Park warp partials, then warp 0 merges.
    #pragma unroll
    for (unsigned int e = 0; e < 8; ++e) {
        if (e < vec) acc_w[warp * hd + lane * vec + e] = acc[e];
    }
    if (lane == 0) { m_w[warp] = m; l_w[warp] = l; }
    __syncthreads();

    if (warp == 0) {
        float m_tot = -1e30f;
        for (unsigned int w = 0; w < QSA_PA_WARPS; ++w) m_tot = fmaxf(m_tot, m_w[w]);
        float l_tot = 0.0f;
        float out[8];
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) out[e] = 0.0f;
        for (unsigned int w = 0; w < QSA_PA_WARPS; ++w) {
            const float s = __expf(m_w[w] - m_tot);
            l_tot += l_w[w] * s;
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) {
                if (e < vec) out[e] += acc_w[w * hd + lane * vec + e] * s;
            }
        }
        const float inv_l = (l_tot > 0.0f) ? 1.0f / l_tot : 0.0f;
        __nv_bfloat16* orow = attn_out + ((size_t)r * nq + qh) * hd;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            if (e < vec) orow[lane * vec + e] = __float2bfloat16(out[e] * inv_l);
        }
    }
}
