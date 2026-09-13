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
// ── QSA prefill top-k: radix select on device ──────────────────────────────
//
// Replaces the host top-k in `prefill_select`, which D2H'd every block score
// and sorted per query row on the CPU. At 36K context that is ~18 MB copied
// and 8192 sorts of 562 elements PER attention layer PER chunk — measured as
// the dominant prefill cost once the dense attention was skipped.
//
// Shape here is top-512 of up to ~25000 blocks (100K ctx / ratio 4), so the
// moe_topk iterative-argmax approach (K passes over N) is hopeless: 512
// passes. Radix select finds the exact K-th value in 4 fixed passes over the
// row, then emits in one more.
//
// Scores are relu(q.k) sums, hence NON-NEGATIVE, so the IEEE-754 bit pattern
// reinterpreted as uint32 is monotonically ordered — no sign-flip mapping
// needed. NaN cannot appear (relu of a finite dot product); an inf would sort
// to the top, which is the correct behaviour anyway.
//
// Tie-break matches the host it replaces: larger score first, LOWER INDEX on
// ties. Ties are common because relu floors at 0.0, so the tie pass walks in
// index order rather than racing on an atomic.
//
// Output order within the list is MATHEMATICALLY irrelevant to the consumer
// (`qsa_prefill_attn` softmaxes over the selected set), but it is NOT
// bit-irrelevant: that kernel's online softmax accumulates in list order, so
// a different order is a different fp32 rounding sequence. The greater-than
// emit below hands out slots in atomicAdd race order — same SET, different
// ORDER every run — which made temp-0 prefill non-reproducible past the
// inert bound (the second nondeterminism source; the first was the MoE
// prefill epilogue). The list is therefore canonicalised — ascending block
// id, bitonic in shared memory — before the kernel returns: the same trick
// `qsa_expand_sel` uses to reproduce the host's `sort_unstable()` for
// decode. topk <= QSA_TOPK_SORT_MAX is enforced host-side in
// `QsaIndexer::new` (decode's `qsa_expand_sel` shares the same 512 cap).

#define QSA_TOPK_THREADS 256
#define QSA_TOPK_SORT_MAX 512

extern "C" __global__ void qsa_topk_rows(
    const float* __restrict__ scores,  // [rows, stride] f32
    int* __restrict__ lists,           // [rows, topk]   i32 block ids
    unsigned int rows,
    unsigned int stride,
    unsigned int topk,
    unsigned int first_pos,            // GLOBAL position of row 0
    unsigned int ratio)                // tokens per block
{
    const unsigned int r = blockIdx.x;
    if (r >= rows) return;

    const unsigned int tid = threadIdx.x;
    const float* __restrict__ row = scores + (size_t)r * stride;
    int* __restrict__ out = lists + (size_t)r * topk;

    // Blocks fully covered by this query's visible prefix. Each row sees a
    // different prefix, which is why this is computed per row rather than
    // passed in.
    const unsigned int complete = (first_pos + r + 1u) / ratio;

    // Degenerate guard. Callers only invoke past the inert bound, where
    // complete > topk, but emitting a valid id beats writing uninitialised
    // memory that the attention would then gather from.
    if (complete <= topk) {
        for (unsigned int i = tid; i < topk; i += QSA_TOPK_THREADS) {
            out[i] = (int)(i < complete ? i : (complete ? complete - 1u : 0u));
        }
        return;
    }

    __shared__ unsigned int s_hist[256];
    __shared__ unsigned int s_prefix;   // bit pattern fixed so far
    __shared__ unsigned int s_need;     // still to take from the current bucket
    __shared__ unsigned int s_above;    // count strictly greater than threshold
    __shared__ unsigned int s_emitted;

    if (tid == 0) {
        s_prefix = 0u;
        s_need = topk;
        s_above = 0u;
    }
    __syncthreads();

    // ── 4 radix passes, 8 bits at a time, most-significant first ──
    for (int pass = 0; pass < 4; ++pass) {
        const unsigned int shift = 24u - 8u * (unsigned int)pass;
        // Bits already pinned by previous passes. Pass 0 pins nothing; the
        // shift would be 32 (UB), hence the explicit zero.
        const unsigned int mask_hi =
            (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8u));

        for (unsigned int i = tid; i < 256u; i += QSA_TOPK_THREADS) s_hist[i] = 0u;
        __syncthreads();

        for (unsigned int i = tid; i < complete; i += QSA_TOPK_THREADS) {
            const unsigned int b = __float_as_uint(row[i]);
            if ((b & mask_hi) == s_prefix) {
                atomicAdd(&s_hist[(b >> shift) & 0xFFu], 1u);
            }
        }
        __syncthreads();

        // Walk buckets high->low until the K-th element's bucket is reached.
        if (tid == 0) {
            unsigned int acc = 0u;
            unsigned int chosen = 0u;
            for (int d = 255; d >= 0; --d) {
                const unsigned int c = s_hist[d];
                if (acc + c >= s_need) { chosen = (unsigned int)d; break; }
                acc += c;
            }
            s_above += acc;                 // strictly above the chosen bucket
            s_need -= acc;                  // remaining, all inside it
            s_prefix |= chosen << shift;
        }
        __syncthreads();
    }

    // s_prefix is now the exact bit pattern of the K-th largest score.
    const unsigned int thresh = s_prefix;

    // ── emit everything strictly greater ──
    if (tid == 0) s_emitted = 0u;
    __syncthreads();
    for (unsigned int i = tid; i < complete; i += QSA_TOPK_THREADS) {
        if (__float_as_uint(row[i]) > thresh) {
            const unsigned int slot = atomicAdd(&s_emitted, 1u);
            if (slot < topk) out[slot] = (int)i;
        }
    }
    __syncthreads();

    // ── fill the remainder from ties, LOWEST INDEX FIRST ──
    // One warp walks in index order so the choice is deterministic and matches
    // the host tie-break. Ties are the common case at the 0.0 floor, so this
    // must not be an atomic race.
    if (tid < 32u) {
        unsigned int emitted = s_emitted;
        for (unsigned int base = 0u; base < complete && emitted < topk; base += 32u) {
            const unsigned int i = base + tid;
            const bool tie = (i < complete) && (__float_as_uint(row[i]) == thresh);
            const unsigned int ballot = __ballot_sync(0xFFFFFFFFu, tie);
            if (ballot) {
                // Rank within this 32-wide window, so lower index wins.
                const unsigned int rank =
                    __popc(ballot & ((1u << tid) - 1u));
                if (tie && emitted + rank < topk) {
                    out[emitted + rank] = (int)i;
                }
                emitted += __popc(ballot);
            }
        }
        if (tid == 0) s_emitted = emitted < topk ? emitted : topk;
    }
    __syncthreads();

    // Defensive: if the row somehow produced fewer than topk (impossible when
    // complete > topk, but a silent short list would make the attention read
    // stale ids), pad with block 0.
    for (unsigned int i = s_emitted + tid; i < topk; i += QSA_TOPK_THREADS) {
        out[i] = 0;
    }
    __syncthreads();

    // ── canonicalise: ascending block id ──
    // The emit pass assigned slots in atomicAdd race order; sorting makes the
    // order a pure function of the SET (which is already deterministic: radix
    // threshold + index-ordered tie fill), so the downstream fp32 accumulation
    // order — and with it temp-0 prefill — is bit-reproducible. Ascending is
    // also decode's canonical order. Bitonic over the next power of two,
    // INT_MAX-padded; the pad sorts to the tail and is never written back.
    __shared__ int s_sort[QSA_TOPK_SORT_MAX];
    unsigned int n = 1u;
    while (n < topk) n <<= 1u;
    if (n <= QSA_TOPK_SORT_MAX) {
        for (unsigned int i = tid; i < n; i += QSA_TOPK_THREADS) {
            s_sort[i] = (i < topk) ? out[i] : 0x7FFFFFFF;
        }
        __syncthreads();
        for (unsigned int k = 2u; k <= n; k <<= 1u) {
            for (unsigned int j = k >> 1u; j > 0u; j >>= 1u) {
                for (unsigned int i = tid; i < n; i += QSA_TOPK_THREADS) {
                    const unsigned int ixj = i ^ j;
                    if (ixj > i) {
                        const bool up = ((i & k) == 0u);
                        const int a = s_sort[i];
                        const int b = s_sort[ixj];
                        if ((a > b) == up) { s_sort[i] = b; s_sort[ixj] = a; }
                    }
                }
                __syncthreads();
            }
        }
        for (unsigned int i = tid; i < topk; i += QSA_TOPK_THREADS) {
            out[i] = s_sort[i];
        }
    }
}

// ── QSA decode selection: sort + expand, on device ─────────────────────────
//
// The DECODE consumer differs from prefill's. `qsa_prefill_attn` reads the
// block-id list directly and softmaxes over the set, so `qsa_topk_rows`
// leaves the order within the list unspecified. Decode instead EXPANDS the
// chosen blocks into a token-index array which `qsa_gather` packs into
// contiguous scratch, and that scratch — viewed through an identity block
// table — is handed to the stock paged decode attention. The host code this
// replaces did `blocks.sort_unstable()` before expanding, so the scratch slot
// order was ascending by position. Softmax is order-invariant and the decode
// call passes sliding_window = 0 (no position-dependent masking), so only the
// SET changes the math — but the accumulation order is not bit-invariant, and
// "identical to the host path" is the correctness bar here. The ascending
// sort is therefore reproduced exactly rather than dropped.
//
// One block per query row. topk <= QSA_EXPAND_MAX_K; the ids go through a
// bitonic sort in shared memory, padded to a power of two with INT_MAX which
// sorts to the tail and is never read back.
//
// This kernel also writes `seq_lens` and the identity block table, both pure
// functions of host-known quantities. Emitting them here makes the whole
// decode selection transfer-free: at 12 full-attention layers and C
// sequences that removes 12*C small pageable H2Ds per decode step, on top of
// the D2H stream drain the device top-k removed.
//
// Rows are addressed like `qsa_topk_rows`: row r's visible prefix is
// `first_pos + r + 1` unless `visible_rows` is non-null, in which case it is
// `visible_rows[r]` — the shape a batched multi-sequence decode needs, where
// rows are independent sequences at unrelated lengths.

#define QSA_EXPAND_THREADS 256
#define QSA_EXPAND_MAX_K 512

extern "C" __global__ __launch_bounds__(QSA_EXPAND_THREADS) void qsa_expand_sel(
    const int* __restrict__ lists,        // [rows, topk] block ids, any order
    const int* __restrict__ visible_rows, // [rows] visible prefix, or null
    int* __restrict__ sel,                // [rows, sel_stride] token indices
    int* __restrict__ seq_lens,           // [rows] n_sel, or null
    int* __restrict__ tables,             // [rows, table_stride], or null
    unsigned int topk,
    unsigned int ratio,
    unsigned int first_pos,               // GLOBAL position of row 0
    unsigned int sel_stride,
    unsigned int table_stride,
    unsigned int block_size)
{
    const unsigned int r = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const int* __restrict__ in = lists + (size_t)r * topk;
    int* __restrict__ out = sel + (size_t)r * sel_stride;

    const unsigned int visible =
        visible_rows ? (unsigned int)visible_rows[r] : (first_pos + r + 1u);
    const unsigned int complete = visible / ratio;
    const unsigned int n_sel = topk * ratio + (visible - complete * ratio);

    __shared__ int s[QSA_EXPAND_MAX_K];

    unsigned int n = 1u;
    while (n < topk) n <<= 1;
    for (unsigned int i = tid; i < n; i += QSA_EXPAND_THREADS) {
        s[i] = (i < topk) ? in[i] : 0x7FFFFFFF;
    }
    __syncthreads();

    // Bitonic sort, ascending. A stage's compare-exchange pairs are disjoint,
    // so one barrier per stage suffices and none is needed inside it.
    for (unsigned int k = 2u; k <= n; k <<= 1) {
        for (unsigned int j = k >> 1; j > 0u; j >>= 1) {
            for (unsigned int i = tid; i < n; i += QSA_EXPAND_THREADS) {
                const unsigned int ixj = i ^ j;
                if (ixj > i) {
                    const bool up = ((i & k) == 0u);
                    const int a = s[i];
                    const int b = s[ixj];
                    if ((a > b) == up) { s[i] = b; s[ixj] = a; }
                }
            }
            __syncthreads();
        }
    }

    // sel[i] = block[i / ratio] * ratio + (i % ratio), then the tail tokens
    // complete*ratio .. visible — byte-for-byte the host expansion.
    for (unsigned int i = tid; i < topk * ratio; i += QSA_EXPAND_THREADS) {
        out[i] = s[i / ratio] * (int)ratio + (int)(i % ratio);
    }
    for (unsigned int i = topk * ratio + tid; i < n_sel; i += QSA_EXPAND_THREADS) {
        out[i] = (int)(complete * ratio + (i - topk * ratio));
    }

    if (seq_lens && tid == 0) seq_lens[r] = (int)n_sel;
    if (tables) {
        // Identity for a single row; for a shared multi-row scratch pool the
        // row's pages start at r * table_stride, which is what this writes.
        const unsigned int pages = (n_sel + block_size - 1u) / block_size;
        for (unsigned int i = tid; i < pages && i < table_stride;
             i += QSA_EXPAND_THREADS) {
            tables[(size_t)r * table_stride + i] = (int)(r * table_stride + i);
        }
    }
}
// ── Tensor-core `qsa_prefill_attn`. Same result, one CTA per QUERY ROW. ──
//
// The shipped scalar kernel launches one CTA per (row, HEAD) and, inside,
// gives ONE WARP to ONE TOKEN: 8 FMAs per lane, then a six-shuffle tree to
// reduce them, per token, per head. nsys (2026-09-12, TP=2 x EP=2, chunk
// 8192) put it at 23.4% of an 8K prefill and 24.1% at 64K — 26.6 ms/call for
// 51.5 GFLOP, i.e. **1.94 TFLOP/s**, about 5% of what the CUDA cores alone
// can do. It is not bandwidth: at 8K the whole visible K/V is ~8 MB. It is
// the reduction tree and the one-token-per-warp serialisation.
//
// Two facts make the tensor-core form fall out:
//   * `nkv == 1` on a TP=2 rank, so `kvh = qh / (nq/nkv)` is 0 for EVERY
//     head — all 12 query heads read the SAME K/V rows. The scalar grid
//     re-streams them once per head.
//   * `lists` is indexed `lists + r * topk` — the selected set is per ROW,
//     not per head. So every head of a row attends the same tokens.
// One row is therefore a dense little attention problem, Q[nq,hd] against
// the row's own gathered K/V — no list union, no cross-row approximation,
// and the selected set is EXACTLY the scalar kernel's.
//
// Shape: Q [16 (nq<=16, padded), 256] x K^T [256, TB] -> S [16, TB], then
// P [16, TB] x V [TB, 256] -> O [16, 256], flash-style online softmax over
// token tiles. m16n8k16.row.col throughout, fp32 accumulate, lane mapping
// cribbed from `qsa_score_rows_tc` above (A rows gid/gid+8, A cols
// d0+tid*2 (+8), B n = gid, B k = d0+tid*2 (+8), D[gid+8*part][tid*2+cc]
// = acc[2*part+cc]).
//
// OCCUPANCY — why K and V SHARE one buffer. The first cut gave each its own
// (sKT 36864 + sV 33792 + Q/P/S), 85952 B per CTA. The sm_121 opt-in ceiling
// is 101376 B and an SM has 102400, so that is ONE CTA PER SM: 8 warps, no
// second block to hide latency behind, where the scalar kernel's ~8 KB let
// many CTAs stay resident. It measured +4.4% end-to-end against a 23.4%
// profile share — the mma win was real and the occupancy loss ate most of it.
// K and V are used in DISJOINT phases of a tile (K only for QK^T, V only for
// PV), so they now alias one `sKV` buffer sized to the larger, and the tile
// re-sequences to gather K, score, soften, gather V, accumulate. That costs
// one extra __syncthreads and splits the gather in two — the same bytes read,
// just later — and takes the CTA to 50112 B, which is TWO CTAs per SM.
// Padding is chosen for bank behaviour, not just size: sQ keeps pad 8 because
// its A-fragment reads stride by `gid` and a 256-element row makes all eight
// gids land on one bank (an 8-way conflict); 264 elements spreads them.
//
// PRECISION. q, k and v are ALL bf16 in memory already, so the QK^T mma is
// exact where the scalar path was (it widened the same bf16 to f32 and did
// f32 FMAs; mma does bf16 inputs with an f32 accumulator). The one place
// this loses ground is P: the scalar path keeps probabilities in f32, and
// the PV mma needs them in bf16 (~8 mantissa bits). Measured against the same
// CPU reference the scalar kernel is held to: worst cos 0.999997 vs the
// scalar's 0.999998, so the hi/lo split `qsa_score_rows_tc` needs for q is
// not needed here. If that ever changes, split P rather than reverting.
// Softmax is order-invariant and rope is baked into cached K, so this equals
// the reference mask, exactly as the scalar kernel's own comment claims.
//
// Grid: (rows)  Block: (256) = 8 warps.
#define QSA_PATC_TB 64          // tokens per tile
#define QSA_PATC_HD 256         // head_dim (checked at the call site)
#define QSA_PATC_M 16           // mma M — nq padded to 16
#define QSA_PATC_QPAD 8         // sQ row pad: kills an 8-way A-fragment conflict
// sKV-as-K^T row pad. 2, not 4, and the reason is the STORE, not the size.
// The gather reads k_cache coalesced but writes sKV[d*KT_ROW + j], striding by
// KT_ROW across consecutive d. Bank = (d*(TB+KPAD) + j)/2 mod 32, so the pad
// decides the stride in banks:
//   KPAD 4 -> 68 elems -> d*34 mod 32 = d*2 -> banks 0,2,..30: 32 lanes into
//             16 banks, a 2-way conflict on every K store of every tile;
//   KPAD 2 -> 66 elems -> d*33 mod 32 = d   -> 32 lanes into 32 banks, none.
// It is also 1024 B smaller. (sQ keeps pad 8 for the same class of reason on
// its A-fragment read; V is stored row-contiguous and does not care.)
#define QSA_PATC_KPAD 2
#define QSA_PATC_VPAD 4         // sKV-as-V row pad
#define QSA_PATC_PPAD 8
extern "C" __global__ __launch_bounds__(256) void qsa_prefill_attn_tc(
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
    const int TB = QSA_PATC_TB, HD = QSA_PATC_HD, M = QSA_PATC_M;
    const int KT_ROW = TB + QSA_PATC_KPAD;   // sKV as K^T: [hd][TB+pad]
    const int V_ROW  = HD + QSA_PATC_VPAD;   // sKV as V:   [TB][hd+pad]
    const int Q_ROW  = HD + QSA_PATC_QPAD;
    const int P_ROW  = TB + QSA_PATC_PPAD;
    // Dynamic, not static: static __shared__ is capped at 49152 B. The Rust
    // side's `QSA_PA_TC_SMEM` recomputes this same total from the same tile
    // constants — keep the two in step.
    extern __shared__ unsigned char smem_raw[];
    __nv_bfloat16* sKV = (__nv_bfloat16*)smem_raw;          // K^T, then V
    const size_t KV_ELEMS = (size_t)HD * KT_ROW > (size_t)TB * V_ROW
                          ? (size_t)HD * KT_ROW : (size_t)TB * V_ROW;
    __nv_bfloat16* sQ_ = sKV + KV_ELEMS;                    // M*(HD+QPAD)
    __nv_bfloat16* sP_ = sQ_ + (size_t)M * Q_ROW;           // M*(TB+PPAD)
    float* sS_   = (float*)(sP_ + (size_t)M * P_ROW);       // M*TB
    float* sM    = sS_ + (size_t)M * TB;
    float* sL    = sM + M;
    float* sCorr = sL + M;
    unsigned int* sTok = (unsigned int*)(sCorr + M);

    const unsigned int r = blockIdx.x;
    const unsigned int tidx = threadIdx.x, NT = 256;
    const unsigned int warp = tidx >> 5, lane = tidx & 31u;
    const unsigned int gid = lane >> 2, tid = lane & 3u;

    const unsigned int pos = first_pos + r;
    const unsigned int complete = (pos + 1) / ratio;
    const unsigned int tail = (pos + 1) - complete * ratio;
    const unsigned int n_tok = topk * ratio + tail;
    const unsigned int row_elems = nkv * hd;
    const unsigned long long page_stride = (unsigned long long)block_size * row_elems;

    // Q tile: [nq, hd] for this row, zero-padded to 16 rows. Lives for the
    // whole kernel; only K/V churn per tile.
    for (unsigned int i = tidx; i < (unsigned int)(M * HD); i += NT) {
        unsigned int h = i / HD, d = i % HD;
        sQ_[(size_t)h * Q_ROW + d] =
            (h < nq) ? q[((size_t)r * nq + h) * hd + d] : __float2bfloat16(0.0f);
    }
    for (unsigned int i = tidx; i < (unsigned int)M; i += NT) {
        sM[i] = -1e30f; sL[i] = 0.0f;
    }

    // O accumulator: each warp owns 32 of the 256 output dims = 4 n-tiles.
    float o[4][4];
#pragma unroll
    for (int t = 0; t < 4; t++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[t][e] = 0.0f;

    const int* my_list = lists + (size_t)r * topk;
    __syncthreads();

    for (unsigned int t0 = 0; t0 < n_tok; t0 += TB) {
        const unsigned int n_this = min((unsigned int)TB, n_tok - t0);

        // Token ids for this tile, resolved exactly as the scalar kernel does.
        for (unsigned int j = tidx; j < (unsigned int)TB; j += NT) {
            unsigned int t = t0 + j, tok = 0u;
            if (j < n_this) {
                tok = (t < topk * ratio)
                    ? (unsigned int)my_list[t / ratio] * ratio + (t % ratio)
                    : complete * ratio + (t - topk * ratio);
            }
            sTok[j] = tok;
        }
        __syncthreads();

        // ── Phase A: sKV holds K^T. kvh is 0 (nkv == 1 is gated at the call
        // site), so every head shares these rows.
        for (unsigned int i = tidx; i < (unsigned int)(TB * HD); i += NT) {
            unsigned int j = i / HD, d = i % HD;
            __nv_bfloat16 kv = __float2bfloat16(0.0f);
            if (j < n_this) {
                unsigned int tok = sTok[j];
                unsigned long long off =
                    (unsigned long long)(unsigned int)block_table[tok / block_size] * page_stride
                    + (unsigned long long)(tok % block_size) * row_elems;
                kv = k_cache[off + d];
            }
            sKV[(size_t)d * KT_ROW + j] = kv;
        }
        __syncthreads();

        // S[16, TB] = Q . K^T — each warp owns one 8-token n-tile.
        {
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            const unsigned short* sA = (const unsigned short*)sQ_;
            const unsigned short* sB = (const unsigned short*)sKV;
            const unsigned int nc = warp * 8 + gid;   // token column
#pragma unroll
            for (int d0 = 0; d0 < HD; d0 += 16) {
                unsigned int fr0 = gid, fr1 = gid + 8;
                unsigned int fc0 = d0 + tid * 2, fc1 = fc0 + 8;
                unsigned int a0 = ((unsigned int)sA[fr0*Q_ROW+fc0+1]<<16) | (unsigned int)sA[fr0*Q_ROW+fc0];
                unsigned int a1 = ((unsigned int)sA[fr1*Q_ROW+fc0+1]<<16) | (unsigned int)sA[fr1*Q_ROW+fc0];
                unsigned int a2 = ((unsigned int)sA[fr0*Q_ROW+fc1+1]<<16) | (unsigned int)sA[fr0*Q_ROW+fc1];
                unsigned int a3 = ((unsigned int)sA[fr1*Q_ROW+fc1+1]<<16) | (unsigned int)sA[fr1*Q_ROW+fc1];
                unsigned int k0 = d0 + tid * 2, k1 = k0 + 8;
                unsigned int b0 = ((unsigned int)sB[(k0+1)*KT_ROW+nc]<<16) | (unsigned int)sB[k0*KT_ROW+nc];
                unsigned int b1 = ((unsigned int)sB[(k1+1)*KT_ROW+nc]<<16) | (unsigned int)sB[k1*KT_ROW+nc];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    :"=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                     "f"(acc[0]),"f"(acc[1]),"f"(acc[2]),"f"(acc[3]));
            }
            // D[gid + 8*part][tid*2 + cc] = acc[2*part + cc], n-tile at warp*8.
#pragma unroll
            for (int part = 0; part < 2; part++) {
#pragma unroll
                for (int cc = 0; cc < 2; cc++) {
                    unsigned int row = gid + part * 8;
                    unsigned int col = warp * 8 + tid * 2 + cc;
                    if (col < (unsigned int)TB)
                        sS_[(size_t)row * TB + col] =
                            (col < n_this) ? acc[2*part+cc] * inv_sqrt_d : -1e30f;
                }
            }
        }
        __syncthreads();

        // Online softmax: one warp per two rows, over this tile's TB scores.
        for (unsigned int row = warp * 2; row < (unsigned int)M && row < warp * 2 + 2; row++) {
            float mx = -1e30f;
            for (unsigned int c = lane; c < (unsigned int)TB; c += 32)
                mx = fmaxf(mx, sS_[(size_t)row * TB + c]);
#pragma unroll
            for (int o2 = 16; o2 > 0; o2 >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xFFFFFFFFu, mx, o2));
            mx = __shfl_sync(0xFFFFFFFFu, mx, 0);
            const float m_old = sM[row];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
            for (unsigned int c = lane; c < (unsigned int)TB; c += 32) {
                float p = (c < n_this) ? __expf(sS_[(size_t)row * TB + c] - m_new) : 0.0f;
                sP_[(size_t)row * P_ROW + c] = __float2bfloat16(p);
                sum += p;
            }
#pragma unroll
            for (int o2 = 16; o2 > 0; o2 >>= 1) sum += __shfl_down_sync(0xFFFFFFFFu, sum, o2);
            sum = __shfl_sync(0xFFFFFFFFu, sum, 0);   // every lane must reach this
            if (lane == 0) {
                const float corr = __expf(m_old - m_new);
                sCorr[row] = corr;
                sM[row] = m_new;
                sL[row] = sL[row] * corr + sum;
            }
        }
        __syncthreads();

        // Rescale the carried output while sKV is still K — the read of sCorr
        // is the only thing phase B needs from phase A.
        {
            const float c0 = sCorr[gid], c1 = sCorr[gid + 8];
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                o[nt][0] *= c0; o[nt][1] *= c0;
                o[nt][2] *= c1; o[nt][3] *= c1;
            }
        }
        __syncthreads();   // everyone is done reading K^T before V overwrites it

        // ── Phase B: sKV now holds V, natural [token][dim].
        for (unsigned int i = tidx; i < (unsigned int)(TB * HD); i += NT) {
            unsigned int j = i / HD, d = i % HD;
            __nv_bfloat16 vv = __float2bfloat16(0.0f);
            if (j < n_this) {
                unsigned int tok = sTok[j];
                unsigned long long off =
                    (unsigned long long)(unsigned int)block_table[tok / block_size] * page_stride
                    + (unsigned long long)(tok % block_size) * row_elems;
                vv = v_cache[off + d];
            }
            sKV[(size_t)j * V_ROW + d] = vv;
        }
        __syncthreads();

        // O += P . V
        {
            const unsigned short* sA = (const unsigned short*)sP_;
            const unsigned short* sB = (const unsigned short*)sKV;
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const unsigned int nc = warp * 32 + nt * 8 + gid;   // output dim
#pragma unroll
                for (int k0 = 0; k0 < TB; k0 += 16) {
                    unsigned int fr0 = gid, fr1 = gid + 8;
                    unsigned int fc0 = k0 + tid * 2, fc1 = fc0 + 8;
                    unsigned int a0 = ((unsigned int)sA[fr0*P_ROW+fc0+1]<<16) | (unsigned int)sA[fr0*P_ROW+fc0];
                    unsigned int a1 = ((unsigned int)sA[fr1*P_ROW+fc0+1]<<16) | (unsigned int)sA[fr1*P_ROW+fc0];
                    unsigned int a2 = ((unsigned int)sA[fr0*P_ROW+fc1+1]<<16) | (unsigned int)sA[fr0*P_ROW+fc1];
                    unsigned int a3 = ((unsigned int)sA[fr1*P_ROW+fc1+1]<<16) | (unsigned int)sA[fr1*P_ROW+fc1];
                    unsigned int kk0 = k0 + tid * 2, kk1 = kk0 + 8;
                    unsigned int b0 = ((unsigned int)sB[(kk0+1)*V_ROW+nc]<<16) | (unsigned int)sB[kk0*V_ROW+nc];
                    unsigned int b1 = ((unsigned int)sB[(kk1+1)*V_ROW+nc]<<16) | (unsigned int)sB[kk1*V_ROW+nc];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(o[nt][0]),"=f"(o[nt][1]),"=f"(o[nt][2]),"=f"(o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(o[nt][0]),"f"(o[nt][1]),"f"(o[nt][2]),"f"(o[nt][3]));
                }
            }
        }
        __syncthreads();   // V must survive until every warp's PV mma is done
    }

    // Epilogue: divide by l and write [nq, hd] for this row.
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
#pragma unroll
        for (int part = 0; part < 2; part++) {
            unsigned int h = gid + part * 8;
            if (h >= nq) continue;
            const float inv_l = 1.0f / sL[h];
#pragma unroll
            for (int cc = 0; cc < 2; cc++) {
                unsigned int d = warp * 32 + nt * 8 + tid * 2 + cc;
                attn_out[((size_t)r * nq + h) * hd + d] =
                    __float2bfloat16(o[nt][2*part+cc] * inv_l);
            }
        }
    }
}
