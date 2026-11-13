// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after Quest (page min/max upper-bound selection) and the page-sparse prototype in
// arczhi/llama-tierkv scripts-5060ti/proto/proto_page_sparse.cu (MIT, Copyright (c) 2023-2026 The
// ggml authors; file has no header of its own). Kernels rewritten for Atlas paged KV; changes: GQA
// grouping, warp-per-page scoring, a radix top-k in one CTA instead of a thread-0 merge, a compacted
// block table written into selector-owned buffers instead of host staging, summaries keyed by the
// page's leading physical block with a completeness mask, a key fingerprint and a slot lock, and
// directed f16 rounding so the stored bound stays an upper bound, by Atlas contributors. The
// prototype source was not read; the formula is the one written in the port sprint's report 77
// (section 5.2 and BR-4): score_p = sum_c max(q_c * kmin_c, q_c * kmax_c).
//
// Responsibility: Quest page-sparse decode for the Ada full-attention layers (port lane L30, row
// `attn_page_sparse`, APPROXIMATE, off by default on every target). Three entry points, all
// launched by `crates/spark-model/src/port/sparse_quest/launch.rs`:
//
//   attn_ps_summary_update  after the K/V write of a step or prefill chunk: fold the new keys into
//                           the per-page channel min/max of every page they touch.
//   attn_ps_page_score      per decode/verify step and layer: the Quest upper bound of every
//                           candidate page against the step's (post-RoPE) queries.
//   attn_ps_compact         per sequence: exact top-k of the candidates, then the compacted block
//                           table [sink pages | top-k pages | recent window] in chronological order
//                           and the shortened lengths, into the selector's fixed View buffers. Below
//                           the minimum context, or when the budget covers every page, the row is a
//                           verbatim copy of the dense table (the identity view, bit-identical).
//
// Layouts (host mirror: crates/spark-runtime/src/kv_cache/page_summary.rs):
//   K pool    [num_blocks][block_size][kv_heads][head_dim], BF16 (eb = 2) or FP8 E4M3 (eb = 1),
//             block stride `block_stride` bytes (the layer's own pool, as the paged decode reads it).
//   summary   per layer [cap_blocks][kv_heads][2][head_dim] f16: kmin then kmax, keyed by the
//             physical block that holds the page's first token.
//   meta      per layer [cap_blocks][kv_heads][4] u64: completeness mask, key fingerprint, lock, 0.
// Everything a kernel indexes comes from the device block table and lengths, so a captured CUDA
// graph replays correctly as sequences grow, and the graph topology never changes.
//
// FP8 keys are summarized unscaled: the per-layer K scale is one positive constant, so the page
// ranking is unchanged by it.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

#define PS_META_WORDS 4u
#define PS_UPDATE_THREADS 128u
#define PS_SCORE_WARPS 8u
#define PS_PAGES_PER_WARP 4u
#define PS_PAGES_PER_CTA (PS_SCORE_WARPS * PS_PAGES_PER_WARP)
#define PS_MAX_CPL 16u
#define PS_COMPACT_THREADS 1024u
#define PS_KEYS_PER_THREAD 8u
#define PS_MAX_CANDIDATES (PS_COMPACT_THREADS * PS_KEYS_PER_THREAD)

namespace {

__device__ __forceinline__ const unsigned char* k_addr(const unsigned char* pool,
                                                       unsigned long long block_stride,
                                                       unsigned phys, unsigned off, unsigned kvh,
                                                       unsigned h, unsigned hd, unsigned c,
                                                       unsigned eb) {
    return pool + (unsigned long long)phys * block_stride
         + ((unsigned long long)(off * kvh + h) * hd + c) * eb;
}

__device__ __forceinline__ float load_k(const unsigned char* p, unsigned eb) {
    if (eb == 2u) {
        return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(p));
    }
    __nv_fp8_e4m3 v;
    v.__x = *reinterpret_cast<const __nv_fp8_storage_t*>(p);
    return static_cast<float>(v);
}

// The first 8 bytes of head h's key at the block's first token.
__device__ __forceinline__ unsigned long long k_fingerprint(const unsigned char* pool,
                                                            unsigned long long block_stride,
                                                            unsigned phys, unsigned kvh, unsigned h,
                                                            unsigned hd, unsigned eb) {
    return *reinterpret_cast<const unsigned long long*>(
        k_addr(pool, block_stride, phys, 0u, kvh, h, hd, 0u, eb));
}

__device__ __forceinline__ unsigned long long page_bits(unsigned lo, unsigned hi, unsigned page_lo) {
    const unsigned span = hi - lo;
    const unsigned long long bits = span >= 64u ? ~0ull : ((1ull << span) - 1ull);
    return bits << (lo - page_lo);
}

__device__ __forceinline__ unsigned long long full_mask(unsigned pt) {
    return pt >= 64u ? ~0ull : ((1ull << pt) - 1ull);
}

__device__ __forceinline__ void slot_lock(unsigned int* l) {
    while (atomicCAS(l, 0u, 1u) != 0u) {
        __nanosleep(64);
    }
    __threadfence();
}

__device__ __forceinline__ void slot_unlock(unsigned int* l) {
    __threadfence();
    atomicExch(l, 0u);
}

// The first page of the recent window: the window starts `recent` tokens before the first new
// token's extent, rounded down to a page, and always covers the page of the first new token.
__device__ __forceinline__ unsigned recent_page(unsigned e_lo, unsigned recent, unsigned pt) {
    const unsigned start = e_lo > recent ? e_lo - recent : 0u;
    const unsigned a = start / pt;
    const unsigned b = (e_lo - 1u) / pt;
    return a < b ? a : b;
}

// Float to an order-preserving u32 (NaN ranks as +inf: an unknown page is kept).
__device__ __forceinline__ unsigned sortable(float s) {
    if (isnan(s)) {
        s = INFINITY;
    }
    const unsigned u = __float_as_uint(s);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

}  // namespace

// grid (pages touched, kv_heads, groups), block PS_UPDATE_THREADS.
// Decode/verify (seq_lens != NULL): group g owns rows [g*R, g*R + R) with consecutive extents; its
// new tokens are positions [seq_lens[g*R] - 1, + R). Prefill (seq_lens == NULL): one group whose
// written range is [first_pos, first_pos + count). The group's table is bt_ptrs[g] when bt_ptrs is
// set, else bt + g*R*row_stride.
extern "C" __global__ void attn_ps_summary_update(
    const unsigned char* __restrict__ k_pool, __half* summ, unsigned long long* meta,
    const int* __restrict__ bt, const unsigned long long* __restrict__ bt_ptrs,
    const int* __restrict__ seq_lens, unsigned first_pos, unsigned count, unsigned row_stride,
    unsigned rows_per_seq, unsigned group_base, unsigned kvh, unsigned hd, unsigned bs,
    unsigned pb, unsigned long long block_stride, unsigned eb, unsigned cap_blocks) {
    const unsigned g = group_base + blockIdx.z;
    const unsigned h = blockIdx.y;
    const int* row;
    if (bt_ptrs != nullptr) {
        row = reinterpret_cast<const int*>(bt_ptrs[g]);
        if (row == nullptr) {
            return;
        }
    } else {
        row = bt + (unsigned long long)g * rows_per_seq * row_stride;
    }
    unsigned first, cnt;
    if (seq_lens != nullptr) {
        const int e = seq_lens[g * rows_per_seq];
        if (e <= 0) {
            return;
        }
        first = (unsigned)e - 1u;
        cnt = rows_per_seq;
    } else {
        first = first_pos;
        cnt = count;
    }
    if (cnt == 0u) {
        return;
    }
    const unsigned pt = pb * bs;
    const unsigned pg = first / pt + blockIdx.x;
    const unsigned page_lo = pg * pt;
    const unsigned end = first + cnt;
    if (page_lo >= end || pg * pb >= row_stride) {
        return;
    }
    const unsigned lo = first > page_lo ? first : page_lo;
    const unsigned hi = end < page_lo + pt ? end : page_lo + pt;
    const int lead = row[pg * pb];
    if (lead < 0 || (unsigned)lead >= cap_blocks) {
        return;
    }
    const unsigned long long slot = (unsigned long long)lead * kvh + h;
    unsigned long long* m = meta + slot * PS_META_WORDS;
    __half* vals = summ + slot * 2ull * hd;

    __shared__ int s_reset;
    if (threadIdx.x == 0) {
        slot_lock(reinterpret_cast<unsigned int*>(m + 2));
        const unsigned long long cur = k_fingerprint(k_pool, block_stride, (unsigned)lead, kvh, h, hd, eb);
        const unsigned long long stored = __ldcg(m + 1);
        // Writing the page's first token means the block is new to this page; a fingerprint that
        // no longer matches means the block was reused or restored. Either way the old summary is
        // not this page's.
        const int reset = (lo == page_lo) || (stored != cur);
        const unsigned long long bits = page_bits(lo, hi, page_lo);
        const unsigned long long old = __ldcg(m);
        __stcg(m, reset ? bits : (old | bits));
        __stcg(m + 1, cur);
        s_reset = reset;
    }
    __syncthreads();
    const int reset = s_reset;
    for (unsigned c = threadIdx.x; c < hd; c += blockDim.x) {
        float mn = INFINITY;
        float mx = -INFINITY;
        if (!reset) {
            mn = __half2float(__ldcg(vals + c));
            mx = __half2float(__ldcg(vals + hd + c));
        }
        for (unsigned pos = lo; pos < hi; ++pos) {
            const int b = row[pos / bs];
            if (b < 0) {
                continue;
            }
            const float v = load_k(k_addr(k_pool, block_stride, (unsigned)b, pos % bs, kvh, h, hd, c, eb), eb);
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }
        __stcg(vals + c, __float2half_rd(mn));
        __stcg(vals + hd + c, __float2half_ru(mx));
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        slot_unlock(reinterpret_cast<unsigned int*>(m + 2));
    }
}

// grid (ceil(score_stride / PS_PAGES_PER_CTA), groups), block PS_SCORE_WARPS * 32.
// One warp per candidate page; lanes own head_dim / 32 consecutive channels. A page whose summary
// is not trusted (mask not full, fingerprint mismatch) is rebuilt from its keys and written back.
// scores[g * score_stride + p] holds the bound of candidate page p (pages in [sink, recent_page)).
extern "C" __global__ void attn_ps_page_score(
    const unsigned char* __restrict__ k_pool, __half* summ, unsigned long long* meta,
    const int* __restrict__ bt, const int* __restrict__ seq_lens,
    const __nv_bfloat16* __restrict__ q, float* __restrict__ scores, unsigned rows_per_seq,
    unsigned row_stride, unsigned q_row_stride, unsigned nq, unsigned kvh, unsigned hd,
    unsigned bs, unsigned pb, unsigned long long block_stride, unsigned eb, unsigned cap_blocks,
    unsigned sink_pages, unsigned recent_tokens, unsigned score_stride, unsigned gqa_sum,
    unsigned topk, unsigned min_ctx) {
    const unsigned g = blockIdx.y;
    const unsigned r0 = g * rows_per_seq;
    const int e_lo_i = seq_lens[r0];
    const int e_hi_i = seq_lens[r0 + rows_per_seq - 1u];
    if (e_lo_i <= 0 || e_hi_i <= (int)min_ctx) {
        return;
    }
    const unsigned pt = pb * bs;
    const unsigned rp = recent_page((unsigned)e_lo_i, recent_tokens, pt);
    if (rp <= sink_pages || rp - sink_pages <= topk) {
        return;
    }
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned cpl = hd >> 5;
    const unsigned c0 = lane * cpl;
    const unsigned group = nq / kvh;
    const unsigned long long fullm = full_mask(pt);
    const int* row = bt + (unsigned long long)r0 * row_stride;

    for (unsigned i = 0; i < PS_PAGES_PER_WARP; ++i) {
        const unsigned p = blockIdx.x * PS_PAGES_PER_CTA + warp * PS_PAGES_PER_WARP + i;
        if (p < sink_pages || p >= rp || p >= score_stride) {
            continue;
        }
        const int lead = row[p * pb];
        float page = gqa_sum ? 0.0f : -INFINITY;
        for (unsigned h = 0; h < kvh; ++h) {
            float mn[PS_MAX_CPL];
            float mx[PS_MAX_CPL];
            const bool has_slot = lead >= 0 && (unsigned)lead < cap_blocks;
            const unsigned long long slot = has_slot ? (unsigned long long)lead * kvh + h : 0ull;
            unsigned long long* m = meta + slot * PS_META_WORDS;
            __half* vals = summ + slot * 2ull * hd;
            unsigned long long cur = 0ull;
            int trusted = 0;
            if (has_slot) {
                if (lane == 0) {
                    cur = k_fingerprint(k_pool, block_stride, (unsigned)lead, kvh, h, hd, eb);
                    trusted = __ldcg(m) == fullm && __ldcg(m + 1) == cur;
                }
                cur = __shfl_sync(0xffffffffu, cur, 0);
                trusted = __shfl_sync(0xffffffffu, trusted, 0);
            }
            if (trusted) {
#pragma unroll
                for (unsigned j = 0; j < PS_MAX_CPL; ++j) {
                    if (j < cpl) {
                        mn[j] = __half2float(__ldcg(vals + c0 + j));
                        mx[j] = __half2float(__ldcg(vals + hd + c0 + j));
                    }
                }
            } else {
#pragma unroll
                for (unsigned j = 0; j < PS_MAX_CPL; ++j) {
                    mn[j] = INFINITY;
                    mx[j] = -INFINITY;
                }
                for (unsigned t = 0; t < pt; ++t) {
                    const int b = row[p * pb + t / bs];
                    if (b < 0) {
                        continue;
                    }
                    const unsigned char* base = k_addr(k_pool, block_stride, (unsigned)b, t % bs, kvh, h, hd, c0, eb);
#pragma unroll
                    for (unsigned j = 0; j < PS_MAX_CPL; ++j) {
                        if (j < cpl) {
                            const float v = load_k(base + j * eb, eb);
                            mn[j] = fminf(mn[j], v);
                            mx[j] = fmaxf(mx[j], v);
                        }
                    }
                }
#pragma unroll
                for (unsigned j = 0; j < PS_MAX_CPL; ++j) {
                    if (j < cpl) {
                        mn[j] = __half2float(__float2half_rd(mn[j]));
                        mx[j] = __half2float(__float2half_ru(mx[j]));
                    }
                }
                if (has_slot) {
                    // Write the rebuilt summary back. A matching fingerprint means the stored
                    // values belong to this block's pages (a prefix shared with another sequence
                    // whose later blocks differ), so merge; a mismatch means they are stale.
                    int merge = 0;
                    if (lane == 0) {
                        slot_lock(reinterpret_cast<unsigned int*>(m + 2));
                        merge = __ldcg(m + 1) == cur;
                    }
                    merge = __shfl_sync(0xffffffffu, merge, 0);
                    __syncwarp();
#pragma unroll
                    for (unsigned j = 0; j < PS_MAX_CPL; ++j) {
                        if (j < cpl) {
                            if (merge) {
                                mn[j] = fminf(mn[j], __half2float(__ldcg(vals + c0 + j)));
                                mx[j] = fmaxf(mx[j], __half2float(__ldcg(vals + hd + c0 + j)));
                            }
                            __stcg(vals + c0 + j, __float2half_rd(mn[j]));
                            __stcg(vals + hd + c0 + j, __float2half_ru(mx[j]));
                        }
                    }
                    __threadfence();
                    __syncwarp();
                    if (lane == 0) {
                        __stcg(m, fullm);
                        __stcg(m + 1, cur);
                        slot_unlock(reinterpret_cast<unsigned int*>(m + 2));
                    }
                    __syncwarp();
                }
            }
            for (unsigned jq = 0; jq < group; ++jq) {
                const unsigned qh = h * group + jq;
                float head = -INFINITY;
                for (unsigned r = 0; r < rows_per_seq; ++r) {
                    const __nv_bfloat16* qp = q + (unsigned long long)(r0 + r) * q_row_stride
                                            + (unsigned long long)qh * hd + c0;
                    float part = 0.0f;
#pragma unroll
                    for (unsigned j = 0; j < PS_MAX_CPL; ++j) {
                        if (j < cpl) {
                            const float qc = __bfloat162float(qp[j]);
                            if (qc != 0.0f) {
                                part += fmaxf(qc * mn[j], qc * mx[j]);
                            }
                        }
                    }
                    part = warp_sum(part);
                    head = fmaxf(head, part);
                }
                page = gqa_sum ? page + head : fmaxf(page, head);
            }
        }
        if (lane == 0) {
            scores[(unsigned long long)g * score_stride + p] = isnan(page) ? INFINITY : page;
        }
    }
}

// grid (groups), block PS_COMPACT_THREADS. Writes out_bt rows [g*R, g*R + R) (row stride
// row_stride) and out_lens, both selector-owned, and when count_stats is set adds this group's
// sparse selection to stats[0] (sequence steps) and stats[1] (pages kept).
extern "C" __global__ void attn_ps_compact(
    const int* __restrict__ bt, const int* __restrict__ seq_lens,
    const float* __restrict__ scores, int* __restrict__ out_bt, int* __restrict__ out_lens,
    unsigned long long* stats, unsigned rows_per_seq, unsigned row_stride, unsigned score_stride,
    unsigned bs, unsigned pb, unsigned sink_pages, unsigned recent_tokens, unsigned topk,
    unsigned min_ctx, unsigned count_stats) {
    const unsigned g = blockIdx.x;
    const unsigned tid = threadIdx.x;
    const unsigned r0 = g * rows_per_seq;
    const int e_lo_i = seq_lens[r0];
    const int e_hi_i = seq_lens[r0 + rows_per_seq - 1u];
    const unsigned pt = pb * bs;
    unsigned rp = 0u, cand = 0u, nb_hi = 0u;
    if (e_lo_i > 0 && e_hi_i > 0) {
        rp = recent_page((unsigned)e_lo_i, recent_tokens, pt);
        cand = rp > sink_pages ? rp - sink_pages : 0u;
        nb_hi = ((unsigned)e_hi_i + bs - 1u) / bs;
    }
    const bool sparse = e_lo_i > 0 && e_hi_i > (int)min_ctx && cand > topk
                     && cand <= PS_MAX_CANDIDATES && rp <= score_stride && nb_hi <= row_stride;
    if (!sparse) {
        // The identity view: each row's dense table prefix and length, verbatim.
        for (unsigned r = 0; r < rows_per_seq; ++r) {
            const int e = seq_lens[r0 + r];
            unsigned n = ((unsigned)(e > 1 ? e : 1) + bs - 1u) / bs;
            n = n < row_stride ? n : row_stride;
            const int* src = bt + (unsigned long long)(r0 + r) * row_stride;
            int* dst = out_bt + (unsigned long long)(r0 + r) * row_stride;
            for (unsigned j = tid; j < n; j += blockDim.x) {
                dst[j] = src[j];
            }
            if (tid == 0) {
                out_lens[r0 + r] = e;
            }
        }
        return;
    }

    // ── Exact top-k: keys (sortable score << 32 | page) are distinct, ties go to the newer page.
    unsigned long long keys[PS_KEYS_PER_THREAD];
    const float* sc = scores + (unsigned long long)g * score_stride;
#pragma unroll
    for (unsigned i = 0; i < PS_KEYS_PER_THREAD; ++i) {
        const unsigned c = tid + i * PS_COMPACT_THREADS;
        const unsigned p = sink_pages + c;
        keys[i] = c < cand ? (((unsigned long long)sortable(sc[p]) << 32) | p) : 0ull;
    }
    __shared__ unsigned hist[256];
    __shared__ unsigned long long s_prefix;
    __shared__ unsigned long long s_mask;
    __shared__ unsigned s_need;
    if (tid == 0) {
        s_prefix = 0ull;
        s_mask = 0ull;
        s_need = topk;
    }
    __syncthreads();
    for (int d = 7; d >= 0; --d) {
        const unsigned shift = (unsigned)d * 8u;
        for (unsigned b = tid; b < 256u; b += blockDim.x) {
            hist[b] = 0u;
        }
        __syncthreads();
        const unsigned long long prefix = s_prefix;
        const unsigned long long mask = s_mask;
#pragma unroll
        for (unsigned i = 0; i < PS_KEYS_PER_THREAD; ++i) {
            const unsigned c = tid + i * PS_COMPACT_THREADS;
            if (c < cand && (keys[i] & mask) == prefix) {
                atomicAdd(&hist[(keys[i] >> shift) & 255ull], 1u);
            }
        }
        __syncthreads();
        if (tid == 0) {
            unsigned need = s_need;
            unsigned acc = 0u;
            int b = 255;
            for (; b > 0; --b) {
                if (acc + hist[b] >= need) {
                    break;
                }
                acc += hist[b];
            }
            s_need = need - acc;
            s_prefix = prefix | ((unsigned long long)b << shift);
            s_mask = mask | (255ull << shift);
        }
        __syncthreads();
    }
    const unsigned long long thr = s_prefix;  // the topk-th largest key

    // ── Compaction: sinks, then the selected candidates in page order, then the recent window.
    const unsigned kept_before = sink_pages + topk;
    const int* row0 = bt + (unsigned long long)r0 * row_stride;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;
    __shared__ unsigned warp_tot[32];
    __shared__ unsigned warp_off[32];
    __shared__ unsigned s_tile_total;
    unsigned base = 0u;
#pragma unroll
    for (unsigned i = 0; i < PS_KEYS_PER_THREAD; ++i) {
        if (i * PS_COMPACT_THREADS >= cand) {
            break;
        }
        const unsigned c = tid + i * PS_COMPACT_THREADS;
        const bool sel = c < cand && keys[i] >= thr;
        const unsigned ballot = __ballot_sync(0xffffffffu, sel);
        if (lane == 0) {
            warp_tot[warp] = __popc(ballot);
        }
        __syncthreads();
        if (warp == 0) {
            const unsigned v = warp_tot[lane];
            unsigned incl = v;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned n = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= (unsigned)o) {
                    incl += n;
                }
            }
            warp_off[lane] = incl - v;
            if (lane == 31) {
                s_tile_total = incl;
            }
        }
        __syncthreads();
        if (sel) {
            const unsigned rank = base + warp_off[warp] + __popc(ballot & ((1u << lane) - 1u));
            const unsigned o = sink_pages + rank;
            const unsigned p = sink_pages + c;
            for (unsigned r = 0; r < rows_per_seq; ++r) {
                int* dst = out_bt + (unsigned long long)(r0 + r) * row_stride + o * pb;
                for (unsigned j = 0; j < pb; ++j) {
                    dst[j] = row0[p * pb + j];
                }
            }
        }
        const unsigned tile_total = s_tile_total;
        __syncthreads();
        base += tile_total;
    }
    const unsigned sink_blocks = sink_pages * pb;
    const unsigned recent_first = rp * pb;
    const unsigned recent_blocks = nb_hi - recent_first;
    for (unsigned r = 0; r < rows_per_seq; ++r) {
        int* dst = out_bt + (unsigned long long)(r0 + r) * row_stride;
        for (unsigned j = tid; j < sink_blocks; j += blockDim.x) {
            dst[j] = row0[j];
        }
        for (unsigned j = tid; j < recent_blocks; j += blockDim.x) {
            dst[kept_before * pb + j] = row0[recent_first + j];
        }
        if (tid == 0) {
            out_lens[r0 + r] = (int)(kept_before * pt + ((unsigned)seq_lens[r0 + r] - rp * pt));
        }
    }
    if (count_stats && tid == 0) {
        const unsigned pages = (nb_hi + pb - 1u) / pb;
        atomicAdd(&stats[0], 1ull);
        atomicAdd(&stats[1], (unsigned long long)(kept_before + (pages - rp)));
    }
}
