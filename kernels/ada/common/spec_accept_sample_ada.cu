// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself
// derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0.
// Source: src/ops/kernel/speculative_round.cuh:323-455 (speculative_accept_greedy_drafts_kernel: accept a
// one-hot draft d with probability p(d), else sample the residual with d excluded, bonus pick from the
// column after the last draft) and src/ops/kernel/sampling_device.cuh:79-114 and 401-438 (the splitmix64
// uniform keyed by seed, position and purpose; the higher-value, lower-index candidate order; the
// inverse-CDF pick with one excluded token). Changes: the sampling semantics are re-derived for Atlas
// `SamplingParams` (temperature, top-k, top-p, min-p as `spark_runtime::sampler` applies them, no top-20
// cap: an uncapped top-k with top-p or min-p is served exactly when its support closes within 64
// candidates, else the row reports a fallback); exact top-k by radix select over order keys; the
// pre-sample pipeline's masks, penalties and biases arrive as a sparse edit list; greedy rows reproduce
// the host first-index-wins argmax bit for bit; one CTA per (position, regime) job; 64-bit positions;
// by Atlas contributors.
//
// Ada (sm_89) device-side speculative acceptance (lane L12, row `mtp_device_accept`).
//
// The MTP verify path used to copy every verify row ([K, vocab] BF16, 0.5 MB per row on a 248K vocab) to
// the host, dequantize it, run the pre-sample pipeline and argmax or sample there. For the rows this
// kernel accepts (no grammar, no logit_bias, no LZ/DRY, no stateful thinking stage; the host planner in
// `spark-server/src/scheduler/verify_pipeline_helper/device_accept/plan.rs` decides) it does the same on
// the device and returns two u32 per job.
//
// ── Inputs ────────────────────────────────────────────────────────────────
//   logits  [n_pos, vocab] BF16, row stride vocab (this sequence's verify rows)
//   blob    the request, laid out by `crates/spark-model/src/port/spec_accept/blob.rs`:
//           words 0..16 header, 16..48 drafts (u32::MAX = none), 48.. edit descriptors (6 words each,
//           regime 0 then regime 1, each sorted by token id)
//   out     [jobs, 2] u32: pick, status bits (1 nonfinite seen, 2 empty, 4 support, 8 draft accepted)
//
// Job j = regime * n_pos + pos. Regime 0 is the sequence's state at step start; regime 1 exists only
// when the sequence is inside thinking and is the state after a `</think>` pick (the host walks the
// positions and switches regime after the first `</think>`, as `pick_positions_from_host` does).
//
// ── Numerics ──────────────────────────────────────────────────────────────
//   * Edited values: bf16 -> f32 (exact), then mask (-inf) or `x - 12` (tool opener suppression), then
//     the repetition penalty once per occurrence in the window (`x > 0 ? x / r : x * r`), then
//     `x - delta` (presence + frequency, delta computed on the host exactly as
//     `freq * count as f32 + pres`), then each bias in order. Every operation is an explicit
//     round-to-nearest intrinsic, so the result is the host's `apply_penalties_and_bias` bit for bit.
//   * Greedy: max over non-NaN values, first index attaining it (-0.0 == +0.0, NaN never wins, all-NaN
//     gives 0): `spark_runtime::sampler::argmax_first_wins_f32`, exactly. The block reduction combines
//     (value, index) pairs with that order, which is associative and commutative.
//   * Sampled: distribution-exact, not bit-identical (the RNG is keyed by
//     seed and position, not the host sampler's stream). Finite values only (the host
//     sampler drops non-finite logits); s = v / T; w = expf(s - s_max). The truncated support follows
//     the host sampler's order of top-k, min-p (w >= min_p * w_max), top-p (cumulative w / sum >= top_p,
//     inclusive), in the same sequential f32 arithmetic on one thread; the parallel sums (full-vocabulary
//     normalizers) differ from a sequential host sum only by rounding. The RNG is a splitmix64 hash of
//     (key, absolute position, purpose): the request seed when it has one, else a fresh host key.
//
// Grid: (jobs, 1, 1). Block: (1024, 1, 1). Dynamic shared memory: ceil(vocab / 32) * 4 bytes (edit
// bitmap) + max_edits * 8 bytes (edited ids and values), at most 48 KiB with the static part (the host
// checks, `spec_accept::max_edits`).

#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int kBlock = 1024;
constexpr int kWarps = kBlock / 32;
constexpr unsigned kFull = 0xffffffffu;
constexpr int kCandCap = 64;
constexpr unsigned kInvalid = 0xffffffffu;
constexpr unsigned kNoDraft = 0xffffffffu;

// Header words (blob.rs `h::*`).
constexpr int H_VOCAB = 0;
constexpr int H_NPOS = 1;
constexpr int H_NREG = 2;
constexpr int H_FLAGS = 3;
constexpr int H_TEMP = 4;
constexpr int H_TOPK = 5;
constexpr int H_TOPP = 6;
constexpr int H_MINP = 7;
constexpr int H_REP = 8;
constexpr int H_EOS_UNTIL = 9;
constexpr int H_KEY_LO = 10;
constexpr int H_KEY_HI = 11;
constexpr int H_POS_LO = 12;
constexpr int H_POS_HI = 13;
constexpr int H_NEDIT0 = 14;
constexpr int H_NEDIT1 = 15;
constexpr int kDraftsAt = 16;
constexpr int kDescAt = 48;
constexpr int kDescWords = 6;

constexpr unsigned F_SAMPLE = 1u;

constexpr unsigned E_MASK = 1u;
constexpr unsigned E_EOS = 2u;
constexpr unsigned E_SUB12 = 4u;
constexpr unsigned E_DELTA = 8u;
constexpr unsigned E_NBIAS_SHIFT = 4u;

constexpr unsigned ST_NONFINITE = 1u;
constexpr unsigned ST_EMPTY = 2u;
constexpr unsigned ST_SUPPORT = 4u;
constexpr unsigned ST_ACCEPTED = 8u;

constexpr unsigned kPurposeAccept = 1u;
constexpr unsigned kPurposeSample = 2u;

__device__ __forceinline__ float sa_neg_inf() { return __int_as_float(0xff800000); }

struct EditView {
    const unsigned* bitmap;
    const unsigned* ids;
    const float* vals;
    unsigned n;
};

// One processed logit: the edited value when the bitmap marks `i`, else the BF16 logit widened.
__device__ __forceinline__ float sa_value(const __nv_bfloat16* __restrict__ row, unsigned i,
                                          const EditView& ev) {
    if ((ev.bitmap[i >> 5] >> (i & 31u)) & 1u) {
        unsigned lo = 0, hi = ev.n;
        while (lo < hi) {
            const unsigned mid = (lo + hi) >> 1;
            if (ev.ids[mid] < i) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return ev.vals[lo];
    }
    return __bfloat162float(row[i]);
}

// The host pipeline's effect on one token: masks, then penalties, then biases (see the header).
__device__ __forceinline__ float sa_apply_edit(float x, const unsigned* __restrict__ d, unsigned pos,
                                               unsigned eos_until, float rep) {
    const unsigned fl = d[1];
    if ((fl & E_MASK) || ((fl & E_EOS) && pos < eos_until)) {
        x = sa_neg_inf();
    } else if (fl & E_SUB12) {
        x = __fsub_rn(x, 12.0f);
    }
    const unsigned reps = d[2];
    for (unsigned c = 0; c < reps; ++c) {
        x = (x > 0.0f) ? __fdiv_rn(x, rep) : __fmul_rn(x, rep);
    }
    if (fl & E_DELTA) {
        x = __fsub_rn(x, __uint_as_float(d[3]));
    }
    const unsigned nb = (fl >> E_NBIAS_SHIFT) & 3u;
    if (nb > 0) {
        x = __fadd_rn(x, __uint_as_float(d[4]));
    }
    if (nb > 1) {
        x = __fadd_rn(x, __uint_as_float(d[5]));
    }
    return x;
}

// Monotone u32 key of a float: a larger key is a larger float (-0.0 sorts below +0.0).
__device__ __forceinline__ unsigned sa_order_key(float v) {
    const unsigned b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Candidate order: higher value first, lower index on a tie (Cinference `sampling_better`).
__device__ __forceinline__ bool sa_better(float v, unsigned i, float bv, unsigned bi) {
    return v > bv || (v == bv && i < bi);
}

__device__ __forceinline__ unsigned long long sa_splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Uniform in [0, 1) with 24 random bits, a pure function of (key, position, purpose), so it is the same
// on every replay. `reference.rs::uniform` is the same function on the host.
__device__ __forceinline__ float sa_uniform(unsigned long long key, unsigned long long position,
                                            unsigned purpose) {
    unsigned long long k = sa_splitmix64(key ^ (position * 0xD1B54A32D192ED03ull));
    k = sa_splitmix64(k ^ (static_cast<unsigned long long>(purpose) << 21));
    return static_cast<float>(static_cast<unsigned>(k >> 40)) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ void sa_block_argmax(float& v, unsigned& i, float* sv, unsigned* si) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(kFull, v, off);
        const unsigned oi = __shfl_down_sync(kFull, i, off);
        if (sa_better(ov, oi, v, i)) {
            v = ov;
            i = oi;
        }
    }
    if (lane == 0) {
        sv[warp] = v;
        si[warp] = i;
    }
    __syncthreads();
    if (warp == 0) {
        v = sv[lane];
        i = si[lane];
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(kFull, v, off);
            const unsigned oi = __shfl_down_sync(kFull, i, off);
            if (sa_better(ov, oi, v, i)) {
                v = ov;
                i = oi;
            }
        }
        if (lane == 0) {
            sv[0] = v;
            si[0] = i;
        }
    }
    __syncthreads();
    v = sv[0];
    i = si[0];
    __syncthreads();
}

__device__ __forceinline__ float sa_block_sum(float x, float* s) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        x = __fadd_rn(x, __shfl_down_sync(kFull, x, off));
    }
    if (lane == 0) {
        s[warp] = x;
    }
    __syncthreads();
    if (warp == 0) {
        x = s[lane];
        for (int off = 16; off > 0; off >>= 1) {
            x = __fadd_rn(x, __shfl_down_sync(kFull, x, off));
        }
        if (lane == 0) {
            s[0] = x;
        }
    }
    __syncthreads();
    const float r = s[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float sa_block_max(float x, float* s) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        x = fmaxf(x, __shfl_down_sync(kFull, x, off));
    }
    if (lane == 0) {
        s[warp] = x;
    }
    __syncthreads();
    if (warp == 0) {
        x = s[lane];
        for (int off = 16; off > 0; off >>= 1) {
            x = fmaxf(x, __shfl_down_sync(kFull, x, off));
        }
        if (lane == 0) {
            s[0] = x;
        }
    }
    __syncthreads();
    const float r = s[0];
    __syncthreads();
    return r;
}

// Sum (op 0), max (op 1) or bitwise or (op 2) of one u32 per thread.
__device__ __forceinline__ unsigned sa_block_u32(unsigned x, unsigned* s, int op) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        const unsigned o = __shfl_down_sync(kFull, x, off);
        x = op == 0 ? x + o : (op == 1 ? (o > x ? o : x) : (x | o));
    }
    if (lane == 0) {
        s[warp] = x;
    }
    __syncthreads();
    if (warp == 0) {
        x = s[lane];
        for (int off = 16; off > 0; off >>= 1) {
            const unsigned o = __shfl_down_sync(kFull, x, off);
            x = op == 0 ? x + o : (op == 1 ? (o > x ? o : x) : (x | o));
        }
        if (lane == 0) {
            s[0] = x;
        }
    }
    __syncthreads();
    const unsigned r = s[0];
    __syncthreads();
    return r;
}

// Inverse-CDF pick over a sorted support of n entries held by thread 0, `exclude` skipped (Cinference
// `sampling_pick_from_support`): the first entry whose running weight exceeds u * mass.
__device__ __forceinline__ unsigned sa_pick_support(const unsigned* ids, const float* w, int n,
                                                    unsigned exclude, float u) {
    float mass = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (ids[j] != exclude) {
            mass = __fadd_rn(mass, w[j]);
        }
    }
    if (!(mass > 0.0f)) {
        return ids[0];
    }
    const float goal = __fmul_rn(u, mass);
    float acc = 0.0f;
    unsigned picked = ids[0];
    for (int j = 0; j < n; ++j) {
        if (ids[j] == exclude) {
            continue;
        }
        acc = __fadd_rn(acc, w[j]);
        picked = ids[j];
        if (goal < acc) {
            return picked;
        }
    }
    return picked;
}

// Weight of one processed value under temperature t and max scaled logit smax.
__device__ __forceinline__ float sa_weight(float v, float t, float smax) {
    return expf(__fsub_rn(__fdiv_rn(v, t), smax));
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kBlock)
    spec_accept_rows_ada(const __nv_bfloat16* __restrict__ logits, const unsigned* __restrict__ blob,
                         unsigned* __restrict__ out) {
    extern __shared__ unsigned sa_dyn[];
    __shared__ float s_f[kWarps];
    __shared__ unsigned s_u[kWarps];
    __shared__ unsigned s_hist[256];
    __shared__ unsigned s_cand_idx[kCandCap];
    __shared__ float s_cand_val[kCandCap];
    __shared__ float s_w[kCandCap];
    __shared__ unsigned s_ctl[8];
    __shared__ float s_ctl_f[2];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const unsigned vocab = blob[H_VOCAB];
    const unsigned n_pos = blob[H_NPOS];
    const unsigned job = blockIdx.x;
    const unsigned pos = job % n_pos;
    const unsigned reg = job / n_pos;
    const unsigned flags = blob[H_FLAGS];
    const float temp = __uint_as_float(blob[H_TEMP]);
    const unsigned top_k = blob[H_TOPK];
    const float top_p = __uint_as_float(blob[H_TOPP]);
    const float min_p = __uint_as_float(blob[H_MINP]);
    const float rep = __uint_as_float(blob[H_REP]);
    const unsigned eos_until = blob[H_EOS_UNTIL];
    const unsigned long long key =
        (static_cast<unsigned long long>(blob[H_KEY_HI]) << 32) | blob[H_KEY_LO];
    const unsigned long long pos_abs =
        ((static_cast<unsigned long long>(blob[H_POS_HI]) << 32) | blob[H_POS_LO]) + pos;
    const unsigned n_edit0 = blob[H_NEDIT0];
    const unsigned n_edit1 = blob[H_NREG] > 1 ? blob[H_NEDIT1] : 0u;
    const unsigned n_edits = reg == 0 ? n_edit0 : n_edit1;
    const unsigned n_edit_max = n_edit0 > n_edit1 ? n_edit0 : n_edit1;
    const unsigned* desc = blob + kDescAt + (reg == 0 ? 0u : n_edit0 * kDescWords);
    const unsigned draft = blob[kDraftsAt + pos];

    const __nv_bfloat16* __restrict__ row = logits + static_cast<size_t>(pos) * vocab;
    const unsigned bitmap_words = (vocab + 31u) >> 5;
    unsigned* bitmap = sa_dyn;
    unsigned* e_ids = bitmap + bitmap_words;
    float* e_vals = reinterpret_cast<float*>(e_ids + n_edit_max);

    // ── Edits: the bitmap of touched ids and their processed values for this position ──
    for (unsigned w = tid; w < bitmap_words; w += kBlock) {
        bitmap[w] = 0u;
    }
    __syncthreads();
    for (unsigned e = tid; e < n_edits; e += kBlock) {
        const unsigned* d = desc + e * kDescWords;
        const unsigned tok = d[0];
        e_ids[e] = tok;
        e_vals[e] = sa_apply_edit(__bfloat162float(row[tok]), d, pos, eos_until, rep);
        atomicOr(&bitmap[tok >> 5], 1u << (tok & 31u));
    }
    __syncthreads();
    const EditView ev{bitmap, e_ids, e_vals, n_edits};

    // ── Phase 1: greedy argmax, finite max and count, non-finite flag ──
    float best_v = sa_neg_inf();
    unsigned best_i = kInvalid;
    float vmax = sa_neg_inf();
    unsigned fin = 0;
    unsigned nonfinite = 0;
    for (unsigned i = tid; i < vocab; i += kBlock) {
        const float v = sa_value(row, i, ev);
        if (sa_better(v, i, best_v, best_i)) {
            best_v = v;
            best_i = i;
        }
        if (isfinite(v)) {
            vmax = fmaxf(vmax, v);
            ++fin;
        } else if (isnan(v) || v > 0.0f) {
            nonfinite = 1u;
        }
    }
    sa_block_argmax(best_v, best_i, s_f, s_u);
    vmax = sa_block_max(vmax, s_f);
    fin = sa_block_u32(fin, s_u, 0);
    nonfinite = sa_block_u32(nonfinite, s_u, 2);
    unsigned status = nonfinite ? ST_NONFINITE : 0u;

    if (!(flags & F_SAMPLE)) {
        if (tid == 0) {
            out[job * 2] = best_i == kInvalid ? 0u : best_i;
            out[job * 2 + 1] = status;
        }
        return;
    }

    if (fin == 0u) {
        if (tid == 0) {
            out[job * 2] = 0u;
            out[job * 2 + 1] = status | ST_EMPTY;
        }
        return;
    }
    const unsigned kk = (top_k > 0u && top_k < fin) ? top_k : 0u;
    if (kk > static_cast<unsigned>(kCandCap)) {
        if (tid == 0) {
            out[job * 2] = 0u;
            out[job * 2 + 1] = status | ST_SUPPORT;
        }
        return;
    }
    const bool truncated = kk > 0u || min_p > 0.0f || top_p < 1.0f;
    const float smax = __fdiv_rn(vmax, temp);
    const float u_acc = sa_uniform(key, pos_abs, kPurposeAccept);
    const float u_res = sa_uniform(key, pos_abs, kPurposeSample);

    if (!truncated) {
        // ── FULL: every finite token, weights in index order (the host's unsorted multinomial) ──
        float total = 0.0f;
        unsigned last_pos = 0u;
        bool any_pos = false;
        for (unsigned i = tid; i < vocab; i += kBlock) {
            const float v = sa_value(row, i, ev);
            if (isfinite(v)) {
                const float w = sa_weight(v, temp, smax);
                total = __fadd_rn(total, w);
                if (w > 0.0f && i != draft) {
                    last_pos = i;
                    any_pos = true;
                }
            }
        }
        total = sa_block_sum(total, s_f);
        last_pos = sa_block_u32(any_pos ? last_pos + 1u : 0u, s_u, 1);  // 0 = none
        float wd = 0.0f;
        if (draft != kNoDraft && draft < vocab) {
            const float vd = sa_value(row, draft, ev);
            if (isfinite(vd)) {
                wd = sa_weight(vd, temp, smax);
            }
        }
        if (draft != kNoDraft && wd > 0.0f && u_acc < __fdiv_rn(wd, total)) {
            if (tid == 0) {
                out[job * 2] = draft;
                out[job * 2 + 1] = status | ST_ACCEPTED;
            }
            return;
        }
        const unsigned exclude = draft != kNoDraft ? draft : kInvalid;
        const float mass = exclude != kInvalid ? __fsub_rn(total, wd) : total;
        const float goal = __fmul_rn(u_res, mass);
        if (tid == 0) {
            s_ctl_f[0] = 0.0f;
            s_ctl[0] = kInvalid;
        }
        __syncthreads();
        for (unsigned base = 0; base < vocab; base += kBlock) {
            const unsigned i = base + tid;
            float w = 0.0f;
            if (i < vocab && i != exclude) {
                const float v = sa_value(row, i, ev);
                if (isfinite(v)) {
                    w = sa_weight(v, temp, smax);
                }
            }
            float x = w;
            for (int off = 1; off < 32; off <<= 1) {
                const float y = __shfl_up_sync(kFull, x, off);
                if (lane >= off) {
                    x = __fadd_rn(x, y);
                }
            }
            if (lane == 31) {
                s_f[warp] = x;
            }
            __syncthreads();
            float before = s_ctl_f[0];
            for (int w2 = 0; w2 < warp; ++w2) {
                before = __fadd_rn(before, s_f[w2]);
            }
            const bool hit = w > 0.0f && goal < __fadd_rn(before, x);
            const unsigned hb = __ballot_sync(kFull, hit);
            if (lane == 0 && hb != 0u) {
                atomicMin(&s_ctl[0], base + static_cast<unsigned>(warp) * 32u + (__ffs(hb) - 1));
            }
            __syncthreads();
            if (s_ctl[0] != kInvalid) {
                break;
            }
            if (tid == 0) {
                float t = s_ctl_f[0];
                for (int w2 = 0; w2 < kWarps; ++w2) {
                    t = __fadd_rn(t, s_f[w2]);
                }
                s_ctl_f[0] = t;
            }
            __syncthreads();
        }
        if (tid == 0) {
            unsigned pick = s_ctl[0];
            if (pick == kInvalid) {
                pick = last_pos > 0u ? last_pos - 1u : (draft != kNoDraft ? draft : 0u);
            }
            out[job * 2] = pick;
            out[job * 2 + 1] = status;
        }
        return;
    }

    // ── TRUNC: exact top-K by radix select over order keys, then the host's sequential filters ──
    const unsigned kc = kk > 0u ? kk : (fin < static_cast<unsigned>(kCandCap) ? fin : kCandCap);
    float sum_full = 0.0f;
    unsigned cnt_full = 0u;
    if (kk == 0u) {
        // min-p and top-p over the whole finite vocabulary need its retained mass and size.
        // The host threshold is min_p * max_prob, and the full set's max weight is expf(0) = 1.
        const float thr = min_p > 0.0f ? min_p : -1.0f;
        for (unsigned i = tid; i < vocab; i += kBlock) {
            const float v = sa_value(row, i, ev);
            if (isfinite(v)) {
                const float w = sa_weight(v, temp, smax);
                if (w >= thr) {
                    sum_full = __fadd_rn(sum_full, w);
                    ++cnt_full;
                }
            }
        }
        sum_full = sa_block_sum(sum_full, s_f);
        cnt_full = sa_block_u32(cnt_full, s_u, 0);
    }

    // Radix select: after it, `need` of the elements with (key & mask) == prefix are taken, together
    // with every element whose masked key is larger (kc - need of them).
    unsigned prefix = 0u, mask = 0u, need = kc, eq = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) {
            s_hist[tid] = 0u;
        }
        __syncthreads();
        for (unsigned base = 0; base < vocab; base += kBlock) {
            const unsigned i = base + tid;
            int bin = -1;
            if (i < vocab) {
                const float v = sa_value(row, i, ev);
                if (isfinite(v)) {
                    const unsigned k = sa_order_key(v);
                    if ((k & mask) == prefix) {
                        bin = static_cast<int>((k >> shift) & 255u);
                    }
                }
            }
            const unsigned peers = __match_any_sync(kFull, bin);
            if (bin >= 0 && lane == __ffs(peers) - 1) {
                atomicAdd(&s_hist[bin], static_cast<unsigned>(__popc(peers)));
            }
        }
        __syncthreads();
        if (tid == 0) {
            unsigned above = 0u;
            int b = 255;
            for (; b > 0; --b) {
                if (above + s_hist[b] >= need) {
                    break;
                }
                above += s_hist[b];
            }
            s_ctl[0] = static_cast<unsigned>(b);
            s_ctl[1] = need - above;
            s_ctl[2] = s_hist[b];
        }
        __syncthreads();
        const unsigned b = s_ctl[0];
        need = s_ctl[1];
        eq = s_ctl[2];
        __syncthreads();
        prefix |= b << shift;
        mask |= 255u << shift;
        if (eq == need) {
            break;
        }
    }

    // Gather: every larger masked key, plus `need` ties (all of them, or the lowest indices).
    const unsigned tie_base = kc - need;
    const bool all_eq = eq == need;
    if (tid == 0) {
        s_ctl[3] = 0u;
        s_ctl[4] = 0u;
    }
    __syncthreads();
    for (unsigned base = 0; base < vocab; base += kBlock) {
        const unsigned i = base + tid;
        int cls = 0;
        float v = 0.0f;
        if (i < vocab) {
            v = sa_value(row, i, ev);
            if (isfinite(v)) {
                const unsigned km = sa_order_key(v) & mask;
                cls = km > prefix ? 1 : (km == prefix ? 2 : 0);
            }
        }
        if (cls == 1) {
            const unsigned slot = atomicAdd(&s_ctl[3], 1u);
            s_cand_idx[slot] = i;
            s_cand_val[slot] = v;
        }
        if (all_eq) {
            if (cls == 2) {
                const unsigned slot = tie_base + atomicAdd(&s_ctl[4], 1u);
                s_cand_idx[slot] = i;
                s_cand_val[slot] = v;
            }
            continue;
        }
        const unsigned tb = __ballot_sync(kFull, cls == 2);
        if (lane == 0) {
            s_u[warp] = static_cast<unsigned>(__popc(tb));
        }
        __syncthreads();
        unsigned rank = s_ctl[4] + static_cast<unsigned>(__popc(tb & ((1u << lane) - 1u)));
        for (int w2 = 0; w2 < warp; ++w2) {
            rank += s_u[w2];
        }
        if (cls == 2 && rank < need) {
            s_cand_idx[tie_base + rank] = i;
            s_cand_val[tie_base + rank] = v;
        }
        __syncthreads();
        if (tid == 0) {
            unsigned t = 0u;
            for (int w2 = 0; w2 < kWarps; ++w2) {
                t += s_u[w2];
            }
            s_ctl[4] += t;
        }
        __syncthreads();
        if (s_ctl[4] >= need && s_ctl[3] == tie_base) {
            break;
        }
    }
    __syncthreads();

    if (tid != 0) {
        return;
    }
    // Sort the kc candidates: higher value first, lower index on a tie.
    for (unsigned a = 1; a < kc; ++a) {
        const float va = s_cand_val[a];
        const unsigned ia = s_cand_idx[a];
        unsigned b = a;
        while (b > 0 && sa_better(va, ia, s_cand_val[b - 1], s_cand_idx[b - 1])) {
            s_cand_val[b] = s_cand_val[b - 1];
            s_cand_idx[b] = s_cand_idx[b - 1];
            --b;
        }
        s_cand_val[b] = va;
        s_cand_idx[b] = ia;
    }
    const float s0 = __fdiv_rn(s_cand_val[0], temp);
    for (unsigned j = 0; j < kc; ++j) {
        s_w[j] = expf(__fsub_rn(__fdiv_rn(s_cand_val[j], temp), s0));
    }
    int cutoff = -1;
    if (kk > 0u) {
        const float thr = min_p > 0.0f ? __fmul_rn(min_p, s_w[0]) : -1.0f;
        unsigned n_ret = 0;
        while (n_ret < kc && s_w[n_ret] >= thr) {
            ++n_ret;
        }
        float sum_ret = 0.0f;
        for (unsigned j = 0; j < n_ret; ++j) {
            sum_ret = __fadd_rn(sum_ret, s_w[j]);
        }
        cutoff = static_cast<int>(n_ret);
        if (top_p < 1.0f) {
            float cum = 0.0f;
            for (unsigned j = 0; j < n_ret; ++j) {
                cum = __fadd_rn(cum, __fdiv_rn(s_w[j], sum_ret));
                if (cum >= top_p) {
                    cutoff = static_cast<int>(j + 1);
                    break;
                }
            }
        }
    } else if (top_p < 1.0f) {
        float cum = 0.0f;
        for (unsigned j = 0; j < kc; ++j) {
            if (j >= cnt_full) {
                cutoff = static_cast<int>(cnt_full);
                break;
            }
            cum = __fadd_rn(cum, __fdiv_rn(s_w[j], sum_full));
            if (cum >= top_p) {
                cutoff = static_cast<int>(j + 1);
                break;
            }
        }
        if (cutoff < 0 && cnt_full <= kc) {
            cutoff = static_cast<int>(cnt_full);
        }
    } else if (cnt_full <= kc) {
        cutoff = static_cast<int>(cnt_full);
    }
    if (cutoff <= 0) {
        out[job * 2] = 0u;
        out[job * 2 + 1] = status | ST_SUPPORT;
        return;
    }
    const int n = cutoff;
    unsigned pick;
    if (draft != kNoDraft) {
        float total = 0.0f;
        float wd = 0.0f;
        for (int j = 0; j < n; ++j) {
            total = __fadd_rn(total, s_w[j]);
            if (s_cand_idx[j] == draft && wd == 0.0f) {
                wd = s_w[j];
            }
        }
        if (wd > 0.0f && u_acc < __fdiv_rn(wd, total)) {
            out[job * 2] = draft;
            out[job * 2 + 1] = status | ST_ACCEPTED;
            return;
        }
        pick = sa_pick_support(s_cand_idx, s_w, n, draft, u_res);
    } else {
        pick = sa_pick_support(s_cand_idx, s_w, n, kInvalid, u_res);
    }
    out[job * 2] = pick;
    out[job * 2 + 1] = status;
}
