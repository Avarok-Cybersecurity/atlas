// SPDX-License-Identifier: AGPL-3.0-only
//
// GLM-5.3-Flash DSA (DeepSeek Sparse Attention) — kpool indexer + NoPE MLA. Slice 8.
//
// Pipeline, in the order it must be proven:
//   dsa_kpool_compress    -> pool keys / pool token ids / pool validity
//   dsa_index_scores      -> per-(query, pool) score, with candidate masking
//   dsa_topk_pools        -> deterministic top-k over pools
//   dsa_expand_selection  -> pools -> raw token indices, tail appended, -1 padded
//   dsa_topk_to_mask      -> per-query visibility mask (duplicates collapse)
//   dsa_mla_masked_attn   -> NoPE MLA over the selected tokens
//
// ★ THE SENTINEL CONTRACT. `-1` marks an invalid index and **every destination is fully
//   written before anything else happens**. vLLM's day-0 GLM DSA defect was a `torch.empty`
//   top-k buffer whose tail was never written when the valid pool count fell below the budget,
//   so uninitialised memory was consumed as token indices. `dsa_expand_selection` fills the
//   whole row with -1 and __syncthreads() before a single real index is stored, which makes
//   that state unreachable rather than merely unlikely.
//
// ★ NoPE. `qk_rope_head_dim == 0`, so `qk_head_dim == qk_nope_head_dim` and there is no rope
//   section to skip, pad or rotate. `scale` is passed in rather than derived, so a checkpoint
//   with a real rope section cannot silently reuse this path.

#include <cuda_bf16.h>
#include <math_constants.h>
#include <float.h>
#include <limits.h>

#define DSA_INVALID (-1)

__device__ __forceinline__ float dsa_block_sum(float v, float* smem, unsigned tid, unsigned nthreads) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    if ((tid & 31u) == 0u) smem[tid >> 5] = v;
    __syncthreads();
    if (tid < 32u) {
        float x = (tid < ((nthreads + 31u) / 32u)) ? smem[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffff, x, off);
        if (tid == 0u) smem[0] = x;
    }
    __syncthreads();
    return smem[0];
}

// ── 1. kpool compression ────────────────────────────────────────────────────────────────
// One block per pool, one thread per channel. The softmax runs over the POOL-SLOT axis,
// independently per channel -- NOT over head_dim and NOT over pools. A wrong axis still
// produces a well-formed weighted average, so this is asserted by the oracle, not by shape.
//
// Pools are formed from `first_key + p*KP + s`: pooling starts at the first VALID token, so
// left padding is skipped rather than pooled. A pool counts only if EVERY slot is valid, which
// is why a trailing partial pool is never a pool.

// ── Replay-safe geometry ────────────────────────────────────────────────────────────────
//
// 🔴 Under CUDA-graph capture every scalar below is FROZEN at capture time, and the DSA
// selector's scalars all grow with the context: S, the pool count, the padded sort axis and
// select_k. A replayed graph would select over the capture-time context for ever.
//
// So each selector kernel optionally takes `geom` — a 5-int DEVICE vector written once per
// step by `dsa_write_geom` from the same `seq_len` the attention metadata already carries.
// When `geom` is null the scalar arguments are used verbatim and the kernel is exactly what
// it was; the eager path passes null and is bit-identical by construction.
//
// Blocks past the live pool count return immediately, so the launcher can fix the grid at the
// context CEILING under capture and let one graph serve every length.
#define DSA_GEOM_S        0   // tokens resident in the indexer cache
#define DSA_GEOM_NPOOLS_F 1   // pools including the trailing partial one
#define DSA_GEOM_NPOOLS   2   // complete pools — the prefix everything downstream reads
#define DSA_GEOM_SELECT_K 3   // pools selected per query
#define DSA_GEOM_NP2      4   // TILE width the top-k select walks the pool axis in

// One thread. Derives the selector geometry on device from `seq_len` (the same `[seq_len+1]`
// i32 the attention metadata holds), so nothing about the pass is decided on the host.
extern "C" __global__ void dsa_write_geom(
    const int* __restrict__ seq_len,   // [1] tokens in the cache
    int* __restrict__ geom,            // [5] out
    unsigned int KP,
    unsigned int topk,                 // index_topk
    unsigned int tile                  // dsa_topk_pools tile width (power of two, >= select_k)
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int S = seq_len[0];
    const int np = S / (int)KP;
    // 🔴 The sort axis no longer grows with the context: `dsa_topk_pools` walks the pools in
    // fixed `tile`-wide passes, so this is the tile, clamped down for a short context.
    int np2 = 2;
    while (np2 < np && np2 < (int)tile) np2 <<= 1;
    const int cap = (int)(topk / KP);
    geom[DSA_GEOM_S] = S;
    geom[DSA_GEOM_NPOOLS_F] = (S + (int)KP - 1) / (int)KP;
    geom[DSA_GEOM_NPOOLS] = np;
    geom[DSA_GEOM_SELECT_K] = np < cap ? np : cap;
    geom[DSA_GEOM_NP2] = np2;
}

// Copy one staged indexer row into the cache at a DEVICE-side position, and mark it valid.
//
// 🔴 The write offset used to be `k_normed.offset(pos * D * 2)` computed on the HOST, which is
// exactly what froze under graph capture: every replayed step rewrote the capture-time row.
// The projections now land in a fixed staging row and this kernel places them.
extern "C" __global__ void dsa_indexer_store(
    const __nv_bfloat16* __restrict__ stage_k,     // [D]
    const __nv_bfloat16* __restrict__ stage_gate,  // [D]
    const int* __restrict__ pos,                   // [1] row to write
    __nv_bfloat16* __restrict__ k_normed,          // [capacity, D]
    __nv_bfloat16* __restrict__ gate,              // [capacity, D]
    unsigned char* __restrict__ valid,             // [capacity]
    unsigned int D
) {
    const size_t base = (size_t)pos[0] * D;
    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        k_normed[base + d] = stage_k[d];
        gate[base + d] = stage_gate[d];
    }
    if (threadIdx.x == 0) valid[pos[0]] = 1;
}

extern "C" __global__ void dsa_kpool_compress(
    const __nv_bfloat16* __restrict__ k,      // [S, D] bf16
    const __nv_bfloat16* __restrict__ gate,   // [S, D] bf16
    const unsigned char* __restrict__ valid,  // [S]
    const float* __restrict__ ape,            // [KP, D] fp32
    float* __restrict__ pool_keys,            // [P, D] fp32
    int* __restrict__ pool_indices,           // [P, KP] int32, -1 where the slot is not real
    unsigned char* __restrict__ pool_valid,   // [P]
    unsigned int S,
    unsigned int D,
    unsigned int KP,
    int first_key,
    const int* __restrict__ geom              // [5] or null — see "Replay-safe geometry"
) {
    const unsigned int p = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        // Grid is fixed at the context ceiling under capture; this pool is not live yet.
        if (p >= (unsigned int)geom[DSA_GEOM_NPOOLS_F]) return;
    }

    // Slot bookkeeping is identical for every channel, so thread 0 owns the index/validity write.
    bool all_valid = true;
    for (unsigned int s = 0; s < KP; ++s) {
        long long raw = (long long)first_key + (long long)p * KP + s;
        bool in_range = raw >= 0 && raw < (long long)S;
        bool ok = in_range && valid[in_range ? (unsigned)raw : 0] != 0;
        all_valid &= ok;
        if (tid == 0) pool_indices[p * KP + s] = ok ? (int)raw : DSA_INVALID;
    }
    if (tid == 0) pool_valid[p] = all_valid ? 1 : 0;

    for (unsigned int d = tid; d < D; d += blockDim.x) {
        float mx = -CUDART_INF_F;
        float lg[8];
        for (unsigned int s = 0; s < KP && s < 8; ++s) {
            long long raw = (long long)first_key + (long long)p * KP + s;
            bool in_range = raw >= 0 && raw < (long long)S;
            bool ok = in_range && valid[in_range ? (unsigned)raw : 0] != 0;
            lg[s] = ok ? (__bfloat162float(gate[(size_t)raw * D + d]) + ape[s * D + d])
                       : -CUDART_INF_F;
            mx = fmaxf(mx, lg[s]);
        }
        float sum = 0.0f;
        for (unsigned int s = 0; s < KP && s < 8; ++s) {
            lg[s] = (lg[s] == -CUDART_INF_F) ? 0.0f : __expf(lg[s] - mx);
            sum += lg[s];
        }
        // A fully invalid pool softmaxes to NaN in torch; HF nan_to_num's it to zero.
        float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        float acc = 0.0f;
        for (unsigned int s = 0; s < KP && s < 8; ++s) {
            long long raw = (long long)first_key + (long long)p * KP + s;
            bool in_range = raw >= 0 && raw < (long long)S;
            bool ok = in_range && valid[in_range ? (unsigned)raw : 0] != 0;
            if (ok) acc += lg[s] * inv * __bfloat162float(k[(size_t)raw * D + d]);
        }
        pool_keys[p * D + d] = acc;
    }
}

// ── 2. per-(query, pool) index score ────────────────────────────────────────────────────
// One block per (pool, query). `weights` must already carry the index_heads^-0.5 factor.
// ReLU is applied AFTER the scale; relu(s*x) == s*relu(x) for s > 0, so the two orders agree,
// but only because the scale is positive.
extern "C" __global__ void dsa_index_scores(
    const float* __restrict__ q,              // [Q, H, D] fp32
    const float* __restrict__ pool_keys,      // [P, D] fp32
    const float* __restrict__ weights,        // [Q, H] fp32, pre-scaled
    const int* __restrict__ pool_indices,     // [P, KP]
    const unsigned char* __restrict__ pool_valid,   // [P]
    const unsigned char* __restrict__ valid_keys,   // [S]
    const int* __restrict__ q_pos,            // [Q] absolute position of each query
    float* __restrict__ out,                  // [Q, P] fp32
    unsigned char* __restrict__ valid_cand,   // [Q, P]
    unsigned int Q,
    unsigned int P,
    unsigned int H,
    unsigned int D,
    unsigned int KP,
    unsigned int S,
    float scale,
    const int* __restrict__ geom              // [5] or null
) {
    const unsigned int p = blockIdx.x;
    const unsigned int r = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    extern __shared__ float sh[];
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        // 🪤 `P` is also the row stride of `out`/`valid_cand`. Legal to vary ONLY because the
        // device-geometry path is decode-only (Q == 1), so every `r * P` is `0 * P`. The
        // launcher refuses Q > 1 rather than writing a stride the next step disagrees with.
        if (p >= P) return;
    }

    // A pool is a candidate only when it is complete AND its LAST token is visible to this
    // query (causal + not padding). The clamp mirrors HF's `pool_end.clamp(0, kv_len-1)`.
    int end = pool_indices[p * KP + KP - 1];
    int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
    bool vis = (end_c <= q_pos[r]) && (valid_keys[end_c] != 0);
    bool cand = (pool_valid[p] != 0) && vis;
    if (tid == 0) valid_cand[(size_t)r * P + p] = cand ? 1 : 0;
    if (!cand) {
        if (tid == 0) out[(size_t)r * P + p] = -FLT_MAX;
        return;
    }

    float acc = 0.0f;
    for (unsigned int h = 0; h < H; ++h) {
        float dot = 0.0f;
        for (unsigned int d = tid; d < D; d += blockDim.x)
            dot += q[((size_t)r * H + h) * D + d] * pool_keys[(size_t)p * D + d];
        dot = dsa_block_sum(dot, sh, tid, blockDim.x);
        if (tid == 0) acc += weights[(size_t)r * H + h] * fmaxf(scale * dot, 0.0f);
        __syncthreads();
    }
    if (tid == 0) out[(size_t)r * P + p] = acc;
}

// ── 3. deterministic top-k over pools ───────────────────────────────────────────────────
// One block per query. TILED bitonic select: the pool axis is walked in fixed-size tiles and
// a running best-`T` list is kept in shared memory, so shared memory no longer scales with
// the context.
//
// ★ TIEBREAK IS PART OF THE CONTRACT: score DESCENDING, then pool index ASCENDING.
//   `torch.topk`'s tie order is implementation-defined, so the reference's own pool identities
//   are not a legal target on a tied row -- Atlas pins a total order instead so it is at least
//   reproducible, and the oracle compares the selected SET.
//
// ★ The tiled result is BIT-IDENTICAL to the old whole-axis sort. The comparator is a total
//   order (score, then index -- and indices are unique), so the top-`select_k` prefix is
//   unique; merge-and-truncate over tiles cannot reach a different set or a different order.
//
// Capacity: `NP2` is now the TILE width, not the padded pool count. The block holds two
// tiles (running best + candidate) of [f32, i32], i.e. `16 * NP2` bytes; against the 49,152 B
// runtime ceiling that is `NP2 <= 2048`. `select_k` must be <= `NP2`; the launcher checks it.
// Context is bounded by the indexer cache allocation now, not by this kernel.
//
// Each tile costs one bitonic sort of the tile (log^2 T stages) plus one merge (log T), so
// the walk is O(P/T * log^2 T) -- linear in context, with no shared-memory growth.
extern "C" __global__ void dsa_topk_pools(
    const float* __restrict__ scores,   // [Q, P]
    int* __restrict__ selected,         // [Q, select_k]
    unsigned int Q,
    unsigned int P,
    unsigned int NP2,                   // TILE width: min(next_pow2(P), DSA_TOPK_TILE)
    unsigned int select_k,
    const int* __restrict__ geom        // [5] or null
) {
    if (geom) {
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        NP2 = (unsigned int)geom[DSA_GEOM_NP2];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    // Dynamic shared memory is requested at the CEILING under capture; the layout and the
    // walk still run over this step's `NP2` and `P`, so the result is the eager result.
    extern __shared__ char raw_sh[];
    const unsigned int T = NP2;
    float* sv = (float*)raw_sh;                                  // [2T] best | candidate
    int*   si = (int*)(raw_sh + (size_t)(2 * T) * sizeof(float)); // [2T]
    const unsigned int r = blockIdx.x;
    const unsigned int tid = threadIdx.x;

    // Running best-T, sorted DESCENDING. Starts empty: pad sorts last on BOTH keys.
    for (unsigned int i = tid; i < T; i += blockDim.x) {
        sv[i] = -FLT_MAX;
        si[i] = INT_MAX;
    }
    __syncthreads();

#define DSA_TOPK_GT(a, b) ((sv[(a)] > sv[(b)]) || (sv[(a)] == sv[(b)] && si[(a)] < si[(b)]))
#define DSA_TOPK_SWAP(a, b)                                                                  \
    do {                                                                                     \
        float tv_ = sv[(a)]; sv[(a)] = sv[(b)]; sv[(b)] = tv_;                               \
        int   ti_ = si[(a)]; si[(a)] = si[(b)]; si[(b)] = ti_;                               \
    } while (0)

    for (unsigned int base = 0; base < P; base += T) {
        // Candidate tile into [T, 2T). Short tiles pad with the same sorts-last sentinel.
        for (unsigned int i = tid; i < T; i += blockDim.x) {
            unsigned int idx = base + i;
            sv[T + i] = (idx < P) ? scores[(size_t)r * P + idx] : -FLT_MAX;
            si[T + i] = (idx < P) ? (int)idx : INT_MAX;
        }
        __syncthreads();

        // Sort the candidate tile DESCENDING -- the same network as before, offset by T.
        for (unsigned int k = 2; k <= T; k <<= 1) {
            for (unsigned int j = k >> 1; j > 0; j >>= 1) {
                for (unsigned int i = tid; i < T; i += blockDim.x) {
                    unsigned int l = i ^ j;
                    if (l > i) {
                        bool gt = DSA_TOPK_GT(T + i, T + l);
                        bool want_desc = ((i & k) == 0);
                        if (want_desc != gt) DSA_TOPK_SWAP(T + i, T + l);
                    }
                }
                __syncthreads();
            }
        }

        // Batcher half-cleaner across the two DESCENDING runs: pairing best[i] with
        // cand[T-1-i] leaves the global top-T in [0, T) -- bitonic, not yet sorted.
        for (unsigned int i = tid; i < T; i += blockDim.x) {
            unsigned int a = i, b = T + (T - 1 - i);
            if (!DSA_TOPK_GT(a, b)) DSA_TOPK_SWAP(a, b);
        }
        __syncthreads();

        // Bitonic merge restores DESCENDING order over [0, T) for the next tile's half-cleaner.
        for (unsigned int j = T >> 1; j > 0; j >>= 1) {
            for (unsigned int i = tid; i < T; i += blockDim.x) {
                unsigned int l = i ^ j;
                if (l > i && !DSA_TOPK_GT(i, l)) DSA_TOPK_SWAP(i, l);
            }
            __syncthreads();
        }
    }

#undef DSA_TOPK_GT
#undef DSA_TOPK_SWAP

    for (unsigned int i = tid; i < select_k; i += blockDim.x)
        selected[(size_t)r * select_k + i] = (si[i] == INT_MAX) ? DSA_INVALID : si[i];
}

// ── 4. expand selected pools into raw token indices ─────────────────────────────────────
// ★ The row is filled with -1 by every thread and __syncthreads()'d BEFORE any real index is
//   written. Short rows, invalid pools and a missing tail therefore all leave the sentinel;
//   there is no code path that leaves a byte of this destination unwritten.
extern "C" __global__ void dsa_expand_selection(
    const int* __restrict__ selected,               // [Q, select_k]
    const int* __restrict__ pool_indices,           // [P, KP]
    const unsigned char* __restrict__ valid_cand,   // [Q, P]
    const unsigned char* __restrict__ valid_keys,   // [S]
    const int* __restrict__ q_pos,                  // [Q]
    const unsigned char* __restrict__ q_mask,       // [Q]
    int* __restrict__ out,                          // [Q, width]
    unsigned int Q,
    unsigned int P,
    unsigned int KP,
    unsigned int S,
    unsigned int select_k,
    unsigned int width,
    int first_key,
    int always_tail,
    const int* __restrict__ geom                    // [5] or null
) {
    const unsigned int r = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    int* row = out + (size_t)r * width;

    for (unsigned int i = tid; i < width; i += blockDim.x) row[i] = DSA_INVALID;
    __syncthreads();
    if (q_mask[r] == 0) return;   // a padded query selects nothing, and the row stays all -1

    for (unsigned int j = tid; j < select_k; j += blockDim.x) {
        int p = selected[(size_t)r * select_k + j];
        bool ok = (p >= 0) && (valid_cand[(size_t)r * P + p] != 0);
        for (unsigned int s = 0; s < KP; ++s) {
            unsigned int w = j * KP + s;
            if (w < width) row[w] = ok ? pool_indices[(size_t)p * KP + s] : DSA_INVALID;
        }
    }

    if (always_tail && tid == 0) {
        // The in-progress (incomplete) pool, as raw indices.
        int vis_count = 0;
        for (unsigned int t = 0; t < S; ++t)
            if ((int)t <= q_pos[r] && valid_keys[t] != 0) ++vis_count;
        int tail_count = vis_count % (int)KP;
        int tail_start = first_key + vis_count - tail_count;
        unsigned int base = select_k * KP;
        for (unsigned int t = 0; t + 1 < KP; ++t) {
            long long idx = (long long)tail_start + t;
            bool ok = ((int)t < tail_count) && idx >= 0 && idx < (long long)S
                      && ((int)idx <= q_pos[r]) && valid_keys[(unsigned)idx] != 0;
            if (base + t < width) row[base + t] = ok ? (int)idx : DSA_INVALID;
        }
    }
}

// ── 5. index row -> visibility mask ─────────────────────────────────────────────────────
// Mirrors HF's `scatter_add(...).ne(0)`: duplicates COLLAPSE, so a repeated token is attended
// once. Out-of-range and -1 entries are dropped.
extern "C" __global__ void dsa_topk_to_mask(
    const int* __restrict__ topk,        // [Q, width]
    unsigned char* __restrict__ mask,    // [Q, S]
    unsigned int Q,
    unsigned int width,
    unsigned int S
) {
    const unsigned int r = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    unsigned char* row = mask + (size_t)r * S;
    for (unsigned int i = tid; i < S; i += blockDim.x) row[i] = 0;
    __syncthreads();
    for (unsigned int j = tid; j < width; j += blockDim.x) {
        int i = topk[(size_t)r * width + j];
        if (i >= 0 && i < (int)S) row[i] = 1;   // idempotent: duplicates collapse
    }
}

// ── 6. NoPE MLA over the selected tokens ────────────────────────────────────────────────
// One block per (query, head).
//
// ★ Parallelism is over KEYS for the scores and over DIMS for the accumulation, with the score
//   row staged in shared memory in between. The obvious alternative -- parallelise over dims and
//   block-reduce per key -- costs one __syncthreads() PER KEY, i.e. thousands of barriers per
//   block, and is orders of magnitude slower. Measured the hard way.
//
// Capacity: the score row is `S` floats of shared memory, so at the 49,152 B runtime ceiling this
// kernel handles S <= 12,288 keys. It does not silently truncate -- the launcher refuses.
//
// NoPE: `qd` is the FULL qk head dim and equals qk_nope_head_dim because the rope section is
// zero-width. `scale` is an argument, never derived here, so a checkpoint with a real rope
// section cannot quietly reuse this path.
extern "C" __global__ void dsa_mla_masked_attn(
    const __nv_bfloat16* __restrict__ q,   // [Q, H, qd] bf16
    const __nv_bfloat16* __restrict__ k,   // [S, H, qd] bf16
    const __nv_bfloat16* __restrict__ v,   // [S, H, vd] bf16
    const unsigned char* __restrict__ mask,// [Q, S]
    float* __restrict__ out,               // [Q, H, vd] fp32
    unsigned int Q,
    unsigned int S,
    unsigned int H,
    unsigned int qd,
    unsigned int vd,
    float scale,
    // 🪤 HF's eager path computes `matmul(q, k) * scaling` in the MODULE dtype and only then
    // upcasts for the softmax, so on a bf16 module the pre-softmax score is bf16-rounded. Keeping
    // the score in fp32 is *more* accurate and still WRONG against that reference — visibly so on
    // short sequences, where the softmax has few terms and cannot average the difference away.
    unsigned int round_scores_bf16
) {
    extern __shared__ float sc[];          // [S] score row
    __shared__ float red[32];
    __shared__ float s_m, s_l;
    const unsigned int r = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const __nv_bfloat16* qrow = q + ((size_t)r * H + h) * qd;
    const unsigned char* mrow = mask + (size_t)r * S;

    // ── scores, parallel over keys ──
    float local_max = -CUDART_INF_F;
    for (unsigned int t = tid; t < S; t += blockDim.x) {
        if (mrow[t] == 0) { sc[t] = -CUDART_INF_F; continue; }
        float dot = 0.0f;
        const __nv_bfloat16* krow = k + ((size_t)t * H + h) * qd;
        for (unsigned int d = 0; d < qd; ++d)
            dot += __bfloat162float(qrow[d]) * __bfloat162float(krow[d]);
        float sv = dot * scale;
        if (round_scores_bf16) sv = __bfloat162float(__float2bfloat16(sv));
        sc[t] = sv;
        local_max = fmaxf(local_max, sv);
    }
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));
    if ((tid & 31u) == 0u) red[tid >> 5] = local_max;
    __syncthreads();
    if (tid < 32u) {
        float x = (tid < ((blockDim.x + 31u) / 32u)) ? red[tid] : -CUDART_INF_F;
        for (int off = 16; off > 0; off >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, off));
        if (tid == 0u) s_m = x;
    }
    __syncthreads();

    // ── exponentiate in place, sum ──
    const float m = s_m;
    float local_sum = 0.0f;
    for (unsigned int t = tid; t < S; t += blockDim.x) {
        float e = (sc[t] == -CUDART_INF_F) ? 0.0f : __expf(sc[t] - m);
        sc[t] = e;
        local_sum += e;
    }
    local_sum = dsa_block_sum(local_sum, red, tid, blockDim.x);
    if (tid == 0) s_l = local_sum;
    __syncthreads();

    // ── weighted value sum, parallel over dims ──
    const float inv = (s_l > 0.0f) ? (1.0f / s_l) : 0.0f;
    for (unsigned int d = tid; d < vd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int t = 0; t < S; ++t) {
            float p = sc[t];
            if (p != 0.0f) acc += p * __bfloat162float(v[((size_t)t * H + h) * vd + d]);
        }
        out[((size_t)r * H + h) * vd + d] = acc * inv;
    }
}

// ── 1b. pool-axis compaction ────────────────────────────────────────────────────────────
// ★ HF returns `pool_keys[:, keep]` where `keep = pool_valid.any(0)` — pools that are invalid
//   for EVERY batch element are DROPPED from the axis entirely. That is not cosmetic: it shrinks
//   `n_pools`, and `select_k = min(index_topk / index_kpool, n_pools)` is computed from the
//   compacted count. So the selection budget is DATA-DEPENDENT — a 7-token sequence has one pool
//   and a budget of one, not the two the raw grid would suggest.
//
// `keep` is derived from padding + sequence length only, so the caller computes it; this kernel
// just gathers, keeping the compacted arrays on-device.
extern "C" __global__ void dsa_compact_pools(
    const float* __restrict__ keys_in,          // [P_full, D]
    const int* __restrict__ idx_in,             // [P_full, KP]
    const unsigned char* __restrict__ valid_in, // [P_full]
    const int* __restrict__ keep,               // [P_kept] original pool ids
    float* __restrict__ keys_out,               // [P_kept, D]
    int* __restrict__ idx_out,                  // [P_kept, KP]
    unsigned char* __restrict__ valid_out,      // [P_kept]
    unsigned int P_kept,
    unsigned int D,
    unsigned int KP
) {
    const unsigned int p = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const int src = keep[p];
    for (unsigned int d = tid; d < D; d += blockDim.x)
        keys_out[(size_t)p * D + d] = keys_in[(size_t)src * D + d];
    for (unsigned int s = tid; s < KP; s += blockDim.x)
        idx_out[(size_t)p * KP + s] = idx_in[(size_t)src * KP + s];
    if (tid == 0) valid_out[p] = valid_in[src];
}
