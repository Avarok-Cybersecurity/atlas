// SPDX-License-Identifier: AGPL-3.0-only

// DFlash2 candidate selector on device (z-lab `CandidateSelector`).
//
// Replaces the host selector (selector.rs): a ~4 MB logits D2H + scalar
// top-16 scan over gamma-1 rows x 248320 bf16 + a rank x hidden projection
// per row, measured at ~6 ms per propose on GB10. On device the same math
// is three tiny launches and the only D2H left is the gamma x u32 drafts
// the propose already copies.
//
// Contract (mirrors selector.rs::dflash2_selector_pick exactly):
//   1. per draft row t in 1..gamma-1: cand = top-16 logits of row t
//   2. g[t] = hidden_projection[rank, H] @ normed_hidden[t]     (f32 acc)
//   3. greedy chain, prev = anchor (last_token):
//        e      = pred_cb[prev] * g[t]                (elementwise, rank)
//        score  = unary(cand) + dot(succ_cb[cand], e)
//        pick   = argmax score  (strict >, earlier candidate wins ties;
//                 candidates are in descending-unary order, like the host)
//        prev   = pick; drafts[t] = pick
//
// Tie note: the host top-16 uses sort_unstable, so equal-logit tie order is
// UNSPECIFIED there; the device rule (equal value -> smaller token id) is a
// superset-deterministic refinement, not a divergence. gamma <= 16,
// rank <= 256, vocab ~248320 — trivially small launches; clarity over
// cleverness (same stance as dflash2_conv.cu).

#include <cuda_bf16.h>

#define SEL_TOPK 16
#define SEL_BLOCK 256

// Monotone map from f32 bits to an ascending-comparable u32.
static __device__ __forceinline__ unsigned int f32_sortable(float v) {
    unsigned int b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// ── 1. Per-row top-16 over the draft logits ────────────────────────────
// grid.x = rows-1 (block b handles draft row b+1), block = SEL_BLOCK.
// SINGLE global pass: each thread keeps a register-local descending top-16
// of its strided slice (~970 elements), then the block merges the 256 local
// lists via 16 rounds of block-argmax over each thread's current head.
// (The first cut re-scanned global memory once per selected candidate —
// 16 full row reads — and measured ~6.8 ms/propose, erasing the port's win.)
// BATCHED: grid.y = n sequences. Sequence b owns the row band
// [b*gamma, (b+1)*gamma) of every buffer, so grid.y == 1 reduces to the
// single-sequence launch bit-for-bit (b == 0 => all strides vanish).
extern "C" __global__ void dflash2_selector_topk16(
    const __nv_bfloat16* __restrict__ logits,  // [n*gamma, vocab]
    unsigned int* __restrict__ cand_ids,       // [n*gamma, 16]
    float* __restrict__ cand_vals,             // [n*gamma, 16]
    unsigned int vocab,
    unsigned int gamma
) {
    const unsigned int b = blockIdx.y;
    const unsigned int row = b * gamma + blockIdx.x + 1;
    const unsigned int tid = threadIdx.x;
    const __nv_bfloat16* rl = logits + (size_t)row * vocab;

    // Local top-16, descending (keys pack value-desc, id-asc-on-tie).
    unsigned long long local[SEL_TOPK];
#pragma unroll
    for (unsigned int j = 0; j < SEL_TOPK; ++j) {
        local[j] = 0;  // key 0 == -inf sentinel
    }
    for (unsigned int i = tid; i < vocab; i += SEL_BLOCK) {
        float v = __bfloat162float(rl[i]);
        unsigned long long key =
            ((unsigned long long)f32_sortable(v) << 32) | (0xFFFFFFFFu - i);
        if (key > local[SEL_TOPK - 1]) {
            // Insertion into the sorted register array.
            unsigned int j = SEL_TOPK - 1;
            while (j > 0 && local[j - 1] < key) {
                local[j] = local[j - 1];
                --j;
            }
            local[j] = key;
        }
    }

    __shared__ unsigned long long red[SEL_BLOCK];
    __shared__ unsigned int red_tid[SEL_BLOCK];
    __shared__ unsigned int head[SEL_BLOCK];  // consumed count per thread
    head[tid] = 0;
    __syncthreads();

    for (unsigned int k = 0; k < SEL_TOPK; ++k) {
        red[tid] = (head[tid] < SEL_TOPK) ? local[head[tid]] : 0;
        red_tid[tid] = tid;
        __syncthreads();
        for (unsigned int s = SEL_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s && red[tid + s] > red[tid]) {
                red[tid] = red[tid + s];
                red_tid[tid] = red_tid[tid + s];
            }
            __syncthreads();
        }
        if (tid == red_tid[0]) {
            head[tid] += 1;  // winner advances its list
        }
        if (tid == 0) {
            unsigned long long key = red[0];
            unsigned int id = 0xFFFFFFFFu - (unsigned int)(key & 0xFFFFFFFFu);
            cand_ids[row * SEL_TOPK + k] = id;
            cand_vals[row * SEL_TOPK + k] = __bfloat162float(rl[id]);
        }
        __syncthreads();
    }
}

// ── 2. g[t] = hidden_projection @ normed_hidden[t] ─────────────────────
// grid = (rank, rows-1) (y index r handles draft row y+1), block = 128.
extern "C" __global__ void dflash2_selector_proj(
    const __nv_bfloat16* __restrict__ hidden_projection,  // [rank, hidden]
    const __nv_bfloat16* __restrict__ normed_hidden,      // [gamma, hidden]
    float* __restrict__ g_out,                            // [gamma, rank]
    unsigned int rank,
    unsigned int hidden,
    unsigned int gamma
) {
    const unsigned int r = blockIdx.x;
    // BATCHED: grid.z = n sequences; band b owns rows [b*gamma, ..).
    const unsigned int b = blockIdx.z;
    const unsigned int row = b * gamma + blockIdx.y + 1;
    const unsigned int tid = threadIdx.x;
    const __nv_bfloat16* w = hidden_projection + (size_t)r * hidden;
    const __nv_bfloat16* h = normed_hidden + (size_t)row * hidden;

    float acc = 0.0f;
    for (unsigned int i = tid; i < hidden; i += blockDim.x) {
        acc += __bfloat162float(w[i]) * __bfloat162float(h[i]);
    }
    __shared__ float red[128];
    red[tid] = acc;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_out[row * rank + r] = red[0];
    }
}

// ── 3. Greedy chain walk (sequential over rows; one block) ─────────────
// block = 256 (>= rank). Thread layout for scoring: cand c = tid / 16,
// lane l = tid % 16, partial dot strided by 16 over rank.
extern "C" __global__ void dflash2_selector_chain(
    const unsigned int* __restrict__ cand_ids,  // [gamma, 16]
    const float* __restrict__ cand_vals,        // [gamma, 16]
    const float* __restrict__ g,                // [gamma, rank]
    const __nv_bfloat16* __restrict__ pred_cb,  // [vocab, rank]
    const __nv_bfloat16* __restrict__ succ_cb,  // [vocab, rank]
    unsigned int* __restrict__ drafts,          // [n*gamma] (rows 1.. rewritten)
    unsigned int gamma,
    unsigned int rank,
    unsigned int anchor,
    // BATCHED: grid.y = n sequences. Each block walks ITS OWN chain over
    // band b, seeded from anchors[b] (the per-sequence last_token). NULL
    // falls back to the scalar `anchor`, so the single-sequence launch is
    // unchanged. The walk stays sequential WITHIN a band — that is the
    // data dependence — but bands are independent, so n of them run
    // concurrently instead of n kernel launches.
    const unsigned int* __restrict__ anchors
) {
    const unsigned int b = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    cand_ids += (size_t)b * gamma * SEL_TOPK;
    cand_vals += (size_t)b * gamma * SEL_TOPK;
    g += (size_t)b * gamma * rank;
    drafts += (size_t)b * gamma;
    __shared__ float e[256];
    __shared__ float part[SEL_TOPK][SEL_TOPK];
    __shared__ float score[SEL_TOPK];
    __shared__ unsigned int prev;

    if (tid == 0) {
        prev = (anchors == nullptr) ? anchor : anchors[b];
    }
    __syncthreads();

    for (unsigned int t = 1; t < gamma; ++t) {
        if (tid < rank) {
            e[tid] = __bfloat162float(pred_cb[(size_t)prev * rank + tid]) * g[t * rank + tid];
        }
        __syncthreads();

        const unsigned int c = tid / SEL_TOPK;
        const unsigned int l = tid % SEL_TOPK;
        float p = 0.0f;
        const unsigned int cid = cand_ids[t * SEL_TOPK + c];
        for (unsigned int r = l; r < rank; r += SEL_TOPK) {
            p += __bfloat162float(succ_cb[(size_t)cid * rank + r]) * e[r];
        }
        part[c][l] = p;
        __syncthreads();
        if (tid < SEL_TOPK) {
            float s = cand_vals[t * SEL_TOPK + tid];
            for (unsigned int l2 = 0; l2 < SEL_TOPK; ++l2) {
                s += part[tid][l2];
            }
            score[tid] = s;
        }
        __syncthreads();
        if (tid == 0) {
            // Strict >, first (highest-unary) candidate wins ties — the
            // host walk's iteration order.
            float best = score[0];
            unsigned int bi = 0;
            for (unsigned int c2 = 1; c2 < SEL_TOPK; ++c2) {
                if (score[c2] > best) {
                    best = score[c2];
                    bi = c2;
                }
            }
            unsigned int pick = cand_ids[t * SEL_TOPK + bi];
            drafts[t] = pick;
            prev = pick;
        }
        __syncthreads();
    }
}
