// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

// DFlash2 drafter kernels: grouped dynamic two-tap conv, per-row top-16,
// and the candidate-selector chain walk.
//
// Reference semantics: z-lab dflash/model.py (GroupedDynamicCausalConv,
// CandidateSelector) + inco.ai/blog/dflash2 formulas:
//   Conv_k(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t-1}
//   S_t(a,b)    = U_t(b) + ⟨A(a) ⊙ H(h_t), B(b)⟩
// All math in f32, storage BF16 (matching the drafter's training dtype).

#include <cuda_bf16.h>

// ─────────────────────────────────────────────────────────────────────────
// Two-tap grouped dynamic causal convolution — ONE application (prepare OR
// finish). Per row t, channel c, with group g = c / group_size:
//   coef_k(t,c) = base[k*h + c] + dyn[t*dyn_stride + app_off + k*groups + g]
//   out[t][c]   = coef_0 * x[t][c] + (t > 0 ? coef_1 * x[t-1][c] : 0)
// Row 0's second tap reads nothing (zero pad) — in DFlash blocks row 0 IS
// the anchor (last verified token's embedding), so "the first draft
// position reads the last verified token" holds by construction.
//
// dyn layout per row (from kernel_projection GEMM, row-major view
// [2 applications, kernel_size, groups]): app_off selects the application
// (0 = prepare, kernel_size*groups = finish).
//
// NOT in-place safe: out must not alias x (row t reads x[t-1] after row
// t-1 was consumed elsewhere only if distinct buffers). Grid: (rows,1,1),
// Block: (256,1,1).
extern "C" __global__ void dflash2_conv2(
    const __nv_bfloat16* __restrict__ x,     // [rows, h]
    const __nv_bfloat16* __restrict__ dyn,   // [rows, dyn_stride]
    const __nv_bfloat16* __restrict__ base,  // [kernel_size, h] (app slice)
    __nv_bfloat16* __restrict__ out,         // [rows, h]
    unsigned int h,
    unsigned int group_size,
    unsigned int dyn_stride,   // elements per dyn row (both applications)
    unsigned int app_off       // 0 (prepare) or kernel_size*groups (finish)
) {
    const unsigned int t = blockIdx.x;
    const unsigned int groups = h / group_size;
    const __nv_bfloat16* xr = x + (size_t)t * h;
    const __nv_bfloat16* xp = (t > 0) ? x + (size_t)(t - 1) * h : nullptr;
    const __nv_bfloat16* dr = dyn + (size_t)t * dyn_stride + app_off;
    __nv_bfloat16* orow = out + (size_t)t * h;

    for (unsigned int c = threadIdx.x; c < h; c += blockDim.x) {
        const unsigned int g = c / group_size;
        const float c0 = __bfloat162float(base[c]) + __bfloat162float(dr[g]);
        float acc = c0 * __bfloat162float(xr[c]);
        if (xp != nullptr) {
            const float c1 =
                __bfloat162float(base[h + c]) + __bfloat162float(dr[groups + g]);
            acc += c1 * __bfloat162float(xp[c]);
        }
        orow[c] = __float2bfloat16(acc);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Per-row top-16 over BF16 logits. DESTRUCTIVE: masks selected entries in
// `logits` to -inf (drafter logits are not consumed after selection).
// 16 sequential tree-argmax passes per row — simple, proven reduction
// pattern, one launch for all rows. Grid: (rows,1,1), Block: (1024,1,1).
extern "C" __global__ void dflash2_topk16(
    __nv_bfloat16* __restrict__ logits,  // [rows, n] — masked in place
    float* __restrict__ top_vals,        // [rows, 16]
    unsigned int* __restrict__ top_idx,  // [rows, 16]
    unsigned int n
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    __nv_bfloat16* lrow = logits + (size_t)row * n;

    for (unsigned int k = 0; k < 16; ++k) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < n; i += stride) {
            const float v = __bfloat162float(lrow[i]);
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }
        s_val[tid] = local_max;
        s_idx[tid] = local_idx;
        __syncthreads();
        for (unsigned int off = stride >> 1; off > 0; off >>= 1) {
            if (tid < off && s_val[tid + off] > s_val[tid]) {
                s_val[tid] = s_val[tid + off];
                s_idx[tid] = s_idx[tid + off];
            }
            __syncthreads();
        }
        if (tid == 0) {
            top_vals[(size_t)row * 16 + k] = s_val[0];
            top_idx[(size_t)row * 16 + k] = s_idx[0];
            lrow[s_idx[0]] = __float2bfloat16(-1e30f);
        }
        __syncthreads();
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Candidate-selector chain walk — the ONLY sequential piece, fused into a
// single launch for the whole block (vs one launch per position).
//
//   S_t(a,b) = U_t(b) + Σ_r pred[a][r] * hproj[t][r] * succ[b][r]
//
// Walks rows first_row..rows-1. prev for the first walked row is read from
// `tokens[0]`, which at tail entry still holds last_token (H2D'd fresh
// every propose BEFORE graph capture — same capture-safe device-side chain
// as the DSpark Markov path). Each chosen token is written to `tokens[t]`
// and becomes the next row's predecessor.
//
// Grid: (1,1,1), Block: (512,1,1) = 16 warps; warp k owns candidate k's
// rank-256 dot (8 lanes-elements each, shuffle-reduced).
extern "C" __global__ void dflash2_selector_walk(
    const float* __restrict__ top_vals,        // [rows, 16] unary logits
    const unsigned int* __restrict__ top_idx,  // [rows, 16] candidate ids
    const __nv_bfloat16* __restrict__ hproj,   // [rows, rank] context gate H(h_t)
    const __nv_bfloat16* __restrict__ pred,    // [vocab, rank] A codebook
    const __nv_bfloat16* __restrict__ succ,    // [vocab, rank] B codebook
    unsigned int* __restrict__ tokens,         // [rows] draft token slots
    unsigned int rows,
    unsigned int rank,       // 256
    unsigned int first_row   // 1 (row 0 = anchor, not selected)
) {
    __shared__ float s_gate[256];
    __shared__ float s_score[16];
    __shared__ unsigned int s_prev;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;

    if (tid == 0) {
        s_prev = tokens[0];  // last_token — slot 0 pre-argmax state
    }
    __syncthreads();

    for (unsigned int t = first_row; t < rows; ++t) {
        // gate[r] = pred[prev][r] * hproj[t][r]
        const __nv_bfloat16* pr = pred + (size_t)s_prev * rank;
        const __nv_bfloat16* hr = hproj + (size_t)t * rank;
        for (unsigned int r = tid; r < rank; r += blockDim.x) {
            s_gate[r] = __bfloat162float(pr[r]) * __bfloat162float(hr[r]);
        }
        __syncthreads();

        // warp k: score_k = unary_k + Σ_r gate[r] * succ[cand_k][r]
        if (warp < 16) {
            const unsigned int cand = top_idx[(size_t)t * 16 + warp];
            const __nv_bfloat16* sr = succ + (size_t)cand * rank;
            float acc = 0.0f;
            for (unsigned int r = lane; r < rank; r += 32) {
                acc += s_gate[r] * __bfloat162float(sr[r]);
            }
            #pragma unroll
            for (unsigned int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffff, acc, off);
            }
            if (lane == 0) {
                s_score[warp] = top_vals[(size_t)t * 16 + warp] + acc;
            }
        }
        __syncthreads();

        if (tid == 0) {
            float best = s_score[0];
            unsigned int best_k = 0;
            #pragma unroll
            for (unsigned int k = 1; k < 16; ++k) {
                if (s_score[k] > best) {
                    best = s_score[k];
                    best_k = k;
                }
            }
            const unsigned int chosen = top_idx[(size_t)t * 16 + best_k];
            tokens[t] = chosen;
            s_prev = chosen;
        }
        __syncthreads();
    }
}
