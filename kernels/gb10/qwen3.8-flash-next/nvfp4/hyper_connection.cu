// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next multi-hyperconnection (mHC) — the LOW-RANK mixer.
//
// Same four entry points and the same `[T, hc, H]` FP32 highway as
// DeepSeek-V4's `hyper_connection.cu`, and a DIFFERENT mixer. DeepSeek mixes
// with a Sinkhorn-normalized matrix over `hc_fn` / `hc_scale` / `hc_base`;
// Qwen mixes through a low-rank pair of rank `hc_lowrank` (320). The layouts
// coincide, the math does not — running DeepSeek's kernel against these
// weights produces fluent, confident, wrong output, which is why this file
// exists rather than a symlink.
//
// Transcribed from `Qwen4ExpTextGatedResidual.forward` (see
// `bench/qwen4_exp/ARCHITECTURE.md` §1):
//
//     normed = hc_norm(hyper_input)              # GROUPED RMSNorm, group=H
//     w = silu(down(normed) / hc)                # [hc*H] -> [R]
//     w = sigmoid(up(w))                         # [R] -> [hc*H]
//     mixed = (w.unflatten * normed.unflatten).mean(dim=-2)     # -> [H]
//     inj   = 2 * sigmoid(block_inject(normed) / hc)            # -> [hc]
//
// and the block output is injected back by `hc_post`:
//
//     residual[t, s*H + d] = hyper_input[t, s*H + d] + hidden[t, d] * inj[t, s]
//
// TWO THINGS THAT DO NOT FAIL LOUDLY IF GOT WRONG, both load-bearing:
//
//   1. `hc_norm` is GROUPED with `group_size = hidden_size`: the `hc` streams
//      normalize INDEPENDENTLY inside the `hc*H` vector. One RMS across all
//      `hc*H` is a different function that still produces plausible numbers.
//   2. The reduction over streams is a MEAN, not a sum. With hc = 4 a sum is
//      4x the intended magnitude — survivable-looking, and wrong.
//
// `normed` is recomputed on the fly from the per-stream RMS rather than
// staged: at hc*H = 10240 floats per token it would be 40 KB of shared (over
// budget) or ~84 MB of global traffic at T=2048. Only the `hc` reciprocals
// and the rank-R vector are kept resident.
//
// Grid: (T,1,1)   Block: (256,1,1)

#include <cuda_bf16.h>

#define QHC_BLOCK 256
#define QHC_MAX_MULT 8
#define QHC_MAX_RANK 512

__device__ __forceinline__ float qhc_silu(float v) {
    return v / (1.0f + __expf(-v));
}

__device__ __forceinline__ float qhc_sigmoid(float v) {
    return 1.0f / (1.0f + __expf(-v));
}

// Per-stream RMS reciprocals for one token: rms_inv[s] over x[s*H .. s*H+H).
// Leaves the result in `smem_rms`, block-wide visible after __syncthreads().
__device__ __forceinline__ void qhc_stream_rms(
    const float* __restrict__ x,
    unsigned int H,
    unsigned int hc,
    float eps,
    float* __restrict__ smem_rms,   // [hc]
    float* __restrict__ smem_red    // [QHC_BLOCK / 32]
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warps = QHC_BLOCK / 32;

    for (unsigned int s = 0; s < hc; ++s) {
        const float* xs = x + (size_t)s * H;
        float acc = 0.0f;
        for (unsigned int d = tid; d < H; d += QHC_BLOCK) {
            float v = xs[d];
            acc += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) smem_red[warp] = acc;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (unsigned int w = 0; w < warps; ++w) tot += smem_red[w];
            smem_rms[s] = rsqrtf(tot / (float)H + eps);
        }
        __syncthreads();
    }
}

// ── hc_expand ──
// Broadcast a single hidden state into `hc` identical streams. Identical in
// behaviour to the DeepSeek twin; duplicated because a model shadow overrides
// a whole FILE, not individual entry points.
extern "C" __global__ void hc_expand(
    const __nv_bfloat16* __restrict__ hidden, // [T, H]
    float* __restrict__ streams,              // [T, hc, H] FP32 highway
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const __nv_bfloat16* x = hidden + (size_t)t * H;
    float* s = streams + (size_t)t * hc_mult * H;
    for (unsigned int d = tid; d < H; d += QHC_BLOCK) {
        float v = (float)x[d];
        for (unsigned int i = 0; i < hc_mult; ++i) s[i * H + d] = v;
    }
}

// Shared core for `hc_pre` and `hc_head`: both run the identical low-rank
// collapse; `hc_head` is the model-level mixer built with `use_combine=False`,
// so it simply has no `block_inject_weight` and emits no injection vector.
// Passing `inject_w == nullptr` selects that form.
//
// PERFORMANCE SHAPE (this core was the entire decode budget — 4.5 ms per
// call, x96 calls/token ~= 435 ms of a 455 ms token). Three rules:
//
//  1. The normed vector is staged ONCE in shared memory (hc*H floats = 40 KB
//     at 4x2560). The first cut recomputed `x * rms * (1 + w)` — three loads
//     and two multiplies — at every one of its ~6.6M uses.
//  2. The down projection runs one WARP per rank row: lanes stride the
//     10240-wide row (coalesced), then warp-reduce. The first cut gave each
//     THREAD a serial row: uncoalesced and 32x less parallel.
//  3. The up projection runs one warp per 32 output elements, each lane
//     owning one element's rank-320 loop per stream; `up_w` rows for
//     adjacent outputs are adjacent, so the lane-parallel reads stay warm in
//     L2.
//
// The launcher passes block=1024 (32 warps). Grid stays [num_tokens]: at
// prefill that is thousands of independent blocks; at decode it is one block,
// which rule 2 finally keeps busy.
//
// The `1.0f +` in the norm is NOT optional — see the offset-from-1 note in
// the header. The parity probe (`hyper_connection_lowrank_tests.rs`) holds
// this core to the reference at every entry point.
#define QHC_WBLOCK 1024
#define QHC_SMEM_NORMED (QHC_MAX_MULT * 2560)

__device__ __forceinline__ void qhc_collapse(
    const float* __restrict__ streams,
    const __nv_bfloat16* __restrict__ hc_norm_w,
    const __nv_bfloat16* __restrict__ down_w,
    const __nv_bfloat16* __restrict__ up_w,
    const __nv_bfloat16* __restrict__ inject_w,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ inj_out,
    unsigned int H,
    unsigned int hc,
    unsigned int rank,
    float eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warps = blockDim.x >> 5;
    const unsigned int hc_dim = hc * H;
    const float* x = streams + (size_t)t * hc_dim;

    extern __shared__ float smem[];
    float* smem_normed = smem;                 // [hc*H]
    float* smem_low = smem + hc_dim;           // [rank]
    __shared__ float smem_rms[QHC_MAX_MULT];
    __shared__ float smem_red[QHC_WBLOCK / 32];

    // ── per-stream RMS ──
    for (unsigned int s2 = 0; s2 < hc; ++s2) {
        const float* xs = x + (size_t)s2 * H;
        float acc = 0.0f;
        for (unsigned int d = tid; d < H; d += blockDim.x) {
            float v = xs[d];
            acc += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) smem_red[warp] = acc;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (unsigned int w2 = 0; w2 < warps; ++w2) tot += smem_red[w2];
            smem_rms[s2] = rsqrtf(tot / (float)H + eps);
        }
        __syncthreads();
    }

    // ── stage normed = x * rms * (1 + w) once ──
    for (unsigned int i = tid; i < hc_dim; i += blockDim.x) {
        smem_normed[i] = x[i] * smem_rms[i / H] * (1.0f + (float)hc_norm_w[i]);
    }
    __syncthreads();

    // ── down: warp per rank row, lanes stride the row ──
    const float inv_hc = 1.0f / (float)hc;
    for (unsigned int r = warp; r < rank; r += warps) {
        const __nv_bfloat16* row = down_w + (size_t)r * hc_dim;
        float acc = 0.0f;
        for (unsigned int i = lane; i < hc_dim; i += 32) {
            acc += (float)row[i] * smem_normed[i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) smem_low[r] = qhc_silu(acc * inv_hc);
    }
    __syncthreads();

    // ── up + gate + mean over streams: lane owns one output element ──
    __nv_bfloat16* y = y_out + (size_t)t * H;
    for (unsigned int d = tid; d < H; d += blockDim.x) {
        float mixed = 0.0f;
        for (unsigned int s2 = 0; s2 < hc; ++s2) {
            const unsigned int i = s2 * H + d;
            const __nv_bfloat16* urow = up_w + (size_t)i * rank;
            float acc = 0.0f;
            for (unsigned int r = 0; r < rank; ++r) {
                acc += (float)urow[r] * smem_low[r];
            }
            mixed += qhc_sigmoid(acc) * smem_normed[i];
        }
        y[d] = __float2bfloat16(mixed * inv_hc);
    }

    // ── injection weights: warp per stream ──
    if (inject_w != nullptr) {
        __syncthreads();
        for (unsigned int s2 = warp; s2 < hc; s2 += warps) {
            const __nv_bfloat16* row = inject_w + (size_t)s2 * hc_dim;
            float acc = 0.0f;
            for (unsigned int i = lane; i < hc_dim; i += 32) {
                acc += (float)row[i] * smem_normed[i];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
            }
            if (lane == 0) {
                inj_out[(size_t)t * hc + s2] = 2.0f * qhc_sigmoid(acc * inv_hc);
            }
        }
    }
}

// ── hc_pre ──
// streams [T, hc, H] -> y_out [T, H] collapsed, inj_out [T, hc].
extern "C" __global__ void hc_pre(
    const float* __restrict__ streams,
    const __nv_bfloat16* __restrict__ hc_norm_w,  // [hc*H]
    const __nv_bfloat16* __restrict__ down_w,     // [rank, hc*H]
    const __nv_bfloat16* __restrict__ up_w,       // [hc*H, rank]
    const __nv_bfloat16* __restrict__ inject_w,   // [hc, hc*H]
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ inj_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int rank,
    const float norm_eps
) {
    qhc_collapse(streams, hc_norm_w, down_w, up_w, inject_w, y_out, inj_out,
                 hidden_size, hc_mult, rank, norm_eps);
}

// ── hc_head ──
// The model-level `hyper_connection_mixer` (`use_combine=False`): the same
// collapse with no injection. This IS the model's final normalization — the
// checkpoint ships no `model.norm.weight` because `hc_norm` here plays that
// role.
extern "C" __global__ void hc_head(
    const float* __restrict__ streams,
    const __nv_bfloat16* __restrict__ hc_norm_w,
    const __nv_bfloat16* __restrict__ down_w,
    const __nv_bfloat16* __restrict__ up_w,
    __nv_bfloat16* __restrict__ y_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int rank,
    const float norm_eps
) {
    qhc_collapse(streams, hc_norm_w, down_w, up_w, nullptr, y_out, nullptr,
                 hidden_size, hc_mult, rank, norm_eps);
}

// ── hc_post ──
// residual[t, s*H + d] = hyper_input[t, s*H + d] + block_out[t, d] * inj[t, s]
//
// `hyper_input` is the PRE-NORM highway, not the normalized one — the
// reference keeps the raw residual and adds to it.
extern "C" __global__ void hc_post(
    const __nv_bfloat16* __restrict__ block_out, // [T, H]
    const float* __restrict__ residual,          // [T, hc, H]
    const float* __restrict__ inj,               // [T, hc]
    float* __restrict__ out,                     // [T, hc, H]
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;

    const __nv_bfloat16* x = block_out + (size_t)t * H;
    const float* res = residual + (size_t)t * hc * H;
    const float* w = inj + (size_t)t * hc;
    float* o = out + (size_t)t * hc * H;

    float wv[QHC_MAX_MULT];
    for (unsigned int s = 0; s < hc; ++s) wv[s] = w[s];

    // 2026-09-09: gridDim.y > 1 slices the hidden dimension across blocks
    // (grid [T] was T blocks on T SMs for T*hc*H*8 bytes of traffic, 8.3 us at
    // T=3); each element's arithmetic is unchanged, so the output is the same
    // bytes at any gridDim.y.
    const unsigned int chunk = (H + gridDim.y - 1u) / gridDim.y;
    const unsigned int d0 = blockIdx.y * chunk;
    const unsigned int d1 = (d0 + chunk < H) ? (d0 + chunk) : H;
    for (unsigned int d = d0 + tid; d < d1; d += QHC_BLOCK) {
        float xd = (float)x[d];
        for (unsigned int s = 0; s < hc; ++s) {
            o[s * H + d] = res[s * H + d] + xd * wv[s];
        }
    }
}

// ── Split collapse, for SMALL T (decode) ─────────────────────────────────
// grid=[1] starves the fused kernel at decode: one block, one SM, ~13 MB of
// weights per call (measured 2.0 ms). These three launches spread the same
// math across the whole GPU; the Rust dispatcher picks them when
// `num_tokens` is small and keeps the fused kernel for prefill.

// Stage 1: normed = x * rms * (1 + w) -> global scratch [T, hc*H].
extern "C" __global__ void hc_pre_stage(
    const float* __restrict__ streams,
    const __nv_bfloat16* __restrict__ hc_norm_w,
    float* __restrict__ normed_out,            // [T, hc*H]
    const unsigned int hidden_size,
    const unsigned int hc,
    const float eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc_dim = hc * H;
    const float* x = streams + (size_t)t * hc_dim;
    float* out = normed_out + (size_t)t * hc_dim;

    __shared__ float smem_rms[QHC_MAX_MULT];
    __shared__ float smem_red[QHC_WBLOCK / 32];
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warps = blockDim.x >> 5;

    // 2026-09-09: gridDim.y == hc puts one STREAM per block (grid [T, hc]:
    // 12 blocks at T=3 instead of 3); gridDim.y == 1 keeps the whole-token
    // block. The per-thread striding, the shuffle tree and the warp-partial
    // order are the same either way, so the bytes are identical.
    const unsigned int s_lo = (gridDim.y > 1u) ? blockIdx.y : 0u;
    const unsigned int s_hi = (gridDim.y > 1u) ? blockIdx.y + 1u : hc;
    for (unsigned int s2 = s_lo; s2 < s_hi; ++s2) {
        const float* xs = x + (size_t)s2 * H;
        float acc = 0.0f;
        for (unsigned int d = tid; d < H; d += blockDim.x) {
            float v = xs[d];
            acc += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) smem_red[warp] = acc;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (unsigned int w2 = 0; w2 < warps; ++w2) tot += smem_red[w2];
            smem_rms[s2] = rsqrtf(tot / (float)H + eps);
        }
        __syncthreads();
    }
    for (unsigned int i = s_lo * H + tid; i < s_hi * H; i += blockDim.x) {
        out[i] = x[i] * smem_rms[i / H] * (1.0f + (float)hc_norm_w[i]);
    }
}

// Stage 2: low[r] = silu(down[r] . normed / hc), rank rows split over
// blockIdx.y. Warp per row, coalesced lane strides.
extern "C" __global__ void hc_pre_down(
    const float* __restrict__ normed,          // [T, hc*H]
    const __nv_bfloat16* __restrict__ down_w,  // [rank, hc*H]
    float* __restrict__ low_out,               // [T, rank]
    const unsigned int hidden_size,
    const unsigned int hc,
    const unsigned int rank
) {
    const unsigned int t = blockIdx.x;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warps = blockDim.x >> 5;
    const unsigned int hc_dim = hc * hidden_size;
    const float* nx = normed + (size_t)t * hc_dim;
    const float inv_hc = 1.0f / (float)hc;

    // Rows split first across grid.y, then across warps in the block.
    const unsigned int rows_per_split = (rank + gridDim.y - 1) / gridDim.y;
    const unsigned int r0 = blockIdx.y * rows_per_split;
    const unsigned int r1 = min(r0 + rows_per_split, rank);
    for (unsigned int r = r0 + warp; r < r1; r += warps) {
        const __nv_bfloat16* row = down_w + (size_t)r * hc_dim;
        float acc = 0.0f;
        for (unsigned int i = lane; i < hc_dim; i += 32) {
            acc += (float)row[i] * nx[i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) low_out[(size_t)t * rank + r] = qhc_silu(acc * inv_hc);
    }
}

// Stage 3: y[d] = mean_s sigmoid(up[s*H+d] . low) * normed[s*H+d], the
// d-range split over blockIdx.y; block y==0 also emits the injection vector.
extern "C" __global__ void hc_pre_finish(
    const float* __restrict__ normed,          // [T, hc*H]
    const float* __restrict__ low,             // [T, rank]
    const __nv_bfloat16* __restrict__ up_w,    // [hc*H, rank]
    const __nv_bfloat16* __restrict__ inject_w,// [hc, hc*H] or null
    __nv_bfloat16* __restrict__ y_out,         // [T, H]
    float* __restrict__ inj_out,               // [T, hc] (unused if null inject)
    const unsigned int hidden_size,
    const unsigned int hc,
    const unsigned int rank
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc_dim = hc * H;
    const float* nx = normed + (size_t)t * hc_dim;
    const float inv_hc = 1.0f / (float)hc;

    extern __shared__ float smem_lo[];         // [rank]
    for (unsigned int r = tid; r < rank; r += blockDim.x) {
        smem_lo[r] = low[(size_t)t * rank + r];
    }
    __syncthreads();

    const unsigned int d_per_split = (H + gridDim.y - 1) / gridDim.y;
    const unsigned int d0 = blockIdx.y * d_per_split;
    const unsigned int d1 = min(d0 + d_per_split, H);
    __nv_bfloat16* y = y_out + (size_t)t * H;
    for (unsigned int d = d0 + tid; d < d1; d += blockDim.x) {
        float mixed = 0.0f;
        for (unsigned int s2 = 0; s2 < hc; ++s2) {
            const unsigned int i = s2 * H + d;
            const __nv_bfloat16* urow = up_w + (size_t)i * rank;
            float acc = 0.0f;
            for (unsigned int r = 0; r < rank; ++r) {
                acc += (float)urow[r] * smem_lo[r];
            }
            mixed += qhc_sigmoid(acc) * nx[i];
        }
        y[d] = __float2bfloat16(mixed * inv_hc);
    }

    if (inject_w != nullptr && blockIdx.y == 0) {
        const unsigned int lane = tid & 31u;
        const unsigned int warp = tid >> 5;
        const unsigned int warps = blockDim.x >> 5;
        for (unsigned int s2 = warp; s2 < hc; s2 += warps) {
            const __nv_bfloat16* row = inject_w + (size_t)s2 * hc_dim;
            float acc = 0.0f;
            for (unsigned int i = lane; i < hc_dim; i += 32) {
                acc += (float)row[i] * nx[i];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
            }
            if (lane == 0) {
                inj_out[(size_t)t * hc + s2] = 2.0f * qhc_sigmoid(acc * inv_hc);
            }
        }
    }
}

// ───────────────────────── GEMM-path collapse (large T) ─────────────────────
//
// PERFORMANCE SHAPE: at prefill the fused kernel measured ~45 ms per call —
// 47% of the whole prefill (two calls per layer x 48 layers). Its down/up
// projections are GEMM-shaped ([T,hc*H]x[hc*H,rank] and back), but ran as
// hand-rolled FP32 warp loops at ~4% of the machine. For T > 64 the collapse
// instead stages `normed` in BF16 and hands both projections to
// `dense_gemm_bf16_pipelined` (tensor cores), keeping only the cheap
// elementwise seams as custom kernels:
//
//   hc_pre_stage_bf16   grid=[T]    rms + (1+w) scale -> normed  [T, hc*H] BF16
//   dense_gemm          low_pre  = normed x down_w^T             [T, rank]
//   hc_silu_scale       low      = silu(low_pre / hc)            in place
//   dense_gemm          up_pre   = low x up_w^T                  [T, hc*H]
//   dense_gemm          inj_pre  = normed x inject_w^T           [T, hc]
//   hc_pre_mix          grid=[T]    y = mean_s sigmoid(up_pre)*normed;
//                                   inj = 2*sigmoid(inj_pre / hc)
//
// Numerics: normed is rounded to BF16 before the GEMMs (the fused kernel kept
// it FP32 in smem). The checkpoint's hyper-connection weights are BF16 and the
// reference module computes in BF16, so this is parity-gated the same way as
// every other collapse variant (probe cosine vs the FP32 fused path).

extern "C" __global__ void hc_pre_stage_bf16(
    const float* __restrict__ streams,
    const __nv_bfloat16* __restrict__ hc_norm_w,
    __nv_bfloat16* __restrict__ normed_out,    // [T, hc*H] BF16
    const unsigned int hidden_size,
    const unsigned int hc,
    const float eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc_dim = hc * H;
    const float* x = streams + (size_t)t * hc_dim;
    __nv_bfloat16* out = normed_out + (size_t)t * hc_dim;

    __shared__ float smem_rms[QHC_MAX_MULT];
    __shared__ float smem_red[QHC_WBLOCK / 32];
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warps = blockDim.x >> 5;

    for (unsigned int s2 = 0; s2 < hc; ++s2) {
        const float* xs = x + (size_t)s2 * H;
        float acc = 0.0f;
        for (unsigned int d = tid; d < H; d += blockDim.x) {
            float v = xs[d];
            acc += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) smem_red[warp] = acc;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (unsigned int w2 = 0; w2 < warps; ++w2) tot += smem_red[w2];
            smem_rms[s2] = rsqrtf(tot / (float)H + eps);
        }
        __syncthreads();
    }
    for (unsigned int i = tid; i < hc_dim; i += blockDim.x) {
        out[i] = __float2bfloat16(
            x[i] * smem_rms[i / H] * (1.0f + (float)hc_norm_w[i]));
    }
}

// low = silu(low_pre * inv_hc), elementwise in place over n = T*rank.
extern "C" __global__ void hc_silu_scale(
    __nv_bfloat16* __restrict__ low,
    const unsigned int n,
    const float inv_hc
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = (float)low[i] * inv_hc;
        low[i] = __float2bfloat16(qhc_silu(v));
    }
}

// y[d] = mean_s sigmoid(up_pre[s*H+d]) * normed[s*H+d];
// inj[s] = 2*sigmoid(inj_pre[s] * inv_hc) (skipped when inj_pre is null).
extern "C" __global__ void hc_pre_mix(
    const __nv_bfloat16* __restrict__ normed,  // [T, hc*H]
    const __nv_bfloat16* __restrict__ up_pre,  // [T, hc*H]
    const __nv_bfloat16* __restrict__ inj_pre, // [T, hc] or null
    __nv_bfloat16* __restrict__ y_out,         // [T, H]
    float* __restrict__ inj_out,               // [T, hc]
    const unsigned int hidden_size,
    const unsigned int hc,
    const float inv_hc
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc_dim = hc * H;
    const __nv_bfloat16* nx = normed + (size_t)t * hc_dim;
    const __nv_bfloat16* ux = up_pre + (size_t)t * hc_dim;
    __nv_bfloat16* y = y_out + (size_t)t * H;

    for (unsigned int d = tid; d < H; d += blockDim.x) {
        float mixed = 0.0f;
        for (unsigned int s2 = 0; s2 < hc; ++s2) {
            const unsigned int i = s2 * H + d;
            mixed += qhc_sigmoid((float)ux[i]) * (float)nx[i];
        }
        y[d] = __float2bfloat16(mixed * inv_hc);
    }
    if (inj_pre != nullptr && tid < hc) {
        inj_out[(size_t)t * hc + tid] =
            2.0f * qhc_sigmoid((float)inj_pre[(size_t)t * hc + tid] * inv_hc);
    }
}

// ─── MTP combiner tail ───────────────────────────────────────────────────────
// Write the MTP draft's FP32 highway from a PER-STREAM BF16 projection plus one
// BROADCAST BF16 row:
//
//   streams[i*H + d] = (float)per_stream[i*H + d] + (float)bcast[d]
//
// This exists because the highway is FP32 while the projections that build it
// (`fc_hidden` per stream, `fc_embedding` once) are BF16 GEMVs. Without it the
// combiner would write BF16 into an FP32 buffer — which is silent garbage, not
// an error, and reads as "MTP just doesn't predict well on this model".
//
// `hc_expand` cannot serve: it BROADCASTS a single row to every stream, and the
// MTP combiner's streams differ per stream. Single block, T=1 (a draft step is
// one token); `hc` is bounded by QHC_MAX_MULT.
extern "C" __global__ void qhc_mtp_combine_streams(
    const __nv_bfloat16* __restrict__ per_stream, // [hc, H] BF16
    const __nv_bfloat16* __restrict__ bcast,      // [H]     BF16
    float* __restrict__ streams,                  // [hc, H] FP32 (out)
    const unsigned int hidden_size,
    const unsigned int hc
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    for (unsigned int i = 0; i < hc; ++i) {
        const __nv_bfloat16* ps = per_stream + (size_t)i * H;
        float* s = streams + (size_t)i * H;
        for (unsigned int d = tid; d < H; d += blockDim.x) {
            s[d] = (float)ps[d] + (float)bcast[d];
        }
    }
}


// ───────────────────── Decode-shaped collapse, T <= QHC_DEC_MAX_T ────────────
// provenance-id: 526f6e616c6420522e205374657369616b
//
// PERFORMANCE SHAPE. At decode (T = 1..8 rows) the split arm reads the 13 MB
// of low-rank weights once PER TOKEN (grid.x = T) with one 2-byte load per
// lane, and the cuBLASLt arm pays six library launches per site at M <= 8.
// Both sit at ~100-150 us per site against a ~56 us bandwidth floor
// (2026-09-07 trace: ~10 ms of a 59 ms MTP step across 96 sites). These two
// kernels read every weight row EXACTLY ONCE per site with 16-byte lane loads
// and apply it to all T tokens at the same time, keeping `normed` in FP32 (the
// split arm's numerics, not the cuBLASLt arm's BF16-rounded ones):
//
//   hc_pre_stage   grid=[T]                    normed = x * rms * (1 + w)  [T, hc*H] F32 (existing)
//   hc_dec_down    grid=[ceil(rows/8)] x 256   warp per weight row (rank rows + inject rows):
//                                              low[t, r]  = silu(down[r] . normed[t] / hc)
//                                              inj[t, s]  = 2 sigmoid(inject[s] . normed[t] / hc)
//   hc_dec_up      grid=[H/16] x (hc*64)       4 threads per (stream, d) row (quarter rows, xor-shuffle reduce):
//                                              sigmoid(up[s*H+d] . low[t]) * normed[t, s*H+d], mean over streams
//                                              in shared memory -> y[t, d] BF16
//
// Shape contract (checked by the Rust dispatcher, which falls back to the
// existing arms otherwise): hc*H % 256 == 0, H % 64 == 0, rank % 8 == 0,
// hc <= QHC_MAX_MULT, T <= QHC_DEC_MAX_T, 16-byte aligned rows (rank*2 % 16 == 0).

#define QHC_DEC_MAX_T 8

__device__ __forceinline__ void qhc_unpack8(const uint4 raw, float* w) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float2 f = __bfloat1622float2(p[k]);
        w[2 * k] = f.x;
        w[2 * k + 1] = f.y;
    }
}

// hc_dec_down (2026-09-09 rework): QHC_DOWN_SPLIT warps per weight row.
// The single-warp-per-row form issued 32 dependent 16-byte loads per lane
// (hc_dim = 8192) with no unrolling and put only rows/8 blocks on the machine
// (41 blocks for rank 320 + 4 inject rows on 48 SMs): latency-bound at
// 34.8 us against a ~19 us floor for the 5.3 MB of weights (09-08 trace).
// Now each row is a contiguous slice per warp (hc_dim / QHC_DOWN_SPLIT), the
// slice's loads are issued QHC_DOWN_UNROLL at a time, and the split partials
// are summed in a fixed order through shared memory, so every token sees the
// same reduction order (the T=3-vs-T=1 byte test) regardless of T.
//
// Grid: ceil(rows / QHC_DOWN_ROWS_PER_BLOCK)   Block: 256 (8 warps)
// Contract: hc_dim % (QHC_DOWN_SPLIT * 256) == 0.
#define QHC_DOWN_SPLIT 4
#define QHC_DOWN_ROWS_PER_BLOCK (8 / QHC_DOWN_SPLIT)
#define QHC_DOWN_UNROLL 8

extern "C" __global__ void hc_dec_down(
    const float* __restrict__ normed,          // [T, hc*H] F32
    const __nv_bfloat16* __restrict__ down_w,  // [rank, hc*H]
    const __nv_bfloat16* __restrict__ inject_w,// [hc, hc*H] or null
    float* __restrict__ low_out,               // [T, rank]
    float* __restrict__ inj_out,               // [T, hc] (unused if inject_w null)
    const unsigned int num_tokens,
    const unsigned int hc_dim,
    const unsigned int hc,
    const unsigned int rank
) {
    __shared__ float smem_part[QHC_DOWN_ROWS_PER_BLOCK][QHC_DOWN_SPLIT][QHC_DEC_MAX_T];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int row_in_block = warp / QHC_DOWN_SPLIT;
    const unsigned int part = warp % QHC_DOWN_SPLIT;
    const unsigned int row = blockIdx.x * QHC_DOWN_ROWS_PER_BLOCK + row_in_block;
    const unsigned int total = rank + (inject_w != nullptr ? hc : 0u);
    const bool live = row < total;

    float acc[QHC_DEC_MAX_T];
    #pragma unroll
    for (int t = 0; t < QHC_DEC_MAX_T; ++t) acc[t] = 0.0f;

    if (live) {
        const bool is_inj = row >= rank;
        const __nv_bfloat16* wrow = is_inj
            ? inject_w + (size_t)(row - rank) * hc_dim
            : down_w + (size_t)row * hc_dim;
        const unsigned int seg = hc_dim / QHC_DOWN_SPLIT;
        const unsigned int base = part * seg;
        const unsigned int end = base + seg;
        // Each lane owns 8 consecutive elements per iteration (one 16-byte
        // weight load, two float4 loads per token); the warp covers 256
        // elements per iteration and QHC_DOWN_UNROLL iterations are in flight.
        for (unsigned int i0 = base + lane * 8u; i0 < end; i0 += 256u * QHC_DOWN_UNROLL) {
            uint4 raw[QHC_DOWN_UNROLL];
            #pragma unroll
            for (int u = 0; u < QHC_DOWN_UNROLL; ++u) {
                const unsigned int i = i0 + 256u * u;
                raw[u] = (i < end) ? *reinterpret_cast<const uint4*>(wrow + i)
                                   : make_uint4(0u, 0u, 0u, 0u);
            }
            #pragma unroll
            for (int u = 0; u < QHC_DOWN_UNROLL; ++u) {
                const unsigned int i = i0 + 256u * u;
                if (i < end) {
                    float w[8];
                    qhc_unpack8(raw[u], w);
                    #pragma unroll
                    for (int t = 0; t < QHC_DEC_MAX_T; ++t) {
                        if (t < (int)num_tokens) {
                            const float* nx = normed + (size_t)t * hc_dim + i;
                            const float4 a = *reinterpret_cast<const float4*>(nx);
                            const float4 b = *reinterpret_cast<const float4*>(nx + 4);
                            acc[t] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w
                                    + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
                        }
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < QHC_DEC_MAX_T; ++t) {
        float v = acc[t];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xFFFFFFFFu, v, off);
        }
        if (lane == 0) smem_part[row_in_block][part][t] = v;
    }
    __syncthreads();
    if (live && part == 0 && lane == 0) {
        const bool is_inj = row >= rank;
        const float inv_hc = 1.0f / (float)hc;
        #pragma unroll
        for (int t = 0; t < QHC_DEC_MAX_T; ++t) {
            if (t < (int)num_tokens) {
                float v = 0.0f;
                #pragma unroll
                for (int q = 0; q < QHC_DOWN_SPLIT; ++q) v += smem_part[row_in_block][q][t];
                if (is_inj) {
                    inj_out[(size_t)t * hc + (row - rank)] = 2.0f * qhc_sigmoid(v * inv_hc);
                } else {
                    low_out[(size_t)t * rank + row] = qhc_silu(v * inv_hc);
                }
            }
        }
    }
}

// hc_dec_up (2026-09-09 rework): EIGHT lanes per up_w row, four rows per
// warp, QHC_UP_D_PER_BLOCK hidden columns per block.
// The quarter-row form (four lanes per row, 16 columns per block, grid H/16
// = 128 blocks) read 5.2 MB of weights at 47.9 us (09-08 trace) against a
// ~19 us floor: each load instruction touched four 160-byte-strided 16-byte
// chunks per row and the grid was under three blocks per SM. Now the eight
// lanes of a row read one contiguous 128-byte segment per instruction (rank
// 320 = 5 segments per row), all of a lane's loads are issued before the
// FMAs, and the grid is H/8 = 256 blocks of hc*64 threads.
//
// Block: hc * 64 threads = 2*hc warps; warp w: stream s = w / 2, columns
// d = blockIdx.x*8 + (w % 2)*4 + lane/8; lane % 8 = the 16-byte chunk lane.
// Grid: H / 8.  Contract: H % 8 == 0, rank % 64 == 0 (each lane's rank/8
// elements are 16-byte aligned), hc <= 8 (block <= 512).
#define QHC_UP_D_PER_BLOCK 8
#define QHC_UP_LANES_PER_ROW 8
#define QHC_UP_MAX_CHUNKS 8   // rank <= 512 = 8 lanes x 8 chunks x 8 elements

extern "C" __global__ void hc_dec_up(
    const float* __restrict__ normed,          // [T, hc*H] F32
    const float* __restrict__ low,             // [T, rank] F32
    const __nv_bfloat16* __restrict__ up_w,    // [hc*H, rank]
    __nv_bfloat16* __restrict__ y_out,         // [T, H]
    const unsigned int num_tokens,
    const unsigned int hidden_size,
    const unsigned int hc,
    const unsigned int rank
) {
    extern __shared__ float qhc_dec_smem[];
    float* smem_low = qhc_dec_smem;                                  // [T, rank]
    float* smem_part = qhc_dec_smem + (size_t)QHC_DEC_MAX_T * rank;  // [hc*8, QHC_DEC_MAX_T]
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int s = warp >> 1;
    const unsigned int dl = (warp & 1u) * 4u + (lane >> 3);   // 0..7
    const unsigned int c = lane & 7u;                          // chunk lane
    const unsigned int H = hidden_size;
    const unsigned int hc_dim = hc * H;
    const unsigned int d = blockIdx.x * QHC_UP_D_PER_BLOCK + dl;
    const unsigned int i = s * H + d;
    const unsigned int slice = rank / QHC_UP_LANES_PER_ROW;    // rank % 64 == 0 => slice % 8 == 0
    const unsigned int chunks = slice >> 3;

    for (unsigned int k = tid; k < num_tokens * rank; k += blockDim.x) {
        smem_low[k] = low[k];
    }
    __syncthreads();

    const __nv_bfloat16* urow = up_w + (size_t)i * rank + (size_t)c * slice;
    uint4 raw[QHC_UP_MAX_CHUNKS];
    #pragma unroll
    for (int u = 0; u < QHC_UP_MAX_CHUNKS; ++u) {
        raw[u] = (u < (int)chunks) ? *reinterpret_cast<const uint4*>(urow + 8u * u)
                                   : make_uint4(0u, 0u, 0u, 0u);
    }
    float acc[QHC_DEC_MAX_T];
    #pragma unroll
    for (int t = 0; t < QHC_DEC_MAX_T; ++t) acc[t] = 0.0f;
    const unsigned int r0 = c * slice;
    #pragma unroll
    for (int u = 0; u < QHC_UP_MAX_CHUNKS; ++u) {
        if (u < (int)chunks) {
            float w[8];
            qhc_unpack8(raw[u], w);
            #pragma unroll
            for (int t = 0; t < QHC_DEC_MAX_T; ++t) {
                if (t < (int)num_tokens) {
                    const float* lo = smem_low + (size_t)t * rank + r0 + 8u * u;
                    acc[t] += w[0] * lo[0] + w[1] * lo[1] + w[2] * lo[2] + w[3] * lo[3]
                            + w[4] * lo[4] + w[5] * lo[5] + w[6] * lo[6] + w[7] * lo[7];
                }
            }
        }
    }
    // Reduce the eight chunk partials (adjacent lanes c = 0..7).
    #pragma unroll
    for (int t = 0; t < QHC_DEC_MAX_T; ++t) {
        if (t < (int)num_tokens) {
            float v = acc[t];
            v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
            v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
            v += __shfl_xor_sync(0xFFFFFFFFu, v, 4);
            acc[t] = v;
        }
    }
    if (c == 0) {
        #pragma unroll
        for (int t = 0; t < QHC_DEC_MAX_T; ++t) {
            if (t < (int)num_tokens) {
                smem_part[((size_t)s * QHC_UP_D_PER_BLOCK + dl) * QHC_DEC_MAX_T + t] =
                    qhc_sigmoid(acc[t]) * normed[(size_t)t * hc_dim + i];
            }
        }
    }
    __syncthreads();
    if (tid < QHC_UP_D_PER_BLOCK) {
        const unsigned int dd = blockIdx.x * QHC_UP_D_PER_BLOCK + tid;
        const float inv_hc = 1.0f / (float)hc;
        #pragma unroll
        for (int t = 0; t < QHC_DEC_MAX_T; ++t) {
            if (t < (int)num_tokens) {
                float mixed = 0.0f;
                for (unsigned int s2 = 0; s2 < hc; ++s2) {
                    mixed += smem_part[((size_t)s2 * QHC_UP_D_PER_BLOCK + tid) * QHC_DEC_MAX_T + t];
                }
                y_out[(size_t)t * H + dd] = __float2bfloat16(mixed * inv_hc);
            }
        }
    }
}

// ───────────────────── hc_dec_down_v5: the shipped down kernel (2026-09-09) ──
// provenance-id: 526f6e616c6420522e205374657369616b
// Measured against hc_dec_down on the DRAM-true microbench (hc_rows_microbench,
// 32 weight copies cycled): 77-78 us vs 83 us per site at T=3, and it is
// BYTE-IDENTICAL to hc_dec_down (same per-lane slice order, same shuffle tree,
// same four-part sum), so the switch moved no stream. The dispatcher runs it
// by default; ATLAS_HC_DOWN_KERNEL=hc_dec_down restores the one-row form.
// (hc_dec_up_v3 and hc_dec_down_v4 were measured within noise and removed.)

// hc_dec_down_v5: register-blocked, TWO weight rows per warp (each `normed`
// read serves two rows, halving the L2 traffic per weight byte), four warps
// per row pair, two pairs per block. Grid: ceil(rows / 4).
#define QHC_DOWN5_SPLIT 4
#define QHC_DOWN5_UNROLL 4
extern "C" __global__ void hc_dec_down_v5(
    const float* __restrict__ normed,
    const __nv_bfloat16* __restrict__ down_w,
    const __nv_bfloat16* __restrict__ inject_w,
    float* __restrict__ low_out,
    float* __restrict__ inj_out,
    const unsigned int num_tokens,
    const unsigned int hc_dim,
    const unsigned int hc,
    const unsigned int rank
) {
    __shared__ float smem_part[2][2][QHC_DOWN5_SPLIT][QHC_DEC_MAX_T];   // [pair][row][part][t]
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int pair = warp / QHC_DOWN5_SPLIT;      // 0..1
    const unsigned int part = warp % QHC_DOWN5_SPLIT;
    const unsigned int row0 = (blockIdx.x * 2u + pair) * 2u;
    const unsigned int total = rank + (inject_w != nullptr ? hc : 0u);
    const bool live0 = row0 < total;
    const bool live1 = row0 + 1u < total;
    const __nv_bfloat16* wr0 = nullptr;
    const __nv_bfloat16* wr1 = nullptr;
    if (live0) wr0 = (row0 >= rank) ? inject_w + (size_t)(row0 - rank) * hc_dim : down_w + (size_t)row0 * hc_dim;
    if (live1) wr1 = (row0 + 1u >= rank) ? inject_w + (size_t)(row0 + 1u - rank) * hc_dim : down_w + (size_t)(row0 + 1u) * hc_dim;
    const unsigned int seg = hc_dim / QHC_DOWN5_SPLIT;
    const unsigned int base = part * seg;
    const unsigned int end = base + seg;
    float acc0[QHC_DEC_MAX_T], acc1[QHC_DEC_MAX_T];
    #pragma unroll
    for (int t = 0; t < QHC_DEC_MAX_T; ++t) { acc0[t] = 0.0f; acc1[t] = 0.0f; }
    if (live0) {
        for (unsigned int i0 = base + lane * 8u; i0 < end; i0 += 256u * QHC_DOWN5_UNROLL) {
            uint4 r0[QHC_DOWN5_UNROLL], r1[QHC_DOWN5_UNROLL];
            #pragma unroll
            for (int u = 0; u < QHC_DOWN5_UNROLL; ++u) {
                const unsigned int i = i0 + 256u * u;
                r0[u] = (i < end) ? *reinterpret_cast<const uint4*>(wr0 + i) : make_uint4(0u, 0u, 0u, 0u);
                r1[u] = (i < end && live1) ? *reinterpret_cast<const uint4*>(wr1 + i) : make_uint4(0u, 0u, 0u, 0u);
            }
            #pragma unroll
            for (int u = 0; u < QHC_DOWN5_UNROLL; ++u) {
                const unsigned int i = i0 + 256u * u;
                if (i < end) {
                    float w0[8], w1[8];
                    qhc_unpack8(r0[u], w0);
                    qhc_unpack8(r1[u], w1);
                    #pragma unroll
                    for (int t = 0; t < QHC_DEC_MAX_T; ++t) {
                        if (t < (int)num_tokens) {
                            const float* nx = normed + (size_t)t * hc_dim + i;
                            const float4 a = *reinterpret_cast<const float4*>(nx);
                            const float4 b = *reinterpret_cast<const float4*>(nx + 4);
                            acc0[t] += w0[0] * a.x + w0[1] * a.y + w0[2] * a.z + w0[3] * a.w
                                     + w0[4] * b.x + w0[5] * b.y + w0[6] * b.z + w0[7] * b.w;
                            acc1[t] += w1[0] * a.x + w1[1] * a.y + w1[2] * a.z + w1[3] * a.w
                                     + w1[4] * b.x + w1[5] * b.y + w1[6] * b.z + w1[7] * b.w;
                        }
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < QHC_DEC_MAX_T; ++t) {
        float v0 = acc0[t], v1 = acc1[t];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v0 += __shfl_down_sync(0xFFFFFFFFu, v0, off);
            v1 += __shfl_down_sync(0xFFFFFFFFu, v1, off);
        }
        if (lane == 0) { smem_part[pair][0][part][t] = v0; smem_part[pair][1][part][t] = v1; }
    }
    __syncthreads();
    if (part == 0 && lane < 2u) {
        const unsigned int row = row0 + lane;
        if (row < total) {
            const bool is_inj = row >= rank;
            const float inv_hc = 1.0f / (float)hc;
            #pragma unroll
            for (int t = 0; t < QHC_DEC_MAX_T; ++t) {
                if (t < (int)num_tokens) {
                    float v = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < QHC_DOWN5_SPLIT; ++q) v += smem_part[pair][lane][q][t];
                    if (is_inj) inj_out[(size_t)t * hc + (row - rank)] = 2.0f * qhc_sigmoid(v * inv_hc);
                    else        low_out[(size_t)t * rank + row] = qhc_silu(v * inv_hc);
                }
            }
        }
    }
}
