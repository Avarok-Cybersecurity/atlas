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

    for (unsigned int d = tid; d < H; d += QHC_BLOCK) {
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
