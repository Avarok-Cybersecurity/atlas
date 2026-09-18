// SPDX-License-Identifier: AGPL-3.0-only
//
// Control vectors (activation steering) on the mHC highway.
// Qwen3.8-Flash-Next. See docs/design/qwen4exp-control-vectors.md.
//
// A control vector is a per-layer direction `v` in the residual stream. Two
// interventions, applied at the END of a layer, after that layer's `hc_post`:
//
//   project:  h <- h - s * (h . v) * v     (v unit norm; s = 1 fully ablates)
//   add:      h <- h + s * v
//
// NO WEIGHT IS READ OR WRITTEN. The highway is FP32 regardless of how the
// model's weights are quantized, so NVFP4 never enters this arithmetic.
//
// ── The highway is [T, hc, H], and all hc streams take the hook ───────────
//
// There is no per-layer `[H]` residual vector in this architecture: the
// residual stream IS the hc-row highway (`hidden` inside a layer is scratch).
// So a block handles one (token, stream) pair and the grid is (T, hc).
//
// llama.cpp's reference hook does the same thing — `build_cvec(res_hc, il)`
// over `[n_embd, hc, n_tokens]`, broadcast over the stream axis — even though
// the direction is DERIVED from the stream mean. That is self-consistent only
// because projection is LINEAR: mean(project(h_r)) == project(mean(h_r)).
// Do not "optimize" this into a single projection of the mean; the streams do
// not stay equal to their mean once they diverge, and the mean is not what the
// next layer reads.
//
// ── Determinism ──────────────────────────────────────────────────────────
//
// The dot product is a block reduction, so its ROUNDING is a function of
// `blockDim.x` — the same property that pins `hc_pre_stage_bf16` to block
// 1024. `CVEC_BLOCK` below is that pin for these kernels: changing it changes
// the answer in the last bits, which on a speculative path means a different
// accept/reject and therefore different text. Change it only with a re-gate.
//
// Every thread reduces the warp partials in the SAME order, so all threads
// leave the reduction holding bit-identical values and no broadcast barrier is
// needed. Under `--fmad=false` (this shadow's KERNEL.toml) the multiply and
// add do not contract, which keeps the result stable across nvcc versions.

#include <cuda_runtime.h>

// Pinned launch width — see "Determinism" above. 8 warps.
#define CVEC_BLOCK 256u

// Sum `x` across the block. `warp_partials` needs `blockDim.x / 32` floats.
// Contains one `__syncthreads()`, which also separates the caller's reads of
// the highway (before) from its writes (after).
__device__ __forceinline__ float cvec_block_reduce_sum(
    float x,
    float* __restrict__ warp_partials
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        x += __shfl_down_sync(0xFFFFFFFFu, x, off);
    }
    if (lane == 0) {
        warp_partials[warp] = x;
    }
    __syncthreads();
    const unsigned int warps = blockDim.x >> 5;
    float total = 0.0f;
    for (unsigned int w = 0; w < warps; ++w) {
        total += warp_partials[w];
    }
    return total;
}

// ── cvec_project_highway ──
// `h -= scale * (h . v) * v` for every (token, stream).
//
// grid (T, hc), block CVEC_BLOCK, dynamic shared `(hidden_size + 32) * 4`.
// `v` is staged in shared memory because both passes read all of it.
extern "C" __global__ void cvec_project_highway(
    float* __restrict__ highway,        // [T, hc*H] FP32, in/out
    const float* __restrict__ v,        // [H] FP32, unit norm
    const float scale,
    const unsigned int hidden_size,
    const unsigned int hc
) {
    extern __shared__ float smem[];
    float* __restrict__ sv = smem;                    // [hidden_size]
    float* __restrict__ swarp = smem + hidden_size;   // [blockDim.x / 32]

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sv[i] = v[i];
    }
    __syncthreads();

    float* __restrict__ h =
        highway + ((size_t)blockIdx.x * hc + blockIdx.y) * hidden_size;

    float acc = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        acc += h[i] * sv[i];
    }
    // The barrier inside this call is what makes the write loop below safe:
    // every thread's reads of `h` have retired by the time it returns.
    const float dot = cvec_block_reduce_sum(acc, swarp);

    const float k = scale * dot;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        h[i] -= k * sv[i];
    }
}

// ── cvec_add_highway ──
// `h += scale * v` for every (token, stream). The additive arm; a different
// vector and a much smaller scale than projection (~0.1 vs 1.0). Elementwise,
// so no reduction and no shared memory.
//
// grid `div_ceil(n, CVEC_BLOCK)`, block CVEC_BLOCK, where `n = T * hc * H`.
extern "C" __global__ void cvec_add_highway(
    float* __restrict__ highway,        // [T, hc*H] FP32, in/out
    const float* __restrict__ v,        // [H] FP32
    const float scale,
    const unsigned int hidden_size,
    const unsigned int n                // T * hc * hidden_size
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        highway[i] += scale * v[i % hidden_size];
    }
}

// ── cvec_cos_highway ──
// The validation probe: `|cos(h, v)|` per (token, stream), written to
// `out[t * hc + r]`. `v` is unit norm, so the denominator is just `|h|`.
//
// This is the gate that proves the hook actually landed on a forward path —
// run it before and after the projection, and `post` must be ~0 while `pre`
// is clearly non-zero. Far more sensitive than any behavioural eval, and the
// only thing that can prove no call site was missed.
//
// grid (T, hc), block CVEC_BLOCK, dynamic shared `(hidden_size + 64) * 4`.
extern "C" __global__ void cvec_cos_highway(
    const float* __restrict__ highway,  // [T, hc*H] FP32
    const float* __restrict__ v,        // [H] FP32, unit norm
    float* __restrict__ out,            // [T * hc] FP32
    const unsigned int hidden_size,
    const unsigned int hc
) {
    extern __shared__ float smem[];
    float* __restrict__ sv = smem;                    // [hidden_size]
    float* __restrict__ swarp = smem + hidden_size;   // [2 * blockDim.x / 32]

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sv[i] = v[i];
    }
    __syncthreads();

    const float* __restrict__ h =
        highway + ((size_t)blockIdx.x * hc + blockIdx.y) * hidden_size;

    float adot = 0.0f;
    float asq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float x = h[i];
        adot += x * sv[i];
        asq += x * x;
    }

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warps = blockDim.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        adot += __shfl_down_sync(0xFFFFFFFFu, adot, off);
        asq += __shfl_down_sync(0xFFFFFFFFu, asq, off);
    }
    // Two disjoint partial arrays, so one barrier serves both reductions.
    if (lane == 0) {
        swarp[warp] = adot;
        swarp[warps + warp] = asq;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float d = 0.0f;
        float s = 0.0f;
        for (unsigned int w = 0; w < warps; ++w) {
            d += swarp[w];
            s += swarp[warps + w];
        }
        // A zero-norm stream has no defined angle; report 0 rather than NaN so
        // an aggregate over padded rows stays readable.
        out[(size_t)blockIdx.x * hc + blockIdx.y] =
            s > 0.0f ? fabsf(d) * rsqrtf(s) : 0.0f;
    }
}
