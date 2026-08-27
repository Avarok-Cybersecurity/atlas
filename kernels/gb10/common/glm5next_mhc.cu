// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3-Flash mHC (Manifold-Constrained Hyper-Connections) — GLM-specific `hc_pre`.
//
// Atlas already has an mHC kernel: `deepseek-v4-flash/nvfp4/hyper_connection.cu`. Its
// `hc_pre` ends the Sinkhorn with an EXACT column projection (no eps) that DeepSeek-V4
// wants and GLM-5.3's reference does NOT have. HF's `Glm5NextTextHyperConnection` divides
// by `(colsum + hc_eps)` on every pass and stops there, so its columns settle at
// `1 - O(hc_eps)` rather than exactly 1.
//
// Slice 9's numeric oracle isolated that projection as the ENTIRE residual between Atlas
// and the reference: re-normalising the reference's own `comb` columns to exactly 1 drops
// the max abs difference 1.1325e-6 -> 5.9605e-8 (19x, onto f32 rounding). So this kernel is
// `hc_pre` with that one block removed, and nothing else.
//
// ── Why this is a COPY and not a shared header ──
// `hyper_connection.cu` is a frozen, proven DeepSeek-V4 path. The projection it carries was
// A/B-tested there (portv4b11) and REGRESSED coherence onset when removed, so it is load-
// bearing for V4 — and that A/B cannot be re-run from the GLM lane (no DS4F checkpoint here,
// lane closed). Refactoring its body into a shared `.cuh` would recompile V4's kernel from
// new source for a benefit measured in lines. The ~120 duplicated lines buy V4 being
// BYTE-IDENTICAL: `hyper_connection.cu` is not touched at all by this change.
//
// Placement: `common/`, not a GLM target dir — GLM-5.3 has no kernel target yet, and its
// other new kernels (`kda_layer_ops.cu`, `dsa_indexer.cu`) already live here. Unlisted `.cu`
// files take their file stem as the module name, so this is `glm5next_mhc::glm5next_hc_pre`.
//
// The launch signature is IDENTICAL to `hyper_connection::hc_pre`, deliberately: the GLM path
// reuses `ops::hc_pre` and passes a different KernelHandle. No new Rust surface, and the V4
// call sites (which resolve `"hyper_connection"`/`"hc_pre"` in `qwen3_attention/init.rs`)
// cannot reach this one.
//
// ⚠️ `hc_post` is NOT duplicated here: Slice 9 measured it as deviation-free, so GLM uses
// `hyper_connection::hc_post` verbatim. When GLM gets its own kernel target that module will
// not be present, and the pair will have to be co-located. Recorded, not pre-built.

#include <cuda_bf16.h>

#define GLM_HC_BLOCK 256
#define GLM_HC_MAX_MULT 4
#define GLM_HC_MAX_MIX 24 // (2 + GLM_HC_MAX_MULT) * GLM_HC_MAX_MULT

// Block-wide sum reduction over red[0..GLM_HC_BLOCK).
__device__ __forceinline__ float glm_hc_block_reduce(float* red, unsigned int tid) {
    for (unsigned int s = GLM_HC_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    return red[0];
}

// ── glm5next_hc_pre ──
// streams [T, hc, H] -> y_out [T, H] (collapsed), post_out [T, hc],
// comb_out [T, hc, hc].  Grid: (T,1,1)  Block: (256,1,1).
//
// Mirrors `Glm5NextTextHyperConnection.forward`:
//   flat  = unweighted_rms_norm(streams.flatten(2).float())
//   mixes = F.linear(flat, fn.float())                      -> [pre | post | comb]
//   pre   = sigmoid(pre*scale0 + base) + hc_eps
//   post  = 2 * sigmoid(post*scale1 + base)
//   comb  = softmax(comb*scale2 + base, dim=-1) + hc_eps
//           then col-norm, then (iters-1) x (row-norm, col-norm), ALL with +hc_eps
//   y     = sum_i pre[i] * streams[i]
extern "C" __global__ void glm5next_hc_pre(
    const float* __restrict__ streams,  // [T, hc, H] FP32 highway (mHC)
    const float* __restrict__ hc_fn,    // [mix_hc, hc*H]  (BF16 on disk; upcast by the loader)
    const float* __restrict__ hc_scale, // [3]
    const float* __restrict__ hc_base,  // [mix_hc]
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int sinkhorn_iters,
    const float norm_eps,
    const float hc_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;
    const unsigned int hc_dim = hc * H;
    const unsigned int mix_hc = (2 + hc) * hc;

    const float* x = streams + (size_t)t * hc_dim;

    __shared__ float red[GLM_HC_BLOCK];
    __shared__ float s_rsqrt;
    __shared__ float s_mix[GLM_HC_MAX_MIX];
    __shared__ float s_pre[GLM_HC_MAX_MULT];

    // Pass 1: RMS over the flattened hc*H vector.
    float ss = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += GLM_HC_BLOCK) {
        float v = (float)x[k];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    float ssum = glm_hc_block_reduce(red, tid);
    if (tid == 0) s_rsqrt = rsqrtf(ssum / (float)hc_dim + norm_eps);
    __syncthreads();
    const float rsqrt = s_rsqrt;

    // Pass 2: mixes[m] = (sum_k fn[m,k] * x[k]) * rsqrt
    // (`linear(x * rsqrt, fn)` == `rsqrt * linear(x, fn)`; rsqrt is a per-token scalar.)
    for (unsigned int m = 0; m < mix_hc; ++m) {
        const float* fn_row = hc_fn + (size_t)m * hc_dim;
        float acc = 0.f;
        for (unsigned int k = tid; k < hc_dim; k += GLM_HC_BLOCK) {
            acc += fn_row[k] * (float)x[k];
        }
        red[tid] = acc;
        __syncthreads();
        float r = glm_hc_block_reduce(red, tid);
        if (tid == 0) s_mix[m] = r * rsqrt;
        __syncthreads();
    }

    // Thread 0: split + Sinkhorn (tiny hc x hc problem).
    if (tid == 0) {
        float comb[GLM_HC_MAX_MULT * GLM_HC_MAX_MULT];
        for (unsigned int i = 0; i < hc; ++i) {
            float pr = s_mix[i] * hc_scale[0] + hc_base[i];
            s_pre[i] = 1.f / (1.f + expf(-pr)) + hc_eps;
            float po = s_mix[hc + i] * hc_scale[1] + hc_base[hc + i];
            post_out[(size_t)t * hc + i] = 2.f * (1.f / (1.f + expf(-po)));
        }
        for (unsigned int i = 0; i < hc; ++i)
            for (unsigned int j = 0; j < hc; ++j)
                comb[i * hc + j] =
                    s_mix[2 * hc + i * hc + j] * hc_scale[2] + hc_base[2 * hc + i * hc + j];
        // softmax over j (dim=-1) + eps
        for (unsigned int i = 0; i < hc; ++i) {
            float mx = -1e30f;
            for (unsigned int j = 0; j < hc; ++j) mx = fmaxf(mx, comb[i * hc + j]);
            float sum = 0.f;
            for (unsigned int j = 0; j < hc; ++j) {
                float e = expf(comb[i * hc + j] - mx);
                comb[i * hc + j] = e;
                sum += e;
            }
            for (unsigned int j = 0; j < hc; ++j) comb[i * hc + j] = comb[i * hc + j] / sum + hc_eps;
        }
        // col-norm first (dim=-2, over i)
        for (unsigned int j = 0; j < hc; ++j) {
            float c = hc_eps;
            for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
            for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] /= c;
        }
        // Sinkhorn: (iters - 1) alternating row/col passes
        for (unsigned int it = 0; it + 1 < sinkhorn_iters; ++it) {
            for (unsigned int i = 0; i < hc; ++i) {
                float r = hc_eps;
                for (unsigned int j = 0; j < hc; ++j) r += comb[i * hc + j];
                for (unsigned int j = 0; j < hc; ++j) comb[i * hc + j] /= r;
            }
            for (unsigned int j = 0; j < hc; ++j) {
                float c = hc_eps;
                for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
                for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] /= c;
            }
        }
        // 🔴 AND STOP. `hyper_connection.cu` adds one more EXACT column projection here.
        // GLM's reference does not, and Slice 9 measured that block as the whole difference.
        // Do not "restore the manifold constraint": the eps-ending Sinkhorn IS the reference's
        // semantics, and the columns it leaves (1 - O(hc_eps)) are non-expansive already.
        for (unsigned int i = 0; i < hc; ++i)
            for (unsigned int j = 0; j < hc; ++j)
                comb_out[(size_t)t * hc * hc + i * hc + j] = comb[i * hc + j];
    }
    __syncthreads();

    // Pass 3: collapse y[d] = sum_i pre[i] * x[i, d]
    for (unsigned int d = tid; d < H; d += GLM_HC_BLOCK) {
        float acc = 0.f;
        for (unsigned int i = 0; i < hc; ++i) acc += s_pre[i] * (float)x[i * H + d];
        y_out[(size_t)t * H + d] = __float2bfloat16(acc);
    }
}

// ── glm5next_hc_post ──
// out[t,j,d] = post[t,j]*block_out[t,d] + sum_i comb[t,i,j]*residual[t,i,d].
// `out` may alias `residual` (all hc residual values are read before write).
// Grid: (T,1,1)  Block: (256,1,1).
//
// Mirrors the decoder layer's own residual write:
//   post.unsqueeze(-1) * block_out.unsqueeze(-2) + matmul(comb.transpose(-1,-2), residual)
//
// ⚠️ This is byte-for-byte the same arithmetic as `hyper_connection::hc_post` — Slice 9 measured
// that half of mHC as deviation-free, so there is NO semantic duplication here, only a second
// entry point. It exists solely for TARGET INDEPENDENCE: `hyper_connection` lives in the
// deepseek-v4-flash target, and a GLM kernel target must not have to carry a DeepSeek module to
// resolve half of its own hyper-connection. `hc_pre` is where the two models genuinely differ
// (GLM omits the final exact column projection); this one differs in name only, and that is
// stated rather than hidden.
extern "C" __global__ void glm5next_hc_post(
    const __nv_bfloat16* __restrict__ block_out, // [T, H]
    const float* __restrict__ residual,          // [T, hc, H] FP32 highway (mHC)
    const float* __restrict__ post,              // [T, hc]
    const float* __restrict__ comb,              // [T, hc, hc]
    float* __restrict__ out,                     // [T, hc, H] FP32 highway (mHC)
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;

    const __nv_bfloat16* x = block_out + (size_t)t * H;
    const float* res = residual + (size_t)t * hc * H;
    const float* p = post + (size_t)t * hc;
    const float* c = comb + (size_t)t * hc * hc;
    float* o = out + (size_t)t * hc * H;

    for (unsigned int d = tid; d < H; d += GLM_HC_BLOCK) {
        float xd = (float)x[d];
        float rv[GLM_HC_MAX_MULT];
        for (unsigned int i = 0; i < hc; ++i) rv[i] = res[i * H + d];
        for (unsigned int j = 0; j < hc; ++j) {
            float acc = p[j] * xd;
            for (unsigned int i = 0; i < hc; ++i) acc += c[i * hc + j] * rv[i];
            o[j * H + d] = acc;
        }
    }
}

// ── glm5next_hc_head ──
// Final collapse before the LM head: streams [T, hc, H] -> y_out [T, H].
// Grid: (T,1,1)  Block: (256,1,1).
//
// 🔴 GLM's `Glm5NextTextHyperHead` is an UNWEIGHTED MEAN and has NO PARAMETERS:
//     return hidden_streams.mean(dim=2)
// HF's own comment: "Unlike DeepSeek-V4, this is an unweighted mean."
//
// `hyper_connection::hc_head` is DeepSeek-V4's LEARNED sigmoid-weighted sum and reads
// `hc_head.{fn,base,scale}`. This checkpoint contains **ZERO** `hc_head` tensors — reusing that
// kernel would read weights that do not exist. Hence a separate kernel with NO weight arguments
// at all: the absence of the pointers is the guard.
//
// Divides ONCE by hc after accumulating, matching `mean`, rather than pre-scaling each stream.
extern "C" __global__ void glm5next_hc_head(
    const float* __restrict__ streams, // [T, hc, H] FP32 highway (mHC)
    __nv_bfloat16* __restrict__ y_out, // [T, H]
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;

    const float* x = streams + (size_t)t * hc * H;
    const float inv = 1.0f / (float)hc;

    for (unsigned int d = tid; d < H; d += GLM_HC_BLOCK) {
        float acc = 0.f;
        for (unsigned int i = 0; i < hc; ++i) acc += x[i * H + d];
        y_out[(size_t)t * H + d] = __float2bfloat16(acc * inv);
    }
}
