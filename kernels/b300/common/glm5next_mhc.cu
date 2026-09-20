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

// ── glm5next_hc_mix + glm5next_hc_finish ──
// `glm5next_hc_pre` SPLIT IN TWO, and the split is the whole point.
//
// 🔴 The fused kernel runs on grid (T,1,1) — ONE BLOCK, one SM. Its pass 2 walks the
// [mix_hc, hc*H] mixing matrix ROW BY ROW, so a single block pulls mix_hc x hc*H floats of
// `hc_fn` plus mix_hc re-reads of the stream vector: at hc=4, H=5120 that is ~3.9 MB through
// one SM, per site, per layer. Measured 2026-08-28 at ~193 us/call averaged over the three
// mHC launches, 26.1 ms/token, 21 % of the decode step — the same low-parallelism defect as
// `glm5next_router_topk`, in a different kernel.
//
// `glm5next_hc_mix` gives every mixing row its OWN block: grid (T, mix_hc). The reduction is
// unchanged — same 256-wide block, same strided accumulation order, same tree reduce — so the
// mixes come out BIT-IDENTICAL, which `examples/glm5next_hc_split_gate.rs` asserts against the
// fused kernel byte for byte.
//
// 🪤 `glm5next_hc_pre` is KEPT and still built. It is the oracle the split is gated against,
// and `examples/mhc_microtest.rs` (the Slice-9 numeric gate vs HF, and the V4-divergence
// control) launches it through the shared `ops::hc_pre` signature. Deleting it would delete
// both gates.
// 🪤 Each `hc_mix` block recomputes the RMS over the same stream vector. That is hc*H floats
// (80 KB at GLM's shape) re-read per block, L2-resident after the first, and it is what keeps
// the kernel free of a cross-block dependency. Do not "optimise" it into a separate pass
// without re-running the bit-identity gate: the redundancy IS the bit-identity.

// streams [T, hc, H] -> mix_out [T, mix_hc].  Grid: (T, mix_hc, 1)  Block: (256,1,1).
extern "C" __global__ void glm5next_hc_mix(
    const float* __restrict__ streams,  // [T, hc, H] FP32 highway (mHC)
    const float* __restrict__ hc_fn,    // [mix_hc, hc*H]
    float* __restrict__ mix_out,        // [T, mix_hc]
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const float norm_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int m = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int hc_dim = hc_mult * hidden_size;
    const unsigned int mix_hc = (2 + hc_mult) * hc_mult;

    const float* x = streams + (size_t)t * hc_dim;
    __shared__ float red[GLM_HC_BLOCK];

    // Pass 1: RMS over the flattened hc*H vector — identical order to the fused kernel.
    float ss = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += GLM_HC_BLOCK) {
        float v = (float)x[k];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    const float ssum = glm_hc_block_reduce(red, tid);
    const float rsqrt = rsqrtf(ssum / (float)hc_dim + norm_eps);
    // Every thread read red[0] above; `red` is reused below, so nobody may write it yet.
    __syncthreads();

    // Pass 2: this block's ONE mixing row.
    const float* fn_row = hc_fn + (size_t)m * hc_dim;
    float acc = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += GLM_HC_BLOCK) {
        acc += fn_row[k] * (float)x[k];
    }
    red[tid] = acc;
    __syncthreads();
    const float r = glm_hc_block_reduce(red, tid);
    if (tid == 0) mix_out[(size_t)t * mix_hc + m] = r * rsqrt;
}

// mix [T, mix_hc] -> y_out [T, H], post_out [T, hc], comb_out [T, hc, hc].
// Grid: (T, NB, 1)  Block: (256,1,1). Body copied from `glm5next_hc_pre`'s passes 3+4; the
// changes are that `s_mix` is read from global instead of computed in shared, and that the
// final collapse is spread over the grid.
//
// 🔴 SECOND BLOCK AXIS — the Sinkhorn needs one block, the COLLAPSE does not.
// The Sinkhorn is an hc x hc problem with cross-lane dependencies, so it cannot leave a single
// block. The collapse that follows it is `y[d] = sum_i pre[i] * x[i,d]` over H = 4096 — hc*H
// floats in, H bf16 out, every `d` independent — and it was pinned to that same ONE block, i.e.
// 64 KB pulled through 1 of the GB10's 48 SMs, 90 times per token.
//
// So: grid (T, NB). `blockIdx.y == 0` runs the Sinkhorn and writes `post_out`/`comb_out`;
// blocks 1..NB-1 split the collapse between them, and block 0 sits the collapse out so the
// Sinkhorn is fully overlapped rather than serialised in front of block 0's share.
// (NB == 1 degenerates to the old single-block behaviour, which is what `mhc_microtest` and
// any direct caller still get.)
//
// 🪤 EVERY block recomputes `pre` — hc sigmoids off the same `mix` row. That is the same
// deliberate redundancy as `hc_mix`'s repeated RMS: it is four `expf`s, and it is what keeps
// the collapse blocks free of a cross-block dependency on block 0. It is bit-identical by
// construction (same expression, same inputs, no reduction), and `glm5next_hc_split_gate.rs`
// asserts that against the fused `glm5next_hc_pre` oracle byte for byte.
// ── glm5next_hc_mix_bf16 ──
// `glm5next_hc_mix` with `hc_fn` read at the width the CHECKPOINT stores it.
//
// 🔴 `hc_*_fn` is **BF16 on disk** (`[24, 16384]`, verified in the safetensors headers) and the
// loader was uploading it as F32. Nothing needed the extra width — every value is exactly a
// BF16 — but the decode path paid for it: 1.57 MB instead of 0.79 MB per site, 90 sites per
// token, **141 MB/token of which half was waste**. nsys 2026-08-28 put `hc_mix` at 1.18 ms of
// a 63.7 ms step, ~120 GB/s, and its unique traffic IS `hc_fn`.
//
// BIT-IDENTICAL, and the reason is exact: widening BF16 to F32 is lossless, so the F32 value
// this kernel used to read and `__bfloat162float` of the BF16 it reads now are the SAME float.
// Same multiply, same strided accumulation order, same tree reduce.
//
// 🪤 This is the MIRROR of the #341/#347 dtype class. There, a wide checkpoint tensor was read
// as narrow and the values were wrong. Here a narrow tensor was stored wide: the values stayed
// right, so nothing failed — it just doubled the bandwidth of the hottest small kernel in the
// model, silently, for the life of the port. Check the on-disk dtype of anything a decode
// kernel streams, in BOTH directions.
extern "C" __global__ void glm5next_hc_mix_bf16(
    const float* __restrict__ streams,  // [T, hc, H] FP32 highway (mHC)
    const __nv_bfloat16* __restrict__ hc_fn, // [mix_hc, hc*H] BF16 — as on disk
    float* __restrict__ mix_out,        // [T, mix_hc]
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const float norm_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int m = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int hc_dim = hc_mult * hidden_size;
    const unsigned int mix_hc = (2 + hc_mult) * hc_mult;

    const float* x = streams + (size_t)t * hc_dim;
    __shared__ float red[GLM_HC_BLOCK];

    // Pass 1: RMS over the flattened hc*H vector — identical order to the fused kernel.
    float ss = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += GLM_HC_BLOCK) {
        float v = (float)x[k];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    const float ssum = glm_hc_block_reduce(red, tid);
    const float rsqrt = rsqrtf(ssum / (float)hc_dim + norm_eps);
    // Every thread read red[0] above; `red` is reused below, so nobody may write it yet.
    __syncthreads();

    // Pass 2: this block's ONE mixing row.
    const __nv_bfloat16* fn_row = hc_fn + (size_t)m * hc_dim;
    float acc = 0.f;
    for (unsigned int k = tid; k < hc_dim; k += GLM_HC_BLOCK) {
        acc += __bfloat162float(fn_row[k]) * (float)x[k];
    }
    red[tid] = acc;
    __syncthreads();
    const float r = glm_hc_block_reduce(red, tid);
    if (tid == 0) mix_out[(size_t)t * mix_hc + m] = r * rsqrt;
}

// mix [T, mix_hc] -> y_out [T, H], post_out [T, hc], comb_out [T, hc, hc].
// Grid: (T, NB, 1)  Block: (256,1,1). Body copied from `glm5next_hc_pre`'s passes 3+4; the
// changes are that `s_mix` is read from global instead of computed in shared, and that the
// final collapse is spread over the grid.
//
// 🔴 SECOND BLOCK AXIS — the Sinkhorn needs one block, the COLLAPSE does not.
// The Sinkhorn is an hc x hc problem with cross-lane dependencies, so it cannot leave a single
// block. The collapse that follows it is `y[d] = sum_i pre[i] * x[i,d]` over H = 4096 — hc*H
// floats in, H bf16 out, every `d` independent — and it was pinned to that same ONE block, i.e.
// 64 KB pulled through 1 of the GB10's 48 SMs, 90 times per token.
//
// So: grid (T, NB). `blockIdx.y == 0` runs the Sinkhorn and writes `post_out`/`comb_out`;
// blocks 1..NB-1 split the collapse between them, and block 0 sits the collapse out so the
// Sinkhorn is fully overlapped rather than serialised in front of block 0's share.
// (NB == 1 degenerates to the old single-block behaviour, which is what `mhc_microtest` and
// any direct caller still get.)
//
// 🪤 EVERY block recomputes `pre` — hc sigmoids off the same `mix` row. That is the same
// deliberate redundancy as `hc_mix`'s repeated RMS: it is four `expf`s, and it is what keeps
// the collapse blocks free of a cross-block dependency on block 0. It is bit-identical by
// construction (same expression, same inputs, no reduction), and `glm5next_hc_split_gate.rs`
// asserts that against the fused `glm5next_hc_pre` oracle byte for byte.
extern "C" __global__ void glm5next_hc_finish(
    const float* __restrict__ streams,  // [T, hc, H] FP32 highway (mHC)
    const float* __restrict__ mix,      // [T, mix_hc]
    const float* __restrict__ hc_scale, // [3]
    const float* __restrict__ hc_base,  // [mix_hc]
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int sinkhorn_iters,
    const float hc_eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int by = blockIdx.y;
    const unsigned int nb = gridDim.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;
    const unsigned int mix_hc = (2 + hc) * hc;

    const float* x = streams + (size_t)t * hc * H;
    const float* s_mix = mix + (size_t)t * mix_hc;
    __shared__ float s_pre[GLM_HC_MAX_MULT];
    // 🔴 SHARED, not a thread-local array. `comb[i * hc + j]` indexes with a RUNTIME `hc`, so a
    // local `float comb[16]` cannot live in registers — nvcc puts it in local memory and every
    // one of the Sinkhorn's ~1,280 accesses becomes a memory op. Measured 2026-08-28 on GLM's
    // shape: hc_finish 68.1 us at sinkhorn_iters=20 vs 13.2 us at 1, i.e. 2.9 us per iteration
    // for 64 flops. Shared memory is the same arithmetic in the same order — bit-identical, and
    // `examples/glm5next_hc_split_gate.rs` asserts exactly that against the fused kernel.
    __shared__ float comb[GLM_HC_MAX_MULT * GLM_HC_MAX_MULT];

    // 🔴 ONE THREAD PER ROW / PER COLUMN, not one thread for the whole Sinkhorn.
    // Every accumulation a thread performs is still the reference's own sequential order — row
    // i sums j ascending, column j sums i ascending, both starting from `hc_eps` — so this is
    // BIT-IDENTICAL to the serial form and the split gate asserts it. What changes is that the
    // hc row (and then column) normalisations run concurrently instead of chained, which is
    // what the divides were waiting on: measured 2.9 us per Sinkhorn iteration serial, for 64
    // flops, because a single thread cannot overlap `div.rn.f32` latency with anything.
    // 🪤 Do NOT "simplify" a normalisation to `x * (1/r)`. That is not the same float, and it
    // is not what HF computes.
    const bool lane = tid < hc;

    // `pre` in EVERY block — see the header note. Four sigmoids, no reduction, no dependency.
    if (lane) {
        const unsigned int i = tid;
        float pr = s_mix[i] * hc_scale[0] + hc_base[i];
        s_pre[i] = 1.f / (1.f + expf(-pr)) + hc_eps;
    }

    // `post`, `comb` and the whole Sinkhorn: block 0 only.
    if (by == 0) {
        if (lane) {
            const unsigned int i = tid;
            float po = s_mix[hc + i] * hc_scale[1] + hc_base[hc + i];
            post_out[(size_t)t * hc + i] = 2.f * (1.f / (1.f + expf(-po)));
            for (unsigned int j = 0; j < hc; ++j)
                comb[i * hc + j] =
                    s_mix[2 * hc + i * hc + j] * hc_scale[2] + hc_base[2 * hc + i * hc + j];
        }
        __syncthreads();

        // softmax over j (dim=-1) + eps — row i, sequential in j.
        if (lane) {
            const unsigned int i = tid;
            float mx = -1e30f;
            for (unsigned int j = 0; j < hc; ++j) mx = fmaxf(mx, comb[i * hc + j]);
            float sum = 0.f;
            for (unsigned int j = 0; j < hc; ++j) {
                float e = expf(comb[i * hc + j] - mx);
                comb[i * hc + j] = e;
                sum += e;
            }
            for (unsigned int j = 0; j < hc; ++j)
                comb[i * hc + j] = comb[i * hc + j] / sum + hc_eps;
        }
        __syncthreads();

        // col-norm first (dim=-2, over i) — column j, sequential in i.
        if (lane) {
            const unsigned int j = tid;
            float c = hc_eps;
            for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
            for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] /= c;
        }
        __syncthreads();

        // Sinkhorn: (iters - 1) alternating row/col passes.
        for (unsigned int it = 0; it + 1 < sinkhorn_iters; ++it) {
            if (lane) {
                const unsigned int i = tid;
                float r = hc_eps;
                for (unsigned int j = 0; j < hc; ++j) r += comb[i * hc + j];
                for (unsigned int j = 0; j < hc; ++j) comb[i * hc + j] /= r;
            }
            __syncthreads();
            if (lane) {
                const unsigned int j = tid;
                float c = hc_eps;
                for (unsigned int i = 0; i < hc; ++i) c += comb[i * hc + j];
                for (unsigned int i = 0; i < hc; ++i) comb[i * hc + j] /= c;
            }
            __syncthreads();
        }
        // 🔴 AND STOP — see `glm5next_hc_pre`. No final exact column projection.
        for (unsigned int k = tid; k < hc * hc; k += GLM_HC_BLOCK)
            comb_out[(size_t)t * hc * hc + k] = comb[k];
    }
    __syncthreads();

    // Collapse y[d] = sum_i pre[i] * x[i, d], split over blocks 1..NB-1 (all of block 0 when
    // NB == 1). Every `d` is an independent output element, so this is the same arithmetic in
    // the same order on a different thread — bit-identical, whatever NB is.
    if (nb == 1 || by > 0) {
        const unsigned int slot = (nb == 1) ? 0 : by - 1;
        const unsigned int nslot = (nb == 1) ? 1 : nb - 1;
        for (unsigned int d = slot * GLM_HC_BLOCK + tid; d < H; d += nslot * GLM_HC_BLOCK) {
            float acc = 0.f;
            for (unsigned int i = 0; i < hc; ++i) acc += s_pre[i] * (float)x[i * H + d];
            y_out[(size_t)t * H + d] = __float2bfloat16(acc);
        }
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
// 🪤 KEPT AS THE ORACLE, not dead code. `glm5next_hc_post` below is this body with two changes
// — compile-time trip counts and a second grid axis — and `examples/glm5next_hc_post_gate.rs`
// asserts the two agree BYTE FOR BYTE. Same role `glm5next_hc_pre` plays for the mix/finish
// split: nothing here is an approximation, so the gate is identity, not tolerance.
// Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void glm5next_hc_post_ref(
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

// The serve-path `hc_post`. Two changes against `glm5next_hc_post_ref`, both bit-identical:
//
// 🔴 1. `rv` LIVES IN REGISTERS NOW. `float rv[GLM_HC_MAX_MULT]` indexed by a loop whose bound
//    is the RUNTIME `hc` cannot be register-allocated — nvcc puts it in LOCAL MEMORY, so all
//    2 * hc accesses per `d` become memory ops, on top of the loads they were caching. This is
//    the same defect that cost `glm5next_hc_finish` 2.9 us per Sinkhorn iteration (fixed there
//    by moving `comb` to shared) and the same one the routed-expert and router-top-k passes hit
//    earlier in this port. The fix is a COMPILE-TIME trip count with a predicate:
//    `for (i = 0; i < GLM_HC_MAX_MULT; ++i) if (i < hc)`. The `i < hc` guard means the executed
//    arithmetic — and its order, i ascending — is exactly the reference's.
//
// 🔴 2. GRID (T, NB, 1). This kernel is `hc_mult * H` floats in and out per token with EVERY
//    `d` independent, and it ran on ONE block: 128 KB through 1 of the GB10's 48 SMs, twice per
//    layer, 90 times per token. Splitting `d` across blocks is safe even though `out` may alias
//    `residual`: a block touching column `d` reads only `res[i*H+d]` and writes only
//    `o[j*H+d]`, and `j*H+d == i*H+d'` requires `d == d'`. Columns never cross blocks.
//    NB == 1 reproduces the old launch exactly.
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

    // `post` and `comb` are hc + hc*hc floats re-read for every `d`. Hoist once per block:
    // same values, so the arithmetic below is untouched.
    __shared__ float s_p[GLM_HC_MAX_MULT];
    __shared__ float s_c[GLM_HC_MAX_MULT * GLM_HC_MAX_MULT];
    if (tid < hc) s_p[tid] = p[tid];
    if (tid < hc * hc) s_c[tid] = c[tid];
    __syncthreads();

    const unsigned int stride = gridDim.y * GLM_HC_BLOCK;
    for (unsigned int d = blockIdx.y * GLM_HC_BLOCK + tid; d < H; d += stride) {
        float xd = (float)x[d];
        float rv[GLM_HC_MAX_MULT];
#pragma unroll
        for (unsigned int i = 0; i < GLM_HC_MAX_MULT; ++i)
            if (i < hc) rv[i] = res[i * H + d];
#pragma unroll
        for (unsigned int j = 0; j < GLM_HC_MAX_MULT; ++j) {
            if (j < hc) {
                float acc = s_p[j] * xd;
#pragma unroll
                for (unsigned int i = 0; i < GLM_HC_MAX_MULT; ++i)
                    if (i < hc) acc += s_c[i * hc + j] * rv[i];
                o[j * H + d] = acc;
            }
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

// ── glm5next_hc_expand ──
// Broadcast the single embedding stream into the `hc_mult` highway streams at the FIRST
// layer: streams[t, i, d] = hidden[t, d].
//
// 🪤 This is byte-for-byte the same broadcast as `hyper_connection::hc_expand`, and it is
// duplicated here ANYWAY because that file lives in `kernels/gb10/deepseek-v4-flash/nvfp4/`.
// A kernel target merges `common/` plus its OWN model dir and cannot reach into another
// target's, so the GLM target could never resolve it — the module simply would not exist.
// Copying ~10 lines of a parameterless broadcast is the whole cost of that isolation.
//
// The highway is FP32 (see `buffers/sizes.rs`: mHC mixing is norm-preserving, so BF16
// storage swamps the per-layer signal at depth). The input embedding is BF16.
// Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void glm5next_hc_expand(
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
    for (unsigned int d = tid; d < H; d += GLM_HC_BLOCK) {
        float v = (float)x[d];
        for (unsigned int i = 0; i < hc_mult; ++i) s[i * H + d] = v;
    }
}
