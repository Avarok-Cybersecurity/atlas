// SPDX-License-Identifier: AGPL-3.0-only
//
// GLM-5.3-Flash KDA (Kimi Delta Attention) recurrent DECODE step — one token.
//
//   S[h,k,v] <- S[h,k,v] * exp(gate[h,k])          per-channel decay on the KEY axis
//   delta[h,v] = (v[h,v] - SUM_k S[h,k,v]*k[h,k]) * beta[h]
//   S[h,k,v] <- S[h,k,v] + k[h,k] * delta[h,v]
//   o[h,v]   = SUM_k S[h,k,v] * q[h,k]
//
// Production geometry: H = 64, D = 128, state [64,128,128] fp32.
//
// ─────────────────────────────────────────────────────────────────────────────
// GATE CONVENTION — the exp() belongs HERE, and this is the trap
// ─────────────────────────────────────────────────────────────────────────────
// `gate` is the **log-decay** produced by `kda_gate` — `lower_bound*sigmoid(...)`,
// which lies in [lower_bound, 0]. This kernel exponentiates it, so exp(gate) lies
// in (0, 1]. All three sources agree:
//
//   HF 5.16.1 recurrent_kimi_delta_attention:  g_i = g[:, i][..., None].exp()
//   vLLM fused_recurrent.py:                   b_state *= exp(b_gate[None, :])
//   Slice 2 CPU reference:                     let decay = gate[base + kd].exp();
//
// ⚠️ Atlas's OWN Qwen GDN uses the OPPOSITE convention on BOTH axes:
// `ssm_preprocess.cu::compute_gdn_gates` stores `gate_tok[vh] = __expf(g)` — already
// exponentiated — and `gated_delta_rule.cu::gated_delta_rule_decode` documents its
// input as `const float* gate  // [batch, num_v_heads] exp(g_t) decay`, one SCALAR
// PER HEAD. Feeding a KDA gate into a GDN consumer is a missing-exp bug; feeding a
// GDN gate here is a double-exp bug. Neither would crash. Both would silently change
// decay dynamics.
//
// ⚠️ `gated_delta_rule_decode` additionally CLAMPS the gate to (0,1) and clamps the
// per-head h-state Frobenius norm ("Stuffed Mamba" state-explosion guard). Neither
// clamp exists in the HF or vLLM KDA path, so neither is applied here. KDA does not
// need them: the bounded gate already confines exp(gate) to (0, 1] by construction.
//
// ─────────────────────────────────────────────────────────────────────────────
// INPUT CONTRACT
// ─────────────────────────────────────────────────────────────────────────────
// `q` and `k` arrive **already L2-normalised** — Atlas's convention, where
// `causal_conv1d_update_l2norm` fuses conv + SiLU + L2 upstream. HF instead carries
// raw q/k into its kernel and normalises there (`use_qk_l2norm_in_kernel=True`); the
// two are the same computation in a different place. `scale` (= 1/sqrt(D)) is applied
// to q BEFORE the accumulation, matching HF's `query = query * scale`.
//
// `beta` arrives **already sigmoided**, matching vLLM's non-`APPLY_BETA_SIGMOID` path.
//
// ─────────────────────────────────────────────────────────────────────────────
// LAYOUT AND WHY IT IS THIS ONE
// ─────────────────────────────────────────────────────────────────────────────
// state is [H, K, V], K-major — HF's `last_recurrent_state` shape (B, H, k_dim, v_dim)
// and Atlas's existing `h_state_bytes` comment `FP32 [nv, kd, vd]`. vLLM stores the
// transpose [H, V, K]; same math, different traversal.
//
// One thread per V, looping K sequentially in ascending order. That choice does two
// things at once:
//   * threads in a warp read consecutive `v` for a fixed `k`, so every state access is
//     fully coalesced under a K-major layout;
//   * the per-thread accumulation order over K is IDENTICAL to the CPU reference's, so
//     any residual against it is arithmetic, not reduction-order.
//
// Correctness-first, deliberately not optimised: the state is read twice and written
// twice per token (decay+kv pass, then update+out pass). Holding a whole V-row in
// shared memory would halve that to 1R+1W but costs D*D*4 = 64 KiB of shared memory per
// block at production geometry, which is an occupancy decision that belongs to a later
// slice, not to a correctness gate.
//
// Shared memory is used only for the three per-head [D] vectors (decay, k, scaled q):
// without it, each of the D threads would recompute all D `expf` calls, D^2 = 16384
// transcendentals per head instead of D = 128.

#include <cuda_bf16.h>
#include <math.h>

// dynamic shared: 3*D floats -> exp(gate) | k | q*scale
#define KDA_REC_BODY(LOAD_QKV)                                                        \
    extern __shared__ float sh[];                                                     \
    const unsigned int h = blockIdx.x;                                                \
    if (h >= H) return;                                                               \
    float* sh_decay = sh;                                                             \
    float* sh_k = sh + D;                                                             \
    float* sh_q = sh + 2u * D;                                                        \
    const size_t hd = (size_t)h * D;                                                  \
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {                       \
        sh_decay[i] = expf(gate[hd + i]);                                             \
        sh_k[i] = LOAD_QKV(k[hd + i]);                                                \
        sh_q[i] = LOAD_QKV(q[hd + i]) * scale;                                        \
    }                                                                                 \
    __syncthreads();                                                                  \
    const float b = beta[h];                                                          \
    float* S = state + hd * D;                                                        \
    for (unsigned int vi = threadIdx.x; vi < D; vi += blockDim.x) {                    \
        float kv = 0.0f;                                                              \
        /* 🔴 `#pragma unroll 8` is a MEMORY-LEVEL-PARALLELISM fix, not a code-size one. \
           Production geometry is grid (H,1,1) x 128 threads = 4096 threads per rank, \
           each walking D=128 state elements ONE dependent global load at a time, so   \
           the kernel had at most 4096 loads in flight and measured 69 us/call at ~121 \
           GB/s — half of what every other decode GEMV on this model reaches. Eight     \
           independent loads per thread give it 8x the outstanding requests.            \
           BIT-IDENTICAL: unrolling does not reassociate. The `kv`/`o` accumulator      \
           chains stay strictly kk = 0..D-1, which is the ordering the CPU reference and \
           the byte-identity gate pin. nvcc emits a remainder loop for the runtime D. */ \
        _Pragma("unroll 8")                                                           \
        for (unsigned int kk = 0; kk < D; ++kk) {                                      \
            const size_t idx = (size_t)kk * D + vi;                                   \
            const float s = S[idx] * sh_decay[kk];                                    \
            S[idx] = s;                                                               \
            kv += s * sh_k[kk];                                                       \
        }                                                                             \
        const float delta = (LOAD_QKV(v[hd + vi]) - kv) * b;                          \
        float o = 0.0f;                                                               \
        _Pragma("unroll 8")                                                           \
        for (unsigned int kk = 0; kk < D; ++kk) {                                      \
            const size_t idx = (size_t)kk * D + vi;                                   \
            const float s = S[idx] + sh_k[kk] * delta;                                \
            S[idx] = s;                                                               \
            o += s * sh_q[kk];                                                        \
        }                                                                             \
        out[hd + vi] = o;                                                             \
    }

#define KDA_REC_IDENT(x) (x)
#define KDA_REC_BF16(x) __bfloat162float(x)

// Oracle / high-precision entry point: fp32 q/k/v.
extern "C" __global__ void kda_recurrent_decode_f32(
    const float* __restrict__ q,      // [H, D] fp32, ALREADY L2-normalised
    const float* __restrict__ k,      // [H, D] fp32, ALREADY L2-normalised
    const float* __restrict__ v,      // [H, D] fp32
    const float* __restrict__ gate,   // [H, D] fp32, LOG-decay from kda_gate
    const float* __restrict__ beta,   // [H]    fp32, ALREADY sigmoided
    float* __restrict__ state,        // [H, D, D] fp32, K-major, read-modify-write
    float* __restrict__ out,          // [H, D] fp32
    unsigned int H,
    unsigned int D,
    float scale                       // 1/sqrt(D)
) {
    KDA_REC_BODY(KDA_REC_IDENT)
}

// Production entry point: bf16 q/k/v (Atlas's conv writes bf16). Gate, beta, state and
// output stay fp32 — the recurrent state is fp32 by REFERENCE SEMANTICS, not by Atlas
// policy: HF stores it via `last_recurrent_state.to(torch.float32)` and vLLM's
// `MambaStateDtypeCalculator.kda_state_dtype` returns `(conv_dtype, torch.float32)`.
extern "C" __global__ void kda_recurrent_decode_bf16(
    const __nv_bfloat16* __restrict__ q,   // [H, D] bf16, ALREADY L2-normalised
    const __nv_bfloat16* __restrict__ k,   // [H, D] bf16, ALREADY L2-normalised
    const __nv_bfloat16* __restrict__ v,   // [H, D] bf16
    const float* __restrict__ gate,        // [H, D] fp32, LOG-decay
    const float* __restrict__ beta,        // [H]    fp32, ALREADY sigmoided
    float* __restrict__ state,             // [H, D, D] fp32, K-major
    float* __restrict__ out,               // [H, D] fp32
    unsigned int H,
    unsigned int D,
    float scale
) {
    KDA_REC_BODY(KDA_REC_BF16)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1R+1W VARIANT — the state column lives in shared memory between the two passes
// ─────────────────────────────────────────────────────────────────────────────
// The body above is deliberately unoptimised: it reads and writes the whole [D, D]
// state TWICE per token because `delta` needs the complete `kv` sum before the second
// pass can start, and each thread re-reads its column from global to get it. nsys
// (2026-08-28, graph-node trace) charged that 2.34 ms/token across 34 layers at ~121
// GB/s — half of what every other decode kernel on this model reaches, and 3.6 % of
// the step.
//
// The header called holding a V-row "an occupancy decision": 128 threads x D floats is
// 64 KiB of shared memory per block. That framing missed the free axis — the V axis has
// **no cross-thread dependency at all** (`kv`, `delta` and `o` are per `(h, vi)`), so
// the block can own a SLICE of it. `VPB` v-columns per block need `VPB * D` floats;
// grid.y covers `D / VPB`, and each block still touches only its own columns, so the
// total global traffic is 1R+1W however the axis is cut.
//
// BIT-IDENTICAL to `kda_recurrent_decode_bf16`: same thread-to-`vi` mapping, same
// strictly ascending `kk = 0..D-1` accumulation for `kv` and `o`, same operands. A
// float written to shared memory and read back is the same float; only the address
// space changed. Threads in a warp still walk consecutive `vi` for a fixed `kk`, so the
// global reads stay fully coalesced.
//
// 🪤 `VPB` must divide `D` and `blockDim.x` must equal `VPB` — the launcher owns both,
// and a mismatch silently drops or double-counts columns. Kill switch on the Rust side:
// `ATLAS_GLM_KDA_NO_SMEM=1` restores the 2R+2W kernel.
extern "C" __global__ void kda_recurrent_decode_bf16_smem(
    const __nv_bfloat16* __restrict__ q,   // [H, D] bf16, ALREADY L2-normalised
    const __nv_bfloat16* __restrict__ k,   // [H, D] bf16, ALREADY L2-normalised
    const __nv_bfloat16* __restrict__ v,   // [H, D] bf16
    const float* __restrict__ gate,        // [H, D] fp32, LOG-decay
    const float* __restrict__ beta,        // [H]    fp32, ALREADY sigmoided
    float* __restrict__ state,             // [H, D, D] fp32, K-major
    float* __restrict__ out,               // [H, D] fp32
    unsigned int H,
    unsigned int D,
    float scale,
    unsigned int VPB                       // v-columns this block owns; == blockDim.x
) {
    extern __shared__ float sh[];
    const unsigned int h = blockIdx.x;
    if (h >= H) return;
    const unsigned int v0 = blockIdx.y * VPB;
    if (v0 >= D) return;

    float* sh_decay = sh;
    float* sh_k = sh + D;
    float* sh_q = sh + 2u * D;
    // [VPB, D + 1] — column `vi` of this block's slice, k-major within the column.
    //
    // 🪤 The `+1` pad is not slack, it is the whole point. At a stride of exactly D = 128
    // floats every thread in a warp hits the SAME shared-memory bank (128 % 32 == 0) and
    // the two passes serialise 32 ways. A stride of D+1 rotates each thread by one bank
    // and the access is conflict-free. The launcher sizes the request to match.
    float* sh_s = sh + 3u * D;
    const unsigned int col_stride = D + 1u;

    const size_t hd = (size_t)h * D;
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
        sh_decay[i] = expf(gate[hd + i]);
        sh_k[i] = __bfloat162float(k[hd + i]);
        sh_q[i] = __bfloat162float(q[hd + i]) * scale;
    }
    __syncthreads();

    const float b = beta[h];
    float* S = state + hd * D;
    const unsigned int vi = v0 + threadIdx.x;
    if (threadIdx.x >= VPB || vi >= D) return;
    float* col = sh_s + (size_t)threadIdx.x * col_stride;

    float kv = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = S[(size_t)kk * D + vi] * sh_decay[kk];
        col[kk] = s;
        kv += s * sh_k[kk];
    }
    const float delta = (__bfloat162float(v[hd + vi]) - kv) * b;
    float o = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = col[kk] + sh_k[kk] * delta;
        S[(size_t)kk * D + vi] = s;
        o += s * sh_q[kk];
    }
    out[hd + vi] = o;
}
