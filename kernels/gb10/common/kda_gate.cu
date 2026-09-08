// SPDX-License-Identifier: AGPL-3.0-only
//
// GLM-5.3-Flash KDA (Kimi Delta Attention) bounded forget gate.
//
//   decay[h]      = exp(A_log[h])                                  per HEAD
//   gate[t,h,d]   = lower_bound * sigmoid(decay[h] * (g_raw[t,h,d] + dt_bias[h,d]))
//
// Production geometry: H = 64, D = 128, gate output [T, 64, 128] fp32.
//
// ─────────────────────────────────────────────────────────────────────────────
// WHY THIS IS NOT `compute_gdn_gates`
// ─────────────────────────────────────────────────────────────────────────────
// `ssm_preprocess.cu::compute_gdn_gates` is Qwen3-Next GDN and is structurally
// unable to express this gate. Its signature is:
//
//     const float* A_log;     // [num_v_heads]
//     const float* dt_bias;   // [num_v_heads]     <-- ONE SCALAR PER HEAD
//     float*       gate_out;  // [num_tokens, num_v_heads]
//     float g = -A_val * dt;  gate_tok[vh] = __expf(g);
//
// KDA's decay is per (head, KEY-CHANNEL): `dt_bias` is [H*D] = 8192 and the gate
// output is [T, H, D] — 128x wider per head. Reusing `compute_gdn_gates` would
// silently collapse the channel axis, so this is a separate kernel by necessity,
// not by preference. See docs/glm5next/KDA-VS-QWEN-GDN.md.
//
// The law also differs. Qwen GDN is unbounded `exp(-exp(A_log)*softplus(a+bias))`;
// KDA is bounded `lower_bound * sigmoid(exp(A_log)*(g+bias))`, saturating at
// `lower_bound` instead of growing without limit.
//
// ─────────────────────────────────────────────────────────────────────────────
// NUMERICS
// ─────────────────────────────────────────────────────────────────────────────
// `expf`, not `__expf`. The fast intrinsic carries ~2 ulp of error, and this
// kernel is bandwidth-bound (it reads T*H*D and writes T*H*D), so the accurate
// call is effectively free and buys bit-exact agreement with the HuggingFace
// `transformers` 5.16.1 reference.
//
// The sigmoid is written as the naive `1/(1+exp(-x))` deliberately, to match
// torch's formulation element for element rather than an algebraically-equal
// rearrangement that would differ in the last ulp. It is safe at both limits:
//   x = +100  -> expf(-100) subnormal      -> gate = lower_bound
//   x = -1000 -> expf(1000) = +inf, 1/inf  -> gate = -0.0
// No NaN is reachable for finite input.
//
// `lower_bound` is a KERNEL ARGUMENT, never a compiled-in constant. It comes
// from the checkpoint's `linear_attn_config.gate_lower_bound`. It happens to be
// -5.0 for GLM-5.3-Flash, and vLLM happens to arrive at the same number only
// because it looks up a legacy key that is absent and falls back to a matching
// default. Atlas must not inherit that coincidence.
//
// ─────────────────────────────────────────────────────────────────────────────
// LAUNCH GEOMETRY
// ─────────────────────────────────────────────────────────────────────────────
//   grid  = (num_tokens * H, 1, 1)   one block per (token, head) row
//   block = (128, 1, 1)              threads stride over D
//
// One block owns a whole row, so `A_log[h]` and its `expf` are loaded and
// evaluated once per block rather than once per element, and `dt_bias`/`g_raw`
// are read fully coalesced. The `% H` is likewise once per block. D is looped so
// the kernel stays correct for D != blockDim.x.

#include <cuda_bf16.h>
#include <math.h>

// Shared scalar core. Kept in one place so the bf16 and fp32 entry points cannot
// drift apart.
__device__ __forceinline__ float kda_gate_scalar(float g_raw, float dt_bias,
                                                 float decay, float lower_bound) {
    return lower_bound * (1.0f / (1.0f + expf(-(decay * (g_raw + dt_bias)))));
}

// Production entry point: `g_raw` is the bf16 output of the low-rank `f_b`
// projection, matching Atlas's convention for GDN staging buffers.
extern "C" __global__ void kda_gate_bf16(
    const __nv_bfloat16* __restrict__ g_raw,   // [num_tokens, H, D] bf16
    const float* __restrict__ dt_bias,         // [H, D]             fp32, PER CHANNEL
    const float* __restrict__ A_log,           // [H]                fp32, PER HEAD
    float* __restrict__ gate_out,              // [num_tokens, H, D] fp32
    unsigned int num_tokens,
    unsigned int H,
    unsigned int D,
    float lower_bound
) {
    unsigned int row = blockIdx.x;             // row = t * H + h
    if (row >= num_tokens * H) return;
    unsigned int h = row % H;

    const float decay = expf(A_log[h]);
    const __nv_bfloat16* g_row = g_raw + (size_t)row * D;
    const float* b_row = dt_bias + (size_t)h * D;
    float* o_row = gate_out + (size_t)row * D;

    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        o_row[d] = kda_gate_scalar(__bfloat162float(g_row[d]), b_row[d], decay, lower_bound);
    }
}

// Oracle / high-precision entry point: identical arithmetic, fp32 `g_raw`.
// Used by `kda_gate_microtest` so the comparison against the HuggingFace golden
// measures THIS kernel's error rather than the bf16 rounding of its input.
extern "C" __global__ void kda_gate_f32(
    const float* __restrict__ g_raw,           // [num_tokens, H, D] fp32
    const float* __restrict__ dt_bias,         // [H, D]             fp32, PER CHANNEL
    const float* __restrict__ A_log,           // [H]                fp32, PER HEAD
    float* __restrict__ gate_out,              // [num_tokens, H, D] fp32
    unsigned int num_tokens,
    unsigned int H,
    unsigned int D,
    float lower_bound
) {
    unsigned int row = blockIdx.x;
    if (row >= num_tokens * H) return;
    unsigned int h = row % H;

    const float decay = expf(A_log[h]);
    const float* g_row = g_raw + (size_t)row * D;
    const float* b_row = dt_bias + (size_t)h * D;
    float* o_row = gate_out + (size_t)row * D;

    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        o_row[d] = kda_gate_scalar(g_row[d], b_row[d], decay, lower_bound);
    }
}
