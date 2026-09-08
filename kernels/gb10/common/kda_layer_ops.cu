// SPDX-License-Identifier: AGPL-3.0-only
//
// GLM-5.3-Flash KDA layer glue — Slice 6.
//
// Three small entry points the integrated KDA layer needs and Atlas does not already have.
//
// ★ kda_o_norm_gated_* is NOT a duplicate of `gated_rms_norm_f32_input`.
//   Every Atlas gated RMSNorm applies **SiLU** to the gate (`rms_norm.cu:1087`,
//   `g / (1 + expf(-g))`). GLM's `Glm5NextTextRMSNormGated` sets `self.activation = "sigmoid"`
//   and applies **sigmoid** (`transformers` 5.16.1, `modeling_glm5_next.py:343-359`).
//   There is no substitution that turns one into the other, so `o_norm` is ADAPT, not REUSE —
//   correcting the Slice-2D classification. Everything else about the two kernels agrees:
//   strict-FP32 norm over the trailing `head_dim`, FP32 core input, BF16 weight/gate/output.
//
// The other two are plumbing between kernels whose dtypes differ by design:
//   * `kda_widen_bf16_f32` — Atlas's prefill conv emits BF16; `kda_chunk_*` consume FP32.
//   * `kda_sigmoid_bf16_f32` — `b_proj` emits BF16; `kda_chunk_*` / `kda_recurrent_*` want
//     an already-sigmoided FP32 `beta` (HF: `beta = torch.sigmoid(self.b_proj(hidden))`).

#include <cuda_bf16.h>

// out[row, d] = rms(x[row])[d] * weight[d] * sigmoid(gate[row, d])
//
// grid = (num_rows), block = (head_dim). `num_rows = num_tokens * heads`; each row is one
// head's `head_dim` slice, matching HF's `o_norm(core_attn_out, gate)` over the trailing dim.
#define KDA_ONORM_BODY(GATE_LD, W_LD, OUT_ST)                                                 \
    const unsigned int row = blockIdx.x;                                                      \
    const unsigned int tid = threadIdx.x;                                                     \
    const float* x = input + (size_t)row * head_dim;                                          \
    float acc = 0.0f;                                                                         \
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) { float f = x[i]; acc += f * f; }\
    __shared__ float red[32];                                                                 \
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);     \
    if ((tid & 31) == 0) red[tid >> 5] = acc;                                                 \
    __syncthreads();                                                                          \
    if (tid < 32) {                                                                           \
        float v = (tid < ((blockDim.x + 31) / 32)) ? red[tid] : 0.0f;                          \
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);      \
        if (tid == 0) red[0] = v;                                                             \
    }                                                                                         \
    __syncthreads();                                                                          \
    const float inv = rsqrtf(red[0] / (float)head_dim + eps);                                 \
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) {                                \
        const float g = (GATE_LD);                                                            \
        const float s = 1.0f / (1.0f + __expf(-g));   /* sigmoid, NOT SiLU */                 \
        OUT_ST(x[i] * inv * (W_LD) * s);                                                      \
    }

// Production: FP32 core in (the recurrent path is FP32 by reference semantics), BF16 out.
extern "C" __global__ void kda_o_norm_gated_bf16(
    const float* __restrict__ input,           // [num_rows, head_dim] fp32
    const __nv_bfloat16* __restrict__ gate,    // [num_rows, head_dim] bf16
    const __nv_bfloat16* __restrict__ weight,  // [head_dim]           bf16
    __nv_bfloat16* __restrict__ output,        // [num_rows, head_dim] bf16
    unsigned int head_dim,
    float eps
) {
#define KDA_ONORM_OUT_BF16(val) output[(size_t)row * head_dim + i] = __float2bfloat16(val)
    KDA_ONORM_BODY(__bfloat162float(gate[(size_t)row * head_dim + i]),
                   __bfloat162float(weight[i]),
                   KDA_ONORM_OUT_BF16)
#undef KDA_ONORM_OUT_BF16
}

// Oracle: everything FP32, so the microtest can separate kernel residual from bf16 rounding.
extern "C" __global__ void kda_o_norm_gated_f32(
    const float* __restrict__ input,
    const float* __restrict__ gate,
    const float* __restrict__ weight,
    float* __restrict__ output,
    unsigned int head_dim,
    float eps
) {
#define KDA_ONORM_OUT_F32(val) output[(size_t)row * head_dim + i] = (val)
    KDA_ONORM_BODY(gate[(size_t)row * head_dim + i], weight[i], KDA_ONORM_OUT_F32)
#undef KDA_ONORM_OUT_F32
}

// De-interleave + widen: Atlas's prefill conv writes ONE BF16 buffer whose channels are
// `q | k | v` per token, while `kda_chunk_*` take three separate FP32 `[T_pad, H*D]` buffers.
// BF16 is a strict subset of FP32, so the widen itself contributes exactly zero error.
//
// Positions `t >= T` are NOT written. That is deliberate: the Slice-5 pad-corruption bug wrote
// correct outputs while poisoning the carried recurrent state, and the fix made `kda_chunk_*`
// self-guarding. Leaving the pad tail at whatever the caller put there is what lets the
// microtest fill it with poison and prove the guard.
extern "C" __global__ void kda_split_widen(
    const __nv_bfloat16* __restrict__ src,  // [T, 3 * qkv] bf16, channels q | k | v
    float* __restrict__ q,                  // [T_pad, qkv] fp32
    float* __restrict__ k,
    float* __restrict__ v,
    unsigned int T,
    unsigned int qkv
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int t = blockIdx.y;
    if (ch >= qkv || t >= T) return;
    const size_t s = (size_t)t * 3 * qkv + ch;
    const size_t d = (size_t)t * qkv + ch;
    q[d] = __bfloat162float(src[s]);
    k[d] = __bfloat162float(src[s + qkv]);
    v[d] = __bfloat162float(src[s + 2 * qkv]);
}

// Device-side fill, so the padded chunk buffers can be primed (with zero, or with poison for
// the pad-guard regression) without a host round trip.
extern "C" __global__ void kda_fill_f32(float* __restrict__ dst, unsigned int n, float value) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = value;
}

// beta = sigmoid(b_proj(hidden)), BF16 in / FP32 out. The KDA kernels take beta ALREADY
// sigmoided; feeding them the raw projection is silent (no shape or dtype change).
extern "C" __global__ void kda_sigmoid_bf16_f32(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __bfloat162float(src[i]);
        dst[i] = 1.0f / (1.0f + __expf(-x));
    }
}

// Interleave three [T, qkv] projections into the [T, 3*qkv] q|k|v layout the fused depthwise
// conv consumes.
//
// ★ Needed because `dense_gemm_bf16` writes `C[row * N + col]` — its output row stride is N,
//   not a caller-chosen stride. Aiming the three GEMMs at offsets inside one [T, 3*qkv] buffer
//   therefore has them overwrite each other for T > 1, and is silently CORRECT at T = 1 because
//   the two layouts coincide there. A decode-only test cannot catch it.
extern "C" __global__ void kda_pack_qkv_bf16(
    const __nv_bfloat16* __restrict__ q,   // [T, qkv]
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ dst,       // [T, 3 * qkv]
    unsigned int T,
    unsigned int qkv
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int t = blockIdx.y;
    if (ch >= qkv || t >= T) return;
    const size_t s = (size_t)t * qkv + ch;
    const size_t d = (size_t)t * 3 * qkv + ch;
    dst[d] = q[s];
    dst[d + qkv] = k[s];
    dst[d + 2 * qkv] = v[s];
}
