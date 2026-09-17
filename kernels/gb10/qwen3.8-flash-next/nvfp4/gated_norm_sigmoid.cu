// SPDX-License-Identifier: AGPL-3.0-only
//
// SIGMOID-gated RMS norm — the Qwen3.8-Flash-Next GDN twins.
//
// `Qwen4ExpTextRMSNormGated` takes its gate activation from
// `config.output_gate_type or config.hidden_act`, and this checkpoint says
// `output_gate_type: "sigmoid"`. Every other Qwen-family GDN model gates
// with SiLU, which is what `common/rms_norm.cu` hardcodes — correct there,
// wrong here, on the output of all 36 GDN layers.
//
// Found by the phase-E bisect (Avarok #753): with the recurrence PROVEN
// textbook-correct (token-0 output parallel to v, magnitude ratio 1.00) and
// every input verified, the norm stage still diverged at cos 0.81 — and the
// reference module disagreed with its own printed source, because the
// activation is a CONSTRUCTOR ARGUMENT the printed forward hides behind
// `ACT2FN[self.activation]`. Swapping sigmoid in matched to max|diff| = 0.0.
//
// DERIVED from common/rms_norm.cu: entry points renamed `*_sigmoid`, each
// `g / (1 + exp(-g))` on a gate value became `1 / (1 + exp(-g))`. Nothing
// else differs; a fix to the SiLU originals should be re-derived here.
//
// Selected at LAYER INIT (`gdn_norm_sigmoid`): handles swapped once, no
// forward call-site changes, no other model resolves these.

#include <cuda_bf16.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ void unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

// Fused RMS Norm + Gated variant (for Mamba layers).
// out = rms_norm(x) * SiLU(gate)   where SiLU(x) = x * sigmoid(x)
//
// Optimized: register-cached x values (eliminates second read of input),
// 4-wide BF16 vectorized loads (64-bit), fused normalize+gate pass.
extern "C" __global__ void gated_rms_norm_sigmoid(
    const __nv_bfloat16* __restrict__ input,   // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ gate,    // [num_tokens, gate_stride] (gate_stride >= hidden_size)
    const __nv_bfloat16* __restrict__ weight,  // [hidden_size]
    __nv_bfloat16* __restrict__ output,         // [num_tokens, hidden_size]
    unsigned int hidden_size,
    float eps,
    unsigned int gate_stride,                   // elements between gate rows (may differ from hidden_size)
    unsigned int group_size                     // unused in Qwen3 (norm_before_gate=True), kept for API compat
) {
    (void)group_size;
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + token * hidden_size;
    const __nv_bfloat16* g = gate + (unsigned long long)token * gate_stride;
    __nv_bfloat16* out = output + token * hidden_size;

    // 4-wide BF16 loads: process 4 elements per iteration via 64-bit reads.
    // Register cache: store x values to avoid re-reading in pass 2.
    // Max 16 elements per thread (supports hidden_size up to 16*1024 = 16K).
    const unsigned int quad_size = hidden_size / 4;
    const unsigned long long* x64 = (const unsigned long long*)x;

    float x_cache[16];
    float sum_sq = 0.0f;
    unsigned int n_cached = 0;

    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned long long v = x64[i];
        float f0, f1, f2, f3;
        unpack_bf16x2((unsigned int)v, f0, f1);
        unpack_bf16x2((unsigned int)(v >> 32), f2, f3);
        x_cache[n_cached]     = f0;
        x_cache[n_cached + 1] = f1;
        x_cache[n_cached + 2] = f2;
        x_cache[n_cached + 3] = f3;
        n_cached += 4;
        sum_sq += f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3;
    }

    // Block-level reduction
    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    // Pass 2: Apply normalization + gate using cached x values (no re-read of input).
    // 4-wide vectorized weight and gate loads, 4-wide output stores.
    const unsigned long long* g64 = (const unsigned long long*)g;
    const unsigned long long* w64 = (const unsigned long long*)weight;
    unsigned long long* out64 = (unsigned long long*)out;

    unsigned int ci = 0;
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        float f0 = x_cache[ci];
        float f1 = x_cache[ci + 1];
        float f2 = x_cache[ci + 2];
        float f3 = x_cache[ci + 3];
        ci += 4;

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        unpack_bf16x2((unsigned int)wv, w0, w1);
        unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        unpack_bf16x2((unsigned int)gv, g0, g1);
        unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = 1.0f / (1.0f + expf(-g0));  // SiLU
        float s1 = 1.0f / (1.0f + expf(-g1));  // SiLU
        float s2 = 1.0f / (1.0f + expf(-g2));  // SiLU
        float s3 = 1.0f / (1.0f + expf(-g3));  // SiLU

        unsigned int lo = pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}

// FP32-input variant: accepts GDN output in FP32 (no BF16 truncation in the
// recurrent path). Gate is still BF16 (from Z projection), weight is BF16,
// output is BF16 (feeds into the BF16 output projection).
extern "C" __global__ void gated_rms_norm_f32_input_sigmoid(
    const float* __restrict__ input,              // [num_tokens, hidden_size] FP32
    const __nv_bfloat16* __restrict__ gate,       // [num_tokens, gate_stride]
    const __nv_bfloat16* __restrict__ weight,     // [hidden_size]
    __nv_bfloat16* __restrict__ output,            // [num_tokens, hidden_size]
    unsigned int hidden_size,
    float eps,
    unsigned int gate_stride,
    unsigned int group_size
) {
    (void)group_size;
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const float* x = input + token * hidden_size;
    const __nv_bfloat16* g = gate + (unsigned long long)token * gate_stride;
    __nv_bfloat16* out = output + token * hidden_size;

    // Pass 1: compute sum of squares (FP32 input — no BF16 unpack needed)
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float f = x[i];
        sum_sq += f * f;
    }

    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    // Pass 2: Apply normalization + gate (re-read FP32 from L1 cache)
    const unsigned long long* g64 = (const unsigned long long*)g;
    const unsigned long long* w64 = (const unsigned long long*)weight;
    unsigned long long* out64 = (unsigned long long*)out;

    const unsigned int quad_size = hidden_size / 4;
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned int base = i * 4;
        float f0 = x[base];
        float f1 = x[base + 1];
        float f2 = x[base + 2];
        float f3 = x[base + 3];

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        unpack_bf16x2((unsigned int)wv, w0, w1);
        unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        unpack_bf16x2((unsigned int)gv, g0, g1);
        unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = 1.0f / (1.0f + expf(-g0));
        float s1 = 1.0f / (1.0f + expf(-g1));
        float s2 = 1.0f / (1.0f + expf(-g2));
        float s3 = 1.0f / (1.0f + expf(-g3));

        unsigned int lo = pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}

// Batched Gated RMS Norm for prefill: processes all (head, token) pairs
// in a single kernel launch instead of N separate launches per actual token.
//
// Grid: (heads_per_token, num_actual_tokens, 1)
// Block: (min(head_dim, 1024), 1, 1)
extern "C" __global__ void gated_rms_norm_prefill_sigmoid(
    const __nv_bfloat16* __restrict__ input,   // GDN output base
    const __nv_bfloat16* __restrict__ gate,    // Z gate base
    const __nv_bfloat16* __restrict__ weight,  // [head_dim]
    __nv_bfloat16* __restrict__ output,         // normed output base
    unsigned int head_dim,
    float eps,
    unsigned int input_token_stride,            // BF16 elements between actual tokens in input/output
    unsigned int gate_token_stride              // BF16 elements between actual tokens in gate
) {
    unsigned int head = blockIdx.x;
    unsigned int token = blockIdx.y;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + (unsigned long long)token * input_token_stride + head * head_dim;
    const __nv_bfloat16* g = gate + (unsigned long long)token * gate_token_stride + head * head_dim;
    __nv_bfloat16* out = output + (unsigned long long)token * input_token_stride + head * head_dim;

    const unsigned int quad_size = head_dim / 4;
    const unsigned long long* x64 = (const unsigned long long*)x;

    float x_cache[16];
    float sum_sq = 0.0f;
    unsigned int n_cached = 0;

    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned long long v = x64[i];
        float f0, f1, f2, f3;
        unpack_bf16x2((unsigned int)v, f0, f1);
        unpack_bf16x2((unsigned int)(v >> 32), f2, f3);
        x_cache[n_cached]     = f0;
        x_cache[n_cached + 1] = f1;
        x_cache[n_cached + 2] = f2;
        x_cache[n_cached + 3] = f3;
        n_cached += 4;
        sum_sq += f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)head_dim + eps);

    const unsigned long long* g64 = (const unsigned long long*)g;
    const unsigned long long* w64 = (const unsigned long long*)weight;
    unsigned long long* out64 = (unsigned long long*)out;

    unsigned int ci = 0;
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        float f0 = x_cache[ci];
        float f1 = x_cache[ci + 1];
        float f2 = x_cache[ci + 2];
        float f3 = x_cache[ci + 3];
        ci += 4;

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        unpack_bf16x2((unsigned int)wv, w0, w1);
        unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        unpack_bf16x2((unsigned int)gv, g0, g1);
        unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = 1.0f / (1.0f + expf(-g0));
        float s1 = 1.0f / (1.0f + expf(-g1));
        float s2 = 1.0f / (1.0f + expf(-g2));
        float s3 = 1.0f / (1.0f + expf(-g3));

        unsigned int lo = pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}
