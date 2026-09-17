// SPDX-License-Identifier: AGPL-3.0-only
//
// PLE — hashed n-gram injection into the hyper-connection highway.
// Qwen3.8-Flash-Next, ONE layer (model layer 1). Avarok #753 item C.
//
// Reference: `Qwen4ExpTextPLELayer.forward`,
// bench/qwen4_exp/ref/modeling_qwen4_exp.py L1168.
//
//   key_normed   = norm_key(key_proj(emb)).unflatten(-1, (hc, H))
//   value        = value_proj(emb)                            # [T, H]
//   query_normed = norm_query(hidden).unflatten(-1, (hc, H))  # hidden [T, hc*H]
//   gate  = (key_normed * query_normed).sum(-1) / sqrt(H)     # [T, hc]
//   gate  = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()   # SIGNED SQRT
//   gated = sigmoid(gate) * value.unsqueeze(-2)               # [T, hc, H]
//   out   = gated.flatten(-2) + silu(conv1d(norm_conv(gated.flatten(-2))))
//
// THREE THINGS HERE ARE EASY TO GET WRONG AND IMPOSSIBLE TO SEE:
//
//  1. All three norms are `Qwen4ExpTextRMSNorm` — `normed * (1.0 + weight)`,
//     initialised to ZEROS. NOT `normed * weight`. The checkpoint settles it:
//     `ple.norm_key` has mean -0.1067. Dropping the offset for w~0 gives a
//     near-null gate: finite, plausible, wrong. Same trap as `hc_norm`; see
//     bench/qwen4_exp/ARCHITECTURE.md §6.
//  2. They are GROUPED, `group_size = hidden_size`: the `hc` streams
//     normalize INDEPENDENTLY inside the `hc*H` vector. One RMS over all
//     10240 is a different function of the same shape.
//  3. The gate's SIGNED SQUARE ROOT. Nobody would guess it from the tensor
//     names, and omitting it leaves the gate distribution wrong but finite.
//
// The conv is depthwise (`groups = hc*H`) with kernel 4 and DILATION 3, so
// its state is (4-1)*3 = 9 steps — not 3. It lives in `ple.cu` rather than
// reusing `causal_conv1d.cu` for exactly that reason: that kernel has no
// dilation, and silently treating dilation as 1 reads the wrong 4 timesteps.

// PRECISION NOTE — the whole PLE chain is FP32, deliberately.
//
// PLE's output is added to the FP32 mHC highway, so rounding it to BF16 on
// the way there is a round-trip that buys nothing and costs real accuracy.
//
// It matters more than it looks. `norm_conv`'s output peaks near |12| while
// its RMS is ~0.8, and the conv sums four taps of it into an output whose own
// RMS is 0.019 — heavy cancellation, so the absolute error is set by the LARGE
// terms and lands, relatively, on the small result. Measured end to end:
// BF16 intermediates gave 2.7% relative error on the output; FP32 throughout
// gives well under 1%.
//
// The cost is `T * 10240 * 4` on ONE layer of 48. Only `key`/`value` (the
// projection outputs, which are read once and are not cancelled against
// anything) stay BF16.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define PLE_BLOCK 256
#define PLE_MAX_STREAMS 8

__device__ __forceinline__ float ple_silu(float v) {
    return v / (1.0f + __expf(-v));
}

__device__ __forceinline__ float ple_sigmoid(float v) {
    return 1.0f / (1.0f + __expf(-v));
}

// Block-wide sum of `val`, result broadcast to every thread.
__device__ __forceinline__ float ple_block_sum(float val, float* smem_red) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int warps = PLE_BLOCK / 32;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFFu, val, off);
    }
    if (lane == 0) smem_red[warp] = val;
    __syncthreads();
    float tot = 0.0f;
    for (unsigned int w = 0; w < warps; ++w) tot += smem_red[w];
    __syncthreads();
    return tot;
}

// ── ple_gate ──
// One block per token. Computes the gate, applies it to `value`, and emits
// BOTH the gated value (which the final add uses un-normalized) and its
// norm_conv'd twin (which the conv consumes).
//
// `hidden` is the FP32 mHC highway; `key`/`value` are BF16 projection
// outputs. Everything accumulates in FP32.
extern "C" __global__ void ple_gate(
    const float* __restrict__ hidden,          // [T, hc*H] FP32 highway
    const __nv_bfloat16* __restrict__ key,     // [T, hc*H] key_proj out
    const __nv_bfloat16* __restrict__ value,   // [T, H]    value_proj out
    const __nv_bfloat16* __restrict__ norm_query_w, // [hc*H]
    const __nv_bfloat16* __restrict__ norm_key_w,   // [hc*H]
    const __nv_bfloat16* __restrict__ norm_conv_w,  // [hc*H]
    float* __restrict__ gated_out,             // [T, hc*H] FP32
    float* __restrict__ gated_normed,          // [T, hc*H] FP32 — see below
    const unsigned int hidden_size,            // H
    const unsigned int hc,                     // hc_count
    const float eps
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc_dim = hc * H;

    const float* q = hidden + (size_t)t * hc_dim;
    const __nv_bfloat16* k = key + (size_t)t * hc_dim;
    const __nv_bfloat16* v = value + (size_t)t * H;
    float* g_out = gated_out + (size_t)t * hc_dim;
    float* gn_out = gated_normed + (size_t)t * hc_dim;

    __shared__ float smem_red[PLE_BLOCK / 32];
    __shared__ float smem_gate[PLE_MAX_STREAMS];

    // Per-stream grouped RMS for query and key, then their dot product —
    // all three reductions are over the same 2560-wide slice, so they share
    // one pass per stream.
    for (unsigned int s = 0; s < hc; ++s) {
        const float* qs = q + (size_t)s * H;
        const __nv_bfloat16* ks = k + (size_t)s * H;
        float sq = 0.0f, sk = 0.0f;
        for (unsigned int d = tid; d < H; d += PLE_BLOCK) {
            float qv = qs[d];
            float kv = (float)ks[d];
            sq += qv * qv;
            sk += kv * kv;
        }
        float rq = ple_block_sum(sq, smem_red);
        float rk = ple_block_sum(sk, smem_red);
        rq = rsqrtf(rq / (float)H + eps);
        rk = rsqrtf(rk / (float)H + eps);

        // dot(norm_query(q), norm_key(k)) over this stream. The `1.0f +` is
        // the offset-from-1 convention — see the header.
        float dot = 0.0f;
        for (unsigned int d = tid; d < H; d += PLE_BLOCK) {
            const unsigned int i = s * H + d;
            float qn = qs[d] * rq * (1.0f + (float)norm_query_w[i]);
            float kn = (float)ks[d] * rk * (1.0f + (float)norm_key_w[i]);
            dot += qn * kn;
        }
        dot = ple_block_sum(dot, smem_red);
        if (tid == 0) {
            float gate = dot * rsqrtf((float)H);
            // SIGNED SQRT: sign(g) * sqrt(max(|g|, 1e-6)).
            float mag = fabsf(gate);
            mag = mag < 1e-6f ? 1e-6f : mag;
            float sgn = (gate > 0.0f) ? 1.0f : ((gate < 0.0f) ? -1.0f : 0.0f);
            smem_gate[s] = ple_sigmoid(sgn * sqrtf(mag));
        }
        __syncthreads();
    }

    // gated[s*H + d] = sigmoid_gate[s] * value[d]
    for (unsigned int i = tid; i < hc_dim; i += PLE_BLOCK) {
        float gv = smem_gate[i / H] * (float)v[i % H];
        g_out[i] = gv;
    }
    __syncthreads();

    // norm_conv over the gated value — grouped and offset-from-1 again.
    for (unsigned int s = 0; s < hc; ++s) {
        float acc = 0.0f;
        for (unsigned int d = tid; d < H; d += PLE_BLOCK) {
            float x = g_out[s * H + d];
            acc += x * x;
        }
        float r = ple_block_sum(acc, smem_red);
        r = rsqrtf(r / (float)H + eps);
        for (unsigned int d = tid; d < H; d += PLE_BLOCK) {
            const unsigned int i = s * H + d;
            float x = g_out[i];
            gn_out[i] = x * r * (1.0f + (float)norm_conv_w[i]);
        }
        __syncthreads();
    }
}

// ── ple_conv ──
// Depthwise causal conv1d, kernel K, DILATION D, over `num_tokens` new steps
// with `state_len = (K-1)*D` carried steps in front.
//
//   y[t][c] = sum_{j<K} w[c][j] * x[t - (K-1-j)*D][c]
//   out[t][c] = gated[t][c] + silu(y[t][c])
//
// `state` holds the last `state_len` rows of the PREVIOUS call's normed input
// (EOS/zero-filled at sequence start), and is updated in place to the last
// `state_len` rows of `state ++ x`. Handles prefill (T>1) and decode (T==1)
// with the same code — no separate decode twin to drift.
//
// One thread per channel per token. Channels are the fast axis, so the loads
// coalesce.
extern "C" __global__ void ple_conv(
    const float* __restrict__ x,               // [T, C] norm_conv'd, FP32
    const float* __restrict__ gated,           // [T, C] un-normalized, FP32
    const __nv_bfloat16* __restrict__ weight,  // [C, K] depthwise
    float* __restrict__ state,                 // [state_len, C] FP32 in/out
    float* __restrict__ out,                   // [T, C] FP32
    const unsigned int num_tokens,
    const unsigned int channels,               // C = hc*H
    const unsigned int k_size,                 // K
    const unsigned int dilation                // D
) {
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const unsigned int state_len = (k_size - 1) * dilation;

    for (unsigned int t = 0; t < num_tokens; ++t) {
        float acc = 0.0f;
        for (unsigned int j = 0; j < k_size; ++j) {
            // Offset back in time for tap j. j = K-1 is the current step.
            const int back = (int)((k_size - 1 - j) * dilation);
            const int src = (int)t - back;
            float xv;
            if (src >= 0) {
                xv = x[(size_t)src * channels + c];
            } else {
                // Into the carried state: index `state_len + src`, which is
                // >= 0 because back <= state_len.
                const int si = (int)state_len + src;
                xv = (si >= 0) ? state[(size_t)si * channels + c] : 0.0f;
            }
            acc += xv * (float)weight[(size_t)c * k_size + j];
        }
        float base = gated[(size_t)t * channels + c];
        out[(size_t)t * channels + c] = base + ple_silu(acc);
    }

    // Roll the state: the last `state_len` rows of `state ++ x`.
    // Read every value BEFORE writing, so a short `x` that overlaps the tail
    // of the old state cannot clobber its own source.
    float carry[16];
    const unsigned int keep = state_len;
    for (unsigned int i = 0; i < keep; ++i) {
        // Position `i` of the new state is global index
        // `num_tokens + state_len - keep + i` in `state ++ x`.
        const int gi = (int)(num_tokens + state_len - keep + i);
        const int xi = gi - (int)state_len;
        carry[i] = (xi >= 0) ? x[(size_t)xi * channels + c]
                             : state[(size_t)gi * channels + c];
    }
    for (unsigned int i = 0; i < keep; ++i) {
        state[(size_t)i * channels + c] = carry[i];
    }
}

// ── ple_add_highway ──
// `hidden_states = hidden_states + ple(...)`, straight into the FP32 highway
// (the reference adds BEFORE that layer's attention hyper-connection).
extern "C" __global__ void ple_add_highway(
    const float* __restrict__ ple_out,         // [T, C] FP32
    float* __restrict__ hidden,                // [T, C] FP32 highway, in/out
    const unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) hidden[i] += ple_out[i];
}
