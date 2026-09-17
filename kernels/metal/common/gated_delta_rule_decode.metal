// SPDX-License-Identifier: AGPL-3.0-only
//
// Atlas Gated Delta Rule decode kernel — MSL port of
// `kernels/gb10/common/gated_delta_rule.cu::gated_delta_rule_decode`.
// Same math, same arg layout, same kernel name so spark-model's
// existing `ops::gdn_decode` dispatches through the metal backend
// without any orchestration-side changes.
//
// State equation (per (batch, value_head) pair, k_dim = v_dim = 128):
//
//   hk_dot[c]   = sum_j H[j, c] * k[j]                  (matvec)
//   v_new[c]    = (v[c] - g * hk_dot[c]) * beta          (delta-rule residual)
//   H[j, c]     = g * H[j, c] + k[j] * v_new[c]          (state update)
//   y[c]        = (sum_j H[j, c] * q[j]) * 1/sqrt(k_dim) (output dot)
//
// State H[batch, num_v_heads, k_dim, v_dim] is FP32; v_dim is the
// fast (contiguous) dimension. Threads map to v_dim → coalesced
// reads/writes. 128 threads/threadgroup = 4 simdgroups on Apple
// Silicon (simd_size = 32 like CUDA warps).
//
// Layout:
//   h_state : float  [batch, num_v_heads, k_dim, v_dim]   (in/out)
//   query   : bfloat [batch, num_k_heads,   k_dim]
//   key     : bfloat [batch, num_k_heads,   k_dim]
//   value   : bfloat [batch, num_v_heads,   v_dim]
//   gate    : float  [batch, num_v_heads]                 (exp(g_t) decay)
//   beta    : float  [batch, num_v_heads]                 (sigmoid(b_t))
//   output  : bfloat [batch, num_v_heads,   v_dim]

#include <metal_stdlib>
using namespace metal;

// SSM state-norm clamp (Stuffed Mamba mitigation). Same value as
// CUDA reference: 1000.0 lets the state grow naturally for long
// contexts but caps it before FP32 overflow.
constant float SSM_STATE_MAX_NORM = 1000.0f;

kernel void gated_delta_rule_decode(
    device float        *h_state    [[buffer(0)]],
    device const bfloat *query      [[buffer(1)]],
    device const bfloat *key        [[buffer(2)]],
    device const bfloat *value      [[buffer(3)]],
    device const float  *gate       [[buffer(4)]],
    device const float  *beta       [[buffer(5)]],
    device bfloat       *output     [[buffer(6)]],
    constant uint &batch_size       [[buffer(7)]],
    constant uint &num_k_heads      [[buffer(8)]],
    constant uint &num_v_heads      [[buffer(9)]],
    constant uint &k_dim            [[buffer(10)]],
    constant uint &v_dim            [[buffer(11)]],
    uint  tg_idx    [[threadgroup_position_in_grid]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_grp  [[simdgroup_index_in_threadgroup]])
{
    // Flat 1-D grid: caller dispatches `num_v_heads * batch_size`
    // threadgroups; we decode (vh, b) here. Metal forbids mixing
    // scalar and uint3 position attributes in one entry point.
    const uint vh = tg_idx % num_v_heads;
    const uint b  = tg_idx / num_v_heads;
    if (vh >= num_v_heads || b >= batch_size) {
        return;
    }
    const uint head_repeat = num_v_heads / num_k_heads;
    const uint kh = vh / head_repeat;

    // H slice: [k_dim, v_dim] for this (batch, value_head)
    device float *H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    device const bfloat *q_ptr = query + (b * num_k_heads + kh) * k_dim;
    device const bfloat *k_ptr = key   + (b * num_k_heads + kh) * k_dim;
    device const bfloat *v_ptr = value + (b * num_v_heads + vh) * v_dim;

    // Gate decay clamped to (0, 1) — same numeric guard as CUDA ref.
    float g_raw = gate[b * num_v_heads + vh];
    const float g  = fmin(fmax(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[b * num_v_heads + vh];

    // Shared K and Q vectors (k_dim ≤ 128 here; sized for the
    // Qwen3.5 GDN where k_dim = v_dim = 128).
    threadgroup float smem_k[128];
    threadgroup float smem_q[128];
    if (tid < k_dim) {
        smem_k[tid] = float(k_ptr[tid]);
        smem_q[tid] = float(q_ptr[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= v_dim) {
        return;
    }
    float v_i = float(v_ptr[tid]);

    // Step 1: hk_dot = sum_j H[j, tid] * k[j]
    float hk_dot = 0.0f;
    for (uint j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        hk_dot += h0 * smem_k[j] + h1 * smem_k[j + 1]
                + h2 * smem_k[j + 2] + h3 * smem_k[j + 3];
    }

    // Step 2: gated residual value (HF reference order — decay applied
    // before the correction term, then beta scales the residual).
    float v_new_i = (v_i - g * hk_dot) * bt;

    // Steps 3+4 fused: state update + output dot product in one pass.
    float q_dot = 0.0f;
    for (uint j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        h0 = g * h0 + smem_k[j]     * v_new_i;
        h1 = g * h1 + smem_k[j + 1] * v_new_i;
        h2 = g * h2 + smem_k[j + 2] * v_new_i;
        h3 = g * h3 + smem_k[j + 3] * v_new_i;
        H[(j + 0) * v_dim + tid] = h0;
        H[(j + 1) * v_dim + tid] = h1;
        H[(j + 2) * v_dim + tid] = h2;
        H[(j + 3) * v_dim + tid] = h3;
        q_dot += h0 * smem_q[j] + h1 * smem_q[j + 1]
               + h2 * smem_q[j + 2] + h3 * smem_q[j + 3];
    }

    // ── SSM state-norm clamp (Stuffed Mamba mitigation) ──
    // Per-thread sum of squares over k_dim rows for this v_dim
    // column → block-wide reduction → if ||H||_F > MAX, scale down.
    {
        float local_sq = 0.0f;
        for (uint j = 0; j < k_dim; ++j) {
            float hv = H[j * v_dim + tid];
            local_sq += hv * hv;
        }
        // simdgroup (32-lane) sum, then cross-simdgroup reduction.
        float warp_sum = simd_sum(local_sq);
        threadgroup float norm_sums[4];
        if (simd_lane == 0) {
            norm_sums[simd_grp] = warp_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float head_norm_sq_storage;
        if (simd_grp == 0) {
            // 4 simdgroups for tg_size = 128.
            float s = (tid < 4u) ? norm_sums[tid] : 0.0f;
            s = simd_sum(s);
            if (tid == 0) {
                head_norm_sq_storage = s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float head_norm_sq = head_norm_sq_storage;
        if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
            float scale = SSM_STATE_MAX_NORM * rsqrt(head_norm_sq);
            for (uint j = 0; j < k_dim; ++j) {
                H[j * v_dim + tid] *= scale;
            }
        }
    }

    // Output scaled by 1/sqrt(k_dim) — matches CUDA + HF reference.
    float inv_sqrt_d = rsqrt(float(k_dim));
    output[(b * num_v_heads + vh) * v_dim + tid] = bfloat(q_dot * inv_sqrt_d);
}
