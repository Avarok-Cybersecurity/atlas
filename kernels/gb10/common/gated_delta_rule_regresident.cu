// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Gated Delta Rule — REGISTER-RESIDENT sequential prefill recurrence.
//
// Drop-in alternative to `gated_delta_rule_prefill_persistent_wy4` for the
// WARM Marconi-replay path (short suffix after a restored snapshot). Same
// token-sequential FP32 math as `gated_delta_rule_decode` (the exact-replay
// reference class), but the 128x128 H state lives in REGISTERS instead of
// 64KB shared memory — modeled on llama.cpp's `gated_delta_net_cuda`.
//
// WHY: WY4 keeps H in 64KB smem => launch_bounds(128,1) = 1 CTA/SM, 4 warps,
// 6 __syncthreads per 4-token iter. nsys (graphs ON) measured this ~5.5x
// slower per warm turn than llama's register-resident kernel. Keeping H in
// registers removes the smem occupancy cap (>=2 CTA/SM) and the per-token
// barriers (warp-private recurrence, intra-warp shuffle reductions only).
//
// LAYOUT (k_dim=v_dim=128): one WARP owns one v-column `col`; its 32 lanes
// split the 128 k-rows, lane l holding rows {l, l+32, l+64, l+96} as s[0..3].
// State is read from global ONCE at start, written ONCE at end; the whole
// token loop runs register-resident with no smem-H and no block barriers.
//
//   kv   = sum_i H[i][col] * k[i]            (warp all-reduce over the 4 rows/lane)
//   vnew = (v[col] - g*kv) * beta
//   H[i][col] = g*H[i][col] + k[i]*vnew      (per-lane, on its 4 rows)
//   out[col]  = (sum_i H[i][col] * q[i]) * rsqrt(k_dim)
//
// Token-equal (not bit-identical) to the scalar reference — same acceptance
// class as WY4 (warp-reduce summation order differs from the per-thread
// sequential sum, ~1e-6, exactly the tolerance WY4 already operates under).
//
// NOTE: the per-token Stuffed-Mamba H-norm clamp (SSM_STATE_NORM) is OMITTED:
// it needs the whole-head Frobenius norm, but this layout splits the head's
// 128 columns across grid.z blocks. The warm-replay suffix is short and starts
// from an already-norm-controlled restored state, so the clamp does not trigger
// (validated by cos=1.0 vs the clamped scalar reference). llama omits it too.

#include <cuda_bf16.h>

#ifndef GDN_RR_WARPS_PER_BLOCK
#define GDN_RR_WARPS_PER_BLOCK 4
#endif

// Grid:  (num_v_heads, batch, v_dim / GDN_RR_WARPS_PER_BLOCK)
// Block: (32 * GDN_RR_WARPS_PER_BLOCK, 1, 1)   — one warp per v-column.
// Requires k_dim == 128 and v_dim divisible by GDN_RR_WARPS_PER_BLOCK.
extern "C" __global__ void __launch_bounds__(32 * GDN_RR_WARPS_PER_BLOCK, 4)
gated_delta_rule_prefill_regresident(
    float* __restrict__ h_state,               // [batch, num_v_heads, k_dim, v_dim] FP32
    const __nv_bfloat16* __restrict__ query,   // Q[t] at: query + t*qk_stride + kh*k_dim
    const __nv_bfloat16* __restrict__ key,     // K[t] at: key   + t*qk_stride + kh*k_dim
    const __nv_bfloat16* __restrict__ value,   // V[t] at: value + t*v_stride  + vh*v_dim
    const float* __restrict__ gate,            // g[t] at: gate + t*gb_stride + vh
    const float* __restrict__ beta,            // b[t] at: beta + t*gb_stride + vh
    __nv_bfloat16* __restrict__ output,        // [batch, seq_len, num_v_heads, v_dim]
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,                    // BF16 elements between consecutive Q/K tokens
    unsigned int v_stride,                     // BF16 elements between consecutive V tokens
    unsigned int gb_stride                     // FP32 elements between consecutive gate/beta tokens
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b  = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int warp_id = threadIdx.x >> 5;          // 0..WARPS_PER_BLOCK-1
    const unsigned int lane    = threadIdx.x & 31;          // 0..31
    const unsigned int col     = blockIdx.z * GDN_RR_WARPS_PER_BLOCK + warp_id;
    if (col >= v_dim) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    // k_dim == 128 => 4 rows per lane: {lane, lane+32, lane+64, lane+96}.
    const unsigned int r0 = lane, r1 = lane + 32, r2 = lane + 64, r3 = lane + 96;

    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh) * k_dim * v_dim);

    // Load this lane's 4 H rows for column `col` into registers — ONCE.
    float s0 = H[r0 * v_dim + col];
    float s1 = H[r1 * v_dim + col];
    float s2 = H[r2 * v_dim + col];
    float s3 = H[r3 * v_dim + col];

    const float inv_sqrt_d = rsqrtf((float)k_dim);

    for (unsigned int t = 0; t < seq_len; t++) {
        const __nv_bfloat16* k_t = key   + (unsigned long long)t * qk_stride + kh * k_dim;
        const __nv_bfloat16* q_t = query + (unsigned long long)t * qk_stride + kh * k_dim;

        // This lane's 4 K/Q values (direct global read; 4 warps share kh —
        // tiny redundant bf16 traffic, avoids a per-token block barrier).
        float k0 = (float)k_t[r0], k1 = (float)k_t[r1], k2 = (float)k_t[r2], k3 = (float)k_t[r3];
        float q0 = (float)q_t[r0], q1 = (float)q_t[r1], q2 = (float)q_t[r2], q3 = (float)q_t[r3];

        float g_raw = gate[(unsigned long long)t * gb_stride + vh];
        const float g  = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);   // match decode clamp
        const float bt = beta[(unsigned long long)t * gb_stride + vh];
        float v_col = (float)value[(unsigned long long)t * v_stride + vh * v_dim + col];

        // kv = sum_i H[i][col]*k[i] — partial over this lane's 4 rows, then warp all-reduce.
        float kv = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) kv += __shfl_xor_sync(0xFFFFFFFFu, kv, off);

        float v_new = (v_col - g * kv) * bt;

        // State update (per-lane on its 4 rows) fused with q_dot accumulation.
        s0 = g * s0 + k0 * v_new;
        s1 = g * s1 + k1 * v_new;
        s2 = g * s2 + k2 * v_new;
        s3 = g * s3 + k3 * v_new;

        float q_dot = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) q_dot += __shfl_xor_sync(0xFFFFFFFFu, q_dot, off);

        if (lane == 0) {
            output[((unsigned long long)(b * seq_len + t) * num_v_heads + vh) * v_dim + col] =
                __float2bfloat16(q_dot * inv_sqrt_d);
        }
    }

    // Write the final H column back to global — ONCE.
    H[r0 * v_dim + col] = s0;
    H[r1 * v_dim + col] = s1;
    H[r2 * v_dim + col] = s2;
    H[r3 * v_dim + col] = s3;
}
