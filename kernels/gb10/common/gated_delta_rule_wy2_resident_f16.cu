// SPDX-License-Identifier: AGPL-3.0-only

// FP16 h-state twin of `gated_delta_rule_wy2_resident` (K=2 MTP-verify GDN,
// register-resident Pass 2). Stage 2 of `ATLAS_SSM_H_FP16`.
//
// This is the kernel the C=32 rung runs: the default ladder is
// `4:3,8:3,16:2,32:1`, so a width in 17..32 verifies 1 draft = K=2 rows, and
// `wy_resident_min_width()` is 16, so the resident twin — not the base wy2 —
// is dispatched. With `--speculative` on it is therefore the ONLY GDN h-state
// reader/writer in the decode step, which is why stage 1's FP16 lever could not
// reach the rung at all: preflight refused the flag together with speculation.
//
// STORAGE-ONLY NARROWING (see gdn_f16_state.cuh). Every float expression,
// accumulation order and gate clamp below is copied verbatim from
// `gated_delta_rule_wy2_resident`, which in turn copied them verbatim from the
// base `gated_delta_rule_wy2`. The ONLY differences are that H and H_inter are
// `__half` in memory, and that the state is round-tripped through FP16 at each
// token boundary (see the next paragraph). Arithmetic remains FP32 in
// registers, and `H_reg` stays `float[128]` exactly as in the FP32 parent — the
// narrowing buys HBM traffic, not registers.
//
// ★ PER-TOKEN ROUND-TRIP, and why it is not optional. The FP32 parent carries
// the token-0 state forward in `H_reg` with the SAME bits it writes to H_inter,
// because there is only one dtype. Under FP16 storage those could differ: the
// checkpoint is rounded, the register is not. They must not. `H_inter[0]` is
// the state a rollback restores when exactly 1 draft is accepted, and the
// carried register is the state used when 2 are; if they disagree, "verify K
// tokens then accept n" stops being equal to "decode n tokens", which is the
// invariant the whole MTP rollback design rests on. So token 0's update is
// stored as `__half` and read BACK as the value token 1 consumes — the state is
// FP16, and every reader of it sees the same FP16 bits. This also makes the
// twin agree with stage 1's decode scan, which rounds to FP16 once per step for
// the same reason. `q0_dot` is likewise accumulated from the rounded value: the
// output projection reads the state that exists, not a wider ghost of it.
//
// TRAFFIC. The parent is 1R+2W of a 64KB/head FP32 state (Pass 1 read retained
// in registers, then Hi0 + final H written). Halving the element width makes it
// 1R+2W of 32KB/head. The register residency lever and the FP16 lever therefore
// COMPOSE — residency removes a redundant pass, FP16 halves every remaining
// one.
//
// `__launch_bounds__(128, 1)` is mandatory and unchanged: it forces
// minBlocksPerSM=1 so ptxas may allocate beyond 255 registers for `H_reg[128]`.
// Without it the array spills to local memory and the kernel is pointless.
// Verify with `ptxas -v`: this twin must report 0 bytes stack frame and 0
// spill stores/loads, exactly like its parent.
//
// DISPATCH CONTRACT: identical to the parent — compile-time WY2R_KD/WY2R_VD of
// 128 so the k-row loops fully unroll and every H access folds to an immediate
// offset; the Rust selector only picks it at k_dim == v_dim == 128. Under FP16
// the contiguous (`state_is_table == 0`) form is only ever launched at
// batch_size == 1, because consecutive slots are `h_state_bytes` apart — TWICE
// the dense FP16 footprint — so the parent's `(b*num_v_heads+vh)` flat indexing
// would walk into the wrong slot. The Rust launcher refuses that combination.
//
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1)

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#include "gdn_f16_state.cuh"
#define BLOCK_SIZE 128
#define WY2RF_KD 128u
#define WY2RF_VD 128u

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_wy2_resident_f16(
    __half* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    __half* __restrict__ h_state_intermediate,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    // Same contract as the FP32 parent:
    // 0 = the two state args are CONTIGUOUS bases indexed by
    //     (b*num_v_heads+vh) — FP16 permits this only at batch_size == 1;
    // 1 = they are device POINTER TABLES of `batch_size` entries, one per
    //     sequence (the form the cross-sequence batched MTP verify needs).
    unsigned int state_is_table
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    const unsigned int hv_size = k_dim * v_dim;
    const unsigned long long head_off = (unsigned long long)vh * hv_size;
    const unsigned long long flat_off =
        (unsigned long long)(b * num_v_heads + vh) * hv_size;
    __half* H = state_is_table ? ((__half* const*)h_state)[b] + head_off
                               : h_state + flat_off;
    __half* H_inter = state_is_table
                          ? ((__half* const*)h_state_intermediate)[b] + head_off
                          : h_state_intermediate + flat_off;

    // Token pointers
    const __nv_bfloat16* q0 = query + (b * 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k0 = key   + (b * 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v0 = value + (b * 2) * v_stride  + vh * v_dim;
    // Gate clamp MUST match per-token gated_delta_rule_decode — verbatim
    // from the FP32 parent (see base wy2 for the drift failure mode).
    const float g0 = fminf(fmaxf(gate[(b * 2) * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
    const float bt0 = beta[(b * 2) * gb_stride + vh];

    const __nv_bfloat16* q1 = query + (b * 2 + 1) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k1 = key   + (b * 2 + 1) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v1 = value + (b * 2 + 1) * v_stride  + vh * v_dim;
    const float g1 = fminf(fmaxf(gate[(b * 2 + 1) * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
    const float bt1 = beta[(b * 2 + 1) * gb_stride + vh];

    __shared__ float smem_k0[128], smem_q0[128];
    __shared__ float smem_k1[128], smem_q1[128];
    __shared__ float smem_kdot;
    __shared__ float smem_warp[4];

    if (tid < k_dim) {
        smem_k0[tid] = (float)k0[tid]; smem_q0[tid] = (float)q0[tid];
        smem_k1[tid] = (float)k1[tid]; smem_q1[tid] = (float)q1[tid];
    }
    __syncthreads();

    // ── Compute kdot = k_1^T @ k_0 ──
    {
        float partial = (tid < k_dim) ? smem_k1[tid] * smem_k0[tid] : 0.0f;
        float result = atlas_block_reduce_sum(partial, smem_warp, tid);
        if (tid == 0) smem_kdot = result;
    }
    __syncthreads();

    if (tid < v_dim) {
        float vi0 = (float)v0[tid];
        float vi1 = (float)v1[tid];
        float kdot_10 = smem_kdot;

        // Thread tid owns state column tid: H[j][tid] for j = 0..127.
        float H_reg[WY2RF_KD];

        // ── PASS 1: read H ONCE from HBM into registers (widening on the
        //    way in), compute hk_prev[0] and hk_prev[1] ──
        float hk0 = 0.0f, hk1_prev = 0.0f;
        #pragma unroll
        for (unsigned int j = 0; j < WY2RF_KD; j += 4) {
            float h0 = __half2float(H[(j+0) * WY2RF_VD + tid]);
            float h1 = __half2float(H[(j+1) * WY2RF_VD + tid]);
            float h2 = __half2float(H[(j+2) * WY2RF_VD + tid]);
            float h3 = __half2float(H[(j+3) * WY2RF_VD + tid]);
            H_reg[j+0] = h0; H_reg[j+1] = h1;
            H_reg[j+2] = h2; H_reg[j+3] = h3;
            hk0      += h0*smem_k0[j] + h1*smem_k0[j+1] + h2*smem_k0[j+2] + h3*smem_k0[j+3];
            hk1_prev += h0*smem_k1[j] + h1*smem_k1[j+1] + h2*smem_k1[j+2] + h3*smem_k1[j+3];
        }

        // ── WY Correction (verbatim) ──
        float v_new_0 = (vi0 - g0 * hk0) * bt0;
        float hk1_corr = g0 * hk1_prev + kdot_10 * v_new_0;
        float v_new_1 = (vi1 - g1 * hk1_corr) * bt1;

        // ── PASS 2a (token 0): served from registers (no H re-read),
        //    H_1 = g0*H + k0 ⊗ v_new_0, written to Hi0 as FP16 and read BACK
        //    so token 1 and q0_dot consume the same FP16 bits a rollback
        //    would restore (see the round-trip note in the header) ──
        float q0_dot = 0.0f, q1_dot = 0.0f;
        #pragma unroll
        for (unsigned int j = 0; j < WY2RF_KD; j += 4) {
            float h0 = H_reg[j+0];
            float h1 = H_reg[j+1];
            float h2 = H_reg[j+2];
            float h3 = H_reg[j+3];
            h0 = g0*h0 + smem_k0[j]  *v_new_0;
            h1 = g0*h1 + smem_k0[j+1]*v_new_0;
            h2 = g0*h2 + smem_k0[j+2]*v_new_0;
            h3 = g0*h3 + smem_k0[j+3]*v_new_0;
            __half s0 = gdn_f16_store(h0);
            __half s1 = gdn_f16_store(h1);
            __half s2 = gdn_f16_store(h2);
            __half s3 = gdn_f16_store(h3);
            H_inter[(j+0)*WY2RF_VD+tid]=s0; H_inter[(j+1)*WY2RF_VD+tid]=s1;
            H_inter[(j+2)*WY2RF_VD+tid]=s2; H_inter[(j+3)*WY2RF_VD+tid]=s3;
            h0 = __half2float(s0); h1 = __half2float(s1);
            h2 = __half2float(s2); h3 = __half2float(s3);
            H_reg[j+0] = h0; H_reg[j+1] = h1;
            H_reg[j+2] = h2; H_reg[j+3] = h3;
            q0_dot += h0*smem_q0[j] + h1*smem_q0[j+1] + h2*smem_q0[j+2] + h3*smem_q0[j+3];
        }

        // ── PASS 2b (token 1): H_2 = g1*H_1 + k1 ⊗ v_new_1, write final H ──
        #pragma unroll
        for (unsigned int j = 0; j < WY2RF_KD; j += 4) {
            float h0 = H_reg[j+0];
            float h1 = H_reg[j+1];
            float h2 = H_reg[j+2];
            float h3 = H_reg[j+3];
            h0 = g1*h0 + smem_k1[j]  *v_new_1;
            h1 = g1*h1 + smem_k1[j+1]*v_new_1;
            h2 = g1*h2 + smem_k1[j+2]*v_new_1;
            h3 = g1*h3 + smem_k1[j+3]*v_new_1;
            __half s0 = gdn_f16_store(h0);
            __half s1 = gdn_f16_store(h1);
            __half s2 = gdn_f16_store(h2);
            __half s3 = gdn_f16_store(h3);
            H[(j+0)*WY2RF_VD+tid]=s0; H[(j+1)*WY2RF_VD+tid]=s1;
            H[(j+2)*WY2RF_VD+tid]=s2; H[(j+3)*WY2RF_VD+tid]=s3;
            h0 = __half2float(s0); h1 = __half2float(s1);
            h2 = __half2float(s2); h3 = __half2float(s3);
            q1_dot += h0*smem_q1[j] + h1*smem_q1[j+1] + h2*smem_q1[j+2] + h3*smem_q1[j+3];
        }

        float inv_sqrt_d = rsqrtf((float)k_dim);
        unsigned int out0 = (b * 2 * num_v_heads + vh) * v_dim;
        unsigned int out1 = ((b * 2 + 1) * num_v_heads + vh) * v_dim;
        output[out0 + tid] = __float2bfloat16(q0_dot * inv_sqrt_d);
        output[out1 + tid] = __float2bfloat16(q1_dot * inv_sqrt_d);
    }
}
