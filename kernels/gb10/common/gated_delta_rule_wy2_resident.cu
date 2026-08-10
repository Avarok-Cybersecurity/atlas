// SPDX-License-Identifier: AGPL-3.0-only

// Register-resident twin of `gated_delta_rule_wy2` (K=2 MTP-verify GDN).
//
// The base wy2 kernel touches the FP32 state H [k_dim x v_dim = 64KB/head]
// FOUR times per verify: Pass 1 reads H (hk dot products), Pass 2 re-reads H
// and writes Hi0 (rollback intermediate) + final H. Nothing writes H between
// the passes and each block owns its (b, vh) slice exclusively, so the Pass 2
// re-read returns exactly the Pass 1 bytes — pure redundant HBM traffic. At
// C=32/K=2 this family is 89.5ms/step (nsys, wave-10 fixer r2), 2R+2W vs the
// plain decode kernel's 1R+1W.
//
// This twin retains the Pass 1 read in registers (H_reg[128], one state
// column per thread — the proven `gated_delta_rule_prefill` regresident
// pattern, PR#369) and serves Pass 2 from them: 2R+2W -> 1R+2W, -25% of the
// kernel's HBM traffic. The Hi0 and final-H writes are mandatory (rollback +
// state carry) and unchanged.
//
// `__launch_bounds__(128, 1)` forces minBlocksPerSM=1 so the compiler may
// allocate up to 512 registers/thread; without it H_reg spills to local
// memory and the point of the kernel is lost (same note as the prefill
// regresident kernel).
//
// BIT-EXACTNESS CONTRACT: every float expression, accumulation order and
// clamp is copied verbatim from `gated_delta_rule_wy2` — only the Pass 2
// H loads are replaced by the register copy of the same bits, so output,
// Hi0 and final H are byte-identical to the base kernel (asserted bitwise
// by `gdn_wy_verify_microtest`'s resident-parity leg; do not edit one
// kernel without the other).
//
// DISPATCH CONTRACT: `H_reg` is indexed by the k-row loop, which must fully
// unroll for the array to live in registers, so k_dim is the compile-time
// WY2R_KD = 128 here; v_dim is likewise the compile-time WY2R_VD = 128 so
// every H/H_inter access folds to an immediate LDG/STG offset (no per-row
// address-arithmetic registers — the 128-float array leaves only ~127 regs
// for everything else under the 255-reg ISA cap). The Rust selector
// (`Qwen3SsmLayer::wy2_kernel`) only picks this kernel when
// k_dim == v_dim == 128 (the production GDN head shape) and falls back to
// the base wy2 otherwise. Kill switch: ATLAS_NO_GDN_WY2_RESIDENT (presence).
//
// Pass 2 is deliberately SPLIT into two sequential per-token loops (token 0:
// H_inter writes + q0_dot; token 1: H writes + q1_dot) instead of the base
// kernel's interleaved loop: halves the simultaneously-live accumulator
// chains / smem operands / write streams, which is what keeps the spill
// footprint near the proven prefill-regresident kernel's. The float
// EXPRESSIONS and their accumulation order are unchanged — token 1 consumes
// the token-0-updated H_reg values, which are the same bits the interleaved
// form produces mid-iteration.
//
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1)

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#define BLOCK_SIZE 128
#define WY2R_KD 128u
#define WY2R_VD 128u

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_wy2_resident(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_intermediate,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    // Same contract as gated_delta_rule_wy2:
    // 0 = the two state args are CONTIGUOUS bases indexed by
    //     (b*num_v_heads+vh);
    // 1 = they are device POINTER TABLES of `batch_size` entries, one per
    //     sequence (the form the cross-sequence batched MTP verify needs —
    //     see the base kernel for the stride-corruption rationale).
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
    float* H = state_is_table ? ((float* const*)h_state)[b] + head_off
                              : h_state + flat_off;
    float* H_inter = state_is_table
                         ? ((float* const*)h_state_intermediate)[b] + head_off
                         : h_state_intermediate + flat_off;

    // Token pointers
    const __nv_bfloat16* q0 = query + (b * 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* k0 = key   + (b * 2) * qk_stride + kh * k_dim;
    const __nv_bfloat16* v0 = value + (b * 2) * v_stride  + vh * v_dim;
    // Gate clamp MUST match per-token gated_delta_rule_decode — verbatim
    // from the base wy2 (see its comment for the drift failure mode).
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
        float H_reg[WY2R_KD];

        // ── PASS 1: read H ONCE from HBM into registers, compute
        //    hk_prev[0] and hk_prev[1] (expressions verbatim from wy2) ──
        float hk0 = 0.0f, hk1_prev = 0.0f;
        #pragma unroll
        for (unsigned int j = 0; j < WY2R_KD; j += 4) {
            float h0 = H[(j+0) * WY2R_VD + tid];
            float h1 = H[(j+1) * WY2R_VD + tid];
            float h2 = H[(j+2) * WY2R_VD + tid];
            float h3 = H[(j+3) * WY2R_VD + tid];
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
        //    H_1 = g0*H + k0 ⊗ v_new_0 kept in H_reg, write Hi0
        //    (expressions verbatim from wy2's interleaved loop) ──
        float q0_dot = 0.0f, q1_dot = 0.0f;
        #pragma unroll
        for (unsigned int j = 0; j < WY2R_KD; j += 4) {
            float h0 = H_reg[j+0];
            float h1 = H_reg[j+1];
            float h2 = H_reg[j+2];
            float h3 = H_reg[j+3];
            h0 = g0*h0 + smem_k0[j]  *v_new_0;
            h1 = g0*h1 + smem_k0[j+1]*v_new_0;
            h2 = g0*h2 + smem_k0[j+2]*v_new_0;
            h3 = g0*h3 + smem_k0[j+3]*v_new_0;
            H_inter[(j+0)*WY2R_VD+tid]=h0; H_inter[(j+1)*WY2R_VD+tid]=h1;
            H_inter[(j+2)*WY2R_VD+tid]=h2; H_inter[(j+3)*WY2R_VD+tid]=h3;
            H_reg[j+0] = h0; H_reg[j+1] = h1;
            H_reg[j+2] = h2; H_reg[j+3] = h3;
            q0_dot += h0*smem_q0[j] + h1*smem_q0[j+1] + h2*smem_q0[j+2] + h3*smem_q0[j+3];
        }

        // ── PASS 2b (token 1): H_2 = g1*H_1 + k1 ⊗ v_new_1, write final H ──
        #pragma unroll
        for (unsigned int j = 0; j < WY2R_KD; j += 4) {
            float h0 = H_reg[j+0];
            float h1 = H_reg[j+1];
            float h2 = H_reg[j+2];
            float h3 = H_reg[j+3];
            h0 = g1*h0 + smem_k1[j]  *v_new_1;
            h1 = g1*h1 + smem_k1[j+1]*v_new_1;
            h2 = g1*h2 + smem_k1[j+2]*v_new_1;
            h3 = g1*h3 + smem_k1[j+3]*v_new_1;
            H[(j+0)*WY2R_VD+tid]=h0; H[(j+1)*WY2R_VD+tid]=h1;
            H[(j+2)*WY2R_VD+tid]=h2; H[(j+3)*WY2R_VD+tid]=h3;
            q1_dot += h0*smem_q1[j] + h1*smem_q1[j+1] + h2*smem_q1[j+2] + h3*smem_q1[j+3];
        }

        float inv_sqrt_d = rsqrtf((float)k_dim);
        unsigned int out0 = (b * 2 * num_v_heads + vh) * v_dim;
        unsigned int out1 = ((b * 2 + 1) * num_v_heads + vh) * v_dim;
        output[out0 + tid] = __float2bfloat16(q0_dot * inv_sqrt_d);
        output[out1 + tid] = __float2bfloat16(q1_dot * inv_sqrt_d);
    }
}
