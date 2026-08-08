// SPDX-License-Identifier: AGPL-3.0-only

// Register-resident twin of `gated_delta_rule_wy3` (K=3 MTP-verify GDN).
//
// Same lever as `gated_delta_rule_wy2_resident` (wave 10), generalized to
// K=3 for the 16:2 ladder rung (2 drafts = 3 rows/seq — the shape that
// measured +6% at C=16 once the verify chunk cap was fixed, 2026-07-30) and
// the 24:2/32:2 rungs the 96-row verify envelope opens. The base wy3 kernel
// reads the FP32 state H [k_dim x v_dim = 64KB/head] TWICE per verify:
// Pass 1 for the three hk dot products, Pass 2 re-reads H and writes Hi0 +
// Hi1 (rollback intermediates) + final H. Nothing writes H between the
// passes and each block owns its (b, vh) slice exclusively, so the Pass 2
// re-read returns exactly the Pass 1 bytes — pure redundant HBM traffic.
//
// This twin retains the Pass 1 read in registers (H_reg[128], one state
// column per thread — the proven `gated_delta_rule_prefill` regresident
// pattern, PR#369) and serves Pass 2 from them: 2R+3W -> 1R+3W, -20% of the
// kernel's HBM traffic. The Hi0/Hi1 and final-H writes are mandatory
// (rollback + state carry) and unchanged.
//
// `__launch_bounds__(128, 1)` forces minBlocksPerSM=1 so the compiler may
// allocate up to 512 registers/thread; without it H_reg spills to local
// memory and the point of the kernel is lost (same note as the wy2 twin and
// the prefill regresident kernel).
//
// BIT-EXACTNESS CONTRACT: every float expression, accumulation order and
// clamp is copied verbatim from `gated_delta_rule_wy3` — only the Pass 2
// H loads are replaced by the register copy of the same bits, so output,
// Hi0, Hi1 and final H are byte-identical to the base kernel (asserted
// bitwise by `gdn_wy_verify_microtest`'s wy3 resident-parity leg; do not
// edit one kernel without the other).
//
// DISPATCH CONTRACT: `H_reg` is indexed by the k-row loop, which must fully
// unroll for the array to live in registers, so k_dim is the compile-time
// WY3R_KD = 128 here; v_dim is likewise the compile-time WY3R_VD = 128 so
// every H/Hi access folds to an immediate LDG/STG offset. The Rust selector
// (`Qwen3SsmLayer::wy3_kernel`) only picks this kernel when
// k_dim == v_dim == 128 (the production GDN head shape) and the launch is
// wide enough to carry 1-block/SM occupancy (n >= wy_resident_min_width());
// base wy3 otherwise. Kill switch: ATLAS_NO_GDN_WY3_RESIDENT (presence).
//
// Pass 2 is deliberately SPLIT into three sequential per-token loops
// (token 0: Hi0 writes + qd0; token 1: Hi1 writes + qd1; token 2: H writes
// + qd2) instead of the base kernel's interleaved loop: it cuts the
// simultaneously-live accumulator chains / smem operands / write streams to
// a third, which is what keeps the spill footprint near the wy2 twin's. The
// float EXPRESSIONS and their accumulation order are unchanged — each later
// token consumes the previous token's updated H_reg values, which are the
// same bits the interleaved form produces mid-iteration.
//
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1)

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#define BLOCK_SIZE 128
#define WY3R_KD 128u
#define WY3R_VD 128u

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_wy3_resident(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_inter0,
    float* __restrict__ h_state_inter1,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    // Same contract as gated_delta_rule_wy3:
    // 0 = the three state args are CONTIGUOUS bases indexed by
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
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;

    const unsigned long long head_off = (unsigned long long)vh * hv;
    const unsigned long long flat_off = (unsigned long long)(b * num_v_heads + vh) * hv;
    float* H   = state_is_table ? ((float* const*)h_state)[b]        + head_off
                                : h_state        + flat_off;
    float* Hi0 = state_is_table ? ((float* const*)h_state_inter0)[b] + head_off
                                : h_state_inter0 + flat_off;
    float* Hi1 = state_is_table ? ((float* const*)h_state_inter1)[b] + head_off
                                : h_state_inter1 + flat_off;

    // Token pointers.
    // Gate clamp MUST match per-token gated_delta_rule_decode — verbatim
    // from the base wy3 (see gated_delta_rule_wy.cu for full rationale).
    #define TP(T) \
        const __nv_bfloat16* q##T = query + (b*3+T)*qk_stride + kh*k_dim; \
        const __nv_bfloat16* k##T = key   + (b*3+T)*qk_stride + kh*k_dim; \
        const __nv_bfloat16* v##T = value + (b*3+T)*v_stride  + vh*v_dim; \
        const float g##T = fminf(fmaxf(gate[(b*3+T)*gb_stride + vh], 1e-6f), 1.0f - 1e-6f); \
        const float bt##T = beta[(b*3+T)*gb_stride + vh];
    TP(0) TP(1) TP(2)
    #undef TP

    __shared__ float sk0[128], sq0[128], sk1[128], sq1[128], sk2[128], sq2[128];
    __shared__ float smem_warp[4];
    __shared__ float kd10, kd20, kd21;

    if (tid < k_dim) {
        sk0[tid]=(float)k0[tid]; sq0[tid]=(float)q0[tid];
        sk1[tid]=(float)k1[tid]; sq1[tid]=(float)q1[tid];
        sk2[tid]=(float)k2[tid]; sq2[tid]=(float)q2[tid];
    }
    __syncthreads();

    // ── Compute 3 k_dot products via block reduction (verbatim) ──
    {
        float p = (tid<k_dim) ? sk1[tid]*sk0[tid] : 0.0f;
        float r = atlas_block_reduce_sum(p, smem_warp, tid);
        if (tid==0) kd10 = r;
    }
    __syncthreads();
    {
        float p = (tid<k_dim) ? sk2[tid]*sk0[tid] : 0.0f;
        float r = atlas_block_reduce_sum(p, smem_warp, tid);
        if (tid==0) kd20 = r;
    }
    __syncthreads();
    {
        float p = (tid<k_dim) ? sk2[tid]*sk1[tid] : 0.0f;
        float r = atlas_block_reduce_sum(p, smem_warp, tid);
        if (tid==0) kd21 = r;
    }
    __syncthreads();

    if (tid < v_dim) {
        float vi0=(float)v0[tid], vi1=(float)v1[tid], vi2=(float)v2[tid];

        // Thread tid owns state column tid: H[j][tid] for j = 0..127.
        float H_reg[WY3R_KD];

        // ── PASS 1: read H ONCE from HBM into registers, compute all 3
        //    dot products (expressions verbatim from wy3) ──
        float hk0=0, hk1p=0, hk2p=0;
        #pragma unroll
        for (unsigned int j=0; j<WY3R_KD; j+=4) {
            float h0=H[(j+0)*WY3R_VD+tid], h1=H[(j+1)*WY3R_VD+tid];
            float h2=H[(j+2)*WY3R_VD+tid], h3=H[(j+3)*WY3R_VD+tid];
            H_reg[j+0]=h0; H_reg[j+1]=h1;
            H_reg[j+2]=h2; H_reg[j+3]=h3;
            hk0  += h0*sk0[j]+h1*sk0[j+1]+h2*sk0[j+2]+h3*sk0[j+3];
            hk1p += h0*sk1[j]+h1*sk1[j+1]+h2*sk1[j+2]+h3*sk1[j+3];
            hk2p += h0*sk2[j]+h1*sk2[j+1]+h2*sk2[j+2]+h3*sk2[j+3];
        }

        // ── WY Correction (verbatim) ──
        float vn0 = (vi0 - g0*hk0) * bt0;
        float hk1c = g0*hk1p + kd10*vn0;
        float vn1 = (vi1 - g1*hk1c) * bt1;
        float hk2c = g0*g1*hk2p + g1*kd20*vn0 + kd21*vn1;
        float vn2 = (vi2 - g2*hk2c) * bt2;

        // ── PASS 2a (token 0): served from registers (no H re-read),
        //    H_1 = g0*H + k0 ⊗ vn0 kept in H_reg, write Hi0
        //    (expressions verbatim from wy3's interleaved loop) ──
        float qd0=0, qd1=0, qd2=0;
        #pragma unroll
        for (unsigned int j=0; j<WY3R_KD; j+=4) {
            float h0=H_reg[j+0], h1=H_reg[j+1];
            float h2=H_reg[j+2], h3=H_reg[j+3];
            h0=g0*h0+sk0[j]*vn0; h1=g0*h1+sk0[j+1]*vn0;
            h2=g0*h2+sk0[j+2]*vn0; h3=g0*h3+sk0[j+3]*vn0;
            Hi0[(j+0)*WY3R_VD+tid]=h0; Hi0[(j+1)*WY3R_VD+tid]=h1;
            Hi0[(j+2)*WY3R_VD+tid]=h2; Hi0[(j+3)*WY3R_VD+tid]=h3;
            H_reg[j+0]=h0; H_reg[j+1]=h1;
            H_reg[j+2]=h2; H_reg[j+3]=h3;
            qd0 += h0*sq0[j]+h1*sq0[j+1]+h2*sq0[j+2]+h3*sq0[j+3];
        }

        // ── PASS 2b (token 1): H_2 = g1*H_1 + k1 ⊗ vn1, write Hi1 ──
        #pragma unroll
        for (unsigned int j=0; j<WY3R_KD; j+=4) {
            float h0=H_reg[j+0], h1=H_reg[j+1];
            float h2=H_reg[j+2], h3=H_reg[j+3];
            h0=g1*h0+sk1[j]*vn1; h1=g1*h1+sk1[j+1]*vn1;
            h2=g1*h2+sk1[j+2]*vn1; h3=g1*h3+sk1[j+3]*vn1;
            Hi1[(j+0)*WY3R_VD+tid]=h0; Hi1[(j+1)*WY3R_VD+tid]=h1;
            Hi1[(j+2)*WY3R_VD+tid]=h2; Hi1[(j+3)*WY3R_VD+tid]=h3;
            H_reg[j+0]=h0; H_reg[j+1]=h1;
            H_reg[j+2]=h2; H_reg[j+3]=h3;
            qd1 += h0*sq1[j]+h1*sq1[j+1]+h2*sq1[j+2]+h3*sq1[j+3];
        }

        // ── PASS 2c (token 2): H_3 = g2*H_2 + k2 ⊗ vn2, write final H ──
        #pragma unroll
        for (unsigned int j=0; j<WY3R_KD; j+=4) {
            float h0=H_reg[j+0], h1=H_reg[j+1];
            float h2=H_reg[j+2], h3=H_reg[j+3];
            h0=g2*h0+sk2[j]*vn2; h1=g2*h1+sk2[j+1]*vn2;
            h2=g2*h2+sk2[j+2]*vn2; h3=g2*h3+sk2[j+3]*vn2;
            H[(j+0)*WY3R_VD+tid]=h0; H[(j+1)*WY3R_VD+tid]=h1;
            H[(j+2)*WY3R_VD+tid]=h2; H[(j+3)*WY3R_VD+tid]=h3;
            qd2 += h0*sq2[j]+h1*sq2[j+1]+h2*sq2[j+2]+h3*sq2[j+3];
        }

        float s = rsqrtf((float)k_dim);
        output[(b*3*num_v_heads+vh)*v_dim+tid]     = __float2bfloat16(qd0*s);
        output[((b*3+1)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd1*s);
        output[((b*3+2)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd2*s);
    }
}
