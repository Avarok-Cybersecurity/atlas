// SPDX-License-Identifier: AGPL-3.0-only

// FP16 h-state twin of `gated_delta_rule_wy3_resident` (K=3 MTP-verify GDN,
// register-resident Pass 2). Stage 2 of `ATLAS_SSM_H_FP16`.
//
// This is the kernel the C=16 rung runs: the default ladder `4:3,8:3,16:2,32:1`
// verifies 2 drafts = K=3 rows at widths 9..16, and `wy_resident_min_width()`
// is 16, so at exactly n=16 the resident twin is dispatched. C=16 is the one
// rung Atlas already wins (180.11 vs a ~174 bar), so this twin's job is to not
// give that back while the flag is on — the no-regression gate.
//
// STORAGE-ONLY NARROWING and the PER-TOKEN ROUND-TRIP rule are identical to
// `gated_delta_rule_wy2_resident_f16.cu`; see that file's header and
// gdn_f16_state.cuh for the full rationale. In short: every float expression
// and accumulation order below is verbatim from the FP32 parent, arithmetic
// stays FP32 in registers (`H_reg` is still `float[128]`), and the ONLY change
// is that H/Hi0/Hi1 are `__half` in memory with the state rounded once per
// token boundary — so the two rollback checkpoints hold exactly the bits the
// forward chain carried, and "verify K then accept n" still equals "decode n".
//
// K=3 writes TWO intermediates plus the final state, so the parent is 1R+3W of
// a 64KB/head FP32 state; the twin makes that 1R+3W of 32KB/head. The write
// side dominates here, which is precisely where halving the element width pays
// most — residency already removed the redundant read.
//
// `__launch_bounds__(128, 1)` is mandatory (H_reg[128] must not spill to local
// memory). Note the FP32 parent itself reports 168 bytes of stack frame at
// sm_121a; this twin is expected to report LESS, not zero, because FP16 loads
// need fewer address registers. Verify with `ptxas -v` and compare against the
// parent rather than against zero.
//
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1)

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#include "gdn_f16_state.cuh"
#define BLOCK_SIZE 128
#define WY3RF_KD 128u
#define WY3RF_VD 128u

// Each Pass-2 loop converts its four updated elements to FP16 FIRST, stores
// them, and only then widens them back into the registers that carry forward.
// Interleaving the convert/store/reload per element instead (the obvious macro
// form) lengthens every live range across the store and cost 880 bytes of spill
// at sm_121a versus 168 for the FP32 parent; batching the four keeps it near
// the parent. Structure matters here, not just arithmetic.

extern "C" __global__ void __launch_bounds__(128, 1)
gated_delta_rule_wy3_resident_f16(
    __half* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    __half* __restrict__ h_state_inter0,
    __half* __restrict__ h_state_inter1,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    // Same contract as the FP32 parent:
    // 0 = the three state args are CONTIGUOUS bases indexed by
    //     (b*num_v_heads+vh) — FP16 permits this only at batch_size == 1;
    // 1 = they are device POINTER TABLES of `batch_size` entries, one per
    //     sequence (the cross-sequence batched MTP verify form).
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
    __half* H   = state_is_table ? ((__half* const*)h_state)[b]        + head_off
                                 : h_state        + flat_off;
    __half* Hi0 = state_is_table ? ((__half* const*)h_state_inter0)[b] + head_off
                                 : h_state_inter0 + flat_off;
    __half* Hi1 = state_is_table ? ((__half* const*)h_state_inter1)[b] + head_off
                                 : h_state_inter1 + flat_off;

    // Token pointers.
    // Gate clamp MUST match per-token gated_delta_rule_decode — verbatim
    // from the FP32 parent (see gated_delta_rule_wy.cu for full rationale).
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
        float H_reg[WY3RF_KD];

        // ── PASS 1: read H ONCE from HBM into registers (widening on the
        //    way in), compute all 3 dot products ──
        float hk0=0, hk1p=0, hk2p=0;
        #pragma unroll
        for (unsigned int j=0; j<WY3RF_KD; j+=4) {
            float h0=__half2float(H[(j+0)*WY3RF_VD+tid]);
            float h1=__half2float(H[(j+1)*WY3RF_VD+tid]);
            float h2=__half2float(H[(j+2)*WY3RF_VD+tid]);
            float h3=__half2float(H[(j+3)*WY3RF_VD+tid]);
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

        // ── PASS 2a (token 0): served from registers, write Hi0 ──
        float qd0=0, qd1=0, qd2=0;
        #pragma unroll
        for (unsigned int j=0; j<WY3RF_KD; j+=4) {
            float h0=H_reg[j+0], h1=H_reg[j+1];
            float h2=H_reg[j+2], h3=H_reg[j+3];
            h0=g0*h0+sk0[j]*vn0; h1=g0*h1+sk0[j+1]*vn0;
            h2=g0*h2+sk0[j+2]*vn0; h3=g0*h3+sk0[j+3]*vn0;
            __half s0 = gdn_f16_store(h0);
            __half s1 = gdn_f16_store(h1);
            __half s2 = gdn_f16_store(h2);
            __half s3 = gdn_f16_store(h3);
            Hi0[(j+0)*WY3RF_VD+tid]=s0; Hi0[(j+1)*WY3RF_VD+tid]=s1;
            Hi0[(j+2)*WY3RF_VD+tid]=s2; Hi0[(j+3)*WY3RF_VD+tid]=s3;
            h0=__half2float(s0); h1=__half2float(s1);
            h2=__half2float(s2); h3=__half2float(s3);
            H_reg[j+0]=h0; H_reg[j+1]=h1;
            H_reg[j+2]=h2; H_reg[j+3]=h3;
            qd0 += h0*sq0[j]+h1*sq0[j+1]+h2*sq0[j+2]+h3*sq0[j+3];
        }

        // ── PASS 2b (token 1): H_2 = g1*H_1 + k1 ⊗ vn1, write Hi1 ──
        #pragma unroll
        for (unsigned int j=0; j<WY3RF_KD; j+=4) {
            float h0=H_reg[j+0], h1=H_reg[j+1];
            float h2=H_reg[j+2], h3=H_reg[j+3];
            h0=g1*h0+sk1[j]*vn1; h1=g1*h1+sk1[j+1]*vn1;
            h2=g1*h2+sk1[j+2]*vn1; h3=g1*h3+sk1[j+3]*vn1;
            __half s0 = gdn_f16_store(h0);
            __half s1 = gdn_f16_store(h1);
            __half s2 = gdn_f16_store(h2);
            __half s3 = gdn_f16_store(h3);
            Hi1[(j+0)*WY3RF_VD+tid]=s0; Hi1[(j+1)*WY3RF_VD+tid]=s1;
            Hi1[(j+2)*WY3RF_VD+tid]=s2; Hi1[(j+3)*WY3RF_VD+tid]=s3;
            h0=__half2float(s0); h1=__half2float(s1);
            h2=__half2float(s2); h3=__half2float(s3);
            H_reg[j+0]=h0; H_reg[j+1]=h1;
            H_reg[j+2]=h2; H_reg[j+3]=h3;
            qd1 += h0*sq1[j]+h1*sq1[j+1]+h2*sq1[j+2]+h3*sq1[j+3];
        }

        // ── PASS 2c (token 2): H_3 = g2*H_2 + k2 ⊗ vn2, write final H ──
        #pragma unroll
        for (unsigned int j=0; j<WY3RF_KD; j+=4) {
            float h0=H_reg[j+0], h1=H_reg[j+1];
            float h2=H_reg[j+2], h3=H_reg[j+3];
            h0=g2*h0+sk2[j]*vn2; h1=g2*h1+sk2[j+1]*vn2;
            h2=g2*h2+sk2[j+2]*vn2; h3=g2*h3+sk2[j+3]*vn2;
            __half s0 = gdn_f16_store(h0);
            __half s1 = gdn_f16_store(h1);
            __half s2 = gdn_f16_store(h2);
            __half s3 = gdn_f16_store(h3);
            H[(j+0)*WY3RF_VD+tid]=s0; H[(j+1)*WY3RF_VD+tid]=s1;
            H[(j+2)*WY3RF_VD+tid]=s2; H[(j+3)*WY3RF_VD+tid]=s3;
            h0=__half2float(s0); h1=__half2float(s1);
            h2=__half2float(s2); h3=__half2float(s3);
            qd2 += h0*sq2[j]+h1*sq2[j+1]+h2*sq2[j+2]+h3*sq2[j+3];
        }

        float s = rsqrtf((float)k_dim);
        output[(b*3*num_v_heads+vh)*v_dim+tid]     = __float2bfloat16(qd0*s);
        output[((b*3+1)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd1*s);
        output[((b*3+2)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd2*s);
    }
}
