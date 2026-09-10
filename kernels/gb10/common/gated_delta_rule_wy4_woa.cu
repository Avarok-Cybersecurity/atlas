// SPDX-License-Identifier: AGPL-3.0-only

// Atlas WY-Chunkwise Gated Delta Rule — K=4 verification, WRITE-ON-ACCEPT.
//
// The parent (gated_delta_rule_wy4.cu) reads H twice and writes four full
// state widths per launch (Hi0, Hi1, Hi2, H), and a partial accept then
// copies one of those back over H. At C=16 that kernel is ~30% of the step
// and is at the DRAM wall — so the lever is bytes, not cycles.
//
// This pair moves the SAME arithmetic in the SAME order and writes the state
// exactly once, after the verdict:
//
//   gated_delta_rule_wy4_woa  — pass 1 loads the head's H slice into
//     shared memory (k_dim x v_dim floats, one column per thread), computes the four
//     k-dots, the WY correction, then the four in-register updates and the
//     four q-dots — identical expressions to the parent, so `output` is
//     byte-identical. It writes NO state. It stashes what the fold needs:
//     the four per-row vn vectors, the four gate scalars, and the four key
//     rows (as the same float values the parent used).
//
//   gated_delta_rule_wy4_fold — after the host knows how many rows were
//     accepted (na in 1..=4), applies rows 0..na-1 to H with the parent's
//     update expression `h = g*h + sk[j]*vn` in row order, reading H once
//     and writing it once. H_na is what the parent's Hi(na-1) / final H held.
//
// Traffic per step: parent 2 reads + 4 writes (+1 restore copy) -> here
// 1 read (woa) + 1 read + 1 write (fold). Grid (num_v_heads, batch), 128.
// State args are device POINTER TABLES (one entry per sequence), the same
// slab-0 table the parent's state_is_table=1 form reads.
//
// The twin needs 64 KB of dynamic shared memory (the head's 128x128 f32 H
// slice). The launch path sets CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
// for every launch over 48 KB (atlas-core registry.rs), so no separate
// opt-in is needed here; a refused launch is an error to the caller, never
// a silent swap to the parent.
//
// Engaged ONLY on the caller's request: the DFlash batched verify asks for
// it and folds; the MTP batched K-row verify never asks and keeps the
// parent wy4 (see spark-model qwen3_ssm/woa.rs).
//
// provenance-id: 526f6e616c6420522e205374657369616b

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#define BLOCK_SIZE 128
#define WOA_KD 128

// Stash layout per sequence (floats): vn[4][num_v_heads][v_dim] |
// g[4][num_v_heads] | sk[4][num_k_heads][k_dim]. The host passes the per-
// sequence float count so the layout can grow without a kernel change.
#define WOA_VN(base, T, VH, VD)      ((base) + ((T) * num_v_heads + (VH)) * (VD))
#define WOA_G(base, T, VH)           ((base) + 4 * num_v_heads * v_dim + (T) * num_v_heads + (VH))
#define WOA_SK(base, T, KH, KD)      ((base) + 4 * num_v_heads * v_dim + 4 * num_v_heads + ((T) * num_k_heads + (KH)) * (KD))

extern "C" __global__ void gated_delta_rule_wy4_woa(
    const float* __restrict__ h_state_table,   // float* const[batch]
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ stash,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int stash_seq_floats,
    // Device word this layer's fold reads: 1 = the woa twin ran for the last
    // batched verify (set here, INSIDE the captured graph, so replays count).
    // The HOST clears it (gated_delta_rule_wy4_flag_clear) at the top of
    // every batched verify that requests write-on-accept, in the same
    // capture, so the word describes that launch only. 0 = the parent ran.
    unsigned int* __restrict__ engaged_flag
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;
    // k_dim == v_dim == 128 is the HOST's check (woa.rs `woa_decision`):
    // no in-kernel dim guard on purpose. A silent early return here would
    // leave `output` unwritten with no kernel having run in its place.
    if (threadIdx.x == 0) *engaged_flag = 1u;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;
    const unsigned long long head_off = (unsigned long long)vh * hv;
    const float* H = ((const float* const*)h_state_table)[b] + head_off;
    float* sb = stash + (unsigned long long)b * stash_seq_floats;

    #define TP(T) \
        const __nv_bfloat16* q##T = query + (b*4+T)*qk_stride + kh*k_dim; \
        const __nv_bfloat16* k##T = key   + (b*4+T)*qk_stride + kh*k_dim; \
        const __nv_bfloat16* v##T = value + (b*4+T)*v_stride  + vh*v_dim; \
        const float g##T = fminf(fmaxf(gate[(b*4+T)*gb_stride + vh], 1e-6f), 1.0f - 1e-6f); \
        const float bt##T = beta[(b*4+T)*gb_stride + vh];
    TP(0) TP(1) TP(2) TP(3)
    #undef TP

    __shared__ float sk0[128], sq0[128], sk1[128], sq1[128];
    __shared__ float sk2[128], sq2[128], sk3[128], sq3[128];
    __shared__ float smem_warp[4];
    __shared__ float kd10, kd20, kd21, kd30, kd31, kd32;

    if (tid < k_dim) {
        sk0[tid]=(float)k0[tid]; sq0[tid]=(float)q0[tid];
        sk1[tid]=(float)k1[tid]; sq1[tid]=(float)q1[tid];
        sk2[tid]=(float)k2[tid]; sq2[tid]=(float)q2[tid];
        sk3[tid]=(float)k3[tid]; sq3[tid]=(float)q3[tid];
        // Key rows for the fold: one writer per k-head (the first v-head
        // that maps to it), the same float values used below.
        if (vh % hr == 0) {
            WOA_SK(sb, 0, kh, k_dim)[tid] = sk0[tid];
            WOA_SK(sb, 1, kh, k_dim)[tid] = sk1[tid];
            WOA_SK(sb, 2, kh, k_dim)[tid] = sk2[tid];
            WOA_SK(sb, 3, kh, k_dim)[tid] = sk3[tid];
        }
    }
    if (tid == 0) {
        *WOA_G(sb, 0, vh) = g0; *WOA_G(sb, 1, vh) = g1;
        *WOA_G(sb, 2, vh) = g2; *WOA_G(sb, 3, vh) = g3;
    }
    __syncthreads();

    #define KDOT(NAME, A, B) { \
        float p = (tid<k_dim) ? s##A[tid]*s##B[tid] : 0.0f; \
        float r = atlas_block_reduce_sum(p, smem_warp, tid); \
        if (tid==0) NAME = r; \
        __syncthreads(); \
    }
    KDOT(kd10, k1, k0)
    KDOT(kd20, k2, k0)
    KDOT(kd21, k2, k1)
    KDOT(kd30, k3, k0)
    KDOT(kd31, k3, k1)
    KDOT(kd32, k3, k2)
    #undef KDOT

    // The head's H slice lives in DYNAMIC shared memory for the launch
    // (k_dim x v_dim floats = 64 KB): pass 1 fills it from DRAM once, pass 2
    // reads it back. Column-per-thread indexing (j*v_dim + tid) is bank-
    // conflict-free. (A register-resident copy spilled 1.8 KB/thread.)
    extern __shared__ float hs[];

    if (tid < v_dim) {
        float vi0=(float)v0[tid], vi1=(float)v1[tid];
        float vi2=(float)v2[tid], vi3=(float)v3[tid];

        // ── PASS 1: read H ONCE (DRAM -> smem), compute the 4 dot products ──
        float hk0=0, hk1p=0, hk2p=0, hk3p=0;
        #pragma unroll 8
        for (unsigned int j=0; j<WOA_KD; j+=4) {
            float h0=H[(j+0)*v_dim+tid], h1=H[(j+1)*v_dim+tid];
            float h2=H[(j+2)*v_dim+tid], h3=H[(j+3)*v_dim+tid];
            hs[(j+0)*v_dim+tid]=h0; hs[(j+1)*v_dim+tid]=h1;
            hs[(j+2)*v_dim+tid]=h2; hs[(j+3)*v_dim+tid]=h3;
            hk0  += h0*sk0[j]+h1*sk0[j+1]+h2*sk0[j+2]+h3*sk0[j+3];
            hk1p += h0*sk1[j]+h1*sk1[j+1]+h2*sk1[j+2]+h3*sk1[j+3];
            hk2p += h0*sk2[j]+h1*sk2[j+1]+h2*sk2[j+2]+h3*sk2[j+3];
            hk3p += h0*sk3[j]+h1*sk3[j+1]+h2*sk3[j+2]+h3*sk3[j+3];
        }

        // ── WY Correction (verbatim) ──
        float vn0 = (vi0 - g0*hk0) * bt0;
        float hk1c = g0*hk1p + kd10*vn0;
        float vn1 = (vi1 - g1*hk1c) * bt1;
        float hk2c = g0*g1*hk2p + g1*kd20*vn0 + kd21*vn1;
        float vn2 = (vi2 - g2*hk2c) * bt2;
        float hk3c = g0*g1*g2*hk3p + g1*g2*kd30*vn0 + g2*kd31*vn1 + kd32*vn2;
        float vn3 = (vi3 - g3*hk3c) * bt3;

        // Stash the per-row update vectors for the fold.
        WOA_VN(sb, 0, vh, v_dim)[tid] = vn0;
        WOA_VN(sb, 1, vh, v_dim)[tid] = vn1;
        WOA_VN(sb, 2, vh, v_dim)[tid] = vn2;
        WOA_VN(sb, 3, vh, v_dim)[tid] = vn3;

        // ── PASS 2: the 4 updates from smem, no state writes ──
        float qd0=0, qd1=0, qd2=0, qd3=0;
        #pragma unroll 8
        for (unsigned int j=0; j<WOA_KD; j+=4) {
            float h0=hs[(j+0)*v_dim+tid], h1=hs[(j+1)*v_dim+tid];
            float h2=hs[(j+2)*v_dim+tid], h3=hs[(j+3)*v_dim+tid];
            h0=g0*h0+sk0[j]*vn0; h1=g0*h1+sk0[j+1]*vn0;
            h2=g0*h2+sk0[j+2]*vn0; h3=g0*h3+sk0[j+3]*vn0;
            qd0 += h0*sq0[j]+h1*sq0[j+1]+h2*sq0[j+2]+h3*sq0[j+3];
            h0=g1*h0+sk1[j]*vn1; h1=g1*h1+sk1[j+1]*vn1;
            h2=g1*h2+sk1[j+2]*vn1; h3=g1*h3+sk1[j+3]*vn1;
            qd1 += h0*sq1[j]+h1*sq1[j+1]+h2*sq1[j+2]+h3*sq1[j+3];
            h0=g2*h0+sk2[j]*vn2; h1=g2*h1+sk2[j+1]*vn2;
            h2=g2*h2+sk2[j+2]*vn2; h3=g2*h3+sk2[j+3]*vn2;
            qd2 += h0*sq2[j]+h1*sq2[j+1]+h2*sq2[j+2]+h3*sq2[j+3];
            h0=g3*h0+sk3[j]*vn3; h1=g3*h1+sk3[j+1]*vn3;
            h2=g3*h2+sk3[j+2]*vn3; h3=g3*h3+sk3[j+3]*vn3;
            qd3 += h0*sq3[j]+h1*sq3[j+1]+h2*sq3[j+2]+h3*sq3[j+3];
        }

        float s = rsqrtf((float)k_dim);
        output[(b*4*num_v_heads+vh)*v_dim+tid]     = __float2bfloat16(qd0*s);
        output[((b*4+1)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd1*s);
        output[((b*4+2)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd2*s);
        output[((b*4+3)*num_v_heads+vh)*v_dim+tid] = __float2bfloat16(qd3*s);
    }
}

// Apply the accepted rows 0..na-1 (na = na_tab[b], 0..=4) to H, once.
// `hi_tables`: the verify's Hi0 slab; Hi(t) is `slab_entries` pointers
// further on (the parent's intermediates). When `engaged_flag` is 0 the
// parent kernel ran and this launch performs the parent's partial-accept
// restore instead (H = Hi[na-1], nothing at na == 4), so the host never has
// to know which kernel the graph replayed.
extern "C" __global__ void gated_delta_rule_wy4_fold(
    float* __restrict__ h_state_table,          // float* const[batch]
    const float* __restrict__ stash,
    const unsigned int* __restrict__ na_tab,
    const float* __restrict__ hi_tables,        // float* const[slab_entries][batch]
    unsigned int slab_entries,
    const unsigned int* __restrict__ engaged_flag,
    unsigned int k_rows,                        // verify width K of the launch that ran
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int stash_seq_floats
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;
    const unsigned int na = na_tab[b];
    if (na == 0) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;
    float* H = ((float* const*)h_state_table)[b] + (unsigned long long)vh * hv;
    const float* sb = stash + (unsigned long long)b * stash_seq_floats;

    if (*engaged_flag == 0u) {
        // Parent (wy2/3/4 or the wyN table twin) ran: it wrote H (final) and
        // Hi0..Hi(K-2). Full accept keeps H; a partial accept of na rows
        // restores Hi(na-1). K comes from the host: this is NOT K=4-only.
        if (na >= k_rows) return;
        const float* src = ((const float* const*)hi_tables)[(na - 1) * slab_entries + b]
                           + (unsigned long long)vh * hv;
        for (unsigned int i = tid; i < hv; i += BLOCK_SIZE) H[i] = src[i];
        return;
    }

    __shared__ float sk0[128], sk1[128], sk2[128], sk3[128];
    if (tid < k_dim) {
        sk0[tid] = WOA_SK(sb, 0, kh, k_dim)[tid];
        sk1[tid] = WOA_SK(sb, 1, kh, k_dim)[tid];
        sk2[tid] = WOA_SK(sb, 2, kh, k_dim)[tid];
        sk3[tid] = WOA_SK(sb, 3, kh, k_dim)[tid];
    }
    __syncthreads();
    if (tid >= v_dim) return;

    const float g0 = *WOA_G(sb, 0, vh), g1 = *WOA_G(sb, 1, vh);
    const float g2 = *WOA_G(sb, 2, vh), g3 = *WOA_G(sb, 3, vh);
    const float vn0 = WOA_VN(sb, 0, vh, v_dim)[tid], vn1 = WOA_VN(sb, 1, vh, v_dim)[tid];
    const float vn2 = WOA_VN(sb, 2, vh, v_dim)[tid], vn3 = WOA_VN(sb, 3, vh, v_dim)[tid];

    // Dims are host-checked (see the twin above); no guard here either.
    #pragma unroll
    for (unsigned int j=0; j<WOA_KD; j+=4) {
        float h0=H[(j+0)*v_dim+tid], h1=H[(j+1)*v_dim+tid];
        float h2=H[(j+2)*v_dim+tid], h3=H[(j+3)*v_dim+tid];
        h0=g0*h0+sk0[j]*vn0; h1=g0*h1+sk0[j+1]*vn0;
        h2=g0*h2+sk0[j+2]*vn0; h3=g0*h3+sk0[j+3]*vn0;
        if (na > 1) {
            h0=g1*h0+sk1[j]*vn1; h1=g1*h1+sk1[j+1]*vn1;
            h2=g1*h2+sk1[j+2]*vn1; h3=g1*h3+sk1[j+3]*vn1;
        }
        if (na > 2) {
            h0=g2*h0+sk2[j]*vn2; h1=g2*h1+sk2[j+1]*vn2;
            h2=g2*h2+sk2[j+2]*vn2; h3=g2*h3+sk2[j+3]*vn2;
        }
        if (na > 3) {
            h0=g3*h0+sk3[j]*vn3; h1=g3*h1+sk3[j+1]*vn3;
            h2=g3*h2+sk3[j+2]*vn3; h3=g3*h3+sk3[j+3]*vn3;
        }
        H[(j+0)*v_dim+tid]=h0; H[(j+1)*v_dim+tid]=h1;
        H[(j+2)*v_dim+tid]=h2; H[(j+3)*v_dim+tid]=h3;
    }
}

// Reset the layer's engaged word after the fold (same stream, so ordered).
extern "C" __global__ void gated_delta_rule_wy4_flag_clear(unsigned int* __restrict__ engaged_flag) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *engaged_flag = 0u;
}
