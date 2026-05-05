// SPDX-License-Identifier: AGPL-3.0-only

// Inferspark Prefill Attention v47 — V-load overlap with QK^T
//
// Key change from v45: V load starts BEFORE QK^T (not after SYNC 1)
//   V goes to smem[32..63] which is separate from K in smem[0..31]
//   So V load can happen concurrently with QK^T, hiding V's memory latency
//   Only K[next] load needs to wait for SYNC 1 (overwrites K smem)
//
// Flow: cp.async V → QK^T → SYNC1 → cp.async K[next] → softmax → wait V → SYNC2 → PV → wait K
//
// Architecture: BC=32, separate K/V smem, 2 syncs/iter, 2 CTAs/SM
// Same as v42/v45: 64 × 264 × 2 = 33,792 bytes smem

#include <cuda_bf16.h>

#define BR 64
#define BC 32
#define HDIM 256
#define PAD 8
#define HDIM_PAD (HDIM + PAD)    // 264
#define Q_CHUNKS (BR * (HDIM / 8))    // 2048
#define KV_CHUNKS (BC * (HDIM / 8))   // 1024

extern "C" __global__ __launch_bounds__(128, 2) void inferspark_prefill_v47(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d,
    const unsigned int causal
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = (gridDim.y - 1) - blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;

    const unsigned int q_start = q_block * BR;
    if (q_start >= seq_len) return;
    const unsigned int q_end = min(q_start + BR, seq_len);
    const unsigned int q_len = q_end - q_start;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_seq_stride = num_kv_heads * head_dim;

    const __nv_bfloat16* Q_batch = Q + batch * seq_len * q_seq_stride;
    const __nv_bfloat16* K_batch = K + batch * seq_len * kv_seq_stride;
    const __nv_bfloat16* V_batch = V + batch * seq_len * kv_seq_stride;
    __nv_bfloat16* O_batch = O + batch * seq_len * q_seq_stride;

    __shared__ __nv_bfloat16 smem[64][HDIM_PAD];

    unsigned int kv_end_max = causal ? min(q_start + BR, seq_len) : seq_len;
    unsigned int num_kv_blocks = (kv_end_max + BC - 1) / BC;

    // ---- cp.async addressing ----
    const unsigned int cpr = HDIM / 8;  // 32
    const unsigned int cp_first_row = tid / cpr;
    const unsigned int cp_first_col = (tid % cpr) * 8;
    const unsigned int sa_inc = 4u * (unsigned int)(HDIM_PAD * 2u);

    const unsigned int sa_base = __cvta_generic_to_shared(&smem[cp_first_row][cp_first_col]);
    const unsigned int sa_base_v = __cvta_generic_to_shared(&smem[32 + cp_first_row][cp_first_col]);

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int warp_m = warp_id * 16;

    // ====== Phase 1: Load Q to shared memory ======
    {
        const __nv_bfloat16* ga_Q = Q_batch + (q_start + cp_first_row) * q_seq_stride
                                    + q_head * head_dim + cp_first_col;
        const unsigned int q_ga_inc = 4u * q_seq_stride;
        #pragma unroll
        for (unsigned int i = 0; i < 16; i++) {
            unsigned int sa = sa_base + i * sa_inc;
            unsigned int q_row = q_start + cp_first_row + i * 4;
            if (q_row < seq_len) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(sa), "l"((const void*)(ga_Q + (unsigned long long)i * q_ga_inc)));
            } else {
                *((uint4*)((char*)smem + (sa - __cvta_generic_to_shared(smem)))) = make_uint4(0,0,0,0);
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    // ====== Phase 2: Extract Q fragments to registers ======
    unsigned int q_frag[64];
    {
        unsigned int q_ldm_base = __cvta_generic_to_shared(
            &smem[warp_m + (lane_id & 15)][(lane_id >> 4) * 8]);
        #pragma unroll
        for (unsigned int ks = 0; ks < 16; ks++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(q_frag[ks*4+0]), "=r"(q_frag[ks*4+1]),
                  "=r"(q_frag[ks*4+2]), "=r"(q_frag[ks*4+3])
                : "r"(q_ldm_base + ks * 32u));
        }
    }
    __syncthreads();

    // ====== Phase 3: Load K[0] to smem rows 0-31 ======
    {
        const __nv_bfloat16* ga_K = K_batch + cp_first_row * kv_seq_stride
                                    + kv_head * head_dim + cp_first_col;
        const unsigned int kv_ga_inc = 4u * kv_seq_stride;
        #pragma unroll
        for (unsigned int i = 0; i < 8; i++) {
            unsigned int sa = sa_base + i * sa_inc;
            unsigned int k_row = cp_first_row + i * 4;
            if (k_row < seq_len) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(sa), "l"((const void*)(ga_K + (unsigned long long)i * kv_ga_inc)));
            } else {
                *((uint4*)((char*)smem + (sa - __cvta_generic_to_shared(smem)))) = make_uint4(0,0,0,0);
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    // ====== Precompute addresses outside main loop ======
    // SM121 workaround: manual K loading instead of ldmatrix.trans
    const unsigned short* sK_u16 = (const unsigned short*)&smem[0][0];

    // V ldmatrix addressing
    const unsigned int v_row_in_8 = lane_id & 7;
    const unsigned int v_grp = lane_id >> 3;
    const unsigned int v_kv_off = (v_grp & 1) * 8;
    const unsigned int v_col_off = (v_grp >> 1) * 8;

    const unsigned int vb_base = __cvta_generic_to_shared(
        &smem[32 + v_kv_off + v_row_in_8][v_col_off]);

    // Softmax row indices (OPTIMIZATION: moved outside main loop)
    const unsigned int s_row0 = warp_m + group_id;
    const unsigned int s_row1 = s_row0 + 8;

    // Precompute KV global base (OPTIMIZATION: avoid per-iter recomputation)
    const __nv_bfloat16* kv_gbase = K_batch + kv_head * head_dim + cp_first_col;
    const __nv_bfloat16* vv_gbase = V_batch + kv_head * head_dim + cp_first_col;
    const unsigned int kv_ga_inc = 4u * kv_seq_stride;

    // Fast-path threshold for causal masking (OPTIMIZATION 2)
    const bool is_full_q = (q_len >= BR);

    // Initialize accumulators
    float acc_o[32][4];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    // ====== Main loop ======
    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, seq_len);
        unsigned int kv_len = kv_end - kv_start;

        // ---- Load V[j] → smem[32..63] FIRST (commit group A) ----
        // V smem doesn't overlap K smem, so this can run concurrently with QK^T
        {
            const __nv_bfloat16* ga_V = vv_gbase + (kv_start + cp_first_row) * kv_seq_stride;
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int sa = sa_base_v + i * sa_inc;
                unsigned int v_row = kv_start + cp_first_row + i * 4;
                if (v_row < seq_len) {
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                        :: "r"(sa), "l"((const void*)(ga_V + (unsigned long long)i * kv_ga_inc)));
                } else {
                    *((uint4*)((char*)smem + (sa - __cvta_generic_to_shared(smem)))) = make_uint4(0,0,0,0);
                }
            }
            asm volatile("cp.async.commit_group;");  // Group A: V (loads during QK^T)
        }

        // ---- QK^T: 2 N-pairs × 16 K-steps ----
        // V loading concurrently via cp.async into smem[32..63]
        // K is in smem[0..31] — no conflict
        float acc_s[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            acc_s[i][0] = 0.0f; acc_s[i][1] = 0.0f;
            acc_s[i][2] = 0.0f; acc_s[i][3] = 0.0f;
        }

        {
            // SM121 workaround: manual B-operand loading instead of ldmatrix.trans
            #pragma unroll
            for (unsigned int ks = 0; ks < 16; ks++) {
                unsigned int a0 = q_frag[ks*4+0], a1 = q_frag[ks*4+1];
                unsigned int a2 = q_frag[ks*4+2], a3 = q_frag[ks*4+3];

                unsigned int kd0 = ks * 16 + tid_in_group * 2;
                unsigned int kd1 = ks * 16 + tid_in_group * 2 + 8;

                // n-pair 0: K positions 0-7 and 8-15
                {
                    unsigned int n0 = group_id;
                    unsigned int n1 = 8 + group_id;
                    unsigned int b0 = ((unsigned int)sK_u16[n0 * HDIM_PAD + kd0 + 1] << 16) |
                                      (unsigned int)sK_u16[n0 * HDIM_PAD + kd0];
                    unsigned int b1 = ((unsigned int)sK_u16[n0 * HDIM_PAD + kd1 + 1] << 16) |
                                      (unsigned int)sK_u16[n0 * HDIM_PAD + kd1];
                    unsigned int b2 = ((unsigned int)sK_u16[n1 * HDIM_PAD + kd0 + 1] << 16) |
                                      (unsigned int)sK_u16[n1 * HDIM_PAD + kd0];
                    unsigned int b3 = ((unsigned int)sK_u16[n1 * HDIM_PAD + kd1 + 1] << 16) |
                                      (unsigned int)sK_u16[n1 * HDIM_PAD + kd1];
                    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_s[0][0]), "=f"(acc_s[0][1]),
                          "=f"(acc_s[0][2]), "=f"(acc_s[0][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_s[0][0]), "f"(acc_s[0][1]),
                          "f"(acc_s[0][2]), "f"(acc_s[0][3]));
                    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_s[1][0]), "=f"(acc_s[1][1]),
                          "=f"(acc_s[1][2]), "=f"(acc_s[1][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b2), "r"(b3),
                          "f"(acc_s[1][0]), "f"(acc_s[1][1]),
                          "f"(acc_s[1][2]), "f"(acc_s[1][3]));
                }

                // n-pair 1: K positions 16-23 and 24-31
                {
                    unsigned int n2 = 16 + group_id;
                    unsigned int n3 = 24 + group_id;
                    unsigned int c0 = ((unsigned int)sK_u16[n2 * HDIM_PAD + kd0 + 1] << 16) |
                                      (unsigned int)sK_u16[n2 * HDIM_PAD + kd0];
                    unsigned int c1 = ((unsigned int)sK_u16[n2 * HDIM_PAD + kd1 + 1] << 16) |
                                      (unsigned int)sK_u16[n2 * HDIM_PAD + kd1];
                    unsigned int c2 = ((unsigned int)sK_u16[n3 * HDIM_PAD + kd0 + 1] << 16) |
                                      (unsigned int)sK_u16[n3 * HDIM_PAD + kd0];
                    unsigned int c3 = ((unsigned int)sK_u16[n3 * HDIM_PAD + kd1 + 1] << 16) |
                                      (unsigned int)sK_u16[n3 * HDIM_PAD + kd1];
                    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_s[2][0]), "=f"(acc_s[2][1]),
                          "=f"(acc_s[2][2]), "=f"(acc_s[2][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(c0), "r"(c1),
                          "f"(acc_s[2][0]), "f"(acc_s[2][1]),
                          "f"(acc_s[2][2]), "f"(acc_s[2][3]));
                    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_s[3][0]), "=f"(acc_s[3][1]),
                          "=f"(acc_s[3][2]), "=f"(acc_s[3][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(c2), "r"(c3),
                          "f"(acc_s[3][0]), "f"(acc_s[3][1]),
                          "f"(acc_s[3][2]), "f"(acc_s[3][3]));
                }
            }
        }

        // ---- SYNC 1: all warps done reading K[j] from smem[0..31] ----
        __syncthreads();

        // ---- Load K[j+1] → smem[0..31] (commit group B) ----
        // Safe now: SYNC 1 ensures all K reads are complete
        {
            if (kv_block + 1 < num_kv_blocks) {
                const __nv_bfloat16* ga_Kn = kv_gbase
                    + ((kv_block + 1) * BC + cp_first_row) * kv_seq_stride;
                #pragma unroll
                for (unsigned int i = 0; i < 8; i++) {
                    unsigned int sa = sa_base + i * sa_inc;
                    unsigned int k_row = (kv_block + 1) * BC + cp_first_row + i * 4;
                    if (k_row < seq_len) {
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                            :: "r"(sa), "l"((const void*)(ga_Kn + (unsigned long long)i * kv_ga_inc)));
                    } else {
                        *((uint4*)((char*)smem + (sa - __cvta_generic_to_shared(smem)))) = make_uint4(0,0,0,0);
                    }
                }
            }
            asm volatile("cp.async.commit_group;");  // Group B: K[next]
        }

        // ---- Softmax ----
        // OPTIMIZATION 2: Fast path for tiles entirely below causal diagonal
        bool tile_clean = is_full_q && (kv_len == BC) &&
                          (!causal || kv_end <= q_start + 1);

        if (tile_clean) {
            // Fast path: just scale, no causal or bounds checks
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                acc_s[nt][0] *= inv_sqrt_d; acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d; acc_s[nt][3] *= inv_sqrt_d;
            }
        } else {
            // Slow path: scale + causal mask + bounds checks
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                acc_s[nt][0] *= inv_sqrt_d; acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d; acc_s[nt][3] *= inv_sqrt_d;
                unsigned int col0 = nt * 8 + tid_in_group * 2;
                unsigned int col1 = col0 + 1;
                if (causal) {
                    unsigned int qr0 = q_start + s_row0, qr1 = q_start + s_row1;
                    if (kv_start + col0 > qr0) acc_s[nt][0] = -1e30f;
                    if (kv_start + col1 > qr0) acc_s[nt][1] = -1e30f;
                    if (kv_start + col0 > qr1) acc_s[nt][2] = -1e30f;
                    if (kv_start + col1 > qr1) acc_s[nt][3] = -1e30f;
                }
                if (col0 >= kv_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (col1 >= kv_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (s_row0 >= q_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (s_row1 >= q_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }
        }

        // Row-wise max
        float rmax0 = -1e30f, rmax1 = -1e30f;
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
            rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
        }
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

        // Online softmax update
        float m_new0 = fmaxf(m_r0, rmax0);
        float exp_old0 = __expf(m_r0 - m_new0);
        l_r0 *= exp_old0;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            acc_o[i][0] *= exp_old0; acc_o[i][1] *= exp_old0;
        }
        m_r0 = m_new0;

        float m_new1 = fmaxf(m_r1, rmax1);
        float exp_old1 = __expf(m_r1 - m_new1);
        l_r1 *= exp_old1;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            acc_o[i][2] *= exp_old1; acc_o[i][3] *= exp_old1;
        }
        m_r1 = m_new1;

        // Exp and sum
        float sum0 = 0.0f, sum1 = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            acc_s[nt][0] = __expf(acc_s[nt][0] - m_r0);
            acc_s[nt][1] = __expf(acc_s[nt][1] - m_r0);
            acc_s[nt][2] = __expf(acc_s[nt][2] - m_r1);
            acc_s[nt][3] = __expf(acc_s[nt][3] - m_r1);
            sum0 += acc_s[nt][0] + acc_s[nt][1];
            sum1 += acc_s[nt][2] + acc_s[nt][3];
        }
        sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
        sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
        sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
        sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
        l_r0 += sum0;
        l_r1 += sum1;

        // ---- Wait for V[j] (group A) — K[j+1] (group B) may still be in flight ----
        // OPTIMIZATION 3: wait_group 1 — only wait for V, K continues loading during PV
        asm volatile("cp.async.wait_group 1;");
        // ---- SYNC 2 ----
        __syncthreads();

        // ---- PV ----
        {
            #pragma unroll
            for (unsigned int ks = 0; ks < 2; ks++) {
                unsigned int pa0, pa1, pa2, pa3;
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(pa0) : "f"(acc_s[2*ks][1]), "f"(acc_s[2*ks][0]));
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(pa1) : "f"(acc_s[2*ks][3]), "f"(acc_s[2*ks][2]));
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(pa2) : "f"(acc_s[2*ks+1][1]), "f"(acc_s[2*ks+1][0]));
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(pa3) : "f"(acc_s[2*ks+1][3]), "f"(acc_s[2*ks+1][2]));

                unsigned int va = vb_base + ks * 16u * (unsigned int)(HDIM_PAD * 2);

                unsigned int vf0, vf1, vf2, vf3;
                asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(vf0), "=r"(vf1), "=r"(vf2), "=r"(vf3) : "r"(va));

                #pragma unroll
                for (int p = 0; p < 15; p++) {
                    unsigned int vn0, vn1, vn2, vn3;
                    asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                        : "=r"(vn0), "=r"(vn1), "=r"(vn2), "=r"(vn3)
                        : "r"(va + (unsigned int)(p + 1) * 32u));

                    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_o[p*2][0]), "=f"(acc_o[p*2][1]),
                          "=f"(acc_o[p*2][2]), "=f"(acc_o[p*2][3])
                        : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3),
                          "r"(vf0), "r"(vf1),
                          "f"(acc_o[p*2][0]), "f"(acc_o[p*2][1]),
                          "f"(acc_o[p*2][2]), "f"(acc_o[p*2][3]));
                    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_o[p*2+1][0]), "=f"(acc_o[p*2+1][1]),
                          "=f"(acc_o[p*2+1][2]), "=f"(acc_o[p*2+1][3])
                        : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3),
                          "r"(vf2), "r"(vf3),
                          "f"(acc_o[p*2+1][0]), "f"(acc_o[p*2+1][1]),
                          "f"(acc_o[p*2+1][2]), "f"(acc_o[p*2+1][3]));
                    vf0 = vn0; vf1 = vn1; vf2 = vn2; vf3 = vn3;
                }
                // Last V fragment
                asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    : "=f"(acc_o[30][0]), "=f"(acc_o[30][1]),
                      "=f"(acc_o[30][2]), "=f"(acc_o[30][3])
                    : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3),
                      "r"(vf0), "r"(vf1),
                      "f"(acc_o[30][0]), "f"(acc_o[30][1]),
                      "f"(acc_o[30][2]), "f"(acc_o[30][3]));
                asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    : "=f"(acc_o[31][0]), "=f"(acc_o[31][1]),
                      "=f"(acc_o[31][2]), "=f"(acc_o[31][3])
                    : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3),
                      "r"(vf2), "r"(vf3),
                      "f"(acc_o[31][0]), "f"(acc_o[31][1]),
                      "f"(acc_o[31][2]), "f"(acc_o[31][3]));
            }
        }

        // Ensure K[j+1] completes before next iteration reads it
        // This wait is AFTER PV, so K[next] had PV duration to complete
        if (kv_block + 1 < num_kv_blocks) {
            asm volatile("cp.async.wait_group 0;");
        }
    }

    // ====== Output ======
    {
        unsigned int out_row0 = warp_m + group_id;
        unsigned int out_row1 = out_row0 + 8;
        float inv_l0 = (l_r0 > 0.0f) ? (1.0f / l_r0) : 0.0f;
        float inv_l1 = (l_r1 > 0.0f) ? (1.0f / l_r1) : 0.0f;
        #pragma unroll
        for (int nt = 0; nt < 32; nt++) {
            unsigned int col0 = nt * 8 + tid_in_group * 2;
            unsigned int col1 = col0 + 1;
            unsigned int gr0 = q_start + out_row0;
            unsigned int gr1 = q_start + out_row1;
            __nv_bfloat16* o_base = O_batch + q_head * head_dim;
            if (gr0 < seq_len && out_row0 < q_len && col1 < head_dim) {
                __nv_bfloat16* o_ptr = o_base + gr0 * q_seq_stride;
                unsigned int packed;
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed)
                    : "f"(acc_o[nt][1] * inv_l0), "f"(acc_o[nt][0] * inv_l0));
                *(unsigned int*)&o_ptr[col0] = packed;
            } else if (gr0 < seq_len && out_row0 < q_len && col0 < head_dim) {
                __nv_bfloat16* o_ptr = o_base + gr0 * q_seq_stride;
                o_ptr[col0] = __float2bfloat16(acc_o[nt][0] * inv_l0);
            }
            if (gr1 < seq_len && out_row1 < q_len && col1 < head_dim) {
                __nv_bfloat16* o_ptr = o_base + gr1 * q_seq_stride;
                unsigned int packed;
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed)
                    : "f"(acc_o[nt][3] * inv_l1), "f"(acc_o[nt][2] * inv_l1));
                *(unsigned int*)&o_ptr[col0] = packed;
            } else if (gr1 < seq_len && out_row1 < q_len && col0 < head_dim) {
                __nv_bfloat16* o_ptr = o_base + gr1 * q_seq_stride;
                o_ptr[col0] = __float2bfloat16(acc_o[nt][2] * inv_l1);
            }
        }
    }
}
