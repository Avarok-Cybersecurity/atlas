// SPDX-License-Identifier: AGPL-3.0-only

// inferspark_prefill_128 — HDIM=128 variants of the inferspark flash-attention
// kernels, for MLA unabsorbed prefill (Mistral Small 4, head_dim=128).
//
// Root cause of the Mistral Small 4 long-context bug: inferspark_prefill.cu
// hardcodes HDIM=256. For MLA head_dim=128, the Q/K tile loads cover columns
// 0..255 in shared memory but each head only has 128 valid elements. Columns
// 128..255 read from the adjacent head (Q_head+1) and the next K row
// (K[k_row+1][0..127]), polluting QK^T with cross-head and cross-row data.
// Short sequences happen to tolerate the noise; long-range retrieval (>1K
// tokens) fails because the contaminated scores suppress correct early-context
// attention, producing repetitive or incoherent output.
//
// These HDIM=128 variants are structurally identical to the HDIM=256 originals:
//   - BR=32 kernel: 4 warps (128 threads), for any seq_len
//   - BR=64 kernel: 8 warps (256 threads), for seq_len >= 256
// Only compile-time constants differ (N_TILES_PER_WARP=8, TILE_CHUNKS=512, etc.)

#include <cuda_bf16.h>

#define BR   32
#define BC   32
#define HDIM 128
#define PAD_KV 8
#define HDIM_PAD (HDIM + PAD_KV)            // 136
#define PAD_P 8
#define N_TILES_PER_WARP ((HDIM / 8) / 2)   // 8
#define TILE_CHUNKS (BR * (HDIM / 8))        // 512
#define BR64 64
#define TILE_CHUNKS_Q64 (BR64 * (HDIM / 8)) // 1024
#define TILE_CHUNKS_KV  (BC  * (HDIM / 8))  // 512

// ============================================================================
// BR=32 HDIM=128 variant (4 warps / 128 threads).
// Grid: (num_q_heads, ceil(seq_len/32), batch)   Block: (128, 1, 1)
// Shared memory (~37 KB):
//   smem_Q  [32][136] BF16  =  8.5 KB
//   smem_K  [2][32][136] BF16 = 17.0 KB  (double-buffered)
//   smem_V  [32][136] BF16  =  8.5 KB
//   smem_P  [32][40]  BF16  =  2.5 KB
//   smem_ml [32][2]   FP32  =  0.25 KB
// ============================================================================
extern "C" __global__ void inferspark_prefill_hd128(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window
) {
    const unsigned int q_head  = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch   = blockIdx.z;
    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;

    const unsigned int q_start = q_block * BR;
    if (q_start >= seq_len) return;
    const unsigned int q_end = min(q_start + BR, seq_len);
    const unsigned int q_len = q_end - q_start;

    const unsigned int gqa_ratio   = num_q_heads / num_kv_heads;
    const unsigned int kv_head     = q_head / gqa_ratio;
    const unsigned int q_seq_stride  = num_q_heads * head_dim;
    const unsigned int kv_seq_stride = num_kv_heads * head_dim;

    const __nv_bfloat16* Q_batch = Q + (unsigned long long)batch * seq_len * q_seq_stride;
    const __nv_bfloat16* K_batch = K + (unsigned long long)batch * seq_len * kv_seq_stride;
    const __nv_bfloat16* V_batch = V + (unsigned long long)batch * seq_len * kv_seq_stride;
    __nv_bfloat16*       O_batch = O + (unsigned long long)batch * seq_len * q_seq_stride;

    __shared__ __nv_bfloat16 smem_Q[BR][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[2][BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P[BR][BC + PAD_P];
    __shared__ float          smem_ml[BR][2];

    const unsigned int group_id    = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m   = (warp_id & 1) * 16;
    const unsigned int pv_warp_m   = (warp_id & 1) * 16;
    const unsigned int pv_n_start  = (warp_id >> 1) * N_TILES_PER_WARP;

    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }

    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f,   l_r1 = 0.0f;
    const unsigned int p_smem_stride = BC + PAD_P;

    unsigned int num_kv_blocks = (seq_len + BC - 1) / BC;
    if (causal) {
        unsigned int max_kv_block = (q_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1);
    }

    // Merged Q + K[0] load
    {
        const unsigned int chunks_per_row = HDIM / 8;  // 16
        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
            unsigned int row   = idx / chunks_per_row;
            unsigned int col   = (idx % chunks_per_row) * 8;
            unsigned int q_row = q_start + row;
            unsigned int addr  = __cvta_generic_to_shared(&smem_Q[row][col]);
            if (q_row < seq_len) {
                const void* g = (const void*)&Q_batch[q_row * q_seq_stride + q_head * head_dim + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(g));
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
        if (num_kv_blocks > 0) {
            for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
                unsigned int row  = idx / chunks_per_row;
                unsigned int col  = (idx % chunks_per_row) * 8;
                unsigned int addr = __cvta_generic_to_shared(&smem_K[0][row][col]);
                if (row < seq_len) {
                    const void* g = (const void*)&K_batch[row * kv_seq_stride + kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(g));
                } else {
                    *((uint4*)&smem_K[0][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end   = min(kv_start + BC, seq_len);
        unsigned int kv_len   = kv_end - kv_start;
        unsigned int buf      = kv_block & 1;

        // Async V load (overlaps QK^T)
        {
            const unsigned int chunks_per_row = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
                unsigned int row   = idx / chunks_per_row;
                unsigned int col   = (idx % chunks_per_row) * 8;
                unsigned int v_row = kv_start + row;
                unsigned int addr  = __cvta_generic_to_shared(&smem_V[row][col]);
                if (v_row < seq_len) {
                    const void* g = (const void*)&V_batch[v_row * kv_seq_stride + kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(g));
                } else {
                    *((uint4*)&smem_V[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // QK^T (warps 0-1)
        float acc_s[4][4];
        if (warp_id < 2) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                acc_s[i][0] = 0.0f; acc_s[i][1] = 0.0f;
                acc_s[i][2] = 0.0f; acc_s[i][3] = 0.0f;
            }
            const unsigned short* sQ = (const unsigned short*)smem_Q;
            const unsigned short* sK = (const unsigned short*)smem_K[buf];
            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM / 16); ks++) {  // 8 k-tiles
                unsigned int k_base = ks * 16;
                unsigned int ar0 = qk_warp_m + group_id;
                unsigned int ar1 = ar0 + 8;
                unsigned int ac0 = k_base + tid_in_group * 2;
                unsigned int ac1 = ac0 + 8;
                unsigned int a0 = *(const unsigned int*)&sQ[ar0 * HDIM_PAD + ac0];
                unsigned int a1 = *(const unsigned int*)&sQ[ar1 * HDIM_PAD + ac0];
                unsigned int a2 = *(const unsigned int*)&sQ[ar0 * HDIM_PAD + ac1];
                unsigned int a3 = *(const unsigned int*)&sQ[ar1 * HDIM_PAD + ac1];
                #pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    unsigned int n_col = nt * 8 + group_id;
                    unsigned int k0 = k_base + tid_in_group * 2;
                    unsigned int k1 = k0 + 8;
                    unsigned int b0 = ((unsigned int)sK[n_col * HDIM_PAD + k0 + 1] << 16) |
                                       (unsigned int)sK[n_col * HDIM_PAD + k0];
                    unsigned int b1 = ((unsigned int)sK[n_col * HDIM_PAD + k1 + 1] << 16) |
                                       (unsigned int)sK[n_col * HDIM_PAD + k1];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_s[nt][0]),"=f"(acc_s[nt][1]),
                          "=f"(acc_s[nt][2]),"=f"(acc_s[nt][3])
                        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                          "f"(acc_s[nt][0]),"f"(acc_s[nt][1]),
                          "f"(acc_s[nt][2]),"f"(acc_s[nt][3])
                    );
                }
            }

            unsigned int row0 = qk_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                acc_s[nt][0] *= inv_sqrt_d; acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d; acc_s[nt][3] *= inv_sqrt_d;
                unsigned int col0 = nt * 8 + tid_in_group * 2;
                unsigned int col1 = col0 + 1;
                if (causal) {
                    unsigned int qr0 = q_start + row0, qr1 = q_start + row1;
                    if (kv_start + col0 > qr0) acc_s[nt][0] = -1e30f;
                    if (kv_start + col1 > qr0) acc_s[nt][1] = -1e30f;
                    if (kv_start + col0 > qr1) acc_s[nt][2] = -1e30f;
                    if (kv_start + col1 > qr1) acc_s[nt][3] = -1e30f;
                    if (sliding_window > 0) {
                        unsigned int k0 = kv_start + col0, k1 = kv_start + col1;
                        if (k0 <= qr0 && qr0 - k0 >= sliding_window) acc_s[nt][0] = -1e30f;
                        if (k1 <= qr0 && qr0 - k1 >= sliding_window) acc_s[nt][1] = -1e30f;
                        if (k0 <= qr1 && qr1 - k0 >= sliding_window) acc_s[nt][2] = -1e30f;
                        if (k1 <= qr1 && qr1 - k1 >= sliding_window) acc_s[nt][3] = -1e30f;
                    }
                }
                if (col0 >= kv_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (col1 >= kv_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (row0 >= q_len)  { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (row1 >= q_len)  { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }

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

            float m_new0 = fmaxf(m_r0, rmax0), exp_old0 = __expf(m_r0 - m_new0);
            l_r0 *= exp_old0;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_old0; acc_o[i][1] *= exp_old0;
            }
            m_r0 = m_new0;

            float m_new1 = fmaxf(m_r1, rmax1), exp_old1 = __expf(m_r1 - m_new1);
            l_r1 *= exp_old1;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][2] *= exp_old1; acc_o[i][3] *= exp_old1;
            }
            m_r1 = m_new1;

            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                float p00 = __expf(acc_s[nt][0] - m_r0), p01 = __expf(acc_s[nt][1] - m_r0);
                float p10 = __expf(acc_s[nt][2] - m_r1), p11 = __expf(acc_s[nt][3] - m_r1);
                sum0 += p00 + p01; sum1 += p10 + p11;
                unsigned int col0 = nt * 8 + tid_in_group * 2;
                smem_P[row0][col0]     = __float2bfloat16(p00);
                smem_P[row0][col0 + 1] = __float2bfloat16(p01);
                smem_P[row1][col0]     = __float2bfloat16(p10);
                smem_P[row1][col0 + 1] = __float2bfloat16(p11);
            }
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0; l_r1 += sum1;
            if (tid_in_group == 0) {
                smem_ml[row0][0] = m_r0; smem_ml[row0][1] = l_r0;
                smem_ml[row1][0] = m_r1; smem_ml[row1][1] = l_r1;
            }
        }

        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Warps 2-3: rescale accumulators to match current m
        if (warp_id >= 2) {
            unsigned int row0 = pv_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            float cur_m0 = smem_ml[row0][0], cur_m1 = smem_ml[row1][0];
            float exp_r0 = __expf(m_r0 - cur_m0), exp_r1 = __expf(m_r1 - cur_m1);
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_r0; acc_o[i][1] *= exp_r0;
                acc_o[i][2] *= exp_r1; acc_o[i][3] *= exp_r1;
            }
            m_r0 = cur_m0; m_r1 = cur_m1;
        }

        // Prefetch K[kv_block+1] (overlaps PV below)
        if (kv_block + 1 < num_kv_blocks) {
            unsigned int next_start = (kv_block + 1) * BC;
            const unsigned int cpr = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
                unsigned int row  = idx / cpr;
                unsigned int col  = (idx % cpr) * 8;
                unsigned int k_row = next_start + row;
                unsigned int addr = __cvta_generic_to_shared(&smem_K[1 - buf][row][col]);
                if (k_row < seq_len) {
                    const void* g = (const void*)&K_batch[k_row * kv_seq_stride + kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(g));
                } else {
                    *((uint4*)&smem_K[1 - buf][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // PV MMA (all 4 warps)
        {
            const unsigned short* sP = (const unsigned short*)smem_P;
            const unsigned short* sV = (const unsigned short*)smem_V;
            #pragma unroll
            for (unsigned int ks = 0; ks < 2; ks++) {
                unsigned int k_off = ks * 16;
                unsigned int ar0 = pv_warp_m + group_id;
                unsigned int ar1 = ar0 + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = ac0 + 8;
                unsigned int a0 = *(const unsigned int*)&sP[ar0 * p_smem_stride + ac0];
                unsigned int a1 = *(const unsigned int*)&sP[ar1 * p_smem_stride + ac0];
                unsigned int a2 = *(const unsigned int*)&sP[ar0 * p_smem_stride + ac1];
                unsigned int a3 = *(const unsigned int*)&sP[ar1 * p_smem_stride + ac1];
                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
                    unsigned int n_col = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k0 + 8;
                    unsigned int b0 = ((unsigned int)sV[(k0+1) * HDIM_PAD + n_col] << 16) |
                                       (unsigned int)sV[ k0    * HDIM_PAD + n_col];
                    unsigned int b1 = ((unsigned int)sV[(k1+1) * HDIM_PAD + n_col] << 16) |
                                       (unsigned int)sV[ k1    * HDIM_PAD + n_col];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),
                          "=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                          "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),
                          "f"(acc_o[nt][2]),"f"(acc_o[nt][3])
                    );
                }
            }
        }

        if (kv_block + 1 < num_kv_blocks) {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
    }

    // Final normalize + store
    {
        unsigned int row0 = pv_warp_m + group_id;
        unsigned int row1 = row0 + 8;
        float inv_l0, inv_l1;
        if (warp_id < 2) {
            inv_l0 = (l_r0 > 0.0f) ? (1.0f / l_r0) : 0.0f;
            inv_l1 = (l_r1 > 0.0f) ? (1.0f / l_r1) : 0.0f;
        } else {
            inv_l0 = (smem_ml[row0][1] > 0.0f) ? (1.0f / smem_ml[row0][1]) : 0.0f;
            inv_l1 = (smem_ml[row1][1] > 0.0f) ? (1.0f / smem_ml[row1][1]) : 0.0f;
        }
        __nv_bfloat16* o_base = O_batch + q_head * head_dim;
        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            unsigned int col0 = (pv_n_start + nt) * 8 + tid_in_group * 2;
            unsigned int gr0  = q_start + row0;
            unsigned int gr1  = q_start + row1;
            if (gr0 < seq_len && row0 < q_len && col0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * inv_l0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * inv_l0));
                *(unsigned int*)&o_base[gr0 * q_seq_stride + col0] = lo | (hi << 16);
            }
            if (gr1 < seq_len && row1 < q_len && col0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * inv_l1));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * inv_l1));
                *(unsigned int*)&o_base[gr1 * q_seq_stride + col0] = lo | (hi << 16);
            }
        }
    }
}

// ============================================================================
// BR=64 HDIM=128 variant (8 warps / 256 threads) for seq_len >= 256.
// Grid: (num_q_heads, ceil(seq_len/64), batch)   Block: (256, 1, 1)
// Shared memory (~49 KB):
//   smem_Q   [64][136] BF16  = 17.0 KB
//   smem_K64 [2][32][136] BF16 = 17.0 KB  (double-buffered)
//   smem_V64 [32][136] BF16  =  8.5 KB
//   smem_P64 [64][40]  BF16  =  5.0 KB
//   smem_ml64[64][2]   FP32  =  0.5 KB
// ============================================================================
extern "C" __global__ void inferspark_prefill_64_hd128(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window
) {
    const unsigned int q_head  = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch   = blockIdx.z;
    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;

    const unsigned int q_start = q_block * BR64;
    if (q_start >= seq_len) return;
    const unsigned int q_end = min(q_start + BR64, seq_len);
    const unsigned int q_len = q_end - q_start;

    const unsigned int gqa_ratio    = num_q_heads / num_kv_heads;
    const unsigned int kv_head      = q_head / gqa_ratio;
    const unsigned int q_seq_stride  = num_q_heads * head_dim;
    const unsigned int kv_seq_stride = num_kv_heads * head_dim;

    const __nv_bfloat16* Q_batch = Q + (unsigned long long)batch * seq_len * q_seq_stride;
    const __nv_bfloat16* K_batch = K + (unsigned long long)batch * seq_len * kv_seq_stride;
    const __nv_bfloat16* V_batch = V + (unsigned long long)batch * seq_len * kv_seq_stride;
    __nv_bfloat16*       O_batch = O + (unsigned long long)batch * seq_len * q_seq_stride;

    __shared__ __nv_bfloat16 smem_Q[BR64][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K64[2][BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V64[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P64[BR64][BC + PAD_P];
    __shared__ float          smem_ml64[BR64][2];

    const unsigned int group_id     = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m    = warp_id * 16;           // valid for warp_id < 4
    const unsigned int pv_warp_m    = (warp_id & 3) * 16;
    const unsigned int pv_n_start   = (warp_id >> 2) * N_TILES_PER_WARP;

    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }

    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f,   l_r1 = 0.0f;
    const unsigned int p_smem_stride64 = BC + PAD_P;

    unsigned int num_kv_blocks = (seq_len + BC - 1) / BC;
    if (causal) {
        unsigned int max_kv_block = (q_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1);
    }

    // Merged Q + K[0] load
    {
        const unsigned int cpr = HDIM / 8;  // 16 chunks per row
        for (unsigned int idx = tid; idx < TILE_CHUNKS_Q64; idx += 256) {
            unsigned int row   = idx / cpr;
            unsigned int col   = (idx % cpr) * 8;
            unsigned int q_row = q_start + row;
            unsigned int addr  = __cvta_generic_to_shared(&smem_Q[row][col]);
            if (q_row < seq_len) {
                const void* g = (const void*)&Q_batch[q_row * q_seq_stride + q_head * head_dim + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(g));
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
        if (num_kv_blocks > 0) {
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 256) {
                unsigned int row  = idx / cpr;
                unsigned int col  = (idx % cpr) * 8;
                unsigned int addr = __cvta_generic_to_shared(&smem_K64[0][row][col]);
                if (row < seq_len) {
                    const void* g = (const void*)&K_batch[row * kv_seq_stride + kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(g));
                } else {
                    *((uint4*)&smem_K64[0][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end   = min(kv_start + BC, seq_len);
        unsigned int kv_len   = kv_end - kv_start;
        unsigned int buf      = kv_block & 1;

        // Async V load
        {
            const unsigned int cpr = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 256) {
                unsigned int row   = idx / cpr;
                unsigned int col   = (idx % cpr) * 8;
                unsigned int v_row = kv_start + row;
                unsigned int addr  = __cvta_generic_to_shared(&smem_V64[row][col]);
                if (v_row < seq_len) {
                    const void* g = (const void*)&V_batch[v_row * kv_seq_stride + kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(g));
                } else {
                    *((uint4*)&smem_V64[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // QK^T (warps 0-3)
        float acc_s[4][4];
        if (warp_id < 4) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                acc_s[i][0] = 0.0f; acc_s[i][1] = 0.0f;
                acc_s[i][2] = 0.0f; acc_s[i][3] = 0.0f;
            }
            const unsigned short* sQ = (const unsigned short*)smem_Q;
            const unsigned short* sK = (const unsigned short*)smem_K64[buf];
            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM / 16); ks++) {  // 8 k-tiles
                unsigned int k_base = ks * 16;
                unsigned int ar0 = qk_warp_m + group_id;
                unsigned int ar1 = ar0 + 8;
                unsigned int ac0 = k_base + tid_in_group * 2;
                unsigned int ac1 = ac0 + 8;
                unsigned int a0 = *(const unsigned int*)&sQ[ar0 * HDIM_PAD + ac0];
                unsigned int a1 = *(const unsigned int*)&sQ[ar1 * HDIM_PAD + ac0];
                unsigned int a2 = *(const unsigned int*)&sQ[ar0 * HDIM_PAD + ac1];
                unsigned int a3 = *(const unsigned int*)&sQ[ar1 * HDIM_PAD + ac1];
                #pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    unsigned int n_col = nt * 8 + group_id;
                    unsigned int k0 = k_base + tid_in_group * 2;
                    unsigned int k1 = k0 + 8;
                    unsigned int b0 = ((unsigned int)sK[n_col * HDIM_PAD + k0 + 1] << 16) |
                                       (unsigned int)sK[n_col * HDIM_PAD + k0];
                    unsigned int b1 = ((unsigned int)sK[n_col * HDIM_PAD + k1 + 1] << 16) |
                                       (unsigned int)sK[n_col * HDIM_PAD + k1];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_s[nt][0]),"=f"(acc_s[nt][1]),
                          "=f"(acc_s[nt][2]),"=f"(acc_s[nt][3])
                        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                          "f"(acc_s[nt][0]),"f"(acc_s[nt][1]),
                          "f"(acc_s[nt][2]),"f"(acc_s[nt][3])
                    );
                }
            }

            unsigned int row0 = qk_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                acc_s[nt][0] *= inv_sqrt_d; acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d; acc_s[nt][3] *= inv_sqrt_d;
                unsigned int col0 = nt * 8 + tid_in_group * 2;
                unsigned int col1 = col0 + 1;
                if (causal) {
                    unsigned int qr0 = q_start + row0, qr1 = q_start + row1;
                    if (kv_start + col0 > qr0) acc_s[nt][0] = -1e30f;
                    if (kv_start + col1 > qr0) acc_s[nt][1] = -1e30f;
                    if (kv_start + col0 > qr1) acc_s[nt][2] = -1e30f;
                    if (kv_start + col1 > qr1) acc_s[nt][3] = -1e30f;
                    if (sliding_window > 0) {
                        unsigned int k0 = kv_start + col0, k1 = kv_start + col1;
                        if (k0 <= qr0 && qr0 - k0 >= sliding_window) acc_s[nt][0] = -1e30f;
                        if (k1 <= qr0 && qr0 - k1 >= sliding_window) acc_s[nt][1] = -1e30f;
                        if (k0 <= qr1 && qr1 - k0 >= sliding_window) acc_s[nt][2] = -1e30f;
                        if (k1 <= qr1 && qr1 - k1 >= sliding_window) acc_s[nt][3] = -1e30f;
                    }
                }
                if (col0 >= kv_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (col1 >= kv_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (row0 >= q_len)  { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (row1 >= q_len)  { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }

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

            float m_new0 = fmaxf(m_r0, rmax0), exp_old0 = __expf(m_r0 - m_new0);
            l_r0 *= exp_old0;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_old0; acc_o[i][1] *= exp_old0;
            }
            m_r0 = m_new0;

            float m_new1 = fmaxf(m_r1, rmax1), exp_old1 = __expf(m_r1 - m_new1);
            l_r1 *= exp_old1;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][2] *= exp_old1; acc_o[i][3] *= exp_old1;
            }
            m_r1 = m_new1;

            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                float p00 = __expf(acc_s[nt][0] - m_r0), p01 = __expf(acc_s[nt][1] - m_r0);
                float p10 = __expf(acc_s[nt][2] - m_r1), p11 = __expf(acc_s[nt][3] - m_r1);
                sum0 += p00 + p01; sum1 += p10 + p11;
                unsigned int col0 = nt * 8 + tid_in_group * 2;
                smem_P64[row0][col0]     = __float2bfloat16(p00);
                smem_P64[row0][col0 + 1] = __float2bfloat16(p01);
                smem_P64[row1][col0]     = __float2bfloat16(p10);
                smem_P64[row1][col0 + 1] = __float2bfloat16(p11);
            }
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0; l_r1 += sum1;
            if (tid_in_group == 0) {
                smem_ml64[row0][0] = m_r0; smem_ml64[row0][1] = l_r0;
                smem_ml64[row1][0] = m_r1; smem_ml64[row1][1] = l_r1;
            }
        }

        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Warps 4-7: rescale accumulators to match current m
        if (warp_id >= 4) {
            unsigned int row0 = pv_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            float cur_m0 = smem_ml64[row0][0], cur_m1 = smem_ml64[row1][0];
            float exp_r0 = __expf(m_r0 - cur_m0), exp_r1 = __expf(m_r1 - cur_m1);
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_r0; acc_o[i][1] *= exp_r0;
                acc_o[i][2] *= exp_r1; acc_o[i][3] *= exp_r1;
            }
            m_r0 = cur_m0; m_r1 = cur_m1;
        }

        // Prefetch K[kv_block+1]
        if (kv_block + 1 < num_kv_blocks) {
            unsigned int next_start = (kv_block + 1) * BC;
            const unsigned int cpr = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 256) {
                unsigned int row  = idx / cpr;
                unsigned int col  = (idx % cpr) * 8;
                unsigned int k_row = next_start + row;
                unsigned int addr = __cvta_generic_to_shared(&smem_K64[1 - buf][row][col]);
                if (k_row < seq_len) {
                    const void* g = (const void*)&K_batch[k_row * kv_seq_stride + kv_head * head_dim + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(g));
                } else {
                    *((uint4*)&smem_K64[1 - buf][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // PV MMA (all 8 warps)
        {
            const unsigned short* sP = (const unsigned short*)smem_P64;
            const unsigned short* sV = (const unsigned short*)smem_V64;
            #pragma unroll
            for (unsigned int ks = 0; ks < 2; ks++) {
                unsigned int k_off = ks * 16;
                unsigned int ar0 = pv_warp_m + group_id;
                unsigned int ar1 = ar0 + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = ac0 + 8;
                unsigned int a0 = *(const unsigned int*)&sP[ar0 * p_smem_stride64 + ac0];
                unsigned int a1 = *(const unsigned int*)&sP[ar1 * p_smem_stride64 + ac0];
                unsigned int a2 = *(const unsigned int*)&sP[ar0 * p_smem_stride64 + ac1];
                unsigned int a3 = *(const unsigned int*)&sP[ar1 * p_smem_stride64 + ac1];
                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
                    unsigned int n_col = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k0 + 8;
                    unsigned int b0 = ((unsigned int)sV[(k0+1) * HDIM_PAD + n_col] << 16) |
                                       (unsigned int)sV[ k0    * HDIM_PAD + n_col];
                    unsigned int b1 = ((unsigned int)sV[(k1+1) * HDIM_PAD + n_col] << 16) |
                                       (unsigned int)sV[ k1    * HDIM_PAD + n_col];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),
                          "=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                          "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),
                          "f"(acc_o[nt][2]),"f"(acc_o[nt][3])
                    );
                }
            }
        }

        if (kv_block + 1 < num_kv_blocks) {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
    }

    // Final normalize + store
    {
        unsigned int row0 = pv_warp_m + group_id;
        unsigned int row1 = row0 + 8;
        float inv_l0, inv_l1;
        if (warp_id < 4) {
            inv_l0 = (l_r0 > 0.0f) ? (1.0f / l_r0) : 0.0f;
            inv_l1 = (l_r1 > 0.0f) ? (1.0f / l_r1) : 0.0f;
        } else {
            inv_l0 = (smem_ml64[row0][1] > 0.0f) ? (1.0f / smem_ml64[row0][1]) : 0.0f;
            inv_l1 = (smem_ml64[row1][1] > 0.0f) ? (1.0f / smem_ml64[row1][1]) : 0.0f;
        }
        __nv_bfloat16* o_base = O_batch + q_head * head_dim;
        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            unsigned int col0 = (pv_n_start + nt) * 8 + tid_in_group * 2;
            unsigned int gr0  = q_start + row0;
            unsigned int gr1  = q_start + row1;
            if (gr0 < seq_len && row0 < q_len && col0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * inv_l0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * inv_l0));
                *(unsigned int*)&o_base[gr0 * q_seq_stride + col0] = lo | (hi << 16);
            }
            if (gr1 < seq_len && row1 < q_len && col0 < head_dim) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * inv_l1));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * inv_l1));
                *(unsigned int*)&o_base[gr1 * q_seq_stride + col0] = lo | (hi << 16);
            }
        }
    }
}
