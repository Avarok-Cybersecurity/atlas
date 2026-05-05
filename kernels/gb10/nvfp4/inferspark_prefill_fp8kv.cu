// SPDX-License-Identifier: AGPL-3.0-only

// Contiguous Prefill Flash Attention — FP8 E4M3 K/V variant (BR=64).
//
// Q is BF16 (contiguous), K/V are FP8 E4M3 (contiguous, dequantized to BF16
// in shared memory). Same Flash Attention v2 algorithm as inferspark_prefill_64
// but halves K/V memory reads by loading 1-byte FP8 instead of 2-byte BF16.
//
// Grid: (num_q_heads, ceil(seq_len/64), batch)  Block: (256, 1, 1)
//
// Shared memory (~88 KB):
//   Q:   [64][264] BF16 = 33.0 KB
//   K:   [2][32][264] BF16 = 33.0 KB  (double-buffered)
//   V:   [32][264] BF16 = 16.5 KB
//   P:   [64][40]  BF16 =  5.0 KB
//   m/l: [64][2]   FP32 =  0.5 KB

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BR64 64
#define BC 32
#ifndef HDIM
#define HDIM 256
#endif
#define PAD_KV 8
#define HDIM_PAD (HDIM + PAD_KV)
#define PAD_P 8
#define N_TILES_PER_WARP ((HDIM / 8) / 2)  // 16

#define TILE_CHUNKS_Q64 (BR64 * (HDIM / 8))   // 2048
#define TILE_CHUNKS_KV  (BC * (HDIM / 8))      // 1024

// FP8 E4M3 → BF16 conversion (no scaling).
__device__ __forceinline__ __nv_bfloat16 fp8_to_bf16(__nv_fp8_storage_t b) {
    return __float2bfloat16(__half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
}

// Load 8 FP8 elements from global memory, dequant to BF16, store to smem.
// `smem_row` is a BF16 array, `col` is the starting column, `src` is FP8 ptr.
#define LOAD_FP8_CHUNK(smem_row, col, src) \
    do { \
        const __nv_fp8_storage_t* _p = (src); \
        __nv_bfloat16 _bv[8]; \
        _bv[0] = fp8_to_bf16(_p[0]); _bv[1] = fp8_to_bf16(_p[1]); \
        _bv[2] = fp8_to_bf16(_p[2]); _bv[3] = fp8_to_bf16(_p[3]); \
        _bv[4] = fp8_to_bf16(_p[4]); _bv[5] = fp8_to_bf16(_p[5]); \
        _bv[6] = fp8_to_bf16(_p[6]); _bv[7] = fp8_to_bf16(_p[7]); \
        *((uint4*)&(smem_row)[(col)]) = *((uint4*)_bv); \
    } while(0)

extern "C" __global__ void inferspark_prefill_fp8kv_64(
    const __nv_bfloat16* __restrict__ Q,              // [batch, seq, nq*hd] BF16
    const __nv_fp8_storage_t* __restrict__ K,          // [batch, seq, nkv*hd] FP8
    const __nv_fp8_storage_t* __restrict__ V,          // [batch, seq, nkv*hd] FP8
    __nv_bfloat16* __restrict__ O,                     // [batch, seq, nq*hd] BF16
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d,
    const unsigned int causal
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;

    const unsigned int q_start = q_block * BR64;
    if (q_start >= seq_len) return;
    const unsigned int q_end = min(q_start + BR64, seq_len);
    const unsigned int q_len = q_end - q_start;

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_seq_stride = num_kv_heads * head_dim;

    const __nv_bfloat16* Q_batch = Q + batch * seq_len * q_seq_stride;
    const __nv_fp8_storage_t* K_batch = K + (unsigned long long)batch * seq_len * kv_seq_stride;
    const __nv_fp8_storage_t* V_batch = V + (unsigned long long)batch * seq_len * kv_seq_stride;
    __nv_bfloat16* O_batch = O + batch * seq_len * q_seq_stride;

    __shared__ __nv_bfloat16 smem_Q[BR64][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K64[2][BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V64[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P64[BR64][BC + PAD_P];
    __shared__ float smem_ml64[BR64][2];

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;

    // QK^T: warps 0-3, each handles 16 M-rows
    const unsigned int qk_warp_m = warp_id * 16;

    // PV: all 8 warps, 4 pairs
    const unsigned int pv_warp_m = (warp_id & 3) * 16;
    const unsigned int pv_n_start = (warp_id >> 2) * N_TILES_PER_WARP;

    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }

    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    const unsigned int p_smem_stride64 = BC + PAD_P;

    // === KV block count ===
    unsigned int num_kv_blocks = (seq_len + BC - 1) / BC;
    if (causal) {
        unsigned int max_kv_block = (q_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1);
    }

    // === Load Q via cp.async (BF16, 16 bytes per chunk) ===
    {
        const unsigned int chunks_per_row = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_Q64; idx += 256) {
            unsigned int row = idx / chunks_per_row;
            unsigned int chunk = idx % chunks_per_row;
            unsigned int col = chunk * 8;
            unsigned int q_row = q_start + row;
            unsigned int smem_addr = __cvta_generic_to_shared(&smem_Q[row][col]);

            if (q_row < seq_len) {
                const void* gmem = (const void*)&Q_batch[q_row * q_seq_stride + q_head * head_dim + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
        asm volatile("cp.async.commit_group;");
    }

    // === Load K[0] via manual FP8 dequant (synchronous) ===
    if (num_kv_blocks > 0) {
        const unsigned int chunks_per_row = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 256) {
            unsigned int row = idx / chunks_per_row;
            unsigned int chunk = idx % chunks_per_row;
            unsigned int col = chunk * 8;
            if (row < seq_len) {
                LOAD_FP8_CHUNK(smem_K64[0][row], col,
                    &K_batch[row * kv_seq_stride + kv_head * head_dim + col]);
            } else {
                *((uint4*)&smem_K64[0][row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
    }

    // Wait for Q cp.async
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, seq_len);
        unsigned int kv_len = kv_end - kv_start;
        unsigned int buf = kv_block & 1;

        // === V load: manual FP8 dequant (all threads) ===
        // Warps 0-3 will proceed to QK^T after finishing their V loads.
        // Warps 4-7 may still be loading V — no conflict since QK^T
        // reads smem_Q/smem_K, not smem_V.
        {
            const unsigned int chunks_per_row = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 256) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int v_row = kv_start + row;
                if (v_row < seq_len) {
                    LOAD_FP8_CHUNK(smem_V64[row], col,
                        &V_batch[v_row * kv_seq_stride + kv_head * head_dim + col]);
                } else {
                    *((uint4*)&smem_V64[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }

        // === QK^T (warps 0-3, each 16 M-rows) ===
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
            for (unsigned int ks = 0; ks < (HDIM / 16); ks++) {
                unsigned int k_off = ks * 16;
                unsigned int ar0 = qk_warp_m + group_id;
                unsigned int ar1 = qk_warp_m + group_id + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = k_off + tid_in_group * 2 + 8;

                unsigned int a0 = *(const unsigned int*)&sQ[ar0 * HDIM_PAD + ac0];
                unsigned int a1 = *(const unsigned int*)&sQ[ar1 * HDIM_PAD + ac0];
                unsigned int a2 = *(const unsigned int*)&sQ[ar0 * HDIM_PAD + ac1];
                unsigned int a3 = *(const unsigned int*)&sQ[ar1 * HDIM_PAD + ac1];

                #pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    unsigned int n_col = nt * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k_off + tid_in_group * 2 + 8;

                    unsigned int b0 = ((unsigned int)sK[n_col * HDIM_PAD + k0 + 1] << 16) |
                                      (unsigned int)sK[n_col * HDIM_PAD + k0];
                    unsigned int b1 = ((unsigned int)sK[n_col * HDIM_PAD + k1 + 1] << 16) |
                                      (unsigned int)sK[n_col * HDIM_PAD + k1];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]),
                          "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_s[nt][0]), "f"(acc_s[nt][1]),
                          "f"(acc_s[nt][2]), "f"(acc_s[nt][3])
                    );
                }
            }

            // === Register-based softmax (warps 0-3) ===
            unsigned int row0 = qk_warp_m + group_id;
            unsigned int row1 = row0 + 8;

            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                acc_s[nt][0] *= inv_sqrt_d;
                acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d;
                acc_s[nt][3] *= inv_sqrt_d;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                unsigned int col1 = col0 + 1;

                if (causal) {
                    unsigned int qr0 = q_start + row0, qr1 = q_start + row1;
                    if (kv_start + col0 > qr0) acc_s[nt][0] = -1e30f;
                    if (kv_start + col1 > qr0) acc_s[nt][1] = -1e30f;
                    if (kv_start + col0 > qr1) acc_s[nt][2] = -1e30f;
                    if (kv_start + col1 > qr1) acc_s[nt][3] = -1e30f;
                }
                if (col0 >= kv_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (col1 >= kv_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (row0 >= q_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (row1 >= q_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
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

            float m_new0 = fmaxf(m_r0, rmax0);
            float exp_old0 = __expf(m_r0 - m_new0);
            l_r0 *= exp_old0;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_old0; acc_o[i][1] *= exp_old0;
            }
            m_r0 = m_new0;

            float m_new1 = fmaxf(m_r1, rmax1);
            float exp_old1 = __expf(m_r1 - m_new1);
            l_r1 *= exp_old1;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][2] *= exp_old1; acc_o[i][3] *= exp_old1;
            }
            m_r1 = m_new1;

            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                float p00 = __expf(acc_s[nt][0] - m_r0);
                float p01 = __expf(acc_s[nt][1] - m_r0);
                float p10 = __expf(acc_s[nt][2] - m_r1);
                float p11 = __expf(acc_s[nt][3] - m_r1);
                sum0 += p00 + p01;
                sum1 += p10 + p11;

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
            l_r0 += sum0;
            l_r1 += sum1;

            if (tid_in_group == 0) {
                smem_ml64[row0][0] = m_r0; smem_ml64[row0][1] = l_r0;
                smem_ml64[row1][0] = m_r1; smem_ml64[row1][1] = l_r1;
            }
        }

        // V was loaded synchronously before QK^T — no cp.async.wait needed.
        __syncthreads();

        // Warps 4-7: rescale accumulators to match current m
        if (warp_id >= 4) {
            unsigned int row0 = pv_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            float cur_m0 = smem_ml64[row0][0];
            float cur_m1 = smem_ml64[row1][0];
            float exp_r0 = __expf(m_r0 - cur_m0);
            float exp_r1 = __expf(m_r1 - cur_m1);
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) {
                acc_o[i][0] *= exp_r0; acc_o[i][1] *= exp_r0;
                acc_o[i][2] *= exp_r1; acc_o[i][3] *= exp_r1;
            }
            m_r0 = cur_m0; m_r1 = cur_m1;
        }

        // === Preload K[i+1] via manual FP8 dequant ===
        // Writes to smem_K64[1-buf] which is not read by PV below.
        if (kv_block + 1 < num_kv_blocks) {
            unsigned int next_kv_start = (kv_block + 1) * BC;
            const unsigned int chunks_per_row_k = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 256) {
                unsigned int row = idx / chunks_per_row_k;
                unsigned int chunk = idx % chunks_per_row_k;
                unsigned int col = chunk * 8;
                unsigned int k_row = next_kv_start + row;
                if (k_row < seq_len) {
                    LOAD_FP8_CHUNK(smem_K64[1 - buf][row], col,
                        &K_batch[k_row * kv_seq_stride + kv_head * head_dim + col]);
                } else {
                    *((uint4*)&smem_K64[1 - buf][row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }

        // === PV MMA (all 8 warps, 16 n-tiles each) ===
        {
            const unsigned short* sP = (const unsigned short*)smem_P64;
            const unsigned short* sV = (const unsigned short*)smem_V64;

            #pragma unroll
            for (unsigned int ks = 0; ks < 2; ks++) {
                unsigned int k_off = ks * 16;
                unsigned int ar0 = pv_warp_m + group_id;
                unsigned int ar1 = pv_warp_m + group_id + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = k_off + tid_in_group * 2 + 8;

                unsigned int a0 = *(const unsigned int*)&sP[ar0 * p_smem_stride64 + ac0];
                unsigned int a1 = *(const unsigned int*)&sP[ar1 * p_smem_stride64 + ac0];
                unsigned int a2 = *(const unsigned int*)&sP[ar0 * p_smem_stride64 + ac1];
                unsigned int a3 = *(const unsigned int*)&sP[ar1 * p_smem_stride64 + ac1];

                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
                    unsigned int n_col = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k_off + tid_in_group * 2 + 8;

                    unsigned int b0 = ((unsigned int)sV[(k0 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV[k0 * HDIM_PAD + n_col];
                    unsigned int b1 = ((unsigned int)sV[(k1 + 1) * HDIM_PAD + n_col] << 16) |
                                      (unsigned int)sV[k1 * HDIM_PAD + n_col];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]),
                          "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_o[nt][0]), "f"(acc_o[nt][1]),
                          "f"(acc_o[nt][2]), "f"(acc_o[nt][3])
                    );
                }
            }
        }

        __syncthreads();
    }

    // === Final normalization and store ===
    {
        unsigned int row0 = pv_warp_m + group_id;
        unsigned int row1 = row0 + 8;

        float inv_l0, inv_l1;
        if (warp_id < 4) {
            inv_l0 = (l_r0 > 0.0f) ? (1.0f / l_r0) : 0.0f;
            inv_l1 = (l_r1 > 0.0f) ? (1.0f / l_r1) : 0.0f;
        } else {
            float l0 = smem_ml64[row0][1];
            float l1 = smem_ml64[row1][1];
            inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
            inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
        }

        __nv_bfloat16* o_base = O_batch + q_head * head_dim;

        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            unsigned int col0 = (pv_n_start + nt) * 8 + tid_in_group * 2;
            unsigned int gr0 = q_start + row0;
            unsigned int gr1 = q_start + row1;

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
