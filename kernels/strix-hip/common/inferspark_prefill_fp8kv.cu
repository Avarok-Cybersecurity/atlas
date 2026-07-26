// SPDX-License-Identifier: AGPL-3.0-only

// Contiguous Prefill Flash Attention — FP8 E4M3 K/V variant. AMD HIP/WMMA port.
//
// Q is BF16 (contiguous); K/V are FP8 E4M3 (contiguous, dequantized to BF16 in
// shared memory, no per-group scale). Same WMMA Flash-Attention + smem-staged
// online softmax approach as inferspark_prefill_wmma.cu (HDIM=256). BR=64 is
// clamped to 32 to fit RDNA3.5 LDS; this kernel is a compile-only orphan on AMD
// (no runtime callers — never dispatched). cp.async -> synchronous load+dequant.
//
// Signature note: takes no sliding_window parameter (full/causal only).
//
// STATUS: compiles-pending / numerics-pending GPU.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

// Standard E4M3 -> FP32 decode (matches the NVIDIA reference scl_fp8).
__device__ __forceinline__ float scl_fp8_kv(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}

#define BR 32
#define BC 32
#define HDIM 256
#define PAD_KV 8
#define HDIM_PAD (HDIM + PAD_KV)

#define K16 16
#define WMMA_K_STEPS (HDIM / K16)        // 16
#define QK_N_TILES   (BC / K16)          // 2
#define PV_K_STEPS   (BC / K16)          // 2
#define PV_N_TILES   ((HDIM / K16) / 2)  // 8
#define TILE_CHUNKS (BR * (HDIM / 8))

__device__ __forceinline__ float sw_exp_fp8kv(float x) {
    float t = x * 1.4426950408889634f;
    float ti = floorf(t);
    float tf = t - ti;
    float p = 1.0f + tf * (0.6931471805599453f +
              tf * (0.2402265069591007f +
              tf * 0.05550410866482158f));
    return ldexpf(p, (int)ti);
}

// Dequant 8 contiguous FP8 elements from gmem -> BF16, store 16 bytes to smem.
__device__ __forceinline__ void load_fp8_chunk_kv(
    __nv_bfloat16* smem_dst, const __nv_fp8_storage_t* src) {
    __nv_bfloat16 bv[8];
    #pragma unroll
    for (int j = 0; j < 8; j++)
        bv[j] = __float2bfloat16(scl_fp8_kv((unsigned char)src[j]));
    *((uint4*)smem_dst) = *((uint4*)bv);
}

extern "C" __global__ void inferspark_prefill_fp8kv_64(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_fp8_storage_t* __restrict__ K,
    const __nv_fp8_storage_t* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
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
    const unsigned int lane_lo = lane_id & 15;
    const unsigned int lane_hi = lane_id >> 4;

    if (warp_id >= 4) return;  // clamped BR=32 uses 4 warps (256-thread launch ok)
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
    const __nv_fp8_storage_t* K_batch = K + (unsigned long long)batch * seq_len * kv_seq_stride;
    const __nv_fp8_storage_t* V_batch = V + (unsigned long long)batch * seq_len * kv_seq_stride;
    __nv_bfloat16* O_batch = O + batch * seq_len * q_seq_stride;

    __shared__ __nv_bfloat16 smem_Q[BR][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P[BR][BC];
    __shared__ float smem_S[BR][BC];
    __shared__ float smem_ml[BR][2];
    __shared__ float smem_resc[BR];

    const unsigned int pv_warp_m  = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * PV_N_TILES;

    v8f acc_o[PV_N_TILES];
    #pragma unroll
    for (int i = 0; i < PV_N_TILES; i++) acc_o[i] = v8f{0,0,0,0,0,0,0,0};

    unsigned int num_kv_blocks = (seq_len + BC - 1) / BC;
    if (causal) {
        unsigned int max_kv_block = (q_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1);
    }

    for (unsigned int r = tid; r < BR; r += 128) {
        smem_ml[r][0] = -1e30f;
        smem_ml[r][1] = 0.0f;
    }

    {
        const unsigned int cpr = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            unsigned int q_row = q_start + row;
            if (q_row < seq_len) {
                *((uint4*)&smem_Q[row][col]) =
                    *((const uint4*)&Q_batch[q_row * q_seq_stride + q_head * head_dim + col]);
            } else { *((uint4*)&smem_Q[row][col]) = make_uint4(0,0,0,0); }
        }
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, seq_len);
        unsigned int kv_len = kv_end - kv_start;

        {
            const unsigned int cpr = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
                unsigned int row = idx / cpr, col = (idx % cpr) * 8;
                unsigned int kv_row = kv_start + row;
                if (kv_row < seq_len) {
                    load_fp8_chunk_kv(&smem_K[row][col],
                        &K_batch[(unsigned long long)kv_row * kv_seq_stride + kv_head * head_dim + col]);
                    load_fp8_chunk_kv(&smem_V[row][col],
                        &V_batch[(unsigned long long)kv_row * kv_seq_stride + kv_head * head_dim + col]);
                } else {
                    *((uint4*)&smem_K[row][col]) = make_uint4(0,0,0,0);
                    *((uint4*)&smem_V[row][col]) = make_uint4(0,0,0,0);
                }
            }
        }
        __syncthreads();

        if (warp_id < 2) {
            const unsigned int qk_m = warp_id * 16;
            v8f acc_s[QK_N_TILES];
            #pragma unroll
            for (int n = 0; n < QK_N_TILES; n++) acc_s[n] = v8f{0,0,0,0,0,0,0,0};

            #pragma unroll
            for (unsigned int ks = 0; ks < WMMA_K_STEPS; ks++) {
                unsigned int k_off = ks * K16;
                v16bf a;
                #pragma unroll
                for (int i = 0; i < 16; i++) a[i] = (__bf16)(float)smem_Q[qk_m + lane_lo][k_off + i];
                #pragma unroll
                for (int nt = 0; nt < QK_N_TILES; nt++) {
                    unsigned int key_row = nt * 16 + lane_lo;
                    v16bf bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++) bb[k] = (__bf16)(float)smem_K[key_row][k_off + k];
                    acc_s[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_s[nt]);
                }
            }

            #pragma unroll
            for (int nt = 0; nt < QK_N_TILES; nt++) {
                unsigned int col = nt * 16 + lane_lo;
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    unsigned int row = qk_m + 2 * e + lane_hi;
                    smem_S[row][col] = acc_s[nt][e];
                }
            }
        }
        __syncthreads();

        if (tid < BR) {
            unsigned int r = tid;
            unsigned int qr = q_start + r;
            bool row_valid = (r < q_len);
            float rmax = -1e30f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float s = smem_S[r][c] * inv_sqrt_d;
                unsigned int kpos = kv_start + c;
                bool masked = (c >= kv_len) || !row_valid;
                if (causal && kpos > qr) masked = true;
                if (masked) s = -1e30f;
                smem_S[r][c] = s;
                rmax = fmaxf(rmax, s);
            }
            float m_old = smem_ml[r][0];
            float l_old = smem_ml[r][1];
            float m_new = fmaxf(m_old, rmax);
            float resc = sw_exp_fp8kv(m_old - m_new);
            float sum = 0.0f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float p = sw_exp_fp8kv(smem_S[r][c] - m_new);
                smem_P[r][c] = __float2bfloat16(p);
                sum += p;
            }
            smem_ml[r][0] = m_new;
            smem_ml[r][1] = l_old * resc + sum;
            smem_resc[r] = resc;
        }
        __syncthreads();

        {
            float resc_e[8];
            #pragma unroll
            for (int e = 0; e < 8; e++) resc_e[e] = smem_resc[pv_warp_m + 2 * e + lane_hi];
            #pragma unroll
            for (int nt = 0; nt < PV_N_TILES; nt++)
                #pragma unroll
                for (int e = 0; e < 8; e++) acc_o[nt][e] *= resc_e[e];
        }

        {
            #pragma unroll
            for (unsigned int ks = 0; ks < PV_K_STEPS; ks++) {
                unsigned int k_off = ks * K16;
                v16bf a;
                #pragma unroll
                for (int i = 0; i < 16; i++) a[i] = (__bf16)(float)smem_P[pv_warp_m + lane_lo][k_off + i];
                #pragma unroll
                for (int nt = 0; nt < PV_N_TILES; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;
                    v16bf bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++) bb[k] = (__bf16)(float)smem_V[k_off + k][d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_o[nt]);
                }
            }
        }
        __syncthreads();
    }

    {
        __nv_bfloat16* o_base = O_batch + q_head * head_dim;
        #pragma unroll
        for (int nt = 0; nt < PV_N_TILES; nt++) {
            unsigned int col = (pv_n_start + nt) * 16 + lane_lo;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                unsigned int row = pv_warp_m + 2 * e + lane_hi;
                unsigned int gr = q_start + row;
                if (gr < seq_len && row < q_len && col < head_dim) {
                    float l = smem_ml[row][1];
                    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                    o_base[gr * q_seq_stride + col] = __float2bfloat16(acc_o[nt][e] * inv_l);
                }
            }
        }
    }
}
