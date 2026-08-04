// SPDX-License-Identifier: AGPL-3.0-only

// HDIM=512 paged Flash Attention compute (Gemma-4 full-attention layers).
// AMD HIP/WMMA port (gfx1151, RDNA3.5) of the NVIDIA SM121 mma.sync version.
//
// Transforms (same as prefill_paged_compute.cuh):
//   mma.sync.m16n8k16 -> __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32
//   cp.async          -> synchronous uint4 smem copies (commit/wait dropped)
//   __shfl register softmax -> shared-memory-staged online softmax with the
//                              validated C-frag map S[2*e+(l>>4)][l&15].
//
// Layout: BR=32, BC=32, HDIM=512, dynamic shared memory, 8 warps (256 threads).
//   QK^T: warps 0-1 (each owns 16 Q rows).
//   PV:   8 warps in 4 d-groups × 2 query M-tiles.
//         pv_warp_m  = (warp_id & 1) * 16      (query rows 0-15 / 16-31)
//         pv_n_start = (warp_id >> 1) * 8       (d-tile groups 0/8/16/24)
//         → each warp owns 8 of the 32 d-tiles (128 of 512 head_dim cols).
//
// STATUS: compiles-pending / numerics-pending GPU.
//
// Expects the including .cu file to define:
//   LOAD_KV_TILE_512(cache, bt, smem_ptr, kv_s, kv_l, kvh, t, stride)
//   KERNEL_NAME, K_CACHE_TYPE, V_CACHE_TYPE, KERNEL_EXTRA_PARAMS, KERNEL_PREAMBLE

#include <cuda_bf16.h>

// Async global→shared 16-byte copy helpers. AMD/gfx1151 has no cp.async
// (hipcc rejects the PTX "l" constraint), so these degrade to synchronous
// uint4 copies; commit/wait become no-ops. The NVIDIA/SCALE copy of this
// header defines the same names as real cp.async. Per-tree behavior comes
// purely from which header is included — no #if at the call sites.
__device__ __forceinline__ void atlas_cp16(void* smem_dst, const void* gmem_src) {
    *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
}
__device__ __forceinline__ void atlas_cp16_pred(void* smem_dst, const void* gmem_src, bool pred) {
    if (pred) *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
    else      *reinterpret_cast<uint4*>(smem_dst) = make_uint4(0,0,0,0);
}
__device__ __forceinline__ void atlas_cp_commit() {}
__device__ __forceinline__ void atlas_cp_wait()   {}

typedef __bf16 v16bf_512 __attribute__((ext_vector_type(16)));
typedef float  v8f_512   __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float sw_exp_512(float x) {
    float t = x * 1.4426950408889634f;
    float ti = floorf(t);
    float tf = t - ti;
    float p = 1.0f + tf * (0.6931471805599453f +
              tf * (0.2402265069591007f +
              tf * 0.05550410866482158f));
    return ldexpf(p, (int)ti);
}

#define BR_512   32
#define BC_512   32
#define HDIM_512 512
#define PAD_P_512 8
#define K16_512 16
#define WMMA_K_STEPS_512 (HDIM_512 / K16_512)   // 32 (QK^T contraction)
#define QK_N_TILES_512   (BC_512 / K16_512)     // 2
#define PV_K_STEPS_512   (BC_512 / K16_512)     // 2
#define N_TILES_PER_WARP_512 8                  // 8 d-tiles per warp (128 cols)
#define TILE_CHUNKS_512 (BR_512 * (HDIM_512 / 8))

extern "C" __global__ void KERNEL_NAME(
    const __nv_bfloat16* __restrict__ Q,
    K_CACHE_TYPE K_cache,
    V_CACHE_TYPE V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_table,
    const unsigned int q_len,
    const unsigned int kv_len,
    const unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size,
    const unsigned int sliding_window,
    const unsigned int causal_mask_enabled
    KERNEL_EXTRA_PARAMS
) {
    const unsigned int q_head  = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int lane_lo = lane_id & 15;
    const unsigned int lane_hi = lane_id >> 4;

    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * BR_512;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + BR_512, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);

    extern __shared__ __align__(16) unsigned char smem_dyn_512[];
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem_dyn_512);
    __nv_bfloat16* smem_K = smem_Q + (unsigned int)BR_512 * HDIM_512;
    __nv_bfloat16* smem_V = smem_K + (unsigned int)BC_512 * HDIM_512;
    __nv_bfloat16* smem_P = smem_V + (unsigned int)BC_512 * HDIM_512;
    float* smem_S = reinterpret_cast<float*>(
                       smem_P + (unsigned int)BR_512 * (BC_512 + PAD_P_512));
    float* smem_ml = smem_S + (unsigned int)BR_512 * BC_512;       // [BR][2]
    float* smem_resc = smem_ml + (unsigned int)BR_512 * 2;         // [BR]

    KERNEL_PREAMBLE

    const unsigned int pv_warp_m  = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * N_TILES_PER_WARP_512;  // 0/8/16/24

    v8f_512 acc_o[N_TILES_PER_WARP_512];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP_512; i++)
        acc_o[i] = v8f_512{0,0,0,0,0,0,0,0};

    unsigned int num_kv_blocks = (kv_len + BC_512 - 1) / BC_512;
    { unsigned int mx = (q_offset + q_tile_end - 1) / BC_512;
      num_kv_blocks = min(num_kv_blocks, mx + 1); }

    for (unsigned int r = tid; r < BR_512; r += blockDim.x) {
        smem_ml[r * 2 + 0] = -1e30f;
        smem_ml[r * 2 + 1] = 0.0f;
    }

    // ---- Q tile load (synchronous) ----
    {
        const unsigned int cpr = HDIM_512 / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_512; idx += blockDim.x) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            if (q_start + row < q_len) {
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + q_head*head_dim + col];
                *((uint4*)&smem_Q[row * HDIM_512 + col]) = *((const uint4*)gm);
            } else {
                *((uint4*)&smem_Q[row * HDIM_512 + col]) = make_uint4(0,0,0,0);
            }
        }
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC_512;
        unsigned int kv_end   = min(kv_start + BC_512, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;

        LOAD_KV_TILE_512(K_cache, block_table, smem_K, kv_start, kv_len, kv_head, tid, blockDim.x);
        LOAD_KV_TILE_512(V_cache, block_table, smem_V, kv_start, kv_len, kv_head, tid, blockDim.x);
        __syncthreads();

        // ---- QK^T (warps 0-1) ----
        if (warp_id < 2) {
            const unsigned int qk_m = warp_id * 16;
            v8f_512 acc_s[QK_N_TILES_512];
            #pragma unroll
            for (int n = 0; n < QK_N_TILES_512; n++) acc_s[n] = v8f_512{0,0,0,0,0,0,0,0};

            #pragma unroll
            for (unsigned int ks = 0; ks < WMMA_K_STEPS_512; ks++) {
                unsigned int k_off = ks * K16_512;
                v16bf_512 a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_Q[(qk_m + lane_lo) * HDIM_512 + k_off + i];
                #pragma unroll
                for (int nt = 0; nt < QK_N_TILES_512; nt++) {
                    unsigned int key_row = nt * 16 + lane_lo;
                    v16bf_512 bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__bf16)(float)smem_K[key_row * HDIM_512 + k_off + k];
                    acc_s[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_s[nt]);
                }
            }

            #pragma unroll
            for (int nt = 0; nt < QK_N_TILES_512; nt++) {
                unsigned int col = nt * 16 + lane_lo;
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    unsigned int row = qk_m + 2 * e + lane_hi;
                    smem_S[row * BC_512 + col] = acc_s[nt][e];
                }
            }
        }
        __syncthreads();

        // ---- Online softmax in smem ----
        if (tid < BR_512) {
            unsigned int r = tid;
            unsigned int qr = q_offset + q_start + r;
            bool row_valid = (r < q_tile_len);
            float rmax = -1e30f;
            #pragma unroll
            for (unsigned int c = 0; c < BC_512; c++) {
                float s = smem_S[r * BC_512 + c] * inv_sqrt_d;
                unsigned int kpos = kv_start + c;
                bool masked = (c >= kv_tile_len) || !row_valid;
                if (causal_mask_enabled && kpos > qr) masked = true;
                if (sliding_window > 0 && kpos <= qr &&
                    (qr - kpos) >= sliding_window) masked = true;
                if (masked) s = -1e30f;
                smem_S[r * BC_512 + c] = s;
                rmax = fmaxf(rmax, s);
            }
            float m_old = smem_ml[r * 2 + 0];
            float l_old = smem_ml[r * 2 + 1];
            float m_new = fmaxf(m_old, rmax);
            float resc = sw_exp_512(m_old - m_new);
            float sum = 0.0f;
            #pragma unroll
            for (unsigned int c = 0; c < BC_512; c++) {
                float p = sw_exp_512(smem_S[r * BC_512 + c] - m_new);
                smem_P[r * (BC_512 + PAD_P_512) + c] = __float2bfloat16(p);
                sum += p;
            }
            smem_ml[r * 2 + 0] = m_new;
            smem_ml[r * 2 + 1] = l_old * resc + sum;
            smem_resc[r] = resc;
        }
        __syncthreads();

        // ---- Rescale acc_o ----
        {
            float resc_e[8];
            #pragma unroll
            for (int e = 0; e < 8; e++)
                resc_e[e] = smem_resc[pv_warp_m + 2 * e + lane_hi];
            #pragma unroll
            for (int nt = 0; nt < N_TILES_PER_WARP_512; nt++)
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    acc_o[nt][e] *= resc_e[e];
        }

        // ---- PV: O += P @ V ----
        {
            const unsigned int p_stride = BC_512 + PAD_P_512;
            #pragma unroll
            for (unsigned int ks = 0; ks < PV_K_STEPS_512; ks++) {
                unsigned int k_off = ks * K16_512;
                v16bf_512 a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_P[(pv_warp_m + lane_lo) * p_stride + k_off + i];
                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP_512; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;
                    v16bf_512 bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__bf16)(float)smem_V[(k_off + k) * HDIM_512 + d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_o[nt]);
                }
            }
        }
        __syncthreads();
    }

    // ---- Final normalization and store ----
    {
        __nv_bfloat16* ob = O + q_head * head_dim;
        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP_512; nt++) {
            unsigned int col = (pv_n_start + nt) * 16 + lane_lo;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                unsigned int row = pv_warp_m + 2 * e + lane_hi;
                unsigned int gr = q_start + row;
                if (gr < q_len && row < q_tile_len && col < head_dim) {
                    float l = smem_ml[row * 2 + 1];
                    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                    ob[gr * q_seq_stride + col] = __float2bfloat16(acc_o[nt][e] * inv_l);
                }
            }
        }
    }
}
