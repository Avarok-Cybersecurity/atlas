// SPDX-License-Identifier: AGPL-3.0-only

// Per-Q-head paged Flash Attention compute — AMD HIP/WMMA port (gfx1151, RDNA3.5).
//
// Ported from the NVIDIA SM121 mma.sync.m16n8k16 implementation. Transforms:
//   1. mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//        -> __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32  (wave32, n16 tiles)
//   2. cp.async.cg.shared.global / commit_group / wait_group
//        -> synchronous 16-byte smem copies (correctness-first, no pipelining).
//        The BF16 LOAD_KV_TILE macros in the including .cu use synchronous uint4
//        copies; FP8/NVFP4 macros already do manual load+dequant. All commit/wait
//        group PTX is removed here and a __syncthreads() brackets each tile load.
//   3. __shfl_xor_sync register softmax (which assumed the NVIDIA m16n8k16 lane
//        map) -> shared-memory-staged online softmax. After QK^T WMMA the scores
//        land as S[2*e+(lane>>4)][lane&15]; we stage them to smem_S with that
//        VALIDATED C-fragment mapping (see w4a16_wmma_ref.hip line 45), run a
//        plain per-row online softmax (one thread per query row) in smem, write P,
//        and feed P back into the PV WMMA. This decouples the softmax math from
//        the fragment layout — the exact approach proven in inferspark_prefill_wmma.cu.
//
// STATUS: compiles-pending / numerics-pending GPU. Not validated on hardware.
//
// Covers (via the including .cu wrapper macros): inferspark_prefill_paged,
// _paged_batched, _paged_fp8, _paged_fp8_batched, _paged_nvfp4, _paged_nvfp4_batched.
//
// Fragment layout (gfx1151 wave32, VALIDATED idiom):
//   typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
//   typedef float  v8f   __attribute__((ext_vector_type(8)));
//   v8f d = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(v16bf a, v16bf b, v8f c);
//   A (MxK row-major): lane l -> a[i] = A[mrow + (l&15)][i]
//   B (KxN):           lane l -> b[k] = B[k][ncol + (l&15)]
//   C/D out:           lane l, elem e(0..7) -> row=2*e+(l>>4), col=l&15
//
// Expects the including file to define:
//   LOAD_KV_TILE(cache, block_table, smem, kv_start, kv_len, kv_head, tid, stride)
//   KERNEL_NAME, K_CACHE_TYPE, V_CACHE_TYPE, KERNEL_EXTRA_PARAMS, KERNEL_PREAMBLE
//   (optionally PREFILL_BATCHED)

#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef __fp16 v16h  __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

// Softmax exponential. Forward-port of NVIDIA PR #90 (Phase 2b precision
// fix, 2026-05-24): the prior degree-3 Taylor polynomial advertised
// "max err ~1e-4" but numerical verification showed **max relative error
// ~0.5%** near tf~1.0, compounding to ~5% cosine drift vs the PyTorch
// reference softmax over long attention rows (full-attention layers only;
// GDN linear-attention layers don't use softmax). Default path is now the
// hardware exp (matches the reference); the FA4-style polynomial is opt-in
// via ATLAS_FAST_SOFTMAX_EXP. (HIP/gfx1151: __expf is the transcendental
// SFU exp, the same accuracy class as the NVIDIA SFU __expf this replaces.)
__device__ __forceinline__ float sw_exp(float x) {
#ifdef ATLAS_FAST_SOFTMAX_EXP
    // FA4-style: degree-3 polynomial for 2^tf, max err ~0.5% at tf~1.
    float t = x * 1.4426950408889634f; // x * log2(e)
    float ti = floorf(t);
    float tf = t - ti;
    float p = 1.0f + tf * (0.6931471805599453f +
              tf * (0.2402265069591007f +
              tf * 0.05550410866482158f));
    return ldexpf(p, (int)ti);
#else
    // SSOT for prefill-attention softmax exp. Matches PyTorch reference.
    return __expf(x);
#endif
}

#define BR 32
#define BC 32
#ifndef HDIM
#define HDIM 256
#endif
#define PAD_KV 8
#define HDIM_PAD (HDIM + PAD_KV)
#define PAD_P 8

// WMMA tiling (16x16x16).
#define K16 16
#define WMMA_K_STEPS (HDIM / K16)      // contraction steps for QK^T (head_dim)
#define QK_N_TILES   (BC / K16)        // key columns split into 16-wide N-tiles (2)
#define PV_K_STEPS   (BC / K16)        // PV contraction over key dimension (2)
#define PV_N_TILES   ((HDIM / K16) / 2) // d-cols per warp half (8 at HDIM=256)
#define TILE_CHUNKS (BR * (HDIM / 8))

// ==========================================================================
// BR=32 variant: 4 warps (128 threads, wave32).
// ==========================================================================
extern "C" __global__ void KERNEL_NAME(
    const __nv_bfloat16* __restrict__ Q,
    K_CACHE_TYPE K_cache,
    V_CACHE_TYPE V_cache,
    __nv_bfloat16* __restrict__ O,
#ifdef PREFILL_BATCHED
    const int* const* __restrict__ block_table_ptrs,
    const unsigned int batch_size,
#else
    const int* __restrict__ block_table,
#endif
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
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
#ifdef PREFILL_BATCHED
    const unsigned int b = blockIdx.z;
    if (b >= batch_size) return;
    const int* const __restrict__ block_table = block_table_ptrs[b];
#endif
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int lane_lo = lane_id & 15;
    const unsigned int lane_hi = lane_id >> 4;

    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * BR;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + BR, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);
#ifdef PREFILL_BATCHED
    const unsigned long long q_batch_off = (unsigned long long)b * q_len * q_seq_stride;
#endif

    __shared__ __nv_bfloat16 smem_Q[BR][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    // PR #90 Phase 2c: P stored as FP16 (10-bit mantissa) vs BF16 (7-bit) →
    // 8× finer softmax-probability precision in the P×V WMMA, the largest
    // remaining attention-output drift source vs the FP32 PyTorch reference.
    // smem_V stays BF16 (the LOAD_KV_TILE macros write BF16); V is converted
    // to FP16 per-MMA in registers. Bisect: ATLAS_DISABLE_FP16_PV reverts to
    // the pre-#90 BF16 P×V (smem_P=BF16, __float2bfloat16 store, bf16 WMMA).
#ifdef ATLAS_DISABLE_FP16_PV
    __shared__ __nv_bfloat16 smem_P[BR][BC];
#else
    __shared__ __half smem_P[BR][BC];
#endif
    __shared__ float smem_S[BR][BC];
    __shared__ float smem_ml[BR][2];
    __shared__ float smem_resc[BR];

    KERNEL_PREAMBLE

    // PV warp role mapping (4 warps): (warp_id&1) selects query M-tile;
    // (warp_id>>1) selects d N-tile half (cols 0-127 vs 128-255 at HDIM=256).
    const unsigned int pv_warp_m  = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * PV_N_TILES;

    v8f acc_o[PV_N_TILES];
    #pragma unroll
    for (int i = 0; i < PV_N_TILES; i++) acc_o[i] = v8f{0,0,0,0,0,0,0,0};

    unsigned int num_kv_blocks = (kv_len + BC - 1) / BC;
    { unsigned int mx = (q_offset + q_tile_end - 1) / BC;
      num_kv_blocks = min(num_kv_blocks, mx + 1); }

    for (unsigned int r = tid; r < BR; r += blockDim.x) {
        smem_ml[r][0] = -1e30f;
        smem_ml[r][1] = 0.0f;
    }

    // ---- Q tile load (synchronous, BF16 contiguous) ----
    {
        const unsigned int cpr = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += blockDim.x) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            if (q_start + row < q_len) {
#ifdef PREFILL_BATCHED
                const void* gm = (const void*)&Q[q_batch_off + (q_start+row)*q_seq_stride + q_head*head_dim + col];
#else
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + q_head*head_dim + col];
#endif
                *((uint4*)&smem_Q[row][col]) = *((const uint4*)gm);
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0,0,0,0);
            }
        }
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;

        // ---- K and V tile loads (synchronous; FP8/NVFP4 dequant in macro) ----
        LOAD_KV_TILE(K_cache, block_table, smem_K, kv_start, kv_len, kv_head, tid, blockDim.x);
        LOAD_KV_TILE(V_cache, block_table, smem_V, kv_start, kv_len, kv_head, tid, blockDim.x);
        __syncthreads();

        // ---- QK^T: S = Q @ K^T (warps 0-1, each owns 16 query rows) ----
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
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_Q[qk_m + lane_lo][k_off + i];

                #pragma unroll
                for (int nt = 0; nt < QK_N_TILES; nt++) {
                    unsigned int key_row = nt * 16 + lane_lo;
                    v16bf bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__bf16)(float)smem_K[key_row][k_off + k];
                    acc_s[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_s[nt]);
                }
            }

            // Stage scores to smem_S via validated C-frag mapping.
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

        // ---- Online softmax in smem — one thread per query row ----
        if (tid < BR) {
            unsigned int r = tid;
            unsigned int qr = q_offset + q_start + r;
            bool row_valid = (r < q_tile_len);

            float rmax = -1e30f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float s = smem_S[r][c] * inv_sqrt_d;
                unsigned int kpos = kv_start + c;
                bool masked = (c >= kv_tile_len) || !row_valid;
                if (causal_mask_enabled && kpos > qr) masked = true;
                if (sliding_window > 0 && kpos <= qr &&
                    (qr - kpos) >= sliding_window) masked = true;
                if (masked) s = -1e30f;
                smem_S[r][c] = s;
                rmax = fmaxf(rmax, s);
            }

            float m_old = smem_ml[r][0];
            float l_old = smem_ml[r][1];
            float m_new = fmaxf(m_old, rmax);
            float resc = sw_exp(m_old - m_new);

            float sum = 0.0f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float p = sw_exp(smem_S[r][c] - m_new);
#ifdef ATLAS_DISABLE_FP16_PV
                smem_P[r][c] = __float2bfloat16(p);
#else
                smem_P[r][c] = __float2half(p);
#endif
                sum += p;
            }

            smem_ml[r][0] = m_new;
            smem_ml[r][1] = l_old * resc + sum;
            smem_resc[r] = resc;
        }
        __syncthreads();

        // ---- Rescale acc_o by per-row exp(m_old - m_new) ----
        {
            float resc_e[8];
            #pragma unroll
            for (int e = 0; e < 8; e++)
                resc_e[e] = smem_resc[pv_warp_m + 2 * e + lane_hi];
            #pragma unroll
            for (int nt = 0; nt < PV_N_TILES; nt++)
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    acc_o[nt][e] *= resc_e[e];
        }

        // ---- PV: O += P @ V (all 4 warps) ----
        // PR #90 Phase 2c: FP16 P×V WMMA (vs prior BF16) — 8× finer P
        // precision, same 16x16x16 wave32 tile/throughput. P is already FP16
        // in smem_P; V is converted BF16→FP16 in registers per-MMA. Q×K above
        // stays BF16 (Q/K come from the BF16 cache — no precision to recover).
        // Bisect: ATLAS_DISABLE_FP16_PV restores the pre-#90 BF16 P×V WMMA.
        {
            #pragma unroll
            for (unsigned int ks = 0; ks < PV_K_STEPS; ks++) {
                unsigned int k_off = ks * K16;
#ifdef ATLAS_DISABLE_FP16_PV
                v16bf a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_P[pv_warp_m + lane_lo][k_off + i];

                #pragma unroll
                for (int nt = 0; nt < PV_N_TILES; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;
                    v16bf bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__bf16)(float)smem_V[k_off + k][d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_o[nt]);
                }
#else
                v16h a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__fp16)__half2float(smem_P[pv_warp_m + lane_lo][k_off + i]);

                #pragma unroll
                for (int nt = 0; nt < PV_N_TILES; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;
                    v16h bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__fp16)(float)smem_V[k_off + k][d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bb, acc_o[nt]);
                }
#endif
            }
        }
        __syncthreads();
    }

    // ---- Final normalization and store ----
    {
#ifdef PREFILL_BATCHED
        __nv_bfloat16* ob = O + q_batch_off + q_head * head_dim;
#else
        __nv_bfloat16* ob = O + q_head * head_dim;
#endif
        #pragma unroll
        for (int nt = 0; nt < PV_N_TILES; nt++) {
            unsigned int col = (pv_n_start + nt) * 16 + lane_lo;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                unsigned int row = pv_warp_m + 2 * e + lane_hi;
                unsigned int gr = q_start + row;
                if (gr < q_len && row < q_tile_len && col < head_dim) {
                    float l = smem_ml[row][1];
                    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                    ob[gr * q_seq_stride + col] = __float2bfloat16(acc_o[nt][e] * inv_l);
                }
            }
        }
    }
}

// ==========================================================================
// BR=64 entry (KERNEL_NAME##_64). On AMD/gfx1151 the BR=64 large-chunk paged
// prefill kernels are COMPILE-ONLY (force_br32_prefill routes all dispatch to
// the BR=32 kernel above — see HARDWARE.toml / paged_attn.rs). To keep LDS
// within RDNA3.5's 64 KB cap and the grid math harmless, BR64 is clamped to 32
// here (mirroring inferspark_prefill_wmma.cu). The wave32 WMMA + smem-softmax
// body is identical to the BR=32 kernel; only the entry symbol differs so the
// registry links. NVIDIA keeps BR64=64 verbatim.
#define BR64 32
#define TILE_CHUNKS_Q64 (BR64 * (HDIM / 8))

#define _PAGED_CONCAT(a, b) a##b
#define PAGED_CONCAT(a, b) _PAGED_CONCAT(a, b)

extern "C" __global__ void PAGED_CONCAT(KERNEL_NAME, _64)(
    const __nv_bfloat16* __restrict__ Q,
    K_CACHE_TYPE K_cache,
    V_CACHE_TYPE V_cache,
    __nv_bfloat16* __restrict__ O,
#ifdef PREFILL_BATCHED
    const int* const* __restrict__ block_table_ptrs,
    const unsigned int batch_size,
#else
    const int* __restrict__ block_table,
#endif
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
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
#ifdef PREFILL_BATCHED
    const unsigned int b = blockIdx.z;
    if (b >= batch_size) return;
    const int* const __restrict__ block_table = block_table_ptrs[b];
#endif
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int lane_lo = lane_id & 15;
    const unsigned int lane_hi = lane_id >> 4;

    if (warp_id >= 4) return;  // clamped BR64=32 uses only 4 warps (128 threads)

    if (q_head >= num_q_heads) return;
    const unsigned int q_start = q_block * BR64;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + BR64, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);
#ifdef PREFILL_BATCHED
    const unsigned long long q_batch_off = (unsigned long long)b * q_len * q_seq_stride;
#endif

    __shared__ __nv_bfloat16 smem_Q[BR64][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    // PR #90 Phase 2c: smem_P64 FP16 — same rationale as the BR=32 path.
#ifdef ATLAS_DISABLE_FP16_PV
    __shared__ __nv_bfloat16 smem_P[BR64][BC];
#else
    __shared__ __half smem_P[BR64][BC];
#endif
    __shared__ float smem_S[BR64][BC];
    __shared__ float smem_ml[BR64][2];
    __shared__ float smem_resc[BR64];

    KERNEL_PREAMBLE

    const unsigned int pv_warp_m  = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * PV_N_TILES;

    v8f acc_o[PV_N_TILES];
    #pragma unroll
    for (int i = 0; i < PV_N_TILES; i++) acc_o[i] = v8f{0,0,0,0,0,0,0,0};

    unsigned int num_kv_blocks = (kv_len + BC - 1) / BC;
    { unsigned int mx = (q_offset + q_tile_end - 1) / BC;
      num_kv_blocks = min(num_kv_blocks, mx + 1); }

    for (unsigned int r = tid; r < BR64; r += 128) {
        smem_ml[r][0] = -1e30f;
        smem_ml[r][1] = 0.0f;
    }

    {
        const unsigned int cpr = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_Q64; idx += 128) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            if (q_start + row < q_len) {
#ifdef PREFILL_BATCHED
                const void* gm = (const void*)&Q[q_batch_off + (q_start+row)*q_seq_stride + q_head*head_dim + col];
#else
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + q_head*head_dim + col];
#endif
                *((uint4*)&smem_Q[row][col]) = *((const uint4*)gm);
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0,0,0,0);
            }
        }
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;

        LOAD_KV_TILE(K_cache, block_table, smem_K, kv_start, kv_len, kv_head, tid, 128);
        LOAD_KV_TILE(V_cache, block_table, smem_V, kv_start, kv_len, kv_head, tid, 128);
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
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_Q[qk_m + lane_lo][k_off + i];
                #pragma unroll
                for (int nt = 0; nt < QK_N_TILES; nt++) {
                    unsigned int key_row = nt * 16 + lane_lo;
                    v16bf bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__bf16)(float)smem_K[key_row][k_off + k];
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

        if (tid < BR64) {
            unsigned int r = tid;
            unsigned int qr = q_offset + q_start + r;
            bool row_valid = (r < q_tile_len);
            float rmax = -1e30f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float s = smem_S[r][c] * inv_sqrt_d;
                unsigned int kpos = kv_start + c;
                bool masked = (c >= kv_tile_len) || !row_valid;
                if (causal_mask_enabled && kpos > qr) masked = true;
                if (sliding_window > 0 && kpos <= qr &&
                    (qr - kpos) >= sliding_window) masked = true;
                if (masked) s = -1e30f;
                smem_S[r][c] = s;
                rmax = fmaxf(rmax, s);
            }
            float m_old = smem_ml[r][0];
            float l_old = smem_ml[r][1];
            float m_new = fmaxf(m_old, rmax);
            float resc = sw_exp(m_old - m_new);
            float sum = 0.0f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float p = sw_exp(smem_S[r][c] - m_new);
#ifdef ATLAS_DISABLE_FP16_PV
                smem_P[r][c] = __float2bfloat16(p);
#else
                smem_P[r][c] = __float2half(p);
#endif
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
            for (int e = 0; e < 8; e++)
                resc_e[e] = smem_resc[pv_warp_m + 2 * e + lane_hi];
            #pragma unroll
            for (int nt = 0; nt < PV_N_TILES; nt++)
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    acc_o[nt][e] *= resc_e[e];
        }

        // PR #90 Phase 2c: FP16 P×V WMMA — see BR=32 path for rationale.
        {
            #pragma unroll
            for (unsigned int ks = 0; ks < PV_K_STEPS; ks++) {
                unsigned int k_off = ks * K16;
#ifdef ATLAS_DISABLE_FP16_PV
                v16bf a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_P[pv_warp_m + lane_lo][k_off + i];
                #pragma unroll
                for (int nt = 0; nt < PV_N_TILES; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;
                    v16bf bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__bf16)(float)smem_V[k_off + k][d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_o[nt]);
                }
#else
                v16h a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__fp16)__half2float(smem_P[pv_warp_m + lane_lo][k_off + i]);
                #pragma unroll
                for (int nt = 0; nt < PV_N_TILES; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;
                    v16h bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        bb[k] = (__fp16)(float)smem_V[k_off + k][d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bb, acc_o[nt]);
                }
#endif
            }
        }
        __syncthreads();
    }

    {
#ifdef PREFILL_BATCHED
        __nv_bfloat16* ob = O + q_batch_off + q_head * head_dim;
#else
        __nv_bfloat16* ob = O + q_head * head_dim;
#endif
        #pragma unroll
        for (int nt = 0; nt < PV_N_TILES; nt++) {
            unsigned int col = (pv_n_start + nt) * 16 + lane_lo;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                unsigned int row = pv_warp_m + 2 * e + lane_hi;
                unsigned int gr = q_start + row;
                if (gr < q_len && row < q_tile_len && col < head_dim) {
                    float l = smem_ml[row][1];
                    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                    ob[gr * q_seq_stride + col] = __float2bfloat16(acc_o[nt][e] * inv_l);
                }
            }
        }
    }
}
