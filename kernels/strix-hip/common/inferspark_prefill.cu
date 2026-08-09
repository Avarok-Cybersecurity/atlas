// SPDX-License-Identifier: AGPL-3.0-only

// Inferspark Prefill Attention v2 — AMD HIP/WMMA port for gfx1151 (RDNA 3.5).
//
// Ported from the NVIDIA SM121 mma.sync.m16n8k16 implementation. Two transforms:
//   1. mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//        -> __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32  (wave32, n16 tiles)
//   2. cp.async.cg.shared.global / commit_group / wait_group
//        -> synchronous 16-byte smem copies (correctness-first, no pipelining)
//
// SOFTMAX APPROACH: shared-memory staging (NOT register rewrite).
//   In AMD WMMA the QK^T scores land as S[2*e + (lane>>4)][lane&15] — each lane
//   owns ONE key column across 8 query rows. The NVIDIA register softmax assumed
//   the m16n8k16 lane map (thread owns 2 rows x 2 col-pairs, reduce over
//   tid_in_group). Re-deriving the per-row max/sum reduction over the AMD lane
//   topology (cross-lane over l&15 plus cross-N-tile) is exactly where a naive
//   port produces garbage. Instead we store the QK^T score tile to smem_S using
//   the VALIDATED C-fragment store mapping (row=2*e+(l>>4), col=l&15; see
//   w4a16_wmma_ref.hip line 45), then perform a plain dense per-row online
//   softmax in shared memory (one thread per query row), write P to smem, and
//   feed P back into the PV WMMA. This is slower (extra smem round-trip) but the
//   softmax math is decoupled from the fragment layout and provably correct.
//
// STATUS: compiles-pending / numerics-pending GPU. Not validated on hardware.
//
// Fragment layout (gfx1151 wave32, VALIDATED idiom):
//   typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
//   typedef float  v8f   __attribute__((ext_vector_type(8)));
//   v8f d = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(v16bf a, v16bf b, v8f c);
//   A (MxK row-major): lane l -> a[i] = A[l&15][i]          (full K-row)
//   B (KxN):           lane l -> b[k] = B[k][l&15]          (full K-col / col l&15)
//   C/D out:           lane l, elem e(0..7) -> row=2*e+(l>>4), col=l&15

#include <cuda_bf16.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

#define BR 32
#define BC 32
#ifndef HDIM
#define HDIM 256
#endif
#define PAD_KV 8                       // 16-byte row alignment: (256+8)*2 = 528 bytes
#define HDIM_PAD (HDIM + PAD_KV)       // 264
#define PAD_S 0                        // smem_S is FP32, BC=32 cols already 16-byte aligned

// WMMA tiling constants (16x16x16):
#define K16 16
#define WMMA_K_STEPS (HDIM / K16)      // 16  (QK^T contraction over head_dim)
#define QK_N_TILES   (BC / K16)        // 2   (key columns split into 16-wide N-tiles)
#define PV_K_STEPS   (BC / K16)        // 2   (PV contraction over key dimension)
// PV: head_dim split across 2 warp-pairs; each warp owns half the d N-tiles.
#define PV_N_TILES   ((HDIM / K16) / 2)  // 8  (128 d-cols / 16 per warp)

#define TILE_CHUNKS (BR * (HDIM / 8))  // 32 * 32 = 1024 (16-byte chunks per tile)

// --------------------------------------------------------------------------
// Synchronous 16-byte smem copy helper (replaces cp.async.cg.shared.global).
// gmem may be out of range -> caller guards; on guard-fail writes zero.
// --------------------------------------------------------------------------
__device__ __forceinline__ void cp16(void* smem_dst, const void* gmem_src) {
    *(uint4*)smem_dst = *(const uint4*)gmem_src;
}

// ==========================================================================
// BR=32 variant: 4 warps (128 threads, wave32).
// ==========================================================================
extern "C" __global__ void inferspark_prefill(
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
    const unsigned int sliding_window   // 0 = no sliding limit
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int lane_lo = lane_id & 15;   // 0..15  -> WMMA col / row index
    const unsigned int lane_hi = lane_id >> 4;   // 0 or 1 -> C-frag row parity

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

    __shared__ __nv_bfloat16 smem_Q[BR][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P[BR][BC];      // P = exp(s - m_new), BC=32 16-byte aligned
    __shared__ float smem_S[BR][BC];              // raw scores staged from QK^T WMMA
    __shared__ float smem_ml[BR][2];              // [row][0]=m, [row][1]=l (online state)
    __shared__ float smem_resc[BR];               // exp(m_old - m_new) per row (acc_o rescale)

    // ---- PV warp role mapping ----
    // 4 warps: (warp_id & 1) selects query M-tile (rows 0-15 vs 16-31);
    //          (warp_id >> 1) selects d N-tile half (cols 0-127 vs 128-255).
    const unsigned int pv_warp_m   = (warp_id & 1) * 16;
    const unsigned int pv_n_start  = (warp_id >> 1) * PV_N_TILES;  // 0 or 8

    // Output accumulators: PV_N_TILES WMMA n16-tiles, each a v8f.
    v8f acc_o[PV_N_TILES];
    #pragma unroll
    for (int i = 0; i < PV_N_TILES; i++) acc_o[i] = v8f{0,0,0,0,0,0,0,0};

    // ---- KV block count ----
    unsigned int num_kv_blocks = (seq_len + BC - 1) / BC;
    if (causal) {
        unsigned int max_kv_block = (q_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1);
    }

    // Initialize online-softmax state (one slot per query row).
    for (unsigned int r = tid; r < BR; r += 128) {
        smem_ml[r][0] = -1e30f;   // m
        smem_ml[r][1] = 0.0f;     // l
    }

    // ---- Q tile load (synchronous) ----
    {
        const unsigned int chunks_per_row = HDIM / 8;  // 32
        for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
            unsigned int row = idx / chunks_per_row;
            unsigned int chunk = idx % chunks_per_row;
            unsigned int col = chunk * 8;
            unsigned int q_row = q_start + row;
            if (q_row < seq_len) {
                cp16(&smem_Q[row][col],
                     (const void*)&Q_batch[q_row * q_seq_stride + q_head * head_dim + col]);
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, seq_len);
        unsigned int kv_len = kv_end - kv_start;

        // ---- K and V tile loads (synchronous) ----
        {
            const unsigned int chunks_per_row = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS; idx += 128) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int kv_row = kv_start + row;
                if (kv_row < seq_len) {
                    cp16(&smem_K[row][col],
                         (const void*)&K_batch[kv_row * kv_seq_stride + kv_head * head_dim + col]);
                    cp16(&smem_V[row][col],
                         (const void*)&V_batch[kv_row * kv_seq_stride + kv_head * head_dim + col]);
                } else {
                    *((uint4*)&smem_K[row][col]) = make_uint4(0, 0, 0, 0);
                    *((uint4*)&smem_V[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        // ============================================================
        // QK^T: S = Q @ K^T  (warps 0-1, each owns 16 query rows)
        // A = Q[mtile][K]   : lane l -> a[i] = Q[mtile_base + (l&15)][k_off+i]
        // B = K^T -> b[k]   = K[key=ntile_base+(l&15)][k_off+k]
        // C out             : S[mtile_base + 2*e+(l>>4)][ntile_base + (l&15)]
        // ============================================================
        if (warp_id < 2) {
            const unsigned int qk_m = warp_id * 16;  // query-row base for this warp

            v8f acc_s[QK_N_TILES];
            #pragma unroll
            for (int n = 0; n < QK_N_TILES; n++) acc_s[n] = v8f{0,0,0,0,0,0,0,0};

            #pragma unroll
            for (unsigned int ks = 0; ks < WMMA_K_STEPS; ks++) {
                unsigned int k_off = ks * K16;

                // A-fragment: query row (qk_m + lane_lo), 16 contiguous K elems.
                v16bf a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_Q[qk_m + lane_lo][k_off + i];

                #pragma unroll
                for (int nt = 0; nt < QK_N_TILES; nt++) {
                    unsigned int key_row = nt * 16 + lane_lo;  // B column = output col
                    // B-fragment: b[k] = K[key_row][k_off+k]  (full K-col)
                    v16bf b;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        b[k] = (__bf16)(float)smem_K[key_row][k_off + k];
                    acc_s[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc_s[nt]);
                }
            }

            // Stage scores to smem_S using validated C-frag mapping.
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

        // ============================================================
        // Online softmax in shared memory — one thread per query row.
        // Threads 0..BR-1 each own one query row r. Compute scaled+masked
        // scores, row max, online m/l update, P=exp(s-m_new), rescale factor.
        // ============================================================
        if (tid < BR) {
            unsigned int r = tid;
            unsigned int qr = q_start + r;
            bool row_valid = (r < q_len);

            // Row max over valid key columns (after scale + mask).
            float rmax = -1e30f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float s = smem_S[r][c] * inv_sqrt_d;
                unsigned int kpos = kv_start + c;
                bool masked = (c >= kv_len) || !row_valid;
                if (causal && kpos > qr) masked = true;
                if (causal && sliding_window > 0 && kpos <= qr &&
                    (qr - kpos) >= sliding_window) masked = true;
                if (masked) s = -1e30f;
                smem_S[r][c] = s;            // store back scaled+masked score
                rmax = fmaxf(rmax, s);
            }

            float m_old = smem_ml[r][0];
            float l_old = smem_ml[r][1];
            float m_new = fmaxf(m_old, rmax);
            float resc = __expf(m_old - m_new);   // exp(m_old - m_new): rescale acc_o & l

            float sum = 0.0f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float p = __expf(smem_S[r][c] - m_new);
                smem_P[r][c] = __float2bfloat16(p);
                sum += p;
            }

            smem_ml[r][0] = m_new;
            smem_ml[r][1] = l_old * resc + sum;
            smem_resc[r] = resc;
        }
        __syncthreads();

        // ============================================================
        // Rescale existing output accumulators by per-row exp(m_old-m_new).
        // acc_o[nt][e] belongs to query row (pv_warp_m + 2*e + lane_hi).
        // ============================================================
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

        // ============================================================
        // PV: O += P @ V   (all 4 warps)
        // A = P[mtile][key]  : lane l -> a[i] = P[pv_warp_m + (l&15)][k_off+i]
        // B = V              : b[k] = V[key=k_off+k][d = pv_n_col]   (B col = l&15)
        // C out              : O[pv_warp_m + 2*e+(l>>4)][d = pv_n_col]
        // ============================================================
        {
            #pragma unroll
            for (unsigned int ks = 0; ks < PV_K_STEPS; ks++) {
                unsigned int k_off = ks * K16;

                // A-fragment: P row (pv_warp_m + lane_lo), 16 contiguous key cols.
                v16bf a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_P[pv_warp_m + lane_lo][k_off + i];

                #pragma unroll
                for (int nt = 0; nt < PV_N_TILES; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;  // B col = out col
                    // B-fragment: b[k] = V[key = k_off+k][d_col]
                    v16bf b;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        b[k] = (__bf16)(float)smem_V[k_off + k][d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc_o[nt]);
                }
            }
        }
        __syncthreads();
    }

    // ============================================================
    // Final normalization and store.
    // acc_o[nt][e] -> O[query_row = pv_warp_m + 2*e + lane_hi]
    //                  [d = (pv_n_start+nt)*16 + lane_lo]
    // ============================================================
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

// ==========================================================================
// BR=64 variant: 8 warps (256 threads, wave32) for longer sequences.
//   QK^T: warps 0-3, each owns 16 query rows (0-15,16-31,32-47,48-63).
//   PV:   8 warps in 4 M-pairs x 2 d-halves:
//           (warp_id & 3) selects query M-tile (0-15,16-31,32-47,48-63);
//           (warp_id >> 2) selects d N-tile half.
//
// SCALE/gfx1151: RDNA3.5 hard 64 KB/workgroup LDS cap. With BR64=64 the smem
// footprint is ~78.8 KB and will NOT fit. Mirroring the NVIDIA source's
// __SCALE__ guard, this variant is COMPILE-ONLY on AMD (non-paged contiguous
// prefill is not dispatched for FP8 chunked serving); BR64 is clamped to 32 so
// the binary fits LDS and links. The 256-thread launch still uses only warps
// 0-3 for QK^T and rows 0-31 for PV, processing 32 query rows. NVIDIA build
// uses the full BR64=64. @human-review: confirm AMD runtime never dispatches
// inferspark_prefill_64 with q_block sizing assuming 64 rows; if it does, this
// clamp silently drops query rows 32-63. The contiguous prefill path is the
// BR=32 kernel above on AMD.
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
#define BR64 32
#else
#define BR64 64
#endif
// QK^T warps = query rows / 16. When BR64 is clamped to 32 on AMD this is 2,
// keeping smem_Q/smem_S/smem_P (sized [BR64]) indexing in-bounds. On NVIDIA
// (BR64=64) this is 4, matching the original.
#define QK64_WARPS (BR64 / 16)
#define TILE_CHUNKS_Q64 (BR64 * (HDIM / 8))  // 2048
#define TILE_CHUNKS_KV  (BC * (HDIM / 8))     // 1024

extern "C" __global__ void inferspark_prefill_64(
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
    const unsigned int sliding_window   // 0 = no sliding limit
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int lane_lo = lane_id & 15;
    const unsigned int lane_hi = lane_id >> 4;

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
    const __nv_bfloat16* K_batch = K + batch * seq_len * kv_seq_stride;
    const __nv_bfloat16* V_batch = V + batch * seq_len * kv_seq_stride;
    __nv_bfloat16* O_batch = O + batch * seq_len * q_seq_stride;

    __shared__ __nv_bfloat16 smem_Q[BR64][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_K[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_V[BC][HDIM_PAD];
    __shared__ __nv_bfloat16 smem_P[BR64][BC];
    __shared__ float smem_S[BR64][BC];
    __shared__ float smem_ml[BR64][2];
    __shared__ float smem_resc[BR64];

    // PV M-tile: (warp_id mod QK64_WARPS) selects the 16-row query tile so the
    // base stays < BR64; (warp_id / QK64_WARPS) selects the d N-tile half.
    const unsigned int pv_warp_m  = (warp_id % QK64_WARPS) * 16;
    const unsigned int pv_n_start = (warp_id / QK64_WARPS) * PV_N_TILES;  // 0 or 8

    v8f acc_o[PV_N_TILES];
    #pragma unroll
    for (int i = 0; i < PV_N_TILES; i++) acc_o[i] = v8f{0,0,0,0,0,0,0,0};

    unsigned int num_kv_blocks = (seq_len + BC - 1) / BC;
    if (causal) {
        unsigned int max_kv_block = (q_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, max_kv_block + 1);
    }

    for (unsigned int r = tid; r < BR64; r += 256) {
        smem_ml[r][0] = -1e30f;
        smem_ml[r][1] = 0.0f;
    }

    // ---- Q tile load (64 rows, synchronous) ----
    {
        const unsigned int chunks_per_row = HDIM / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_Q64; idx += 256) {
            unsigned int row = idx / chunks_per_row;
            unsigned int chunk = idx % chunks_per_row;
            unsigned int col = chunk * 8;
            unsigned int q_row = q_start + row;
            if (q_row < seq_len) {
                cp16(&smem_Q[row][col],
                     (const void*)&Q_batch[q_row * q_seq_stride + q_head * head_dim + col]);
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, seq_len);
        unsigned int kv_len = kv_end - kv_start;

        // ---- K and V tile loads (synchronous) ----
        {
            const unsigned int chunks_per_row = HDIM / 8;
            for (unsigned int idx = tid; idx < TILE_CHUNKS_KV; idx += 256) {
                unsigned int row = idx / chunks_per_row;
                unsigned int chunk = idx % chunks_per_row;
                unsigned int col = chunk * 8;
                unsigned int kv_row = kv_start + row;
                if (kv_row < seq_len) {
                    cp16(&smem_K[row][col],
                         (const void*)&K_batch[kv_row * kv_seq_stride + kv_head * head_dim + col]);
                    cp16(&smem_V[row][col],
                         (const void*)&V_batch[kv_row * kv_seq_stride + kv_head * head_dim + col]);
                } else {
                    *((uint4*)&smem_K[row][col]) = make_uint4(0, 0, 0, 0);
                    *((uint4*)&smem_V[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        // ---- QK^T (warps 0..QK64_WARPS-1, each 16 query rows) ----
        if (warp_id < QK64_WARPS) {
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
                    v16bf b;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        b[k] = (__bf16)(float)smem_K[key_row][k_off + k];
                    acc_s[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc_s[nt]);
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

        // ---- Online softmax in smem — one thread per query row (0..63) ----
        if (tid < BR64) {
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
                if (causal && sliding_window > 0 && kpos <= qr &&
                    (qr - kpos) >= sliding_window) masked = true;
                if (masked) s = -1e30f;
                smem_S[r][c] = s;
                rmax = fmaxf(rmax, s);
            }

            float m_old = smem_ml[r][0];
            float l_old = smem_ml[r][1];
            float m_new = fmaxf(m_old, rmax);
            float resc = __expf(m_old - m_new);

            float sum = 0.0f;
            #pragma unroll
            for (unsigned int c = 0; c < BC; c++) {
                float p = __expf(smem_S[r][c] - m_new);
                smem_P[r][c] = __float2bfloat16(p);
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

        // ---- PV: O += P @ V ----
        // Guard pv_n_start: on the BR64-clamped AMD build QK64_WARPS=2, so
        // warp/2 can produce pv_n_start in {0,8,16,24}; only {0,8} index valid
        // head_dim columns. Out-of-range warps skip PV (compile-only path).
        if ((pv_n_start + PV_N_TILES) * K16 <= HDIM) {
            #pragma unroll
            for (unsigned int ks = 0; ks < PV_K_STEPS; ks++) {
                unsigned int k_off = ks * K16;
                v16bf a;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    a[i] = (__bf16)(float)smem_P[pv_warp_m + lane_lo][k_off + i];

                #pragma unroll
                for (int nt = 0; nt < PV_N_TILES; nt++) {
                    unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;
                    v16bf b;
                    #pragma unroll
                    for (int k = 0; k < 16; k++)
                        b[k] = (__bf16)(float)smem_V[k_off + k][d_col];
                    acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc_o[nt]);
                }
            }
        }
        __syncthreads();
    }

    // ---- Final normalization and store ----
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
