// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Grouped W4A16 GEMM for MoE — All experts in one kernel launch.
//
// C[total_tokens, N] = A[total_tokens, K] * dequant(B[expert, K, N/2])
//
// Each expert has its own packed FP4 weights and FP8 scales.
// expert_offsets[e] gives the starting row in A/C for expert e.
// expert_offsets[e+1] - expert_offsets[e] = number of tokens for expert e.
//
// Grid: (ceil(N/N_TILE), max_m_tiles, num_experts)
//   blockIdx.x: N tile index
//   blockIdx.y: M tile index within this expert's batch
//   blockIdx.z: expert index
//
// Fused dequant: E2M1_LUT[nibble] * fp8_scale * scale2 → BF16 in shared memory
// Compute: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//
// For Qwen3-Next: 256 experts, hidden=2048, inter=512
//   Gate-up: A[M_e, 2048] × W1[2048, 1024] → [M_e, 1024]
//   Down:    A[M_e, 512]  × W2[512, 2048]  → [M_e, 2048]

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define M_TILE 64
#define N_TILE 64
#define K_STEP 16
// Wider K tile used ONLY by moe_w4a16_grouped_gemm_ptrtable (the kernel the sorted
// prefill path dispatches). At K_STEP=16 each B row contributes just 8 bytes per
// tile -> 25% utilisation of a 32B memory sector, capping the expert-weight stream
// at ~1/4 of LPDDR5X bandwidth. 64 gives 32 contiguous bytes/row = one full sector.
#define K_STEP_PT 64
// Wider N tile for moe_w4a16_grouped_gemm_ptrtable ONLY.
//
// The A (activation) tile is re-read ONCE PER n-TILE. At N_TILE=64 the up-GEMM has
// 32 n-tiles, so the kernel moves 2.7 GB per GEMM of which only 537 MB is weights --
// the activations are 4x the weight traffic. Doubling the N tile halves the n-tiles
// (32 -> 16) and cuts total traffic ~40%. This attacks the ACTUAL constraint; the
// kernel is bandwidth-walled (135 GB/s = 71% of what llama.cpp gets on GB10), so
// every smem/pipelining tweak was rearranging deck chairs.
// Costs: 64 accumulator regs (4 m-tiles x 4 n-tiles) and ~7 KB more smem.
#define N_TILE_PT 128
#define PAD 2
#define GROUP_SIZE 16

// Packed-B smem row stride for the cp.async pipeline: K_STEP_PT/2 = 32 bytes of
// payload, padded to 48 so rows stay 16-byte aligned and don't all collide on the
// same shared-memory banks.
// One full 128-byte cache line of packed B per row = 256 k-values.
#define K_OUTER 256
// Pad the staged rows so they do not all land on the same shared-memory banks.
#define BP_PAD 16
#define BP_STRIDE 48
// Scale bytes per row per K tile (K_STEP_PT / GROUP_SIZE).
#define BS_PER_ROW (K_STEP_PT / GROUP_SIZE)

// The 33.8 GB of expert weights are read EXACTLY ONCE per prefill, so caching them in
// L2 is pure pollution -- they evict the A tiles and routing metadata that ARE reused.
// Marlin tags its weight stream with an evict-first L2 policy for exactly this reason.
__device__ __forceinline__ uint4 moe_ld_stream_u4(const void* p, unsigned long long pol) {
    uint4 v;
    asm volatile("ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p), "l"(pol) : "memory");
    return v;
}

__device__ __forceinline__ void moe_cp_async16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void moe_cp_async4(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void moe_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
// Wait until at most N groups remain in flight.
template <int N>
__device__ __forceinline__ void moe_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __constant__ float E2M1_LUT_MOE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void moe_w4a16_grouped_gemm(
    const __nv_bfloat16* __restrict__ A,        // [total_tokens, K] permuted activations
    const unsigned char* __restrict__ B_packed,  // [num_experts, K, N/2] packed FP4 weights
    const unsigned char* __restrict__ B_scale,   // [num_experts, K/GROUP_SIZE, N] FP8 scales
    const float scale2,                          // Per-tensor scale
    __nv_bfloat16* __restrict__ C,               // [total_tokens, N] output
    const int* __restrict__ expert_offsets,       // [num_experts + 1] prefix sum
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    // Which expert am I?
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    // Row range for this expert
    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    // My CTA's M tile within this expert
    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    // Global M offset
    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE;

    // Expert-specific weight pointers — N-major layout: B[N, K/2], S[N, K/GROUP_SIZE]
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int weight_stride_packed = N * half_K;       // bytes per expert in B_packed
    const unsigned int scale_stride = N * num_groups;           // bytes per expert in B_scale
    const unsigned char* B_expert = B_packed + expert_id * weight_stride_packed;
    const unsigned char* S_expert = B_scale + expert_id * scale_stride;

    // Warp/lane setup
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // Shared memory
    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    // Accumulators
    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE + PAD;

    // Effective M for this CTA (may be less than M_TILE for last tile)
    const unsigned int M_eff = (unsigned int)M_expert;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        // === Load A tile ===
        {
            const unsigned int elems_per_thread = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gr = cta_m + row;
                unsigned int gc = k_base + col;
                // Bounds check against actual expert token count and K
                bool valid = (cta_m_local + row) < M_eff && gc < K;
                smem_A[row][col] = valid ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
        }

        // === Load B tile: dequant FP4 → BF16 ===
        {
            // Coalesced B-tile load (N-major B: K is contiguous within a row).
            //
            // The old mapping (idx = tid*8+i; k = idx/N_TILE; n = idx%N_TILE) handed
            // each thread 8 elements sharing one k but spanning 8 different n, i.e.
            // 8 separate 1-byte loads at `gn * half_K` stride — 8 distinct cache
            // lines per thread. On LPDDR5X this is the dominant prefill cost, since
            // B (the expert weights) is by far the largest stream.
            //
            // New mapping: each thread owns ONE row and a contiguous run of K.
            //   thread t -> n = t>>1, khalf = t&1, k0 = k_base + khalf*8
            //   -> 8 k-values == 4 contiguous packed bytes, one vector load.
            // Threads 2i / 2i+1 cover the two halves of row i's 8-byte window, so
            // adjacent lanes touch adjacent addresses.
            //
            // GROUP_SIZE=16 and k0 is a multiple of 8, so a thread's 8 k-values all
            // land in scale group k_base/GROUP_SIZE => one scale load, not eight.
            unsigned int scale_group = k_base / GROUP_SIZE;
            {
                const unsigned int n = threadIdx.x >> 1;
                const unsigned int khalf = threadIdx.x & 1u;
                const unsigned int k0 = k_base + khalf * 8u;
                const unsigned int gn = cta_n + n;
                const unsigned int kl = khalf * 8u;  // local k offset in the tile

                if (n < N_TILE) {
                    if (gn < N && k0 < K) {
                        const unsigned long long byte_off =
                            (unsigned long long)gn * half_K + (k0 >> 1);

                        unsigned char scale_byte =
                            S_expert[(unsigned long long)gn * num_groups + scale_group];
                        float fp8_val;
                        {
                            __nv_fp8_e4m3 fp8;
                            *(unsigned char*)&fp8 = scale_byte;
                            fp8_val = (float)fp8;
                        }
                        const float s = fp8_val * scale2;

                        unsigned char pb[4];
                        if (((half_K & 3u) == 0u) && (k0 + 8u <= K)) {
                            unsigned int w = *(const unsigned int*)(B_expert + byte_off);
                            pb[0] = (unsigned char)(w & 0xFFu);
                            pb[1] = (unsigned char)((w >> 8) & 0xFFu);
                            pb[2] = (unsigned char)((w >> 16) & 0xFFu);
                            pb[3] = (unsigned char)((w >> 24) & 0xFFu);
                        } else {
                            #pragma unroll
                            for (unsigned int j = 0; j < 4; j++) {
                                unsigned int gk = k0 + j * 2u;
                                pb[j] = (gk < K) ? B_expert[byte_off + j] : 0u;
                            }
                        }

                        #pragma unroll
                        for (unsigned int j = 0; j < 8; j++) {
                            unsigned int gk = k0 + j;
                            float dequant_val = 0.0f;
                            if (gk < K) {
                                unsigned char packed_byte = pb[j >> 1];
                                unsigned int nibble =
                                    (gk & 1u) ? (packed_byte >> 4) : (packed_byte & 0xFu);
                                dequant_val = E2M1_LUT_MOE[nibble] * s;
                            }
                            smem_B[kl + j][n] = __float2bfloat16(dequant_val);
                        }
                    } else {
                        #pragma unroll
                        for (unsigned int j = 0; j < 8; j++) {
                            smem_B[kl + j][n] = __float2bfloat16(0.0f);
                        }
                    }
                }
            }
        }

        __syncthreads();

        // === MMA compute ===
        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;

        unsigned int frag_r0 = warp_m_offset + group_id;
        unsigned int frag_r1 = warp_m_offset + group_id + 8;
        unsigned int frag_c0 = tid * 2;
        unsigned int frag_c1 = tid * 2 + 8;

        unsigned int a0 = ((unsigned int)sA[frag_r0 * a_stride + frag_c0 + 1] << 16) |
                          (unsigned int)sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = ((unsigned int)sA[frag_r1 * a_stride + frag_c0 + 1] << 16) |
                          (unsigned int)sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = ((unsigned int)sA[frag_r0 * a_stride + frag_c1 + 1] << 16) |
                          (unsigned int)sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = ((unsigned int)sA[frag_r1 * a_stride + frag_c1 + 1] << 16) |
                          (unsigned int)sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            unsigned int n_col = n_tile * 8 + group_id;
            unsigned int k0 = tid * 2;
            unsigned int k1 = tid * 2 + 8;

            unsigned int b0 = ((unsigned int)sB[(k0 + 1) * b_stride + n_col] << 16) |
                              (unsigned int)sB[k0 * b_stride + n_col];
            unsigned int b1 = ((unsigned int)sB[(k1 + 1) * b_stride + n_col] << 16) |
                              (unsigned int)sB[k1 * b_stride + n_col];

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(acc[n_tile][0]), "=f"(acc[n_tile][1]),
                  "=f"(acc[n_tile][2]), "=f"(acc[n_tile][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(acc[n_tile][0]), "f"(acc[n_tile][1]),
                  "f"(acc[n_tile][2]), "f"(acc[n_tile][3])
            );
        }

        __syncthreads();
    }

    // === Store results ===
    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;
        // Bounds check: row must be within this expert's range AND within total output
        bool row0_valid = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool row1_valid = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;

        if (row0_valid && col0 < N) C[row0 * N + col0] = __float2bfloat16(acc[n_tile][0]);
        if (row0_valid && col1 < N) C[row0 * N + col1] = __float2bfloat16(acc[n_tile][1]);
        if (row1_valid && col0 < N) C[row1 * N + col0] = __float2bfloat16(acc[n_tile][2]);
        if (row1_valid && col1 < N) C[row1 * N + col1] = __float2bfloat16(acc[n_tile][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pointer-table variant with gather-from-input.
//
// Differences from above:
// 1. Per-expert weight pointers via device tables (not stacked buffer)
// 2. Gathers from original input via sorted_token_ids (no permute buffer)
// 3. Per-expert scale2 from device array (not uniform scalar)
//
// Grid: (ceil(N_out/N_TILE), max_m_tiles, num_experts)
// Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__ void moe_w4a16_grouped_gemm_ptrtable_impl(

    const __nv_bfloat16* __restrict__ A,           // [num_tokens, K] original (unpermuted)
    const unsigned long long* __restrict__ B_packed_ptrs, // [num_experts] → expert's B_packed
    const unsigned long long* __restrict__ B_scale_ptrs,  // [num_experts] → expert's B_scale
    const float* __restrict__ scale2_vals,         // [num_experts] per-expert scale2
    __nv_bfloat16* __restrict__ C,                  // [total_expanded, N_out] output
    const int* __restrict__ expert_offsets,          // [num_experts + 1] prefix sum
    const int* __restrict__ sorted_token_ids,       // [total_expanded] → original token index
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
,
    const bool relu2
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE_PT;

    // Per-expert weight pointers from device tables
    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    // EP: NULL pointer means remote expert — skip (output buffer already zeroed by caller)
    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP_PT + PAD];
    // B stays PACKED in shared memory and is dequantised into MMA REGISTERS.
    //
    // This is only correct because the 4 warps now split N (not M): each warp owns a
    // DISTINCT 16-column slice of B, so per-warp dequant duplicates nothing -- total
    // dequant work is identical, it just no longer round-trips through a BF16 smem
    // tile (~4.3 GB of smem traffic per GEMM). Dropping that tile takes the kernel
    // from 17.8 KB -> 12.7 KB of smem, i.e. ~20 -> ~28 warps/SM. Every previous
    // pipelining/staging attempt failed purely because it ADDED smem to an
    // occupancy-starved kernel; this one REMOVES it.
    // Double-buffered so cp.async can stream tile k+1 while tile k is dequantised and
    // fed to the MMAs. Even doubled this is only ~6 KB (it holds fp4 nibbles, not BF16),
    // so total smem is 15.7 KB -- still BELOW the 17.8 KB the un-pipelined kernel used.
    // Pipelining only became affordable once the BF16 dequant tile was deleted.
    __shared__ __align__(16) unsigned char smem_Bp[2][N_TILE_PT][BP_STRIDE];
    __shared__ unsigned int smem_Bs32[2][N_TILE_PT];   // 4 scale bytes/row, packed
    // Byte-indexed dequant LUT -> two BF16 nibbles packed in one word (a divergent
    // __constant__ lookup SERIALISES; shared memory is banked).
    __shared__ unsigned int sLUT[256];

    unsigned long long evict1;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(evict1));

    for (unsigned int i = threadIdx.x; i < 256u; i += blockDim.x) {
        const unsigned short lo =
            __bfloat16_as_ushort(__float2bfloat16(E2M1_LUT_MOE[i & 0xFu]));
        const unsigned short hi =
            __bfloat16_as_ushort(__float2bfloat16(E2M1_LUT_MOE[i >> 4]));
        sLUT[i] = ((unsigned int)hi << 16) | (unsigned int)lo;
    }

    float acc[16][4];   // 4 m-sub-tiles x 4 n-tiles (this warp's quarter of N_TILE_PT)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }
    __syncthreads();

    const unsigned int a_stride = K_STEP_PT + PAD;
    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;

    const unsigned int pf_n    = threadIdx.x >> 1;
    const unsigned int pf_half = threadIdx.x & 1u;
    const unsigned int pf_gn   = cta_n + pf_n;

#define MOE_PF(kb, buf)                                                                \
    do {                                                                               \
        _Pragma("unroll")                                                              \
        for (unsigned int ch = 0; ch < 2u; ch++) {                                     \
            const unsigned int idx  = threadIdx.x + ch * 128u;                         \
            const unsigned int n_   = idx >> 1;                                        \
            const unsigned int hf_  = idx & 1u;                                        \
            const unsigned int gn_  = cta_n + n_;                                      \
            if ((kb) < K && gn_ < N) {                                                 \
                const unsigned long long boff =                                        \
                    (unsigned long long)gn_ * half_K + ((kb) >> 1) + hf_ * 16u;        \
                moe_cp_async16(&smem_Bp[(buf)][n_][hf_ * 16u], B_expert + boff);       \
            } else {                                                                   \
                *(uint4*)&smem_Bp[(buf)][n_][hf_ * 16u] = make_uint4(0u,0u,0u,0u);     \
            }                                                                          \
        }                                                                              \
        {                                                                              \
            const unsigned int rr  = threadIdx.x;                                      \
            const unsigned int grn = cta_n + rr;                                       \
            const unsigned int sg  = (kb) / GROUP_SIZE;                                \
            unsigned int w = 0u;                                                       \
            for (unsigned int q = 0; q < K_STEP_PT / GROUP_SIZE; q++) {                \
                unsigned char sb = (grn < N && (kb) < K && (sg + q) < num_groups)      \
                    ? S_expert[(unsigned long long)grn * num_groups + sg + q]          \
                    : (unsigned char)0u;                                               \
                w |= ((unsigned int)sb) << (q * 8u);                                   \
            }                                                                          \
            smem_Bs32[(buf)][rr] = w;                                                  \
        }                                                                              \
    } while (0)

    MOE_PF(0u, 0u);
    moe_cp_async_commit();

    unsigned int buf = 0u;
    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP_PT, buf ^= 1u) {
        MOE_PF(k_base + K_STEP_PT, buf ^ 1u);   // stream the next tile...
        moe_cp_async_commit();
        moe_cp_async_wait<1>();                 // ...while we consume this one
        __syncthreads();



        // === Load A tile (gather via sorted_token_ids, or direct if NULL) ===
        {
            const unsigned int elems_per_thread = (M_TILE * K_STEP_PT) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int row = idx / K_STEP_PT;
                unsigned int col = idx % K_STEP_PT;
                unsigned int gc = k_base + col;
                bool valid = (cta_m_local + row) < M_eff && gc < K;
                if (valid) {
                    unsigned int a_row = sorted_token_ids
                        ? (unsigned int)sorted_token_ids[cta_m + row]
                        : (cta_m + row);
                    smem_A[row][col] = A[a_row * K + gc];
                } else {
                    smem_A[row][col] = __float2bfloat16(0.0f);
                }
            }
        }

        // === Load B tile: dequant FP4 → BF16 (N-major layout) ===
        {
        }

        // The cp.async wait at the top of the loop covers smem_Bp/smem_Bs32, but the A
        // tile is written by THIS iteration's threads just above -- it needs its own
        // barrier before the MMAs read it.
        __syncthreads();

        // === MMA compute: 4 warps split N (2 n-tiles each), all 4 m-sub-tiles ===
        // warp w owns n-tiles {2w, 2w+1} = 16 distinct B columns. B is dequantised
        // straight into the MMA b-fragments and reused across the 4 m-sub-tiles, so
        // each element is dequantised exactly once per block.
        {
            const unsigned short* sA = (const unsigned short*)smem_A;
            const unsigned int rows_in_tile = M_eff - (unsigned int)cta_m_local;

            #pragma unroll
            for (unsigned int ks = 0; ks < K_STEP_PT; ks += 16u) {
                const unsigned int sgi = ks / GROUP_SIZE;   // 0..3: one scale per k-step

                // --- B fragments for this warp's 4 n-tiles (registers) ---
                unsigned int b0[4], b1[4];
                #pragma unroll
                for (unsigned int nt = 0; nt < 4u; nt++) {
                    const unsigned int n_col = (warp_id * 4u + nt) * 8u + group_id;

                    __nv_fp8_e4m3 f8;
                    *(unsigned char*)&f8 =
                        (unsigned char)((smem_Bs32[buf][n_col] >> (sgi * 8u)) & 0xFFu);
                    const __nv_bfloat162 sv =
                        __bfloat162bfloat162(__float2bfloat16((float)f8 * scale2));

                    // One packed byte holds k (low nibble) and k+1 (high nibble), and
                    // the m16n8k16 b-fragment wants exactly {B[n][k], B[n][k+1]} with k
                    // even -- so one byte -> one 32-bit half of the fragment.
                    const unsigned int by0 = (ks >> 1) + tid;          // k  = ks + tid*2
                    const unsigned int by1 = (ks >> 1) + tid + 4u;     // k' = k + 8
                    const unsigned int p0 = sLUT[smem_Bp[buf][n_col][by0]];
                    const unsigned int p1 = sLUT[smem_Bp[buf][n_col][by1]];
                    __nv_bfloat162 v0 = *(const __nv_bfloat162*)&p0;
                    __nv_bfloat162 v1 = *(const __nv_bfloat162*)&p1;
                    v0 = __hmul2(v0, sv);
                    v1 = __hmul2(v1, sv);
                    b0[nt] = *(const unsigned int*)&v0;
                    b1[nt] = *(const unsigned int*)&v1;
                }

                // --- 4 m-sub-tiles against those fragments ---
                #pragma unroll
                for (unsigned int m = 0; m < 4u; m++) {
                    const unsigned int wm = m * 16u;
                    if (wm >= rows_in_tile) continue;   // whole sub-tile is padding

                    const unsigned int r0 = wm + group_id, r1 = wm + group_id + 8u;
                    const unsigned int c0 = ks + tid * 2u, c1 = ks + tid * 2u + 8u;
                    unsigned int a0 = ((unsigned int)sA[r0*a_stride + c0+1] << 16) | sA[r0*a_stride + c0];
                    unsigned int a1 = ((unsigned int)sA[r1*a_stride + c0+1] << 16) | sA[r1*a_stride + c0];
                    unsigned int a2 = ((unsigned int)sA[r0*a_stride + c1+1] << 16) | sA[r0*a_stride + c1];
                    unsigned int a3 = ((unsigned int)sA[r1*a_stride + c1+1] << 16) | sA[r1*a_stride + c1];

                    #pragma unroll
                    for (unsigned int nt = 0; nt < 4u; nt++) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                            : "=f"(acc[m*4+nt][0]), "=f"(acc[m*4+nt][1]),
                              "=f"(acc[m*4+nt][2]), "=f"(acc[m*4+nt][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(b0[nt]), "r"(b1[nt]),
                              "f"(acc[m*4+nt][0]), "f"(acc[m*4+nt][1]),
                              "f"(acc[m*4+nt][2]), "f"(acc[m*4+nt][3]));
                    }
                }
            }
        }

        __syncthreads();
    }

    // === Store results ===
    // acc[m*2+nt]: M index = m*16 + group_id (+8), N index = (2w+nt)*8 + tid*2 (+1).
    #pragma unroll
    for (unsigned int m = 0; m < 4u; m++) {
        const unsigned int wm = m * 16u;
        #pragma unroll
        for (unsigned int nt = 0; nt < 4u; nt++) {
            const unsigned int base_n = cta_n + (warp_id * 4u + nt) * 8u;
            const unsigned int col0 = base_n + tid * 2u;
            const unsigned int col1 = col0 + 1u;
            const unsigned int lr0 = wm + group_id;
            const unsigned int lr1 = lr0 + 8u;
            const unsigned int row0 = cta_m + lr0;
            const unsigned int row1 = cta_m + lr1;
            const bool v0 = (int)(lr0 + cta_m_local) < M_expert;
            const bool v1 = (int)(lr1 + cta_m_local) < M_expert;
            float o0 = acc[m*4+nt][0], o1 = acc[m*4+nt][1];
            float o2 = acc[m*4+nt][2], o3 = acc[m*4+nt][3];
            if (relu2) {
                // relu^2 epilogue in fp32 on the accumulator -- replaces the
                // elementwise relu_squared_inplace pass over the whole up_out
                // tensor (a full extra read+write of it from DRAM).
                o0 = o0 > 0.f ? o0 * o0 : 0.f;
                o1 = o1 > 0.f ? o1 * o1 : 0.f;
                o2 = o2 > 0.f ? o2 * o2 : 0.f;
                o3 = o3 > 0.f ? o3 * o3 : 0.f;
            }
            if (v0 && col0 < N) C[row0 * N + col0] = __float2bfloat16(o0);
            if (v0 && col1 < N) C[row0 * N + col1] = __float2bfloat16(o1);
            if (v1 && col0 < N) C[row1 * N + col0] = __float2bfloat16(o2);
            if (v1 && col1 < N) C[row1 * N + col1] = __float2bfloat16(o3);
        }
    }
#undef MOE_PF
}

extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable(

    const __nv_bfloat16* __restrict__ A,           // [num_tokens, K] original (unpermuted)
    const unsigned long long* __restrict__ B_packed_ptrs, // [num_experts] → expert's B_packed
    const unsigned long long* __restrict__ B_scale_ptrs,  // [num_experts] → expert's B_scale
    const float* __restrict__ scale2_vals,         // [num_experts] per-expert scale2
    __nv_bfloat16* __restrict__ C,                  // [total_expanded, N_out] output
    const int* __restrict__ expert_offsets,          // [num_experts + 1] prefix sum
    const int* __restrict__ sorted_token_ids,       // [total_expanded] → original token index
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    moe_w4a16_grouped_gemm_ptrtable_impl(A, B_packed_ptrs, B_scale_ptrs, scale2_vals, C, expert_offsets, sorted_token_ids, num_experts, N, K, false);
}

// Same GEMM with a fused relu^2 epilogue for the expert UP projection.
extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable_relu2(

    const __nv_bfloat16* __restrict__ A,           // [num_tokens, K] original (unpermuted)
    const unsigned long long* __restrict__ B_packed_ptrs, // [num_experts] → expert's B_packed
    const unsigned long long* __restrict__ B_scale_ptrs,  // [num_experts] → expert's B_scale
    const float* __restrict__ scale2_vals,         // [num_experts] per-expert scale2
    __nv_bfloat16* __restrict__ C,                  // [total_expanded, N_out] output
    const int* __restrict__ expert_offsets,          // [num_experts + 1] prefix sum
    const int* __restrict__ sorted_token_ids,       // [total_expanded] → original token index
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    moe_w4a16_grouped_gemm_ptrtable_impl(A, B_packed_ptrs, B_scale_ptrs, scale2_vals, C, expert_offsets, sorted_token_ids, num_experts, N, K, true);
}

// ═══════════════════════════════════════════════════════════════════
// Transposed-B variant: weights in [K/2, N] layout for coalesced reads.
//
// Same as moe_w4a16_grouped_gemm_ptrtable but B_packed is [K/2, N]
// and B_scale is [K/GROUP_SIZE, N]. Adjacent threads read consecutive
// N addresses → coalesced 128-byte cache lines on LPDDR5X.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE;

    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE + PAD;
    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        {
            const unsigned int elems_per_thread = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gc = k_base + col;
                bool valid = (cta_m_local + row) < M_eff && gc < K;
                if (valid) {
                    unsigned int a_row = sorted_token_ids
                        ? (unsigned int)sorted_token_ids[cta_m + row]
                        : (cta_m + row);
                    smem_A[row][col] = A[a_row * K + gc];
                } else {
                    smem_A[row][col] = __float2bfloat16(0.0f);
                }
            }
        }

        // Load B tile: transposed [K/2, N] layout — coalesced on N
        {
            const unsigned int elems_per_thread = (K_STEP * N_TILE) / 128;
            unsigned int scale_group = k_base / GROUP_SIZE;

            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int k = idx / N_TILE;
                unsigned int n = idx % N_TILE;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_expert[(unsigned long long)k_pair * N + gn];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);

                    unsigned char scale_byte = S_expert[(unsigned long long)scale_group * N + gn];
                    float fp8_val;
                    {
                        __nv_fp8_e4m3 fp8;
                        *(unsigned char*)&fp8 = scale_byte;
                        fp8_val = (float)fp8;
                    }

                    float dequant_val = E2M1_LUT_MOE[nibble] * fp8_val * scale2;
                    smem_B[k][n] = __float2bfloat16(dequant_val);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();

        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;

        unsigned int frag_r0 = warp_m_offset + group_id;
        unsigned int frag_r1 = warp_m_offset + group_id + 8;
        unsigned int frag_c0 = tid * 2;
        unsigned int frag_c1 = tid * 2 + 8;

        unsigned int a0 = ((unsigned int)sA[frag_r0 * a_stride + frag_c0 + 1] << 16) |
                          (unsigned int)sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = ((unsigned int)sA[frag_r1 * a_stride + frag_c0 + 1] << 16) |
                          (unsigned int)sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = ((unsigned int)sA[frag_r0 * a_stride + frag_c1 + 1] << 16) |
                          (unsigned int)sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = ((unsigned int)sA[frag_r1 * a_stride + frag_c1 + 1] << 16) |
                          (unsigned int)sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            unsigned int n_col = n_tile * 8 + group_id;
            unsigned int k0 = tid * 2;
            unsigned int k1 = tid * 2 + 8;

            unsigned int b0 = ((unsigned int)sB[(k0 + 1) * b_stride + n_col] << 16) |
                              (unsigned int)sB[k0 * b_stride + n_col];
            unsigned int b1 = ((unsigned int)sB[(k1 + 1) * b_stride + n_col] << 16) |
                              (unsigned int)sB[k1 * b_stride + n_col];

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(acc[n_tile][0]), "=f"(acc[n_tile][1]),
                  "=f"(acc[n_tile][2]), "=f"(acc[n_tile][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(acc[n_tile][0]), "f"(acc[n_tile][1]),
                  "f"(acc[n_tile][2]), "f"(acc[n_tile][3])
            );
        }

        __syncthreads();
    }

    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;
        bool row0_valid = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool row1_valid = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;

        if (row0_valid && col0 < N) C[row0 * N + col0] = __float2bfloat16(acc[n_tile][0]);
        if (row0_valid && col1 < N) C[row0 * N + col1] = __float2bfloat16(acc[n_tile][1]);
        if (row1_valid && col0 < N) C[row1 * N + col0] = __float2bfloat16(acc[n_tile][2]);
        if (row1_valid && col1 < N) C[row1 * N + col1] = __float2bfloat16(acc[n_tile][3]);
    }
}
