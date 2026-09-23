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
#define PAD 2
#define GROUP_SIZE 16

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
                    // N-major layout: B_packed[gn, gk/2], nibble = gk & 1
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_expert[(unsigned long long)gn * half_K + k_pair];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);

                    // N-major scale: B_scale[gn, scale_group]
                    unsigned char scale_byte = S_expert[(unsigned long long)gn * num_groups + scale_group];
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
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        // === Load A tile (gather via sorted_token_ids, or direct if NULL) ===
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

        // === Load B tile: dequant FP4 → BF16 (N-major layout) ===
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
                    // N-major layout: B_packed[gn, gk/2], nibble = gk & 1
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_expert[(unsigned long long)gn * half_K + k_pair];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);

                    // N-major scale: B_scale[gn, scale_group]
                    unsigned char scale_byte = S_expert[(unsigned long long)gn * num_groups + scale_group];
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
        bool row0_valid = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool row1_valid = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;

        if (row0_valid && col0 < N) C[row0 * N + col0] = __float2bfloat16(acc[n_tile][0]);
        if (row0_valid && col1 < N) C[row0 * N + col1] = __float2bfloat16(acc[n_tile][1]);
        if (row1_valid && col0 < N) C[row1 * N + col0] = __float2bfloat16(acc[n_tile][2]);
        if (row1_valid && col1 < N) C[row1 * N + col1] = __float2bfloat16(acc[n_tile][3]);
    }
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

// ═══════════════════════════════════════════════════════════════════
// WS1/P3b — TILE-GEOMETRY VARIANTS OF `moe_w4a16_grouped_gemm_ptrtable`.
//
// ADDITIVE ON PURPOSE. `moe_w4a16_grouped_gemm_ptrtable` above is untouched, so every
// model that resolves this shared file keeps launching the exact kernel it launched
// before and stays byte-identical. Only a caller that asks for one of the new names by
// string reaches the new code.
//
// 🔴 WHY. MEASURED on GLM-5.3-Flash NVFP4, 2 x GB10, TP=2/EP=2, rows=256 prefill
// (`runs/ws1-p2-prefill-n1n2/PROFILE.md`): the base kernel runs 10.177 ms per launch
// against 679.5 MB of expert weight = **66.8 GB/s, 24.5 % of the 273 GB/s GB10 roofline**,
// and is 64.04 % of the prefill window. It is neither compute-bound (15.2 TFLOP/s padded
// against a ~250 TFLOP/s BF16 peak) nor short of weight traffic to save — `max_m_tiles` is
// already 1, so each expert's weights are read exactly ONCE per projection. What is left
// is the SHAPE of the read and how much of the machine is in flight while it happens.
//
// Three mechanisms, none of which changes a single arithmetic operand:
//
//   1. **Load redundancy and burst length.** The base B loop is element-at-a-time: it
//      re-fetches the SAME packed byte twice (it holds nibbles `gk` and `gk+1`) and the
//      SAME E4M3 scale byte 16 times (one scale covers a GROUP_SIZE-wide run), and each
//      thread's global read is one byte out of a stream whose neighbouring `n` is `K/2`
//      bytes away. Here one thread owns a whole GROUP_SIZE-wide k run for one `n`, so 8
//      contiguous packed bytes arrive as two `uint32` and the scale is read once for all
//      16 values. This exact transform is `moe_w4a16_grouped_gemm_ptrtable_k32` in
//      `kernels/gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu`, where it measured
//      **82.8 -> 65.4 ms (1.27x), bit-exact**. Replicated, not re-derived.
//
//   2. **Padding warps.** `M_TILE = 64` over GLM's 288-expert / top-8 routing means 2,048
//      routed slots spread ~7.1 rows per expert, so warps 1..3 of a 4-warp CTA compute 48
//      rows that are ENTIRELY out of range and whose results the store already discards.
//      `SPLIT_N = false` skips their `mma` (they still load and still hit every barrier);
//      `SPLIT_N = true` makes `M_TILE = 16` and has the warps split the N tile instead,
//      which removes the padding rather than skipping it.
//
//   3. **Occupancy.** 🔴 MEASURED in `examples/glm5next_moe_grouped_tile_bench.rs`, and the
//      finding that decided the geometry: this kernel is LATENCY-bound, so staging more k
//      per round trip stops paying as soon as the shared-memory footprint costs a resident
//      CTA. At `M_TILE = 64` the sweep went 42.3 GB/s (K16) -> 56.8 (K32) -> 58.0 (K64) ->
//      **45.4 (K128)**: K128's 33.5 KB of smem is the whole regression. The winning shape
//      is the one that is simultaneously wide enough to burst and small enough to keep
//      CTAs resident. Anyone who reads mechanism 1 alone and pushes K_STEP up will land on
//      K128 and lose.
//
// 🔴 BIT-EXACTNESS. Every variant stages the identical BF16 value into shared memory (same
// `E2M1_LUT_MOE[nibble] * (float)e4m3 * scale2`, same multiply order) and consumes the
// staged tile through the identical `mma.sync.aligned.m16n8k16` chain in the identical k
// order — the `kf` loop walks the wider tile in 16-wide fragments, so the FP32 accumulator
// sees the same sequence of the same products it saw at `K_STEP = 16`. Which warp owns an
// output element changes; the arithmetic that produces it does not. Every variant is
// therefore BYTE-IDENTICAL to the base kernel, and the microbench asserts exactly that
// rather than an error bar.
//
// 🪤 `max_m_tiles` is counted in the variant's OWN `M_TILE`, and grid.x in its OWN `N_TILE`.
// An `m16` kernel launched with a grid height computed against 64 silently drops every row
// past the first 16 of any expert; an `n128` kernel launched with `ceil(N/64)` computes the
// left half of the output twice and never writes the right half.
// ═══════════════════════════════════════════════════════════════════

// `MT`      — rows per CTA tile. `WARPS*16` when the warps split M, 16 when they split N.
// `NTILE`   — output columns per CTA. grid.x is counted in these.
// `KS`      — k staged per shared-memory round trip. Multiple of GROUP_SIZE, and both
//             `MT*KS` and `(KS/GROUP_SIZE)*NTILE` must divide the block's thread count.
// `WARPS`   — warps per CTA; the block is `WARPS*32` threads.
// `SPLIT_N` — false: each warp takes 16 rows of M and the whole N tile.
//             true:  every warp takes the same 16 rows and `NTILE/WARPS` columns.
// E2M1 decode with NO memory reference.
//
// 🔴 `E2M1_LUT_MOE` lives in `__constant__`, and a constant-memory read BROADCASTS: it
// serves one address per replay, so a warp whose 32 lanes hold up to 16 distinct nibbles
// costs up to 16 replays for ONE lookup — and this kernel does one lookup per weight
// element, i.e. 1.2e9 of them per launch on GLM's shape. The value is a pure function of
// the nibble, so it can be built in the integer pipe instead:
//   idx = n & 7 -> {0, 0.5, 1, 1.5, 2, 3, 4, 6}: exponent field 126 + (idx >> 1), mantissa
//   bit 22 set for an ODD idx — but only from idx >= 2. `idx == 1` is 0.5, a power of two
//   with an EMPTY significand, and the SUBNORMAL-shaped code in the E2M1 encoding; a
//   formula that sets its mantissa bit returns 0.75. 🪤 The first cut of this function did
//   exactly that and the microbench's byte-identity check caught it on output element 0 —
//   which is what that check is for. `idx == 0` keeps a zero significand so that nibble 8
//   still yields -0.0f exactly as the table does (the sign bit is OR'd in unconditionally).
// Produces the identical FP32 bit pattern for all 16 codes.
__device__ __forceinline__ float e2m1_decode(unsigned int n) {
    unsigned int idx = n & 7u;
    unsigned int mant = (idx >= 2u) ? ((idx & 1u) << 22) : 0u;
    unsigned int bits = (idx == 0u) ? 0u : (((126u + (idx >> 1)) << 23) | mant);
    bits |= (n & 8u) << 28;
    return __int_as_float(bits);
}

template <int MT, int NTILE, int KS, int WARPS, bool SPLIT_N, bool KMAJOR, bool ARITH_LUT,
          bool BT>
__device__ __forceinline__ void moe_w4a16_grouped_core(
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
    constexpr int THREADS = WARPS * 32;
    constexpr int NT = SPLIT_N ? (NTILE / WARPS / 8) : (NTILE / 8);
    static_assert(SPLIT_N ? (MT == 16) : (MT == WARPS * 16), "MT must match the warp split");
    static_assert(NT >= 1, "each warp needs at least one 8-wide n subtile");
    static_assert((MT * KS) % THREADS == 0, "A tile must divide evenly across the block");
    static_assert(((KS / GROUP_SIZE) * NTILE) % THREADS == 0,
                  "B scale-groups must divide evenly across the block");
    static_assert(KS % GROUP_SIZE == 0, "KS must be a whole number of scale groups");

    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * MT;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * NTILE;

    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    // EP: NULL pointer means remote expert — skip (output buffer already zeroed by caller).
    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    const unsigned int warp_m_offset = SPLIT_N ? 0u : warp_id * 16u;
    const unsigned int warp_n_offset = SPLIT_N ? warp_id * (NTILE / WARPS) : 0u;

    // 🔴 The staged B tile, in one of two layouts.
    //
    // `BT = false` is the base kernel's `[KS][NTILE+PAD]`: a thread that owns a
    // GROUP_SIZE-wide k run for one `n` writes its 16 values `B_STRIDE` apart, i.e. 16
    // separate 2-byte shared stores, and the mma's `b0`/`b1` each need two more 2-byte
    // reads because k and k+1 are a whole row apart.
    //
    // `BT = true` transposes it to `[NTILE][KS+8]`: those same 16 values become 32
    // CONTIGUOUS bytes — two 16-byte stores — and `b0` becomes ONE aligned 32-bit read
    // because k and k+1 are now adjacent. Same values, same slots, same order; only the
    // address arithmetic changes. Padding is 8 (not PAD=2) so each row stays 16-byte
    // aligned: `(KS+8)*2 % 16 == 0` for every KS used here.
    constexpr int B_STRIDE = BT ? (KS + 8) : (NTILE + PAD);
    constexpr int B_ELEMS = BT ? (NTILE * B_STRIDE) : (KS * B_STRIDE);
    __shared__ __nv_bfloat16 smem_A[MT][KS + PAD];
    __shared__ __align__(16) __nv_bfloat16 smem_B[B_ELEMS];

    float acc[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = KS + PAD;
    const unsigned int b_stride = (unsigned int)B_STRIDE;
    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;

    // 🪤 This warp's 16 rows may be entirely past the expert's real row count. Its
    // accumulators are discarded by the store's own bound test either way, so skipping the
    // mma is a pure issue-slot saving with no result change. It must NOT skip the loads or
    // the barriers — the tile is loaded cooperatively by the whole block.
    const bool warp_has_rows = (cta_m_local + (int)warp_m_offset) < M_expert;

    for (unsigned int k_base = 0; k_base < K; k_base += KS) {
        // === Load A tile (gather via sorted_token_ids, or direct if NULL) ===
        {
            constexpr unsigned int ept = (unsigned int)((MT * KS) / THREADS);
            #pragma unroll
            for (unsigned int i = 0; i < ept; i++) {
                unsigned int idx = threadIdx.x * ept + i;
                unsigned int row = idx / KS;
                unsigned int col = idx % KS;
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

        // === Load B tile: one thread owns a whole GROUP_SIZE-wide k run for one n ===
        {
            constexpr unsigned int gpt =
                (unsigned int)(((KS / GROUP_SIZE) * NTILE) / THREADS);
            #pragma unroll
            for (unsigned int i = 0; i < gpt; i++) {
                unsigned int g = threadIdx.x * gpt + i;
                // 🔴 THE COALESCING AXIS, and the second-biggest lever in this file.
                // N-major (`KMAJOR = false`, what the base kernel does): consecutive
                // threads take consecutive `n`, whose packed bytes are `K/2` apart, so a
                // warp's 32 reads land in 32 DISTINCT 32-byte sectors — one transaction per
                // thread however few bytes it wants. K-major: consecutive threads take
                // consecutive k GROUPS of the SAME `n`, so `GROUP_SIZE * (KS/GROUP_SIZE)`
                // bytes of one weight row are fetched by adjacent lanes and the warp's
                // requests coalesce into whole sectors. Same bytes, same smem slots, same
                // values — only which lane fetches which byte changes.
                constexpr unsigned int KG = (unsigned int)(KS / GROUP_SIZE);
                unsigned int kg = KMAJOR ? ((g % KG) * GROUP_SIZE) : ((g / NTILE) * GROUP_SIZE);
                unsigned int n  = KMAJOR ? (g / KG) : (g % NTILE);
                unsigned int gk = k_base + kg;
                unsigned int gn = cta_n + n;
                if (gk < K && gn < N) {
                    const unsigned char* bp =
                        B_expert + (unsigned long long)gn * half_K + (gk / 2);
                    unsigned char sb =
                        S_expert[(unsigned long long)gn * num_groups + (gk / GROUP_SIZE)];
                    __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
                    float sc = (float)fp8 * scale2;
                    unsigned int w0 = *(const unsigned int*)(bp);
                    unsigned int w1 = *(const unsigned int*)(bp + 4);
                    // 🪤 16 BF16 = 32 bytes. Staged in registers first so the BT store is
                    // two 16-byte writes; `smem_B + n*B_STRIDE + kg` is 16-byte aligned by
                    // the padding choice above.
                    __nv_bfloat16* st = smem_B + (unsigned int)(n * B_STRIDE) + kg;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        unsigned char c0 = (unsigned char)((w0 >> (j * 8)) & 0xFF);
                        unsigned char c1 = (unsigned char)((w1 >> (j * 8)) & 0xFF);
                        float v0 = ARITH_LUT ? e2m1_decode(c0 & 0xF) : E2M1_LUT_MOE[c0 & 0xF];
                        float v1 = ARITH_LUT ? e2m1_decode(c0 >> 4)   : E2M1_LUT_MOE[c0 >> 4];
                        float v2 = ARITH_LUT ? e2m1_decode(c1 & 0xF) : E2M1_LUT_MOE[c1 & 0xF];
                        float v3 = ARITH_LUT ? e2m1_decode(c1 >> 4)   : E2M1_LUT_MOE[c1 >> 4];
                        if (BT) {
                            st[j * 2]         = __float2bfloat16(v0 * sc);
                            st[j * 2 + 1]     = __float2bfloat16(v1 * sc);
                            st[8 + j * 2]     = __float2bfloat16(v2 * sc);
                            st[8 + j * 2 + 1] = __float2bfloat16(v3 * sc);
                        } else {
                            smem_B[(kg + j * 2) * B_STRIDE + n]         = __float2bfloat16(v0 * sc);
                            smem_B[(kg + j * 2 + 1) * B_STRIDE + n]     = __float2bfloat16(v1 * sc);
                            smem_B[(kg + 8 + j * 2) * B_STRIDE + n]     = __float2bfloat16(v2 * sc);
                            smem_B[(kg + 8 + j * 2 + 1) * B_STRIDE + n] = __float2bfloat16(v3 * sc);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < GROUP_SIZE; j++) {
                        if (BT) smem_B[(unsigned int)(n * B_STRIDE) + kg + j] = __float2bfloat16(0.0f);
                        else smem_B[(kg + j) * B_STRIDE + n] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        __syncthreads();

        if (warp_has_rows) {
            const unsigned short* sA = (const unsigned short*)smem_A;
            const unsigned short* sB = (const unsigned short*)smem_B;
            unsigned int fr0 = warp_m_offset + group_id;
            unsigned int fr1 = fr0 + 8;

            // The wider staged tile holds KS/16 mma fragments; consume them in the SAME
            // order the 16-wide base kernel did, so the FP32 accumulation order — and
            // therefore the result — is unchanged.
            #pragma unroll
            for (unsigned int kf = 0; kf < (unsigned int)KS; kf += 16) {
                unsigned int fc0 = kf + tid * 2, fc1 = fc0 + 8;
                unsigned int a0 = ((unsigned int)sA[fr0 * a_stride + fc0 + 1] << 16) |
                                  (unsigned int)sA[fr0 * a_stride + fc0];
                unsigned int a1 = ((unsigned int)sA[fr1 * a_stride + fc0 + 1] << 16) |
                                  (unsigned int)sA[fr1 * a_stride + fc0];
                unsigned int a2 = ((unsigned int)sA[fr0 * a_stride + fc1 + 1] << 16) |
                                  (unsigned int)sA[fr0 * a_stride + fc1];
                unsigned int a3 = ((unsigned int)sA[fr1 * a_stride + fc1 + 1] << 16) |
                                  (unsigned int)sA[fr1 * a_stride + fc1];

                #pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    unsigned int nc = warp_n_offset + nt * 8 + group_id;
                    unsigned int k0 = kf + tid * 2, k1 = k0 + 8;
                    unsigned int b0, b1;
                    if (BT) {
                        // k and k+1 adjacent: one aligned 32-bit read each.
                        const unsigned int* r = (const unsigned int*)(sB + nc * b_stride);
                        b0 = r[k0 >> 1];
                        b1 = r[k1 >> 1];
                    } else {
                        b0 = ((unsigned int)sB[(k0 + 1) * b_stride + nc] << 16) |
                             (unsigned int)sB[k0 * b_stride + nc];
                        b1 = ((unsigned int)sB[(k1 + 1) * b_stride + nc] << 16) |
                             (unsigned int)sB[k1 * b_stride + nc];
                    }
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                          "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]));
                }
            }
        }

        __syncthreads();
    }

    if (!warp_has_rows) return;

    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        unsigned int c0 = cta_n + warp_n_offset + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc[nt][3]);
    }
}

#define P3B_GROUPED_VARIANT(SUFFIX, MT, NTILE, KS, WARPS, SPLIT_N, KMAJOR, ALUT, BT) \
extern "C" __global__ __launch_bounds__((WARPS) * 32)                         \
void moe_w4a16_grouped_gemm_ptrtable_##SUFFIX(                                \
    const __nv_bfloat16* __restrict__ A,                                      \
    const unsigned long long* __restrict__ B_packed_ptrs,                     \
    const unsigned long long* __restrict__ B_scale_ptrs,                      \
    const float* __restrict__ scale2_vals,                                    \
    __nv_bfloat16* __restrict__ C,                                            \
    const int* __restrict__ expert_offsets,                                   \
    const int* __restrict__ sorted_token_ids,                                 \
    unsigned int num_experts,                                                 \
    unsigned int N,                                                           \
    unsigned int K                                                            \
) {                                                                           \
    moe_w4a16_grouped_core<MT, NTILE, KS, WARPS, SPLIT_N, KMAJOR, ALUT, BT>(  \
        A, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,                       \
        expert_offsets, sorted_token_ids, num_experts, N, K);                 \
}

//                    suffix        MT  NTILE  KS  WARPS  SPLIT_N  KMAJOR
// Base geometry (M_TILE 64, 4 warps over M), wider staging only.
P3B_GROUPED_VARIANT(k32,            64,    64,  32,    4, false, false, false, false)
P3B_GROUPED_VARIANT(k64,            64,    64,  64,    4, false, false, false, false)
P3B_GROUPED_VARIANT(k128,           64,    64, 128,    4, false, false, false, false)
// M_TILE 16 — four warps split the 64-wide N tile. 🪤 max_m_tiles counted in 16s.
P3B_GROUPED_VARIANT(m16_k32,        16,    64,  32,    4, true,  false, false, false)
P3B_GROUPED_VARIANT(m16_k64,        16,    64,  64,    4, true,  false, false, false)
P3B_GROUPED_VARIANT(m16_k128,       16,    64, 128,    4, true,  false, false, false)
P3B_GROUPED_VARIANT(m16_k256,       16,    64, 256,    4, true,  false, false, false)
// M_TILE 16, N_TILE 128, eight warps. 🪤 grid.x counted in 128s.
P3B_GROUPED_VARIANT(m16_n128_k32,   16,   128,  32,    8, true,  false, false, false)
P3B_GROUPED_VARIANT(m16_n128_k64,   16,   128,  64,    8, true,  false, false, false)
P3B_GROUPED_VARIANT(m16_n128_k128,  16,   128, 128,    8, true,  false, false, false)
// K-MAJOR B fetch — the coalesced twin of each shape above.
P3B_GROUPED_VARIANT(km_k64,         64,    64,  64,    4, false, true, false, false)
P3B_GROUPED_VARIANT(km_m16_k32,     16,    64,  32,    4, true,  true, false, false)
P3B_GROUPED_VARIANT(km_m16_k64,     16,    64,  64,    4, true,  true, false, false)
P3B_GROUPED_VARIANT(km_m16_k128,    16,    64, 128,    4, true,  true, false, false)
P3B_GROUPED_VARIANT(km_m16_n128_k64, 16,  128,  64,    8, true,  true, false, false)
P3B_GROUPED_VARIANT(km_m16_n128_k128,16,  128, 128,    8, true,  true, false, false)

// ═══════════════════════════════════════════════════════════════════
// WS1/P3b CONTROL — pure streaming read of exactly the bytes the grouped GEMM reads.
//
// Not a production kernel. It exists so "% of the 273 GB/s roofline" is a MEASURED
// denominator on this part, this allocation and this pointer-table shape, instead of a
// datasheet number. It walks the same per-expert `B_packed` and `B_scale` buffers through
// the same `[num_experts]` pointer tables, with perfectly coalesced `uint4` loads and no
// dequant, no shared memory, no barriers and no mma — the arithmetic is one XOR so the
// loads cannot be eliminated. Whatever this reaches is the ceiling any variant of the GEMM
// can be held to; the gap between it and 273 GB/s belongs to the memory system, not to the
// tile geometry.
//
// Launch: grid (ceil(bytes_per_expert / (BLOCK*16*UNROLL)), 1, num_experts), block 256.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ __launch_bounds__(256)
void moe_w4a16_grouped_stream_probe(
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    unsigned int* __restrict__ sink,
    unsigned int num_experts,
    unsigned int packed_bytes,   // N * K / 2 for this projection
    unsigned int scale_bytes     // N * K / GROUP_SIZE
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;
    const uint4* bp = (const uint4*)B_packed_ptrs[expert_id];
    const uint4* bs = (const uint4*)B_scale_ptrs[expert_id];
    if (bp == 0) return;

    const unsigned int p_vec = packed_bytes / 16;
    const unsigned int s_vec = scale_bytes / 16;
    const unsigned int stride = gridDim.x * blockDim.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint4 acc = make_uint4(0, 0, 0, 0);
    for (unsigned int i = idx; i < p_vec; i += stride) {
        uint4 v = bp[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    for (unsigned int i = idx; i < s_vec; i += stride) {
        uint4 v = bs[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    unsigned int r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (r == 0xFFFFFFFFu) sink[expert_id] = r;  // never true for real data; keeps the loads
}

// ARITHMETIC E2M1 DECODE — the same shapes again with the constant-memory table removed.
P3B_GROUPED_VARIANT(al_k64,          64,    64,  64,    4, false, false, true, false)
P3B_GROUPED_VARIANT(al_m16_k32,      16,    64,  32,    4, true,  false, true, false)
P3B_GROUPED_VARIANT(al_m16_k64,      16,    64,  64,    4, true,  false, true, false)
P3B_GROUPED_VARIANT(al_m16_k128,     16,    64, 128,    4, true,  false, true, false)
P3B_GROUPED_VARIANT(al_m16_n128_k64, 16,   128,  64,    8, true,  false, true, false)
P3B_GROUPED_VARIANT(alkm_m16_k32,    16,    64,  32,    4, true,  true,  true, false)
P3B_GROUPED_VARIANT(alkm_m16_k64,    16,    64,  64,    4, true,  true,  true, false)
P3B_GROUPED_VARIANT(alkm_m16_k128,   16,    64, 128,    4, true,  true,  true, false)
P3B_GROUPED_VARIANT(alkm_m16_k256,   16,    64, 256,    4, true,  true,  true, false)
P3B_GROUPED_VARIANT(alkm_m16_n128_k64,  16, 128,  64,    8, true,  true,  true, false)
P3B_GROUPED_VARIANT(alkm_m16_n128_k128, 16, 128, 128,    8, true,  true,  true, false)
P3B_GROUPED_VARIANT(alkm_k64,        64,    64,  64,    4, false, true,  true, false)
P3B_GROUPED_VARIANT(alkm_k128,       64,    64, 128,    4, false, true,  true, false)

// TRANSPOSED STAGED B TILE — the winner's shape with vectorised smem traffic.
P3B_GROUPED_VARIANT(bt_m16_k64,      16,    64,  64,    4, true,  true,  true, true)
P3B_GROUPED_VARIANT(bt_m16_k128,     16,    64, 128,    4, true,  true,  true, true)
P3B_GROUPED_VARIANT(bt_m16_k256,     16,    64, 256,    4, true,  true,  true, true)
P3B_GROUPED_VARIANT(bt_m16_n128_k128, 16,  128, 128,    8, true,  true,  true, true)
P3B_GROUPED_VARIANT(bt_k128,         64,    64, 128,    4, false, true,  true, true)
