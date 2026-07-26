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
