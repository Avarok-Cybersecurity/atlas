// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Dense GEMM kernel for SM121 (GB10).
//
// C = A * B^T  where:
//   A: [M, K] BF16 (activations, row-major)
//   B: [N, K] BF16 (weights, row-major — standard HuggingFace layout)
//   C: [M, N] BF16 (output, row-major)
//
// The kernel reads B transposed: B^T[k,n] = B[n,k] = B[n*K + k].
//
// Phase 1: Correct scalar implementation with shared memory tiling.
// Phase 2: Will add mma.sync.aligned.m16n8k16 BF16 tensor cores.

#include <cuda_bf16.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// Tiled GEMM: C[M,N] = A[M,K] * B[N,K]^T
// All matrices in BF16, accumulation in FP32.
//
// Grid: (ceil(N/TILE_N), ceil(M/TILE_M))
// Block: (TILE_N, TILE_M) — each thread computes one output element
extern "C" __global__ void dense_gemm_bf16(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major (read transposed)
    __nv_bfloat16* __restrict__ C,         // [M, N] row-major
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    // Each thread computes one element of C
    unsigned int row = blockIdx.y * TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * TILE_N + threadIdx.x;

    // Shared memory tiles
    __shared__ __nv_bfloat16 smem_A[TILE_M][TILE_K];
    __shared__ __nv_bfloat16 smem_B[TILE_K][TILE_N];

    float acc = 0.0f;

    // Loop over K in TILE_K chunks
    for (unsigned int k_base = 0; k_base < K; k_base += TILE_K) {
        // Load A tile
        if (row < M && (k_base + threadIdx.x) < K) {
            smem_A[threadIdx.y][threadIdx.x] = A[row * K + k_base + threadIdx.x];
        } else {
            smem_A[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        // Load B tile (B is [N,K] row-major, read as B^T[K,N])
        if ((k_base + threadIdx.y) < K && col < N) {
            smem_B[threadIdx.y][threadIdx.x] = B[(unsigned long long)col * K + (k_base + threadIdx.y)];
        } else {
            smem_B[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        __syncthreads();

        // Compute partial dot product
        for (unsigned int kk = 0; kk < TILE_K; kk++) {
            acc += __bfloat162float(smem_A[threadIdx.y][kk])
                 * __bfloat162float(smem_B[kk][threadIdx.x]);
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(acc);
    }
}

// FP32-output twin of dense_gemm_bf16 (see gb10 source for rationale): writes
// the FP32 accumulator directly so the MoE router gate logits keep full
// precision into top-K under ATLAS_FP32_GATE. Same scalar math as the BF16
// kernel above; only the store dtype differs. Inputs stay BF16.
//
// Grid: (ceil(N/TILE_N), ceil(M/TILE_M))   Block: (TILE_N, TILE_M)
extern "C" __global__ void dense_gemm_bf16_f32out(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major (read transposed)
    float* __restrict__ C,                 // [M, N] row-major, FP32
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.y * TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ __nv_bfloat16 smem_A[TILE_M][TILE_K];
    __shared__ __nv_bfloat16 smem_B[TILE_K][TILE_N];

    float acc = 0.0f;

    for (unsigned int k_base = 0; k_base < K; k_base += TILE_K) {
        if (row < M && (k_base + threadIdx.x) < K) {
            smem_A[threadIdx.y][threadIdx.x] = A[row * K + k_base + threadIdx.x];
        } else {
            smem_A[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }
        if ((k_base + threadIdx.y) < K && col < N) {
            smem_B[threadIdx.y][threadIdx.x] = B[(unsigned long long)col * K + (k_base + threadIdx.y)];
        } else {
            smem_B[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }
        __syncthreads();
        for (unsigned int kk = 0; kk < TILE_K; kk++) {
            acc += __bfloat162float(smem_A[threadIdx.y][kk])
                 * __bfloat162float(smem_B[kk][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// FP32-input, FP32-output variant (see gb10 source): A = FP32 router_in from
// residual_add_rms_norm_gatef32, B = BF16 gate weight, C = FP32 gate logits.
// ATLAS_FP32_ROUTING path — unrounded gate logits so top-K doesn't flip on a
// bf16 store. Same scalar math as dense_gemm_bf16_f32out; only A's dtype differs.
//
// Grid: (ceil(N/TILE_N), ceil(M/TILE_M))   Block: (TILE_N, TILE_M)
extern "C" __global__ void dense_gemm_f32in_f32out(
    const float* __restrict__ A,          // [M, K] row-major, FP32
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major (read transposed), BF16
    float* __restrict__ C,                 // [M, N] row-major, FP32
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.y * TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ float smem_A[TILE_M][TILE_K];
    __shared__ __nv_bfloat16 smem_B[TILE_K][TILE_N];

    float acc = 0.0f;

    for (unsigned int k_base = 0; k_base < K; k_base += TILE_K) {
        if (row < M && (k_base + threadIdx.x) < K) {
            smem_A[threadIdx.y][threadIdx.x] = A[row * K + k_base + threadIdx.x];
        } else {
            smem_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if ((k_base + threadIdx.y) < K && col < N) {
            smem_B[threadIdx.y][threadIdx.x] = B[(unsigned long long)col * K + (k_base + threadIdx.y)];
        } else {
            smem_B[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }
        __syncthreads();
        for (unsigned int kk = 0; kk < TILE_K; kk++) {
            acc += smem_A[threadIdx.y][kk] * __bfloat162float(smem_B[kk][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ── HIP (gfx1151 / RDNA3.5) port of dense_gemm_bf16_pipelined ──────────────
//
// The gb10 source ships a tensor-core variant (mma.sync.m16n8k16 + a 2-stage
// cp.async prefetch pipeline). Native HIP/clang cannot lower either NVIDIA
// inline PTX construct, so this target supplies a same-contract kernel: same
// name, same (A,B,C,M,N,K) signature, and the SAME launch geometry the op
// wrapper in gemm_dense.rs uses — Grid (ceil(N/128), ceil(M/128), 1),
// Block (256,1,1). The math is FP32 accumulation of the same BF16 products
// (cosine=1.0 vs the gb10 path); only the GEMM micro-architecture differs.
//
// AMD WMMA port: replaces the scalar 8×8 register-FMA inner loop with
// __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 tensor cores + a register-
// prefetch double-buffered software pipeline (gfx1151 has no cp.async, so
// tile-(k+1) global reads are staged in VGPRs across tile-k WMMA, then
// committed to the other smem buffer; single barrier per K-step). Adapts the
// +131% w8a16_gemm recipe (597ef2f) to plain BF16×BF16.
//
//   256 threads = 8 warps; warp w owns M-rows [w*16, w*16+16) of the 128×128
//   tile and all 8 WMMA n-sub-tiles (8×16 = 128 N). K stepped 16 at a time.
//   LDS: smem_A[2][128][16+2] + smem_B[2][16][128+2] bf16
//      = 2*128*18*2 + 2*16*130*2 = 9216 + 8320 = 17536 B/CTA (well < 64 KB).
//   Per-thread prefetch regs: A 8 elems, B 8 elems (128*16/256 = 8 each).
#define DP_M_TILE 128
#define DP_N_TILE 128
#define DP_K_STEP 16
#define DP_THREADS 256
#define DP_PAD 2
#define DP_NSUB (DP_N_TILE / 16)                         // 8
#define DP_A_EPT ((DP_M_TILE * DP_K_STEP) / DP_THREADS)  // 8
#define DP_B_EPT ((DP_K_STEP * DP_N_TILE) / DP_THREADS)  // 8

typedef __bf16 dp_v16bf __attribute__((ext_vector_type(16)));
typedef float  dp_v8f   __attribute__((ext_vector_type(8)));

__device__ __forceinline__ void dp_load_A_regs(
    const __nv_bfloat16* __restrict__ A, __nv_bfloat16 reg_A[DP_A_EPT],
    unsigned int cta_m, unsigned int k_base, unsigned int M, unsigned int K
) {
    #pragma unroll
    for (unsigned int i = 0; i < DP_A_EPT; i++) {
        unsigned int idx = threadIdx.x * DP_A_EPT + i;
        unsigned int row = idx / DP_K_STEP, col = idx % DP_K_STEP;
        unsigned int gr = cta_m + row, gc = k_base + col;
        reg_A[i] = (gr < M && gc < K) ? A[(unsigned long long)gr * K + gc] : __float2bfloat16(0.0f);
    }
}

__device__ __forceinline__ void dp_load_B_regs(
    const __nv_bfloat16* __restrict__ B, __nv_bfloat16 reg_B[DP_B_EPT],
    unsigned int cta_n, unsigned int k_base, unsigned int N, unsigned int K
) {
    // smem_B is [K_STEP][N_TILE]: element (k, n). B is [N, K] row-major.
    #pragma unroll
    for (unsigned int i = 0; i < DP_B_EPT; i++) {
        unsigned int idx = threadIdx.x * DP_B_EPT + i;
        unsigned int k = idx / DP_N_TILE, n = idx % DP_N_TILE;
        unsigned int gk = k_base + k, gn = cta_n + n;
        reg_B[i] = (gk < K && gn < N) ? B[(unsigned long long)gn * K + gk] : __float2bfloat16(0.0f);
    }
}

__device__ __forceinline__ void dp_store_A_regs(
    __nv_bfloat16 smem_A[][DP_K_STEP + DP_PAD], const __nv_bfloat16 reg_A[DP_A_EPT]
) {
    #pragma unroll
    for (unsigned int i = 0; i < DP_A_EPT; i++) {
        unsigned int idx = threadIdx.x * DP_A_EPT + i;
        smem_A[idx / DP_K_STEP][idx % DP_K_STEP] = reg_A[i];
    }
}

__device__ __forceinline__ void dp_store_B_regs(
    __nv_bfloat16 smem_B[][DP_N_TILE + DP_PAD], const __nv_bfloat16 reg_B[DP_B_EPT]
) {
    #pragma unroll
    for (unsigned int i = 0; i < DP_B_EPT; i++) {
        unsigned int idx = threadIdx.x * DP_B_EPT + i;
        smem_B[idx / DP_N_TILE][idx % DP_N_TILE] = reg_B[i];
    }
}

__device__ __forceinline__ void dp_wmma_compute(
    __nv_bfloat16 smem_A[][DP_K_STEP + DP_PAD],
    __nv_bfloat16 smem_B[][DP_N_TILE + DP_PAD],
    dp_v8f acc[DP_NSUB], unsigned int warp_m_offset, unsigned int lane
) {
    dp_v16bf a;
    #pragma unroll
    for (int i = 0; i < 16; i++) a[i] = (__bf16)smem_A[warp_m_offset + (lane & 15)][i];
    #pragma unroll
    for (int nb = 0; nb < DP_NSUB; nb++) {
        dp_v16bf b;
        #pragma unroll
        for (int k = 0; k < 16; k++) b[k] = (__bf16)smem_B[k][nb * 16 + (lane & 15)];
        acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]);
    }
}

extern "C" __global__
__launch_bounds__(256, 2)
void dense_gemm_bf16_pipelined(
    const __nv_bfloat16* __restrict__ A,   // [M, K] BF16 activations
    const __nv_bfloat16* __restrict__ B,   // [N, K] BF16 weights (read transposed)
    __nv_bfloat16* __restrict__ C,          // [M, N] BF16 output
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * DP_M_TILE;
    const unsigned int cta_n = blockIdx.x * DP_N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ __nv_bfloat16 smem_A[2][DP_M_TILE][DP_K_STEP + DP_PAD];
    __shared__ __nv_bfloat16 smem_B[2][DP_K_STEP][DP_N_TILE + DP_PAD];

    dp_v8f acc[DP_NSUB];
    #pragma unroll
    for (int i = 0; i < DP_NSUB; i++) acc[i] = dp_v8f{0, 0, 0, 0, 0, 0, 0, 0};

    const unsigned int num_tiles = (K + DP_K_STEP - 1) / DP_K_STEP;

    __nv_bfloat16 reg_A[DP_A_EPT];
    __nv_bfloat16 reg_B[DP_B_EPT];
    dp_load_A_regs(A, reg_A, cta_m, 0, M, K);
    dp_load_B_regs(B, reg_B, cta_n, 0, N, K);
    dp_store_A_regs(smem_A[0], reg_A);
    dp_store_B_regs(smem_B[0], reg_B);
    __syncthreads();

    for (unsigned int kt = 1; kt < num_tiles; kt++) {
        const unsigned int k_base = kt * DP_K_STEP;
        dp_load_A_regs(A, reg_A, cta_m, k_base, M, K);
        dp_load_B_regs(B, reg_B, cta_n, k_base, N, K);
        dp_wmma_compute(smem_A[(kt - 1) & 1], smem_B[(kt - 1) & 1], acc, warp_m_offset, lane_id);
        dp_store_A_regs(smem_A[kt & 1], reg_A);
        dp_store_B_regs(smem_B[kt & 1], reg_B);
        __syncthreads();
    }
    dp_wmma_compute(smem_A[(num_tiles - 1) & 1], smem_B[(num_tiles - 1) & 1], acc, warp_m_offset, lane_id);

    #pragma unroll
    for (int nb = 0; nb < DP_NSUB; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int r = cta_m + warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) C[(unsigned long long)r * N + c] = __float2bfloat16(acc[nb][e]);
        }
}

// Fused SiLU(gate) * up activation — vectorized 2-wide BF16 loads/stores.
// Input: [N, inter_size*2] where first half is gate, second half is up.
// Output: [N, inter_size]
// out[i] = silu(gate[i]) * up[i]  where silu(x) = x * sigmoid(x)
extern "C" __global__ void fused_silu_mul(
    const __nv_bfloat16* __restrict__ gate_up,  // [num_tokens, inter_size * 2]
    __nv_bfloat16* __restrict__ output,          // [num_tokens, inter_size]
    unsigned int num_tokens,
    unsigned int inter_size
) {
    // Each thread processes 2 elements (vectorized BF16x2)
    unsigned int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half_total = (num_tokens * inter_size) / 2;
    if (idx2 >= half_total) return;

    // Map linear index to (token, col_pair)
    unsigned int half_inter = inter_size / 2;
    unsigned int token = idx2 / half_inter;
    unsigned int col_pair = idx2 % half_inter;

    // Vectorized loads: 2 BF16 per 32-bit read
    const unsigned int* gate32 = (const unsigned int*)(gate_up + token * (inter_size * 2));
    const unsigned int* up32 = (const unsigned int*)(gate_up + token * (inter_size * 2) + inter_size);
    unsigned int* out32 = (unsigned int*)(output + token * inter_size);

    unsigned int g_packed = gate32[col_pair];
    unsigned int u_packed = up32[col_pair];

    float g0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(g_packed & 0xFFFF)));
    float g1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(g_packed >> 16)));
    float u0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(u_packed & 0xFFFF)));
    float u1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(u_packed >> 16)));

    // SiLU(gate) * up for both elements
    float sg0 = 1.0f / (1.0f + __expf(-g0));
    float sg1 = 1.0f / (1.0f + __expf(-g1));
    float r0 = g0 * sg0 * u0;
    float r1 = g1 * sg1 * u1;

    // Vectorized store
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(r0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(r1));
    out32[col_pair] = lo | (hi << 16);
}
