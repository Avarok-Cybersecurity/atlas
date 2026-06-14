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
// Block (256,1,1) — so the dispatch is unchanged. The math is bit-equivalent
// to the gb10 kernel (FP32 accumulation of the same BF16 products → the same
// cosine=1.0 the gb10 path documents); only the GEMM micro-architecture
// differs (synchronous smem staging + register-blocked scalar FMA in place of
// async-copy + tensor cores). RDNA3.5 WMMA acceleration is a perf follow-up
// (shares the fragment-layout work tracked for the HIP FP8 attention kernels);
// correctness here is independent of that optimization.
//
// 256 threads cover a 128×128 output tile as a 16×16 grid of threads, each
// owning a contiguous 8×8 micro-tile. smem: 2 × 128×16 BF16 = 8192 B/CTA.
#define DP_M_TILE 128
#define DP_N_TILE 128
#define DP_K_STEP 16
#define DP_THREADS 256
extern "C" __global__ void dense_gemm_bf16_pipelined(
    const __nv_bfloat16* __restrict__ A,   // [M, K] BF16 activations
    const __nv_bfloat16* __restrict__ B,   // [N, K] BF16 weights (read transposed)
    __nv_bfloat16* __restrict__ C,          // [M, N] BF16 output
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * DP_M_TILE;
    const unsigned int cta_n = blockIdx.x * DP_N_TILE;
    // 16×16 thread grid; thread (tx,ty) owns rows [ty*8, ty*8+8), cols
    // [tx*8, tx*8+8) of the CTA's 128×128 output tile.
    const unsigned int tx = threadIdx.x & 15;          // [0,16) → N
    const unsigned int ty = threadIdx.x >> 4;          // [0,16) → M

    __shared__ __nv_bfloat16 sA[DP_M_TILE][DP_K_STEP];  // A[m, k]
    __shared__ __nv_bfloat16 sB[DP_N_TILE][DP_K_STEP];  // B[n, k] (K-contiguous)

    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;

    const __nv_bfloat16 zero = __float2bfloat16(0.0f);

    for (unsigned int k_base = 0; k_base < K; k_base += DP_K_STEP) {
        // Cooperative load: 128×16 = 2048 elems / 256 threads = 8 each.
        #pragma unroll
        for (unsigned int e = threadIdx.x; e < DP_M_TILE * DP_K_STEP; e += DP_THREADS) {
            unsigned int row = e / DP_K_STEP;
            unsigned int col = e % DP_K_STEP;
            unsigned int gr = cta_m + row;
            unsigned int gc = k_base + col;
            sA[row][col] = (gr < M && gc < K)
                ? A[(unsigned long long)gr * K + gc] : zero;
        }
        #pragma unroll
        for (unsigned int e = threadIdx.x; e < DP_N_TILE * DP_K_STEP; e += DP_THREADS) {
            unsigned int nrow = e / DP_K_STEP;
            unsigned int kcol = e % DP_K_STEP;
            unsigned int gn = cta_n + nrow;
            unsigned int gk = k_base + kcol;
            sB[nrow][kcol] = (gn < N && gk < K)
                ? B[(unsigned long long)gn * K + gk] : zero;
        }
        __syncthreads();

        #pragma unroll
        for (unsigned int kk = 0; kk < DP_K_STEP; kk++) {
            float a_reg[8];
            float b_reg[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) a_reg[i] = __bfloat162float(sA[ty * 8 + i][kk]);
            #pragma unroll
            for (int j = 0; j < 8; j++) b_reg[j] = __bfloat162float(sB[tx * 8 + j][kk]);
            #pragma unroll
            for (int i = 0; i < 8; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        unsigned int r = cta_m + ty * 8 + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned int c = cta_n + tx * 8 + j;
            if (c < N) C[(unsigned long long)r * N + c] = __float2bfloat16(acc[i][j]);
        }
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
