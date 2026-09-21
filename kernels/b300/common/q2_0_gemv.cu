// SPDX-License-Identifier: AGPL-3.0-only

// Atlas native ternary Q2_0 decode GEMV — keep-packed W2A16 for M=1..8 decode.
//
//   out[m,n] = sum_k A[m,k] * (code(n,k) - 1) * d(n, k/group)
//
// The weight NEVER expands to BF16: each 34-byte (group-128) / 18-byte
// (group-64) PrismML `block_q2_0` stays packed in VRAM and is dequantized
// inside the dot-product, exactly like `w4a16_gemv.cu`/`w8a16_gemv.cu` do for
// NVFP4/FP8. Activation is plain BF16 (weight-only quant), matching Atlas's
// decode-GEMV convention (there is no activation-quant step on this path).
//
// Q2_0 block layout (PrismML id 42, validated in `dequant_gguf_bf16.cu`):
//   [ fp16 d @ front ][ group/4 bytes of 2-bit codes, low-bits-first ]
//   code = (qs[j>>2] >> (2*(j&3))) & 3;   value = (code - 1) * d
//   → symbols {-1, 0, +1, +2}. Scale `d` is inline (one per group), so unlike
//   the FP8/NVFP4 kernels there is NO separate scale pointer.
//
// Row `n` of the [N, K] weight is `K/group` contiguous blocks; its byte base is
//   n * (K/group) * (2 + group/4).
//
// 4 outputs/block, 64 threads (2 warps) per output, warp-shuffle reduction —
// same launch geometry as w8a16_gemv. Grid: (ceil(N/4), 1, 1)  Block: (256,1,1).

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define MAX_M 8

// Little-endian IEEE fp16 (2 bytes) -> f32. Matches half::f16::to_f32 on host
// and the `dq_rd_f16` helper in dequant_gguf_bf16.cu (bit-identical scale read).
__device__ __forceinline__ float q2_rd_f16(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(bits));
}

// ── W2A16 GEMV (M=1) ───────────────────────────────────────────────
//
// C[n] = sum_k A[k] * (code(n,k) - 1) * d(n, k/group)
//
// Each of the 64 threads assigned to output row `n` strides over that row's
// group-blocks, reads the inline fp16 scale, and accumulates the packed codes
// against the BF16 activation. FP32 accumulator; cross-warp smem reduction.
extern "C" __global__ void q2_0_gemv(
    const __nv_bfloat16* __restrict__ A,       // [1, K] BF16 activation
    const unsigned char* __restrict__ B,       // [N, K] packed Q2_0 blocks
    __nv_bfloat16* __restrict__ C,             // [1, N] BF16 output
    unsigned int N,
    unsigned int K,
    unsigned int group)                        // 128 or 64
{
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;

    __shared__ float smem[N_PER_BLOCK * 2];
    if (n >= N) {
        // Threads for out-of-range rows still must not touch smem/reduction.
        return;
    }

    const unsigned int blocks_per_row = K / group;         // K is a multiple of group
    const unsigned int block_bytes = 2u + group / 4u;      // 34 (g128) / 18 (g64)
    const unsigned char* row = B + (unsigned long long)n * blocks_per_row * block_bytes;

    float acc = 0.0f;

    // Stride the row's group-blocks across the 64 lanes.
    for (unsigned int b = lane; b < blocks_per_row; b += threads_per_out) {
        const unsigned char* blk = row + (unsigned long long)b * block_bytes;
        const float d = q2_rd_f16(blk);
        const unsigned char* qs = blk + 2;
        const unsigned int base_k = b * group;

        // 4 codes per byte, low-bits-first.
        #pragma unroll 4
        for (unsigned int cb = 0; cb < group / 4u; ++cb) {
            const unsigned int byte = qs[cb];
            const unsigned int k0 = base_k + cb * 4u;
            #pragma unroll
            for (unsigned int t = 0; t < 4u; ++t) {
                const int code = (int)((byte >> (2u * t)) & 3u);
                const float a = __bfloat162float(A[k0 + t]);
                acc += a * (float)(code - 1) * d;
            }
        }
    }

    // Cross-warp reduction: warp-shuffle then a 2-element smem combine.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    const unsigned int warp_in_out = lane / WARP_SIZE;
    if (lane % WARP_SIZE == 0) {
        smem[local_out * 2 + warp_in_out] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        C[n] = __float2bfloat16(smem[local_out * 2] + smem[local_out * 2 + 1]);
    }
}

// ── W2A16 batched GEMV (M=1..8) ────────────────────────────────────
//
// C[m, n] = sum_k A[m, k] * (code(n,k) - 1) * d(n, k/group)
//
// Reads each weight block ONCE and accumulates it against all M activation
// rows (A is `[M, K]` row-major, C is `[M, N]` row-major). Amortizes the
// weight-byte read (the bandwidth cost at decode) across the batch — the same
// trick as `w4a16_gemv_batchm`. `M` in [1, 8]; larger M should use prefill.
extern "C" __global__ void q2_0_gemv_batchm(
    const __nv_bfloat16* __restrict__ A,       // [M, K] BF16
    const unsigned char* __restrict__ B,       // [N, K] packed Q2_0
    __nv_bfloat16* __restrict__ C,             // [M, N] BF16
    unsigned int N,
    unsigned int K,
    unsigned int group,
    unsigned int M)
{
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;

    __shared__ float smem[N_PER_BLOCK * 2 * MAX_M];
    if (n >= N) return;

    const unsigned int blocks_per_row = K / group;
    const unsigned int block_bytes = 2u + group / 4u;
    const unsigned char* row = B + (unsigned long long)n * blocks_per_row * block_bytes;

    float acc[MAX_M];
    #pragma unroll
    for (unsigned int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (unsigned int b = lane; b < blocks_per_row; b += threads_per_out) {
        const unsigned char* blk = row + (unsigned long long)b * block_bytes;
        const float d = q2_rd_f16(blk);
        const unsigned char* qs = blk + 2;
        const unsigned int base_k = b * group;

        #pragma unroll 4
        for (unsigned int cb = 0; cb < group / 4u; ++cb) {
            const unsigned int byte = qs[cb];
            const unsigned int k0 = base_k + cb * 4u;
            #pragma unroll
            for (unsigned int t = 0; t < 4u; ++t) {
                const int code = (int)((byte >> (2u * t)) & 3u);
                const float wv = (float)(code - 1) * d;
                const unsigned int k = k0 + t;
                for (unsigned int m = 0; m < M; ++m) {
                    acc[m] += __bfloat162float(A[m * K + k]) * wv;
                }
            }
        }
    }

    const unsigned int warp_in_out = lane / WARP_SIZE;
    for (unsigned int m = 0; m < M; ++m) {
        float v = acc[m];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xFFFFFFFF, v, offset);
        }
        if (lane % WARP_SIZE == 0) {
            smem[(local_out * MAX_M + m) * 2 + warp_in_out] = v;
        }
    }
    __syncthreads();

    if (lane == 0) {
        for (unsigned int m = 0; m < M; ++m) {
            float r = smem[(local_out * MAX_M + m) * 2] + smem[(local_out * MAX_M + m) * 2 + 1];
            C[m * N + n] = __float2bfloat16(r);
        }
    }
}
