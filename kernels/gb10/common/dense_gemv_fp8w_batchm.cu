// SPDX-License-Identifier: AGPL-3.0-only
//
// dense_gemv_fp8w_batchm.cu — batched row-scaled FP8 GEMV (M<=8) for the
// full-vocab LM head, BIT-IDENTICAL to M separate `dense_gemv_fp8w` calls.
//
//   C[M,N] (bf16) = A[M,K] (bf16) @ dequant(B)^T * row_scale[N]
//
// Why it exists: the FP8 LM-head decode path had a batched arm only at M=2
// and fell to a PER-TOKEN LOOP above it. At a 248K vocab the head is ~1.27 GB
// of FP8, so an M=8 verify step re-read the entire head EIGHT times — which
// is why the FP8 head measured slower than BF16 despite carrying half the
// bytes. One pass over the weight now serves up to 8 activation rows.
//
// PARITY IS THE POINT, not a bonus. These logits are what the sampler
// argmaxes, so a different rounding order is a different token. The
// accumulation below is therefore the SAME per-element chain as the M=1
// kernel — one product per `acc +=` statement, in the same k order:
//
//     acc += a_lo * w0;   acc += a_hi * w1;
//     acc += a_lo * w2;   acc += a_hi * w3;
//
// NOT the 4-term `acc += a0*w0 + a1*w1 + a2*w2 + a3*w3` form used by
// `dense_gemv_fp8w_batch2`'s `mac4`, which contracts each group of four
// before accumulating and is therefore NOT bit-identical to the M=1 path its
// header claims parity with. See dense_gemv_fp8w_bitparity_microtest.
//
// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1) — same geometry as the M=1 and
// M=2 kernels, so launchers and any graph capture are unchanged.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BM_BLOCK 256
#define BM_N_PER_BLOCK 4
#define BM_VEC 16
#define BM_MAX_M 8

__device__ __forceinline__ float bm_fp8_to_f32(unsigned char b) {
    __nv_fp8_e4m3 v;
    *(unsigned char*)&v = b;
    return (float)v;
}

extern "C" __global__ void dense_gemv_fp8w_batchm(
    const __nv_bfloat16* __restrict__ A,   // [M, K] bf16, row-major
    const unsigned char* __restrict__ B,   // [N, K] fp8 e4m3
    const float* __restrict__ row_scale,   // [N] f32
    __nv_bfloat16* __restrict__ C,         // [M, N] bf16
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BM_BLOCK / BM_N_PER_BLOCK; // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * BM_N_PER_BLOCK + local_out;

    // Tail blocks must not return early: the cross-warp reduction below has a
    // __syncthreads() that every thread in the block has to reach (same class
    // as the w4a16/w8a16 tail fixes).
    const bool active = n < N;
    const float scale = active ? row_scale[n] : 0.0f;

    float acc[BM_MAX_M];
    #pragma unroll
    for (int t = 0; t < BM_MAX_M; t++) acc[t] = 0.0f;

    const unsigned int K_VEC = K / BM_VEC;
    for (unsigned int kv = lane; active && kv < K_VEC; kv += threads_per_out) {
        // 16 FP8 weight bytes for this output row, read ONCE and reused by
        // every activation row — this reuse is the whole point of the kernel.
        const uint4 wb = *(const uint4*)(B + (unsigned long long)n * K
                                           + (unsigned long long)kv * BM_VEC);
        const unsigned int w_raw[4] = {wb.x, wb.y, wb.z, wb.w};

        #pragma unroll
        for (int t = 0; t < BM_MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            const __nv_bfloat16* At = A + (unsigned long long)t * K;
            const uint4 a_d0 = ((const uint4*)At)[kv * 2];
            const uint4 a_d1 = ((const uint4*)At)[kv * 2 + 1];
            const unsigned int a_raw[8] = {a_d0.x, a_d0.y, a_d0.z, a_d0.w,
                                           a_d1.x, a_d1.y, a_d1.z, a_d1.w};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned int w32 = w_raw[i];
                const unsigned int a32_lo = a_raw[i * 2];
                const unsigned int a32_hi = a_raw[i * 2 + 1];
                __nv_bfloat16 a0, a1, a2, a3;
                *(unsigned short*)&a0 = (unsigned short)(a32_lo & 0xFFFF);
                *(unsigned short*)&a1 = (unsigned short)(a32_lo >> 16);
                *(unsigned short*)&a2 = (unsigned short)(a32_hi & 0xFFFF);
                *(unsigned short*)&a3 = (unsigned short)(a32_hi >> 16);
                // One product per statement, k ascending — the M=1 chain.
                acc[t] += __bfloat162float(a0) * bm_fp8_to_f32((unsigned char)(w32 & 0xFF));
                acc[t] += __bfloat162float(a1) * bm_fp8_to_f32((unsigned char)((w32 >> 8) & 0xFF));
                acc[t] += __bfloat162float(a2) * bm_fp8_to_f32((unsigned char)((w32 >> 16) & 0xFF));
                acc[t] += __bfloat162float(a3) * bm_fp8_to_f32((unsigned char)((w32 >> 24) & 0xFF));
            }
        }
    }

    // Scale BEFORE the reduction, exactly like the M=1 kernel
    // (`acc *= scale;` ahead of its shuffle). Scaling the summed result
    // instead would be `(a+b)*s` against its `a*s + b*s` — a different
    // rounding, and on a 248K-vocab argmax a different token.
    #pragma unroll
    for (int t = 0; t < BM_MAX_M; t++) {
        if ((unsigned int)t >= M) continue;
        acc[t] *= scale;
    }

    // 64-lane reduction: shuffle within each warp, cross-warp through smem.
    __shared__ float s_red[BM_MAX_M][BM_N_PER_BLOCK * 2];
    const unsigned int warp_in_out = lane / 32u;
    #pragma unroll
    for (int t = 0; t < BM_MAX_M; t++) {
        if ((unsigned int)t >= M) continue;
        float a = acc[t];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, off);
        }
        if ((lane & 31u) == 0u) {
            s_red[t][local_out * 2 + warp_in_out] = a;
        }
    }
    __syncthreads();

    if (lane == 0 && active) {
        #pragma unroll
        for (int t = 0; t < BM_MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            // Same combine order as the M=1 kernel: smem[0] + smem[1],
            // scale already applied per-thread above.
            const float r = s_red[t][local_out * 2] + s_red[t][local_out * 2 + 1];
            C[(unsigned long long)t * N + n] = __float2bfloat16(r);
        }
    }
}
