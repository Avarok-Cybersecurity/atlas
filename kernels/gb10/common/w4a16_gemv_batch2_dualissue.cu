// SPDX-License-Identifier: AGPL-3.0-only

// W4A16 GEMV batch2 — dual-issue (no smem mailbox).
//
// Same bit-exact FMA order and two-phase virtual-lane remap as
// `w4a16_gemv_batchm_impl<2>` in w4a16_gemv.cu. The difference is WHEN
// the packed-weight loads happen:
//
//   template batch2:  load phase-0 → compute → load phase-1 → compute
//   this kernel:      load phase-0 AND phase-1 → compute 0 → compute 1
//
// `#497` cp.async tried the same overlap via a shared-memory mailbox and
// lost (195 vs 227 GB/s on 12288×2048). This path keeps ordinary
// `ld.global` so there is no commit/wait/alignment tax. Not wired to
// production. Do not default-on until
// `w4a16_batch2_bw_oracle` says ≥3% faster than template batch2.
//
// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
// Signature matches `w4a16_gemv_batch2`.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u;
    float v;
    if (e == 0u)
        v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u)
        v = 0.0f;
    else
        v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#endif

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ void dualissue_fma_chunk(
    const __nv_bfloat16* __restrict__ A, unsigned int K, unsigned int kk,
    unsigned long long packed8, float scale, const float* s_lut, float acc[2])
{
    float wl[16];
#pragma unroll
    for (int b = 0; b < 8; b++) {
        unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
        wl[b * 2] = s_lut[byte_val & 0xF];
        wl[b * 2 + 1] = s_lut[byte_val >> 4];
    }
#pragma unroll
    for (int t = 0; t < 2; t++) {
        const __nv_bfloat16* At = A + (unsigned long long)t * K;
        uint4 a_lo = ((const uint4*)At)[kk * 2];
        uint4 a_hi = ((const uint4*)At)[kk * 2 + 1];
        const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                    a_hi.x, a_hi.y, a_hi.z, a_hi.w};
        float part = 0.0f;
#pragma unroll
        for (int b = 0; b < 8; b++) {
            float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&ar[b]);
            part = fmaf(af.x, wl[b * 2], part);
            part = fmaf(af.y, wl[b * 2 + 1], part);
        }
        acc[t] = fmaf(scale, part, acc[t]);
    }
}

extern "C" __global__ void w4a16_gemv_batch2_dualissue(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2,
    __nv_bfloat16* __restrict__ C, unsigned int N, unsigned int K)
{
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;
    const unsigned int stride = threads_per_out * 2u;

    __shared__ float s_vl[2][N_PER_BLOCK][2 * WARP_SIZE];

    // Hoist the first packed+scale load of BOTH phases before either
    // compute. On K=2048 each phase is one iteration, so this is the
    // whole weight stream for this thread.
    unsigned int kk0 = lane;
    unsigned int kk1 = lane + threads_per_out;
    unsigned long long pk0 = 0, pk1 = 0;
    unsigned char sc0 = 0, sc1 = 0;
    const bool live0 = kk0 < K16;
    const bool live1 = kk1 < K16;
    if (live0) {
        pk0 = *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + kk0 * 8);
        sc0 = B_scale[(unsigned long long)n * num_groups + kk0];
    }
    if (live1) {
        pk1 = *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + kk1 * 8);
        sc1 = B_scale[(unsigned long long)n * num_groups + kk1];
    }

#pragma unroll 1
    for (unsigned int phase = 0; phase < 2u; phase++) {
        float acc[2] = {0.0f, 0.0f};
        unsigned int kk = (phase == 0u) ? kk0 : kk1;
        unsigned long long packed8 = (phase == 0u) ? pk0 : pk1;
        unsigned char scale_byte = (phase == 0u) ? sc0 : sc1;
        bool live = (phase == 0u) ? live0 : live1;

        while (live) {
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8(scale_byte) * scale2;
#else
            float scale = (float)fp8 * scale2;
#endif
            const unsigned int kk_next = kk + stride;
            const bool more = kk_next < K16;
            unsigned long long pk_next = 0;
            unsigned char sc_next = 0;
            if (more) {
                pk_next = *(const unsigned long long*)(
                    B_packed + (unsigned long long)n * half_K + kk_next * 8);
                sc_next = B_scale[(unsigned long long)n * num_groups + kk_next];
            }
            dualissue_fma_chunk(A, K, kk, packed8, scale, s_lut, acc);
            if (!more) break;
            kk = kk_next;
            packed8 = pk_next;
            scale_byte = sc_next;
            live = true;
        }

#pragma unroll
        for (int t = 0; t < 2; t++) {
            const float v = acc[t] + __shfl_xor_sync(0xFFFFFFFF, acc[t], 1);
            if ((lane & 1u) == 0u) {
                s_vl[t][local_out][phase * WARP_SIZE + (lane >> 1)] = v;
            }
        }
    }
    __syncthreads();

    __shared__ float smem[2][N_PER_BLOCK * 2];
    const unsigned int warp_in_out = lane / WARP_SIZE;
#pragma unroll
    for (int t = 0; t < 2; t++) {
        float a = s_vl[t][local_out][lane];
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        if (lane % WARP_SIZE == 0) smem[t][local_out * 2 + warp_in_out] = a;
    }
    __syncthreads();

    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < 2; t++) {
            float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * N + n] = __float2bfloat16(r);
        }
    }
}
