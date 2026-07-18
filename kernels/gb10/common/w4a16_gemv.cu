// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 GEMV — Fused NVFP4 weight dequant + BF16 GEMV for M=1 decode.
//
// out[n] = dot(A[0,:], dequant(B_fp4[n,:]))
//
// Specialized for M=1 decode: replaces w4a16_gemm which wastes ~98% of
// threads at M=1 with 64x64 tiles + MMA tensor cores (MMA requires M>=16).
//
// Vectorized: reads 4 packed weight bytes (uint32_t = 8 FP4 values) and
// 8 BF16 activations (uint4 = 16 bytes) per iteration for better bandwidth.
//
// NVFP4 weight format (HuggingFace/compressed-tensors):
//   B_packed: [N, K/2] uint8 — byte at [n, j] holds W[n, 2j] (low) and W[n, 2j+1] (high)
//   B_scale:  [N, K/GROUP_SIZE] FP8-E4M3 — one scale per group of 16 K-dim values
//   scale2:   scalar FP32 — per-tensor second-level scale
//
// K-dim packing: each byte holds 2 consecutive input features for the same output.
// Vectorized reads of 4 bytes = 8 weight values, coalesced across warps.
//
// 4 outputs per block, 64 threads (2 warps) per output. Cross-warp smem reduction.
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Standard E4M3 (1-4-3, bias 7) decode via pure bit-math. On real NVIDIA this is
// byte-identical to (float)__nv_fp8_e4m3; on SCALE/gfx1151 the built-in
// __nv_fp8_e4m3->float decode is a NON-STANDARD narrow format which mismatches the
// standard E4M3 scales written by the encoder -> corrupts every block scale.
// HIP/gfx1151 shares the same software path (no cvt.rn.satfinite.e4m3x2.f32 PTX).
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;            // subnormal m*2^-9
    else if (e == 15u && m == 7u) v = 0.0f;                            // NaN -> 0
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20)); // 2^(e-7)*(1+m/8)
    return s ? -v : v;
}
#endif

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

// E2M1 lookup table (same as w4a16_gemm.cu)
__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// W4A16 GEMV: C[n] = sum_k A[k] * dequant(B_fp4[n, k])
//
// Vectorized: processes 8 K-values per iteration.
// - 4 packed weight bytes (uint32_t) → 8 FP4 values via E2M1 LUT
// - 8 BF16 activations (uint4 = 128-bit load)
// - 1 FP8 scale (all 8 values in same group since GROUP_SIZE=16, stride=8)
//
// Coalescing: within a warp, consecutive threads read consecutive 4-byte
// weight chunks and consecutive 16-byte activation chunks. Perfectly coalesced.
extern "C" __global__ void w4a16_gemv(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [1, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];  // cross-warp reduction
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f;

    // Vectorized: 16 K-values per chunk (2× uint4 activation + uint64 weight),
    // TWO chunks in flight per iteration with independent accumulators. Keeping 2
    // outstanding weight loads per thread hides DRAM latency (ncu: 72% of warp
    // stalls are long-scoreboard on GB10). The FP8 group scale is factored out of
    // the inner 16-FMA block (exact regroup: sum(s*w*a) == s*sum(w*a)).
    const unsigned int stride2 = threads_per_out * 2u;
    for (unsigned int k16 = lane * 2u; k16 < K16 + 1u; k16 += stride2) {
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            const unsigned int kk = k16 + (unsigned int)c;
            if (kk >= K16) break;

            // Load 16 BF16 activations as 2× uint4 (256-bit total)
            uint4 a_lo = ((const uint4*)A)[kk * 2];
            uint4 a_hi = ((const uint4*)A)[kk * 2 + 1];
            const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                            a_hi.x, a_hi.y, a_hi.z, a_hi.w};

            // Load 8 packed weight bytes as uint64 (16 FP4 values)
            unsigned long long packed8 = *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + kk * 8);

            // Load single FP8 scale — 16 values = exactly 1 group (group index == kk)
            unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + kk];
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8(scale_byte) * scale2;
#else
            float scale = (float)fp8 * scale2;
#endif

            // Unpack 8 bytes × 2 nibbles = 16 weight values, FMA with activations
            float part = 0.0f;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&a_raw[b]);
                part = fmaf(af.x, s_lut[byte_val & 0xF], part);
                part = fmaf(af.y, s_lut[byte_val >> 4], part);
            }
            if (c == 0) acc0 = fmaf(scale, part, acc0);
            else        acc1 = fmaf(scale, part, acc1);
        }
    }
    float acc = acc0 + acc1;

    // Warp shuffle reduction within each group of 64 threads
    // First reduce within each warp (32 threads)
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // threads_per_out=64 means 2 warps per output. Use shared memory for cross-warp reduce.
    if (warp_lane == 0) {
        // Each warp writes its partial sum
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    // First thread of each output group writes final result
    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 GEMV — SINGLE-WARP-PER-OUTPUT variant (lossless, opt-in).
//
// Bit-identical to w4a16_gemv above, but uses 32 threads (1 warp) per output
// instead of 64 (2 warps). 8 outputs per 256-thread block (was 4). The cross-
// warp __syncthreads() + shared-memory round-trip is ELIMINATED — the final
// combine collapses to a single FP32 add of two warp-shuffle results.
//
// BIT-IDENTICALITY (the hard gate): the original splits the K-strided partials
// across 64 lanes; warp A (orig lanes 0..31) and warp B (orig lanes 32..63) are
// each shuffle-reduced, then summed `smem[0]+smem[1]`. Here each of the 32 lanes
// holds TWO accumulators that reproduce EXACTLY those two lane-sets:
//   acc_a[lane]  == orig acc[lane]      (chunks lane, lane+64, ...)      -> warp A
//   acc_b[lane]  == orig acc[lane+32]   (chunks lane+32, lane+32+64, ...) -> warp B
// We shuffle-reduce acc_a (== warp-A reduction) and acc_b (== warp-B reduction)
// in the SAME tree order, then `reduced_a + reduced_b` (== smem[0]+smem[1]).
// Every FP32 add is in the identical order/operands -> byte-identical output.
//
// 8 outputs per block, 32 threads (1 warp) per output. NO smem, NO __syncthreads
// in the reduction. Grid: (ceil(N / 8), 1, 1)   Block: (256, 1, 1)

#define N_PER_BLOCK_SW 8

// Accumulate the K-strided partial for one "virtual lane" (start chunk +
// stride 64), matching the inner math of w4a16_gemv exactly.
__device__ __forceinline__ float w4a16_gemv_partial(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    unsigned int n, unsigned int half_K, unsigned int num_groups,
    unsigned int K16, unsigned int start_chunk)
{
    float acc = 0.0f;
    // stride 64 == the original threads_per_out, so the per-chunk membership and
    // accumulation order of orig lane `start_chunk` are reproduced exactly.
    for (unsigned int k16 = start_chunk; k16 < K16; k16 += 64u) {
        const unsigned int base_k = k16 * 16;
        uint4 a_lo = ((const uint4*)A)[k16 * 2];
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};
        unsigned long long packed8 = *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + k16 * 8);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
            float w_lo = E2M1_LUT[byte_val & 0xF] * scale;
            float w_hi = E2M1_LUT[byte_val >> 4] * scale;
            __nv_bfloat16 a_lo_bf, a_hi_bf;
            *(unsigned short*)&a_lo_bf = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi_bf = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo_bf) * w_lo;
            acc += __bfloat162float(a_hi_bf) * w_hi;
        }
    }
    return acc;
}

extern "C" __global__ void w4a16_gemv_sw(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [1, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int local_out = threadIdx.x / WARP_SIZE;       // 0..7
    const unsigned int lane = threadIdx.x % WARP_SIZE;            // 0..31
    const unsigned int n = blockIdx.x * N_PER_BLOCK_SW + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    // acc_a reproduces orig lane `lane` (warp A); acc_b reproduces orig lane
    // `lane+32` (warp B). Same operands, same order as the 64-thread kernel.
    float acc_a = w4a16_gemv_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane);
    float acc_b = w4a16_gemv_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane + 32u);

    // Reduce each accumulator within the warp in the SAME tree order as orig.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xFFFFFFFF, acc_a, offset);
        acc_b += __shfl_down_sync(0xFFFFFFFF, acc_b, offset);
    }

    // lane 0 holds reduced warp-A (acc_a) and warp-B (acc_b). Final combine ==
    // smem[0] + smem[1] in the 64-thread kernel. Bit-identical.
    if (lane == 0) {
        float result = acc_a + acc_b;
        C[n] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 GEMV with FP32 output (for LM head logits).
// Identical to w4a16_gemv but writes float instead of BF16.
// FP32 logits are critical for sampling quality — BF16 collapses
// similar logit values, making stochastic sampling random.
// ============================================================
extern "C" __global__ void w4a16_gemv_logits(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    float* __restrict__ C,  // FP32 output
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;
    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;
        uint4 a_lo = ((const uint4*)A)[k16 * 2];
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};
        unsigned long long packed8 = *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + k16 * 8);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;
            __nv_bfloat16 a_lo_bf, a_hi_bf;
            *(unsigned short*)&a_lo_bf = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi_bf = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo_bf) * w_lo;
            acc += __bfloat162float(a_hi_bf) * w_hi;
        }
    }
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();
    if (lane == 0) {
        C[n] = smem[local_out * 2] + smem[local_out * 2 + 1]; // FP32 output!
    }
}

// ============================================================
// W4A16 double-GEMV (M=2): reads weights once, computes 2 outputs
// ============================================================
// For K=2 speculative verification: processes 2 input vectors through
// the same weight matrix in a single pass. Eliminates the GEMM M=2
// tile waste (64x64 tiles at 3% M-utilization).
//
// A: [2, K] BF16 contiguous (row 0 and row 1)
// B: [N, K/2] NVFP4 packed weights
// C: [2, N] BF16 contiguous (row 0 and row 1)
//
// Same memory bandwidth as M=1 GEMV (weights dominate, read once).
// Extra cost: 2x activation reads (K*2 bytes per vector, fits in L1/L2).
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_batch2(
    const __nv_bfloat16* __restrict__ A,        // [2, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [2, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    // Pointers to second input/output rows
    const __nv_bfloat16* __restrict__ A1 = A + K;  // second input vector
    __nv_bfloat16* __restrict__ C1 = C + N;         // second output vector

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 4];  // 2 warps × 2 accumulators per output
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;  // accumulator for first input
    float acc1 = 0.0f;  // accumulator for second input

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        // Load 8 BF16 activations from BOTH input vectors
        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};

        // Load 4 packed weight bytes (SHARED between both inputs)
        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);

        // Load single FP8 scale
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        // Unpack weights and FMA with BOTH activation vectors
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            // First input vector
            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo;
            acc0 += __bfloat162float(a0_hi) * w_hi;

            // Second input vector (same weights, different activations)
            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo;
            acc1 += __bfloat162float(a1_hi) * w_hi;
        }
    }

    // Warp shuffle reduction for both accumulators
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    // Cross-warp reduction via shared memory (2 warps per output × 2 accumulators)
    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 4 + warp_idx * 2]     = acc0;
        smem[local_out * 4 + warp_idx * 2 + 1] = acc1;
    }
    __syncthreads();

    // First thread of each output group writes both results
    if (lane == 0) {
        float result0 = smem[local_out * 4]     + smem[local_out * 4 + 2];
        float result1 = smem[local_out * 4 + 1] + smem[local_out * 4 + 3];
        C[n]  = __float2bfloat16(result0);
        C1[n] = __float2bfloat16(result1);
    }
}

// ============================================================
// W4A16 batched GEMV (M<=MAX_M) — the NVFP4 sibling of w8a16_gemv_batch4/16.
// ============================================================
// At M-token batched decode the SSM QKVZ / out_proj projections share the same
// NVFP4 weight matrix across all M sequences, so a SINGLE DRAM pass over the
// packed 4-bit weight (dequantized E2M1*scale ONCE) serves all M rows — MAC'd
// into M independent FP32 accumulators. This is what lets FP4 amortize the
// weight read the way w8a16_gemv_batch4/16 does for FP8; without it the FP4
// multi-seq path capped at batch3 and re-streamed the weight ~3x at C=8.
//
// Per-row accumulation order is IDENTICAL to `w4a16_gemv` (M=1), so the output
// is bit-identical to running w4a16_gemv M times.
// A:[M,K] BF16, B_packed:[N,K/2], B_scale:[N,K/16] FP8-E4M3, scale2 FP32,
// C:[M,N] BF16. Grid: (ceil(N/4),1,1) Block: (256,1,1).
template <int MAX_M>
__device__ __forceinline__ void w4a16_gemv_batchm_impl(
    const __nv_bfloat16* __restrict__ A,         // [M, K]
    const unsigned char* __restrict__ B_packed,   // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,    // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,                // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
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

    float acc[MAX_M];
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        // 8 packed weight bytes (16 FP4) + 1 group scale → dequant ONCE.
        unsigned long long packed8 =
            *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + k16 * 8);
        const unsigned int scale_group = base_k / GROUP_SIZE;  // == k16 (GROUP_SIZE=16)
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif
        float wf[16];
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
            wf[b * 2]     = s_lut[byte_val & 0xF] * scale;   // W[2j]   <-> act 2j
            wf[b * 2 + 1] = s_lut[byte_val >> 4] * scale;    // W[2j+1] <-> act 2j+1
        }

        // Reuse the scaled weights across each activation row.
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            const __nv_bfloat16* At = A + (unsigned long long)t * K;
            uint4 a_lo = ((const uint4*)At)[k16 * 2];
            uint4 a_hi = ((const uint4*)At)[k16 * 2 + 1];
            const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                __nv_bfloat16 lo, hi;
                *(unsigned short*)&lo = (unsigned short)(ar[j] & 0xFFFF);
                *(unsigned short*)&hi = (unsigned short)(ar[j] >> 16);
                acc[t] += __bfloat162float(lo) * wf[j * 2]
                        + __bfloat162float(hi) * wf[j * 2 + 1];
            }
        }
    }

    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];  // 2 warps/output, per row
    const unsigned int warp_in_out = lane / WARP_SIZE;
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        if ((unsigned int)t >= M) continue;
        float a = acc[t];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        if (lane % WARP_SIZE == 0) smem[t][local_out * 2 + warp_in_out] = a;
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * N + n] = __float2bfloat16(r);
        }
    }
}

// M<=4 (common-path batched decode) — sibling of w8a16_gemv_batch4.
extern "C" __global__ void w4a16_gemv_batch4(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<4>(A, B_packed, B_scale, scale2, C, M, N, K);
}

// M<=16 (high-concurrency decode, n=5..16) — sibling of w8a16_gemv_batch16.
extern "C" __global__ void w4a16_gemv_batch16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<16>(A, B_packed, B_scale, scale2, C, M, N, K);
}

// ============================================================
// W4A16 GEMV with inline Q/Gate deinterleave on output write
// ============================================================
// Same GEMV as w4a16_gemv but writes Q and Gate to separate halves.
// Eliminates the separate deinterleave_qg kernel (saves 12 graph nodes).
//
// Input layout (interleaved per head): [Q_h0(hd), G_h0(hd), Q_h1(hd), G_h1(hd), ...]
// Output layout (deinterleaved): [Q_h0..Q_nh | G_h0..G_nh]
//
// N = num_heads * head_dim * 2  (total Q+Gate elements)
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qg(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [Q | G] deinterleaved
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;
        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;
            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo) * w_lo;
            acc += __bfloat162float(a_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / WARP_SIZE)] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];

        // Deinterleave: n indexes interleaved [Q_h0(hd), G_h0(hd), Q_h1(hd), ...]
        // head = n / (2 * head_dim), is_gate = (n % (2 * head_dim)) >= head_dim
        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;             // Q region
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);  // Gate region
        }
        C[out_idx] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 GEMV with inline QKVZ deinterleave on output write
// ============================================================
// Same GEMV as w4a16_gemv but writes to deinterleaved output locations.
// Eliminates the separate deinterleave_qkvz kernel (saves 36 graph nodes).
//
// QKVZ interleaved layout (N=12288, 16 groups of 768):
//   Group g: [Q_{g*128..128} | K_{g*128..128} | V_{g*256..256} | Z_{g*256..256}]
//
// Deinterleaved output: [Q_2048 | K_2048 | V_4096 | Z_4096]
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qkvz(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [Q|K|V|Z] deinterleaved
    unsigned int N,
    unsigned int K,
    // Deinterleave params:
    unsigned int num_groups,        // 16
    unsigned int head_k_dim,        // 128
    unsigned int vheads_per_group,  // 2
    unsigned int head_v_dim         // 128
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups_k = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;
        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups_k + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;
            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo) * w_lo;
            acc += __bfloat162float(a_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / WARP_SIZE)] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];

        // Compute deinterleaved output index
        unsigned int v_group_size = vheads_per_group * head_v_dim;
        unsigned int group_dim = 2 * head_k_dim + 2 * v_group_size;
        unsigned int g = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_groups * head_k_dim;
        unsigned int k_total = num_groups * head_k_dim;

        unsigned int out_idx;
        if (idx < head_k_dim) {
            out_idx = g * head_k_dim + idx;
        } else if (idx < 2 * head_k_dim) {
            out_idx = q_total + g * head_k_dim + (idx - head_k_dim);
        } else if (idx < 2 * head_k_dim + v_group_size) {
            out_idx = q_total + k_total + g * v_group_size + (idx - 2 * head_k_dim);
        } else {
            out_idx = q_total + k_total + num_groups * v_group_size
                    + g * v_group_size + (idx - 2 * head_k_dim - v_group_size);
        }
        C[out_idx] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 GEMV batch2 with inline Q/Gate deinterleave
// ============================================================
// Combines w4a16_gemv_batch2 (2-input) with w4a16_gemv_qg (deinterleave).
// Reads Q+Gate weight matrix once for 2 input tokens, produces 2 deinterleaved
// output vectors [Q_all | Gate_all] per token.
//
// Input:  A[2, K] BF16 (2 token hidden states)
// Output: C[2, N] BF16 (deinterleaved: C[0] = [Q0|G0], C[1] = [Q1|G1])
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qg_batch2(
    const __nv_bfloat16* __restrict__ A,        // [2, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [2, N] deinterleaved [Q|G] per token
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    const __nv_bfloat16* __restrict__ A1 = A + K;
    __nv_bfloat16* __restrict__ C1 = C + N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 4];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 4 + warp_idx * 2]     = acc0;
        smem[local_out * 4 + warp_idx * 2 + 1] = acc1;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 4]     + smem[local_out * 4 + 2];
        float result1 = smem[local_out * 4 + 1] + smem[local_out * 4 + 3];

        // Deinterleave: n indexes interleaved [Q_h0(hd), G_h0(hd), Q_h1(hd), ...]
        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);
        }
        C[out_idx]  = __float2bfloat16(result0);
        C1[out_idx] = __float2bfloat16(result1);
    }
}

// ============================================================
// W4A16 GEMV dual batch2: K+V for 2 input tokens in one launch
// ============================================================
// Processes 2 separate weight matrices (K and V) with 2 input vectors each.
// blockIdx.z selects K (0) or V (1). Both projections compute 2 outputs.
//
// Input:  A[2, K_in] BF16 (2 token hidden states)
// Output: C[2, N] where blockIdx.z=0 writes K, blockIdx.z=1 writes V
//
// Grid: (ceil(N / 4), 1, 2)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_dual_batch2(
    const __nv_bfloat16* __restrict__ A,         // [2, K_in] BF16
    const unsigned char* __restrict__ B0_packed,  // [N, K_in/2] first projection
    const unsigned char* __restrict__ B0_scale,
    float B0_scale2,
    __nv_bfloat16* __restrict__ C0,              // [2, N] first projection output
    const unsigned char* __restrict__ B1_packed,  // [N, K_in/2] second projection
    const unsigned char* __restrict__ B1_scale,
    float B1_scale2,
    __nv_bfloat16* __restrict__ C1,              // [2, N] second projection output
    unsigned int N,
    unsigned int K_in
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = (proj == 0) ? B0_packed : B1_packed;
    const unsigned char* B_scale = (proj == 0) ? B0_scale : B1_scale;
    float s2 = (proj == 0) ? B0_scale2 : B1_scale2;
    __nv_bfloat16* C_out = (proj == 0) ? C0 : C1;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K_in / 2;
    const unsigned int num_groups = K_in / GROUP_SIZE;
    const unsigned int K8 = K_in / 8;

    const __nv_bfloat16* A1 = A + K_in;
    __nv_bfloat16* C_out1 = C_out + N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 4];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb = B_scale[(unsigned long long)n * num_groups + sg];
        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(sb) * s2;
#else
        float scale = (float)fp8 * s2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[bv & 0xF] * scale;
            float w_hi = s_lut[bv >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 4 + warp_idx * 2]     = acc0;
        smem[local_out * 4 + warp_idx * 2 + 1] = acc1;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 4]     + smem[local_out * 4 + 2];
        float result1 = smem[local_out * 4 + 1] + smem[local_out * 4 + 3];
        C_out[n]  = __float2bfloat16(result0);
        C_out1[n] = __float2bfloat16(result1);
    }
}

// ============================================================
// W4A16 triple-GEMV (M=3): reads weights once, computes 3 outputs
// ============================================================
// For K=3 speculative verification: processes 3 input vectors through
// the same weight matrix in a single pass.
//
// A: [3, K] BF16 contiguous (row 0, 1, 2)
// B: [N, K/2] NVFP4 packed weights
// C: [3, N] BF16 contiguous (row 0, 1, 2)
//
// Same memory bandwidth as M=1 GEMV (weights dominate, read once).
// Extra cost: 3x activation reads (K*2 bytes per vector, fits in L1/L2).
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_batch3(
    const __nv_bfloat16* __restrict__ A,        // [3, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [3, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    const __nv_bfloat16* __restrict__ A1 = A + K;
    const __nv_bfloat16* __restrict__ A2 = A + 2 * K;
    __nv_bfloat16* __restrict__ C1 = C + N;
    __nv_bfloat16* __restrict__ C2 = C + 2 * N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 6];  // 2 warps × 3 accumulators per output
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo + __bfloat162float(a2_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float result1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float result2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];
        C[n]  = __float2bfloat16(result0);
        C1[n] = __float2bfloat16(result1);
        C2[n] = __float2bfloat16(result2);
    }
}

// ============================================================
// W4A16 GEMV batch3 with inline Q/Gate deinterleave
// ============================================================
// Combines w4a16_gemv_batch3 (3-input) with Q/Gate deinterleave.
// Reads Q+Gate weight matrix once for 3 input tokens, produces 3 deinterleaved
// output vectors [Q_all | Gate_all] per token.
//
// Input:  A[3, K] BF16 (3 token hidden states)
// Output: C[3, N] BF16 (deinterleaved: C[i] = [Qi|Gi])
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qg_batch3(
    const __nv_bfloat16* __restrict__ A,        // [3, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [3, N] deinterleaved [Q|G] per token
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    const __nv_bfloat16* __restrict__ A1 = A + K;
    const __nv_bfloat16* __restrict__ A2 = A + 2 * K;
    __nv_bfloat16* __restrict__ C1 = C + N;
    __nv_bfloat16* __restrict__ C2 = C + 2 * N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 6];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo + __bfloat162float(a2_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float result1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float result2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];

        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);
        }
        C[out_idx]  = __float2bfloat16(result0);
        C1[out_idx] = __float2bfloat16(result1);
        C2[out_idx] = __float2bfloat16(result2);
    }
}

// ============================================================
// W4A16 GEMV dual batch3: K+V for 3 input tokens in one launch
// ============================================================
// Processes 2 separate weight matrices (K and V) with 3 input vectors each.
// blockIdx.z selects K (0) or V (1). Both projections compute 3 outputs.
//
// Input:  A[3, K_in] BF16 (3 token hidden states)
// Output: C[3, N] where blockIdx.z=0 writes K, blockIdx.z=1 writes V
//
// Grid: (ceil(N / 4), 1, 2)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_dual_batch3(
    const __nv_bfloat16* __restrict__ A,         // [3, K_in] BF16
    const unsigned char* __restrict__ B0_packed,  // [N, K_in/2] first projection
    const unsigned char* __restrict__ B0_scale,
    float B0_scale2,
    __nv_bfloat16* __restrict__ C0,              // [3, N] first projection output
    const unsigned char* __restrict__ B1_packed,  // [N, K_in/2] second projection
    const unsigned char* __restrict__ B1_scale,
    float B1_scale2,
    __nv_bfloat16* __restrict__ C1,              // [3, N] second projection output
    unsigned int N,
    unsigned int K_in
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = (proj == 0) ? B0_packed : B1_packed;
    const unsigned char* B_scale = (proj == 0) ? B0_scale : B1_scale;
    float s2 = (proj == 0) ? B0_scale2 : B1_scale2;
    __nv_bfloat16* C_out = (proj == 0) ? C0 : C1;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K_in / 2;
    const unsigned int num_groups = K_in / GROUP_SIZE;
    const unsigned int K8 = K_in / 8;

    const __nv_bfloat16* A1 = A + K_in;
    const __nv_bfloat16* A2 = A + 2 * K_in;
    __nv_bfloat16* C_out1 = C_out + N;
    __nv_bfloat16* C_out2 = C_out + 2 * N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 6];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb = B_scale[(unsigned long long)n * num_groups + sg];
        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(sb) * s2;
#else
        float scale = (float)fp8 * s2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[bv & 0xF] * scale;
            float w_hi = s_lut[bv >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo + __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo + __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo + __bfloat162float(a2_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float result1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float result2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];
        C_out[n]  = __float2bfloat16(result0);
        C_out1[n] = __float2bfloat16(result1);
        C_out2[n] = __float2bfloat16(result2);
    }
}
