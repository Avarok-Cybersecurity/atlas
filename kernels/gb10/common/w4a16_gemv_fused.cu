// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 GEMV Fused — dual projection + silu-input variants.
//
// Reduces shared expert kernels from 4 to 2 per layer (saves 96 launches total):
//   Before: gate (1) + up (1) + silu_mul (1) + down (1) = 4 per layer × 48 = 192
//   After:  gate_up_dual (1) + silu_down (1) = 2 per layer × 48 = 96
//
// w4a16_gemv_dual: blockIdx.z selects projection 0 vs 1.
//   Both projections share the same BF16 input A[1, K].
//   Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
//
// w4a16_gemv_silu_input: reads gate_out + up_out BF16 vectors, computes
//   silu(gate)*up inline as activation, then GEMV with NVFP4 down weights.
//   Eliminates separate silu_mul kernel entirely.
//   Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)

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

__device__ __constant__ float E2M1_LUT_FUSED_W4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
// ── HIP / SCALE portability ──────────────────────────────────────────────────
// This file is SYMLINKED into kernels/strix-hip/common/ and kernels/strix/common/,
// so hipcc compiles it verbatim. `__syncwarp()` is a CUDA intrinsic that target
// does not accept: the strix-hip fork of w4a16_gemm.cu contains zero
// `__syncwarp` and the tree carries no shim. Introducing it unguarded broke the
// windows-x86_64-amd-hip release build (#519).
//
// NOTE the block-wide staging elsewhere in this file (`__shared__ s_lut` +
// `__syncthreads()`, 9 pre-existing sites) is FINE on HIP and is untouched.
// Only the WARP-scoped publish is CUDA-only.
//
// On AMD the warp-scoped path keeps the PRE-EXISTING behaviour: no staging, the
// partial indexes `__constant__ E2M1_LUT` directly, exactly as shipped before
// this change. Provably compilable (it is the shipped code) and provably
// correct; it forgoes the staging win on a target we cannot measure. Doing
// better needs a wave-scoped publish (`__builtin_amdgcn_wave_barrier()` plus
// the right fence) VALIDATED on gfx1151 — no such box exists here, so it is not
// guessed at, for the same reason the strix-hip batched-prefill argument shift
// was left to someone with the hardware.
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
#define ATLAS_WARP_LUT_STAGED 0
#else
#define ATLAS_WARP_LUT_STAGED 1
#endif


// Stage the E2M1 table in SHARED memory before the decode inner loop. Indexing
// __constant__ memory with a data-dependent weight nibble serializes the warp
// (broadcast cache: one replay per distinct address, ~14 of 16 entries live);
// shared memory answers 16 divergent indices from 16 banks in one transaction.
// Bit-exact copy of the constant table, so no fmaf operand changes. Full
// rationale: `w4a16_gemv.cu` / `w8a16_gemm_pipelined.cu` Lever 1.
//
// WARP-SCOPED, for the single-warp kernels: they early-return on a warp-uniform
// `n >= N`, so __syncthreads() here would be a divergent barrier. Their
// reduction stays barrier-free, as documented.
__device__ __forceinline__ void stage_e2m1_lut_fused_warp(float* s_lut, unsigned int lane) {
#if ATLAS_WARP_LUT_STAGED
    if (lane < 16u) s_lut[lane] = E2M1_LUT_FUSED_W4[lane];
    __syncwarp();
#else
    (void)s_lut; (void)lane;   // AMD: nothing staged; callers pass the constant LUT
#endif
}

// One orig-lane's pipelined K16 partial. Shared by `w4a16_gemv_dual` and
// `w4a16_gemv_dual_sw` so the SW path cannot drift off the 2-chunk association.
__device__ __forceinline__ float w4a16_dual_partial(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    unsigned int n, unsigned int half_K, unsigned int num_groups,
    unsigned int K16, unsigned int orig_lane,
    const float* __restrict__ lut)   // shared-staged E2M1 copy; see stage_e2m1_lut_fused_warp
{
    float acc0 = 0.0f, acc1 = 0.0f;
    const unsigned int stride2 = 128u;
    for (unsigned int k16 = orig_lane * 2u; k16 < K16 + 1u; k16 += stride2) {
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            const unsigned int kk = k16 + (unsigned int)c;
            if (kk >= K16) break;

            uint4 a_lo4 = ((const uint4*)A)[kk * 2];
            uint4 a_hi4 = ((const uint4*)A)[kk * 2 + 1];
            const unsigned int a_raw[8] = {a_lo4.x, a_lo4.y, a_lo4.z, a_lo4.w,
                                            a_hi4.x, a_hi4.y, a_hi4.z, a_hi4.w};
            unsigned long long packed8 = *(const unsigned long long*)(
                B_packed + (unsigned long long)n * half_K + kk * 8);
            unsigned char scale_byte = B_scale[
                (unsigned long long)n * num_groups + kk];
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8(scale_byte) * scale2;
#else
            float scale = (float)fp8 * scale2;
#endif
            float part = 0.0f;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&a_raw[b]);
                part = fmaf(af.x, lut[byte_val & 0xF], part);
                part = fmaf(af.y, lut[byte_val >> 4], part);
            }
            if (c == 0) acc0 = fmaf(scale, part, acc0);
            else        acc1 = fmaf(scale, part, acc1);
        }
    }
    return acc0 + acc1;
}

// ── W4A16 GEMV Dual Projection ──
//
// blockIdx.z = 0: first projection (gate), blockIdx.z = 1: second (up).
// Both read same shared BF16 input A[1, K] with different NVFP4 weights.
// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_dual(
    const __nv_bfloat16* __restrict__ A,           // [1, K] shared input
    const unsigned char* __restrict__ B1_packed,    // [N, K/2] proj 0 weights
    const unsigned char* __restrict__ B1_scale,     // [N, K/GROUP_SIZE] proj 0
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,                 // [1, N] proj 0 output
    const unsigned char* __restrict__ B2_packed,    // [N, K/2] proj 1 weights
    const unsigned char* __restrict__ B2_scale,     // [N, K/GROUP_SIZE] proj 1
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,                 // [1, N] proj 1 output
    unsigned int N,
    unsigned int K
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? B1_packed : B2_packed;
    const unsigned char* B_scale = proj == 0 ? B1_scale : B2_scale;
    float scale2 = proj == 0 ? scale2_1 : scale2_2;
    __nv_bfloat16* C = proj == 0 ? C1 : C2;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    // Block-staged here (all 256 threads reach this barrier — no early return).
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_W4[threadIdx.x];
    __syncthreads();

    // Do not return early: N%4!=0 tail warps must still hit the smem barrier.
    float acc = 0.0f;
    if (n < N) {
        acc = w4a16_dual_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane, s_lut);
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0 && n < N) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// ── W4A16 GEMV with SiLU-fused Input ──
//
// Reads gate_out[K] and up_out[K] BF16, computes silu(gate)*up inline
// as the activation, then GEMV with NVFP4 down weights.
// Eliminates the separate silu_mul kernel entirely.
// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_silu_input(
    const __nv_bfloat16* __restrict__ gate_out,    // [1, K] gate proj output
    const __nv_bfloat16* __restrict__ up_out,      // [1, K] up proj output
    const unsigned char* __restrict__ B_packed,     // [N, K/2] down weights
    const unsigned char* __restrict__ B_scale,      // [N, K/GROUP_SIZE]
    const float scale2,
    __nv_bfloat16* __restrict__ C,                  // [1, N] output
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
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_W4[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 g_data = ((const uint4*)gate_out)[k8];
        uint4 u_data = ((const uint4*)up_out)[k8];

        unsigned int packed4 = *(const unsigned int*)(
            B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[
            (unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        const unsigned int g_raw[4] = {g_data.x, g_data.y, g_data.z, g_data.w};
        const unsigned int u_raw[4] = {u_data.x, u_data.y, u_data.z, u_data.w};

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 g_lo, g_hi;
            *(unsigned short*)&g_lo = (unsigned short)(g_raw[b] & 0xFFFF);
            *(unsigned short*)&g_hi = (unsigned short)(g_raw[b] >> 16);
            float gf_lo = __bfloat162float(g_lo);
            float gf_hi = __bfloat162float(g_hi);

            __nv_bfloat16 u_lo, u_hi;
            *(unsigned short*)&u_lo = (unsigned short)(u_raw[b] & 0xFFFF);
            *(unsigned short*)&u_hi = (unsigned short)(u_raw[b] >> 16);

            // SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
            float a_lo = (gf_lo / (1.0f + __expf(-gf_lo))) * __bfloat162float(u_lo);
            float a_hi = (gf_hi / (1.0f + __expf(-gf_hi))) * __bfloat162float(u_hi);

            acc += a_lo * w_lo;
            acc += a_hi * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// ════════════════════════════════════════════════════════════════════
// SINGLE-WARP-PER-OUTPUT variants (lossless; default ON, kill ATLAS_NO_GEMV_SW=1).
//
// Dual SW shares `w4a16_dual_partial` with the 64-thread dual kernel.
// SiLU SW still matches the K8 sequential silu_input kernel (that base is
// not yet pipelined). Grid: (ceil(N/8),1,z).
// ════════════════════════════════════════════════════════════════════

#define N_PER_BLOCK_SW 8

extern "C" __global__ void w4a16_gemv_dual_sw(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    const float scale2_1,
    __nv_bfloat16* __restrict__ C1,
    const unsigned char* __restrict__ B2_packed,
    const unsigned char* __restrict__ B2_scale,
    const float scale2_2,
    __nv_bfloat16* __restrict__ C2,
    unsigned int N,
    unsigned int K
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = proj == 0 ? B1_packed : B2_packed;
    const unsigned char* B_scale = proj == 0 ? B1_scale : B2_scale;
    float scale2 = proj == 0 ? scale2_1 : scale2_2;
    __nv_bfloat16* C = proj == 0 ? C1 : C2;

    const unsigned int local_out = threadIdx.x / WARP_SIZE;  // 0..7
    const unsigned int lane = threadIdx.x % WARP_SIZE;       // 0..31
    const unsigned int n = blockIdx.x * N_PER_BLOCK_SW + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    // One private copy per warp (8 x 16 floats = 512 B/block); no block barrier.
    __shared__ float s_lut[N_PER_BLOCK_SW][16];
    stage_e2m1_lut_fused_warp(s_lut[local_out], lane);
#if ATLAS_WARP_LUT_STAGED
    const float* __restrict__ warp_lut = s_lut[local_out];
#else
    const float* __restrict__ warp_lut = E2M1_LUT_FUSED_W4;
#endif

    float acc_a = w4a16_dual_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane, warp_lut);
    float acc_b = w4a16_dual_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane + 32u, warp_lut);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xFFFFFFFF, acc_a, offset);
        acc_b += __shfl_down_sync(0xFFFFFFFF, acc_b, offset);
    }
    if (lane == 0) {
        C[n] = __float2bfloat16(acc_a + acc_b);
    }
}

// One K8-strided partial for the SiLU-fused-input down kernel.
__device__ __forceinline__ float w4a16_silu_partial(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    unsigned int n, unsigned int half_K, unsigned int num_groups,
    unsigned int K8, unsigned int start_chunk,
    const float* __restrict__ lut)   // shared-staged E2M1 copy; see stage_e2m1_lut_fused_warp
{
    float acc = 0.0f;
    for (unsigned int k8 = start_chunk; k8 < K8; k8 += 64u) {
        const unsigned int base_k = k8 * 8;
        uint4 g_data = ((const uint4*)gate_out)[k8];
        uint4 u_data = ((const uint4*)up_out)[k8];
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
        const unsigned int g_raw[4] = {g_data.x, g_data.y, g_data.z, g_data.w};
        const unsigned int u_raw[4] = {u_data.x, u_data.y, u_data.z, u_data.w};
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = lut[byte_val & 0xF] * scale;
            float w_hi = lut[byte_val >> 4] * scale;
            __nv_bfloat16 g_lo, g_hi;
            *(unsigned short*)&g_lo = (unsigned short)(g_raw[b] & 0xFFFF);
            *(unsigned short*)&g_hi = (unsigned short)(g_raw[b] >> 16);
            float gf_lo = __bfloat162float(g_lo);
            float gf_hi = __bfloat162float(g_hi);
            __nv_bfloat16 u_lo, u_hi;
            *(unsigned short*)&u_lo = (unsigned short)(u_raw[b] & 0xFFFF);
            *(unsigned short*)&u_hi = (unsigned short)(u_raw[b] >> 16);
            float a_lo = (gf_lo / (1.0f + __expf(-gf_lo))) * __bfloat162float(u_lo);
            float a_hi = (gf_hi / (1.0f + __expf(-gf_hi))) * __bfloat162float(u_hi);
            acc += a_lo * w_lo;
            acc += a_hi * w_hi;
        }
    }
    return acc;
}

extern "C" __global__ void w4a16_gemv_silu_input_sw(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    const unsigned int local_out = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK_SW + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    // One private copy per warp (8 x 16 floats = 512 B/block); no block barrier.
    __shared__ float s_lut[N_PER_BLOCK_SW][16];
    stage_e2m1_lut_fused_warp(s_lut[local_out], lane);
#if ATLAS_WARP_LUT_STAGED
    const float* __restrict__ warp_lut = s_lut[local_out];
#else
    const float* __restrict__ warp_lut = E2M1_LUT_FUSED_W4;
#endif

    float acc_a = w4a16_silu_partial(gate_out, up_out, B_packed, B_scale, scale2, n, half_K, num_groups, K8, lane, warp_lut);
    float acc_b = w4a16_silu_partial(gate_out, up_out, B_packed, B_scale, scale2, n, half_K, num_groups, K8, lane + 32u, warp_lut);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xFFFFFFFF, acc_a, offset);
        acc_b += __shfl_down_sync(0xFFFFFFFF, acc_b, offset);
    }
    if (lane == 0) {
        C[n] = __float2bfloat16(acc_a + acc_b);
    }
}
