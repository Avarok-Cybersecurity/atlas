// SPDX-License-Identifier: AGPL-3.0-only
//
// W4A8 integer-DP4A decode GEMV for AMD gfx1151 (RDNA3.5 / Strix Halo).
//
// This is the strix-hip-only DP4A decode path. It is ADDITIVE: the float
// E2M1-LUT path in w4a16_gemv.cu is untouched and remains the gb10/NVIDIA
// default. Dispatch selects these kernels only on strix-hip behind a flag.
//
// Why: Atlas decode GEMV currently dequantizes 4-bit weights to float and
// accumulates in FP32 against BF16 activations (one FMA per weight). On the
// bandwidth-bound LPDDR5X (~256 GB/s) gfx1151 part the win comes from (a) cutting
// activation traffic (int8 not bf16) and (b) replacing 4 FP32 FMAs with one
// hardware `v_dot4` (`__builtin_amdgcn_sudot4`) — 4 int8xint8 MACs/instruction.
//
// FAITHFULNESS to the float path: the NVFP4 weight codebook E2M1_LUT =
// {0,.5,1,1.5,2,3,4,6} is exactly representable as half-integers; multiply by 2
// to get the integer grid {0,1,2,3,4,6,8,12} (+ negatives) and fold the x0.5 into
// the per-group scale. Weights are therefore EXACT in int8; the ONLY new error vs
// W4A16 is the int8 activation quantization (block-q8_1 style, d = amax/127). This
// is the same accuracy/speed trade the rocmfp4-llama reference ships at BFCL parity.
//
//   float path : acc += bf16(a_i) * (E2M1_LUT[nib_i] * wscale_g * scale2)
//   dp4a  path : acc += a_d_g * (wscale_g * 0.5 * scale2) * SUM_i( aq_i * wint_i )
//                where aq_i = round(a_i / a_d_g), a_d_g = amax_g/127,
//                      wint_i = round(E2M1_LUT[nib_i] * 2)   (exact integer)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Standard E4M3 (1-4-3, bias 7) decode via bit-math — IDENTICAL to w4a16_gemv.cu's
// scl_fp8 (SSOT: the block-scale decode must match the encoder; gfx1151's builtin
// __nv_fp8_e4m3 cast is a non-standard narrow format and corrupts scales).
__device__ __forceinline__ float dp4a_scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)                  v = (float)m * 0.001953125f;                 // subnormal m*2^-9
    else if (e == 15u && m == 7u) v = 0.0f;                                    // NaN -> 0
    else                          v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}

// Integer NVFP4 codebook = E2M1_LUT * 2 (exact). Index = 4-bit nibble (sign in bit3).
__device__ __constant__ signed char DP4A_CODEBOOK[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12
};

#define DP4A_BLOCK_SIZE 256
#define DP4A_N_PER_BLOCK 4
#define DP4A_WARP_SIZE 32
#define DP4A_GROUP_SIZE 16   // one weight scale + one act scale per 16 elements

#if defined(__HIP_PLATFORM_AMD__) || defined(__SCALE__)
#define DP4A_DOT(a, b, c) __builtin_amdgcn_sudot4(true, (a), true, (b), (c), false)
#else
#define DP4A_DOT(a, b, c) __dp4a((a), (b), (c))   // NVIDIA fallback (parity / portability)
#endif

// ── Activation int8 quantizer (once per layer, NOT per GEMV) ──────────────
// Quantizes one BF16 activation row [1,K] to int8 [1,K] with per-16-group symmetric
// scales [K/16]. Grid: (K/16) blocks, block: 16 threads (one per group element).
extern "C" __global__ void quantize_act_int8_g16(
    const __nv_bfloat16* __restrict__ A,   // [1, K] BF16
    signed char* __restrict__ a_q,          // [1, K] int8
    float* __restrict__ a_scale,            // [K/16] f32 per-group scale (= amax/127)
    unsigned int K
) {
    const unsigned int g = blockIdx.x;
    const unsigned int i = g * DP4A_GROUP_SIZE + threadIdx.x;
    if (i >= K) return;

    float a = __bfloat162float(A[i]);
    float aa = fabsf(a);

    // amax over the 16-element group (warp-style reduction over 16 lanes via smem).
    __shared__ float s_amax[DP4A_GROUP_SIZE];
    s_amax[threadIdx.x] = aa;
    __syncthreads();
    #pragma unroll
    for (unsigned int off = DP4A_GROUP_SIZE / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            float o = s_amax[threadIdx.x + off];
            if (o > s_amax[threadIdx.x]) s_amax[threadIdx.x] = o;
        }
        __syncthreads();
    }
    float amax = s_amax[0];
    float d = amax * (1.0f / 127.0f);
    float inv = (d > 0.0f) ? (1.0f / d) : 0.0f;

    int q = (int)rintf(a * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    a_q[i] = (signed char)q;
    if (threadIdx.x == 0) a_scale[g] = d;
}

// ── DP4A GEMV (correctness v1: element-order codebook expansion) ──────────
// out[n] = SUM_g a_scale[g] * (wscale_g * 0.5 * scale2) * dot(aq[g], wint[g])
extern "C" __global__ void w4a16_gemv_dp4a(
    const signed char* __restrict__ a_q,    // [1, K] int8 activations
    const float* __restrict__ a_scale,      // [K/16] f32 act scales
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8 (2 nibbles/byte)
    const unsigned char* __restrict__ B_scale,    // [N, K/16] FP8-E4M3 weight scales
    const float scale2,
    __nv_bfloat16* __restrict__ C,           // [1, N] BF16 out
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = DP4A_BLOCK_SIZE / DP4A_N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * DP4A_N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / DP4A_GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ signed char s_cb[16];
    __shared__ float smem[DP4A_N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_cb[threadIdx.x] = DP4A_CODEBOOK[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;
    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        // 16 int8 activations = one int4 (16 bytes), in element order.
        int4 aq4 = *(const int4*)(a_q + base_k);
        const int aq[4] = {aq4.x, aq4.y, aq4.z, aq4.w};

        // 8 packed weight bytes (16 nibbles).
        unsigned long long packed8 = *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + k16 * 8);

        // Expand to 16 int8 codebook values in ELEMENT order so they align lane-wise
        // with aq[]: element 2b = byte b low nibble, element 2b+1 = byte b high nibble
        // (matches the float w4a16_gemv pairing).
        int wint[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned int packed = 0;
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int elem = j * 4 + e;              // 0..15
                int b = elem >> 1;                 // byte index
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                unsigned char nib = (elem & 1) ? (byte_val >> 4) : (byte_val & 0xF);
                packed |= ((unsigned int)(unsigned char)s_cb[nib]) << (e * 8);
            }
            wint[j] = (int)packed;
        }

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) sumi = DP4A_DOT(aq[j], wint[j], sumi);

        unsigned int scale_group = base_k / DP4A_GROUP_SIZE;
        float wscale = dp4a_scl_fp8(B_scale[(unsigned long long)n * num_groups + scale_group]);
        float a_d = a_scale[k16];
        acc += (float)sumi * a_d * (wscale * 0.5f * scale2);
    }

    const unsigned int warp_lane = threadIdx.x % DP4A_WARP_SIZE;
    #pragma unroll
    for (int offset = DP4A_WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (warp_lane == 0) smem[local_out * 2 + (lane / DP4A_WARP_SIZE)] = acc;
    __syncthreads();
    if (lane == 0) C[n] = __float2bfloat16(smem[local_out * 2] + smem[local_out * 2 + 1]);
}
