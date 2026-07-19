// SPDX-License-Identifier: AGPL-3.0-only

// GPU dequant: NVFP4 (E2M1 packed + per-group FP8 E4M3 scales + per-tensor
// global scale) → BF16. Replaces the host round-trip in
// `weight_map/fp8_lut.rs::dequant_nvfp4_to_bf16`, which did a D2H of the packed
// weight + an 83M-iteration single-threaded CPU dequant loop + an H2D per
// projection — ~8s per SSM layer on the dense Qwen3.5-27B (the NVFP4→BF16→NVFP4
// fused-qkvz round-trip's real cost). One async kernel instead; bit-for-bit the
// same math as the CPU path.
//
// Layout (HuggingFace / compressed-tensors NVFP4, row-major):
//   packed [N, K/2] uint8   — two E2M1 nibbles per byte, K-dim packed
//   scales [N, K/16] FP8 E4M3 — one scale per group of 16 elements
//   combined_global: f32     — the caller folds the per-tensor global into one
//                              MULTIPLY: compressed-tensors stores a RECIPROCAL
//                              global (pass 1/global), ModelOpt a direct
//                              multiplier (pass global). So dequant is always
//                              value = E2M1_LUT[nibble] * fp8_scale * combined_global.

#include <cuda_bf16.h>

#define DQ_GROUP_SIZE 16

// E2M1 nibble → float. Matches the Rust CPU table in fp8_lut.rs exactly.
// Bits: [sign(1)][exp(2)][mantissa(1)].
__device__ __constant__ float DQ_E2M1_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// FP8 E4M3 byte → float (software decode; matches fp8_e4m3_to_f32 on the host).
// E4M3: 1 sign, 4 exp (bias 7), 3 mantissa. exp==0 → subnormal (man * 2^-9).
// exp==15 & man==7 → NaN → 0 (E4M3 has no inf).
__device__ __forceinline__ float dq_fp8_e4m3_decode(unsigned char b) {
    unsigned int sign = (b >> 7) & 1u;
    unsigned int exp = (b >> 3) & 0xFu;
    unsigned int man = b & 0x7u;
    float val;
    if (exp == 0u) {
        val = (float)man * 0.001953125f;  // 2^-9 per mantissa unit (subnormal)
    } else if (exp == 15u && man == 7u) {
        val = 0.0f;  // NaN
    } else {
        val = (1.0f + (float)man * 0.125f) * exp2f((float)((int)exp - 7));
    }
    return sign ? -val : val;
}

// Grid: (N, 1, 1)  Block: (256, 1, 1) — one block per row, threads stride over K.
extern "C" __global__ void dequant_nvfp4_to_bf16(
    const unsigned char* __restrict__ packed,   // [N, K/2]
    const unsigned char* __restrict__ scales,   // [N, K/16] FP8 E4M3
    __nv_bfloat16* __restrict__ out,            // [N, K]
    float combined_global,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.x;
    if (row >= N) return;

    const unsigned char* row_packed = packed + (unsigned long long)row * (K / 2);
    const unsigned char* row_scale = scales + (unsigned long long)row * (K / DQ_GROUP_SIZE);
    __nv_bfloat16* row_out = out + (unsigned long long)row * K;

    for (unsigned int col = threadIdx.x; col < K; col += blockDim.x) {
        unsigned int g = col / DQ_GROUP_SIZE;
        float s = dq_fp8_e4m3_decode(row_scale[g]) * combined_global;
        unsigned char byte = row_packed[col >> 1];
        // even col = low nibble, odd col = high nibble (matches CPU packing).
        unsigned int nib = (col & 1u) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
        row_out[col] = __float2bfloat16(DQ_E2M1_LUT[nib] * s);
    }
}
