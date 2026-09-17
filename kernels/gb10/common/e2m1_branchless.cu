// SPDX-License-Identifier: AGPL-3.0-only

// Atlas E2M1 branchless conversion kernel.
//
// Converts FP32 activations to E2M1 (4-bit) using 7 unsigned integer comparisons.
// Zero branches, zero divergence. Runs entirely on integer ALU.
//
// IEEE 754 bit-pattern thresholds for E2M1 rounding boundaries:
//   0x3E800000 = 0.25   (boundary between 0.0 and 0.5)
//   0x3F400000 = 0.75   (boundary between 0.5 and 1.0)
//   0x3FA00000 = 1.25   (boundary between 1.0 and 1.5)
//   0x3FE00000 = 1.75   (boundary between 1.5 and 2.0)
//   0x40200000 = 2.5    (boundary between 2.0 and 3.0)
//   0x40600000 = 3.5    (boundary between 3.0 and 4.0)
//   0x40A00000 = 5.0    (boundary between 4.0 and 6.0)

__device__ __forceinline__ unsigned char branchless_float_to_e2m1(float x) {
    unsigned char sign = (unsigned char)((__float_as_uint(x) >> 28) & 8u);
    unsigned int abits = __float_as_uint(x) & 0x7FFFFFFFu;
    unsigned char mag = (abits >  0x3E800000u)
                      + (abits >= 0x3F400000u)
                      + (abits >  0x3FA00000u)
                      + (abits >= 0x3FE00000u)
                      + (abits >  0x40200000u)
                      + (abits >= 0x40600000u)
                      + (abits >  0x40A00000u);
    return sign | mag;
}

// Pack 8 E2M1 values into one uint32_t (4 bits each)
__device__ __forceinline__ unsigned int pack_8xe2m1(
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7
) {
    unsigned int val = 0;
    val |= (unsigned int)branchless_float_to_e2m1(f0);
    val |= (unsigned int)branchless_float_to_e2m1(f1) << 4;
    val |= (unsigned int)branchless_float_to_e2m1(f2) << 8;
    val |= (unsigned int)branchless_float_to_e2m1(f3) << 12;
    val |= (unsigned int)branchless_float_to_e2m1(f4) << 16;
    val |= (unsigned int)branchless_float_to_e2m1(f5) << 20;
    val |= (unsigned int)branchless_float_to_e2m1(f6) << 24;
    val |= (unsigned int)branchless_float_to_e2m1(f7) << 28;
    return val;
}

// Quantize N floats to E2M1 (packed uint32_t output, 8 values per word)
extern "C" __global__ void e2m1_quantize(
    const float* __restrict__ input,
    unsigned int* __restrict__ output,
    unsigned int n  // number of float elements (must be multiple of 8)
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx + 7 < n) {
        output[idx / 8] = pack_8xe2m1(
            input[idx], input[idx+1], input[idx+2], input[idx+3],
            input[idx+4], input[idx+5], input[idx+6], input[idx+7]
        );
    }
}
