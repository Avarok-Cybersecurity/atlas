// SPDX-License-Identifier: AGPL-3.0-only

// Atlas BF16 → NVFP4 runtime weight quantization.
//
// Two-phase quantization at model load time:
//   Phase 1: nvfp4_global_absmax — find max |weight| across entire matrix
//   Phase 2: quantize_bf16_to_nvfp4 — per-group E2M1 quantization
//
// Output format matches HuggingFace NVFP4 (compressed-tensors):
//   packed:  [N, K/2] uint8 — two E2M1 nibbles per byte (K-dim packing)
//   scales:  [N, K/16] FP8 E4M3 — one scale per group of 16 elements
//   scale2:  scalar FP32 — per-tensor second-level scale
//
// Dequant: weight = E2M1_LUT[nibble] * fp8_scale * scale2

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#define GROUP_SIZE 16

// ── Software float→FP8 E4M3 conversion ──
//
// SM121 (GB10) may not support the cvt.rn.satfinite.e4m3x2.f32 PTX instruction.
// This software fallback uses IEEE 754 bit manipulation.
__device__ unsigned char float_to_fp8_e4m3(float v) {
    unsigned int bits = __float_as_uint(v);
    unsigned int sign = (bits >> 31) & 1;
    int f32_exp = (int)((bits >> 23) & 0xFF) - 127;  // unbiased exponent
    unsigned int f32_man = bits & 0x7FFFFF;  // 23-bit mantissa

    // Handle zero / denorm
    if ((bits & 0x7FFFFFFF) == 0) return (unsigned char)(sign << 7);

    // Clamp to E4M3 range: [-448, 448]
    // E4M3 max normal: exp=14, man=7 → (1+7/8)*2^(14-7) = 1.875*128 = 240...
    // Actually max = (1+7/8)*2^7 = 240. Wait, exp=14, bias=7: 2^(14-7) = 128.
    // (1+7/8)*128 = 240. But E4M3 max is 448 = (1+7/8)*2^8? No, exp=15 man<7.
    // exp=15, man=6: (1+6/8)*2^(15-7) = 1.75*256 = 448. Correct.
    // For NaN: exp=15, man=7 → skip.

    // Saturate: if |v| > 448, clamp
    float absv = fabsf(v);
    if (absv > 448.0f) absv = 448.0f;

    // Recompute from clamped
    bits = __float_as_uint(absv);
    f32_exp = (int)((bits >> 23) & 0xFF) - 127;
    f32_man = bits & 0x7FFFFF;

    int e4m3_exp;
    unsigned int e4m3_man;

    if (f32_exp < -9) {
        // Too small → zero
        return (unsigned char)(sign << 7);
    } else if (f32_exp < -6) {
        // Subnormal in E4M3: val = man * 2^(-9)
        // man = round(absv / 2^(-9)) = round(absv * 512)
        int man = (int)(absv * 512.0f + 0.5f);
        if (man > 7) man = 7;
        if (man < 0) man = 0;
        return (unsigned char)((sign << 7) | man);
    } else {
        // Normal: val = (1 + man/8) * 2^(exp-7)
        e4m3_exp = f32_exp + 7;  // E4M3 bias = 7
        if (e4m3_exp < 1) e4m3_exp = 1;
        if (e4m3_exp > 15) {
            // Overflow → clamp to max
            e4m3_exp = 15;
            e4m3_man = 6;  // max normal (man=7 is NaN)
        } else {
            // Round mantissa: f32 has 23 bits, E4M3 has 3
            e4m3_man = (f32_man + (1 << 19)) >> 20;  // round to nearest
            if (e4m3_man > 7) {
                e4m3_man = 0;
                e4m3_exp++;
                if (e4m3_exp > 15) {
                    e4m3_exp = 15;
                    e4m3_man = 6;
                }
            }
        }
        return (unsigned char)((sign << 7) | (e4m3_exp << 3) | e4m3_man);
    }
}

// ── E2M1 nearest-value quantizer ──
//
// Maps float v to the nearest E2M1 4-bit value (0..15).
// Positive E2M1 values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
// Negative mirror at bit 3 (sign).
__device__ unsigned int quantize_e2m1(float v) {
    float absv = fabsf(v);
    unsigned int sign = (v < 0.0f) ? 8u : 0u;
    unsigned int idx;

    if      (absv <= 0.25f) idx = 0;  // 0.0
    else if (absv <= 0.75f) idx = 1;  // 0.5
    else if (absv <= 1.25f) idx = 2;  // 1.0
    else if (absv <= 1.75f) idx = 3;  // 1.5
    else if (absv <= 2.5f)  idx = 4;  // 2.0
    else if (absv <= 3.5f)  idx = 5;  // 3.0
    else if (absv <= 5.0f)  idx = 6;  // 4.0
    else                    idx = 7;  // 6.0 (saturate)

    return sign | idx;
}

// ── Phase 1: Global absolute maximum ──
//
// Grid: (min(total/256, 1024), 1, 1)  Block: (256, 1, 1)
// Scans entire [N, K] BF16 matrix, atomicMax to single output float.
// Caller must zero-initialize global_max before launch.
extern "C" __global__ void nvfp4_global_absmax(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ global_max,
    unsigned int total_elements
) {
    float local_max = 0.0f;

    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x) {
        float v = fabsf(__bfloat162float(input[idx]));
        if (v > local_max) local_max = v;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }

    // Cross-warp reduction
    __shared__ float smem[8];
    unsigned int warp_id = threadIdx.x / WARP_SIZE;
    unsigned int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_max = 0.0f;
        for (int w = 0; w < (int)(blockDim.x / WARP_SIZE); w++) {
            if (smem[w] > block_max) block_max = smem[w];
        }
        // AtomicMax for non-negative floats via uint (IEEE 754 ordering)
        atomicMax((unsigned int*)global_max, __float_as_uint(block_max));
    }
}

// ── Phase 2: Quantize BF16 → NVFP4 ──
//
// Grid: (N, 1, 1)  Block: (256, 1, 1)
// One block per row. Each thread processes ceil(K/16/256) groups.
//
// scale2 = global_max / (6.0 * 448.0), passed as kernel arg.
// Per-group fp8_scale = group_max / (6.0 * scale2).
extern "C" __global__ void quantize_bf16_to_nvfp4(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    float scale2,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.x;
    if (row >= N) return;

    const __nv_bfloat16* row_in = input + (unsigned long long)row * K;
    unsigned char* row_packed = packed_out + (unsigned long long)row * (K / 2);
    unsigned char* row_scale = scale_out + (unsigned long long)row * (K / GROUP_SIZE);

    float inv_scale2 = (scale2 > 0.0f) ? (1.0f / scale2) : 0.0f;
    unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int base = g * GROUP_SIZE;

        // Find group max absolute value
        float group_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < GROUP_SIZE; i++) {
            float v = fabsf(__bfloat162float(row_in[base + i]));
            if (v > group_max) group_max = v;
        }

        // Compute FP8 scale: fp8_val ≈ group_max / 6.0 / scale2
        // Use software float→E4M3 conversion (SM121 hardware cast may fail)
        float fp8_float = (group_max > 0.0f) ? (group_max * inv_scale2 / 6.0f) : 0.0f;
        unsigned char fp8_byte = float_to_fp8_e4m3(fp8_float);
        row_scale[g] = fp8_byte;

        // Effective scale for quantization — dequant the FP8 scale back to float
        // using the E4M3 LUT (or recompute from bits)
        unsigned int fp8_sign = (fp8_byte >> 7) & 1;
        unsigned int fp8_exp = (fp8_byte >> 3) & 0xF;
        unsigned int fp8_man = fp8_byte & 0x7;
        float fp8_decoded;
        if (fp8_exp == 0) {
            fp8_decoded = (float)fp8_man * 0.001953125f;  // 2^(-9) per mantissa unit
        } else if (fp8_exp == 15 && fp8_man == 7) {
            fp8_decoded = 0.0f;  // NaN → 0
        } else {
            unsigned int f32_bits = ((fp8_exp + 120u) << 23) | (fp8_man << 20);
            fp8_decoded = __uint_as_float(f32_bits);
        }
        if (fp8_sign) fp8_decoded = -fp8_decoded;

        float effective_scale = fp8_decoded * scale2;
        float inv_eff = (effective_scale > 0.0f) ? (1.0f / effective_scale) : 0.0f;

        // Quantize 16 elements → 8 packed bytes
        #pragma unroll
        for (int i = 0; i < GROUP_SIZE; i += 2) {
            float v0 = __bfloat162float(row_in[base + i]) * inv_eff;
            float v1 = __bfloat162float(row_in[base + i + 1]) * inv_eff;

            unsigned int n0 = quantize_e2m1(v0);
            unsigned int n1 = quantize_e2m1(v1);

            // Pack: low nibble = even position, high nibble = odd position
            row_packed[g * 8 + i / 2] = (unsigned char)((n1 << 4) | (n0 & 0xF));
        }
    }
}
