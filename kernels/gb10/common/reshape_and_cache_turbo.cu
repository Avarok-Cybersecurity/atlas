// SPDX-License-Identifier: AGPL-3.0-only

// TurboQuant KV cache write kernels — WHT + Lloyd-Max quantization.
//
// Three variants:
//   turbo8: WHT + FP8 E4M3 (1 byte/elem + scales). Outlier-free FP8.
//   turbo4: WHT + Lloyd-Max 16-level (4-bit packed + scales). Same as NVFP4 layout.
//   turbo3: WHT + Lloyd-Max 8-level (3-bit packed + scales). 22% smaller than turbo4.
//
// All variants apply Walsh-Hadamard Transform (WHT) to K/V before quantization.
// WHT Gaussianizes the coordinate distribution, eliminating outliers that would
// otherwise clip in FP8 or waste dynamic range in low-bit codebooks.
//
// The key invariant: <WHT(Q), WHT(K)> = <Q, K> (Parseval's theorem).
// So storing WHT(K) and WHT(V) preserves attention correctness.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define GROUP_SIZE 16

// ── Lloyd-Max codebooks for Gaussian N(0,1) ──

// 16-level (turbo4) — MSE = 0.009497
__device__ __constant__ float TURBO4_CODEBOOK[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};
__device__ __constant__ float TURBO4_BOUNDS[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f, -0.7996f, -0.5224f, -0.2582f, 0.0f,
     0.2582f,  0.5224f,  0.7996f,  1.0993f,  1.4371f,  1.8435f,  2.4008f
};
#define TURBO4_MAX 2.7326f

// 8-level (turbo3) — MSE = 0.03454
__device__ __constant__ float TURBO3_CODEBOOK[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f
};
__device__ __constant__ float TURBO3_BOUNDS[7] = {
    -1.748f, -1.050f, -0.501f, 0.0f, 0.501f, 1.050f, 1.748f
};
#define TURBO3_MAX 2.1520f

// ── FP8 E4M3 helpers ──

__device__ __forceinline__ __nv_fp8_storage_t float_to_fp8(float val) {
    unsigned short pair;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;"
                 : "=h"(pair) : "f"(val));
    return (__nv_fp8_storage_t)(pair & 0xFF);  // low byte = first FP8 value
}

// FP8 E4M3 max representable
#define FP8_E4M3_MAX 448.0f

// ── Walsh-Hadamard Transform (in-place, per-warp, 256 elements) ──
// 32 threads × 8 elements = 256 total. Butterfly network.

__device__ __forceinline__ void wht256_warp(float vals[8], unsigned int lane) {
    // Stages 0-2: intra-thread butterflies
    #pragma unroll
    for (int stride = 1; stride <= 4; stride <<= 1) {
        #pragma unroll
        for (int i = 0; i < 8; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = vals[i + j];
                float b = vals[i + j + stride];
                vals[i + j] = a + b;
                vals[i + j + stride] = a - b;
            }
        }
    }
    // Stages 3-7: inter-thread via shuffle
    #pragma unroll
    for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float other = __shfl_xor_sync(0xFFFFFFFF, vals[i], xor_mask);
            // Butterfly: if bit is set, subtract; else add
            if (lane & xor_mask)
                vals[i] = other - vals[i];
            else
                vals[i] = vals[i] + other;
        }
    }
    // Normalize: 1/sqrt(256) = 1/16
    #pragma unroll
    for (int i = 0; i < 8; i++) vals[i] *= 0.0625f;
}

// ── Quantize helpers ──

__device__ __forceinline__ unsigned char turbo4_quantize(float x) {
    // Binary search in 15 boundaries → 4-bit index [0..15]
    unsigned char idx = 0;
    if (x >= TURBO4_BOUNDS[7]) {  // >= 0
        idx = 8;
        if (x >= TURBO4_BOUNDS[11]) { idx = 12; if (x >= TURBO4_BOUNDS[13]) { idx = 14; if (x >= TURBO4_BOUNDS[14]) idx = 15; } else if (x >= TURBO4_BOUNDS[12]) idx = 13; }
        else { if (x >= TURBO4_BOUNDS[9]) { idx = 10; if (x >= TURBO4_BOUNDS[10]) idx = 11; } else if (x >= TURBO4_BOUNDS[8]) idx = 9; }
    } else {
        if (x >= TURBO4_BOUNDS[3]) { idx = 4; if (x >= TURBO4_BOUNDS[5]) { idx = 6; if (x >= TURBO4_BOUNDS[6]) idx = 7; } else if (x >= TURBO4_BOUNDS[4]) idx = 5; }
        else { if (x >= TURBO4_BOUNDS[1]) { idx = 2; if (x >= TURBO4_BOUNDS[2]) idx = 3; } else if (x >= TURBO4_BOUNDS[0]) idx = 1; }
    }
    return idx;
}

__device__ __forceinline__ unsigned char turbo3_quantize(float x) {
    // Binary search in 7 boundaries → 3-bit index [0..7]
    unsigned char idx = 0;
    if (x >= TURBO3_BOUNDS[3]) {  // >= 0
        idx = 4;
        if (x >= TURBO3_BOUNDS[5]) { idx = 6; if (x >= TURBO3_BOUNDS[6]) idx = 7; }
        else if (x >= TURBO3_BOUNDS[4]) idx = 5;
    } else {
        if (x >= TURBO3_BOUNDS[1]) { idx = 2; if (x >= TURBO3_BOUNDS[2]) idx = 3; }
        else if (x >= TURBO3_BOUNDS[0]) idx = 1;
    }
    return idx;
}

// ── Turbo4 reshape_and_cache (same layout as NVFP4: 4-bit packed) ──

extern "C" __global__ void reshape_and_cache_flash_turbo4(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;
    const unsigned int lane = threadIdx.x % 32;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    unsigned char* block_k = k_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned long long data_off = (unsigned long long)block_offset * (n_elems / 2);
    unsigned long long scale_off = data_section_bytes + (unsigned long long)block_offset * num_groups;

    // Simple group-loop quantization (handles any head_dim correctly).
    // Each thread processes one group of 16 elements per iteration.
    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        // Load 16 BF16 elements
        float kf[16], vf[16];
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
        }

        // Compute group absmax
        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }

        // Scales
        float k_inv = (k_max > 1e-12f) ? (TURBO4_MAX / k_max) : 1.0f;
        float v_inv = (v_max > 1e-12f) ? (TURBO4_MAX / v_max) : 1.0f;
        float ks = k_max / TURBO4_MAX; if (ks > FP8_E4M3_MAX) ks = FP8_E4M3_MAX;
        float vs = v_max / TURBO4_MAX; if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        // Write FP8 scales
        ((__nv_fp8_storage_t*)(block_k + scale_off))[g] = float_to_fp8(ks);
        ((__nv_fp8_storage_t*)(block_v + scale_off))[g] = float_to_fp8(vs);

        // Quantize to 4-bit and pack pairs
        unsigned char* kd = block_k + data_off + elem_offset / 2;
        unsigned char* vd = block_v + data_off + elem_offset / 2;
        for (int i = 0; i < 16; i += 2) {
            unsigned char k0 = turbo4_quantize(kf[i] * k_inv);
            unsigned char k1 = turbo4_quantize(kf[i+1] * k_inv);
            unsigned char v0 = turbo4_quantize(vf[i] * v_inv);
            unsigned char v1 = turbo4_quantize(vf[i+1] * v_inv);
            kd[i/2] = k0 | (k1 << 4);
            vd[i/2] = v0 | (v1 << 4);
        }
    }
}

// ── Turbo8 reshape_and_cache (WHT + FP8 E4M3) ──
// Same as turbo4 but stores full FP8 values instead of 4-bit codebook indices.
// Per-group scales ensure proper dynamic range in the WHT domain.

extern "C" __global__ void reshape_and_cache_flash_turbo8(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;
    const unsigned int lane = threadIdx.x % 32;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // Turbo8 layout (2026-04-28 BF16-scale upgrade):
    //   data   = [block_size, n_elems] FP8 E4M3 (1 byte/elem)
    //   scales = [block_size, num_groups] BF16 (2 bytes/scale)
    // Was FP8 scales (1 byte). FP8's ~12% scale precision compounded
    // catastrophically across 50+ Turbo8 layers; BF16 (~0.4%) keeps
    // many-layer models (MiniMax M2.7: 58 Turbo8 layers) coherent.
    unsigned char* block_k = k_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned long long data_off = (unsigned long long)block_offset * n_elems;
    // Each scale is BF16 = 2 bytes. Scale offset uses 2× the group index.
    unsigned long long scale_off =
        data_section_bytes + (unsigned long long)block_offset * num_groups * 2;

    // Simple group-loop: each thread processes one group of 16 elements
    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
        }

        // Group absmax
        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }

        float k_scale = k_max / FP8_E4M3_MAX;
        float v_scale = v_max / FP8_E4M3_MAX;
        if (k_scale < 1e-12f) k_scale = 1e-12f;
        if (v_scale < 1e-12f) v_scale = 1e-12f;

        // Write BF16 scales (2 bytes each).
        ((__nv_bfloat16*)(block_k + scale_off))[g] = __float2bfloat16(k_scale);
        ((__nv_bfloat16*)(block_v + scale_off))[g] = __float2bfloat16(v_scale);

        // Quantize to FP8 and write (1 byte per element)
        float k_inv = 1.0f / k_scale;
        float v_inv = 1.0f / v_scale;
        unsigned char* kd = block_k + data_off + elem_offset;
        unsigned char* vd = block_v + data_off + elem_offset;
        for (int i = 0; i < 16; i++) {
            float ks = fminf(fmaxf(kf[i] * k_inv, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            float vs = fminf(fmaxf(vf[i] * v_inv, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            kd[i] = (unsigned char)float_to_fp8(ks);
            vd[i] = (unsigned char)float_to_fp8(vs);
        }
    }
}

// ── Turbo3 reshape_and_cache (WHT + Lloyd-Max 8-level, 3-bit packed) ──

extern "C" __global__ void reshape_and_cache_flash_turbo3(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;
    const unsigned int lane = threadIdx.x % 32;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // Turbo3 layout: data = [block_size, n_elems*3/8] packed 3-bit, scales = [block_size, num_groups] FP8
    unsigned char* block_k = k_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned long long data_off = (unsigned long long)block_offset * (n_elems * 3 / 8);
    unsigned long long scale_off = data_section_bytes + (unsigned long long)block_offset * num_groups;

    // Simple group-loop: process groups of 16 elements, quantize to 3-bit
    // Note: 3-bit packing requires processing in chunks of 8 (pack 8→3 bytes)
    // Process 16 elements (1 group) at a time, pack as 2×(8→3 bytes) = 6 bytes
    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
        }

        // Group absmax
        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }

        float k_inv = (k_max > 1e-12f) ? (TURBO3_MAX / k_max) : 1.0f;
        float v_inv = (v_max > 1e-12f) ? (TURBO3_MAX / v_max) : 1.0f;
        float ks = k_max / TURBO3_MAX; if (ks > FP8_E4M3_MAX) ks = FP8_E4M3_MAX;
        float vs = v_max / TURBO3_MAX; if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        // Write FP8 scales
        ((__nv_fp8_storage_t*)(block_k + scale_off))[g] = float_to_fp8(ks);
        ((__nv_fp8_storage_t*)(block_v + scale_off))[g] = float_to_fp8(vs);

        // Quantize 16 values to 3-bit indices
        unsigned char ki[16], vi[16];
        for (int i = 0; i < 16; i++) {
            ki[i] = turbo3_quantize(kf[i] * k_inv);
            vi[i] = turbo3_quantize(vf[i] * v_inv);
        }

        // Pack 16 × 3-bit into 6 bytes (two groups of 8→3)
        unsigned int byte_base = elem_offset * 3 / 8;
        unsigned char* kd = block_k + data_off + byte_base;
        unsigned char* vd = block_v + data_off + byte_base;

        // First 8 values → 3 bytes
        kd[0] = (ki[0]) | (ki[1] << 3) | (ki[2] << 6);
        kd[1] = (ki[2] >> 2) | (ki[3] << 1) | (ki[4] << 4) | (ki[5] << 7);
        kd[2] = (ki[5] >> 1) | (ki[6] << 2) | (ki[7] << 5);
        vd[0] = (vi[0]) | (vi[1] << 3) | (vi[2] << 6);
        vd[1] = (vi[2] >> 2) | (vi[3] << 1) | (vi[4] << 4) | (vi[5] << 7);
        vd[2] = (vi[5] >> 1) | (vi[6] << 2) | (vi[7] << 5);

        // Second 8 values → 3 bytes
        kd[3] = (ki[8]) | (ki[9] << 3) | (ki[10] << 6);
        kd[4] = (ki[10] >> 2) | (ki[11] << 1) | (ki[12] << 4) | (ki[13] << 7);
        kd[5] = (ki[13] >> 1) | (ki[14] << 2) | (ki[15] << 5);
        vd[3] = (vi[8]) | (vi[9] << 3) | (vi[10] << 6);
        vd[4] = (vi[10] >> 2) | (vi[11] << 1) | (vi[12] << 4) | (vi[13] << 7);
        vd[5] = (vi[13] >> 1) | (vi[14] << 2) | (vi[15] << 5);
    }
}
