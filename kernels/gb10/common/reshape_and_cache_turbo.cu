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

// 4-level (turbo2) — 2-bit Lloyd-Max for N(0,1). 6.4x compression vs bf16.
// MSE ≈ 0.117 — quality cost is real but the canonical two-sided WHT
// rotation makes outlier mass tractable, mirroring the turbo3 win.
__device__ __constant__ float TURBO2_CODEBOOK[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f
};
__device__ __constant__ float TURBO2_BOUNDS[3] = {
    -0.9816f, 0.0f, 0.9816f
};
#define TURBO2_MAX 1.5104f

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

__device__ __forceinline__ unsigned char turbo2_quantize(float x) {
    // Binary search in 3 boundaries → 2-bit index [0..3]
    if (x >= TURBO2_BOUNDS[1]) return (x >= TURBO2_BOUNDS[2]) ? 3 : 2;
    else                       return (x >= TURBO2_BOUNDS[0]) ? 1 : 0;
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

    // Per-group quantization with L2 norm correction (matched-norm scale).
    // After indexing each element into the codebook, replace the raw amax
    // scale with `||original|| / ||centroid_vec||` so that the dequantized
    // group has the same L2 norm as the input — compensating for systematic
    // shrinkage from rounding-to-centroid. Free quality win (~0.5% PPL on
    // turbo3, similar magnitude on turbo4); only adds 16 FMAs per group on
    // the (cold) write path.
    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        float k_norm_sq = 0.0f, v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            k_norm_sq += kf[i] * kf[i];
            v_norm_sq += vf[i] * vf[i];
        }

        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }

        float k_inv = (k_max > 1e-12f) ? (TURBO4_MAX / k_max) : 1.0f;
        float v_inv = (v_max > 1e-12f) ? (TURBO4_MAX / v_max) : 1.0f;

        // Quantize + accumulate centroid recon L2 norm
        unsigned char k_idx[16], v_idx[16];
        float k_recon_sq = 0.0f, v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_idx[i] = turbo4_quantize(kf[i] * k_inv);
            v_idx[i] = turbo4_quantize(vf[i] * v_inv);
            float kc = TURBO4_CODEBOOK[k_idx[i]];
            float vc = TURBO4_CODEBOOK[v_idx[i]];
            k_recon_sq += kc * kc;
            v_recon_sq += vc * vc;
        }
        float k_recon_norm = sqrtf(k_recon_sq);
        float v_recon_norm = sqrtf(v_recon_sq);

        // Matched-norm scale: dequant(group) has L2 norm = original L2 norm.
        // Fall back to amax scale on degenerate (all-zero) groups.
        float ks = (k_recon_norm > 1e-10f) ? (sqrtf(k_norm_sq) / k_recon_norm) : (k_max / TURBO4_MAX);
        float vs = (v_recon_norm > 1e-10f) ? (sqrtf(v_norm_sq) / v_recon_norm) : (v_max / TURBO4_MAX);
        if (ks > FP8_E4M3_MAX) ks = FP8_E4M3_MAX;
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_k + scale_off))[g] = float_to_fp8(ks);
        ((__nv_fp8_storage_t*)(block_v + scale_off))[g] = float_to_fp8(vs);

        unsigned char* kd = block_k + data_off + elem_offset / 2;
        unsigned char* vd = block_v + data_off + elem_offset / 2;
        for (int i = 0; i < 16; i += 2) {
            kd[i/2] = k_idx[i] | (k_idx[i+1] << 4);
            vd[i/2] = v_idx[i] | (v_idx[i+1] << 4);
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

    // Per-group quantization with L2 norm correction (matched-norm scale).
    // Replaces the amax scale with `||original|| / ||centroid_vec||` so the
    // dequantized group has the same L2 norm as the input — compensates for
    // centroid rounding shrinkage. 16 extra FMAs per group on the cold write
    // path.
    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        float k_norm_sq = 0.0f, v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            k_norm_sq += kf[i] * kf[i];
            v_norm_sq += vf[i] * vf[i];
        }

        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }

        float k_inv = (k_max > 1e-12f) ? (TURBO3_MAX / k_max) : 1.0f;
        float v_inv = (v_max > 1e-12f) ? (TURBO3_MAX / v_max) : 1.0f;

        unsigned char ki[16], vi[16];
        float k_recon_sq = 0.0f, v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            ki[i] = turbo3_quantize(kf[i] * k_inv);
            vi[i] = turbo3_quantize(vf[i] * v_inv);
            float kc = TURBO3_CODEBOOK[ki[i]];
            float vc = TURBO3_CODEBOOK[vi[i]];
            k_recon_sq += kc * kc;
            v_recon_sq += vc * vc;
        }
        float k_recon_norm = sqrtf(k_recon_sq);
        float v_recon_norm = sqrtf(v_recon_sq);

        float ks = (k_recon_norm > 1e-10f) ? (sqrtf(k_norm_sq) / k_recon_norm) : (k_max / TURBO3_MAX);
        float vs = (v_recon_norm > 1e-10f) ? (sqrtf(v_norm_sq) / v_recon_norm) : (v_max / TURBO3_MAX);
        if (ks > FP8_E4M3_MAX) ks = FP8_E4M3_MAX;
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_k + scale_off))[g] = float_to_fp8(ks);
        ((__nv_fp8_storage_t*)(block_v + scale_off))[g] = float_to_fp8(vs);

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

// ── Turbo2 reshape_and_cache (WHT + Lloyd-Max 4-level, 2-bit packed) ──
//
// 6.4x compression vs bf16. Block layout: 4 bytes data + 1 FP8 scale byte
// per 16-element group = 5 bytes per 16 elems = 2.5 bits/elem (data) +
// 0.5 bits/elem (scale) = 3.0 bits/elem total. Pack: 4 indices per byte.
// Adapted to Atlas's GROUP_SIZE=16 layout to stay NVFP4-compatible (the
// alternative 32-elem grouping would buy q4_0-style parity at the cost of
// breaking the per-group scale section's NVFP4 alignment).

extern "C" __global__ void reshape_and_cache_flash_turbo2(
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

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // Turbo2 layout: data = [block_size, n_elems*2/8] = [block_size, n_elems/4] packed 2-bit.
    // Scales = [block_size, num_groups] FP8 (same as nvfp4/turbo3/turbo4).
    unsigned char* block_k = k_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned long long data_off = (unsigned long long)block_offset * (n_elems / 4);
    unsigned long long scale_off = data_section_bytes + (unsigned long long)block_offset * num_groups;

    // Per-group with L2 norm correction (matched-norm scale, same trick as turbo3/4).
    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        float k_norm_sq = 0.0f, v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            k_norm_sq += kf[i] * kf[i];
            v_norm_sq += vf[i] * vf[i];
        }

        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }

        float k_inv = (k_max > 1e-12f) ? (TURBO2_MAX / k_max) : 1.0f;
        float v_inv = (v_max > 1e-12f) ? (TURBO2_MAX / v_max) : 1.0f;

        unsigned char ki[16], vi[16];
        float k_recon_sq = 0.0f, v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            ki[i] = turbo2_quantize(kf[i] * k_inv);
            vi[i] = turbo2_quantize(vf[i] * v_inv);
            float kc = TURBO2_CODEBOOK[ki[i]];
            float vc = TURBO2_CODEBOOK[vi[i]];
            k_recon_sq += kc * kc;
            v_recon_sq += vc * vc;
        }
        float k_recon_norm = sqrtf(k_recon_sq);
        float v_recon_norm = sqrtf(v_recon_sq);

        float ks = (k_recon_norm > 1e-10f) ? (sqrtf(k_norm_sq) / k_recon_norm) : (k_max / TURBO2_MAX);
        float vs = (v_recon_norm > 1e-10f) ? (sqrtf(v_norm_sq) / v_recon_norm) : (v_max / TURBO2_MAX);
        if (ks > FP8_E4M3_MAX) ks = FP8_E4M3_MAX;
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_k + scale_off))[g] = float_to_fp8(ks);
        ((__nv_fp8_storage_t*)(block_v + scale_off))[g] = float_to_fp8(vs);

        // Pack 16 × 2-bit into 4 bytes (4 indices per byte).
        unsigned int byte_base = elem_offset / 4;
        unsigned char* kd = block_k + data_off + byte_base;
        unsigned char* vd = block_v + data_off + byte_base;
        kd[0] = ki[0]  | (ki[1]  << 2) | (ki[2]  << 4) | (ki[3]  << 6);
        kd[1] = ki[4]  | (ki[5]  << 2) | (ki[6]  << 4) | (ki[7]  << 6);
        kd[2] = ki[8]  | (ki[9]  << 2) | (ki[10] << 4) | (ki[11] << 6);
        kd[3] = ki[12] | (ki[13] << 2) | (ki[14] << 4) | (ki[15] << 6);
        vd[0] = vi[0]  | (vi[1]  << 2) | (vi[2]  << 4) | (vi[3]  << 6);
        vd[1] = vi[4]  | (vi[5]  << 2) | (vi[6]  << 4) | (vi[7]  << 6);
        vd[2] = vi[8]  | (vi[9]  << 2) | (vi[10] << 4) | (vi[11] << 6);
        vd[3] = vi[12] | (vi[13] << 2) | (vi[14] << 4) | (vi[15] << 6);
    }
}

// ── Asymmetric Bf16K + Turbo3V reshape_and_cache (TurboQuant+ safer-asym) ──
//
// K is written as raw BF16 (contiguous NHD layout, identical to the
// `reshape_and_cache_flash` baseline). V is written as turbo3 (3-bit Lloyd-Max
// + FP8 per-group scale + matched-norm L2 correction). The two sides use
// separate strides because the K pool is sized for bf16 (2 b/elem) and the
// V pool is sized for turbo3 (~0.5 b/elem + scale).
//
// Caller passes:
//   k_block_stride_bytes: bytes per K block in the K pool (= 2 * block_size *
//                        num_kv_heads * head_dim).
//   v_block_stride_bytes: bytes per V block in the V pool (= turbo3 size).
//   v_data_section_bytes: V-pool data section size (= block_size * n_elems * 3/8).
//
// Grid: (num_tokens, 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void reshape_and_cache_flash_bf16k_turbo3v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ k_cache,        // bf16-typed K pool
    unsigned char* __restrict__ v_cache,         // turbo3 byte-addressed V pool
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long k_block_stride_bytes,   // K-pool: bf16 bytes/block
    const unsigned long long v_block_stride_bytes,   // V-pool: turbo3 bytes/block
    const unsigned long long v_data_section_bytes    // V-pool: 3-bit data section
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // ── K-side write: raw BF16 copy (same as reshape_and_cache_flash) ──
    // K pool is bf16-typed; stride in elements is k_block_stride_bytes/2.
    // For sliding-layer/MQA correctness we must compute the per-block element
    // stride directly from the n_elems geometry to match the kernel reader's
    // assumption that contiguous tokens within a block lie head-major NHD.
    {
        const unsigned long long k_block_stride_elems = (unsigned long long)block_size * n_elems;
        __nv_bfloat16* key_dst = k_cache
            + (unsigned long long)block_idx * k_block_stride_elems
            + (unsigned long long)block_offset * n_elems;
        // Vectorized 8-byte copy: uint2 = 4 BF16 elems per thread step.
        const unsigned int n_vec = n_elems / 4;
        const unsigned int n_rem = n_elems % 4;
        const uint2* key_src_vec = (const uint2*)key_src;
        uint2* key_dst_vec = (uint2*)key_dst;
        for (unsigned int i = threadIdx.x; i < n_vec; i += blockDim.x) {
            key_dst_vec[i] = key_src_vec[i];
        }
        if (n_rem > 0) {
            unsigned int base = n_vec * 4;
            for (unsigned int i = threadIdx.x; i < n_rem; i += blockDim.x) {
                key_dst[base + i] = key_src[base + i];
            }
        }
    }

    // ── V-side write: turbo3 (3-bit packed + FP8 group scale + matched-norm) ──
    // Identical group-by-group quantization to reshape_and_cache_flash_turbo3,
    // but applied only to V (K already written above). Block addresses use the
    // V-pool's stride which differs from the K-pool stride for this asym dtype.
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * (n_elems * 3 / 8);
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float vf[16];
        float v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            v_norm_sq += vf[i] * vf[i];
        }
        float v_max = 0.0f;
        for (int i = 0; i < 16; i++) v_max = fmaxf(v_max, fabsf(vf[i]));

        float v_inv = (v_max > 1e-12f) ? (TURBO3_MAX / v_max) : 1.0f;

        unsigned char vi[16];
        float v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vi[i] = turbo3_quantize(vf[i] * v_inv);
            float vc = TURBO3_CODEBOOK[vi[i]];
            v_recon_sq += vc * vc;
        }
        float v_recon_norm = sqrtf(v_recon_sq);
        float vs = (v_recon_norm > 1e-10f)
            ? (sqrtf(v_norm_sq) / v_recon_norm)
            : (v_max / TURBO3_MAX);
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_v + v_scale_off))[g] = float_to_fp8(vs);

        // Pack 16 × 3-bit into 6 bytes (two groups of 8 → 3 bytes each).
        unsigned int byte_base = elem_offset * 3 / 8;
        unsigned char* vd = block_v + v_data_off + byte_base;
        // First 8 values
        vd[0] = (vi[0]) | (vi[1] << 3) | (vi[2] << 6);
        vd[1] = (vi[2] >> 2) | (vi[3] << 1) | (vi[4] << 4) | (vi[5] << 7);
        vd[2] = (vi[5] >> 1) | (vi[6] << 2) | (vi[7] << 5);
        // Second 8 values
        vd[3] = (vi[8]) | (vi[9] << 3) | (vi[10] << 6);
        vd[4] = (vi[10] >> 2) | (vi[11] << 1) | (vi[12] << 4) | (vi[13] << 7);
        vd[5] = (vi[13] >> 1) | (vi[14] << 2) | (vi[15] << 5);
    }
}


// ── Bf16K + Turbo4V reshape_and_cache (asymmetric: K=bf16, V=4-bit packed) ──
//
// K: raw BF16 copy into bf16-typed K pool (same as reshape_and_cache_flash).
// V: turbo4 4-bit Lloyd-Max + per-group FP8 scale with matched-norm L2 correction
//    (identical math to reshape_and_cache_flash_turbo4's V side).
//
// Grid: (num_tokens, 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void reshape_and_cache_flash_bf16k_turbo4v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ k_cache,        // bf16-typed K pool
    unsigned char* __restrict__ v_cache,         // turbo4 byte-addressed V pool
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long k_block_stride_bytes,   // K-pool: bf16 bytes/block (unused; geometry-computed)
    const unsigned long long v_block_stride_bytes,   // V-pool: turbo4 bytes/block
    const unsigned long long v_data_section_bytes    // V-pool: 4-bit data section
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // ── K-side write: raw BF16 copy (same as reshape_and_cache_flash) ──
    {
        const unsigned long long k_block_stride_elems = (unsigned long long)block_size * n_elems;
        __nv_bfloat16* key_dst = k_cache
            + (unsigned long long)block_idx * k_block_stride_elems
            + (unsigned long long)block_offset * n_elems;
        const unsigned int n_vec = n_elems / 4;
        const unsigned int n_rem = n_elems % 4;
        const uint2* key_src_vec = (const uint2*)key_src;
        uint2* key_dst_vec = (uint2*)key_dst;
        for (unsigned int i = threadIdx.x; i < n_vec; i += blockDim.x) {
            key_dst_vec[i] = key_src_vec[i];
        }
        if (n_rem > 0) {
            unsigned int base = n_vec * 4;
            for (unsigned int i = threadIdx.x; i < n_rem; i += blockDim.x) {
                key_dst[base + i] = key_src[base + i];
            }
        }
    }

    // ── V-side write: turbo4 (4-bit packed + FP8 group scale + matched-norm) ──
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * (n_elems / 2);
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float vf[16];
        float v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            v_norm_sq += vf[i] * vf[i];
        }
        float v_max = 0.0f;
        for (int i = 0; i < 16; i++) v_max = fmaxf(v_max, fabsf(vf[i]));

        float v_inv = (v_max > 1e-12f) ? (TURBO4_MAX / v_max) : 1.0f;

        unsigned char vi[16];
        float v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vi[i] = turbo4_quantize(vf[i] * v_inv);
            float vc = TURBO4_CODEBOOK[vi[i]];
            v_recon_sq += vc * vc;
        }
        float v_recon_norm = sqrtf(v_recon_sq);
        float vs = (v_recon_norm > 1e-10f)
            ? (sqrtf(v_norm_sq) / v_recon_norm)
            : (v_max / TURBO4_MAX);
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_v + v_scale_off))[g] = float_to_fp8(vs);

        // Pack 16 × 4-bit into 8 bytes (2 indices per byte, low nibble first).
        unsigned char* vd = block_v + v_data_off + elem_offset / 2;
        for (int i = 0; i < 16; i += 2) {
            vd[i/2] = vi[i] | (vi[i+1] << 4);
        }
    }
}

// ── Bf16K + Turbo2V reshape_and_cache (asymmetric: K=bf16, V=2-bit packed) ──
//
// K: raw BF16 copy into bf16-typed K pool (same as reshape_and_cache_flash).
// V: turbo2 2-bit Lloyd-Max + per-group FP8 scale with matched-norm L2 correction
//    (identical math to reshape_and_cache_flash_turbo2's V side).
//
// Grid: (num_tokens, 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void reshape_and_cache_flash_bf16k_turbo2v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ k_cache,        // bf16-typed K pool
    unsigned char* __restrict__ v_cache,         // turbo2 byte-addressed V pool
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long k_block_stride_bytes,   // K-pool: bf16 bytes/block (unused; geometry-computed)
    const unsigned long long v_block_stride_bytes,   // V-pool: turbo2 bytes/block
    const unsigned long long v_data_section_bytes    // V-pool: 2-bit data section
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // ── K-side write: raw BF16 copy ──
    {
        const unsigned long long k_block_stride_elems = (unsigned long long)block_size * n_elems;
        __nv_bfloat16* key_dst = k_cache
            + (unsigned long long)block_idx * k_block_stride_elems
            + (unsigned long long)block_offset * n_elems;
        const unsigned int n_vec = n_elems / 4;
        const unsigned int n_rem = n_elems % 4;
        const uint2* key_src_vec = (const uint2*)key_src;
        uint2* key_dst_vec = (uint2*)key_dst;
        for (unsigned int i = threadIdx.x; i < n_vec; i += blockDim.x) {
            key_dst_vec[i] = key_src_vec[i];
        }
        if (n_rem > 0) {
            unsigned int base = n_vec * 4;
            for (unsigned int i = threadIdx.x; i < n_rem; i += blockDim.x) {
                key_dst[base + i] = key_src[base + i];
            }
        }
    }

    // ── V-side write: turbo2 (2-bit packed + FP8 group scale + matched-norm) ──
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * (n_elems / 4);
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float vf[16];
        float v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            v_norm_sq += vf[i] * vf[i];
        }
        float v_max = 0.0f;
        for (int i = 0; i < 16; i++) v_max = fmaxf(v_max, fabsf(vf[i]));

        float v_inv = (v_max > 1e-12f) ? (TURBO2_MAX / v_max) : 1.0f;

        unsigned char vi[16];
        float v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vi[i] = turbo2_quantize(vf[i] * v_inv);
            float vc = TURBO2_CODEBOOK[vi[i]];
            v_recon_sq += vc * vc;
        }
        float v_recon_norm = sqrtf(v_recon_sq);
        float vs = (v_recon_norm > 1e-10f)
            ? (sqrtf(v_norm_sq) / v_recon_norm)
            : (v_max / TURBO2_MAX);
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_v + v_scale_off))[g] = float_to_fp8(vs);

        // Pack 16 × 2-bit into 4 bytes (4 indices per byte).
        unsigned char* vd = block_v + v_data_off + elem_offset / 4;
        vd[0] = vi[0]  | (vi[1]  << 2) | (vi[2]  << 4) | (vi[3]  << 6);
        vd[1] = vi[4]  | (vi[5]  << 2) | (vi[6]  << 4) | (vi[7]  << 6);
        vd[2] = vi[8]  | (vi[9]  << 2) | (vi[10] << 4) | (vi[11] << 6);
        vd[3] = vi[12] | (vi[13] << 2) | (vi[14] << 4) | (vi[15] << 6);
    }
}

// ── Asymmetric Fp8K + Turbo3V reshape_and_cache (TurboQuant+ asym) ──
//
// K is written as FP8 E4M3 (contiguous NHD layout, identical byte layout to
// the FP8 baseline — caller passes `k_scale` so quant = bf16 / k_scale → fp8).
// V is written as turbo3 (3-bit Lloyd-Max + per-group FP8 scale + matched-norm
// L2 correction), identical math to reshape_and_cache_flash_bf16k_turbo3v's
// V side. K and V pools have separate strides because they're sized for
// different dtypes (FP8 = 1 b/elem K; turbo3 = ~0.5 b/elem V + scales).
//
// Grid: (num_tokens, 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void reshape_and_cache_flash_fp8k_turbo3v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,        // FP8-typed K pool (byte-addressed)
    unsigned char* __restrict__ v_cache,         // turbo3 byte-addressed V pool
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const float k_scale,                                // FP8 K per-tensor quant scale
    const unsigned long long k_block_stride_bytes,      // K-pool: fp8 bytes/block (= block_size*nkv*hd)
    const unsigned long long v_block_stride_bytes,      // V-pool: turbo3 bytes/block
    const unsigned long long v_data_section_bytes       // V-pool: 3-bit data section
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // ── K-side write: BF16 → FP8 E4M3 (per-tensor scale) ──
    // K pool element stride = block_size * n_elems (bytes), since 1 b/fp8 elem.
    {
        const unsigned long long k_block_stride_elems = (unsigned long long)block_size * n_elems;
        unsigned char* key_dst = k_cache
            + (unsigned long long)block_idx * k_block_stride_elems
            + (unsigned long long)block_offset * n_elems;
        const float inv_k_scale = 1.0f / k_scale;
        // Vectorized BF16x2 → FP8x2 path. Pack 2 BF16 from a uint32, quant
        // both, store as packed uint16. Same as reshape_and_cache_flash_fp8.
        const unsigned int n_pairs = n_elems / 2;
        const unsigned int n_rem = n_elems % 2;
        const unsigned int* key_src32 = (const unsigned int*)key_src;
        __nv_fp8x2_storage_t* key_dst16 = (__nv_fp8x2_storage_t*)key_dst;
        for (unsigned int i = threadIdx.x; i < n_pairs; i += blockDim.x) {
            unsigned int pk = key_src32[i];
            float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(pk & 0xFFFF)));
            float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(pk >> 16)));
            float2 scaled = make_float2(v0 * inv_k_scale, v1 * inv_k_scale);
            key_dst16[i] = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
        }
        if (n_rem > 0 && threadIdx.x == 0) {
            unsigned int base = n_pairs * 2;
            float kf = __bfloat162float(key_src[base]) * inv_k_scale;
            ((__nv_fp8_storage_t*)key_dst)[base] = __nv_cvt_float_to_fp8(kf, __NV_SATFINITE, __NV_E4M3);
        }
    }

    // ── V-side write: turbo3 (3-bit packed + FP8 group scale + matched-norm) ──
    // Identical to reshape_and_cache_flash_bf16k_turbo3v's V side.
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * (n_elems * 3 / 8);
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float vf[16];
        float v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            v_norm_sq += vf[i] * vf[i];
        }
        float v_max = 0.0f;
        for (int i = 0; i < 16; i++) v_max = fmaxf(v_max, fabsf(vf[i]));

        float v_inv = (v_max > 1e-12f) ? (TURBO3_MAX / v_max) : 1.0f;

        unsigned char vi[16];
        float v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vi[i] = turbo3_quantize(vf[i] * v_inv);
            float vc = TURBO3_CODEBOOK[vi[i]];
            v_recon_sq += vc * vc;
        }
        float v_recon_norm = sqrtf(v_recon_sq);
        float vs = (v_recon_norm > 1e-10f)
            ? (sqrtf(v_norm_sq) / v_recon_norm)
            : (v_max / TURBO3_MAX);
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_v + v_scale_off))[g] = float_to_fp8(vs);

        unsigned int byte_base = elem_offset * 3 / 8;
        unsigned char* vd = block_v + v_data_off + byte_base;
        vd[0] = (vi[0]) | (vi[1] << 3) | (vi[2] << 6);
        vd[1] = (vi[2] >> 2) | (vi[3] << 1) | (vi[4] << 4) | (vi[5] << 7);
        vd[2] = (vi[5] >> 1) | (vi[6] << 2) | (vi[7] << 5);
        vd[3] = (vi[8]) | (vi[9] << 3) | (vi[10] << 6);
        vd[4] = (vi[10] >> 2) | (vi[11] << 1) | (vi[12] << 4) | (vi[13] << 7);
        vd[5] = (vi[13] >> 1) | (vi[14] << 2) | (vi[15] << 5);
    }
}


// ── Fp8K + Turbo4V reshape_and_cache (asymmetric: K=fp8, V=4-bit packed) ──
//
// K: BF16 → FP8 E4M3 (per-tensor `k_scale`), packed contiguous into the K pool.
// V: turbo4 4-bit Lloyd-Max + per-group FP8 scale with matched-norm L2 correction.
//
// Grid: (num_tokens, 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void reshape_and_cache_flash_fp8k_turbo4v(
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
    const float k_scale,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // ── K-side write: BF16 → FP8 E4M3 ──
    {
        const unsigned long long k_block_stride_elems = (unsigned long long)block_size * n_elems;
        unsigned char* key_dst = k_cache
            + (unsigned long long)block_idx * k_block_stride_elems
            + (unsigned long long)block_offset * n_elems;
        const float inv_k_scale = 1.0f / k_scale;
        const unsigned int n_pairs = n_elems / 2;
        const unsigned int n_rem = n_elems % 2;
        const unsigned int* key_src32 = (const unsigned int*)key_src;
        __nv_fp8x2_storage_t* key_dst16 = (__nv_fp8x2_storage_t*)key_dst;
        for (unsigned int i = threadIdx.x; i < n_pairs; i += blockDim.x) {
            unsigned int pk = key_src32[i];
            float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(pk & 0xFFFF)));
            float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(pk >> 16)));
            float2 scaled = make_float2(v0 * inv_k_scale, v1 * inv_k_scale);
            key_dst16[i] = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
        }
        if (n_rem > 0 && threadIdx.x == 0) {
            unsigned int base = n_pairs * 2;
            float kf = __bfloat162float(key_src[base]) * inv_k_scale;
            ((__nv_fp8_storage_t*)key_dst)[base] = __nv_cvt_float_to_fp8(kf, __NV_SATFINITE, __NV_E4M3);
        }
    }

    // ── V-side write: turbo4 (4-bit packed + FP8 group scale + matched-norm) ──
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * (n_elems / 2);
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float vf[16];
        float v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            v_norm_sq += vf[i] * vf[i];
        }
        float v_max = 0.0f;
        for (int i = 0; i < 16; i++) v_max = fmaxf(v_max, fabsf(vf[i]));

        float v_inv = (v_max > 1e-12f) ? (TURBO4_MAX / v_max) : 1.0f;

        unsigned char vi[16];
        float v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vi[i] = turbo4_quantize(vf[i] * v_inv);
            float vc = TURBO4_CODEBOOK[vi[i]];
            v_recon_sq += vc * vc;
        }
        float v_recon_norm = sqrtf(v_recon_sq);
        float vs = (v_recon_norm > 1e-10f)
            ? (sqrtf(v_norm_sq) / v_recon_norm)
            : (v_max / TURBO4_MAX);
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_v + v_scale_off))[g] = float_to_fp8(vs);

        unsigned char* vd = block_v + v_data_off + elem_offset / 2;
        for (int i = 0; i < 16; i += 2) {
            vd[i/2] = vi[i] | (vi[i+1] << 4);
        }
    }
}

// ── Fp8K + Turbo2V reshape_and_cache (asymmetric: K=fp8, V=2-bit packed) ──
//
// K: BF16 → FP8 E4M3 (per-tensor `k_scale`), packed contiguous into the K pool.
// V: turbo2 2-bit Lloyd-Max + per-group FP8 scale with matched-norm L2 correction.
//
// Grid: (num_tokens, 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void reshape_and_cache_flash_fp8k_turbo2v(
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
    const float k_scale,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // ── K-side write: BF16 → FP8 E4M3 ──
    {
        const unsigned long long k_block_stride_elems = (unsigned long long)block_size * n_elems;
        unsigned char* key_dst = k_cache
            + (unsigned long long)block_idx * k_block_stride_elems
            + (unsigned long long)block_offset * n_elems;
        const float inv_k_scale = 1.0f / k_scale;
        const unsigned int n_pairs = n_elems / 2;
        const unsigned int n_rem = n_elems % 2;
        const unsigned int* key_src32 = (const unsigned int*)key_src;
        __nv_fp8x2_storage_t* key_dst16 = (__nv_fp8x2_storage_t*)key_dst;
        for (unsigned int i = threadIdx.x; i < n_pairs; i += blockDim.x) {
            unsigned int pk = key_src32[i];
            float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(pk & 0xFFFF)));
            float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(pk >> 16)));
            float2 scaled = make_float2(v0 * inv_k_scale, v1 * inv_k_scale);
            key_dst16[i] = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
        }
        if (n_rem > 0 && threadIdx.x == 0) {
            unsigned int base = n_pairs * 2;
            float kf = __bfloat162float(key_src[base]) * inv_k_scale;
            ((__nv_fp8_storage_t*)key_dst)[base] = __nv_cvt_float_to_fp8(kf, __NV_SATFINITE, __NV_E4M3);
        }
    }

    // ── V-side write: turbo2 (2-bit packed + FP8 group scale + matched-norm) ──
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * (n_elems / 4);
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float vf[16];
        float v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            v_norm_sq += vf[i] * vf[i];
        }
        float v_max = 0.0f;
        for (int i = 0; i < 16; i++) v_max = fmaxf(v_max, fabsf(vf[i]));

        float v_inv = (v_max > 1e-12f) ? (TURBO2_MAX / v_max) : 1.0f;

        unsigned char vi[16];
        float v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            vi[i] = turbo2_quantize(vf[i] * v_inv);
            float vc = TURBO2_CODEBOOK[vi[i]];
            v_recon_sq += vc * vc;
        }
        float v_recon_norm = sqrtf(v_recon_sq);
        float vs = (v_recon_norm > 1e-10f)
            ? (sqrtf(v_norm_sq) / v_recon_norm)
            : (v_max / TURBO2_MAX);
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_v + v_scale_off))[g] = float_to_fp8(vs);

        unsigned char* vd = block_v + v_data_off + elem_offset / 4;
        vd[0] = vi[0]  | (vi[1]  << 2) | (vi[2]  << 4) | (vi[3]  << 6);
        vd[1] = vi[4]  | (vi[5]  << 2) | (vi[6]  << 4) | (vi[7]  << 6);
        vd[2] = vi[8]  | (vi[9]  << 2) | (vi[10] << 4) | (vi[11] << 6);
        vd[3] = vi[12] | (vi[13] << 2) | (vi[14] << 4) | (vi[15] << 6);
    }
}

// ============================================================================
// TurboQuant+ both-sides-quantized asymmetric write kernels.
// Each variant quantizes K and V independently into separate-stride pools.
// K and V codebook + packing live in their own routines; matched-norm L2
// scale correction applied to both sides (free quality win, ~0.5% PPL).
// ============================================================================

// ── Turbo4K + Turbo3V reshape_and_cache (asymmetric) ──
//
// K: turbo4 4-bit Lloyd-Max + per-group FP8 scale with matched-norm L2.
// V: turbo3 3-bit Lloyd-Max + per-group FP8 scale with matched-norm L2.
// Each pool has its own block stride and data-section size.
//
// Grid: (num_tokens, 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void reshape_and_cache_flash_turbo4k_turbo3v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,        // turbo4 byte pool
    unsigned char* __restrict__ v_cache,        // turbo3 byte pool
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long k_data_section_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // K-side: turbo4 (4-bit packed, 2 idx per byte) + FP8 group scale.
    unsigned char* block_k = k_cache + (unsigned long long)block_idx * k_block_stride_bytes;
    unsigned long long k_data_off = (unsigned long long)block_offset * (n_elems / 2);
    unsigned long long k_scale_off = k_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    // V-side: turbo3 (3-bit packed) + FP8 group scale.
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * (n_elems * 3 / 8);
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        float k_norm_sq = 0.0f, v_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            k_norm_sq += kf[i] * kf[i];
            v_norm_sq += vf[i] * vf[i];
        }
        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }

        float k_inv = (k_max > 1e-12f) ? (TURBO4_MAX / k_max) : 1.0f;
        float v_inv = (v_max > 1e-12f) ? (TURBO3_MAX / v_max) : 1.0f;

        unsigned char ki[16], vi[16];
        float k_recon_sq = 0.0f, v_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            ki[i] = turbo4_quantize(kf[i] * k_inv);
            vi[i] = turbo3_quantize(vf[i] * v_inv);
            float kc = TURBO4_CODEBOOK[ki[i]];
            float vc = TURBO3_CODEBOOK[vi[i]];
            k_recon_sq += kc * kc;
            v_recon_sq += vc * vc;
        }
        float k_recon_norm = sqrtf(k_recon_sq);
        float v_recon_norm = sqrtf(v_recon_sq);

        float ks = (k_recon_norm > 1e-10f) ? (sqrtf(k_norm_sq) / k_recon_norm) : (k_max / TURBO4_MAX);
        float vs = (v_recon_norm > 1e-10f) ? (sqrtf(v_norm_sq) / v_recon_norm) : (v_max / TURBO3_MAX);
        if (ks > FP8_E4M3_MAX) ks = FP8_E4M3_MAX;
        if (vs > FP8_E4M3_MAX) vs = FP8_E4M3_MAX;

        ((__nv_fp8_storage_t*)(block_k + k_scale_off))[g] = float_to_fp8(ks);
        ((__nv_fp8_storage_t*)(block_v + v_scale_off))[g] = float_to_fp8(vs);

        // Pack K: 16 × 4-bit into 8 bytes (low nibble first).
        unsigned char* kd = block_k + k_data_off + elem_offset / 2;
        for (int i = 0; i < 16; i += 2) {
            kd[i/2] = ki[i] | (ki[i+1] << 4);
        }

        // Pack V: 16 × 3-bit into 6 bytes (two halves of 8 → 3 bytes each).
        unsigned int v_byte_base = elem_offset * 3 / 8;
        unsigned char* vd = block_v + v_data_off + v_byte_base;
        vd[0] = (vi[0]) | (vi[1] << 3) | (vi[2] << 6);
        vd[1] = (vi[2] >> 2) | (vi[3] << 1) | (vi[4] << 4) | (vi[5] << 7);
        vd[2] = (vi[5] >> 1) | (vi[6] << 2) | (vi[7] << 5);
        vd[3] = (vi[8]) | (vi[9] << 3) | (vi[10] << 6);
        vd[4] = (vi[10] >> 2) | (vi[11] << 1) | (vi[12] << 4) | (vi[13] << 7);
        vd[5] = (vi[13] >> 1) | (vi[14] << 2) | (vi[15] << 5);
    }
}

// ── Turbo4K + Turbo8V reshape_and_cache (asymmetric) ──
//
// K: turbo4 4-bit Lloyd-Max + FP8 group scale (matched-norm L2).
// V: turbo8 FP8 E4M3 + BF16 group scale (amax scaling, 2 b/scale upgrade).
extern "C" __global__ void reshape_and_cache_flash_turbo4k_turbo8v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,        // turbo4 byte pool
    unsigned char* __restrict__ v_cache,        // turbo8 byte pool
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long k_data_section_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // K-side: turbo4 (4-bit packed) + FP8 group scale.
    unsigned char* block_k = k_cache + (unsigned long long)block_idx * k_block_stride_bytes;
    unsigned long long k_data_off = (unsigned long long)block_offset * (n_elems / 2);
    unsigned long long k_scale_off = k_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    // V-side: turbo8 (1 byte/elem FP8) + BF16 group scale (2 bytes per group).
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * n_elems;
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups * 2;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        float k_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            k_norm_sq += kf[i] * kf[i];
        }
        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }

        // K-side: turbo4 quantize with matched-norm L2.
        float k_inv = (k_max > 1e-12f) ? (TURBO4_MAX / k_max) : 1.0f;
        unsigned char ki[16];
        float k_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            ki[i] = turbo4_quantize(kf[i] * k_inv);
            float kc = TURBO4_CODEBOOK[ki[i]];
            k_recon_sq += kc * kc;
        }
        float k_recon_norm = sqrtf(k_recon_sq);
        float ks = (k_recon_norm > 1e-10f) ? (sqrtf(k_norm_sq) / k_recon_norm) : (k_max / TURBO4_MAX);
        if (ks > FP8_E4M3_MAX) ks = FP8_E4M3_MAX;
        ((__nv_fp8_storage_t*)(block_k + k_scale_off))[g] = float_to_fp8(ks);

        unsigned char* kd = block_k + k_data_off + elem_offset / 2;
        for (int i = 0; i < 16; i += 2) {
            kd[i/2] = ki[i] | (ki[i+1] << 4);
        }

        // V-side: turbo8 amax scale (BF16 scale).
        float v_scale = v_max / FP8_E4M3_MAX;
        if (v_scale < 1e-12f) v_scale = 1e-12f;
        ((__nv_bfloat16*)(block_v + v_scale_off))[g] = __float2bfloat16(v_scale);

        float v_inv = 1.0f / v_scale;
        unsigned char* vd = block_v + v_data_off + elem_offset;
        for (int i = 0; i < 16; i++) {
            float vs = fminf(fmaxf(vf[i] * v_inv, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            vd[i] = (unsigned char)float_to_fp8(vs);
        }
    }
}

// ── Turbo3K + Turbo8V reshape_and_cache (asymmetric) ──
//
// K: turbo3 3-bit Lloyd-Max + FP8 group scale (matched-norm L2).
// V: turbo8 FP8 E4M3 + BF16 group scale.
extern "C" __global__ void reshape_and_cache_flash_turbo3k_turbo8v(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,        // turbo3 byte pool
    unsigned char* __restrict__ v_cache,        // turbo8 byte pool
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long k_block_stride_bytes,
    const unsigned long long k_data_section_bytes,
    const unsigned long long v_block_stride_bytes,
    const unsigned long long v_data_section_bytes
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / GROUP_SIZE;

    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // K-side: turbo3 (3-bit packed) + FP8 group scale.
    unsigned char* block_k = k_cache + (unsigned long long)block_idx * k_block_stride_bytes;
    unsigned long long k_data_off = (unsigned long long)block_offset * (n_elems * 3 / 8);
    unsigned long long k_scale_off = k_data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    // V-side: turbo8 (1 byte/elem FP8) + BF16 group scale.
    unsigned char* block_v = v_cache + (unsigned long long)block_idx * v_block_stride_bytes;
    unsigned long long v_data_off = (unsigned long long)block_offset * n_elems;
    unsigned long long v_scale_off = v_data_section_bytes
        + (unsigned long long)block_offset * num_groups * 2;

    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * GROUP_SIZE;

        float kf[16], vf[16];
        float k_norm_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            kf[i] = __bfloat162float(key_src[elem_offset + i]);
            vf[i] = __bfloat162float(val_src[elem_offset + i]);
            k_norm_sq += kf[i] * kf[i];
        }
        float k_max = 0.0f, v_max = 0.0f;
        for (int i = 0; i < 16; i++) {
            k_max = fmaxf(k_max, fabsf(kf[i]));
            v_max = fmaxf(v_max, fabsf(vf[i]));
        }

        // K-side: turbo3 quantize with matched-norm L2.
        float k_inv = (k_max > 1e-12f) ? (TURBO3_MAX / k_max) : 1.0f;
        unsigned char ki[16];
        float k_recon_sq = 0.0f;
        for (int i = 0; i < 16; i++) {
            ki[i] = turbo3_quantize(kf[i] * k_inv);
            float kc = TURBO3_CODEBOOK[ki[i]];
            k_recon_sq += kc * kc;
        }
        float k_recon_norm = sqrtf(k_recon_sq);
        float ks = (k_recon_norm > 1e-10f) ? (sqrtf(k_norm_sq) / k_recon_norm) : (k_max / TURBO3_MAX);
        if (ks > FP8_E4M3_MAX) ks = FP8_E4M3_MAX;
        ((__nv_fp8_storage_t*)(block_k + k_scale_off))[g] = float_to_fp8(ks);

        // Pack K: 16 × 3-bit into 6 bytes.
        unsigned int k_byte_base = elem_offset * 3 / 8;
        unsigned char* kd = block_k + k_data_off + k_byte_base;
        kd[0] = (ki[0]) | (ki[1] << 3) | (ki[2] << 6);
        kd[1] = (ki[2] >> 2) | (ki[3] << 1) | (ki[4] << 4) | (ki[5] << 7);
        kd[2] = (ki[5] >> 1) | (ki[6] << 2) | (ki[7] << 5);
        kd[3] = (ki[8]) | (ki[9] << 3) | (ki[10] << 6);
        kd[4] = (ki[10] >> 2) | (ki[11] << 1) | (ki[12] << 4) | (ki[13] << 7);
        kd[5] = (ki[13] >> 1) | (ki[14] << 2) | (ki[15] << 5);

        // V-side: turbo8 amax scale (BF16 scale).
        float v_scale = v_max / FP8_E4M3_MAX;
        if (v_scale < 1e-12f) v_scale = 1e-12f;
        ((__nv_bfloat16*)(block_v + v_scale_off))[g] = __float2bfloat16(v_scale);

        float v_inv = 1.0f / v_scale;
        unsigned char* vd = block_v + v_data_off + elem_offset;
        for (int i = 0; i < 16; i++) {
            float vs = fminf(fmaxf(vf[i] * v_inv, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            vd[i] = (unsigned char)float_to_fp8(vs);
        }
    }
}
