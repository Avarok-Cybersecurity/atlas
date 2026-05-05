// SPDX-License-Identifier: AGPL-3.0-only

// reshape_and_cache_flash — Write K/V tokens to paged KV cache (NHD layout).
//
// Compatible with vLLM's reshape_and_cache_flash interface for FlashAttention/FlashInfer.
//
// Input:
//   key:   [num_tokens, num_kv_heads, head_dim]  BF16 contiguous
//   value: [num_tokens, num_kv_heads, head_dim]  BF16 contiguous
//
// Output (paged cache, NHD layout):
//   k_cache: [num_blocks, block_size, num_kv_heads, head_dim]  BF16
//   v_cache: [num_blocks, block_size, num_kv_heads, head_dim]  BF16
//
// Addressing:
//   slot_mapping[token_idx] = physical_block_number * block_size + offset_within_block
//   slot_mapping[token_idx] == -1  means padding token (skip)
//
// Grid: (num_tokens, 1, 1)
// Block: (256, 1, 1)
//
// Each block copies one token's K and V (num_kv_heads * head_dim BF16 elements each)
// to the correct paged cache location.

#include <cuda_bf16.h>

extern "C" __global__ void reshape_and_cache_flash(
    const __nv_bfloat16* __restrict__ key,       // [num_tokens, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ value,     // [num_tokens, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ k_cache,         // [num_blocks, block_size, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ v_cache,         // [num_blocks, block_size, num_kv_heads, head_dim]
    const long long* __restrict__ slot_mapping,  // [num_tokens], int64
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,               // key.stride(0) in elements (may differ from n_elems for non-contiguous views)
    const unsigned int value_stride              // value.stride(0) in elements
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];

    // Skip padding tokens (slot == -1)
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);

    // Total contiguous elements within a single token's K/V data
    const unsigned int n_elems = num_kv_heads * head_dim;

    // Source: strided [token_idx, :] — inner dims contiguous, row stride may differ
    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // Destination: paged [block_idx, block_offset, :]
    const unsigned long long cache_stride = (unsigned long long)block_size * n_elems;
    __nv_bfloat16* key_dst = k_cache + (unsigned long long)block_idx * cache_stride
                                      + (unsigned long long)block_offset * n_elems;
    __nv_bfloat16* val_dst = v_cache + (unsigned long long)block_idx * cache_stride
                                      + (unsigned long long)block_offset * n_elems;

    // Vectorized copy: 4 BF16 elements (8 bytes) per thread per step
    // using uint2 (8 bytes = 4 x BF16)
    const unsigned int n_vec = n_elems / 4;  // number of uint2 vectors
    const unsigned int n_rem = n_elems % 4;  // remaining elements

    const uint2* key_src_vec = (const uint2*)key_src;
    const uint2* val_src_vec = (const uint2*)val_src;
    uint2* key_dst_vec = (uint2*)key_dst;
    uint2* val_dst_vec = (uint2*)val_dst;

    for (unsigned int i = threadIdx.x; i < n_vec; i += blockDim.x) {
        key_dst_vec[i] = key_src_vec[i];
        val_dst_vec[i] = val_src_vec[i];
    }

    // Handle remainder (if num_kv_heads * head_dim not divisible by 4)
    if (n_rem > 0) {
        unsigned int base = n_vec * 4;
        for (unsigned int i = threadIdx.x; i < n_rem; i += blockDim.x) {
            key_dst[base + i] = key_src[base + i];
            val_dst[base + i] = val_src[base + i];
        }
    }
}


// ============================================================================
// FP8 E4M3 variant — Quantize BF16 input to FP8 paged cache with per-tensor scale.
//
// Conversion: fp8_val = clamp(bf16_val / scale, -448, 448)
// Uses vectorized BF16 reads (uint32 = 2 BF16) and paired FP8 writes
// (__nv_fp8x2_storage_t = 2 FP8 elements per 16-bit store).
//
// Grid: (num_tokens, 1, 1)
// Block: (256, 1, 1)
// ============================================================================

#include <cuda_fp8.h>

// Vectorized BF16->FP8 quantization: read 2 BF16, scale, convert to 2 packed FP8
__device__ __forceinline__ __nv_fp8x2_storage_t
bf16x2_to_fp8x2(unsigned int packed_bf16, float inv_scale) {
    // Unpack 2 BF16 values from uint32
    float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed_bf16 & 0xFFFF)));
    float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed_bf16 >> 16)));
    // Scale and convert to paired FP8
    float2 scaled = make_float2(v0 * inv_scale, v1 * inv_scale);
    return __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
}

extern "C" __global__ void reshape_and_cache_flash_fp8(
    const __nv_bfloat16* __restrict__ key,                // [num_tokens, num_kv_heads, head_dim] BF16
    const __nv_bfloat16* __restrict__ value,              // [num_tokens, num_kv_heads, head_dim] BF16
    __nv_fp8_storage_t* __restrict__ k_cache,             // [num_blocks, block_size, num_kv_heads, head_dim] FP8
    __nv_fp8_storage_t* __restrict__ v_cache,             // [num_blocks, block_size, num_kv_heads, head_dim] FP8
    const long long* __restrict__ slot_mapping,           // [num_tokens], int64
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float k_scale,                                  // dequant: bf16 = fp8 * k_scale
    const float v_scale,                                  // dequant: bf16 = fp8 * v_scale
    const unsigned int key_stride,                        // key.stride(0) in elements
    const unsigned int value_stride,                      // value.stride(0) in elements
    const unsigned long long cache_stride                 // k_cache.stride(0) in elements (block-level stride)
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];

    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;

    // Source: BF16 strided [token_idx, :] — inner dims contiguous, row stride may differ
    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // Destination: FP8 paged [block_idx, block_offset, :]
    // cache_stride passed from host — may differ from block_size * n_elems when
    // vLLM allocates BF16-sized cache and views as uint8 for FP8 storage.
    __nv_fp8_storage_t* key_dst = k_cache + (unsigned long long)block_idx * cache_stride
                                           + (unsigned long long)block_offset * n_elems;
    __nv_fp8_storage_t* val_dst = v_cache + (unsigned long long)block_idx * cache_stride
                                           + (unsigned long long)block_offset * n_elems;

    // Precompute reciprocal scales (quantize: fp8 = bf16 / scale)
    const float inv_k_scale = 1.0f / k_scale;
    const float inv_v_scale = 1.0f / v_scale;

    // Vectorized path: process 2 BF16 elements per iteration
    // Read uint32 (2 BF16), convert to fp8x2 (2 FP8 packed in uint16), store
    const unsigned int n_pairs = n_elems / 2;
    const unsigned int n_rem = n_elems % 2;

    const unsigned int* key_src32 = (const unsigned int*)key_src;
    const unsigned int* val_src32 = (const unsigned int*)val_src;
    __nv_fp8x2_storage_t* key_dst16 = (__nv_fp8x2_storage_t*)key_dst;
    __nv_fp8x2_storage_t* val_dst16 = (__nv_fp8x2_storage_t*)val_dst;

    for (unsigned int i = threadIdx.x; i < n_pairs; i += blockDim.x) {
        key_dst16[i] = bf16x2_to_fp8x2(key_src32[i], inv_k_scale);
        val_dst16[i] = bf16x2_to_fp8x2(val_src32[i], inv_v_scale);
    }

    // Handle odd remainder element (unlikely for head_dim=256, but correct)
    if (n_rem > 0 && threadIdx.x == 0) {
        unsigned int base = n_pairs * 2;
        float kf = __bfloat162float(key_src[base]) * inv_k_scale;
        float vf = __bfloat162float(val_src[base]) * inv_v_scale;
        key_dst[base] = __nv_cvt_float_to_fp8(kf, __NV_SATFINITE, __NV_E4M3);
        val_dst[base] = __nv_cvt_float_to_fp8(vf, __NV_SATFINITE, __NV_E4M3);
    }
}


// ============================================================================
// NVFP4 E2M1 variant — Quantize BF16 input to NVFP4 paged cache with
// per-group FP8 scales (GROUP_SIZE=16).
//
// Block layout per K or V:
//   [data section: packed E2M1 nibbles][scale section: FP8 per-group scales]
//   data:   block_size * num_kv_heads * (head_dim / 2) bytes
//   scales: block_size * num_kv_heads * (head_dim / 16) bytes
//
// Each thread processes one group of 16 BF16 elements:
//   1. Compute group absmax
//   2. Compute FP8 scale = absmax / 6.0
//   3. Quantize each element to E2M1 nibble
//   4. Pack pairs into bytes, write to data section
//   5. Write FP8 scale to scale section
//
// Grid: (num_tokens, 1, 1)
// Block: (256, 1, 1)
// ============================================================================

#define NVFP4_GROUP_SIZE 16

// Branchless float-to-E2M1 conversion (7 unsigned int comparisons, zero divergence)
__device__ __forceinline__ unsigned char nvfp4_float_to_e2m1(float x) {
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

// Quantize one group of 16 BF16 elements → 8 packed bytes + 1 FP8 scale
__device__ void nvfp4_quantize_group(
    const __nv_bfloat16* __restrict__ src,
    unsigned char* __restrict__ data_dst,
    __nv_fp8_storage_t* __restrict__ scale_dst
) {
    // Load 16 BF16 values and compute absmax
    float vals[NVFP4_GROUP_SIZE];
    float absmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < NVFP4_GROUP_SIZE; i++) {
        vals[i] = __bfloat162float(src[i]);
        float av = fabsf(vals[i]);
        absmax = fmaxf(absmax, av);
    }

    // Compute FP8 scale: maps absmax to E2M1 range [0, 6.0]
    // fp8_scale = absmax / 6.0 (clamped to avoid divide-by-zero)
    float fp8_scale_f = absmax * (1.0f / 6.0f);
    // Convert scale to FP8 E4M3 for storage
    __nv_fp8_storage_t fp8_scale = __nv_cvt_float_to_fp8(
        fp8_scale_f, __NV_SATFINITE, __NV_E4M3
    );
    *scale_dst = fp8_scale;

    // Effective scale for dequant = fp8_to_float(fp8_scale)
    // For quant, we need inv_scale = 1.0 / fp8_to_float(fp8_scale)
    float dequant_scale = __half2float(__nv_cvt_fp8_to_halfraw(fp8_scale, __NV_E4M3));
    float inv_scale = (dequant_scale > 0.0f) ? (1.0f / dequant_scale) : 0.0f;

    // Quantize each element and pack pairs into bytes
    #pragma unroll
    for (int i = 0; i < NVFP4_GROUP_SIZE; i += 2) {
        unsigned char lo = nvfp4_float_to_e2m1(vals[i] * inv_scale);
        unsigned char hi = nvfp4_float_to_e2m1(vals[i + 1] * inv_scale);
        data_dst[i / 2] = lo | (hi << 4);
    }
}

extern "C" __global__ void reshape_and_cache_flash_nvfp4(
    const __nv_bfloat16* __restrict__ key,            // [num_tokens, num_kv_heads, head_dim] BF16
    const __nv_bfloat16* __restrict__ value,          // [num_tokens, num_kv_heads, head_dim] BF16
    unsigned char* __restrict__ k_cache,              // [num_blocks, block_bytes] raw bytes
    unsigned char* __restrict__ v_cache,              // [num_blocks, block_bytes] raw bytes
    const long long* __restrict__ slot_mapping,       // [num_tokens], int64
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,                    // key.stride(0) in BF16 elements
    const unsigned int value_stride,                  // value.stride(0) in BF16 elements
    const unsigned long long block_stride_bytes,      // total bytes per block (data + scales)
    const unsigned long long data_section_bytes       // bytes in data section per block
) {
    const unsigned int token_idx = blockIdx.x;
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int block_idx = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned int num_groups = n_elems / NVFP4_GROUP_SIZE;

    // Source pointers (BF16)
    const __nv_bfloat16* key_src = key + (unsigned long long)token_idx * key_stride;
    const __nv_bfloat16* val_src = value + (unsigned long long)token_idx * value_stride;

    // Destination: within the block, data section is at offset 0,
    // scale section is at offset data_section_bytes.
    // Within each section, layout is [block_size, num_kv_heads, ...].
    unsigned char* block_base_k = k_cache + (unsigned long long)block_idx * block_stride_bytes;
    unsigned char* block_base_v = v_cache + (unsigned long long)block_idx * block_stride_bytes;

    // Data offset for this token's position within the block
    unsigned long long data_offset = (unsigned long long)block_offset * (n_elems / 2);
    // Scale offset for this token's position within the block
    unsigned long long scale_offset = data_section_bytes
        + (unsigned long long)block_offset * num_groups;

    unsigned char* key_data = block_base_k + data_offset;
    unsigned char* val_data = block_base_v + data_offset;
    __nv_fp8_storage_t* key_scales = (__nv_fp8_storage_t*)(block_base_k + scale_offset);
    __nv_fp8_storage_t* val_scales = (__nv_fp8_storage_t*)(block_base_v + scale_offset);

    // Each thread handles one group of 16 elements
    for (unsigned int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        unsigned int elem_offset = g * NVFP4_GROUP_SIZE;
        nvfp4_quantize_group(
            key_src + elem_offset,
            key_data + elem_offset / 2,
            key_scales + g
        );
        nvfp4_quantize_group(
            val_src + elem_offset,
            val_data + elem_offset / 2,
            val_scales + g
        );
    }
}


// ============================================================================
// bf16_absmax — Compute max absolute value of a BF16 buffer.
//
// Used for FP8 KV cache online scale calibration: track max |K| and max |V|
// during warmup tokens to compute per-tensor scales.
//
// Input:
//   data:       [n_elems] BF16 values
//   n_elems:    total number of BF16 elements
//
// Output:
//   out_max:    [1] f32 — atomicMax updates (caller must initialize to 0.0)
//
// Uses warp-level reduction + shared memory + global atomic for single-kernel
// absmax over arbitrary-length buffers.
//
// Grid: (1, 1, 1)   or   (ceil(n_elems / (256*8)), 1, 1) for large buffers
// Block: (256, 1, 1)
// ============================================================================

// Atomic max for float using CAS loop (CUDA lacks native atomicMax for float)
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    if (val <= 0.0f) return;
    unsigned int* addr_as_ui = (unsigned int*)addr;
    unsigned int old = *addr_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        float old_val = __uint_as_float(assumed);
        if (old_val >= val) return;
        old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
    } while (assumed != old);
}

extern "C" __global__ void bf16_absmax(
    const __nv_bfloat16* __restrict__ data,  // [n_elems] BF16
    float* __restrict__ out_max,              // [1] f32 output (atomicMax)
    const unsigned int n_elems
) {
    // Each thread processes multiple elements (coalesced reads)
    float local_max = 0.0f;
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;

    // Vectorized: read 2 BF16 per iteration via uint32
    const unsigned int n_pairs = n_elems / 2;
    const unsigned int* data32 = (const unsigned int*)data;
    for (unsigned int i = tid; i < n_pairs; i += stride) {
        unsigned int packed = data32[i];
        float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
        float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
        float a0 = fabsf(v0);
        float a1 = fabsf(v1);
        local_max = fmaxf(local_max, fmaxf(a0, a1));
    }
    // Handle odd remainder
    if (n_elems % 2 != 0 && tid == 0) {
        float v = fabsf(__bfloat162float(data[n_elems - 1]));
        local_max = fmaxf(local_max, v);
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    // First thread in each warp writes to shared memory
    __shared__ float warp_max[8]; // 256 threads / 32 = 8 warps
    unsigned int warp_id = threadIdx.x / 32;
    unsigned int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    // First warp reduces across all warps in this block
    if (warp_id == 0 && lane_id < 8) {
        float val = warp_max[lane_id];
        for (int offset = 4; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xff, val, offset));
        }
        if (lane_id == 0) {
            atomicMaxFloat(out_max, val);
        }
    }
}
