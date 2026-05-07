// SPDX-License-Identifier: AGPL-3.0-only

// Paged Prefill Flash Attention — NVFP4 (E2M1 + FP8 group scales) KV cache variant.
//
// Reads NVFP4-packed K/V from paged cache, dequantizes to BF16 in shared memory,
// then runs Flash Attention with contiguous BF16 Q.
//
// Block layout: [data: block_size * nkv * hd/2 bytes][scales: block_size * nkv * hd/GROUP bytes]
// GROUP_SIZE = 16 elements share one FP8 E4M3 scale.
//
// Grid: (num_q_heads, ceil(q_len/BR), 1)  Block: (128 or 256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define NVFP4_GROUP_SIZE 16

__device__ __forceinline__ float fp8e4m3_f32(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// NVFP4 tile loader: load packed nibbles + FP8 scales, dequant to BF16 in smem.
// `cache` is unsigned char* (raw bytes). `nvfp4_bsb` = block_stride_bytes,
// `nvfp4_dsb` = data_section_bytes.
#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 8; \
        const unsigned int _nkv_hd = num_kv_heads * head_dim; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                /* Data pointer: packed E2M1 nibble pairs (2 elements per byte) */ \
                const unsigned char* _blk = (const unsigned char*)(cache) \
                    + (unsigned long long)_pb * nvfp4_bsb; \
                const unsigned char* _dp = _blk \
                    + (unsigned long long)_bo * _nkv_hd / 2 \
                    + (unsigned long long)(kvh) * head_dim / 2 + _col / 2; \
                /* Scale pointer: 1 FP8 per GROUP_SIZE elements */ \
                const unsigned int _sg = head_dim / NVFP4_GROUP_SIZE; \
                const unsigned char* _sp = _blk + nvfp4_dsb \
                    + (unsigned long long)_bo * num_kv_heads * _sg \
                    + (unsigned long long)(kvh) * _sg + _col / NVFP4_GROUP_SIZE; \
                /* Dequant 8 elements (4 bytes of packed data, 1 FP8 scale per 16 elems) */ \
                float _gs = fp8e4m3_f32((__nv_fp8_storage_t)*_sp); \
                unsigned int _pk = *(const unsigned int*)_dp; \
                __nv_bfloat16 _v[8]; \
                _v[0] = __float2bfloat16(e2m1_lut[(_pk)      & 0xF] * _gs); \
                _v[1] = __float2bfloat16(e2m1_lut[(_pk >> 4) & 0xF] * _gs); \
                _v[2] = __float2bfloat16(e2m1_lut[(_pk >> 8) & 0xF] * _gs); \
                _v[3] = __float2bfloat16(e2m1_lut[(_pk >> 12)& 0xF] * _gs); \
                _v[4] = __float2bfloat16(e2m1_lut[(_pk >> 16)& 0xF] * _gs); \
                _v[5] = __float2bfloat16(e2m1_lut[(_pk >> 20)& 0xF] * _gs); \
                _v[6] = __float2bfloat16(e2m1_lut[(_pk >> 24)& 0xF] * _gs); \
                _v[7] = __float2bfloat16(e2m1_lut[_pk >> 28]        * _gs); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define KERNEL_NAME inferspark_prefill_paged_nvfp4
#define K_CACHE_TYPE const unsigned char* __restrict__
#define V_CACHE_TYPE const unsigned char* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const unsigned long long nvfp4_bsb \
    , const unsigned long long nvfp4_dsb
#define KERNEL_PREAMBLE \
    __shared__ float e2m1_lut[16]; \
    if (tid < 16) { \
        const float _lut[16] = { \
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, \
           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f \
        }; \
        e2m1_lut[tid] = _lut[tid]; \
    } \
    __syncthreads();

#include "prefill_paged_compute.cuh"
