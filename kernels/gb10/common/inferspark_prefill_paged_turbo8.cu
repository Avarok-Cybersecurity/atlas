// SPDX-License-Identifier: AGPL-3.0-only

// Paged Prefill Flash Attention — Turbo8 (FP8 E4M3 + BF16 group scales) KV cache variant.
//
// Reads 1-byte-per-element FP8 K/V from paged cache, dequantizes to BF16 in
// shared memory using per-group BF16 scales, then runs Flash Attention with
// contiguous BF16 Q. Replaces upstream Atlas behavior of routing turbo8
// through the NVFP4_64 prefill kernel, which read the FP8 data section at a
// 4-bit row stride (half of every block's rows aliased) and the BF16 scale
// section as 1-byte E4M3 at half stride — unbounded-magnitude garbage on
// every chunk≥2 history read of a chunked prefill.
//
// Block layout (must match reshape_and_cache_flash_turbo8):
//   [data: block_size * num_kv_heads * head_dim bytes]            FP8 E4M3
//   [scales: block_size * num_kv_heads * (head_dim/16) * 2 bytes] BF16

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define NVFP4_GROUP_SIZE 16

__device__ __forceinline__ float fp8e4m3_f32(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// Turbo8 tile loader: 8 elements = 8 bytes FP8 E4M3 + 1 BF16 group scale.
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
                const unsigned char* _blk = (const unsigned char*)(cache) \
                    + (unsigned long long)_pb * tq8_bsb; \
                const unsigned char* _dp = _blk \
                    + (unsigned long long)_bo * _nkv_hd \
                    + (unsigned long long)(kvh) * head_dim + _col; \
                const unsigned int _sg = head_dim / NVFP4_GROUP_SIZE; \
                /* BF16 scales: 2 bytes per group. */ \
                const unsigned char* _sp = _blk + tq8_dsb \
                    + ((unsigned long long)_bo * num_kv_heads * _sg \
                       + (unsigned long long)(kvh) * _sg + _col / NVFP4_GROUP_SIZE) * 2; \
                float _gs = __bfloat162float(*(const __nv_bfloat16*)_sp); \
                unsigned long long _pk8 = *(const unsigned long long*)_dp; \
                __nv_bfloat16 _v[8]; \
                _v[0] = __float2bfloat16(fp8e4m3_f32((__nv_fp8_storage_t)(_pk8 & 0xFF)) * _gs); \
                _v[1] = __float2bfloat16(fp8e4m3_f32((__nv_fp8_storage_t)((_pk8 >> 8) & 0xFF)) * _gs); \
                _v[2] = __float2bfloat16(fp8e4m3_f32((__nv_fp8_storage_t)((_pk8 >> 16) & 0xFF)) * _gs); \
                _v[3] = __float2bfloat16(fp8e4m3_f32((__nv_fp8_storage_t)((_pk8 >> 24) & 0xFF)) * _gs); \
                _v[4] = __float2bfloat16(fp8e4m3_f32((__nv_fp8_storage_t)((_pk8 >> 32) & 0xFF)) * _gs); \
                _v[5] = __float2bfloat16(fp8e4m3_f32((__nv_fp8_storage_t)((_pk8 >> 40) & 0xFF)) * _gs); \
                _v[6] = __float2bfloat16(fp8e4m3_f32((__nv_fp8_storage_t)((_pk8 >> 48) & 0xFF)) * _gs); \
                _v[7] = __float2bfloat16(fp8e4m3_f32((__nv_fp8_storage_t)(_pk8 >> 56)) * _gs); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define KERNEL_NAME inferspark_prefill_paged_turbo8
#define K_CACHE_TYPE const unsigned char* __restrict__
#define V_CACHE_TYPE const unsigned char* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const unsigned long long tq8_bsb \
    , const unsigned long long tq8_dsb
#define KERNEL_PREAMBLE

#include "prefill_paged_compute.cuh"
