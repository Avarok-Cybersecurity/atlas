// SPDX-License-Identifier: AGPL-3.0-only

// Paged Prefill Flash Attention — Turbo4 (4-bit Lloyd-Max + FP8 group scales) KV cache variant.
//
// Reads 4-bit packed K/V from paged cache (2 values per byte), dequantizes to
// BF16 in shared memory via the 16-level Lloyd-Max Gaussian codebook, then
// runs Flash Attention with contiguous BF16 Q. Replaces upstream Atlas
// behavior of routing turbo4 through the NVFP4_64 prefill kernel: the byte
// layout is identical, but NVFP4 decodes with the E2M1 LUT
// {0, .5, 1, 1.5, 2, 3, 4, 6, -0, ...} while turbo4 data was encoded against
// the Lloyd-Max codebook below — every history value was systematically
// wrong on chunk≥2 reads of a chunked prefill.
//
// Block layout (must match reshape_and_cache_flash_turbo4):
//   [data: block_size * num_kv_heads * head_dim / 2 bytes]      4-bit packed
//   [scales: block_size * num_kv_heads * head_dim / 16 bytes]   FP8 E4M3

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define NVFP4_GROUP_SIZE 16

__device__ __forceinline__ float fp8e4m3_f32(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// Turbo4 tile loader: 8 elements = 4 bytes packed 4-bit + 1 FP8 scale per 16.
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
                    + (unsigned long long)_pb * tq4_bsb; \
                const unsigned char* _dp = _blk \
                    + (unsigned long long)_bo * _nkv_hd / 2 \
                    + (unsigned long long)(kvh) * head_dim / 2 + _col / 2; \
                const unsigned int _sg = head_dim / NVFP4_GROUP_SIZE; \
                const unsigned char* _sp = _blk + tq4_dsb \
                    + (unsigned long long)_bo * num_kv_heads * _sg \
                    + (unsigned long long)(kvh) * _sg + _col / NVFP4_GROUP_SIZE; \
                float _gs = fp8e4m3_f32((__nv_fp8_storage_t)*_sp); \
                unsigned int _pk = *(const unsigned int*)_dp; \
                __nv_bfloat16 _v[8]; \
                _v[0] = __float2bfloat16(tq4_lut[(_pk)       & 0xF] * _gs); \
                _v[1] = __float2bfloat16(tq4_lut[(_pk >> 4)  & 0xF] * _gs); \
                _v[2] = __float2bfloat16(tq4_lut[(_pk >> 8)  & 0xF] * _gs); \
                _v[3] = __float2bfloat16(tq4_lut[(_pk >> 12) & 0xF] * _gs); \
                _v[4] = __float2bfloat16(tq4_lut[(_pk >> 16) & 0xF] * _gs); \
                _v[5] = __float2bfloat16(tq4_lut[(_pk >> 20) & 0xF] * _gs); \
                _v[6] = __float2bfloat16(tq4_lut[(_pk >> 24) & 0xF] * _gs); \
                _v[7] = __float2bfloat16(tq4_lut[_pk >> 28]         * _gs); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define KERNEL_NAME inferspark_prefill_paged_turbo4
#define K_CACHE_TYPE const unsigned char* __restrict__
#define V_CACHE_TYPE const unsigned char* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const unsigned long long tq4_bsb \
    , const unsigned long long tq4_dsb
#define KERNEL_PREAMBLE \
    __shared__ float tq4_lut[16]; \
    if (tid < 16) { \
        const float _kl[16] = { \
            -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f, \
             0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f \
        }; \
        tq4_lut[tid] = _kl[tid]; \
    } \
    __syncthreads();

#include "prefill_paged_compute.cuh"
