// SPDX-License-Identifier: AGPL-3.0-only

// Paged Prefill Flash Attention — Turbo2 (2-bit Lloyd-Max + FP8 group scales) KV cache variant.
//
// Reads 2-bit packed K/V from paged cache, dequantizes to BF16 in shared memory,
// then runs Flash Attention with contiguous BF16 Q.
//
// Block layout:
//   [data: block_size * nkv * hd/4 bytes]  (4 indices per byte)
//   [scales: block_size * nkv * hd/GROUP_SIZE bytes]  (FP8 E4M3, 1 byte per 16 elements)
//
// GROUP_SIZE = 16 elements share one FP8 E4M3 scale.
//
// Grid: (num_q_heads, ceil(q_len/BR), 1)  Block: (128 or 256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define NVFP4_GROUP_SIZE 16

__device__ __forceinline__ float fp8e4m3_f32(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// Turbo2 tile loader: read 2-bit packed data + FP8 group scale, dequant to BF16 in smem.
// Each thread chunk processes 8 elements = 2 bytes of packed 2-bit data.
// `cache` is unsigned char*. `tq2_bsb` = block_stride_bytes, `tq2_dsb` = data_section_bytes.
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
                /* Data pointer: 2-bit packed (4 indices per byte) */ \
                const unsigned char* _blk = (const unsigned char*)(cache) \
                    + (unsigned long long)_pb * tq2_bsb; \
                const unsigned char* _dp = _blk \
                    + (unsigned long long)_bo * _nkv_hd / 4 \
                    + (unsigned long long)(kvh) * head_dim / 4 + _col / 4; \
                /* Scale pointer: 1 FP8 per GROUP_SIZE elements (same as nvfp4/turbo3/turbo4) */ \
                const unsigned int _sg = head_dim / NVFP4_GROUP_SIZE; \
                const unsigned char* _sp = _blk + tq2_dsb \
                    + (unsigned long long)_bo * num_kv_heads * _sg \
                    + (unsigned long long)(kvh) * _sg + _col / NVFP4_GROUP_SIZE; \
                /* Dequant 8 elements: 2 bytes of packed 2-bit data, 1 FP8 scale per 16 elems */ \
                float _gs = fp8e4m3_f32((__nv_fp8_storage_t)*_sp); \
                unsigned short _pk = *(const unsigned short*)_dp; \
                __nv_bfloat16 _v[8]; \
                _v[0] = __float2bfloat16(e2m1_lut[(_pk)       & 0x3] * _gs); \
                _v[1] = __float2bfloat16(e2m1_lut[(_pk >> 2)  & 0x3] * _gs); \
                _v[2] = __float2bfloat16(e2m1_lut[(_pk >> 4)  & 0x3] * _gs); \
                _v[3] = __float2bfloat16(e2m1_lut[(_pk >> 6)  & 0x3] * _gs); \
                _v[4] = __float2bfloat16(e2m1_lut[(_pk >> 8)  & 0x3] * _gs); \
                _v[5] = __float2bfloat16(e2m1_lut[(_pk >> 10) & 0x3] * _gs); \
                _v[6] = __float2bfloat16(e2m1_lut[(_pk >> 12) & 0x3] * _gs); \
                _v[7] = __float2bfloat16(e2m1_lut[(_pk >> 14) & 0x3] * _gs); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define KERNEL_NAME inferspark_prefill_paged_turbo2
#define K_CACHE_TYPE const unsigned char* __restrict__
#define V_CACHE_TYPE const unsigned char* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const unsigned long long tq2_bsb \
    , const unsigned long long tq2_dsb
#define KERNEL_PREAMBLE \
    __shared__ float e2m1_lut[4]; \
    if (tid < 4) { \
        const float _lut[4] = { -1.5104f, -0.4528f, 0.4528f, 1.5104f }; \
        e2m1_lut[tid] = _lut[tid]; \
    } \
    __syncthreads();

#include "prefill_paged_compute.cuh"
