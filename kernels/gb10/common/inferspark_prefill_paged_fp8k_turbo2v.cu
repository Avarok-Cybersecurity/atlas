// SPDX-License-Identifier: AGPL-3.0-only

// Paged Prefill Flash Attention — ASYMMETRIC K=FP8 (E4M3), V=Turbo2.
//
// TurboQuant+ asym variant for FP8-attention models: K stored as FP8 with a
// per-tensor `k_scale`, V stored as 2-bit Lloyd-Max packed + per-group FP8
// scale (6.4x V compression — maximum bandwidth savings at long context).
//
// K-pool block layout (per layer):
//   [block_size, num_kv_heads, head_dim]  FP8 contiguous (1 byte/elem)
// V-pool block layout (per layer):
//   [data: block_size * num_kv_heads * head_dim / 4 bytes]  2-bit packed (4 idx/byte)
//   [scales: block_size * num_kv_heads * head_dim / 16 bytes] FP8 E4M3

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define NVFP4_GROUP_SIZE 16

__device__ __forceinline__ float fp8e4m3_f32_asym_pf(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// ── LOAD_K_TILE: FP8 sync read + dequant to BF16 in smem ──
#define LOAD_K_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
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
                const __nv_fp8_storage_t* _base = (const __nv_fp8_storage_t*)(cache) \
                    + (unsigned long long)_pb * cache_block_size * _nkv_hd \
                    + (unsigned long long)_bo * _nkv_hd \
                    + (unsigned long long)(kvh) * head_dim + _col; \
                __nv_bfloat16 _v[8]; \
                for (int _j = 0; _j < 8; _j++) { \
                    float _f = fp8e4m3_f32_asym_pf(_base[_j]) * k_scale; \
                    _v[_j] = __float2bfloat16(_f); \
                } \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

// ── LOAD_V_TILE: Turbo2 V tile load — sync read + dequant ──
//
// Identical body to inferspark_prefill_paged_bf16k_turbo2v.cu's LOAD_V_TILE.
#define LOAD_V_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
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
                    + (unsigned long long)_pb * tq2_v_bsb; \
                const unsigned char* _dp = _blk \
                    + (unsigned long long)_bo * _nkv_hd / 4 \
                    + (unsigned long long)(kvh) * head_dim / 4 + _col / 4; \
                const unsigned int _sg = head_dim / NVFP4_GROUP_SIZE; \
                const unsigned char* _sp = _blk + tq2_v_dsb \
                    + (unsigned long long)_bo * num_kv_heads * _sg \
                    + (unsigned long long)(kvh) * _sg + _col / NVFP4_GROUP_SIZE; \
                float _gs = fp8e4m3_f32_asym_pf((__nv_fp8_storage_t)*_sp); \
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

#define KERNEL_NAME inferspark_prefill_paged_fp8k_turbo2v
#define K_CACHE_TYPE const void* __restrict__
#define V_CACHE_TYPE const unsigned char* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const float k_scale \
    , const unsigned long long tq2_v_bsb \
    , const unsigned long long tq2_v_dsb
#define KERNEL_PREAMBLE \
    __shared__ float e2m1_lut[4]; \
    if (tid < 4) { \
        const float _lut[4] = { -1.5104f, -0.4528f, 0.4528f, 1.5104f }; \
        e2m1_lut[tid] = _lut[tid]; \
    } \
    __syncthreads();

#include "prefill_paged_compute_asym.cuh"
