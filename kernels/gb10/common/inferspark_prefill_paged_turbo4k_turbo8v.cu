// SPDX-License-Identifier: AGPL-3.0-only

// Paged Prefill Flash Attention — ASYMMETRIC K=Turbo4 (4-bit), V=Turbo8 (FP8).
//
// TurboQuant+ both-sides-quantized asym: K compressed to 4-bit Lloyd-Max
// (16-level codebook) and V kept at FP8 E4M3 + per-group BF16 scale.
// Each pool has its own byte layout, byte block stride, and data-section offset.
//
// K-pool block layout (per layer):
//   [data: block_size * num_kv_heads * head_dim / 2 bytes]  4-bit packed
//   [scales: block_size * num_kv_heads * head_dim / 16 bytes] FP8 E4M3
// V-pool block layout (per layer):
//   [data: block_size * num_kv_heads * head_dim bytes]  FP8 E4M3 (1 byte/elem)
//   [scales: block_size * num_kv_heads * (head_dim/16) * 2 bytes] BF16

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define NVFP4_GROUP_SIZE 16

__device__ __forceinline__ float fp8e4m3_f32_asym_pf(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// ── LOAD_K_TILE: Turbo4 K tile load — sync read + dequant to BF16 in smem ──
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
                const unsigned char* _blk = (const unsigned char*)(cache) \
                    + (unsigned long long)_pb * tq4_k_bsb; \
                const unsigned char* _dp = _blk \
                    + (unsigned long long)_bo * _nkv_hd / 2 \
                    + (unsigned long long)(kvh) * head_dim / 2 + _col / 2; \
                const unsigned int _sg = head_dim / NVFP4_GROUP_SIZE; \
                const unsigned char* _sp = _blk + tq4_k_dsb \
                    + (unsigned long long)_bo * num_kv_heads * _sg \
                    + (unsigned long long)(kvh) * _sg + _col / NVFP4_GROUP_SIZE; \
                float _gs = fp8e4m3_f32_asym_pf((__nv_fp8_storage_t)*_sp); \
                unsigned int _pk = *(const unsigned int*)_dp; \
                __nv_bfloat16 _v[8]; \
                _v[0] = __float2bfloat16(k_lut[(_pk)       & 0xF] * _gs); \
                _v[1] = __float2bfloat16(k_lut[(_pk >> 4)  & 0xF] * _gs); \
                _v[2] = __float2bfloat16(k_lut[(_pk >> 8)  & 0xF] * _gs); \
                _v[3] = __float2bfloat16(k_lut[(_pk >> 12) & 0xF] * _gs); \
                _v[4] = __float2bfloat16(k_lut[(_pk >> 16) & 0xF] * _gs); \
                _v[5] = __float2bfloat16(k_lut[(_pk >> 20) & 0xF] * _gs); \
                _v[6] = __float2bfloat16(k_lut[(_pk >> 24) & 0xF] * _gs); \
                _v[7] = __float2bfloat16(k_lut[_pk >> 28]         * _gs); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

// ── LOAD_V_TILE: Turbo8 V tile load — sync read + dequant to BF16 in smem ──
// 8 elements occupy 8 bytes (uint64) of FP8 E4M3 + 1 BF16 group scale (2 bytes).
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
                    + (unsigned long long)_pb * tq8_v_bsb; \
                const unsigned char* _dp = _blk \
                    + (unsigned long long)_bo * _nkv_hd \
                    + (unsigned long long)(kvh) * head_dim + _col; \
                const unsigned int _sg = head_dim / NVFP4_GROUP_SIZE; \
                /* BF16 scales: 2 bytes per group. */ \
                const unsigned char* _sp = _blk + tq8_v_dsb \
                    + ((unsigned long long)_bo * num_kv_heads * _sg \
                       + (unsigned long long)(kvh) * _sg + _col / NVFP4_GROUP_SIZE) * 2; \
                float _gs = __bfloat162float(*(const __nv_bfloat16*)_sp); \
                unsigned long long _pk8 = *(const unsigned long long*)_dp; \
                __nv_bfloat16 _v[8]; \
                _v[0] = __float2bfloat16(fp8e4m3_f32_asym_pf((__nv_fp8_storage_t)(_pk8 & 0xFF)) * _gs); \
                _v[1] = __float2bfloat16(fp8e4m3_f32_asym_pf((__nv_fp8_storage_t)((_pk8 >> 8) & 0xFF)) * _gs); \
                _v[2] = __float2bfloat16(fp8e4m3_f32_asym_pf((__nv_fp8_storage_t)((_pk8 >> 16) & 0xFF)) * _gs); \
                _v[3] = __float2bfloat16(fp8e4m3_f32_asym_pf((__nv_fp8_storage_t)((_pk8 >> 24) & 0xFF)) * _gs); \
                _v[4] = __float2bfloat16(fp8e4m3_f32_asym_pf((__nv_fp8_storage_t)((_pk8 >> 32) & 0xFF)) * _gs); \
                _v[5] = __float2bfloat16(fp8e4m3_f32_asym_pf((__nv_fp8_storage_t)((_pk8 >> 40) & 0xFF)) * _gs); \
                _v[6] = __float2bfloat16(fp8e4m3_f32_asym_pf((__nv_fp8_storage_t)((_pk8 >> 48) & 0xFF)) * _gs); \
                _v[7] = __float2bfloat16(fp8e4m3_f32_asym_pf((__nv_fp8_storage_t)(_pk8 >> 56)) * _gs); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define KERNEL_NAME inferspark_prefill_paged_turbo4k_turbo8v
#define K_CACHE_TYPE const unsigned char* __restrict__
#define V_CACHE_TYPE const unsigned char* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const unsigned long long tq4_k_bsb \
    , const unsigned long long tq4_k_dsb \
    , const unsigned long long tq8_v_bsb \
    , const unsigned long long tq8_v_dsb
#define KERNEL_PREAMBLE \
    __shared__ float k_lut[16]; \
    if (tid < 16) { \
        const float _kl[16] = { \
            -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f, \
             0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f \
        }; \
        k_lut[tid] = _kl[tid]; \
    } \
    __syncthreads();

#include "prefill_paged_compute_asym.cuh"
