// SPDX-License-Identifier: AGPL-3.0-only

// Paged Prefill Flash Attention — ASYMMETRIC K=BF16, V=Turbo2.
//
// TurboQuant+ safer-asym variant: K kept as raw BF16 (preserves attention-
// score fidelity, K-side bandwidth is the smaller half at long context), V
// stored as 2-bit Lloyd-Max packed + per-group FP8 scale (max V-side bw
// savings — 6.4x compression — at a quality cost the canonical two-sided
// WHT rotation on the write path keeps tractable). Uses the asym prefill
// template (prefill_paged_compute_asym.cuh) which takes separate
// `LOAD_K_TILE` and `LOAD_V_TILE` macros — one for BF16 K, one for turbo2 V.
//
// K-pool block layout (per layer):
//   [block_size, num_kv_heads, head_dim]  BF16 contiguous (same as bf16 KV)
// V-pool block layout (per layer):
//   [data: block_size * num_kv_heads * head_dim / 4 bytes]  2-bit packed (4 idx/byte)
//   [scales: block_size * num_kv_heads * head_dim / 16 bytes] FP8 E4M3
//
// Kernel signature adds tq2_v_bsb (V block-stride bytes) + tq2_v_dsb (V
// data-section bytes). K side has no block-stride byte parameter because
// BF16 strides are computed from (block_size, num_kv_heads, head_dim).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define NVFP4_GROUP_SIZE 16

__device__ __forceinline__ float fp8e4m3_f32_asym_pf(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// ── LOAD_K_TILE: BF16 cp.async tile load (same as bf16 prefill loader) ──
//
// Each thread copies a 16-byte (8 BF16 elem) chunk via cp.async.cg.shared.global.
#define LOAD_K_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 8; \
        const unsigned long long _ps = (unsigned long long)cache_block_size * num_kv_heads * head_dim; \
        const unsigned long long _rs = (unsigned long long)num_kv_heads * head_dim; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            unsigned int _sa = __cvta_generic_to_shared(&(smem)[_row][_col]); \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const void* _gm = (const void*)( \
                    (cache) + _pb * _ps + _bo * _rs + (kvh) * head_dim + _col); \
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" \
                             :: "r"(_sa), "l"(_gm)); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

// ── LOAD_V_TILE: Turbo2 V tile load — sync read + dequant to BF16 in smem ──
//
// Identical body to inferspark_prefill_paged_turbo2.cu's LOAD_KV_TILE. 8
// elements occupy 2 bytes (uint16) of 2-bit indices (4 idx per byte).
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

#define KERNEL_NAME inferspark_prefill_paged_bf16k_turbo2v
#define K_CACHE_TYPE const __nv_bfloat16* __restrict__
#define V_CACHE_TYPE const unsigned char* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
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
