// SPDX-License-Identifier: AGPL-3.0-only

// Paged Prefill Flash Attention for HDIM=512 (Gemma-4 full-attention) — BF16 KV cache.
//
// Reads contiguous BF16 Q from GEMM output and BF16 K/V from paged cache via
// block_table. Uses cp.async vectorized loads. See prefill_paged_compute_512.cuh
// for design notes (single-buffered K, 8 warps, dynamic shared memory).
//
// Grid: (num_q_heads, ceil(q_len/32), 1)   Block: (256, 1, 1)
// Required dynamic shared memory: 101,120 bytes (caller passes via .shared_mem()).

#include <cuda_bf16.h>

// 1D BF16 tile loader: cp.async paged cache → smem (no PAD_KV row stride).
#define LOAD_KV_TILE_512(cache, bt, smem_ptr, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM_512 / 8; \
        const unsigned long long _ps = (unsigned long long)cache_block_size * num_kv_heads * head_dim; \
        const unsigned long long _rs = (unsigned long long)num_kv_heads * head_dim; \
        for (unsigned int _i = (t); _i < TILE_CHUNKS_512; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            unsigned int _sa = __cvta_generic_to_shared(&(smem_ptr)[_row * HDIM_512 + _col]); \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const void* _gm = (const void*)( \
                    (cache) + _pb * _ps + _bo * _rs + (kvh) * head_dim + _col); \
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" \
                             :: "r"(_sa), "l"(_gm)); \
            } else { \
                *((uint4*)&(smem_ptr)[_row * HDIM_512 + _col]) = make_uint4(0,0,0,0); \
            } \
        } \
    } while(0)

#define KERNEL_NAME inferspark_prefill_paged_512
#define K_CACHE_TYPE const __nv_bfloat16* __restrict__
#define V_CACHE_TYPE const __nv_bfloat16* __restrict__
#define KERNEL_EXTRA_PARAMS , const float inv_sqrt_d
#define KERNEL_PREAMBLE /* nothing */

#include "prefill_paged_compute_512.cuh"
