// SPDX-License-Identifier: AGPL-3.0-only
//
// Paged Prefill Flash Attention — BF16 KV cache, INDIRECT scalar args.
//
// Identical to inferspark_prefill_paged.cu except `kv_len`, `q_offset`, and
// `q_rope_pos` are read from device pointers at kernel entry instead of taken
// as scalar kernel args. This makes the kernel graph-friendly: a captured CUDA
// graph holds the pointers; per-call dynamic values are written to the
// pointed-to buffers in pre-graph host code.
//
// Used by the DFlash drafter (BlockDiffusionDraftHead::forward_block) so the
// entire propose path can be captured as a single CUDA graph and replayed
// every step. q_offset addresses the KV cache; q_rope_pos is the true decode
// position for the γ query block's RoPE rotation (decoupled from q_offset).

#include <cuda_bf16.h>

// BF16 tile loader: cp.async from paged cache to shared memory.
// Uses atlas_cp16 (defined in prefill_paged_compute.cuh) so the SCALE/gfx1151
// build degrades to a synchronous uint4 copy. Identical body to
// inferspark_prefill_paged.cu.
#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 8; \
        const unsigned long long _ps = (unsigned long long)cache_block_size * num_kv_heads * head_dim; \
        const unsigned long long _rs = (unsigned long long)num_kv_heads * head_dim; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const void* _gm = (const void*)( \
                    (cache) + _pb * _ps + _bo * _rs + (kvh) * head_dim + _col); \
                atlas_cp16(&(smem)[_row][_col], _gm); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define KERNEL_NAME inferspark_prefill_paged_indirect
#define K_CACHE_TYPE const __nv_bfloat16* __restrict__
#define V_CACHE_TYPE const __nv_bfloat16* __restrict__
#define KERNEL_EXTRA_PARAMS , const float inv_sqrt_d,                          \
                           const unsigned int* __restrict__ kv_len_ptr,        \
                           const unsigned int* __restrict__ q_offset_ptr,      \
                           const unsigned int* __restrict__ q_rope_pos_ptr
// Signal prefill_paged_compute.cuh that KERNEL_PREAMBLE declares q_rope_pos
// (avoids the default `q_rope_pos = q_offset` initializer in the .cuh body).
#define Q_ROPE_POS_OVERRIDE
#define KERNEL_PREAMBLE                                                         \
    /* Read indirect scalar args via shared memory: thread 0 loads, all wait. */\
    /* 12-byte layout: [kv_len, q_offset, q_rope_pos] each u32.              */\
    __shared__ unsigned int s_indirect[3];                                      \
    if (threadIdx.x == 0) {                                                     \
        s_indirect[0] = *kv_len_ptr;                                            \
        s_indirect[1] = *q_offset_ptr;                                          \
        s_indirect[2] = *q_rope_pos_ptr;                                        \
    }                                                                           \
    __syncthreads();                                                            \
    kv_len = s_indirect[0];                                                     \
    q_offset = s_indirect[1];                                                   \
    unsigned int q_rope_pos = s_indirect[2];

#include "prefill_paged_compute.cuh"
