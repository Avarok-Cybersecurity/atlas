// SPDX-License-Identifier: AGPL-3.0-only

// Q12 Phase 3: Paged Prefill Flash Attention — FP8 E4M3 KV cache,
// batched same-chunk-len variant. See inferspark_prefill_paged_fp8.cu
// for the single-stream version.
//
// Differences from single-stream:
//   - Per-stream block tables via block_table_ptrs[b] (one pointer per
//     batched stream); block_table_ptrs is uploaded by the dispatch as
//     a device array of pointers.
//   - Q and O are stacked [batch, q_len, num_q_heads, head_dim] — each
//     stream lands at `b * q_len * q_seq_stride` within Q/O.
//   - Grid extended to (num_q_heads, q_chunks, batch_size). blockIdx.z=b.
//
// Constraint: all batched streams share the same q_len, kv_len, q_offset,
// sliding_window, causal_mask_enabled, and FP8 quantisation scales. The
// scheduler `can_batch_prefill_only` gate enforces this.
//
// Validation status: unvalidated against hardware.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ __nv_bfloat16 fp8_to_bf16(__nv_fp8_storage_t b, float scale) {
    float v = __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)) * scale;
    return __float2bfloat16(v);
}

#define PREFILL_BATCHED

#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const float _sc = ((const void*)(cache) == (const void*)K_cache) ? k_scale : v_scale; \
        const unsigned int _cpr = HDIM / 8; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const __nv_fp8_storage_t* _base = (const __nv_fp8_storage_t*)(cache) \
                    + (unsigned long long)_pb * fp8_cache_stride \
                    + (unsigned long long)_bo * num_kv_heads * head_dim \
                    + (unsigned long long)(kvh) * head_dim + _col; \
                __nv_bfloat16 _v[8]; \
                for (int _j = 0; _j < 8; _j++) \
                    _v[_j] = fp8_to_bf16(_base[_j], _sc); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define KERNEL_NAME inferspark_prefill_paged_fp8_batched
#define K_CACHE_TYPE const void* __restrict__
#define V_CACHE_TYPE const void* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const float k_scale \
    , const float v_scale \
    , const unsigned long long fp8_cache_stride
#define KERNEL_PREAMBLE /* nothing */

#include "prefill_paged_compute.cuh"
