// SPDX-License-Identifier: AGPL-3.0-only
// ldmatrix-enabled rebuild (header content change forces kernel cache miss)

// Paged Prefill Flash Attention — FP8 E4M3 KV cache variant.
//
// Reads FP8 K/V from paged cache, dequantizes to BF16 in shared memory,
// then runs Flash Attention with contiguous BF16 Q.
//
// Grid: (num_q_heads, ceil(q_len/BR), 1)  Block: (128 or 256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ __nv_bfloat16 fp8_to_bf16(__nv_fp8_storage_t b, float scale) {
    float v = __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)) * scale;
    return __float2bfloat16(v);
}

// FP8-smem + cp.async pipelining (Lever #2, 2026-07-19): K/V kept in shared
// memory as raw E4M3 bytes (__nv_fp8_storage_t, halving smem_K + smem_V) and
// dequantized in-register before each MMA. Tile loads use cp.async (16 FP8
// elements per atlas_cp16) so KV-load latency is hidden behind QK^T compute
// of the previous tile — the synchronous fp8→bf16 dequant that made FP8 KV
// 10% SLOWER than BF16 is now deferred to the register read. The in-register
// dequant uses the same fp8_to_bf16 + k_scale/v_scale as the original sync
// path, making the MMA operands (and therefore the kernel output) bit-identical.
// See also: inferspark_prefill_paged_bf16k_turbo4v.cu (BF16 cp.async pattern).
#define ATLAS_ATTN_FP8_SMEM

// FP8 tile loader: cp.async copy of raw E4M3 bytes into fp8-storage smem.
// 16 FP8 elements (16 bytes) per atlas_cp16, matching the 16-byte transaction
// size. Dequant is deferred to in-register MMA read (ATLAS_ATTN_FP8_SMEM path
// in prefill_paged_compute.cuh uses fp8x2_to_*_bits). OOB rows (pos >= kv_len)
// are zeroed synchronously (uint4) — safe because the committed cp.async group
// covers only valid-row entries. cache_stride is in FP8 elements (1 byte each).
//
// ALIGNMENT (2026-07-19 fix): FP8-smem elements are 1 byte, so the row stride
// HDIM_PAD must itself be 16-aligned for the 16-byte atlas_cp16 and uint4
// stores. With the default PAD_KV=8 → HDIM_PAD=264 → 264 mod 16 = 8 (odd rows
// misaligned → silent cp.async misalignment fault, 1/8 tool-call coherence).
// Override PAD_KV=16 → HDIM_PAD=272 → 272 mod 16 = 0 before including the
// shared header (which is #ifndef-guarded so the override sticks). The BF16 /
// NVFP4 paths don't define ATLAS_ATTN_FP8_SMEM, so they keep PAD_KV=8.
#ifdef ATLAS_ATTN_FP8_SMEM
#define PAD_KV 16
#endif
#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 16; \
        for (unsigned int _i = t; _i < TILE_CHUNKS / 2; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 16; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const void* _base = (const void*)( \
                    (const __nv_fp8_storage_t*)(cache) \
                    + (unsigned long long)_pb * fp8_cache_stride \
                    + (unsigned long long)_bo * num_kv_heads * head_dim \
                    + (unsigned long long)(kvh) * head_dim + _col); \
                atlas_cp16(&((__nv_fp8_storage_t(*)[HDIM_PAD])(smem))[_row][_col], _base); \
            } else { \
                *((uint4*)&((__nv_fp8_storage_t(*)[HDIM_PAD])(smem))[_row][_col]) = make_uint4(0,0,0,0); \
            } \
        } \
    } while(0)

#define KERNEL_NAME inferspark_prefill_paged_fp8
#define K_CACHE_TYPE const void* __restrict__
#define V_CACHE_TYPE const void* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const float k_scale \
    , const float v_scale \
    , const unsigned long long fp8_cache_stride
// KERNEL_PREAMBLE: empty — ATLAS_ATTN_FP8_SMEM path defers fp8→bf16 dequant
// to in-register MMA reads (fp8x2_to_*_bits in prefill_paged_compute.cuh), so
// no load-time dq_scale is needed on the host side.
#define KERNEL_PREAMBLE /* nothing */

#include "prefill_paged_compute.cuh"
