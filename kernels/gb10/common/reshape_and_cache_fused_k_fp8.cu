// SPDX-License-Identifier: AGPL-3.0-only

// Fused k_norm + RoPE + FP8 paged cache write (K and V) for the DECODE path.
//
// Replaces this three-launch chain on the `KvCacheDtype::Fp8` decode path:
//
//   ops::rms_norm(k_out, k_norm, k_out, nkv, hd, eps)   // norm.cu   :: rms_norm
//   ops::rope(q_out, k_out, positions, 1, nq, nkv, ...) // rope.cu   :: rope_forward
//   ops::reshape_and_cache_fp8(k_out, v_out, ...)       // reshape_and_cache.cu
//
// with ONE launch that also carries V. The `rope_forward` launch does not
// disappear — Q still needs it — but it is issued with `num_kv_heads = 0`,
// so it costs the same single launch and does strictly less work. Net effect
// per attention layer per token: 4 launches -> 3.
//
// ─── BIT-IDENTITY IS THE POINT ────────────────────────────────────────────
// `kernels/gb10/common/KERNEL.toml` sets `--fmad=false` and the gate records
// are committed against the exact bytes the unfused chain produces, so this
// kernel deliberately REPRODUCES every intermediate rounding step rather than
// keeping intermediates in FP32:
//
//   * the sum-of-squares is accumulated with `rms_norm`'s per-thread element
//     assignment (thread `t` owns the packed BF16 PAIR `t`, threads past
//     `head_dim/2` contribute an exact 0.0f) and reduced through the same
//     butterfly + `warp_sums[32]` tree, so `warp_sums[0]` is bit-equal;
//   * the normalized value is rounded to BF16 exactly where `rms_norm`'s
//     store rounds it, and parked in shared memory as BF16 BITS — that is
//     what `rope_forward` would have read back out of `k_out`;
//   * the rotated value is rounded to BF16 exactly where `rope_forward`'s
//     store rounds it — that is what `reshape_and_cache_flash_fp8` would
//     have read back;
//   * the FP8 cast is the same paired `__nv_cvt_float2_to_fp8x2` over the
//     same element pairing, reached through the same `1.0f / scale`
//     reciprocal computed in-kernel.
//
// A "more accurate" fused kernel that skipped those rounds would change the
// cache bytes and re-open every committed record for FP8-KV models. Do not
// remove a round here to buy accuracy.
//
// Mirrored into `kernels/{hopper,b200}/common` by symlink, like every other
// gb10 common/ file those targets inherit.
//
// Grid:  (num_tokens, num_kv_heads, 1)
// Block: (head_dim, 1, 1) — MUST equal head_dim, which must be a multiple of
//        32 and at most FUSED_KFP8_MAX_HEAD_DIM. The block geometry is not a
//        tuning knob: it is what makes the reduction tree match `rms_norm`,
//        which is launched as grid=(nkv,1,1) block=(head_dim,1,1).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>

// Shared-memory budget for the normalized row (BF16 bits).
#define FUSED_KFP8_MAX_HEAD_DIM 256

// ── Verbatim from `rms_norm.cu` (same TU-local helpers, renamed to keep this
// module self-contained; the bodies must not drift). ──────────────────────
__device__ __forceinline__ void fkfp8_unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int fkfp8_pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

__device__ __forceinline__ float fkfp8_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ── Verbatim from `reshape_and_cache.cu::bf16x2_to_fp8x2`. ────────────────
__device__ __forceinline__ __nv_fp8x2_storage_t
fkfp8_bf16x2_to_fp8x2(unsigned int packed_bf16, float inv_scale) {
    float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed_bf16 & 0xFFFF)));
    float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed_bf16 >> 16)));
    float2 scaled = make_float2(v0 * inv_scale, v1 * inv_scale);
    return __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
}

// One element of `rope_forward`'s rotate-half, reading the BF16 row the
// `rms_norm` store would have left in `k_out`.
//
// `rope_forward` rotates pairs (d, d + rotary_dim/2) for d < rotary_dim/2 and
// leaves `e >= rotary_dim` untouched — those elements stay at the BF16
// `rms_norm` output, which is what the passthrough arm returns.
//
// `s_cos`/`s_sin` are precomputed once per CTA, for the reason `rope_forward`
// precomputes `s_freq`: the angle depends only on `pair_idx` and the CTA's
// single token, and a thread here owns the ADJACENT pair (2t, 2t+1) — two
// different `pair_idx` — so computing per element would run `rotary_dim` FP64
// `pow`s and `rotary_dim` sincos per head where the kernels this replaces run
// half that. FP64 is 1/64 rate on SM121; that exact redundancy is what left
// `rope_forward` 13x above its bandwidth floor before its own `s_freq` fix.
// Sharing through smem is bit-identical — the same `cosf` of the same
// `angle`, evaluated by one thread instead of two.
__device__ __forceinline__ float fkfp8_rope_elem(
    const unsigned short* __restrict__ s_normed,  // [head_dim] BF16 bits
    const float* __restrict__ s_cos,              // [rotary_dim/2]
    const float* __restrict__ s_sin,              // [rotary_dim/2]
    const unsigned int e,
    const unsigned int rotary_dim
) {
    if (e >= rotary_dim) {
        return __bfloat162float(__ushort_as_bfloat16(s_normed[e]));
    }
    const unsigned int half_rot = rotary_dim / 2;
    const bool is_d0 = (e < half_rot);
    const unsigned int pair_idx = is_d0 ? e : (e - half_rot);
    const float cos_val = s_cos[pair_idx];
    const float sin_val = s_sin[pair_idx];
    const float x0 = __bfloat162float(__ushort_as_bfloat16(s_normed[pair_idx]));
    const float x1 = __bfloat162float(__ushort_as_bfloat16(s_normed[pair_idx + half_rot]));
    return is_d0 ? (x0 * cos_val - x1 * sin_val)
                 : (x1 * cos_val + x0 * sin_val);
}

/// `k_in`     BF16 `[num_tokens, key_stride]`, head slice at `kv_head*head_dim`.
/// `value`    BF16 `[num_tokens, value_stride]` — already carries whatever the
///            caller applied to V (e.g. a v_norm); this kernel only quantizes.
/// `k_cache`/`v_cache` FP8 E4M3 pools, `reshape_and_cache_flash_fp8` layout.
/// `k_scale`/`v_scale` DEQUANT scales (`bf16 = fp8 * scale`); the reciprocal is
///            taken in-kernel so it matches `reshape_and_cache_flash_fp8`.
/// `cache_stride` block-level stride in ELEMENTS, passed by the host for the
///            same reason `reshape_and_cache_flash_fp8` takes it: it is not
///            always `block_size * num_kv_heads * head_dim`.
extern "C" __global__ void fused_k_norm_rope_cache_write_fp8_kv(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ k_norm_weight,   // [head_dim]
    const unsigned int*  __restrict__ positions,       // [num_tokens]
    __nv_fp8_storage_t*  __restrict__ k_cache,
    __nv_fp8_storage_t*  __restrict__ v_cache,
    const long long*     __restrict__ slot_mapping,    // [num_tokens], -1 = skip
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const unsigned int block_size,
    const float k_scale,
    const float v_scale,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long cache_stride,
    const float rms_eps,
    const float theta
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int kv_head   = blockIdx.y;
    const unsigned int tid       = threadIdx.x;

    // Uniform across the block (depends only on blockIdx.x), so returning
    // here cannot strand a `__syncthreads()`.
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned int half_size = head_dim / 2;

    const __nv_bfloat16* k_row = k_in
        + (unsigned long long)token_idx * key_stride
        + (unsigned long long)kv_head * head_dim;
    const unsigned int* x32 = (const unsigned int*)k_row;

    // ── Stage 1 — `rms_norm` step 1+2: sum of squares, same tree. ──
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        fkfp8_unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    sum_sq = fkfp8_warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = fkfp8_warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    // ── Stage 2 — `rms_norm` step 3+4, storing to smem instead of `k_out`. ──
    const float rms = rsqrtf(warp_sums[0] / (float)head_dim + rms_eps);

    // `__align__(4)`: stage 2 stores through an `unsigned int*` view of this
    // array (that is what makes the BF16 rounding identical to
    // `rms_norm`'s `pack_bf16x2` store). A `unsigned short` array is only
    // 2-byte aligned by the language rules, and a misaligned 4-byte shared
    // store is a fault, not a slow path.
    __shared__ __align__(4) unsigned short s_normed[FUSED_KFP8_MAX_HEAD_DIM];
    const unsigned int* w32 = (const unsigned int*)k_norm_weight;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float xv0, xv1, wv0, wv1;
        fkfp8_unpack_bf16x2(x32[i], xv0, xv1);
        fkfp8_unpack_bf16x2(w32[i], wv0, wv1);
        ((unsigned int*)s_normed)[i] =
            fkfp8_pack_bf16x2(xv0 * rms * (1.0f + wv0), xv1 * rms * (1.0f + wv1));
    }

    // `rope_forward`'s `s_freq` precompute, carried one step further to
    // (cos, sin) because this CTA covers exactly one token. Shares the
    // barrier below with the store above — no extra `__syncthreads()`.
    __shared__ float s_cos[FUSED_KFP8_MAX_HEAD_DIM / 2];
    __shared__ float s_sin[FUSED_KFP8_MAX_HEAD_DIM / 2];
    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int abs_pos = positions[token_idx];
    for (unsigned int i = tid; i < half_rot; i += blockDim.x) {
        // FP64 `pow`, the same expression `rope_forward`'s `s_freq` uses.
        const double fe = (double)(2 * i) / (double)rotary_dim;
        const float freq = (float)(1.0 / pow((double)theta, fe));
        const float angle = (float)abs_pos * freq;
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    // Past this point the idle upper half of the block has no work and no
    // barrier left to reach.
    if (tid >= half_size) return;

    // ── Stage 3 — `rope_forward`, then its BF16 store. ──
    const float y0 = fkfp8_rope_elem(s_normed, s_cos, s_sin, 2u * tid, rotary_dim);
    const float y1 = fkfp8_rope_elem(s_normed, s_cos, s_sin, 2u * tid + 1u, rotary_dim);
    const unsigned int packed_k = fkfp8_pack_bf16x2(y0, y1);

    // ── Stage 4 — `reshape_and_cache_flash_fp8`, same pairing and scales. ──
    const unsigned int block_idx    = (unsigned int)(slot / block_size);
    const unsigned int block_offset = (unsigned int)(slot % block_size);
    const unsigned int n_elems      = num_kv_heads * head_dim;

    __nv_fp8_storage_t* key_dst = k_cache
        + (unsigned long long)block_idx * cache_stride
        + (unsigned long long)block_offset * n_elems;
    __nv_fp8_storage_t* val_dst = v_cache
        + (unsigned long long)block_idx * cache_stride
        + (unsigned long long)block_offset * n_elems;

    const float inv_k_scale = 1.0f / k_scale;
    const float inv_v_scale = 1.0f / v_scale;

    // `head_dim` is even, so a head slice never straddles a BF16 pair and this
    // index is exactly the `i` the un-fused writer would have used.
    const unsigned int pair = (kv_head * head_dim) / 2u + tid;

    ((__nv_fp8x2_storage_t*)key_dst)[pair] =
        fkfp8_bf16x2_to_fp8x2(packed_k, inv_k_scale);

    const __nv_bfloat16* v_row = value
        + (unsigned long long)token_idx * value_stride
        + (unsigned long long)kv_head * head_dim;
    ((__nv_fp8x2_storage_t*)val_dst)[pair] =
        fkfp8_bf16x2_to_fp8x2(((const unsigned int*)v_row)[tid], inv_v_scale);
}
