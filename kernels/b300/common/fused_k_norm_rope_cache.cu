// SPDX-License-Identifier: AGPL-3.0-only

// Fused k_norm + RoPE + paged cache-write for the prefill K path.
//
// Replaces the legacy three-kernel chain:
//   ops::rms_norm(k_contiguous, k_norm_weight, k_contiguous)   // BF16→FP32→BF16
//   ops::rope(q_contiguous, k_contiguous, positions, ...)       // BF16→FP32→BF16
//   ops::reshape_and_cache_flash(k_contiguous, v_contiguous, ...) // BF16 memcopy
//
// with a single fused pass that keeps K in FP32 internally and rounds
// to the cache dtype ONCE at write time. Eliminates two BF16 rounding
// stages that compound at deep attention layers (L35-L39) where K
// magnitudes peak ~18× vs L0. The doubled rounding is masked by FP8
// KV cache's coarser quantization noise but exposed by BF16 KV cache,
// producing the documented L35-L39 cliff (memory:
// `project_qwen36_phase2b_softmax_expf.md`).
//
// V is NOT processed here — V skips k_norm and RoPE in standard
// (non-Gemma-4) Qwen3.6 inference. Use the existing
// `reshape_and_cache_flash` for V.
//
// Grid: (num_tokens, num_kv_heads, 1)
// Block: (head_dim, 1, 1) — must equal head_dim, must be ≤ 256.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>

// Maximum supported head_dim (smem size budget).
#define FUSED_KV_MAX_HEAD_DIM 256

__device__ __forceinline__ float warp_reduce_sum_fkv(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
    }
    return v;
}

/// Fused K-path: rms_norm → RoPE → BF16 paged cache write.
///
/// Arithmetic stays in FP32 from the BF16 input load all the way to the
/// final BF16 store, matching vLLM's K-side precision regime.
///
/// `k_in`: BF16 from GEMM, layout `[num_tokens, num_kv_heads, head_dim]`.
/// `k_norm_weight`: BF16 `[head_dim]`. RMSNorm formula matches Qwen3-Next:
///     out = x * rsqrt(mean(x²) + eps) * (1 + weight)
/// `positions`: u32 `[num_tokens]`, absolute sequence positions for RoPE.
/// `k_cache`: BF16 paged cache `[num_blocks, block_size, num_kv_heads, head_dim]`.
/// `slot_mapping`: i64 `[num_tokens]`, paged slot index per token (=-1 → skip).
/// `rotary_dim`: number of leading dims to rotate (e.g., 64 for Qwen3.6).
///   Pairs: (d0, d0+rotary_dim/2) for d0 in [0, rotary_dim/2).
/// `block_size`: cache page size (Atlas: 16).
extern "C" __global__ void fused_k_norm_rope_cache_write_bf16(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const unsigned int* __restrict__ positions,
    __nv_bfloat16* __restrict__ k_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const unsigned int block_size,
    const float rms_eps,
    const float theta
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int kv_head = blockIdx.y;
    const unsigned int t = threadIdx.x;
    if (t >= head_dim) return;

    // Page slot lookup. Skip padding tokens early.
    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    // Source: k_in[token_idx, kv_head, :]
    const unsigned long long src_off =
        (unsigned long long)token_idx * num_kv_heads * head_dim
        + (unsigned long long)kv_head * head_dim;
    const __nv_bfloat16* k_row = k_in + src_off;

    // Phase 1: Load BF16 → FP32. Each thread owns one element.
    float x = __bfloat162float(k_row[t]);

    // Phase 2: RMS over the head_dim. Block-wide reduction.
    float sum_sq = x * x;
    sum_sq = warp_reduce_sum_fkv(sum_sq);

    __shared__ float warp_sums[8]; // up to 256 threads / 32 = 8 warps
    const unsigned int warp_id = t / 32;
    const unsigned int lane_id = t % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        const unsigned int num_warps = (head_dim + 31) / 32;
        float v = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        v = warp_reduce_sum_fkv(v);
        if (lane_id == 0) warp_sums[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(warp_sums[0] / (float)head_dim + rms_eps);

    // Phase 3: Apply weight (Qwen3 "(1 + w)" convention) in FP32.
    const float w = __bfloat162float(k_norm_weight[t]);
    const float normed = x * rms * (1.0f + w);

    // Phase 4: RoPE. Share normed values via smem so paired threads can
    // read each other's value without rewriting.
    __shared__ float smem_normed[FUSED_KV_MAX_HEAD_DIM];
    smem_normed[t] = normed;
    __syncthreads();

    float out_val;
    if (t < rotary_dim) {
        const unsigned int half_rot = rotary_dim / 2;
        const bool is_d0 = (t < half_rot);
        const unsigned int pair_idx = is_d0 ? t : (t - half_rot);
        const float x0 = is_d0 ? smem_normed[t] : smem_normed[t - half_rot];
        const float x1 = is_d0 ? smem_normed[t + half_rot] : smem_normed[t];
        // Use FP64 pow to match Atlas's existing rope.cu (precision at high pos).
        const double freq_exp_d = (double)(2u * pair_idx) / (double)rotary_dim;
        const float freq = (float)(1.0 / pow((double)theta, freq_exp_d));
        const unsigned int pos = positions[token_idx];
        const float angle = (float)pos * freq;
        const float cos_val = cosf(angle);
        const float sin_val = sinf(angle);
        out_val = is_d0
            ? (x0 * cos_val - x1 * sin_val)
            : (x1 * cos_val + x0 * sin_val);
    } else {
        out_val = normed; // passthrough for non-rotary dims
    }

    // Phase 5: Single BF16 round + paged-cache write.
    const unsigned int block_idx = (unsigned int)(slot / (long long)block_size);
    const unsigned int block_offset = (unsigned int)(slot % (long long)block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned long long cache_stride = (unsigned long long)block_size * n_elems;
    const unsigned long long dst_off =
        (unsigned long long)block_idx * cache_stride
        + (unsigned long long)block_offset * n_elems
        + (unsigned long long)kv_head * head_dim
        + t;
    k_cache[dst_off] = __float2bfloat16(out_val);
}

/// MRoPE-interleaved variant of `fused_k_norm_rope_cache_write_bf16`.
///
/// Selects the absolute position from one of three streams (pos_t,
/// pos_h, pos_w) based on `pair_idx % 3`, matching Qwen3.6 / Qwen3-VL.
/// For text-only inputs where pos_h == pos_w == pos_t, the result is
/// bit-identical to the scalar-position kernel.
extern "C" __global__ void fused_k_norm_rope_mrope_cache_write_bf16(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    __nv_bfloat16* __restrict__ k_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const unsigned int block_size,
    const float rms_eps,
    const float theta
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int kv_head = blockIdx.y;
    const unsigned int t = threadIdx.x;
    if (t >= head_dim) return;

    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned long long src_off =
        (unsigned long long)token_idx * num_kv_heads * head_dim
        + (unsigned long long)kv_head * head_dim;
    const __nv_bfloat16* k_row = k_in + src_off;

    float x = __bfloat162float(k_row[t]);

    float sum_sq = x * x;
    sum_sq = warp_reduce_sum_fkv(sum_sq);

    __shared__ float warp_sums[8];
    const unsigned int warp_id = t / 32;
    const unsigned int lane_id = t % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        const unsigned int num_warps = (head_dim + 31) / 32;
        float v = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        v = warp_reduce_sum_fkv(v);
        if (lane_id == 0) warp_sums[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(warp_sums[0] / (float)head_dim + rms_eps);

    const float w = __bfloat162float(k_norm_weight[t]);
    const float normed = x * rms * (1.0f + w);

    __shared__ float smem_normed[FUSED_KV_MAX_HEAD_DIM];
    smem_normed[t] = normed;
    __syncthreads();

    float out_val;
    if (t < rotary_dim) {
        const unsigned int half_rot = rotary_dim / 2;
        const bool is_d0 = (t < half_rot);
        const unsigned int pair_idx = is_d0 ? t : (t - half_rot);
        const float x0 = is_d0 ? smem_normed[t] : smem_normed[t - half_rot];
        const float x1 = is_d0 ? smem_normed[t + half_rot] : smem_normed[t];
        // MRoPE: select position stream by pair_idx % 3.
        const unsigned int section = pair_idx % 3u;
        unsigned int abs_pos;
        if (section == 0u) abs_pos = pos_t[token_idx];
        else if (section == 1u) abs_pos = pos_h[token_idx];
        else                    abs_pos = pos_w[token_idx];
        const double freq_exp_d = (double)(2u * pair_idx) / (double)rotary_dim;
        const float freq = (float)(1.0 / pow((double)theta, freq_exp_d));
        const float angle = (float)abs_pos * freq;
        const float cos_val = cosf(angle);
        const float sin_val = sinf(angle);
        out_val = is_d0
            ? (x0 * cos_val - x1 * sin_val)
            : (x1 * cos_val + x0 * sin_val);
    } else {
        out_val = normed;
    }

    const unsigned int block_idx = (unsigned int)(slot / (long long)block_size);
    const unsigned int block_offset = (unsigned int)(slot % (long long)block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned long long cache_stride = (unsigned long long)block_size * n_elems;
    const unsigned long long dst_off =
        (unsigned long long)block_idx * cache_stride
        + (unsigned long long)block_offset * n_elems
        + (unsigned long long)kv_head * head_dim
        + t;
    k_cache[dst_off] = __float2bfloat16(out_val);
}

/// Same as `fused_k_norm_rope_cache_write_bf16` but writes to an FP8
/// paged cache with a per-tensor scale. The final conversion is the
/// only rounding stage.
///
/// `k_cache_fp8`: E4M3 storage layout matching `reshape_and_cache_flash_fp8`.
/// `inv_scale`: 1.0 / k_scale; multiplied into the FP32 value before
///   the saturating FP8 cast.
extern "C" __global__ void fused_k_norm_rope_cache_write_fp8(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const unsigned int* __restrict__ positions,
    __nv_fp8_storage_t* __restrict__ k_cache_fp8,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const unsigned int block_size,
    const float rms_eps,
    const float theta,
    const float inv_scale
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int kv_head = blockIdx.y;
    const unsigned int t = threadIdx.x;
    if (t >= head_dim) return;

    const long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const unsigned long long src_off =
        (unsigned long long)token_idx * num_kv_heads * head_dim
        + (unsigned long long)kv_head * head_dim;
    const __nv_bfloat16* k_row = k_in + src_off;

    float x = __bfloat162float(k_row[t]);

    float sum_sq = x * x;
    sum_sq = warp_reduce_sum_fkv(sum_sq);

    __shared__ float warp_sums[8];
    const unsigned int warp_id = t / 32;
    const unsigned int lane_id = t % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        const unsigned int num_warps = (head_dim + 31) / 32;
        float v = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        v = warp_reduce_sum_fkv(v);
        if (lane_id == 0) warp_sums[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(warp_sums[0] / (float)head_dim + rms_eps);

    const float w = __bfloat162float(k_norm_weight[t]);
    const float normed = x * rms * (1.0f + w);

    __shared__ float smem_normed[FUSED_KV_MAX_HEAD_DIM];
    smem_normed[t] = normed;
    __syncthreads();

    float out_val;
    if (t < rotary_dim) {
        const unsigned int half_rot = rotary_dim / 2;
        const bool is_d0 = (t < half_rot);
        const unsigned int pair_idx = is_d0 ? t : (t - half_rot);
        const float x0 = is_d0 ? smem_normed[t] : smem_normed[t - half_rot];
        const float x1 = is_d0 ? smem_normed[t + half_rot] : smem_normed[t];
        const double freq_exp_d = (double)(2u * pair_idx) / (double)rotary_dim;
        const float freq = (float)(1.0 / pow((double)theta, freq_exp_d));
        const unsigned int pos = positions[token_idx];
        const float angle = (float)pos * freq;
        const float cos_val = cosf(angle);
        const float sin_val = sinf(angle);
        out_val = is_d0
            ? (x0 * cos_val - x1 * sin_val)
            : (x1 * cos_val + x0 * sin_val);
    } else {
        out_val = normed;
    }

    const unsigned int block_idx = (unsigned int)(slot / (long long)block_size);
    const unsigned int block_offset = (unsigned int)(slot % (long long)block_size);
    const unsigned int n_elems = num_kv_heads * head_dim;
    const unsigned long long cache_stride = (unsigned long long)block_size * n_elems;
    const unsigned long long dst_off =
        (unsigned long long)block_idx * cache_stride
        + (unsigned long long)block_offset * n_elems
        + (unsigned long long)kv_head * head_dim
        + t;
    // Saturating FP8 E4M3 cast (matches reshape_and_cache_flash_fp8 semantics).
    const float scaled = out_val * inv_scale;
    k_cache_fp8[dst_off] = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
}
