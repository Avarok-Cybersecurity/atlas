// SPDX-License-Identifier: AGPL-3.0-only
// Idea after Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself derived from
// NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0:
// src/ops/linear/fp8/fp8_rmsnorm_quantize.cu:1-98, an RMSNorm that hands the next FP8 GEMM its
// quantized input in the same pass. Independent implementation for Atlas's residual-add norm and its
// 1x128 activation groups (upstream quantizes per token); no code copied.

// Ada (sm_89) fused residual add + RMSNorm + per-token-group FP8 E4M3
// quantization: the dense FFN's W8A8 PREFILL input in the same pass as the
// norm that produces it (port row `rmsnorm_fp8_quant_fused`, off by default).
//
//   hidden[t, i]   += src[t, i]                              (BF16 store)
//   residual[t, i]  = hidden[t, i]
//   output[t, i]    = bf16( hidden[t, i] * rms_t * (1 + weight[i]) )
//   a_scale[t, g]   = max(max_{i in g} |output[t, i]| / 448, 1e-12)
//   out_fp8[t, i]   = e4m3_satfinite( clamp(output[t, i] / a_scale[t, g], +-448) )
//
// g walks the hidden dimension in groups of 128. It replaces the pair the
// prefill layers launch around the FFN's W8A8 gate/up quantization:
//
//   residual_add_rms_norm(hidden, src, weight, output, residual)  (kernels/gb10/common/rms_norm.cu)
//   per_token_group_quant_fp8(output, out_fp8, a_scale)           (kernels/gb10/common/per_token_group_quant_fp8.cu)
//
// The BF16 `output` is still written (the debug dumps and any non-W8A8 FFN arm
// read it), but the quantizer no longer reads it back: the codes and scales
// come from registers. That removes one read of the normed rows (2 bytes per
// element: 20 MB per layer on the 4B at a 3,921-token chunk) and one launch per
// layer. The host takes this kernel only where the FFN's gate and up both take
// the W8A8 arm (`layers/dense_ffn_w8a8_prefill/norm_fused.rs`).
//
// ── NUMERICS: bit-identical to the pair, by construction ─────────────────
//
//   * Passes 1 and 2 are `residual_add_rms_norm` statement for statement: the
//     same block size (the host launches min(hidden, 1024) threads, as
//     `ops::residual_add_rms_norm` does), the same pair-to-thread map
//     (pair i on thread i mod blockDim, in increasing i), the same FP32 sums
//     `hv + sv` (squared UNROUNDED, as there), the same `warp_reduce_sum`
//     XOR butterfly and two-level block sum, the same
//     `rsqrtf(sum / (float)hidden_size + eps)`, the same
//     `x * rms * (1.0f + w)` rounded by `__float2bfloat16`. Both files sit in
//     kernels/ada/common and get the same nvcc flags (common/KERNEL.toml's
//     `--fmad=false`), so the same expressions lower to the same instructions.
//     The updated hidden pair is kept in a register instead of being re-read;
//     it is the value that was stored.
//   * The quantizer input is that BF16 value widened to FP32, which is exactly
//     what `per_token_group_quant_fp8` reads from memory.
//   * Group max: one warp's 32 pairs in one pass are 64 consecutive elements,
//     half of one 128-group (blockDim and hidden_size are multiples of 128, so
//     a warp's pairs never straddle a group and a warp is either wholly live or
//     wholly idle). Each half-group's max is reduced with an XOR butterfly and
//     the two halves combined. `fmaxf` over absolute values is exact,
//     associative and commutative, and every max is seeded with 0.0f, so a NaN
//     element is dropped exactly as the shared kernel drops it. Same argument
//     as `swiglu_quant_fp8_ada.cu`.
//   * Scale: `amax / 448.0f` (IEEE division) floored at 1e-12f. Element:
//     `v / scale` (IEEE division), the same `fmaxf(fminf(v, 448), -448)`
//     clamp, the same `__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)`.
//
// The norm half is only identical while the layer's norm kernel is the
// offset-from-one `residual_add_rms_norm` (Qwen3 / Qwen3.5). The host never
// takes this kernel for a model that ships vanilla norm weights.
//
// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1).
// Requires hidden_size % 128 == 0 and hidden_size <= RQ_MAX_HIDDEN (the host
// selector's clauses; the kernel traps otherwise rather than write a partial
// row). out_fp8 is [num_tokens, hidden_size], a_scale [num_tokens, hidden/128].

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define RQ_GROUP_K 128
#define RQ_E4M3_MAX 448.0f
#define RQ_MAX_THREADS 1024
// Largest hidden size one CTA covers: 8 BF16 pairs per thread at 1024 threads.
#define RQ_MAX_HIDDEN 16384
#define RQ_MAX_PAIRS (RQ_MAX_HIDDEN / 2 / RQ_MAX_THREADS)
// One warp's pairs in one pass: 32 pairs, 64 elements, half a group.
#define RQ_PAIRS_PER_SEGMENT 32
#define RQ_PAIRS_PER_GROUP (RQ_GROUP_K / 2)
#define RQ_MAX_SEGMENTS (RQ_MAX_HIDDEN / 2 / RQ_PAIRS_PER_SEGMENT)
#define RQ_MAX_GROUPS (RQ_MAX_HIDDEN / RQ_GROUP_K)

// `unpack_bf16x2` / `pack_bf16x2` of rms_norm.cu.
__device__ __forceinline__ void rq_unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int rq_pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

// `warp_reduce_sum` of rms_norm.cu.
__device__ __forceinline__ float rq_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float rq_warp_reduce_max(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, offset));
    }
    return val;
}

extern "C" __global__ void __launch_bounds__(RQ_MAX_THREADS)
    rmsnorm_quant_fp8_ada(__nv_bfloat16* __restrict__ hidden,       // [T, H] in/out (hidden += src)
                          const __nv_bfloat16* __restrict__ src,    // [T, H] added to hidden
                          const __nv_bfloat16* __restrict__ weight, // [H] offset-from-one norm weight
                          __nv_bfloat16* __restrict__ output,       // [T, H] normed BF16
                          __nv_bfloat16* __restrict__ residual,     // [T, H] copy of the updated hidden
                          unsigned char* __restrict__ out_fp8,      // [T, H] E4M3 of `output`
                          float* __restrict__ a_scale,              // [T, H/128] FP32
                          unsigned int hidden_size, float eps) {
    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int half_size = hidden_size / 2;

    // The host's clauses, checked where a violation would corrupt memory.
    // CTA-uniform, so no thread is left waiting at a barrier.
    if (hidden_size > RQ_MAX_HIDDEN || (hidden_size % RQ_GROUP_K) != 0 || (blockDim.x % RQ_GROUP_K) != 0 ||
        blockDim.x * RQ_MAX_PAIRS < half_size) {
        __trap();
    }

    const size_t row = (size_t)token * hidden_size;
    unsigned int* h32 = (unsigned int*)(hidden + row);
    const unsigned int* s32 = (const unsigned int*)(src + row);
    unsigned int* out32 = (unsigned int*)(output + row);
    unsigned int* res32 = (unsigned int*)(residual + row);
    const unsigned int* w32 = (const unsigned int*)weight;

    // Pass 1: hidden += src, sum of squares of the FP32 sums. Pair i lives on
    // thread i mod blockDim; slot j is the j-th trip of the reference loop.
    unsigned int hp[RQ_MAX_PAIRS];
    float sum_sq = 0.0f;
#pragma unroll
    for (int j = 0; j < RQ_MAX_PAIRS; j++) {
        const unsigned int i = tid + (unsigned int)j * blockDim.x;
        hp[j] = 0u;
        if (i < half_size) {
            float hv0, hv1, sv0, sv1;
            rq_unpack_bf16x2(h32[i], hv0, hv1);
            rq_unpack_bf16x2(s32[i], sv0, sv1);
            float new0 = hv0 + sv0;
            float new1 = hv1 + sv1;
            hp[j] = rq_pack_bf16x2(new0, new1);
            h32[i] = hp[j]; // Write updated hidden
            sum_sq += new0 * new0 + new1 * new1;
        }
    }

    // Block-level reduction, as residual_add_rms_norm does it.
    sum_sq = rq_warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = rq_warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    // Pass 2: normalize with (1 + weight), store output and residual, keep the
    // BF16 result in registers, and reduce each warp's half-group max.
    __shared__ float seg_max[RQ_MAX_SEGMENTS];
    __shared__ float group_scale[RQ_MAX_GROUPS];
    float q0[RQ_MAX_PAIRS];
    float q1[RQ_MAX_PAIRS];
#pragma unroll
    for (int j = 0; j < RQ_MAX_PAIRS; j++) {
        const unsigned int i = tid + (unsigned int)j * blockDim.x;
        q0[j] = 0.0f;
        q1[j] = 0.0f;
        // Warp-uniform: half_size and the warp's first pair index are both
        // multiples of 32, so a warp is wholly inside or wholly past the row.
        if (i < half_size) {
            float xv0, xv1, wv0, wv1;
            rq_unpack_bf16x2(hp[j], xv0, xv1);
            rq_unpack_bf16x2(w32[i], wv0, wv1);
            const unsigned int normed = rq_pack_bf16x2(xv0 * rms * (1.0f + wv0), xv1 * rms * (1.0f + wv1));
            out32[i] = normed;
            res32[i] = hp[j]; // Copy to residual
            rq_unpack_bf16x2(normed, q0[j], q1[j]);
            float m = fmaxf(fmaxf(0.0f, fabsf(q0[j])), fabsf(q1[j]));
            m = rq_warp_reduce_max(m);
            if (lane_id == 0) {
                seg_max[i / RQ_PAIRS_PER_SEGMENT] = m;
            }
        }
    }
    __syncthreads();

    // One thread per 128-group: the two half-group maxima, then the scale as
    // `per_token_group_quant_fp8` computes it.
    const unsigned int groups = hidden_size / RQ_GROUP_K;
    if (tid < groups) {
        const float amax = fmaxf(seg_max[2 * tid], seg_max[2 * tid + 1]);
        float scale = amax / RQ_E4M3_MAX;
        if (scale < 1e-12f) scale = 1e-12f;
        a_scale[(size_t)token * groups + tid] = scale;
        group_scale[tid] = scale;
    }
    __syncthreads();

    // Quantize from registers; two E4M3 bytes per pair, low byte first.
    unsigned short* q16 = reinterpret_cast<unsigned short*>(out_fp8 + row);
#pragma unroll
    for (int j = 0; j < RQ_MAX_PAIRS; j++) {
        const unsigned int i = tid + (unsigned int)j * blockDim.x;
        if (i < half_size) {
            const float s = group_scale[i / RQ_PAIRS_PER_GROUP];
            float v0 = q0[j] / s;
            float v1 = q1[j] / s;
            v0 = fmaxf(fminf(v0, RQ_E4M3_MAX), -RQ_E4M3_MAX);
            v1 = fmaxf(fminf(v1, RQ_E4M3_MAX), -RQ_E4M3_MAX);
            const unsigned int b0 = (unsigned int)__nv_cvt_float_to_fp8(v0, __NV_SATFINITE, __NV_E4M3);
            const unsigned int b1 = (unsigned int)__nv_cvt_float_to_fp8(v1, __NV_SATFINITE, __NV_E4M3);
            q16[i] = (unsigned short)(b0 | (b1 << 8));
        }
    }
}
