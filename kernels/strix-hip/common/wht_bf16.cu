// SPDX-License-Identifier: AGPL-3.0-only

// Walsh-Hadamard Transform (WHT) for BF16 vectors.
// Applied per-head to Q before turbo paged decode and to output after.
//
// Grid: (num_heads, 1, 1)  Block: (32, 1, 1) — one warp per head
// Each warp processes one head (256 elements via 32 threads × 8 values).

#include <cuda_bf16.h>

#ifdef TQ_PLUS_SIGNS
#include "tq_plus_signs.cuh"
#endif

// In-place WHT on 256 elements using butterfly network.
// 32 threads × 8 elements = 256. Stages 0-2 intra-thread, 3-7 inter-thread.
__device__ __forceinline__ void wht256_warp_bf16(float vals[8], unsigned int lane) {
    // Stages 0-2: intra-thread butterflies
    #pragma unroll
    for (int stride = 1; stride <= 4; stride <<= 1) {
        #pragma unroll
        for (int i = 0; i < 8; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = vals[i + j];
                float b = vals[i + j + stride];
                vals[i + j] = a + b;
                vals[i + j + stride] = a - b;
            }
        }
    }
    // Stages 3-7: inter-thread via shuffle
    #pragma unroll
    for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float other = __shfl_xor_sync(0xFFFFFFFF, vals[i], xor_mask);
            if (lane & xor_mask)
                vals[i] = other - vals[i];
            else
                vals[i] = vals[i] + other;
        }
    }
    // Normalize: 1/sqrt(256) = 1/16
    #pragma unroll
    for (int i = 0; i < 8; i++) vals[i] *= 0.0625f;
}

// Apply WHT to each head of a BF16 vector.
// Supports head_dim=128, 256, and 512.
// In-place operation on the BF16 buffer.
extern "C" __global__ void wht_bf16_inplace(
    __nv_bfloat16* __restrict__ data,   // [num_heads, head_dim]
    const unsigned int head_dim         // 128, 256, or 512
) {
    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (lane >= 32) return;

    __nv_bfloat16* head_data = data + (unsigned long long)head * head_dim;

    if (head_dim >= 512) {
        // 512-point WHT: 32 threads × 16 values
        float vals[16];
        #pragma unroll
        for (int i = 0; i < 16; i++)
            vals[i] = __bfloat162float(head_data[lane * 16 + i]);

#ifdef TQ_PLUS_SIGNS
        // TurboQuant+ signs1 before butterfly for hd=512.
        tq_plus::apply_signs_512_pre(vals, lane);
#endif
        // Stages 0-3: intra-thread butterflies (stride 1,2,4,8)
        for (int stride = 1; stride <= 8; stride <<= 1) {
            for (int i = 0; i < 16; i += stride * 2) {
                for (int j = 0; j < stride; j++) {
                    float a = vals[i + j];
                    float b = vals[i + j + stride];
                    vals[i + j] = a + b;
                    vals[i + j + stride] = a - b;
                }
            }
        }
        // Stages 4-8: inter-thread via shuffle
        for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
            for (int i = 0; i < 16; i++) {
                float other = __shfl_xor_sync(0xFFFFFFFF, vals[i], xor_mask);
                if (lane & xor_mask) vals[i] = other - vals[i];
                else vals[i] = vals[i] + other;
            }
        }
        // Normalize: 1/sqrt(512)
        float norm = 1.0f / sqrtf(512.0f);
        for (int i = 0; i < 16; i++) vals[i] *= norm;
#ifdef TQ_PLUS_SIGNS
        // TurboQuant+ signs2 after butterfly+normalize for hd=512.
        tq_plus::apply_signs_512_post(vals, lane);
#endif

        #pragma unroll
        for (int i = 0; i < 16; i++)
            head_data[lane * 16 + i] = __float2bfloat16(vals[i]);
    } else if (head_dim >= 256) {
        // 256-point WHT: 32 threads × 8 values
        float vals[8];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            vals[i] = __bfloat162float(head_data[lane * 8 + i]);
#ifdef TQ_PLUS_SIGNS
        // TurboQuant+ signs1 before butterfly for hd=256.
        tq_plus::apply_signs_256_pre(vals, lane);
#endif
        wht256_warp_bf16(vals, lane);
#ifdef TQ_PLUS_SIGNS
        // TurboQuant+ signs2 after butterfly+normalize for hd=256.
        tq_plus::apply_signs_256_post(vals, lane);
#endif
        #pragma unroll
        for (int i = 0; i < 8; i++)
            head_data[lane * 8 + i] = __float2bfloat16(vals[i]);
    } else {
        // 128-point WHT: 32 threads × 4 values
        float vals[4];
        unsigned int elems_per_thread = head_dim / 32;  // 4 for hd=128
        #pragma unroll
        for (unsigned int i = 0; i < 4; i++) {
            unsigned int idx = lane * elems_per_thread + i;
            vals[i] = (idx < head_dim) ? __bfloat162float(head_data[idx]) : 0.0f;
        }

#ifdef TQ_PLUS_SIGNS
        // TurboQuant+ : apply signs1 BEFORE butterfly (Gaussianize input).
        // Per Google ICLR 2026 — Lindeberg-CLT-driven outlier elimination.
        if (head_dim == 128) tq_plus::apply_signs_128_pre(vals, lane);
#endif

        // 128-point WHT butterfly: stages 0-1 intra-thread, 2-6 inter-thread
        // Stages 0-1: intra-thread (stride 1, 2)
        for (int stride = 1; stride <= 2; stride <<= 1) {
            for (int i = 0; i < 4; i += stride * 2) {
                for (int j = 0; j < stride; j++) {
                    float a = vals[i + j];
                    float b = vals[i + j + stride];
                    vals[i + j] = a + b;
                    vals[i + j + stride] = a - b;
                }
            }
        }
        // Stages 2-6: inter-thread (xor masks 1,2,4,8,16)
        for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
            for (int i = 0; i < 4; i++) {
                float other = __shfl_xor_sync(0xFFFFFFFF, vals[i], xor_mask);
                if (lane & xor_mask) vals[i] = other - vals[i];
                else vals[i] = vals[i] + other;
            }
        }
        // Normalize: 1/sqrt(128) = 1/sqrt(128) ≈ 0.08839
        float norm = 1.0f / sqrtf((float)head_dim);
        for (int i = 0; i < 4; i++) vals[i] *= norm;

#ifdef TQ_PLUS_SIGNS
        // TurboQuant+ : apply signs2 AFTER butterfly+normalize.
        if (head_dim == 128) tq_plus::apply_signs_128_post(vals, lane);
#endif

        #pragma unroll
        for (unsigned int i = 0; i < 4; i++) {
            unsigned int idx = lane * elems_per_thread + i;
            if (idx < head_dim) head_data[idx] = __float2bfloat16(vals[i]);
        }
    }
}

// ── Inverse rotation kernel ──────────────────────────────────────────────────
//
// With plain WHT (TQ_PLUS_SIGNS off): identical to wht_bf16_inplace — WHT is
// self-inverse so this is just an alias.
//
// With TQ_PLUS_SIGNS on: applies signs2 BEFORE butterfly and signs1 AFTER
// (reversed order vs forward). Required because (S2·H·S1)·(S2·H·S1) ≠ I when
// S1 ≠ S2 — the forward kernel applied twice does NOT undo itself.
//
// Used by attention_forward to bring softmax(QK^T)·WHT(V) back to original basis.
extern "C" __global__ void wht_bf16_inplace_inv(
    __nv_bfloat16* __restrict__ data,
    const unsigned int head_dim
) {
    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (lane >= 32) return;

    __nv_bfloat16* head_data = data + (unsigned long long)head * head_dim;

    if (head_dim >= 512) {
        // 512-point WHT — no signs yet, identical to forward.
        // (256/512 sign arrays not yet vendored from llama-cpp-turboquant.)
        float vals[16];
        #pragma unroll
        for (int i = 0; i < 16; i++)
            vals[i] = __bfloat162float(head_data[lane * 16 + i]);

#ifdef TQ_PLUS_SIGNS
        // INVERSE: signs2 first (reversed order).
        tq_plus::apply_signs_512_post(vals, lane);
#endif
                for (int stride = 1; stride <= 8; stride <<= 1) {
            for (int i = 0; i < 16; i += stride * 2) {
                for (int j = 0; j < stride; j++) {
                    float a = vals[i + j];
                    float b = vals[i + j + stride];
                    vals[i + j] = a + b;
                    vals[i + j + stride] = a - b;
                }
            }
        }
        for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
            for (int i = 0; i < 16; i++) {
                float other = __shfl_xor_sync(0xFFFFFFFF, vals[i], xor_mask);
                if (lane & xor_mask) vals[i] = other - vals[i];
                else vals[i] = vals[i] + other;
            }
        }
        float norm = 1.0f / sqrtf(512.0f);
        for (int i = 0; i < 16; i++) vals[i] *= norm;
#ifdef TQ_PLUS_SIGNS
        // INVERSE: signs1 last (reversed order).
        tq_plus::apply_signs_512_pre(vals, lane);
#endif
        #pragma unroll
        for (int i = 0; i < 16; i++)
            head_data[lane * 16 + i] = __float2bfloat16(vals[i]);
    } else if (head_dim >= 256) {
        // 256-point WHT — no signs yet, identical to forward.
        float vals[8];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            vals[i] = __bfloat162float(head_data[lane * 8 + i]);
#ifdef TQ_PLUS_SIGNS
        // INVERSE: signs2 first.
        tq_plus::apply_signs_256_post(vals, lane);
#endif
        wht256_warp_bf16(vals, lane);
#ifdef TQ_PLUS_SIGNS
        // INVERSE: signs1 last.
        tq_plus::apply_signs_256_pre(vals, lane);
#endif
        #pragma unroll
        for (int i = 0; i < 8; i++)
            head_data[lane * 8 + i] = __float2bfloat16(vals[i]);
    } else {
        // 128-point inverse — applies signs2 BEFORE butterfly and signs1 AFTER.
        float vals[4];
        unsigned int elems_per_thread = head_dim / 32;
        #pragma unroll
        for (unsigned int i = 0; i < 4; i++) {
            unsigned int idx = lane * elems_per_thread + i;
            vals[i] = (idx < head_dim) ? __bfloat162float(head_data[idx]) : 0.0f;
        }

#ifdef TQ_PLUS_SIGNS
        // INVERSE: apply signs2 first (reversed order vs forward).
        if (head_dim == 128) tq_plus::apply_signs_128_post(vals, lane);
#endif

        for (int stride = 1; stride <= 2; stride <<= 1) {
            for (int i = 0; i < 4; i += stride * 2) {
                for (int j = 0; j < stride; j++) {
                    float a = vals[i + j];
                    float b = vals[i + j + stride];
                    vals[i + j] = a + b;
                    vals[i + j + stride] = a - b;
                }
            }
        }
        for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
            for (int i = 0; i < 4; i++) {
                float other = __shfl_xor_sync(0xFFFFFFFF, vals[i], xor_mask);
                if (lane & xor_mask) vals[i] = other - vals[i];
                else vals[i] = vals[i] + other;
            }
        }
        float norm = 1.0f / sqrtf((float)head_dim);
        for (int i = 0; i < 4; i++) vals[i] *= norm;

#ifdef TQ_PLUS_SIGNS
        // INVERSE: apply signs1 after.
        if (head_dim == 128) tq_plus::apply_signs_128_pre(vals, lane);
#endif

        #pragma unroll
        for (unsigned int i = 0; i < 4; i++) {
            unsigned int idx = lane * elems_per_thread + i;
            if (idx < head_dim) head_data[idx] = __float2bfloat16(vals[i]);
        }
    }
}
