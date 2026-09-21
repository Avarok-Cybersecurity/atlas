// SPDX-License-Identifier: AGPL-3.0-only

// Hopper paged-decode SPLIT-K, FP8 E4M3 KV cache (#928).
//
// An ADDITION, not an override: new stem, new entry names, and the gb10
// `paged_decode_attn_fp8.cu` it is derived from is left byte-for-byte alone.
// The rationale, the geometry receipt and the determinism invariant are in
// `paged_decode_splitk_hopper.cuh`; the numbers are in
// `ATTN-DECODE-SPLITK-ATTRIBUTION.md`.
//
// What this file adds over gb10's `paged_decode_attn_splitk_fp8`:
//   * the PD_BC=4 BATCHED inner loop of the non-split kernel, so a split that
//     fills 132 SMs is not also eleven times the serial dependency chain;
//   * `pd_split_bounds`' minimum work per split, so a split count chosen for a
//     16 k context leaves the high splits empty at 300 tokens instead of
//     paying eleven CTAs and a reduce for them.
// Everything else — the dequant, the hoisted `k_scale`/`v_scale`, the operand
// order inside a warp, the smem tree merge, the `[o, m, l]` F32 workspace
// format — is the gb10 kernel's, so the two are interchangeable per split.
//
// Grid: (num_q_heads, num_splits, num_seqs)  Block: (256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "paged_decode_splitk_hopper.cuh"

// 4 FP8 E4M3 packed in a u32 -> 4 F32, UNSCALED.
//
// Identical arithmetic to `paged_decode_attn_fp8.cu`'s `unpack4_fp8_raw`,
// including the paired `fp8x2` converts and the byte order (low byte =
// element 0): the dequant is linear, so `k_scale` is hoisted into the score
// multiply and `v_scale` into the single per-warp smem write.
__device__ __forceinline__ void pd_unpack4_fp8_raw(
    unsigned int packed,
    float& v0, float& v1, float& v2, float& v3
) {
    __half2_raw h01 = __nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(packed & 0xFFFF), __NV_E4M3);
    __half2_raw h23 = __nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(packed >> 16), __NV_E4M3);
    const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&h01));
    const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&h23));
    v0 = f01.x; v1 = f01.y; v2 = f23.x; v3 = f23.y;
}

extern "C" __global__ void paged_decode_attn_splitk_fp8_hopper(
    const __nv_bfloat16* __restrict__ Q,             // [num_seqs, q_stride] BF16
    const __nv_fp8_storage_t* __restrict__ K_cache,  // [blocks, block_size, kv_heads, hd] FP8
    const __nv_fp8_storage_t* __restrict__ V_cache,
    float* __restrict__ workspace,                   // [seqs, heads, splits, hd+2] F32
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const float k_scale,
    const float v_scale,
    const unsigned int q_stride,            // query.stride(0) in elements
    const unsigned long long cache_stride,  // k_cache.stride(0) in elements
    const unsigned int sliding_window       // 0 = full attention
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int split_id = blockIdx.y;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / PD_WARP_SIZE;
    const unsigned int lane_id = tid % PD_WARP_SIZE;

    if (q_head >= num_q_heads) return;
    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;
    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    __shared__ float smem_m[PD_NUM_WARPS];
    __shared__ float smem_l[PD_NUM_WARPS];
    __shared__ float smem_o[PD_NUM_WARPS][HDIM];

    const unsigned int vec_offset = lane_id * PD_VEC;

    unsigned int kv_start = 0, kv_end = 0;
    const bool has_work =
        pd_split_bounds(seq_len, window_start, split_id, num_splits, kv_start, kv_end);

    float m_val = -1e30f;
    float l_val = 0.0f;
    float o_reg[PD_VEC];
    #pragma unroll
    for (int i = 0; i < PD_VEC; i++) o_reg[i] = 0.0f;

    if (has_work) {
        // Q is BF16 and may be a non-contiguous QKV split view.
        const unsigned int* q32 = (const unsigned int*)(Q
            + (unsigned long long)seq_idx * q_stride
            + (unsigned long long)q_head * head_dim + vec_offset);
        float q_reg[PD_VEC];
        #pragma unroll
        for (int i = 0; i < PD_VEC_U32; i++) {
            pd_unpack2_bf16(q32[i], q_reg[2 * i], q_reg[2 * i + 1]);
        }

        unsigned int my_start = 0, my_end = 0;
        pd_warp_bounds(kv_start, kv_end, warp_id, my_start, my_end);

        const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
        const unsigned int kv_head = q_head / gqa_ratio;
        // k_scale folds into the score multiply, v_scale into the one smem
        // write below — the dequant is linear (see the .cuh).
        const float score_scale = inv_sqrt_d * k_scale;
        const unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;

        unsigned int pos = my_start;
        while (pos < my_end) {
            const unsigned int logical_block = pos / block_size;
            const unsigned int block_offset = pos % block_size;
            const unsigned int remaining_in_block = block_size - block_offset;
            const unsigned int remaining_total = my_end - pos;
            const unsigned int batch_count =
                remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

            const unsigned int physical_block = (unsigned int)block_tables[
                (unsigned long long)seq_idx * max_blocks_per_seq + logical_block];
            const unsigned long long base = (unsigned long long)physical_block * cache_stride
                                          + (unsigned long long)block_offset * head_stride_kv
                                          + (unsigned long long)kv_head * head_dim;
            const __nv_fp8_storage_t* k_block_base = K_cache + base;
            const __nv_fp8_storage_t* v_block_base = V_cache + base;

            unsigned int processed = 0;
            const unsigned int aligned_count = (batch_count / PD_BC) * PD_BC;

            for (; processed < aligned_count; processed += PD_BC) {
                unsigned int k_packed[PD_BC][PD_VEC_U32_F8];
                #pragma unroll
                for (int b = 0; b < PD_BC; b++) {
                    const unsigned int* k32 = (const unsigned int*)(k_block_base
                        + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                    #pragma unroll
                    for (int i = 0; i < PD_VEC_U32_F8; i++) k_packed[b][i] = k32[i];
                }

                float scores[PD_BC];
                #pragma unroll
                for (int b = 0; b < PD_BC; b++) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < PD_VEC_U32_F8; i++) {
                        float k0, k1, k2, k3;
                        pd_unpack4_fp8_raw(k_packed[b][i], k0, k1, k2, k3);
                        dot += q_reg[4 * i] * k0 + q_reg[4 * i + 1] * k1
                             + q_reg[4 * i + 2] * k2 + q_reg[4 * i + 3] * k3;
                    }
                    #pragma unroll
                    for (int offset = PD_WARP_SIZE / 2; offset > 0; offset >>= 1)
                        dot += __shfl_xor_sync(0xffffffff, dot, offset);
                    scores[b] = dot * score_scale;
                }

                unsigned int v_packed[PD_BC][PD_VEC_U32_F8];
                #pragma unroll
                for (int b = 0; b < PD_BC; b++) {
                    const unsigned int* v32 = (const unsigned int*)(v_block_base
                        + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                    #pragma unroll
                    for (int i = 0; i < PD_VEC_U32_F8; i++) v_packed[b][i] = v32[i];
                }

                float m_new = m_val;
                #pragma unroll
                for (int b = 0; b < PD_BC; b++) m_new = fmaxf(m_new, scores[b]);

                const float exp_old = __expf(m_val - m_new);
                #pragma unroll
                for (int i = 0; i < PD_VEC; i++) o_reg[i] *= exp_old;
                l_val *= exp_old;

                float exp_factors[PD_BC];
                #pragma unroll
                for (int b = 0; b < PD_BC; b++) {
                    exp_factors[b] = __expf(scores[b] - m_new);
                    l_val += exp_factors[b];
                }
                m_val = m_new;

                #pragma unroll
                for (int b = 0; b < PD_BC; b++) {
                    const float ef = exp_factors[b];
                    #pragma unroll
                    for (int i = 0; i < PD_VEC_U32_F8; i++) {
                        float v0, v1, v2, v3;
                        pd_unpack4_fp8_raw(v_packed[b][i], v0, v1, v2, v3);
                        o_reg[4 * i]     += ef * v0;
                        o_reg[4 * i + 1] += ef * v1;
                        o_reg[4 * i + 2] += ef * v2;
                        o_reg[4 * i + 3] += ef * v3;
                    }
                }
            }

            // Remainder: the block tail, one position at a time.
            for (; processed < batch_count; processed++) {
                const unsigned int* k32 = (const unsigned int*)(k_block_base
                    + (unsigned long long)processed * head_stride_kv + vec_offset);
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < PD_VEC_U32_F8; i++) {
                    float k0, k1, k2, k3;
                    pd_unpack4_fp8_raw(k32[i], k0, k1, k2, k3);
                    dot += q_reg[4 * i] * k0 + q_reg[4 * i + 1] * k1
                         + q_reg[4 * i + 2] * k2 + q_reg[4 * i + 3] * k3;
                }
                #pragma unroll
                for (int offset = PD_WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);

                const float score = dot * score_scale;
                const float m_new = fmaxf(m_val, score);
                const float exp_old = __expf(m_val - m_new);
                const float exp_new = __expf(score - m_new);
                l_val = l_val * exp_old + exp_new;

                const unsigned int* v32 = (const unsigned int*)(v_block_base
                    + (unsigned long long)processed * head_stride_kv + vec_offset);
                #pragma unroll
                for (int i = 0; i < PD_VEC_U32_F8; i++) {
                    float v0, v1, v2, v3;
                    pd_unpack4_fp8_raw(v32[i], v0, v1, v2, v3);
                    o_reg[4 * i]     = o_reg[4 * i]     * exp_old + exp_new * v0;
                    o_reg[4 * i + 1] = o_reg[4 * i + 1] * exp_old + exp_new * v1;
                    o_reg[4 * i + 2] = o_reg[4 * i + 2] * exp_old + exp_new * v2;
                    o_reg[4 * i + 3] = o_reg[4 * i + 3] * exp_old + exp_new * v3;
                }
                m_val = m_new;
            }

            pos += batch_count;
        }
    }

    // Every CTA emits, including an empty split: it writes l = 0, which is what
    // tells the reduce to skip it. Returning early instead would leave stale
    // workspace bytes for the reduce to merge.
    pd_emit_partial(m_val, l_val, o_reg, v_scale,
                    smem_m, smem_l, &smem_o[0][0],
                    warp_id, lane_id, vec_offset, workspace,
                    seq_idx, q_head, split_id, num_q_heads, num_splits, head_dim);
}

// Reduce the FP8 twin's partials. Grid: (num_q_heads, num_seqs, 1) Block: (32,1,1)
extern "C" __global__ void paged_decode_attn_reduce_fp8_hopper(
    const float* __restrict__ workspace,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ seq_lens,
    const unsigned int num_q_heads,
    const unsigned int head_dim,
    const unsigned int num_splits
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    if (q_head >= num_q_heads) return;
    if (seq_lens[seq_idx] == 0) return;
    pd_reduce_body(workspace, O, seq_idx, q_head, threadIdx.x,
                   num_q_heads, head_dim, num_splits);
}
