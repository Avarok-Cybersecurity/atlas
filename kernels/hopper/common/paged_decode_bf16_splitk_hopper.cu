// SPDX-License-Identifier: AGPL-3.0-only

// Hopper paged-decode SPLIT-K, BF16 KV cache (#928).
//
// The half of this lever that is a NEW capability rather than a re-tuning:
// `run_paged_decode.rs` carried an explicit
//   // BF16 paged decode — no Split-K (not implemented for BF16 yet)
// branch, so the `--kv-high-precision-layers auto` layers had no split-K
// kernel to select even when the policy asked for one. On Qwen3.8-27B those
// are 4 of the 16 full-attention layers and 1 013 us of a 16.69 ms C=1 decode
// step on an H100 (nsys, round 13 cell T1N: `grid=(24,1,1)`, 253.36 us/launch,
// 79.4 MB, 78.4 GB/s = 2.34% of HBM).
//
// Structure mirrors `paged_decode_fp8_splitk_hopper.cu` exactly — same split
// partition, same PD_BC=4 batched loop, same workspace format — with the FP8
// dequant replaced by the BF16 unpack and the k/v scales dropped. The cache
// page stride is computed here rather than passed, matching gb10's
// `paged_decode_attn`, whose launcher has no `cache_stride` argument.
//
// ADDITION (new stem, new entry names): gb10's `paged_decode_attn.cu` is not
// edited and not forked. See `paged_decode_splitk_hopper.cuh` for the
// placement rule and the determinism invariant.
//
// Grid: (num_q_heads, num_splits, num_seqs)  Block: (256, 1, 1)

#include <cuda_bf16.h>

#include "paged_decode_splitk_hopper.cuh"

extern "C" __global__ void paged_decode_attn_splitk_bf16_hopper(
    const __nv_bfloat16* __restrict__ Q,        // [num_seqs, q_stride] BF16
    const __nv_bfloat16* __restrict__ K_cache,  // [blocks, block_size, kv_heads, hd] BF16
    const __nv_bfloat16* __restrict__ V_cache,
    float* __restrict__ workspace,              // [seqs, heads, splits, hd+2] F32
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride,          // query.stride(0) in elements
    const unsigned int sliding_window     // 0 = full attention
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
        const unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;
        const unsigned long long page_stride = (unsigned long long)block_size * head_stride_kv;

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
            const unsigned long long base = (unsigned long long)physical_block * page_stride
                                          + (unsigned long long)block_offset * head_stride_kv
                                          + (unsigned long long)kv_head * head_dim;
            const __nv_bfloat16* k_block_base = K_cache + base;
            const __nv_bfloat16* v_block_base = V_cache + base;

            unsigned int processed = 0;
            const unsigned int aligned_count = (batch_count / PD_BC) * PD_BC;

            for (; processed < aligned_count; processed += PD_BC) {
                unsigned int k_packed[PD_BC][PD_VEC_U32];
                #pragma unroll
                for (int b = 0; b < PD_BC; b++) {
                    const unsigned int* k32 = (const unsigned int*)(k_block_base
                        + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                    #pragma unroll
                    for (int i = 0; i < PD_VEC_U32; i++) k_packed[b][i] = k32[i];
                }

                float scores[PD_BC];
                #pragma unroll
                for (int b = 0; b < PD_BC; b++) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < PD_VEC_U32; i++) {
                        float k0, k1;
                        pd_unpack2_bf16(k_packed[b][i], k0, k1);
                        dot += q_reg[2 * i] * k0 + q_reg[2 * i + 1] * k1;
                    }
                    #pragma unroll
                    for (int offset = PD_WARP_SIZE / 2; offset > 0; offset >>= 1)
                        dot += __shfl_xor_sync(0xffffffff, dot, offset);
                    scores[b] = dot * inv_sqrt_d;
                }

                unsigned int v_packed[PD_BC][PD_VEC_U32];
                #pragma unroll
                for (int b = 0; b < PD_BC; b++) {
                    const unsigned int* v32 = (const unsigned int*)(v_block_base
                        + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                    #pragma unroll
                    for (int i = 0; i < PD_VEC_U32; i++) v_packed[b][i] = v32[i];
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
                    for (int i = 0; i < PD_VEC_U32; i++) {
                        float v0, v1;
                        pd_unpack2_bf16(v_packed[b][i], v0, v1);
                        o_reg[2 * i]     += ef * v0;
                        o_reg[2 * i + 1] += ef * v1;
                    }
                }
            }

            for (; processed < batch_count; processed++) {
                const unsigned int* k32 = (const unsigned int*)(k_block_base
                    + (unsigned long long)processed * head_stride_kv + vec_offset);
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < PD_VEC_U32; i++) {
                    float k0, k1;
                    pd_unpack2_bf16(k32[i], k0, k1);
                    dot += q_reg[2 * i] * k0 + q_reg[2 * i + 1] * k1;
                }
                #pragma unroll
                for (int offset = PD_WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);

                const float score = dot * inv_sqrt_d;
                const float m_new = fmaxf(m_val, score);
                const float exp_old = __expf(m_val - m_new);
                const float exp_new = __expf(score - m_new);
                l_val = l_val * exp_old + exp_new;

                const unsigned int* v32 = (const unsigned int*)(v_block_base
                    + (unsigned long long)processed * head_stride_kv + vec_offset);
                #pragma unroll
                for (int i = 0; i < PD_VEC_U32; i++) {
                    float v0, v1;
                    pd_unpack2_bf16(v32[i], v0, v1);
                    o_reg[2 * i]     = o_reg[2 * i]     * exp_old + exp_new * v0;
                    o_reg[2 * i + 1] = o_reg[2 * i + 1] * exp_old + exp_new * v1;
                }
                m_val = m_new;
            }

            pos += batch_count;
        }
    }

    // 1.0f: there is no v_scale on a BF16 cache. Empty splits emit l = 0 so the
    // reduce skips them — see the FP8 twin's note.
    pd_emit_partial(m_val, l_val, o_reg, 1.0f,
                    smem_m, smem_l, &smem_o[0][0],
                    warp_id, lane_id, vec_offset, workspace,
                    seq_idx, q_head, split_id, num_q_heads, num_splits, head_dim);
}

// Reduce the BF16 twin's partials. Grid: (num_q_heads, num_seqs, 1) Block: (32,1,1)
//
// The workspace format is quantisation-agnostic, so this body is the FP8
// twin's; it carries its own entry point rather than sharing one because a
// module is a file and the two kernels are dispatched independently.
extern "C" __global__ void paged_decode_attn_reduce_bf16_hopper(
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
