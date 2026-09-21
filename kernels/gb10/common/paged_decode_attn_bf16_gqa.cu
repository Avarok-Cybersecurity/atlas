// SPDX-License-Identifier: AGPL-3.0-only

// GQA-PACKED BF16-KV paged decode attention — the `paged_decode_attn` twin
// that reads each K and V row ONCE for the whole query group.
//
// The defect, the fix, the split-K restriction and the bit-identity argument
// are all the FP8 twin's, verbatim: see
// `kernels/gb10/common/paged_decode_attn_fp8_gqa.cu`. This file exists because
// the BF16 cache is a different element type (2 bytes, `unpack2_pd`, no
// per-tensor scale) and a different block-stride convention (`page_stride`
// derived in-kernel rather than a host-passed `cache_stride`), so the two
// cannot share a body without a wrapper that would defeat the register
// residency the whole change is for.
//
// On GB10 this arm matters because `bf16_splitk_pair` is Hopper-only: the BF16
// KV layers — Qwen3.8-27B's four `--kv-high-precision-layers auto` layers —
// take the single-CTA kernel at EVERY batch size on this target, so they are
// the arm with no split-K alternative at all.
//
// Grid: (num_kv_heads, num_seqs, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>

#define WARP_SIZE 32
// Fixed, not `#ifndef`-overridable — see the FP8 twin.
#define PD_HDIM 256
#define VEC_BF16 (PD_HDIM / WARP_SIZE)
#define VEC_U32 (PD_HDIM / (WARP_SIZE * 2))
#define NUM_WARPS 8
#define BC 4

// Query heads packed into one CTA. Must equal
// `avarok_kernels::attn_splitk::DECODE_GQA_PACK_WIDTH`; the Rust test
// `cuda_sources_declare_the_pack_width_rust_dispatches_on` asserts it.
#define PD_GQA 6

// ★ Copy of `paged_decode_attn.cu`'s helper, byte-for-byte. Bit-identity with
// the unpacked kernel requires this body to stay identical, so
// `attn_splitk_tests::gqa_kernels_copy_the_unpack_helpers_verbatim` compares
// it textually against the original.
__device__ __forceinline__ void unpack2_pd(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

extern "C" __global__ void __launch_bounds__(NUM_WARPS* WARP_SIZE, 1)
    paged_decode_attn_bf16_gqa(
        const __nv_bfloat16* __restrict__ Q,        // [num_seqs, num_q_heads, head_dim]
        const __nv_bfloat16* __restrict__ K_cache,  // [num_blocks, block_size, num_kv_heads, head_dim]
        const __nv_bfloat16* __restrict__ V_cache,  // [num_blocks, block_size, num_kv_heads, head_dim]
        __nv_bfloat16* __restrict__ O,              // [num_seqs, num_q_heads, head_dim]
        const int* __restrict__ block_tables,
        const int* __restrict__ seq_lens,
        const unsigned int max_blocks_per_seq,
        const unsigned int num_q_heads,
        const unsigned int num_kv_heads,
        const unsigned int head_dim,
        const unsigned int block_size,
        const float inv_sqrt_d,
        const unsigned int q_stride,
        const unsigned int sliding_window
    ) {
    const unsigned int kv_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (kv_head >= num_kv_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;
    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    const unsigned int q_head_base = kv_head * PD_GQA;
    const unsigned int vec_offset = lane_id * VEC_BF16;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    float q_reg[PD_GQA][VEC_BF16];
    #pragma unroll
    for (int h = 0; h < PD_GQA; h++) {
        const unsigned int* q32 = (const unsigned int*)(Q
            + (unsigned long long)seq_idx * q_stride
            + (unsigned long long)(q_head_base + h) * head_dim + vec_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            unpack2_pd(q32[i], q_reg[h][2 * i], q_reg[h][2 * i + 1]);
        }
    }

    const unsigned int attended = seq_len - window_start;
    unsigned int chunk_size = (attended + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    float m_acc[PD_GQA];
    float l_acc[PD_GQA];
    float o_reg[PD_GQA][VEC_BF16];
    #pragma unroll
    for (int h = 0; h < PD_GQA; h++) {
        m_acc[h] = -1e30f;
        l_acc[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) o_reg[h][i] = 0.0f;
    }

    unsigned int pos = my_start;
    while (pos < my_end) {
        unsigned int logical_block = pos / block_size;
        unsigned int block_offset = pos % block_size;
        unsigned int remaining_in_block = block_size - block_offset;
        unsigned int remaining_total = my_end - pos;
        unsigned int batch_count =
            remaining_in_block < remaining_total ? remaining_in_block : remaining_total;

        unsigned int physical_block = (unsigned int)my_block_table[logical_block];
        unsigned long long page_stride = (unsigned long long)block_size * num_kv_heads * head_dim;
        unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;
        const __nv_bfloat16* k_block_base = K_cache
            + (unsigned long long)physical_block * page_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim;
        const __nv_bfloat16* v_block_base = V_cache
            + (unsigned long long)physical_block * page_stride
            + (unsigned long long)block_offset * head_stride_kv
            + (unsigned long long)kv_head * head_dim;

        unsigned int processed = 0;
        unsigned int aligned_count = (batch_count / BC) * BC;

        for (; processed < aligned_count; processed += BC) {
            // ONE load of each K and V row for the whole query group.
            unsigned int k_packed[BC][VEC_U32];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* k32 = (const unsigned int*)(k_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) k_packed[b][i] = k32[i];
            }
            unsigned int v_packed[BC][VEC_U32];
            #pragma unroll
            for (int b = 0; b < BC; b++) {
                const unsigned int* v32 = (const unsigned int*)(v_block_base
                    + (unsigned long long)(processed + b) * head_stride_kv + vec_offset);
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) v_packed[b][i] = v32[i];
            }

            #pragma unroll
            for (int h = 0; h < PD_GQA; h++) {
                float scores[BC];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < VEC_U32; i++) {
                        float k0, k1;
                        unpack2_pd(k_packed[b][i], k0, k1);
                        dot += q_reg[h][2 * i] * k0 + q_reg[h][2 * i + 1] * k1;
                    }
                    #pragma unroll
                    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                        dot += __shfl_xor_sync(0xffffffff, dot, offset);
                    scores[b] = dot * inv_sqrt_d;
                }

                float m_new = m_acc[h];
                #pragma unroll
                for (int b = 0; b < BC; b++) m_new = fmaxf(m_new, scores[b]);

                float exp_old = __expf(m_acc[h] - m_new);
                #pragma unroll
                for (int i = 0; i < VEC_BF16; i++) o_reg[h][i] *= exp_old;
                l_acc[h] *= exp_old;

                float exp_factors[BC];
                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    exp_factors[b] = __expf(scores[b] - m_new);
                    l_acc[h] += exp_factors[b];
                }
                m_acc[h] = m_new;

                #pragma unroll
                for (int b = 0; b < BC; b++) {
                    float ef = exp_factors[b];
                    #pragma unroll
                    for (int i = 0; i < VEC_U32; i++) {
                        float v0, v1;
                        unpack2_pd(v_packed[b][i], v0, v1);
                        o_reg[h][2 * i]     += ef * v0;
                        o_reg[h][2 * i + 1] += ef * v1;
                    }
                }
            }
        }

        for (; processed < batch_count; processed++) {
            const unsigned int* k32 = (const unsigned int*)(k_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset);
            unsigned int k_one[VEC_U32];
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) k_one[i] = k32[i];
            const unsigned int* v32 = (const unsigned int*)(v_block_base
                + (unsigned long long)processed * head_stride_kv + vec_offset);
            unsigned int v_one[VEC_U32];
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) v_one[i] = v32[i];

            #pragma unroll
            for (int h = 0; h < PD_GQA; h++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    float k0, k1;
                    unpack2_pd(k_one[i], k0, k1);
                    dot += q_reg[h][2 * i] * k0 + q_reg[h][2 * i + 1] * k1;
                }
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);

                float score = dot * inv_sqrt_d;
                float m_new = fmaxf(m_acc[h], score);
                float exp_old = __expf(m_acc[h] - m_new);
                float exp_new = __expf(score - m_new);
                l_acc[h] = l_acc[h] * exp_old + exp_new;

                #pragma unroll
                for (int i = 0; i < VEC_U32; i++) {
                    float v0, v1;
                    unpack2_pd(v_one[i], v0, v1);
                    o_reg[h][2 * i]     = o_reg[h][2 * i]     * exp_old + exp_new * v0;
                    o_reg[h][2 * i + 1] = o_reg[h][2 * i + 1] * exp_old + exp_new * v1;
                }
                m_acc[h] = m_new;
            }
        }

        pos += batch_count;
    }

    // Per-head epilogue through ONE shared buffer — see the FP8 twin.
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][PD_HDIM];

    #pragma unroll
    for (int h = 0; h < PD_GQA; h++) {
        __syncthreads();

        if (lane_id == 0) {
            smem_m[warp_id] = m_acc[h];
            smem_l[warp_id] = l_acc[h];
        }
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) {
            smem_o[warp_id][vec_offset + i] = o_reg[h][i];
        }
        __syncthreads();

        #pragma unroll
        for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
            if (warp_id < (unsigned int)stride) {
                unsigned int other = warp_id + stride;
                float lw = smem_l[other];
                if (lw > 0.0f) {
                    float mw = smem_m[other];
                    float my_m = smem_m[warp_id];
                    float my_l = smem_l[warp_id];
                    float m_new = fmaxf(my_m, mw);
                    float scale_me = __expf(my_m - m_new);
                    float scale_w = __expf(mw - m_new);
                    smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                    smem_m[warp_id] = m_new;
                    #pragma unroll
                    for (int i = 0; i < VEC_BF16; i++) {
                        smem_o[warp_id][vec_offset + i] =
                            smem_o[warp_id][vec_offset + i] * scale_me +
                            smem_o[other][vec_offset + i] * scale_w;
                    }
                }
            }
            __syncthreads();
        }

        if (warp_id == 0) {
            float final_l = smem_l[0];
            float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
            unsigned int* o32 = (unsigned int*)(O
                + (unsigned long long)seq_idx * num_q_heads * head_dim
                + (unsigned long long)(q_head_base + h) * head_dim + vec_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                float v0 = smem_o[0][vec_offset + 2 * i]     * inv_l;
                float v1 = smem_o[0][vec_offset + 2 * i + 1] * inv_l;
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
                o32[i] = lo | (hi << 16);
            }
        }
    }
}
