// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3-Flash — DSA selected-index MLA paged decode, FP8 KV.
//
// Scoped to LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3. Lives in the GLM target, so
// DeepSeek-V4-Flash is untouched by construction.
//
// ── Why this kernel exists ──────────────────────────────────────────────────────────────
//
// Two jobs that turned out to be one:
//
// 1. NoPE. GLM-5.3 has qk_rope_head_dim == 0: the latent IS the whole cache token, 512
//    dims, no rope tail. Neither common/ MLA decode can read that. mla_paged_decode.cu
//    declares kv_cache_dim and never uses it (every stride comes from #define ROPE_DIM
//    64, so its token stride is 288 B where GLM's is 256 B). mla_paged_decode_fp8.cu does
//    use kv_cache_dim for the stride, but then unconditionally overwrites dims 448-511 of
//    k_vals/v_vals with "rope" read from the NEXT token's bytes. Both fail silently: no
//    crash, no shape error, no launch failure. There is no rope arm here at all.
//
// 2. DSA. Selection feeds a per-row GATHER, not a dense mask. HF materialises a
//    [B, Q, kv_len] mask and says so in its own docstring -- "cannot be mapped to FA
//    without a custom kernel that can select on a per indices bases per row" -- with
//    _supports_flash_attn = False. vLLM ships that kernel (FlashMLA sparse / FlashInfer
//    paged MLA) and gathers. The mask is pure set membership (scatter_add(...).ne(0):
//    duplicates collapse, no additive weighting), so a gather is EXACTLY equivalent, not
//    an approximation. A dense-mask kernel would also stage an [S] score row in shared
//    memory, capping context at 12,288 keys regardless of how sparse the selection is.
//
// ── What changed vs mla_paged_decode_fp8 ────────────────────────────────────────────────
//
// The MATH is unchanged -- same per-warp online softmax, same cross-warp merge. Only the
// ITERATION changed: warps split the SELECTION ROW instead of [0, seq_len), and each
// selected token id is gathered through the block table individually. Contiguity is gone,
// so the BC=4 in-block batching is gone with it; one token at a time. The dense kernel
// already issued one warp reduction per token, so this costs the batched max/exp
// amortisation and nothing else. Deliberately NOT optimised.
//
// ── Contract for `sel_indices` (identical in HF and vLLM; do not weaken it) ─────────────
//
//   * -1 is the invalid sentinel, and the row is FULLY written -- never left
//     uninitialised. vLLM's day-0 GLM bug was a torch.empty top-k buffer whose tail
//     became "token indices".
//   * Order is not meaningful.
//   * Duplicates would collapse under the reference mask. They cannot occur here: pools
//     are disjoint blocks of `index_kpool` tokens and the appended tail is the
//     in-progress group beyond the last complete pool. No dedup pass -- but if the
//     selector ever emits a duplicate, this kernel double-counts it. That invariant is
//     pinned host-side.
//   * A row that is entirely -1 must contribute nothing. It does: every warp ends with
//     l == 0, the merge skips warps with l <= 0, and the final normalisation writes zeros
//     when l == 0. That is the (out, lse) = (0, -inf) identity vLLM relies on for its
//     cross-rank merge.
//
// ── BF16 control (later, not now) ───────────────────────────────────────────────────────
//
// Everything above the load is dtype-agnostic. A BF16-KV twin replaces exactly the two
// `load_kv_*` calls and drops k_scale/v_scale; the iteration, softmax and merge are
// copied verbatim. Kept as a separate entry point rather than a runtime branch, matching
// how the common/ NVFP4 and FP8 twins are already split.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#define VEC_BF16 16   // 512 / 32 lanes = 16 elements per lane
#define VEC_U32  8    // 16 bf16 = 8 uint32
#define NUM_WARPS 8

// 🔴 Heads retired per block. Every head of a given row gathers the SAME selected tokens
// — `sel_width` of them, 2,051 at production settings — so a one-head-per-block grid read
// each token's 512-byte latent `num_q_heads` times over. This block loads it once and
// dots it against `HEADS_PER_BLOCK` queries.
//
// 🪤 Bounded by REGISTERS, not by shared memory: each head carries its own `q_reg` and
// `o_reg` (VEC_BF16 floats apiece) plus running `m`/`l`, so per-lane state grows as
// 2*VEC_BF16*HG. At HG=2 that is ~68 floats/lane, which keeps the same occupancy the
// one-head version had. The cross-warp merge is run once per head reusing ONE
// `smem_o[NUM_WARPS][512]` buffer, so shared memory does not scale with HG at all.
#define HEADS_PER_BLOCK 2


// GLM-5.3's latent width. Compile-time because the per-lane register tiling above is
// derived from it (32 lanes * VEC_BF16 == 512 exactly). The host refuses any checkpoint
// whose kv_lora_rank differs -- Glm5NextDsaConfig::validate, KERNEL_KV_LORA_DIM.
#define GLM_KV_LORA_DIM 512

#define DSA_INVALID (-1)

__device__ __forceinline__ float fp8e4m3_to_f32(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// Load this lane's VEC_BF16 slice of one cache token. NoPE: pure latent, no rope arm,
// no overwrite of the tail dims. This is the ONLY dtype-dependent step.
__device__ __forceinline__ void load_kv_fp8(
    const unsigned char* __restrict__ token_base,
    unsigned int lane_offset,
    float scale,
    float* __restrict__ out
) {
    const unsigned char* p = token_base + lane_offset;
    #pragma unroll
    for (int i = 0; i < VEC_BF16; i++)
        out[i] = fp8e4m3_to_f32((__nv_fp8_storage_t)p[i]) * scale;
}

extern "C" __global__ void glm5next_dsa_mla_decode_fp8(
    const __nv_bfloat16* __restrict__ Q,           // [num_q_heads * kv_lora_dim] bf16
    const unsigned char* __restrict__ K_cache,     // FP8 latent cache
    const unsigned char* __restrict__ V_cache,     // same buffer as K in absorbed NoPE MLA
    __nv_bfloat16* __restrict__ O,                 // [num_q_heads * kv_lora_dim] bf16
    const int* __restrict__ block_tables,          // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,              // [num_seqs]
    const int* __restrict__ sel_indices,           // [num_seqs, sel_width] i32, -1 = unused
    const unsigned int sel_width,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int kv_lora_dim,                // 512; NoPE => this IS the token width
    const unsigned int block_size,
    const float inv_sqrt_d,
    const float k_scale,
    const float v_scale,
    const unsigned long long cache_stride_bytes
) {
    const unsigned int q_head_base = blockIdx.x * HEADS_PER_BLOCK;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    if (q_head_base >= num_q_heads) return;

    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    const unsigned int lane_offset = lane_id * VEC_BF16;

    // 🪤 NoPE: no rope term. This is the line both common/ kernels get wrong for GLM.
    const unsigned int token_stride = num_kv_heads * kv_lora_dim;

    const int* my_block_table = block_tables + (size_t)seq_idx * max_blocks_per_seq;
    const int* my_sel         = sel_indices  + (size_t)seq_idx * sel_width;

    // Q and O are `[num_seqs, num_q_heads, kv_lora_dim]`. A K-row speculative verify runs
    // all K rows in ONE launch (gridDim.y == K) so the 32 head-blocks of a single row do
    // not leave most of the GPU idle for three serial launches. At num_seqs == 1 this term
    // is 0 and the decode path is untouched.
    const unsigned long long row_off = (unsigned long long)seq_idx * num_q_heads * kv_lora_dim;

    // Q for each head of this group, this lane's 16 dims. `ng` is how many of
    // HEADS_PER_BLOCK actually exist — the last group is short when num_q_heads is not a
    // multiple, and a missing head must not be scored or written.
    const unsigned int ng = (q_head_base + HEADS_PER_BLOCK <= num_q_heads)
                          ? HEADS_PER_BLOCK
                          : (num_q_heads - q_head_base);
    float q_reg[HEADS_PER_BLOCK][VEC_BF16];
    #pragma unroll
    for (int g = 0; g < HEADS_PER_BLOCK; g++) {
        if ((unsigned int)g >= ng) break;
        const unsigned int* q32 =
            (const unsigned int*)(Q + row_off
                                  + (unsigned long long)(q_head_base + g) * kv_lora_dim
                                  + lane_offset);
        #pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            unsigned int v = q32[i];
            q_reg[g][2*i]     = __bfloat162float(__ushort_as_bfloat16((unsigned short)(v & 0xFFFF)));
            q_reg[g][2*i + 1] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(v >> 16)));
        }
    }

    // Warps split the SELECTION ROW, not the sequence. A warp whose whole slice is -1
    // simply ends with l == 0 and is dropped by the merge.
    const unsigned int chunk = (sel_width + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int j     = warp_id * chunk;
    unsigned int j_end = j + chunk;
    if (j_end > sel_width) j_end = sel_width;

    float m[HEADS_PER_BLOCK];
    float l[HEADS_PER_BLOCK];
    float o_reg[HEADS_PER_BLOCK][VEC_BF16];
    #pragma unroll
    for (int g = 0; g < HEADS_PER_BLOCK; g++) {
        m[g] = -1e30f;
        l[g] = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++) o_reg[g][i] = 0.0f;
    }

    for (; j < j_end; j++) {
        const int t = my_sel[j];
        // Sentinel and range guard. Out-of-range is dropped rather than clamped: clamping
        // would silently attend to a real-but-wrong token.
        if (t == DSA_INVALID || t < 0 || (unsigned int)t >= seq_len) continue;

        const unsigned int logical_block = (unsigned int)t / block_size;
        const unsigned int p             = (unsigned int)t % block_size;
        const unsigned int physical_block = (unsigned int)my_block_table[logical_block];

        const unsigned char* k_tok =
            K_cache + (unsigned long long)physical_block * cache_stride_bytes + p * token_stride;
        const unsigned char* v_tok =
            V_cache + (unsigned long long)physical_block * cache_stride_bytes + p * token_stride;

        float k_tmp[VEC_BF16];
        load_kv_fp8(k_tok, lane_offset, k_scale, k_tmp);

        // The latent is loaded ONCE above; every head of the group now consumes it from
        // registers. Each head keeps its own online-softmax state, so the arithmetic per
        // head is exactly what the one-head-per-block version did.
        float dot_g[HEADS_PER_BLOCK];
        #pragma unroll
        for (int g = 0; g < HEADS_PER_BLOCK; g++) {
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                if (lane_offset + i < kv_lora_dim) dot += q_reg[g][i] * k_tmp[i];
            #pragma unroll
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, off);
            dot_g[g] = dot;
        }

        // 🔴 ABSORBED MLA: K AND V ARE THE SAME LATENT. `attend.rs` is handed one pool for
        // both (`v_cache: pool, // absorbed NoPE MLA: K and V are the same latent`) and one
        // scale for both (`k_scale: self.kv_scale, v_scale: self.kv_scale`), so `v_tok`
        // resolves to the byte-identical address `k_tok` did and `load_kv_fp8` would decode
        // the byte-identical values a second time. `__restrict__` on both pointers is a
        // PROMISE they do not alias, so the compiler is not permitted to notice they do and
        // must emit the second load — the annotation that usually helps is what kept this
        // alive.
        //
        // This kernel is the largest remaining prefill leaf and is bound by exactly this
        // traffic: ~17 GB per launch, of which half was the same bytes twice.
        //
        // 🪤 BIT-IDENTICAL, not approximately equal: same address, same lane offset, same
        // scale, so `load_kv_fp8` is a pure function of inputs that are all equal. The guard
        // is a real runtime check rather than an assumption, so a future caller that passes
        // genuinely distinct K/V or per-tensor scales silently keeps the correct two-load
        // path. Both operands are kernel-uniform, so the branch costs no divergence.
        const bool same_kv = (K_cache == V_cache) && (k_scale == v_scale);
        float v_tmp[VEC_BF16];
        if (same_kv) {
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++) v_tmp[i] = k_tmp[i];
        } else {
            load_kv_fp8(v_tok, lane_offset, v_scale, v_tmp);
        }

        #pragma unroll
        for (int g = 0; g < HEADS_PER_BLOCK; g++) {
            const float score   = dot_g[g] * inv_sqrt_d;
            const float m_new   = fmaxf(m[g], score);
            const float exp_old = __expf(m[g] - m_new);
            const float exp_new = __expf(score - m_new);
            l[g] = l[g] * exp_old + exp_new;
            #pragma unroll
            for (int i = 0; i < VEC_BF16; i++)
                o_reg[g][i] = o_reg[g][i] * exp_old + exp_new * v_tmp[i];
            m[g] = m_new;
        }
    }

    // ── cross-warp merge (verbatim from mla_paged_decode_fp8; no sinks on GLM) ──
    //
    // 🪤 ONE buffer, reused per head, rather than `[HEADS_PER_BLOCK]` of them. `smem_o` is
    // NUM_WARPS x 512 floats = 16 KB on its own; scaling it with the group would blow the
    // 48 KiB ceiling at HG=4 and cap the grouping on shared memory instead of registers.
    // The merge is a fixed cost per head, tiny beside the `sel_width`-long token loop, so
    // serialising it costs far less than the loads the grouping saves.
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][GLM_KV_LORA_DIM];

    for (unsigned int g = 0; g < ng; g++) {
        // Before overwriting: the previous head's merge still had warp 0 READING
        // `smem_o[0]` when the other warps reached here.
        __syncthreads();
        if (lane_id == 0) {
            smem_m[warp_id] = m[g];
            smem_l[warp_id] = l[g];
        }
        #pragma unroll
        for (int i = 0; i < VEC_BF16; i++)
            if (lane_offset + i < GLM_KV_LORA_DIM)
                smem_o[warp_id][lane_offset + i] = o_reg[g][i];
        __syncthreads();

        #pragma unroll
        for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
            if (warp_id < (unsigned int)stride) {
                const unsigned int other = warp_id + stride;
                const float lw = smem_l[other];
                if (lw > 0.0f) {
                    const float mw     = smem_m[other];
                    const float my_m   = smem_m[warp_id];
                    const float my_l   = smem_l[warp_id];
                    const float m_new  = fmaxf(my_m, mw);
                    const float sc_me  = __expf(my_m - m_new);
                    const float sc_w   = __expf(mw - m_new);
                    smem_l[warp_id] = my_l * sc_me + lw * sc_w;
                    smem_m[warp_id] = m_new;
                    #pragma unroll
                    for (int i = 0; i < GLM_KV_LORA_DIM; i++)
                        smem_o[warp_id][i] = smem_o[warp_id][i] * sc_me + smem_o[other][i] * sc_w;
                }
            }
            __syncthreads();
        }

        if (warp_id == 0) {
            const float final_l = smem_l[0];
            // l == 0 means this row selected nothing at all: write zeros, which is the
            // identity element of a cross-rank LSE merge. Never divide by zero.
            const float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
            unsigned int* o32 =
                (unsigned int*)(O + row_off
                                + (unsigned long long)(q_head_base + g) * kv_lora_dim
                                + lane_offset);
            #pragma unroll
            for (int i = 0; i < VEC_U32; i++) {
                const float v0 = smem_o[0][lane_offset + 2*i]     * inv_l;
                const float v1 = smem_o[0][lane_offset + 2*i + 1] * inv_l;
                const unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
                const unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
                o32[i] = lo | (hi << 16);
            }
        }
    }
}
