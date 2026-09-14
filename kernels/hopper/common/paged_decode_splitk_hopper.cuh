// SPDX-License-Identifier: AGPL-3.0-only

// Shared machinery for the Hopper paged-decode SPLIT-K twins (#928).
//
// WHY THESE KERNELS EXIST. nsys on 1xH100 80GB HBM3, Qwen/Qwen3.8-27B-FP8,
// round 13 cell T1N (`ATTN-DECODE-SPLITK-ATTRIBUTION.md`): at C=1 the shipped
// `paged_decode_attn_fp8` launches `grid=(24,1,1)` — 24 CTAs on 132 SMs — and
// spends 231.51 us/launch moving 9.93 MB of KV, i.e. 42.9 GB/s = 1.28% of
// HBM. With the BF16-KV sibling the pair is 3.79 ms of a 16.69 ms C=1 decode
// step (22.7%). The kernel is not bandwidth-bound and not FLOP-bound; it is
// 108 idle SMs.
//
// The gb10 tree already carries an FP8 split-K pair
// (`common/paged_decode_attn_fp8.cu`), but its inner loop is the SCALAR
// remainder path: one KV position per iteration with a serial
// load -> dot -> 5 shuffles -> exp -> load -> accumulate dependency chain. On
// a 132-SM part the split count that fills the device also multiplies that
// chain, so the twins here restore the BC=4 batched path the NON-split kernel
// uses, and add the BF16-KV split-K that gb10 has never had
// (`run_paged_decode.rs` took an explicit "no Split-K (not implemented for
// BF16 yet)" branch for the 4 `--kv-high-precision-layers auto` layers).
//
// PLACEMENT. Maintainer rule, 2026-09-11 (tbraun96): a Hopper-tuned kernel is
// a REAL FILE under `kernels/hopper/common/`, declared in that target's
// `[kernels] overrides`; `kernels/gb10/common/*.cu` is not edited when
// iterating on Hopper. These are ADDITIONS (new stems, new entry names), not
// overrides, for the same reason `gdn_fwd_o_hopper.cu` is: a whole-file
// override of `paged_decode_attn_fp8.cu` would fork the NON-split
// `paged_decode_attn_fp8` entry that five other targets compile and that six
// KV dtypes route through, to change two entry points beside it.
//
// ── THE DETERMINISM INVARIANT, which these kernels must not break ──────────
//
// `num_splits` is a launch parameter derived from CONFIGURATION ONLY
// (`atlas_kernels::attn_splitk`): target SM count, q-head count, the pinned
// max decode batch. It is never a function of the runtime co-batched count.
// Inside the kernel the partition of a sequence's KV range is a pure function
// of THAT SEQUENCE'S OWN `seq_len` and `num_splits` (see `pd_split_bounds`),
// and the reduce merges partials in split-index order, skipping empties. So a
// sequence decoded alone and the same sequence decoded beside fifteen others
// traverses an identical reduction tree and produces identical bytes — which
// is the property `qwen3_attention::split_ref_seqs` was introduced to protect
// (`tasks/determinism_investigation.md`) and the reason none of the code below
// may read `num_seqs` for anything but addressing.
//
// The online-softmax split-merge IS non-associative, so output is NOT
// bit-identical ACROSS different `num_splits`; that is a reassociation, not
// nondeterminism, and the microtest grades it with a tolerance plus a
// KNOWN_BAD control rather than with an equality
// (`native_attn_decode_splitk_hopper_microtest`).

#ifndef ATLAS_PAGED_DECODE_SPLITK_HOPPER_CUH
#define ATLAS_PAGED_DECODE_SPLITK_HOPPER_CUH

#include <cuda_bf16.h>

#define PD_WARP_SIZE 32
#ifndef HDIM
#define HDIM 256
#endif
// Per-lane element counts. A lane owns HDIM/32 head-dim elements; the u32
// counts are how many 32-bit loads that is for a 2-byte (BF16) and a 1-byte
// (FP8 E4M3) cache element.
#define PD_VEC        (HDIM / PD_WARP_SIZE)
#define PD_VEC_U32    (HDIM / (PD_WARP_SIZE * 2))
#define PD_VEC_U32_F8 (HDIM / (PD_WARP_SIZE * 4))
#define PD_NUM_WARPS 8
// KV positions batched per loop iteration. The whole point of the twins: the
// gb10 split-K kernel processes one position per iteration and is latency
// bound, the non-split kernel processes four and is not.
#define PD_BC 4

// Fewest KV positions a split is allowed to own.
//
// This is the "short contexts do not pay the reduce" rule, and it lives HERE
// rather than on the host because the host may not read a sequence's length:
// `seq_lens` is device memory and the launch geometry is captured into a CUDA
// graph. Applying it per sequence from that sequence's own `seq_len` keeps the
// rule co-batch-invariant (see the header note).
//
// 256 = one CTA's eight warps each running eight full PD_BC=4 batched
// iterations. Below that the fixed per-CTA cost — the Q load, the three-level
// smem tree merge, the workspace round trip and the reduce's extra pass — is
// no longer amortised, and a split count chosen for a 16 k context would make
// a 300-token one slower. Round 14 measures the elbow directly: the microtest
// prints GB/s per (num_splits, L) arm.
#define PD_MIN_KV_PER_SPLIT 256

// 2 BF16 packed in a u32 -> 2 F32. Used for Q on both twins, and for K/V on
// the BF16 twin. The FP8 unpack lives in the FP8 twin's own file, not here:
// it needs `<cuda_fp8.h>`, and a header that made every includer pull in a
// dtype it does not use is how the BF16 twin came to fail `nvcc --ptx` on
// `__nv_fp8x2_storage_t` (PTX gate, sm_90a, strict).
__device__ __forceinline__ void pd_unpack2_bf16(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

// This split's half-open KV range, from the SEQUENCE'S OWN length only.
//
// `split_size` is the ceiling division floored at PD_MIN_KV_PER_SPLIT, so a
// short sequence simply leaves the high splits empty (`kv_start == kv_end`)
// instead of giving each one a handful of positions. An empty split writes
// `l = 0` and the reduce skips it, so the set of merged partials — and hence
// the reduction tree — is a function of `(seq_len, num_splits)` and nothing
// else. Returns false when this split has no work.
__device__ __forceinline__ bool pd_split_bounds(
    unsigned int seq_len,
    unsigned int window_start,
    unsigned int split_id,
    unsigned int num_splits,
    unsigned int& kv_start,
    unsigned int& kv_end
) {
    const unsigned int attended = seq_len - window_start;
    unsigned int split_size = (attended + num_splits - 1u) / num_splits;
    if (split_size < PD_MIN_KV_PER_SPLIT) split_size = PD_MIN_KV_PER_SPLIT;
    const unsigned long long start = (unsigned long long)window_start
                                   + (unsigned long long)split_id * split_size;
    if (start >= (unsigned long long)seq_len) return false;
    kv_start = (unsigned int)start;
    kv_end = kv_start + split_size;
    if (kv_end > seq_len) kv_end = seq_len;
    return kv_start < kv_end;
}

// This warp's slice of [kv_start, kv_end), matching the non-split kernel's
// eight-way warp partition so the two agree on operand order within a split.
__device__ __forceinline__ void pd_warp_bounds(
    unsigned int kv_start, unsigned int kv_end, unsigned int warp_id,
    unsigned int& my_start, unsigned int& my_end
) {
    const unsigned int local_len = kv_end - kv_start;
    const unsigned int chunk = (local_len + PD_NUM_WARPS - 1u) / PD_NUM_WARPS;
    my_start = kv_start + warp_id * chunk;
    my_end = my_start + chunk;
    if (my_end > kv_end) my_end = kv_end;
    if (my_start > kv_end) my_start = kv_end;
}

// Inter-warp tree merge, then write this split's partial as
// `[o[head_dim], m, l]` F32 — the quantisation-agnostic workspace format the
// gb10 reduce already uses, so the two are interchangeable.
//
// `out_scale` is `v_scale` on the FP8 twin (o_reg accumulated raw-V; the merge
// and the reduce are both linear in o) and 1.0f on the BF16 twin.
__device__ __forceinline__ void pd_emit_partial(
    float m_val, float l_val, const float* o_reg, float out_scale,
    float* smem_m, float* smem_l, float* smem_o,
    unsigned int warp_id, unsigned int lane_id, unsigned int vec_offset,
    float* __restrict__ workspace,
    unsigned int seq_idx, unsigned int q_head, unsigned int split_id,
    unsigned int num_q_heads, unsigned int num_splits, unsigned int head_dim
) {
    if (lane_id == 0) {
        smem_m[warp_id] = m_val;
        smem_l[warp_id] = l_val;
    }
    #pragma unroll
    for (int i = 0; i < PD_VEC; i++) {
        smem_o[warp_id * HDIM + vec_offset + i] = o_reg[i] * out_scale;
    }
    __syncthreads();

    #pragma unroll
    for (int stride = PD_NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            const unsigned int other = warp_id + stride;
            const float lw = smem_l[other];
            if (lw > 0.0f) {
                const float mw = smem_m[other];
                const float my_m = smem_m[warp_id];
                const float my_l = smem_l[warp_id];
                const float m_new = fmaxf(my_m, mw);
                const float scale_me = __expf(my_m - m_new);
                const float scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < PD_VEC; i++) {
                    smem_o[warp_id * HDIM + vec_offset + i] =
                        smem_o[warp_id * HDIM + vec_offset + i] * scale_me +
                        smem_o[other * HDIM + vec_offset + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    const unsigned int ws_stride = head_dim + 2u;
    float* ws_base = workspace
        + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride
        + (unsigned long long)split_id * ws_stride;
    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < PD_VEC; i++) ws_base[vec_offset + i] = smem_o[vec_offset + i];
        if (lane_id == 0) {
            ws_base[head_dim] = smem_m[0];
            ws_base[head_dim + 1] = smem_l[0];
        }
    }
}

// Merge `num_splits` partials for one (seq, q_head) into the BF16 output.
//
// One warp per (q_head, seq). Splits are merged in INDEX ORDER and empty ones
// (`l <= 0`) are skipped, which is what makes the tree a function of the
// sequence's own length: `pd_split_bounds` decides emptiness from `seq_len`
// and `num_splits` alone.
__device__ __forceinline__ void pd_reduce_body(
    const float* __restrict__ workspace,
    __nv_bfloat16* __restrict__ O,
    unsigned int seq_idx, unsigned int q_head, unsigned int lane_id,
    unsigned int num_q_heads, unsigned int head_dim, unsigned int num_splits
) {
    const unsigned int vec_off = lane_id * PD_VEC;
    const unsigned int ws_stride = head_dim + 2u;
    const float* ws_base = workspace
        + ((unsigned long long)seq_idx * num_q_heads + q_head) * num_splits * ws_stride;

    float m = ws_base[head_dim];
    float l = ws_base[head_dim + 1];
    float o_reg[PD_VEC];
    #pragma unroll
    for (int i = 0; i < PD_VEC; i++) o_reg[i] = ws_base[vec_off + i];

    for (unsigned int s = 1; s < num_splits; s++) {
        const float* ws = ws_base + (unsigned long long)s * ws_stride;
        const float ls = ws[head_dim + 1];
        if (ls <= 0.0f) continue;
        const float ms = ws[head_dim];
        const float m_new = fmaxf(m, ms);
        const float scale_me = __expf(m - m_new);
        const float scale_s = __expf(ms - m_new);
        #pragma unroll
        for (int i = 0; i < PD_VEC; i++)
            o_reg[i] = o_reg[i] * scale_me + ws[vec_off + i] * scale_s;
        l = l * scale_me + ls * scale_s;
        m = m_new;
    }

    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    unsigned int* o32 = (unsigned int*)(O + (unsigned long long)seq_idx * num_q_heads * head_dim
                                          + (unsigned long long)q_head * head_dim + vec_off);
    #pragma unroll
    for (int i = 0; i < PD_VEC_U32; i++) {
        const float v0 = o_reg[2 * i] * inv_l;
        const float v1 = o_reg[2 * i + 1] * inv_l;
        const unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
        const unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
        o32[i] = lo | (hi << 16);
    }
}

#endif  // ATLAS_PAGED_DECODE_SPLITK_HOPPER_CUH
