// SPDX-License-Identifier: AGPL-3.0-only

// Atlas device-side MoE expert down_proj LoRA fold on SM121 (GB10). Two kernels
// — shrink then expand+fold — that together apply, for the SORTED grouped-GEMM
// output, one installed LoRA per adapted expert:
//
//   base_out[r, :] += scale_e * ( x[r, :] @ A_e^T ) @ B_e^T
//     for every sorted row r in [expert_offsets[e], expert_offsets[e+1]).
//
// This is the DEVICE-SIDE, capture-legal analogue of the host-synced per-expert
// loop in crates/spark-model/src/lora/expert_apply.rs (which D2H-copies
// expert_offsets and drives a host launch loop — both illegal under CUDA-graph
// capture). The reduction body here is lora_bgmv.cu VERBATIM, so it is
// BYTE-IDENTICAL to the per-row apply_lora_delta(m=1) recipe (the on-hardware
// oracle):
//   * uint4 K-vectorization, per-element fp32 accumulate under the GLOBAL
//     --fmad=false, __shfl_down_sync reduce order, 2-warp smem reduce,
//   * shrink stores BF16 xa (rounding boundary 1),
//   * expand rounds delta to BF16 (boundary 2) then folds
//     base += scale_e * bf16(delta) in fp32 — scale applied in fp32 AFTER the
//     BF16 delta rounding, never on the raw accumulator (mirrors
//     residual_add.cu bf16_scaled_add).
// Contraction runs at max_rank (A padded to [max_rank, k_in]; B row stride =
// max_rank with zero pad cols) — bit-identical to the true-rank product.
//
// The ONLY structural change vs lora_bgmv is the per-row axis: instead of one
// row per blockIdx.y keyed by seq_slot[row], the base grouped-GEMM's STATIC 3-D
// grid is used — blockIdx.z = expert, blockIdx.y = m-tile within that expert's
// device-resident [expert_offsets[e], expert_offsets[e+1]) span, blockIdx.x =
// out-element group. Empty experts / out-of-range tiles / unadapted experts
// early-return (device-derived, no host value), so the launch shape is a static
// worst-case bound (worst_case_m_tiles) that captures cleanly.
//
// CHUNKING (fixed-cap scratch at any ISL): xa is a [cap, max_rank] BF16 buffer,
// so a prefill chunk with total_expanded (te) > cap is folded in contiguous
// row windows [row_offset, row_end) of <= cap rows each (the host hooks loop).
// Each expert's span is intersected with the window; per-expert tiles rebase to
// max(m_start, row_offset) so grid.y = ceil((row_end-row_offset)/64) covers the
// slice. xa is indexed at the LOCAL row (r - row_offset); x / base_out /
// sorted_token_ids / moe_row_adapter / expert_offsets all stay ABSOLUTE — no
// input re-basing, so the down path is BIT-IDENTICAL per row regardless of which
// window a row lands in (each row's reduction is intact; no cross-row/atomic
// accumulation). row_offset=0, row_end>=te reduces to the pre-chunk kernel
// exactly (max(m_start,0)=m_start, min(m_end,te)=m_end, r-0=r).
//
// Per-row base skip (mixed-batch, Incr-2): when moe_row_adapter != NULL, a row
// whose owning token (moe_row_adapter[sorted_token_ids[r]]) is < 0 is a base row
// and is skipped — the whole block takes the skip uniformly (the predicate
// depends only on r), so every __syncthreads is still reached by all live
// threads. NULL moe_row_adapter (Incr-1, single active adapter) folds every row
// of every adapted expert; the request-granularity opt-out is handled host-side.

#include <cuda_bf16.h>

#define MLG_BLOCK_SIZE 256
#define MLG_N_PER_BLOCK 4
#define MLG_WARP_SIZE 32
#define MLG_VEC_SIZE 8   // BF16 per uint4 (128-bit) load
#define MLG_M_TILE 64    // == base grouped GEMM M_TILE (worst_case_m_tiles pairs with this)

// ── Kernel 1: shrink ────────────────────────────────────────────────────────
//   xa[r, :] = x[r, :] @ A_e^T          (N := max_rank outputs, K := k_in)
// where A_e = (const __nv_bfloat16*)a_expert_table[e] is that expert's padded
// [max_rank, k_in] region (row-major). Emits BF16 xa — the oracle's boundary 1.
//
// Grid: (ceil(max_rank/4), worst_case_m_tiles, num_experts)   Block: (256,1,1)
extern "C" __global__ void moe_lora_grouped_down_shrink(
    const __nv_bfloat16* __restrict__ x,             // [te, k_in] BF16 (post-SiLU sorted activations)
    const int* __restrict__ expert_offsets,          // [num_experts+1] i32 prefix sum
    const int* __restrict__ sorted_token_ids,        // [te] i32 sorted row -> packed token
    const int* __restrict__ moe_row_adapter,         // [num_tokens] i32 or NULL (<0 => base skip)
    const unsigned long long* __restrict__ a_expert_table,  // [num_experts] u64 A_e addr (0 = unadapted)
    __nv_bfloat16* __restrict__ xa,                  // [<=cap, max_rank] BF16 (out, LOCAL row r-row_offset)
    unsigned int num_experts,
    unsigned int max_rank,                           // output dim (== A padded row count)
    unsigned int k_in,                               // contraction dim
    unsigned int x_gather,                           // 0: x row = sorted row r (down); 1: x row = sorted_token_ids[r] (gate/up)
    unsigned int row_offset,                         // first ABSOLUTE sorted row in this chunk's window
    unsigned int row_end                             // one-past-last ABSOLUTE row (== min(row_offset+cap, te))
) {
    const unsigned int e = blockIdx.z;
    if (e >= num_experts) return;
    const int m_start = expert_offsets[e];
    const int m_end = expert_offsets[e + 1];
    if (m_end <= m_start) return;                    // no rows routed to this expert
    const unsigned long long a_addr = a_expert_table[e];
    if (a_addr == 0ULL) return;                      // expert not adapted -> whole tile skips

    // Chunk window: clamp this expert's row span to [row_offset, row_end). Tiles
    // start at the window-relative expert base so grid.y == ceil(window/64)
    // covers every expert slice; an expert fully outside the window early-returns
    // (block-uniform, __syncthreads-safe). row_offset=0, row_end>=te is the
    // unchunked call: win_start==m_start, win_end==m_end, xa index r-0==r.
    const int win_start = max(m_start, (int)row_offset);
    const int win_end = min(m_end, (int)row_end);
    const int r0 = win_start + (int)blockIdx.y * MLG_M_TILE;
    if (r0 >= win_end) return;                        // tile beyond this expert's windowed rows
    const int r1 = min(r0 + MLG_M_TILE, win_end);

    const __nv_bfloat16* A_base = (const __nv_bfloat16*)a_addr;  // [max_rank, k_in]

    const unsigned int threads_per_out = MLG_BLOCK_SIZE / MLG_N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = threadIdx.x % MLG_WARP_SIZE;

    const unsigned int n = blockIdx.x * MLG_N_PER_BLOCK + local_out;  // which xa element
    if (n >= max_rank) return;                        // whole 64-lane group returns together

    const __nv_bfloat16* Brow = A_base + (unsigned long long)n * k_in;  // [k_in]
    const uint4* B_vec = (const uint4*)Brow;
    const unsigned int K_VEC = k_in / MLG_VEC_SIZE;
    const unsigned int tail_start = K_VEC * MLG_VEC_SIZE;

    __shared__ float smem[MLG_N_PER_BLOCK * 2];

    for (int r = r0; r < r1; ++r) {
        // Per-row base skip (uniform across the block — depends only on r).
        if (moe_row_adapter != nullptr && moe_row_adapter[sorted_token_ids[r]] < 0) {
            continue;
        }
        // down: x is already the sorted post-SiLU activation, so the x-row IS the
        // sorted row r. gate/up: x is the TOKEN-MAJOR `expert_input`, so gather the
        // owning token via sorted_token_ids[r] (the same map the base gate/up GEMM
        // fuses). The expand stage still keys xa/base_out by r, so only this read
        // changes — the down path (x_gather==0) stays bit-identical.
        const int x_row = x_gather ? sorted_token_ids[r] : r;
        const __nv_bfloat16* Arow = x + (unsigned long long)x_row * k_in;  // [k_in]
        const uint4* A_vec = (const uint4*)Arow;

        float acc = 0.0f;
        for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
            uint4 a_data = A_vec[kv];
            uint4 b_data = B_vec[kv];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
                *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
                *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
                acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
                acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
            }
        }
        for (unsigned int k = tail_start + lane; k < k_in; k += threads_per_out) {
            acc += __bfloat162float(Arow[k]) * __bfloat162float(Brow[k]);
        }

        #pragma unroll
        for (int offset = MLG_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (warp_lane == 0) {
            smem[local_out * 2 + (lane / MLG_WARP_SIZE)] = acc;
        }
        __syncthreads();
        if (lane == 0) {
            float result = smem[local_out * 2] + smem[local_out * 2 + 1];
            xa[(unsigned long long)(r - (int)row_offset) * max_rank + n] = __float2bfloat16(result);
        }
        __syncthreads();  // guard smem reuse before the next row iteration
    }
}

// ── Kernel 2: expand + fold ──────────────────────────────────────────────────
//   delta[r, :] = xa[r, :] @ B_e^T        (N := n_out outputs, K := max_rank)
//   base_out[r, o] += scale_e * delta[r, o]   (fused, mirrors bf16_scaled_add)
// where B_e = (const __nv_bfloat16*)b_expert_table[e] is that expert's packed
// [n_out, max_rank] region (ROW STRIDE = max_rank), scale_e = scale_expert_table[e].
//
// Grid: (ceil(n_out/4), worst_case_m_tiles, num_experts)   Block: (256,1,1)
extern "C" __global__ void moe_lora_grouped_down_expand_fold(
    const __nv_bfloat16* __restrict__ xa,            // [<=cap, max_rank] BF16 (from kernel 1, LOCAL row r-row_offset)
    const int* __restrict__ expert_offsets,          // [num_experts+1] i32 prefix sum
    const int* __restrict__ sorted_token_ids,        // [te] i32 sorted row -> packed token
    const int* __restrict__ moe_row_adapter,         // [num_tokens] i32 or NULL (<0 => base skip)
    const unsigned long long* __restrict__ b_expert_table,  // [num_experts] u64 B_e addr (0 = unadapted)
    const float* __restrict__ scale_expert_table,    // [num_experts] f32 scale_e
    __nv_bfloat16* __restrict__ base_out,            // [te, n_out] BF16 (expert_down_out, in-place fold)
    unsigned int num_experts,
    unsigned int n_out,                              // output dim (== hidden)
    unsigned int max_rank,                           // contraction dim (== B row stride)
    unsigned int row_offset,                         // first ABSOLUTE sorted row in this chunk's window
    unsigned int row_end                             // one-past-last ABSOLUTE row (== min(row_offset+cap, te))
) {
    const unsigned int e = blockIdx.z;
    if (e >= num_experts) return;
    const int m_start = expert_offsets[e];
    const int m_end = expert_offsets[e + 1];
    if (m_end <= m_start) return;
    const unsigned long long b_addr = b_expert_table[e];
    if (b_addr == 0ULL) return;

    // Same window clamp as the shrink kernel — xa is read at the LOCAL row
    // (r - row_offset), base_out folded at the ABSOLUTE row r.
    const int win_start = max(m_start, (int)row_offset);
    const int win_end = min(m_end, (int)row_end);
    const int r0 = win_start + (int)blockIdx.y * MLG_M_TILE;
    if (r0 >= win_end) return;
    const int r1 = min(r0 + MLG_M_TILE, win_end);

    const __nv_bfloat16* B_base = (const __nv_bfloat16*)b_addr;  // [n_out, max_rank]
    const float scale_e = scale_expert_table[e];

    const unsigned int threads_per_out = MLG_BLOCK_SIZE / MLG_N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = threadIdx.x % MLG_WARP_SIZE;

    const unsigned int n = blockIdx.x * MLG_N_PER_BLOCK + local_out;  // which output element
    if (n >= n_out) return;

    const __nv_bfloat16* Brow = B_base + (unsigned long long)n * max_rank;  // [max_rank]
    const uint4* B_vec = (const uint4*)Brow;
    const unsigned int K_VEC = max_rank / MLG_VEC_SIZE;
    const unsigned int tail_start = K_VEC * MLG_VEC_SIZE;

    __shared__ float smem[MLG_N_PER_BLOCK * 2];

    for (int r = r0; r < r1; ++r) {
        if (moe_row_adapter != nullptr && moe_row_adapter[sorted_token_ids[r]] < 0) {
            continue;
        }
        const __nv_bfloat16* Arow = xa + (unsigned long long)(r - (int)row_offset) * max_rank;  // [max_rank]
        const uint4* A_vec = (const uint4*)Arow;

        float acc = 0.0f;
        for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
            uint4 a_data = A_vec[kv];
            uint4 b_data = B_vec[kv];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
                *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
                *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
                acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
                acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
            }
        }
        for (unsigned int k = tail_start + lane; k < max_rank; k += threads_per_out) {
            acc += __bfloat162float(Arow[k]) * __bfloat162float(Brow[k]);
        }

        #pragma unroll
        for (int offset = MLG_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (warp_lane == 0) {
            smem[local_out * 2 + (lane / MLG_WARP_SIZE)] = acc;
        }
        __syncthreads();
        if (lane == 0) {
            float result = smem[local_out * 2] + smem[local_out * 2 + 1];
            // Fold: reproduce residual_add.cu bf16_scaled_add EXACTLY. Round the
            // expand accumulator to BF16 first (the oracle's lora_delta boundary),
            // widen back to fp32, apply scale_e in fp32, accumulate onto BF16
            // base_out — never scale the raw accumulator. Each (r, n) is owned by
            // exactly one thread-group, so the read-modify-write needs no atomic.
            __nv_bfloat16 delta_bf = __float2bfloat16(result);
            float d = __bfloat162float(delta_bf);
            __nv_bfloat16* dst = base_out + (unsigned long long)r * n_out + n;
            float o = __bfloat162float(*dst);
            *dst = __float2bfloat16(o + scale_e * d);
        }
        __syncthreads();  // guard smem reuse before the next row iteration
    }
}
