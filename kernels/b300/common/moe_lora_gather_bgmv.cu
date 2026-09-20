// SPDX-License-Identifier: AGPL-3.0-only

// Atlas DECODE-path MoE expert down_proj LoRA fold on SM121 (GB10). Two kernels
// — shrink then expand+fold — that fold, for the UNSORTED slot-major decode
// activations, one installed LoRA per adapted expert, keyed per (token, slot)
// row by the same `indices` array the fused expert GEMV already routed on:
//
//   base_out[row, :] += scale_e * ( x[row, :] @ A_e^T ) @ B_e^T
//     for every flat slot `row in [0, n_slots)`, where e = indices[row] and the
//     row's owning token `row / top_k` is NOT a base row (row_adapter check).
//
// This is the composition of the two landed capture-safe folds:
//   * lora_bgmv.cu        — per-row `blockIdx.y` axis + per-row skip predicate,
//   * moe_lora_grouped_down.cu — expert-keyed pointer gather (A_e/B_e/scale_e).
// The reduction body is lora_bgmv.cu / dense_gemv_bf16 VERBATIM (uint4 K-vec,
// per-element fp32 accumulate under the GLOBAL --fmad=false, __shfl_down_sync
// order, 2-warp smem reduce, __float2bfloat16 boundaries), so the folded delta
// is BYTE-IDENTICAL to the prefill grouped fold (moe_lora_grouped_down) for the
// same `x` row — that is the prefill/decode BF16-ULP oracle.
//
// x is NOT recomputed here. The caller runs the EXISTING `moe_silu_mul` kernel
// (the same kernel + ±10 swiglu clamp + __float2bfloat16 round the prefill fold
// uses to produce its `x`) into a packed `[n_slots, k_in]` BF16 scratch, and
// passes it as `x`. Reusing that kernel verbatim is what makes the decode `x`
// boundary bit-identical to prefill's — so both stages here read a materialized
// BF16 row with uint4 loads and never fold a silu into the dot (which would skip
// the mandatory BF16 round and break the oracle).
//
// The ONLY structural deltas vs lora_bgmv are the per-row gather key and the two
// skip predicates:
//   token = row / top_k
//   e     = indices[row]                                    (u32 expert id)
//   if (row_adapter != NULL && row_adapter[token] < 0) return;   // base row
//   if (e >= n_experts) return;                                  // higher-idx expert unadapted
//   a_addr = a_table[e]; if (a_addr == 0) return;                // expert not adapted
//
// CONTRACTION runs at max_rank (A padded to [max_rank, k_in]; B row stride =
// max_rank with zero pad cols) — bit-identical to the true-rank product. The
// per-(row,n) output is owned by exactly one thread-group, so the fold's
// read-modify-write needs no atomic. All args (x, xa, base_out, indices,
// row_adapter, tables) are pointer/value-stable ⇒ CUDA-graph capture legal.

#include <cuda_bf16.h>

#define GBGMV_BLOCK_SIZE 256
#define GBGMV_N_PER_BLOCK 4
#define GBGMV_WARP_SIZE 32
#define GBGMV_VEC_SIZE 8  // BF16 per uint4 (128-bit) load

// ── Kernel 1: shrink ────────────────────────────────────────────────────────
//   xa[row, :] = x[row, :] @ A_e^T          (N := max_rank outputs, K := k_in)
// where A_e = (const __nv_bfloat16*)a_table[indices[row]] is that expert's
// padded [max_rank, k_in] region (row-major). Emits BF16 xa — the oracle's
// boundary 1. `x` is contiguous [n_slots, k_in] (post-swiglu activations).
//
// Grid: (ceil(max_rank / 4), n_slots, 1)   Block: (256, 1, 1)
extern "C" __global__ void moe_lora_gather_bgmv_shrink(
    const __nv_bfloat16* __restrict__ x,        // [n_slots, k_in] BF16 (silu(gate)*up)
    const unsigned int* __restrict__ indices,   // [n_slots] u32 expert id per flat slot
    const int* __restrict__ row_adapter,        // [num_tokens] i32 or NULL (<0 => base skip)
    const unsigned long long* __restrict__ a_table,  // [n_experts] u64 A_e addrs (0 = unadapted)
    __nv_bfloat16* __restrict__ xa,             // [n_slots, max_rank] BF16 (out, row-major)
    unsigned int n_slots,                       // flat (token, slot) rows = num_tokens * top_k
    unsigned int top_k,                         // slots per token (row / top_k = token)
    unsigned int n_experts,                     // A/B/scale table length (max adapted id + 1)
    unsigned int max_rank,                      // output dim (== A padded row count)
    unsigned int k_in,                          // contraction dim (down: moe_intermediate_size; gate/up: hidden)
    unsigned int x_gather                       // 0: x row = flat slot (down); 1: x row = token = row/top_k (gate/up)
) {
    const unsigned int row = blockIdx.y;
    if (row >= n_slots) return;
    // Per-token base skip (whole block takes it uniformly — depends only on row).
    if (row_adapter != nullptr && row_adapter[row / top_k] < 0) return;
    const unsigned int e = indices[row];
    if (e >= n_experts) return;                 // expert beyond the adapted table -> unadapted
    const unsigned long long a_addr = a_table[e];
    if (a_addr == 0ULL) return;                 // expert not adapted -> no delta

    const __nv_bfloat16* A_base = (const __nv_bfloat16*)a_addr;  // [max_rank, k_in]

    const unsigned int threads_per_out = GBGMV_BLOCK_SIZE / GBGMV_N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * GBGMV_N_PER_BLOCK + local_out;  // which xa element
    if (n >= max_rank) return;

    // down: x = the packed per-slot post-swiglu activation, so the x-row IS the
    // flat slot `row`. gate/up: x = the per-TOKEN `expert_input` (one row shared by
    // all top_k slots), so read the owning token `row / top_k` (already computed
    // above for the row_adapter check). down path (x_gather==0) stays bit-identical.
    const unsigned int x_row = x_gather ? (row / top_k) : row;
    const __nv_bfloat16* Arow = x + (unsigned long long)x_row * k_in;  // [k_in]
    const __nv_bfloat16* Brow = A_base + (unsigned long long)n * k_in; // [k_in]

    float acc = 0.0f;
    const unsigned int K_VEC = k_in / GBGMV_VEC_SIZE;
    const uint4* A_vec = (const uint4*)Arow;
    const uint4* B_vec = (const uint4*)Brow;

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
    {
        const unsigned int tail_start = K_VEC * GBGMV_VEC_SIZE;
        for (unsigned int k = tail_start + lane; k < k_in; k += threads_per_out) {
            acc += __bfloat162float(Arow[k]) * __bfloat162float(Brow[k]);
        }
    }

    const unsigned int warp_lane = threadIdx.x % GBGMV_WARP_SIZE;
    #pragma unroll
    for (int offset = GBGMV_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float smem[GBGMV_N_PER_BLOCK * 2];
    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / GBGMV_WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        xa[(unsigned long long)row * max_rank + n] = __float2bfloat16(result);
    }
}

// ── Kernel 2: expand + fold ──────────────────────────────────────────────────
//   delta[row, :] = xa[row, :] @ B_e^T       (N := n_out outputs, K := max_rank)
//   base_out[row, o] += scale_e * delta[row, o]   (fused, mirrors bf16_scaled_add)
// where B_e = (const __nv_bfloat16*)b_table[indices[row]] is that expert's packed
// [n_out, max_rank] region (ROW STRIDE = max_rank), scale_e = scale_table[e].
// base_out is contiguous [n_slots, n_out] = the slot-major `expert_down_out`.
//
// Grid: (ceil(n_out / 4), n_slots, 1)   Block: (256, 1, 1)
extern "C" __global__ void moe_lora_gather_bgmv_expand_fold(
    const __nv_bfloat16* __restrict__ xa,       // [n_slots, max_rank] BF16 (from kernel 1)
    const unsigned int* __restrict__ indices,   // [n_slots] u32 expert id per flat slot
    const int* __restrict__ row_adapter,        // [num_tokens] i32 or NULL (<0 => base skip)
    const unsigned long long* __restrict__ b_table,  // [n_experts] u64 B_e addrs (0 = unadapted)
    const float* __restrict__ scale_table,      // [n_experts] f32 per-expert scale
    __nv_bfloat16* __restrict__ base_out,       // [n_slots, n_out] BF16 (expert_down_out, in place)
    unsigned int n_slots,
    unsigned int top_k,
    unsigned int n_experts,
    unsigned int n_out,                         // output dim (== hidden)
    unsigned int max_rank                       // contraction dim (== B row stride)
) {
    const unsigned int row = blockIdx.y;
    if (row >= n_slots) return;
    if (row_adapter != nullptr && row_adapter[row / top_k] < 0) return;
    const unsigned int e = indices[row];
    if (e >= n_experts) return;
    const unsigned long long b_addr = b_table[e];
    if (b_addr == 0ULL) return;

    const __nv_bfloat16* B_base = (const __nv_bfloat16*)b_addr;  // [n_out, max_rank]

    const unsigned int threads_per_out = GBGMV_BLOCK_SIZE / GBGMV_N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * GBGMV_N_PER_BLOCK + local_out;  // which output element
    if (n >= n_out) return;

    const __nv_bfloat16* Arow = xa + (unsigned long long)row * max_rank;    // [max_rank]
    const __nv_bfloat16* Brow = B_base + (unsigned long long)n * max_rank;  // [max_rank]

    float acc = 0.0f;
    const unsigned int K_VEC = max_rank / GBGMV_VEC_SIZE;
    const uint4* A_vec = (const uint4*)Arow;
    const uint4* B_vec = (const uint4*)Brow;

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
    {
        const unsigned int tail_start = K_VEC * GBGMV_VEC_SIZE;
        for (unsigned int k = tail_start + lane; k < max_rank; k += threads_per_out) {
            acc += __bfloat162float(Arow[k]) * __bfloat162float(Brow[k]);
        }
    }

    const unsigned int warp_lane = threadIdx.x % GBGMV_WARP_SIZE;
    #pragma unroll
    for (int offset = GBGMV_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float smem[GBGMV_N_PER_BLOCK * 2];
    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / GBGMV_WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        // Fold: reproduce residual_add.cu bf16_scaled_add EXACTLY. Round the
        // expand accumulator to BF16 first (the oracle's delta boundary), widen
        // back to fp32, apply per-expert scale in fp32, accumulate onto BF16
        // base_out — never scale the raw accumulator.
        __nv_bfloat16 delta_bf = __float2bfloat16(result);
        float d = __bfloat162float(delta_bf);
        float sc = scale_table[e];
        __nv_bfloat16* dst = base_out + (unsigned long long)row * n_out + n;
        float o = __bfloat162float(*dst);
        *dst = __float2bfloat16(o + sc * d);
    }
}
