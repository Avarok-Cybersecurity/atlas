// SPDX-License-Identifier: AGPL-3.0-only

// Atlas fused batched gather-matrix-vector (bgmv) for per-request LoRA routing
// on SM121 (GB10). Two kernels — shrink then expand+fold — that together apply,
// for a batch of N decode rows each naming its own adapter slot:
//
//   out[i, :] += scale_s * ( x[i, :] @ A_s^T ) @ B_s^T      where s = seq_slot[i]
//
// This is the BATCHED, per-row-routed analogue of the single-adapter global
// path in crates/spark-model/src/layers/ops/lora_delta.rs (apply_lora_delta
// m==1). It MUST be BYTE-IDENTICAL to N sequential apply_lora_delta(m=1) calls
// for the same (x_i, slot s_i) — that is the review/on-hardware oracle.
//
// BIT-IDENTITY: each kernel's per-output reduction body is dense_gemv_bf16.cu
// VERBATIM (same uint4 K-vectorization, same per-element fp32 accumulate under
// the GLOBAL --fmad=false, same __shfl_down_sync order, same 2-warp smem
// reduce, same __float2bfloat16 store). The ONLY additions are:
//   (1) a row axis on blockIdx.y (one row = one decode sequence),
//   (2) a per-row pointer gather: A_s / B_s come from a_table[s] / b_table[s],
//   (3) the fold in kernel 2 reproduces residual_add.cu bf16_scaled_add exactly.
//
// The oracle rounds to BF16 at TWO buffer boundaries (shrink→lora_xa BF16;
// expand→lora_delta BF16, then scaled_add folds base += scale*bf16(delta)).
// Kernel 1 emits BF16 xa to global memory (boundary 1). Kernel 2 reads that
// BF16 xa back, and folds delta_bf = __float2bfloat16(acc) then
// base += scale*__bfloat162float(delta_bf) (boundary 2) — so per-slot scale is
// applied in fp32 AFTER the BF16 delta rounding, never on the raw accumulator.
//
// CONTRACTION runs at max_rank (A padded to max_rank rows, B row stride =
// max_rank with zero pad cols) — bit-identical to true rank, exactly like the
// global path (see LoraPair.max_rank docs). Never contract at true `rank`.
//
// NULL/base handling (twofold, mirrors apply_lora_delta not being called):
//   seq_slot[row] < 0            → base sequence, no delta (early return)
//   a_table[s] == 0 (NULL slot)  → slot resident but doesn't adapt this module

#include <cuda_bf16.h>

#define BGMV_BLOCK_SIZE 256
#define BGMV_N_PER_BLOCK 4
#define BGMV_WARP_SIZE 32
#define BGMV_VEC_SIZE 8  // BF16 per uint4 (128-bit) load

// ── Kernel 1: shrink ────────────────────────────────────────────────────────
//   xa[row, :] = x[row, :] @ A_s^T          (N := max_rank outputs, K := k_in)
// where A_s = (const __nv_bfloat16*)a_table[seq_slot[row]] is the slot's padded
// [max_rank, k_in] region (row-major). Emits BF16 xa — the oracle's boundary 1.
//
// Grid: (ceil(max_rank / 4), N, 1)   Block: (256, 1, 1)
extern "C" __global__ void lora_bgmv_shrink(
    const __nv_bfloat16* __restrict__ x,        // [N, x_row_stride] BF16 (row-major activations)
    const int* __restrict__ seq_slot,           // [N] i32 (<0 => base, skip)
    const unsigned long long* __restrict__ a_table,  // [max_loras] u64 pool addrs (0 = NULL)
    __nv_bfloat16* __restrict__ xa,             // [N, max_rank] BF16 (out, row-major)
    unsigned int N,                             // batch rows
    unsigned int max_rank,                      // output dim (== A padded row count)
    unsigned int k_in,                          // contraction dim
    unsigned int x_row_stride                   // elements between x rows (>= k_in)
) {
    const unsigned int row = blockIdx.y;
    if (row >= N) return;
    const int s = seq_slot[row];
    if (s < 0) return;                          // base sequence: no delta
    const unsigned long long a_addr = a_table[s];
    if (a_addr == 0ULL) return;                 // slot doesn't adapt this module

    const __nv_bfloat16* A_base = (const __nv_bfloat16*)a_addr;  // [max_rank, k_in]

    const unsigned int threads_per_out = BGMV_BLOCK_SIZE / BGMV_N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * BGMV_N_PER_BLOCK + local_out;  // which xa element
    if (n >= max_rank) return;

    // Reduction body: identical numeric recipe to dense_gemv_bf16.
    //   "A"-operand = x row  (the [1,K] activation), "B"-operand = A_s row n.
    const __nv_bfloat16* Arow = x + (unsigned long long)row * x_row_stride;   // [k_in]
    const __nv_bfloat16* Brow = A_base + (unsigned long long)n * k_in;        // [k_in]

    float acc = 0.0f;
    const unsigned int K_VEC = k_in / BGMV_VEC_SIZE;
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
        const unsigned int tail_start = K_VEC * BGMV_VEC_SIZE;
        for (unsigned int k = tail_start + lane; k < k_in; k += threads_per_out) {
            acc += __bfloat162float(Arow[k]) * __bfloat162float(Brow[k]);
        }
    }

    const unsigned int warp_lane = threadIdx.x % BGMV_WARP_SIZE;
    #pragma unroll
    for (int offset = BGMV_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float smem[BGMV_N_PER_BLOCK * 2];
    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / BGMV_WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        xa[(unsigned long long)row * max_rank + n] = __float2bfloat16(result);
    }
}

// ── Kernel 2: expand + fold ──────────────────────────────────────────────────
//   delta[row, :] = xa[row, :] @ B_s^T       (N := n_out outputs, K := max_rank)
//   out[row, o]  += scale_s * delta[row, o]  (fused, mirrors bf16_scaled_add)
// where B_s = (const __nv_bfloat16*)b_table[seq_slot[row]] is the slot's packed
// [n_out, max_rank] region (ROW STRIDE = max_rank), and scale_s = scale_table[s].
//
// out is written in place at base_out[row * out_row_stride + o] (BF16). The
// per-row output stride lets the K/V apply site target the STRIDED qkv_buf
// (row stride = per_seq_qkv elements) while the O site stays contiguous.
//
// Grid: (ceil(n_out / 4), N, 1)   Block: (256, 1, 1)
extern "C" __global__ void lora_bgmv_expand_fold(
    const __nv_bfloat16* __restrict__ xa,       // [N, max_rank] BF16 (from kernel 1)
    const int* __restrict__ seq_slot,           // [N] i32 (<0 => base, skip)
    const unsigned long long* __restrict__ b_table,  // [max_loras] u64 pool addrs (0 = NULL)
    const float* __restrict__ scale_table,      // [max_loras] f32 per-slot scale
    __nv_bfloat16* __restrict__ base_out,       // [N, out_row_stride] BF16 (in-place fold)
    unsigned int N,                             // batch rows
    unsigned int n_out,                         // output dim
    unsigned int max_rank,                      // contraction dim (== B row stride)
    unsigned int out_row_stride                 // elements between base_out rows (>= n_out)
) {
    const unsigned int row = blockIdx.y;
    if (row >= N) return;
    const int s = seq_slot[row];
    if (s < 0) return;                          // base sequence: no delta
    const unsigned long long b_addr = b_table[s];
    if (b_addr == 0ULL) return;                 // slot doesn't adapt this module

    const __nv_bfloat16* B_base = (const __nv_bfloat16*)b_addr;  // [n_out, max_rank]

    const unsigned int threads_per_out = BGMV_BLOCK_SIZE / BGMV_N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * BGMV_N_PER_BLOCK + local_out;  // which output element
    if (n >= n_out) return;

    // Reduction body: identical recipe. "A"-operand = xa row (the [1,max_rank]
    // shrink result), "B"-operand = B_s row n (stride max_rank).
    const __nv_bfloat16* Arow = xa + (unsigned long long)row * max_rank;      // [max_rank]
    const __nv_bfloat16* Brow = B_base + (unsigned long long)n * max_rank;    // [max_rank]

    float acc = 0.0f;
    const unsigned int K_VEC = max_rank / BGMV_VEC_SIZE;
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
        const unsigned int tail_start = K_VEC * BGMV_VEC_SIZE;
        for (unsigned int k = tail_start + lane; k < max_rank; k += threads_per_out) {
            acc += __bfloat162float(Arow[k]) * __bfloat162float(Brow[k]);
        }
    }

    const unsigned int warp_lane = threadIdx.x % BGMV_WARP_SIZE;
    #pragma unroll
    for (int offset = BGMV_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float smem[BGMV_N_PER_BLOCK * 2];
    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / BGMV_WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        // Fold: reproduce residual_add.cu bf16_scaled_add EXACTLY. Round the
        // expand accumulator to BF16 first (the oracle's lora_delta BF16
        // boundary), widen back to fp32, then apply per-slot scale in fp32 and
        // accumulate onto the BF16 base_out — never scale the raw accumulator.
        __nv_bfloat16 delta_bf = __float2bfloat16(result);
        float d = __bfloat162float(delta_bf);
        float sc = scale_table[s];
        __nv_bfloat16* dst = base_out + (unsigned long long)row * out_row_stride + n;
        float o = __bfloat162float(*dst);
        *dst = __float2bfloat16(o + sc * d);
    }
}
