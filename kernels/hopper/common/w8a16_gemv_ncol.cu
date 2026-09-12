// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 batched GEMV, N-COLUMN BLOCKED — block-scaled FP8 weight, BF16
// activations. Computes exactly what `w8a16_gemv_batch4.cu` computes:
//
//   C[t, n] = sum_k A[t, k] * E4M3_LUT[B[n, k]] * block_scale[n/128, k/128]
//             for t in 0..M  (M <= MAX_M)
//
// WHY (#927, decode concurrency). `w8a16_gemv_batchm_impl<16>` gives every
// thread ONE output column, so per 16 weight bytes it pays, per activation row:
// two `uint4` loads of A and 16 BF16->FP32 converts, then the 16 FFMA that are
// the actual work. At MAX_M=16 that is ~36 ALU ops and 32 activation loads per
// weight byte, which is why #927's H100 receipt reads 342 GB/s (gate/up,
// N=17408 K=5120) against ~3,000 GB/s of HBM3 — the kernel is ALU/LSU bound,
// not bandwidth bound, and the whole cost scales with M.
//
// But every column's thread loads and converts the SAME A elements. Giving one
// thread N_COLS ADJACENT columns amortises both over N_COLS weight bytes:
//
//   ops per weight byte   MAX_M=16:  N_COLS=1 -> ~36   N_COLS=2 -> ~27
//                                    N_COLS=4 -> ~23   (floor 18 = the FFMA)
//   A `uint4` loads/byte             N_COLS=1 ->  32   N_COLS=2 ->  16
//
// 🟢 NUMERICS — BIT-IDENTICAL, and that is the entire point of this file next
// to `w8a16_gemm_m16.cu` (which buys more and REASSOCIATES). Every accumulator
// here sees the same operands in the same order as `w8a16_gemv_batchm_impl`:
//   * the lane -> k16 map is unchanged (`threads_per_out` is still 64, still
//     `k16 = lane; k16 += threads_per_out`), so each partial sum covers the
//     same K subset in the same sequence;
//   * the inner `acc += lo*wf[2j]; acc += hi*wf[2j+1]` pair order is unchanged
//     — the BF16->FP32 convert is merely HOISTED out of the column loop, and a
//     BF16->FP32 widening is exact, so the multiplied value is the same bits;
//   * the cross-warp reduction (shfl_down tree, then the two-partial add in
//     shared memory) is unchanged per (row, column).
// So each row stays bit-identical to the scalar `w8a16_gemv` that M=1 decode
// runs, exactly as `w8a16_gemv_batch4`/`_batch16` are. Oracle:
// `examples/native_fp8_attn_decode_batch_microtest`.
//
// A:[M, a_row_stride] BF16 (first K used), B:[N, K] FP8 E4M3,
// block_scale:[N/128, K/128] FP32, C:[M, c_row_stride] BF16 (first N used).
// Block: (256, 1, 1). Grid: (ceil(N / (4 * N_COLS)), 1, 1).
//
// Entry points: `w8a16_gemv_batch16_ncol2` / `_ncol4` (contiguous A and C) and
// their `_strided` siblings, which take the A and C row pitches in ELEMENTS —
// the multi-seq decode QKV buffer is [n, per_seq_qkv] with Q/K/V at fixed
// offsets inside each row, so a strided C targets one projection in place.

#include <cuda_bf16.h>

#include "e4m3_lut.cuh"

#define BLOCK_SIZE 256
// Output GROUPS per block. Each group is one `threads_per_out`=64 lane team,
// exactly as in `w8a16_gemv_batch4.cu`; a group now owns N_COLS columns
// instead of one, so a block covers 4 * N_COLS columns.
#define N_GROUPS_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128

template <int MAX_M, int N_COLS>
__device__ __forceinline__ void w8a16_gemv_ncol_impl(
    const __nv_bfloat16* __restrict__ A,    // [M, a_row_stride] BF16, K used
    const unsigned char* __restrict__ B,     // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,   // [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,           // [M, c_row_stride] BF16, N used
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_GROUPS_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    // First column of this thread's group, and how many of its N_COLS columns
    // are in range. Tail groups keep running (no early `return`) so every
    // thread reaches the `__syncthreads` below.
    const unsigned int n0 = (blockIdx.x * N_GROUPS_PER_BLOCK + local_out) * N_COLS;
    const unsigned int ncols = (n0 < N) ? min((unsigned int)N_COLS, N - n0) : 0u;

    __shared__ float s_lut[256];
    s_lut[threadIdx.x] = E4M3_LUT[threadIdx.x];
    __syncthreads();

    const unsigned int K16 = K / 16;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;

    float acc[MAX_M][N_COLS];
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        #pragma unroll
        for (int c = 0; c < N_COLS; c++) acc[t][c] = 0.0f;
    }

    // A tail group with no in-range column skips the K walk entirely (there is
    // no `__syncthreads` inside it, so this cannot desynchronise the block) —
    // the single-column kernel's `if (n >= N) return;` with the early exit
    // replaced by a mask, because the reduction below must keep every warp.
    for (unsigned int k16 = (ncols != 0) ? lane : K16; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;
        const unsigned int k_block = base_k / FP8_BLOCK;

        // 16 FP8 weight bytes per column, dequantized + scaled ONCE for all M
        // rows — as in `w8a16_gemv_batchm_impl`, now for N_COLS columns.
        //
        // A tail group's out-of-range columns are CLAMPED to the last valid one
        // rather than masked: they then compute a duplicate of a real column
        // into a private accumulator that the store below never writes out. The
        // point is that the FFMA loop underneath carries NO per-iteration
        // predicate — it is 256*N_COLS unrolled ops and a `ncols` compare in
        // each of them is pure overhead on every full group, which is all of
        // them for these models (N is a multiple of 128, and 128 is a multiple
        // of 4*N_COLS for both instantiations).
        float wf[N_COLS][16];
        #pragma unroll
        for (int c = 0; c < N_COLS; c++) {
            const unsigned int n = min(n0 + (unsigned int)c, N - 1);
            const float scale = block_scale[(n / FP8_BLOCK) * k_blocks + k_block];
            uint4 b_data = ((const uint4*)(B + (unsigned long long)n * K))[k16];
            const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                unsigned int w32 = b_raw[i];
                wf[c][i * 4 + 0] = s_lut[(w32      ) & 0xFF] * scale;
                wf[c][i * 4 + 1] = s_lut[(w32 >>  8) & 0xFF] * scale;
                wf[c][i * 4 + 2] = s_lut[(w32 >> 16) & 0xFF] * scale;
                wf[c][i * 4 + 3] = s_lut[(w32 >> 24) & 0xFF] * scale;
            }
        }

        // One A load + one convert per element, reused across the N_COLS
        // columns. The per-accumulator operand order is the single-column
        // kernel's, element for element.
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            const __nv_bfloat16* At = A + (unsigned long long)t * a_row_stride;
            uint4 a_lo = ((const uint4*)At)[k16 * 2];
            uint4 a_hi = ((const uint4*)At)[k16 * 2 + 1];
            const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                __nv_bfloat16 lo, hi;
                *(unsigned short*)&lo = (unsigned short)(ar[j] & 0xFFFF);
                *(unsigned short*)&hi = (unsigned short)(ar[j] >> 16);
                const float flo = __bfloat162float(lo);
                const float fhi = __bfloat162float(hi);
                #pragma unroll
                for (int c = 0; c < N_COLS; c++) {
                    // Match scalar rounding: never sum a pair before the accumulator.
                    acc[t][c] += flo * wf[c][j * 2];
                    acc[t][c] += fhi * wf[c][j * 2 + 1];
                }
            }
        }
    }

    // Cross-warp reduction (threads_per_out=64 -> 2 warps/group), per row and
    // column — the single-column kernel's tree, one slot per (group, column).
    __shared__ float smem[MAX_M][N_GROUPS_PER_BLOCK * N_COLS * 2];
    const unsigned int warp_in_out = lane / WARP_SIZE;
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        if ((unsigned int)t >= M) continue;
        #pragma unroll
        for (int c = 0; c < N_COLS; c++) {
            // The clamped duplicate columns of a tail group reduce too — they
            // are simply not stored. Every lane still reaches every `shfl`.
            float a = acc[t][c];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                a += __shfl_down_sync(0xFFFFFFFF, a, offset);
            }
            if (lane % WARP_SIZE == 0) {
                smem[t][(local_out * N_COLS + c) * 2 + warp_in_out] = a;
            }
        }
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            #pragma unroll
            for (int c = 0; c < N_COLS; c++) {
                if ((unsigned int)c >= ncols) continue;
                const unsigned int slot = (local_out * N_COLS + c) * 2;
                float r = smem[t][slot] + smem[t][slot + 1];
                C[(unsigned long long)t * c_row_stride + (n0 + c)] = __float2bfloat16(r);
            }
        }
    }
}

// ── MAX_M=16, N_COLS=2 (contiguous A and C) ────────────────────────────────
extern "C" __global__ void w8a16_gemv_batch16_ncol2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemv_ncol_impl<16, 2>(A, B, block_scale, C, M, N, K, K, N);
}

// ── MAX_M=16, N_COLS=4 (contiguous A and C) ────────────────────────────────
extern "C" __global__ void w8a16_gemv_batch16_ncol4(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemv_ncol_impl<16, 4>(A, B, block_scale, C, M, N, K, K, N);
}

// ── Strided siblings: explicit A and C row pitches, in ELEMENTS ────────────
extern "C" __global__ void w8a16_gemv_batch16_ncol2_strided(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    w8a16_gemv_ncol_impl<16, 2>(A, B, block_scale, C, M, N, K, a_row_stride, c_row_stride);
}

extern "C" __global__ void w8a16_gemv_batch16_ncol4_strided(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    w8a16_gemv_ncol_impl<16, 4>(A, B, block_scale, C, M, N, K, a_row_stride, c_row_stride);
}
