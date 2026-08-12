// SPDX-License-Identifier: AGPL-3.0-only

// Atlas native ternary Q2_0 decode GEMV — CANDIDATE B (vectorized codes + smem A-stage).
//
//   out[m,n] = sum_k A[m,k] * (code(n,k) - 1) * d(n, k/group)
//
// Drop-in numeric twin of `q2_0_gemv.cu` (value == (code-1)*d in FP32, FP32
// accumulate, warp-shuffle reduce) but rebuilt around the two loads that make
// `w4a16_gemv.cu` fast — WIDE COALESCED reads — plus a reused shared-memory
// activation tile. Kept keep-packed: the 2-bit weight never expands to BF16.
//
// ── Load vectorization ────────────────────────────────────────────────────
//   • Weight codes: each lane reads ONE `uint32` (4 packed code-bytes) with a
//     single 4-byte load. Because the on-disk packing is low-bits-first and the
//     bytes are contiguous, a little-endian uint32 places code j at bits
//     [2j, 2j+1] for j = 0..15 → 16 ternary codes unpacked from one word with
//     `(w >> 2j) & 3`. The `group/16` words of a block are read by adjacent
//     lanes as one sector-perfect 32-byte (g128) / 16-byte (g64) transaction.
//   • Activation: staged once per block-tile into `s_A` by all 256 threads via
//     coalesced `uint4` (128-bit) loads, then each lane pulls its 16 BF16 from
//     smem as 2× `uint4`. The BF16 activation is NEVER re-read from global per
//     output row — all 8 warps in the CTA reuse the same staged tile (the extra
//     lever §7: A traffic is a larger fraction of DRAM for 2-bit weights).
//
// ── Thread mapping / occupancy ────────────────────────────────────────────
//   1 warp (32 lanes) per output row; 8 rows per 256-thread CTA (no cross-warp
//   smem reduction, no per-row __syncthreads). A warp walks the row's blocks
//   `TILE_K/group` at a time (4 blocks g128 / 8 blocks g64 → 512 K per step),
//   lane l owning code-word `l % (group/16)` of block `l / (group/16)`. All 32
//   lanes stay active on the FFN shapes (K=5120 → 40 blocks/8-warp… 10 steps of
//   4 blocks, no idle lanes; K=17408 → 136 blocks, 34 steps). Grid (ceil(N/8),
//   1, 1), block (256,1,1).
//
// Numerics gate: per-lane accumulation is ascending-k within (code-1)*d, FP32;
// the only reorder is the warp-shuffle tail already inside the validated
// rel-err 0.0023 kernel. No LUT, no (code-1) reorder before *d.

#include <cuda_bf16.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)  // 256
#define TILE_K 512                                // staged A elements / row / step
#define TILE_U4 (TILE_K / 8)                      // 64 uint4 per staged row
#define MAX_M 8

// Little-endian IEEE fp16 (2 bytes) -> f32. Bit-identical scale read to
// `q2_rd_f16` in q2_0_gemv.cu / dq_rd_f16 in dequant_gguf_bf16.cu.
__device__ __forceinline__ float q2v_rd_f16(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(bits));
}

// Read 4 packed code-bytes as a little-endian uint32 WITHOUT assuming 4-byte
// alignment. Q2_0 blocks carry a 2-byte fp16 `d` prefix and a 34-byte (g128)
// stride, so `blk+2+...` is only 2-byte aligned — a raw `*(uint32*)` load there
// faults (CUDA_ERROR_MISALIGNED_ADDRESS). Byte-assemble instead.
__device__ __forceinline__ unsigned int q2v_rd_u32(const unsigned char* p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) |
           ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

// Unpack a uint4 (8 packed BF16) into 8 fp32 lanes, low-half first (matches the
// `*(unsigned short*)&bf = raw & 0xFFFF` convention in w4a16_gemv.cu).
__device__ __forceinline__ void q2v_unpack8(uint4 v, float* out) {
    const unsigned int w[4] = {v.x, v.y, v.z, v.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        __nv_bfloat16 lo, hi;
        *(unsigned short*)&lo = (unsigned short)(w[i] & 0xFFFF);
        *(unsigned short*)&hi = (unsigned short)(w[i] >> 16);
        out[i * 2]     = __bfloat162float(lo);
        out[i * 2 + 1] = __bfloat162float(hi);
    }
}

// ── W2A16 GEMV (M=1) ───────────────────────────────────────────────────────
//
// C[n] = sum_k A[k] * (code(n,k) - 1) * d(n, k/group)
extern "C" __global__ void q2_0_gemv_vec(
    const __nv_bfloat16* __restrict__ A,       // [1, K] BF16
    const unsigned char* __restrict__ B,       // [N, K] packed Q2_0
    __nv_bfloat16* __restrict__ C,             // [1, N] BF16
    unsigned int N,
    unsigned int K,
    unsigned int group)                        // 128 or 64
{
    const unsigned int warp = threadIdx.x / WARP_SIZE;   // 0..7 → output row
    const unsigned int lane = threadIdx.x % WARP_SIZE;   // 0..31
    const unsigned int n = blockIdx.x * WARPS_PER_BLOCK + warp;

    const unsigned int blocks_per_row = K / group;
    const unsigned int block_bytes = 2u + group / 4u;    // 34 (g128) / 18 (g64)
    const unsigned int upb = group / 16u;                // code-words / block: 8 or 4
    const unsigned int bpi = WARP_SIZE / upb;            // blocks / warp-step: 4 or 8
    const unsigned int block_in_step = lane / upb;       // which block this lane owns
    const unsigned int jg = lane % upb;                  // which code-word within it

    // Base of this warp's weight row (guarded: OOB warps still stage + sync).
    const unsigned char* row =
        (n < N) ? B + (unsigned long long)n * blocks_per_row * block_bytes : B;

    __shared__ __align__(16) __nv_bfloat16 s_A[TILE_K];

    float acc = 0.0f;

    for (unsigned int tb0 = 0; tb0 < blocks_per_row; tb0 += bpi) {
        const unsigned int base_k = tb0 * group;

        // Coalesced uint4 stage of the A tile (all 256 threads cooperate).
        for (unsigned int u = threadIdx.x; u < TILE_U4; u += BLOCK_SIZE) {
            const unsigned int k = base_k + u * 8u;
            uint4 val;
            if (k < K) {
                val = *((const uint4*)A + k / 8u);
            } else {
                val.x = val.y = val.z = val.w = 0u;
            }
            *((uint4*)s_A + u) = val;
        }
        __syncthreads();

        if (n < N) {
            const unsigned int b = tb0 + block_in_step;
            if (b < blocks_per_row) {
                const unsigned char* blk = row + (unsigned long long)b * block_bytes;
                const float d = q2v_rd_f16(blk);
                const unsigned int codes = q2v_rd_u32(blk + 2 + jg * 4u);

                const unsigned int kl = block_in_step * group + jg * 16u;  // tile-local k
                float a[16];
                q2v_unpack8(*(const uint4*)(s_A + kl),      a);
                q2v_unpack8(*(const uint4*)(s_A + kl + 8u), a + 8);

                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const int code = (int)((codes >> (2 * j)) & 3u);
                    acc += a[j] * (float)(code - 1) * d;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    }
    if (lane == 0 && n < N) {
        C[n] = __float2bfloat16(acc);
    }
}

// ── W2A16 batched GEMV (M=1..8) ────────────────────────────────────────────
//
// C[m, n] = sum_k A[m, k] * (code(n,k) - 1) * d(n, k/group)
//
// Reads each weight word ONCE, dequants to `wv=(code-1)*d`, and MAC's into M
// independent FP32 accumulators — amortizing the 2-bit weight read across the
// batch (same guarantee as w4a16_gemv_batchm: bit-consistent with M×(M=1)).
// All M activation rows are staged in smem per step.
extern "C" __global__ void q2_0_gemv_vec_batchm(
    const __nv_bfloat16* __restrict__ A,       // [M, K] BF16
    const unsigned char* __restrict__ B,       // [N, K] packed Q2_0
    __nv_bfloat16* __restrict__ C,             // [M, N] BF16
    unsigned int N,
    unsigned int K,
    unsigned int group,
    unsigned int M)
{
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int n = blockIdx.x * WARPS_PER_BLOCK + warp;

    const unsigned int blocks_per_row = K / group;
    const unsigned int block_bytes = 2u + group / 4u;
    const unsigned int upb = group / 16u;
    const unsigned int bpi = WARP_SIZE / upb;
    const unsigned int block_in_step = lane / upb;
    const unsigned int jg = lane % upb;

    const unsigned char* row =
        (n < N) ? B + (unsigned long long)n * blocks_per_row * block_bytes : B;

    __shared__ __align__(16) __nv_bfloat16 s_A[MAX_M * TILE_K];

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (unsigned int tb0 = 0; tb0 < blocks_per_row; tb0 += bpi) {
        const unsigned int base_k = tb0 * group;

        // Stage all M rows' tiles, coalesced per row.
        const unsigned int total_u4 = TILE_U4 * M;
        for (unsigned int u = threadIdx.x; u < total_u4; u += BLOCK_SIZE) {
            const unsigned int mm = u / TILE_U4;
            const unsigned int uu = u % TILE_U4;
            const unsigned int k = base_k + uu * 8u;
            uint4 val;
            if (k < K) {
                val = *((const uint4*)(A + (unsigned long long)mm * K) + k / 8u);
            } else {
                val.x = val.y = val.z = val.w = 0u;
            }
            *((uint4*)s_A + mm * TILE_U4 + uu) = val;
        }
        __syncthreads();

        if (n < N) {
            const unsigned int b = tb0 + block_in_step;
            if (b < blocks_per_row) {
                const unsigned char* blk = row + (unsigned long long)b * block_bytes;
                const float d = q2v_rd_f16(blk);
                const unsigned int codes = q2v_rd_u32(blk + 2 + jg * 4u);

                float wv[16];
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    wv[j] = (float)((int)((codes >> (2 * j)) & 3u) - 1) * d;
                }

                const unsigned int kl = block_in_step * group + jg * 16u;
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if ((unsigned int)m >= M) continue;
                    const __nv_bfloat16* sa = s_A + (unsigned int)m * TILE_K + kl;
                    float a[16];
                    q2v_unpack8(*(const uint4*)sa,      a);
                    q2v_unpack8(*(const uint4*)(sa + 8u), a + 8);
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) acc[m] += a[j] * wv[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if ((unsigned int)m >= M) continue;
        float v = acc[m];
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xFFFFFFFF, v, off);
        }
        if (lane == 0 && n < N) {
            C[(unsigned long long)m * N + n] = __float2bfloat16(v);
        }
    }
}
