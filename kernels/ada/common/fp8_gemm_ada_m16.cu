// SPDX-License-Identifier: AGPL-3.0-only

// Ada (sm_89) W8A16 tensor-core GEMM with a 16-row M tile: the dense FFN's
// FP8-E4M3 block-scaled projections over M = 1..16 BF16 activation rows on ONE
// pass over the weight (port row `ffn_fp8_m16_tc`, off by default;
// `layers/dense_ffn/fp8_m16_tc.rs`).
//
//   C[t, n] = bf16( sum_g ( sum_{k in g} A[t, k] * E4M3(B[n, k]) ) * block_scale[n/128, g] ),  t < M
//
// g walks K in 128-wide scale blocks. A is [M, K] BF16 row-major, B is [N, K]
// E4M3 bytes, block_scale is [ceil(N/128), K/128] FP32 (the format
// `fp8_gemv_ada.cu` reads), C is [M, N] BF16 row-major.
//
// Entry points (M is an ARGUMENT, 1..16; one instantiation serves every M):
//
//   fp8_gemm_ada_m16        (A, B, block_scale, C, M, N, K)
//                           grid (ceil(N/32), 1, 1), block 128
//   fp8_gemm_ada_m16_dual   (A, B1, B1_scale, C1, B2, B2_scale, C2, M, N, K)
//                           gate and up of the same rows; blockIdx.z picks
//                           the weight. grid (ceil(N/32), 1, 2), block 128
//   fp8_gemm_ada_m16_ahead1, fp8_gemm_ada_m16_ahead4
//                           the plain entry with 1 or 4 weight blocks in
//                           flight per thread instead of 2. Timing arms of
//                           `examples/fp8_ada_m16_tc_microtest.rs` only; the
//                           host launches neither. Same output bits.
//
// Preconditions the host selector enforces: K % 128 == 0, 1 <= M <= 16, A and
// B 16-byte aligned. N is arbitrary (columns past N are neither read nor
// written).
//
// ── WHY ──────────────────────────────────────────────────────────────────
//
// A decode step over 5 to 8 streams, and an MTP verify over 8 streams (16
// rows), run the dense FFN at 5 to 16 rows. On the RTX 4060 Ti 16 GB neither
// kernel the tree has for those widths holds the weight-bandwidth floor
// (4B gate/up, N = 9216, K = 2560, 23.6 MB: 91 us at the measured 260 GB/s):
//
//   fp8_gemv_ada_batchm (FP32 FMAs, M accumulators)   104 us at M = 1, 97 at
//       M = 4, 184 at M = 8 (fp8_ada_batchm_microtest, 2026-11-13): compute
//       bound above 4 rows, and it stops at 8;
//   fp8_gemm_ada_w8a8 (E4M3 mma, 64-row padded tile)  143 us flat from 1 to
//       64 rows (L07 linear_t_sweep), and it quantizes the activations to
//       E4M3 (W8A8), which no narrower batch does.
//
// The second line is also the reason for the operand choice here. An FP8
// `mma.sync` (m16n8k32, e4m3 x e4m3) needs E4M3 ACTIVATIONS: a 3-bit mantissa
// where BF16 carries 7, so it is a different arithmetic (W8A8), not a faster
// way to compute this one, and it needs a quantizing pass over the
// activations before every projection. So this kernel keeps the activations
// BF16, exactly as stored, and widens the WEIGHTS in registers: E4M3 -> BF16
// is exact (3 mantissa bits and a 4-bit exponent fit in 7 and 8), and a
// 16-row tile is one m16n8k16 per 128 weight bytes with no padded rows at
// M = 16 and half the tile idle at M = 8. That is 1/4 of the tensor work the
// 64-row W8A8 tile does per weight byte, which is what leaves the pass
// bandwidth bound.
//
// ── GEOMETRY ─────────────────────────────────────────────────────────────
//
// CTA = 4 warps = [16 M x 32 N]; each warp owns one m16n8 tile (8 output
// columns) and walks all of K.
//
// WEIGHTS go global -> registers, never through shared memory. Per 64 K the
// four lanes of an MMA group load the four 16-byte chunks of ONE weight row
// (`__ldcs`, the streaming hint the decode GEMVs use), so a warp instruction
// reads 8 rows x 64 bytes. Each thread keeps the next 2 scale blocks (4
// loads, 64 bytes) in flight in a register ring, the same memory-level
// parallelism per warp as `fp8_gemv_ada`'s two 16-byte loads times two CTAs.
// A thread therefore holds K indices 16t .. 16t+15 of a 64-wide step (t =
// thread in group), which is NOT the m16n8k16 fragment's nominal K order (2t,
// 2t+1, 2t+8, 2t+9). The MMA sums over its 16 K slots whatever K each slot
// carries, so the kernel relabels: MMA i of a step (i = 0..3) takes K = 16t +
// 4i .. 4i+3 from every thread, for the weights and for the activations
// alike. Every K is used exactly once.
//
// ACTIVATIONS are staged through shared memory, one 128-wide scale block per
// stage (16 rows x 128 BF16 = 4 KB), three stages, `cp.async.cg`, one barrier
// per block. Rows >= M are zero-filled once and never loaded or stored. The
// row pitch is 272 bytes (256 + 16) so that the 32 lanes of a 128-bit
// fragment load spread 4-way over the 32 banks, the minimum for 16-byte
// reads. 13,056 bytes of static shared memory: 7 CTAs an SM by shared memory.
//
// No ldmatrix (the relabelled K order is not ldmatrix's), no TMA.
//
// ── NUMERICS: REASSOCIATED, NOT BIT-IDENTICAL TO THE GEMV ────────────────
//
// `fp8_gemv_ada` reduces each output in FP32 in strict K order within a lane
// and multiplies every weight by its block scale first. This kernel, like
// `kernels/hopper/common/w8a16_gemm_m16.cu` and the tile GEMMs:
//   * accumulates UNSCALED products, 16 at a time in the tensor core's own
//     order, 8 MMAs per scale block, into an FP32 `inner`;
//   * folds once per block, `outer = outer + inner * block_scale` (separate
//     round-to-nearest multiply and add, spelled `__fmul_rn` / `__fadd_rn`);
//   * rounds `outer` to BF16 once.
// The contract is the M16 tensor-core one (`layers/dense_ffn_m16_tc_oracle.rs`,
// `within_m16_tc_budget`): at most 2 BF16 ULP from the GEMV, or an absolute
// error under the FP32 accumulation floor for a cancelled output.
// `examples/fp8_ada_m16_tc_microtest.rs` measures it against `fp8_gemv_ada`
// and against a host FP64 reference at every M in 1..16.
//
// ROW INVARIANCE. Row t's bits do not depend on M or on the other rows: each
// output element is its own row's dot product, in the same block order and
// the same MMA order at every M. So a 16-row launch equals 16 one-row launches
// of this kernel byte for byte (the microtest checks that too), and a verify
// row equals the same token decoded in any multi-sequence batch on this route.
//
// The dequant is `cvt.rn.f16x2.e4m3x2` then FP16 -> FP32 -> BF16, all exact
// for the 254 finite E4M3 bytes. The two NaN bytes (0x7F, 0xFF) stay NaN; no
// Atlas quantizer or block-FP8 checkpoint emits them.
//
// ── ADDRESSING ───────────────────────────────────────────────────────────
//
// CUDA 13.3's ptxas miscompiled a kernel of this family that walked several
// activation row pointers with a runtime stride (`fgb_act_word` in
// fp8_gemv_ada_batchm.cu has the record). Here every thread has ONE weight
// row pointer, stepped by a constant, and the activation and output rows are
// addressed by unsigned 32-bit element indices added to the base (M * K and
// M * N are below 2^32 for any shape the host admits), so no 64-bit row
// recurrence exists to fold.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 890)
#error "fp8_gemm_ada_m16.cu needs sm_89 or newer (cvt.rn.f16x2.e4m3x2)"
#endif

#define FM16_M_TILE 16
#define FM16_N_TILE 32   // CTA columns: 4 warps x one n8 MMA tile
#define FM16_THREADS 128 // 4 warps
#define FM16_FP8_BLOCK 128
#define FM16_STAGES 3     // activation stages, one scale block each
#define FM16_A_PITCH 272u // bytes per staged activation row: 128 BF16 + 16 pad
#define FM16_A_STAGE (FM16_M_TILE * FM16_A_PITCH) // 4,352 bytes

__device__ __forceinline__ void fm16_cp_async16(unsigned int smem_addr, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void fm16_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void fm16_wait_one_pending() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}

// Two E4M3 bytes (low byte = lower K index) -> one BF16x2 register in the same
// halves, which is the m16n8k16 B fragment's packing. Exact for every finite
// byte.
__device__ __forceinline__ unsigned int fm16_dequant_pair(unsigned int raw16) {
    unsigned int h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((unsigned short)raw16));
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
    const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
    return *reinterpret_cast<const unsigned int*>(&b);
}

// One 16-byte weight load, or zeros without touching memory when `ok` is
// false (a column past N, a block past K).
__device__ __forceinline__ uint4 fm16_load_b(const uint4* p, bool ok) {
    if (ok) {
        return __ldcs(p);
    }
    return make_uint4(0u, 0u, 0u, 0u);
}

// The whole GEMM for one CTA. `AHEAD` is the number of scale blocks of weights
// each thread keeps in flight (2 loads of 16 bytes a block).
template <int AHEAD>
__device__ __forceinline__ void fm16_gemm(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B,
                                          const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C,
                                          unsigned int M, unsigned int N, unsigned int K) {
    __shared__ __align__(16) unsigned char smem_a[FM16_STAGES * FM16_A_STAGE];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int g = lane >> 2; // MMA group: A and C rows g and g + 8, B column g
    const unsigned int t = lane & 3u; // thread in group
    const unsigned int n0 = blockIdx.x * FM16_N_TILE + warp * 8u; // this warp's 8 output columns
    const unsigned int k_blocks = K / FM16_FP8_BLOCK;
    const unsigned int rows = M < FM16_M_TILE ? M : FM16_M_TILE;

    // ── Activation staging roles ──
    // A stage is 16 rows x 128 BF16 = 256 chunks of 16 bytes, two per thread:
    // rows `ra` and `ra + 8`, the 8 BF16 at element `ach * 8` of the block.
    const unsigned int ra = tid >> 4;   // 0..7
    const unsigned int ach = tid & 15u; // 0..15
    const bool a_lo = ra < rows;
    const bool a_hi = ra + 8u < rows;
    // 32-bit element indices from A (see ADDRESSING).
    const unsigned int a_idx_lo = ra * K + ach * 8u;
    const unsigned int a_idx_hi = (ra + 8u) * K + ach * 8u;
    const unsigned int s_base = (unsigned int)__cvta_generic_to_shared(smem_a);
    const unsigned int s_lo = ra * FM16_A_PITCH + ach * 16u;
    const unsigned int s_hi = (ra + 8u) * FM16_A_PITCH + ach * 16u;

    // Rows >= M: zero once in every stage. No load ever writes them.
#pragma unroll
    for (int s = 0; s < FM16_STAGES; s++) {
        if (!a_lo) {
            *reinterpret_cast<uint4*>(smem_a + s * FM16_A_STAGE + s_lo) = make_uint4(0u, 0u, 0u, 0u);
        }
        if (!a_hi) {
            *reinterpret_cast<uint4*>(smem_a + s * FM16_A_STAGE + s_hi) = make_uint4(0u, 0u, 0u, 0u);
        }
    }

    // Stage scale block `kb` of the activations into `stage`.
    auto stage_a = [&](unsigned int stage, unsigned int kb) {
        const unsigned int dst = s_base + stage * FM16_A_STAGE;
        const unsigned int k0 = kb * FM16_FP8_BLOCK;
        if (a_lo) {
            fm16_cp_async16(dst + s_lo, A + (a_idx_lo + k0));
        }
        if (a_hi) {
            fm16_cp_async16(dst + s_hi, A + (a_idx_hi + k0));
        }
    };

    // ── Weight roles ──
    // Thread (g, t) reads weight row n0 + g: per 64 K, the 16 bytes at K
    // offset 16 t. A block is two such loads.
    const unsigned int bn = n0 + g;
    const bool b_live = bn < N;
    const uint4* bp = reinterpret_cast<const uint4*>(B + (unsigned long long)(b_live ? bn : 0u) * K) + t;

    // The CTA's columns start at a multiple of 8 and a warp's tile is 8 wide,
    // so the tile lies in one 128-row scale block.
    const bool tile_live = n0 < N;
    const float* scale_row = block_scale + (unsigned long long)(tile_live ? n0 / FM16_FP8_BLOCK : 0u) * k_blocks;

    float in0 = 0.0f, in1 = 0.0f, in2 = 0.0f, in3 = 0.0f;
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;

    // Prologue: activation stages 0 and 1 in flight (a commit is issued even
    // when there is nothing to load, so `wait_group 1` always counts the same
    // groups), and the first AHEAD weight blocks.
    stage_a(0u, 0u);
    fm16_commit();
    if (k_blocks > 1u) {
        stage_a(1u, 1u);
    }
    fm16_commit();

    uint4 ring[AHEAD][2];
#pragma unroll
    for (int d = 0; d < AHEAD; d++) {
        const bool ok = b_live && (unsigned int)d < k_blocks;
        ring[d][0] = fm16_load_b(bp + (unsigned int)d * 8u, ok);
        ring[d][1] = fm16_load_b(bp + (unsigned int)d * 8u + 4u, ok);
    }

    for (unsigned int kb = 0; kb < k_blocks; kb++) {
        // Groups committed so far: kb + 2; with at most one pending, block
        // kb's stage has landed. The barrier also retires every warp's reads
        // of the stage the next load overwrites ((kb + 2) % 3 == (kb - 1) % 3).
        fm16_wait_one_pending();
        __syncthreads();
        if (kb + 2u < k_blocks) {
            stage_a((kb + 2u) % FM16_STAGES, kb + 2u);
        }
        fm16_commit();

        // This block's weights leave the ring; the block AHEAD further on
        // enters it.
        const uint4 b[2] = {ring[0][0], ring[0][1]};
#pragma unroll
        for (int d = 0; d + 1 < AHEAD; d++) {
            ring[d][0] = ring[d + 1][0];
            ring[d][1] = ring[d + 1][1];
        }
        {
            const unsigned int nb = kb + (unsigned int)AHEAD;
            const bool ok = b_live && nb < k_blocks;
            ring[AHEAD - 1][0] = fm16_load_b(bp + nb * 8u, ok);
            ring[AHEAD - 1][1] = fm16_load_b(bp + nb * 8u + 4u, ok);
        }
        const float sc = tile_live ? __ldg(scale_row + kb) : 0.0f;

        const unsigned char* sa = smem_a + (kb % FM16_STAGES) * FM16_A_STAGE;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            // Rows g and g + 8, the 16 BF16 at K offset 64 h + 16 t.
            const unsigned int off = (unsigned int)h * 128u + t * 32u;
            const uint4 q0 = *reinterpret_cast<const uint4*>(sa + g * FM16_A_PITCH + off);
            const uint4 q1 = *reinterpret_cast<const uint4*>(sa + g * FM16_A_PITCH + off + 16u);
            const uint4 r0 = *reinterpret_cast<const uint4*>(sa + (g + 8u) * FM16_A_PITCH + off);
            const uint4 r1 = *reinterpret_cast<const uint4*>(sa + (g + 8u) * FM16_A_PITCH + off + 16u);
            const unsigned int wg[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            const unsigned int wh[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
            const unsigned int bw[4] = {b[h].x, b[h].y, b[h].z, b[h].w};
#pragma unroll
            for (int i = 0; i < 4; i++) {
                // MMA i: K = 16 t + 4 i + {0, 1} in fragment slot 0 and
                // + {2, 3} in slot 1, for A (rows g, g + 8) and B alike.
                const unsigned int b0 = fm16_dequant_pair(bw[i] & 0xFFFFu);
                const unsigned int b1 = fm16_dequant_pair(bw[i] >> 16);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                             : "=f"(in0), "=f"(in1), "=f"(in2), "=f"(in3)
                             : "r"(wg[2 * i]), "r"(wh[2 * i]), "r"(wg[2 * i + 1]), "r"(wh[2 * i + 1]), "r"(b0), "r"(b1),
                               "f"(in0), "f"(in1), "f"(in2), "f"(in3));
            }
        }

        // The block boundary: fold the unscaled inner sums once, with this
        // block's scale.
        out0 = __fadd_rn(out0, __fmul_rn(in0, sc));
        out1 = __fadd_rn(out1, __fmul_rn(in1, sc));
        out2 = __fadd_rn(out2, __fmul_rn(in2, sc));
        out3 = __fadd_rn(out3, __fmul_rn(in3, sc));
        in0 = 0.0f;
        in1 = 0.0f;
        in2 = 0.0f;
        in3 = 0.0f;
    }

    // Epilogue: accumulator elements 0, 1 are columns n0 + 2t, + 1 of row g;
    // elements 2, 3 the same columns of row g + 8. 32-bit element indices.
    const unsigned int col = n0 + t * 2u;
    if (col < N) {
        const bool second = col + 1u < N;
        if (g < rows) {
            const unsigned int idx = g * N + col;
            C[idx] = __float2bfloat16(out0);
            if (second) {
                C[idx + 1u] = __float2bfloat16(out1);
            }
        }
        if (g + 8u < rows) {
            const unsigned int idx = (g + 8u) * N + col;
            C[idx] = __float2bfloat16(out2);
            if (second) {
                C[idx + 1u] = __float2bfloat16(out3);
            }
        }
    }
}

// ── Entry points ─────────────────────────────────────────────────────────

// C[M, N] = A[M, K] x dequant(B[N, K])^T. Grid (ceil(N/32), 1, 1), block 128.
extern "C" __global__ void __launch_bounds__(FM16_THREADS)
    fp8_gemm_ada_m16(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B,
                     const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C, unsigned int M,
                     unsigned int N, unsigned int K) {
    fm16_gemm<2>(A, B, block_scale, C, M, N, K);
}

// Gate and up of the same rows; blockIdx.z picks the weight and its output.
// Grid (ceil(N/32), 1, 2), block 128.
extern "C" __global__ void __launch_bounds__(FM16_THREADS)
    fp8_gemm_ada_m16_dual(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B1,
                          const float* __restrict__ B1_scale, __nv_bfloat16* __restrict__ C1,
                          const unsigned char* __restrict__ B2, const float* __restrict__ B2_scale,
                          __nv_bfloat16* __restrict__ C2, unsigned int M, unsigned int N, unsigned int K) {
    const bool second = blockIdx.z != 0;
    fm16_gemm<2>(A, second ? B2 : B1, second ? B2_scale : B1_scale, second ? C2 : C1, M, N, K);
}

// Timing arms (the microtest only): 1 and 4 weight blocks in flight.
extern "C" __global__ void __launch_bounds__(FM16_THREADS)
    fp8_gemm_ada_m16_ahead1(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B,
                            const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C, unsigned int M,
                            unsigned int N, unsigned int K) {
    fm16_gemm<1>(A, B, block_scale, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(FM16_THREADS)
    fp8_gemm_ada_m16_ahead4(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B,
                            const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C, unsigned int M,
                            unsigned int N, unsigned int K) {
    fm16_gemm<4>(A, B, block_scale, C, M, N, K);
}
