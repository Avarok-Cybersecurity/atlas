// SPDX-License-Identifier: AGPL-3.0-only

// Ada (sm_89) W8A8 block-scaled FP8 prefill GEMM.
//
//   C[M, N] = bf16( sum_g ( sum_{k in g} A[m, k] * B[n, k] ) * a_scale[m, g] * b_scale[n/128, g] )
//
// g walks K in groups of 128. The operands are the ones the dense-FFN W8A8 arm
// already produces (`layers/dense_ffn_w8a8_prefill.rs`):
//   A[M, K]              E4M3, per-token 1x128 groups (`per_token_group_quant_fp8`)
//   a_scale[M, K/128]    FP32
//   B[N, K]              E4M3, 128x128 blocks (checkpoint, or quantized at load)
//   b_scale[N/128, K/128] FP32
//   C[M, N]              BF16
//
// `fp8_gemm_ada_w8a8` is a drop-in for `fp8_gemm_t_blockscaled`
// (kernels/gb10/common/fp8_gemm_t_blockscaled.cu): same arguments, same grid
// (ceil(N/128), ceil(M/64), 1), same 128 threads, and the same preconditions
// the host selector already enforces (K % 128 == 0, N % 128 == 0). Only the
// kernel handle changes (`layers/dense_ffn_ada_kernels.rs`).
//
// ── MMA AND ACCUMULATION ─────────────────────────────────────────────────
//
// `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`: FP8 operands, FP32
// accumulators. sm_89 is the first arch with FP8 mma.sync.
//
// FP32 and not FP16 accumulation, although GeForce Ada runs the FP16-
// accumulate form at twice the rate: both operands are scaled so their block
// maximum is the E4M3 maximum 448, so ONE product can reach 448^2 = 200,704,
// three times FP16's largest finite value (65,504), and a 128-term inner sum
// can reach 2.6e7. FP16 accumulation would therefore need the quantizers to
// target a smaller maximum, giving up E4M3's top binades of range for every
// operand, and its rounding (11-bit significand over 128 terms per block) is
// coarser than the FP32 epilogue this arm is defined against. Even at the
// halved rate, FP8 with FP32 accumulation is twice BF16 mma.sync with FP32
// accumulation on the same part, so the prefill gain survives the choice.
//
// Two-level accumulation, as in the gb10 kernel: `inner` takes the 4 MMAs of
// one 128-wide K group unscaled, then folds into `outer` as
// `outer += inner * (a_scale * b_scale)` and resets.
//
// ── WHAT IS DIFFERENT FROM THE GB10 KERNEL ───────────────────────────────
//
// The gb10 kernel stages K=32 per step with `cp.async.wait_group 0` and a
// barrier after every step (load latency is exposed once per 16 MMAs per
// warp), loads every fragment with 4-byte LDS from a 32-byte row stride
// (2-way bank conflicts on B), and gives each warp a 16x128 tile, so every
// MMA needs a fresh B fragment from shared memory. GB10's own receipt for it
// is a FLAT ~14 TFLOP/s from M=128 to 2048 (`w8a8_prefill_max_m_*` in
// crates/avarok-kernels/src/target_defaults.rs). Here:
//   * K=64 per stage, THREE stages in flight (`cp.async.cg`, wait_group 1),
//     one barrier per two k32 steps;
//   * 64-byte smem rows with a 16-byte XOR swizzle (chunk ^ ((row >> 1) & 3)),
//     so the 8 rows one `ldmatrix` phase reads land in 8 distinct 16-byte
//     bank groups: conflict-free, no padding, 36 KB of static shared memory
//     (under the 48 KB static ceiling sm_89 has without an opt-in);
//   * `ldmatrix.x4` for every A and B fragment (the b16 view of 8x16-byte
//     rows is exactly the e4m3 m16n8k32 fragment layout);
//   * a 2x2 warp grid with 32x64 warp tiles: 2 A fragments and 8 B
//     fragments feed 16 MMAs per k32 step (the gb10 tile feeds 16 MMAs from
//     1 A and 16 B fragments).
//
// ── NUMERICS: bit-identical to the gb10 kernel ───────────────────────────
//
// Each accumulator tile sees the same MMA instructions on the same operand
// bytes in the same K order, folds at the same 128-wide group boundaries with
// the same `a_scale * b_scale` product, and is rounded to BF16 once with
// round-to-nearest. The multiply and add of the fold are spelled `__fmul_rn`
// / `__fadd_rn`, which is what the gb10 kernel's `+=` compiles to under the
// tree-wide `--fmad=false`. So the swap changes speed, not output bits.
//
// EXPECTED, NOT MEASURED: prefill on a 4060 Ti is compute-bound, and the FP8
// mma.sync rate with FP32 accumulation is twice the BF16 rate, so a W8A8
// prefill GEMM that reaches the same fraction of peak as the BF16 one is up
// to 2x faster per projection. The per-token activation quantization is extra
// memory traffic this arm pays and W8A16 does not; the card measurement
// decides whether a `w8a8_prefill_max_m_*` ceiling is needed on Ada as on GB10.

#include <cuda_bf16.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 890)
#error "fp8_gemm_ada.cu needs sm_89 or newer (mma.sync m16n8k32 e4m3)"
#endif

#define FMA_BM 64       // CTA rows (tokens)
#define FMA_BN 128      // CTA cols (output features), one 128-row weight scale block
#define FMA_BK 64       // K bytes per pipeline stage = two k32 MMA steps
#define FMA_STAGES 3
#define FMA_THREADS 128 // 4 warps, 2 (M) x 2 (N)
#define FMA_KGROUP 128  // scale group along K
#define FMA_WM 32       // warp tile rows
#define FMA_WN 64       // warp tile cols
#define FMA_MT (FMA_WM / 16) // m16 tiles per warp = 2
#define FMA_NT (FMA_WN / 8)  // n8 tiles per warp = 8

// Byte offset of 16-byte chunk `c` (0..3) of row `r` in a 64-byte-row tile.
__device__ __forceinline__ unsigned int fma_swz(unsigned int r, unsigned int c) {
    return r * FMA_BK + ((c ^ ((r >> 1) & 3u)) << 4);
}

__device__ __forceinline__ void fma_cp_async16(unsigned int dst, const void* src, bool pred) {
    const unsigned int bytes = pred ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}

__device__ __forceinline__ void fma_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void fma_wait_one_pending() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ void fma_ldmatrix_x4(unsigned int addr, unsigned int& r0, unsigned int& r1,
                                                unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void fma_mma_e4m3(float* acc, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Issue one stage's cp.async copies: A rows [cta_m, cta_m+64) and B rows
// [cta_n, cta_n+128), K bytes [k0, k0+64). Out-of-range rows are zero-filled
// (src-size 0) from a clamped, valid address.
__device__ __forceinline__ void fma_load_stage(unsigned int sA, unsigned int sB, const unsigned char* __restrict__ A,
                                               const unsigned char* __restrict__ B, unsigned int M, unsigned int N,
                                               unsigned int K, unsigned int cta_m, unsigned int cta_n,
                                               unsigned int k0) {
    const unsigned int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < (FMA_BM * FMA_BK / 16) / FMA_THREADS; i++) { // 2
        const unsigned int idx = tid + i * FMA_THREADS;
        const unsigned int r = idx >> 2, c = idx & 3u;
        const unsigned int gr = cta_m + r;
        const bool ok = gr < M;
        const unsigned char* src = A + (unsigned long long)(ok ? gr : 0u) * K + k0 + c * 16;
        fma_cp_async16(sA + fma_swz(r, c), src, ok);
    }
#pragma unroll
    for (int i = 0; i < (FMA_BN * FMA_BK / 16) / FMA_THREADS; i++) { // 4
        const unsigned int idx = tid + i * FMA_THREADS;
        const unsigned int r = idx >> 2, c = idx & 3u;
        const unsigned int gn = cta_n + r;
        const bool ok = gn < N;
        const unsigned char* src = B + (unsigned long long)(ok ? gn : 0u) * K + k0 + c * 16;
        fma_cp_async16(sB + fma_swz(r, c), src, ok);
    }
}

extern "C" __global__ void __launch_bounds__(FMA_THREADS)
    fp8_gemm_ada_w8a8(const unsigned char* __restrict__ A, const float* __restrict__ a_scale,
                      const unsigned char* __restrict__ B, const float* __restrict__ b_scale,
                      __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {
    __shared__ __align__(128) unsigned char smem_a[FMA_STAGES][FMA_BM * FMA_BK];
    __shared__ __align__(128) unsigned char smem_b[FMA_STAGES][FMA_BN * FMA_BK];

    const unsigned int cta_n = blockIdx.x * FMA_BN;
    const unsigned int cta_m = blockIdx.y * FMA_BM;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warp_m = (warp & 1u) * FMA_WM;
    const unsigned int warp_n = (warp >> 1) * FMA_WN;
    const unsigned int group = lane >> 2; // fragment row / col within an 8-tile
    const unsigned int tig = lane & 3u;   // thread in group

    const unsigned int k_groups = K / FMA_KGROUP;
    const unsigned int n_block = cta_n / FMA_KGROUP; // BN == 128: one weight scale row per CTA
    const unsigned int num_kt = K / FMA_BK;

    // ldmatrix row-address roles: lanes 8j..8j+7 supply the 8 row addresses of
    // matrix j. For A the four matrices are (rows 0-7 | 8-15) x (k 0-15 | 16-31)
    // in a0..a3 order; for B, (k 0-15 | 16-31) x (n-tile t | t+1).
    const unsigned int ld_mat = lane >> 3;
    const unsigned int ld_row = lane & 7u;
    const unsigned int a_row_off = ld_row + (ld_mat & 1u) * 8u;
    const unsigned int a_chunk_off = ld_mat >> 1;
    const unsigned int b_row_off = ld_row + (ld_mat >> 1) * 8u;
    const unsigned int b_chunk_off = ld_mat & 1u;

    float inner[FMA_MT][FMA_NT][4];
    float outer[FMA_MT][FMA_NT][4];
#pragma unroll
    for (int mt = 0; mt < FMA_MT; mt++) {
#pragma unroll
        for (int nt = 0; nt < FMA_NT; nt++) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                inner[mt][nt][i] = 0.0f;
                outer[mt][nt][i] = 0.0f;
            }
        }
    }

    const unsigned int sA0 = (unsigned int)__cvta_generic_to_shared(&smem_a[0][0]);
    const unsigned int sB0 = (unsigned int)__cvta_generic_to_shared(&smem_b[0][0]);

    // Prologue: stages 0 and 1 in flight. A commit is issued even when a stage
    // has nothing to load, so `wait_group 1` always counts the same groups.
    fma_load_stage(sA0, sB0, A, B, M, N, K, cta_m, cta_n, 0);
    fma_commit();
    if (num_kt > 1) {
        fma_load_stage(sA0 + FMA_BM * FMA_BK, sB0 + FMA_BN * FMA_BK, A, B, M, N, K, cta_m, cta_n, FMA_BK);
    }
    fma_commit();

    for (unsigned int kt = 0; kt < num_kt; kt++) {
        // Groups committed so far: kt + 2. With at most one pending, stage kt
        // has landed. The barrier also retires every warp's reads of the stage
        // the next load overwrites ((kt + 2) % 3 == (kt - 1) % 3).
        fma_wait_one_pending();
        __syncthreads();
        if (kt + 2 < num_kt) {
            const unsigned int s = (kt + 2) % FMA_STAGES;
            fma_load_stage(sA0 + s * (FMA_BM * FMA_BK), sB0 + s * (FMA_BN * FMA_BK), A, B, M, N, K, cta_m, cta_n,
                           (kt + 2) * FMA_BK);
        }
        fma_commit();

        const unsigned int s = kt % FMA_STAGES;
        const unsigned int sA = sA0 + s * (FMA_BM * FMA_BK);
        const unsigned int sB = sB0 + s * (FMA_BN * FMA_BK);
#pragma unroll
        for (unsigned int ks = 0; ks < FMA_BK / 32; ks++) {
            unsigned int a[FMA_MT][4];
#pragma unroll
            for (int mt = 0; mt < FMA_MT; mt++) {
                const unsigned int r = warp_m + mt * 16 + a_row_off;
                fma_ldmatrix_x4(sA + fma_swz(r, ks * 2 + a_chunk_off), a[mt][0], a[mt][1], a[mt][2], a[mt][3]);
            }
#pragma unroll
            for (int np = 0; np < FMA_NT / 2; np++) {
                unsigned int b0, b1, b2, b3;
                const unsigned int r = warp_n + np * 16 + b_row_off;
                fma_ldmatrix_x4(sB + fma_swz(r, ks * 2 + b_chunk_off), b0, b1, b2, b3);
#pragma unroll
                for (int mt = 0; mt < FMA_MT; mt++) {
                    fma_mma_e4m3(inner[mt][np * 2], a[mt], b0, b1);
                    fma_mma_e4m3(inner[mt][np * 2 + 1], a[mt], b2, b3);
                }
            }
        }

        // Every second stage closes a 128-wide K group: fold and reset.
        if (kt & 1u) {
            const unsigned int g = kt >> 1;
            const float bs = __ldg(b_scale + (unsigned long long)n_block * k_groups + g);
#pragma unroll
            for (int mt = 0; mt < FMA_MT; mt++) {
                const unsigned int r0 = cta_m + warp_m + mt * 16 + group;
                const unsigned int r1 = r0 + 8;
                const float as0 = r0 < M ? __ldg(a_scale + (unsigned long long)r0 * k_groups + g) : 0.0f;
                const float as1 = r1 < M ? __ldg(a_scale + (unsigned long long)r1 * k_groups + g) : 0.0f;
                const float s0 = __fmul_rn(as0, bs);
                const float s1 = __fmul_rn(as1, bs);
#pragma unroll
                for (int nt = 0; nt < FMA_NT; nt++) {
                    outer[mt][nt][0] = __fadd_rn(outer[mt][nt][0], __fmul_rn(inner[mt][nt][0], s0));
                    outer[mt][nt][1] = __fadd_rn(outer[mt][nt][1], __fmul_rn(inner[mt][nt][1], s0));
                    outer[mt][nt][2] = __fadd_rn(outer[mt][nt][2], __fmul_rn(inner[mt][nt][2], s1));
                    outer[mt][nt][3] = __fadd_rn(outer[mt][nt][3], __fmul_rn(inner[mt][nt][3], s1));
                    inner[mt][nt][0] = 0.0f;
                    inner[mt][nt][1] = 0.0f;
                    inner[mt][nt][2] = 0.0f;
                    inner[mt][nt][3] = 0.0f;
                }
            }
        }
    }

    // Epilogue: accumulator element pairs (0,1) and (2,3) are adjacent
    // columns of rows `group` and `group + 8`; one bf16x2 store each.
#pragma unroll
    for (int mt = 0; mt < FMA_MT; mt++) {
        const unsigned int r0 = cta_m + warp_m + mt * 16 + group;
        const unsigned int r1 = r0 + 8;
#pragma unroll
        for (int nt = 0; nt < FMA_NT; nt++) {
            const unsigned int col = cta_n + warp_n + nt * 8 + tig * 2;
            if (col < N) {
                if (r0 < M) {
                    *reinterpret_cast<__nv_bfloat162*>(C + (unsigned long long)r0 * N + col) =
                        __floats2bfloat162_rn(outer[mt][nt][0], outer[mt][nt][1]);
                }
                if (r1 < M) {
                    *reinterpret_cast<__nv_bfloat162*>(C + (unsigned long long)r1 * N + col) =
                        __floats2bfloat162_rn(outer[mt][nt][2], outer[mt][nt][3]);
                }
            }
        }
    }
}
