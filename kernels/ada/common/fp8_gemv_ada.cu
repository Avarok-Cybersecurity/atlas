// SPDX-License-Identifier: AGPL-3.0-only

// Ada (sm_89) FP8-E4M3 W8A16 decode GEMVs, register dequant.
//
// Three entry points, each a drop-in for the gb10 kernel of the same role:
//
//   fp8_gemv_ada             == w8a16_gemv             (w8a16_gemv.cu)
//   fp8_gemv_ada_dual        == w8a16_gemv_dual        (w8a16_gemv_fused.cu)
//   fp8_gemv_ada_silu_input  == w8a16_gemv_silu_input  (w8a16_gemv_fused.cu)
//
// SAME arguments, SAME launch geometry (grid ceil(N/4) x 1 x {1,2}, 256
// threads), SAME weight format: B[N, K] E4M3 bytes and the 128x128 FP32 block
// scale grid block_scale[ceil(N/128), ceil(K/128)] that both native block-FP8
// checkpoints and `quantize_bf16_to_fp8_blockscaled` produce. The host swaps
// the kernel HANDLE and nothing else (`layers/dense_ffn_ada_kernels.rs`).
//
// ── WHY AN ADA-OWNED FILE ────────────────────────────────────────────────
//
// The gb10 kernels decode each byte through a 256-entry FP32 table staged in
// shared memory: 16 data-dependent LDS per 16-byte chunk, plus a 1 KB table
// fill from __constant__ memory (256 distinct addresses, so the constant
// cache serialises it per warp) and a __syncthreads before the first FMA, in
// EVERY CTA. Hopper replaced the gather with `cvt.rn.f16x2.e4m3x2` in its own
// override (kernels/hopper/common/w8a16_gemv_hopper.cuh, #928). Ada inherits
// gb10's common/, not Hopper's, and new cross-hardware symlinks are refused
// (scripts/check_cross_hardware.py rule S1), so the same instruction choice
// lives here as Ada's own source. `cvt.rn.f16x2.e4m3x2` is an sm_89 feature,
// i.e. this is the oldest arch that has it.
//
// What is different from Hopper's tuning, and why:
//   * UNROLL is 2, not 4. Hopper unrolled to put ~32 KB of weight bytes in
//     flight per SM against HBM3 latency. An RTX 4060 Ti streams 288 GB/s over
//     34 SMs, about 8.5 GB/s per SM; at ~600 ns of GDDR6 latency Little's law
//     asks for ~5 KB in flight per SM, which two outstanding 16-byte loads per
//     lane already give at two resident CTAs. Two also keeps the SiLU-input
//     variant's 32 live activation floats from spilling.
//   * Weight loads use the evict-first streaming hint (`__ldcs`). Decode reads
//     every weight byte exactly once per token and the 32 MB L2 cannot hold a
//     4B model's FFN, so caching them only evicts the activations and scales
//     that ARE reused (by every row of the CTA and every CTA of the launch).
//
// EXPECTED, NOT MEASURED: the gain over the gb10 kernel is largest where a CTA
// has the least work to amortise its table fill over, i.e. at small K (the
// 0.8B model's hidden 1024 is one 16-byte chunk per lane). At large K both are
// within reach of the bandwidth roofline on this part; the decode win of the
// FP8 route comes from the bytes, not from this file.
//
// ── NUMERICS: bit-identical to the gb10 kernels ──────────────────────────
//
// Every arithmetic step and its ORDER is the gb10 kernels': lane l of 64 walks
// chunks l, l+64, l+128, ... (the unroll moves LOADS only); within a chunk,
// byte i of word j meets activation 4j+i, each `w = decode(byte) * scale`
// then `acc += a * w` as separate round-to-nearest multiply and add (spelled
// with `__fmul_rn` / `__fadd_rn`, so the result does not depend on the
// tree-wide `--fmad=false` flag that the gb10 kernels rely on for the same
// thing); the same 5-step shfl.down butterfly, the same two-warp add, the same
// single `__float2bfloat16`. FP16 holds every finite E4M3 value exactly and
// FP16 -> FP32 is exact, so `cvt` returns the table's FP32 bits for all 254
// finite bytes. The two E4M3 NaN bytes (0x7F, 0xFF) differ: the table decodes
// them to +-0, `cvt` to NaN. Neither the block-scaled quantizer (SATFINITE)
// nor a block-FP8 checkpoint emits them.
//
// So swapping these in changes the instruction stream and not one output bit,
// which is what lets the route ship without its own accuracy gate: the
// accuracy question belongs to the weight format, not to this kernel.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 890)
#error "fp8_gemv_ada.cu needs sm_89 or newer (cvt.rn.f16x2.e4m3x2)"
#endif

#define FGA_BLOCK 256
#define FGA_N_PER_BLOCK 4
#define FGA_LANES (FGA_BLOCK / FGA_N_PER_BLOCK) // 64 lanes per output row
#define FGA_WARP 32
#define FGA_FP8_BLOCK 128
#define FGA_CHUNK 16                                   // K values per uint4 of FP8 bytes
#define FGA_CHUNKS_PER_SCALE (FGA_FP8_BLOCK / FGA_CHUNK) // 8
#define FGA_UNROLL 2

// Two E4M3 bytes (low byte = lower K index) -> two FP32. Exact for every
// finite byte; see NUMERICS for the two NaN bytes.
__device__ __forceinline__ float2 fga_e4m3x2_to_f32x2(unsigned short raw) {
    unsigned int h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(raw));
    return __half22float2(*reinterpret_cast<const __half2*>(&h2));
}

// Raw BF16 bits -> FP32. Exactly what `__bfloat162float` does.
__device__ __forceinline__ float fga_bf16_lo(unsigned int word) {
    return __uint_as_float(word << 16);
}
__device__ __forceinline__ float fga_bf16_hi(unsigned int word) {
    return __uint_as_float(word & 0xFFFF0000u);
}

// 8 BF16 values packed in a uint4 -> out[0..8], in K order.
__device__ __forceinline__ void fga_unpack_bf16x8(uint4 d, float* out) {
    out[0] = fga_bf16_lo(d.x);
    out[1] = fga_bf16_hi(d.x);
    out[2] = fga_bf16_lo(d.y);
    out[3] = fga_bf16_hi(d.y);
    out[4] = fga_bf16_lo(d.z);
    out[5] = fga_bf16_hi(d.z);
    out[6] = fga_bf16_lo(d.w);
    out[7] = fga_bf16_hi(d.w);
}

// Activation source: one BF16 row A[1, K].
struct FgaActRow {
    const __nv_bfloat16* __restrict__ a;

    __device__ __forceinline__ void chunk(unsigned int k16, float* out) const {
        const uint4* a4 = reinterpret_cast<const uint4*>(a);
        fga_unpack_bf16x8(__ldg(a4 + k16 * 2), &out[0]);
        fga_unpack_bf16x8(__ldg(a4 + k16 * 2 + 1), &out[8]);
    }
};

// Activation source: silu(gate) * up, computed inline exactly as
// `w8a16_gemv_silu_input` computes it: (g / (1 + __expf(-g))) * u in FP32.
struct FgaActSilu {
    const __nv_bfloat16* __restrict__ gate;
    const __nv_bfloat16* __restrict__ up;

    __device__ __forceinline__ void chunk(unsigned int k16, float* out) const {
        const uint4* g4 = reinterpret_cast<const uint4*>(gate);
        const uint4* u4 = reinterpret_cast<const uint4*>(up);
        float g[FGA_CHUNK];
        float u[FGA_CHUNK];
        fga_unpack_bf16x8(__ldg(g4 + k16 * 2), &g[0]);
        fga_unpack_bf16x8(__ldg(g4 + k16 * 2 + 1), &g[8]);
        fga_unpack_bf16x8(__ldg(u4 + k16 * 2), &u[0]);
        fga_unpack_bf16x8(__ldg(u4 + k16 * 2 + 1), &u[8]);
#pragma unroll
        for (int i = 0; i < FGA_CHUNK; i++) {
            const float sig_den = __fadd_rn(1.0f, __expf(-g[i]));
            out[i] = __fmul_rn(__fdiv_rn(g[i], sig_den), u[i]);
        }
    }
};

// One 16-value chunk: acc += a[i] * (decode(byte_i) * scale), in K order.
template <typename Act>
__device__ __forceinline__ float fga_chunk(float acc, uint4 b, float scale, const Act& act, unsigned int k16) {
    float a[FGA_CHUNK];
    act.chunk(k16, a);
    const unsigned int words[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float2 lo = fga_e4m3x2_to_f32x2((unsigned short)(words[j] & 0xFFFFu));
        const float2 hi = fga_e4m3x2_to_f32x2((unsigned short)(words[j] >> 16));
        const float w0 = __fmul_rn(lo.x, scale);
        const float w1 = __fmul_rn(lo.y, scale);
        const float w2 = __fmul_rn(hi.x, scale);
        const float w3 = __fmul_rn(hi.y, scale);
        acc = __fadd_rn(acc, __fmul_rn(a[j * 4 + 0], w0));
        acc = __fadd_rn(acc, __fmul_rn(a[j * 4 + 1], w1));
        acc = __fadd_rn(acc, __fmul_rn(a[j * 4 + 2], w2));
        acc = __fadd_rn(acc, __fmul_rn(a[j * 4 + 3], w3));
    }
    return acc;
}

// One output row's partial dot product for lane `lane` of 64.
template <typename Act>
__device__ __forceinline__ float fga_row(const unsigned char* __restrict__ b_row, const float* __restrict__ scale_row,
                                         const Act& act, unsigned int k16_count, unsigned int lane) {
    const uint4* b4 = reinterpret_cast<const uint4*>(b_row);
    float acc = 0.0f;
    unsigned int k16 = lane;
    for (; k16 + (FGA_UNROLL - 1) * FGA_LANES < k16_count; k16 += FGA_UNROLL * FGA_LANES) {
        uint4 b[FGA_UNROLL];
        float s[FGA_UNROLL];
#pragma unroll
        for (int u = 0; u < FGA_UNROLL; u++) {
            const unsigned int c = k16 + u * FGA_LANES;
            b[u] = __ldcs(b4 + c);
            s[u] = __ldg(scale_row + c / FGA_CHUNKS_PER_SCALE);
        }
#pragma unroll
        for (int u = 0; u < FGA_UNROLL; u++) {
            acc = fga_chunk(acc, b[u], s[u], act, k16 + u * FGA_LANES);
        }
    }
    for (; k16 < k16_count; k16 += FGA_LANES) {
        acc = fga_chunk(acc, __ldcs(b4 + k16), __ldg(scale_row + k16 / FGA_CHUNKS_PER_SCALE), act, k16);
    }
    return acc;
}

// The gb10 reduction: 5-step shfl.down per warp, then the two warps of a row
// added through shared memory by lane 0, one BF16 round. Every thread of the
// CTA reaches the barrier (a row past N computes nothing but still syncs).
__device__ __forceinline__ void fga_reduce_store(float acc, bool live, unsigned int local_out, unsigned int lane,
                                                 __nv_bfloat16* __restrict__ c, unsigned int n) {
    __shared__ float smem[FGA_N_PER_BLOCK * 2];
#pragma unroll
    for (int offset = FGA_WARP / 2; offset > 0; offset >>= 1) {
        acc = __fadd_rn(acc, __shfl_down_sync(0xFFFFFFFFu, acc, offset));
    }
    if (lane % FGA_WARP == 0) {
        smem[local_out * 2 + lane / FGA_WARP] = acc;
    }
    __syncthreads();
    if (live && lane == 0) {
        c[n] = __float2bfloat16(__fadd_rn(smem[local_out * 2], smem[local_out * 2 + 1]));
    }
}

template <typename Act>
__device__ __forceinline__ void fga_gemv(const Act& act, const unsigned char* __restrict__ B,
                                         const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C,
                                         unsigned int N, unsigned int K) {
    const unsigned int local_out = threadIdx.x / FGA_LANES;
    const unsigned int lane = threadIdx.x % FGA_LANES;
    const unsigned int n = blockIdx.x * FGA_N_PER_BLOCK + local_out;
    const bool live = n < N;
    float acc = 0.0f;
    if (live) {
        const unsigned int k_blocks = (K + FGA_FP8_BLOCK - 1) / FGA_FP8_BLOCK;
        acc = fga_row(B + (unsigned long long)n * K, block_scale + (unsigned long long)(n / FGA_FP8_BLOCK) * k_blocks,
                      act, K / FGA_CHUNK, lane);
    }
    fga_reduce_store(acc, live, local_out, lane, C, n);
}

// C[n] = sum_k A[k] * E4M3(B[n,k]) * block_scale[n/128, k/128].
// Grid (ceil(N/4), 1, 1), block 256. K % 16 == 0.
extern "C" __global__ void __launch_bounds__(FGA_BLOCK) fp8_gemv_ada(const __nv_bfloat16* __restrict__ A,
                                                                    const unsigned char* __restrict__ B,
                                                                    const float* __restrict__ block_scale,
                                                                    __nv_bfloat16* __restrict__ C, unsigned int N,
                                                                    unsigned int K) {
    const FgaActRow act{A};
    fga_gemv(act, B, block_scale, C, N, K);
}

// Two projections of the same input (gate and up); blockIdx.z picks which.
// Grid (ceil(N/4), 1, 2), block 256.
extern "C" __global__ void __launch_bounds__(FGA_BLOCK)
    fp8_gemv_ada_dual(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B1,
                      const float* __restrict__ B1_scale, __nv_bfloat16* __restrict__ C1,
                      const unsigned char* __restrict__ B2, const float* __restrict__ B2_scale,
                      __nv_bfloat16* __restrict__ C2, unsigned int N, unsigned int K) {
    const FgaActRow act{A};
    if (blockIdx.z == 0) {
        fga_gemv(act, B1, B1_scale, C1, N, K);
    } else {
        fga_gemv(act, B2, B2_scale, C2, N, K);
    }
}

// down(silu(gate) * up) with the SwiGLU computed inline per element.
// Grid (ceil(N/4), 1, 1), block 256.
extern "C" __global__ void __launch_bounds__(FGA_BLOCK)
    fp8_gemv_ada_silu_input(const __nv_bfloat16* __restrict__ gate_out, const __nv_bfloat16* __restrict__ up_out,
                            const unsigned char* __restrict__ B, const float* __restrict__ block_scale,
                            __nv_bfloat16* __restrict__ C, unsigned int N, unsigned int K) {
    const FgaActSilu act{gate_out, up_out};
    fga_gemv(act, B, block_scale, C, N, K);
}
