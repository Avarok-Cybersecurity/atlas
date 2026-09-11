// SPDX-License-Identifier: AGPL-3.0-only

// Hopper W8A16 decode-GEMV inner loop — shared by the hardware-tuned
// `w8a16_gemv.cu` and `w8a16_gemv_fused.cu` in THIS directory (#928).
//
// ── WHY A HOPPER-OWNED COPY EXISTS ───────────────────────────────────────
//
// nsys, 1xH100 SXM5, Qwen/Qwen3.8-27B-FP8 (native FP8), 2026-09-11 round 10,
// C=1 steady-state decode step 18.5 ms, GPU idle 5%:
//
//   | kernel              | launches | ms/step | share | shapes                |
//   |---------------------|---------:|--------:|------:|-----------------------|
//   | `w8a16_gemv`        |      224 |    7.39 |   40% | FFN down, SSM in_proj/
//   |                     |          |         |       | out_proj, attn q/k/v/o |
//   | `w8a16_gemv_dual`   |       64 |    5.79 |   31% | gate+up (N=2x17408)   |
//
// 24.3 GB of FP8 weights per token in 13.2 ms = **1.84 TB/s** against HBM3's
// 3.35 TB/s peak (~2.8-3.0 achievable). The gb10 kernel this replaces is not
// bandwidth-bound on H100; it is bound by its own LUT gather and by too few
// bytes in flight per SM. Both are addressed here, in that order.
//
// ── DIAGNOSIS 1: the LUT gather saturates the SM load/store unit ──────────
//
// `kernels/gb10/common/w8a16_gemv.cu` decodes each E4M3 byte with one
// SHARED-MEMORY load from a 256-entry table: 16 `LDS` per 16-byte chunk, with
// DATA-DEPENDENT indices. 256 floats span the 32 banks eight times over, so 32
// random indices collide at roughly the balls-in-bins maximum, ~3.3x
// serialisation. Per SM the H100 retires ~32 shared lanes/cycle.
//
// 2.6 TB/s over 132 SMs at 1.755 GHz is 11.2 weight-bytes/cycle/SM. Charge
// each byte one LDS at ~3.3x conflict plus the chunk's 3 global 16-byte loads
// (1 weight + 2 activation) and one scale load, and the LSU bill is ~38
// lane-slots/cycle against a 32-slot budget — over capacity BEFORE any DRAM
// request is issued. At the MEASURED 7.9 bytes/cycle/SM (1.84 TB/s) the same
// arithmetic lands at ~27 of 32, i.e. ~85% LSU occupancy. The kernel is
// LSU-bound, not memory-bound, which is why more CTAs never helped it.
//
// FIX: `cvt.rn.f16x2.e4m3x2` (sm_89+, present on sm_90a and sm_100a) decodes
// TWO bytes per instruction on the math pipe and touches the LSU zero times.
// 16 LDS + their address arithmetic per chunk become 8 `cvt` + 8 `cvt.f32.f16`
// pairs. That is the same trade `w8a16_gemm_m16.cu` already makes on this
// hardware; the note under NUMERICS below is the same note.
//
// ── DIAGNOSIS 2: bytes in flight per SM ──────────────────────────────────
//
// The gb10 loop issues ONE 16-byte weight load per lane per iteration and then
// consumes it immediately, so a warp has one outstanding weight request at a
// time: 32 lanes x 16 B = 512 B per warp. At 8 CTAs/SM (ptxas: 32 registers,
// 1,056 B smem) that is 64 warps x 512 B = 32 KB in flight — enough to cover
// ~600 ns of HBM3 latency ONLY when the grid actually fills the machine.
//
// It does not, for half the bytes. The Rust launcher is `ceil(N/4)` CTAs of
// 256 threads (`ops::w8a16_gemv`), so grid depends on N alone:
//
//   gate/up N=17408 -> 4,352 CTAs  ~33 CTAs/SM available  (measured 1,979 GB/s)
//   q       N=12288 -> 3,072 CTAs
//   down/o  N=5120  -> 1,280 CTAs  ~9.7 CTAs/SM
//   k/v     N=1024  ->   256 CTAs  ~1.9 CTAs/SM           (measured   861 GB/s)
//
// At grid 256 an SM holds ~2 CTAs = 16 warps = 8 KB in flight, which is ~1.8
// TB/s of Little's-law ceiling before any inefficiency — and 861 GB/s is what
// came out. The occupancy the kernel is ENTITLED to is irrelevant when the
// grid cannot supply it.
//
// FIX: [`HOPPER_GEMV_UNROLL`] independent 16-byte weight loads (and their
// scales) are issued BEFORE any of them is consumed, so per-warp memory-level
// parallelism stops depending on how many CTAs the grid could place. At
// UNROLL=4 the k/v shape goes from 8 KB to 32 KB of weight bytes in flight per
// SM at the SAME grid.
//
// NOT FIXED HERE, deliberately:
//   * More outputs per CTA (8 outputs / 512 threads) would halve the CTA count
//     and the duplicated activation traffic, but the grid is chosen HOST-side
//     in `ops::w8a16_gemv` as `ceil(N/4)`. Changing it is a Rust change; this
//     file is a same-signature target override and makes none.
//   * Split-K for N <= 2048 is the other half of the k/v answer and already
//     exists as `w8a16_gemv_splitk` + `ops::w8a16_decode_gemv::splitk_plan`,
//     behind `ATLAS_FFN_DOWN_SPLITK`. It is a two-kernel launch plan, so it
//     likewise cannot live inside a single-kernel override.
//
// ── NUMERICS: bit-identical, with one impossible byte ────────────────────
//
// Every arithmetic step, and the ORDER of every one of them, is the gb10
// kernel's:
//   * lane l of 64 walks chunks l, l+64, l+128, ... — unrolling reorders the
//     LOADS only; the accumulation visits the chunks in the same sequence;
//   * within a chunk: bytes 0..3 of `b.x` against activations 0..3, then
//     `b.y`, `b.z`, `b.w`, each `w = decode(byte) * scale` then `acc += a * w`;
//   * `--fmad=false` is a tree-wide nvcc flag (`kernels/gb10/common/
//     KERNEL.toml`), so each `acc += a * w` is a separate FMUL and FADD here
//     exactly as it is there — nothing contracts differently;
//   * the same 5-step shfl.down butterfly, the same two-warp smem add, the
//     same single `__float2bfloat16` round at the end.
//
// The ONE divergence is the dequant instruction, and it is confined to two
// byte values. FP16 represents every finite E4M3 exactly (5 exponent bits vs
// 4, 10 mantissa bits vs 3) and FP16->FP32 is exact, so `cvt` reproduces
// `E4M3_LUT`'s FP32 bits for all 254 finite bytes, `-0.0` included. E4M3
// 0x7F/0xFF are the format's only NaNs: `E4M3_LUT` decodes them to +-0 and
// `cvt` yields NaN. A block-scaled FP8 checkpoint cannot contain them — they
// are not in the quantiser's output alphabet — and the oracles draw from the
// same 0x00..0x7E / 0x80..0xFE alphabet the batch4 oracle uses. Arches below
// sm_89 take the `E4M3_LUT` fallback and keep the +-0 behaviour.
//
// Receipt: `examples/native_fp8_gemv_hopper_microtest.rs` asserts
// `unequal == 0` against `w8a16_gemv_splitk` at splits=1 (which is itself
// pinned bit-identical to the gb10 `w8a16_gemv`) at all six decode shapes, and
// against an exact host model of the reduction order.

#ifndef ATLAS_HOPPER_W8A16_GEMV_CUH
#define ATLAS_HOPPER_W8A16_GEMV_CUH

#include <cuda_bf16.h>

#include "e4m3_lut.cuh"  // pre-sm_89 fallback + the FP32 values `cvt` must match

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128
/// K values one lane consumes per chunk — one `uint4` of FP8 bytes.
#define K_PER_CHUNK 16
/// Chunks per 128-wide FP8 scale block, i.e. `k_block = k16 / CHUNKS_PER_SCALE`.
#define CHUNKS_PER_SCALE (FP8_BLOCK / K_PER_CHUNK)
/// Independent chunk loads issued before the first is consumed. 4 is what
/// takes the grid-256 k/v shape from 8 KB to 32 KB of weight bytes in flight
/// per SM; the SiLU-input variant uses 2 because its activation decode already
/// holds 32 live floats (see `w8a16_gemv_fused.cu`).
#define HOPPER_GEMV_UNROLL 4

/// Two E4M3 bytes (low byte first) -> two FP32, equal to `E4M3_LUT`'s entries
/// for every byte a block-scaled FP8 checkpoint can contain. See NUMERICS.
__device__ __forceinline__ float2 hopper_e4m3x2_to_f32x2(unsigned short raw) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890)
    unsigned int h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(raw));
    return __half22float2(*reinterpret_cast<const __half2*>(&h2));
#else
    return make_float2(E4M3_LUT[raw & 0xFFu], E4M3_LUT[raw >> 8]);
#endif
}

/// Raw BF16 bits -> FP32, spelled as the gb10 kernel spells it.
__device__ __forceinline__ float hopper_bf16_bits_to_f32(unsigned short bits) {
    __nv_bfloat16 h;
    *(unsigned short*)&h = bits;
    return __bfloat162float(h);
}

/// One chunk's 16 activation values, in the gb10 kernel's operand order.
struct HopperActChunk {
    float v[K_PER_CHUNK];
};

/// 8 BF16 lanes of a `uint4` -> `out[0..8]`, lo half before hi half — the
/// `a32_lo & 0xFFFF`, `a32_lo >> 16`, `a32_hi & 0xFFFF`, `a32_hi >> 16`
/// sequence of the gb10 loop.
__device__ __forceinline__ void hopper_unpack_bf16x8(uint4 d, float* out) {
    const unsigned int raw[4] = {d.x, d.y, d.z, d.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        out[i * 2 + 0] = hopper_bf16_bits_to_f32((unsigned short)(raw[i] & 0xFFFFu));
        out[i * 2 + 1] = hopper_bf16_bits_to_f32((unsigned short)(raw[i] >> 16));
    }
}

/// Activation source: a plain BF16 row `A[1, K]`.
struct HopperActRow {
    const __nv_bfloat16* __restrict__ a;

    __device__ __forceinline__ void chunk(unsigned int k16, HopperActChunk& out) const {
        hopper_unpack_bf16x8(((const uint4*)a)[k16 * 2], &out.v[0]);
        hopper_unpack_bf16x8(((const uint4*)a)[k16 * 2 + 1], &out.v[8]);
    }
};

/// Accumulate one 16-value chunk. `acc += a[i] * (decode(byte_i) * scale)`, in
/// the gb10 kernel's operand order, byte group by byte group.
template <typename Act>
__device__ __forceinline__ float hopper_gemv_chunk(
    float acc,
    uint4 b,
    float scale,
    const Act& act,
    unsigned int k16
) {
    HopperActChunk a;
    act.chunk(k16, a);
    const unsigned int b_raw[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 lo = hopper_e4m3x2_to_f32x2((unsigned short)(b_raw[i] & 0xFFFFu));
        float2 hi = hopper_e4m3x2_to_f32x2((unsigned short)(b_raw[i] >> 16));
        float w0 = lo.x * scale;
        float w1 = lo.y * scale;
        float w2 = hi.x * scale;
        float w3 = hi.y * scale;
        acc += a.v[i * 4 + 0] * w0;
        acc += a.v[i * 4 + 1] * w1;
        acc += a.v[i * 4 + 2] * w2;
        acc += a.v[i * 4 + 3] * w3;
    }
    return acc;
}

/// One output row's dot product, for lane `lane` of `BLOCK_SIZE/N_PER_BLOCK`.
///
/// `b_row` is `B + n*K`; `scale_row` is `block_scale + n_block*k_blocks`. The
/// lane walks chunks `lane, lane+64, lane+128, ...` exactly as the gb10 loop
/// does. The only change is that `UNROLL` chunk loads (and their scales) are
/// issued before the first is consumed — see DIAGNOSIS 2.
template <int UNROLL, typename Act>
__device__ __forceinline__ float hopper_gemv_row(
    const unsigned char* __restrict__ b_row,
    const float* __restrict__ scale_row,
    const Act& act,
    unsigned int k16_count,
    unsigned int lane
) {
    const unsigned int stride = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const uint4* b4 = (const uint4*)b_row;
    float acc = 0.0f;
    unsigned int k16 = lane;

    // Pipelined body: UNROLL far-latency weight loads in flight at once.
    for (; k16 + (UNROLL - 1) * stride < k16_count; k16 += UNROLL * stride) {
        uint4 b[UNROLL];
        float s[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const unsigned int c = k16 + u * stride;
            b[u] = b4[c];
            s[u] = scale_row[c / CHUNKS_PER_SCALE];
        }
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            acc = hopper_gemv_chunk(acc, b[u], s[u], act, k16 + u * stride);
        }
    }
    // Tail: the same chunks, same order, one at a time.
    for (; k16 < k16_count; k16 += stride) {
        acc = hopper_gemv_chunk(acc, b4[k16], scale_row[k16 / CHUNKS_PER_SCALE], act, k16);
    }
    return acc;
}

/// The gb10 kernel's two-stage reduction and single BF16 round, unchanged:
/// a 5-step `shfl.down` butterfly per warp, then the two warps' partials added
/// through `smem` by lane 0.
__device__ __forceinline__ void hopper_gemv_reduce_store(
    float acc,
    float* smem,
    unsigned int local_out,
    unsigned int lane,
    __nv_bfloat16* __restrict__ c,
    unsigned int n
) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    const unsigned int warp_in_out = lane / WARP_SIZE;
    if (lane % WARP_SIZE == 0) {
        smem[local_out * 2 + warp_in_out] = acc;
    }
    __syncthreads();
    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        c[n] = __float2bfloat16(result);
    }
}

#endif  // ATLAS_HOPPER_W8A16_GEMV_CUH
