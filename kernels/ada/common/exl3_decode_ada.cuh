// SPDX-License-Identifier: AGPL-3.0-only AND MIT
// After ExLlamaV3 (turboderp-org/exllamav3), MIT, Copyright (c) 2025 Turboderp (DeepSeek integration by
// @jayleaton), via TensorFold zig/kernels/cuda/exl3_decode.cuh:1-157 (Apache-2.0, ashhart/TensorFold,
// commit 3885dce). See EXL3-ATTRIBUTION.md and LICENSES/ExLlamaV3-MIT.txt.
// Changes for Atlas: namespace tf_exl3 renamed avarok_exl3; the mma.m16n8k16 f16 wrapper and the
// non-allocating stream load are inlined here instead of coming from TensorFold's common/qmm_frag.cuh and
// common/cp_async.cuh (same PTX); load_lane_words (shared-memory form, unused) is dropped; comments
// expanded with the layout facts the host code (crates/spark-model/src/port/exl3/) relies on.

// EXL3 trellis tiles on Ada (sm_89): decode, fragment placement and the 128-point Hadamard transform.
//
// FORMAT (exl3_format.zig, mirrored on the CPU by crates/spark-model/src/port/exl3/reference.rs).
//   * A weight y = x @ W, W [K, N], is stored as a trellis int16 [K/16, N/16, 8 * K2]: one 16 x 16 tile
//     per (k tile, n tile), K2 = 2 * bits a value (odd K2 are the mul1 half-bit widths).
//   * A tile is a circular bitstream of 128 * K2 bits, read MSB first within little-endian 32-bit words
//     (int16 pair 2w, 2w+1 = low, high half of word w). Value p (0..255) is the 16-bit window ending at
//     E(p) (exclusive): E(p) = (p + 1) * K2 / 2 for even K2, ((p + 1) * K2 - ((p + 1) & 1)) / 2 for odd.
//   * The 16-bit state decodes to an fp16 through one of three codebooks (3INST, MCG, MUL1) with exactly
//     one fp16 rounding.
//   * Stream value p = 8 * lane + j lands at (row = k, col = n) of the tile given by value_row / value_col,
//     which is exactly the B fragment layout of mma.m16n8k16: lane L decodes its 8 values straight into
//     its two B fragments (columns 0-7 in b0, 8-15 in b1).
//   * The full weight is W = diag(suh) H_K Wq H_N diag(svh), H the 128-block Walsh-Hadamard transform
//     scaled by 1/sqrt(128) (Wq the decoded tiles, suh fp16 [K], svh fp16 [N]).
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace avarok_exl3 {

enum Codebook : int { CB_3INST = 0, CB_MCG = 1, CB_MUL1 = 2 };

// 32-bit words of a tile: 256 values of K2 / 2 bits.
template <int K2>
__host__ __device__ constexpr int tile_words() {
    return 4 * K2;
}

// E(p): one past the last stream bit of value p's 16-bit window (exl3_format.zig's streamEnds).
template <int K2>
__host__ __device__ constexpr int stream_end(int p) {
    return (K2 & 1) ? ((p + 1) * K2 - ((p + 1) & 1)) / 2 : (p + 1) * (K2 / 2);
}

// Words a lane reads for its 8 values: the most any lane's windows span (2, or 3 for 3.5 bits and 5 to 8
// bits). Never more than 3 for the widths the host admits (K2 <= 16, K2 != 15); lane_states assumes it.
template <int K2>
__host__ __device__ constexpr int lane_words() {
    int most = 0;
    for (int l = 0; l < 32; ++l) {
        const int first = (stream_end<K2>(8 * l) - 16 + 128 * K2) % 32;
        const int need = first + stream_end<K2>(8 * l + 7) - stream_end<K2>(8 * l) + 16;
        most = need > most ? need : most;
    }
    return (most + 31) / 32;
}

// The index of the first word lane `lane` reads, and the bit offset of its first window in that word.
template <int K2>
__device__ __forceinline__ void lane_start(int lane, int& word, int& offset) {
    const int first = 4 * lane * K2 + K2 / 2 - 16 + 128 * K2;   // E(8 lane) - 16, made non-negative
    word = (first >> 5) % tile_words<K2>();
    offset = first & 31;
}

// The weight stream is read once, so do not allocate L1 for it: same value as __ldg, cache policy only.
__device__ __forceinline__ uint32_t ldg_stream_u32(const uint32_t* p) {
    uint32_t r;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
    return r;
}

// This lane's words of a tile, from global memory through the read-only path.
template <int K2>
__device__ __forceinline__ void ldg_lane_words(const uint32_t* tile, int lane, uint32_t (&w)[lane_words<K2>()]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = ldg_stream_u32(tile + (word + i) % tile_words<K2>());
}

// Bits [d, d + 16) of the 96-bit big-endian stream hi:mid:lo (d a compile-time constant after unrolling).
__device__ __forceinline__ uint32_t window16(uint32_t hi, uint32_t mid, uint32_t lo, int d) {
    if (d <= 16) return (hi >> (16 - d)) & 0xffffu;
    if (d < 32) return __funnelshift_l(mid, hi, d) >> 16;
    if (d <= 48) return (mid >> (48 - d)) & 0xffffu;
    return __funnelshift_l(lo, mid, d - 32) >> 16;
}

// The 16-bit states of this lane's 8 values (8 lane + j, j = 0..7) from its words.
template <int K2>
__device__ __forceinline__ void lane_states(const uint32_t (&w)[lane_words<K2>()], int lane, uint32_t (&s)[8]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
    const uint32_t c = lane_words<K2>() > 2 ? w[lane_words<K2>() > 2 ? 2 : 0] : 0u;
    const uint32_t hi = __funnelshift_l(w[1], w[0], offset);   // the stream from the lane's first window on
    const uint32_t mid = __funnelshift_l(c, w[1], offset);
    const uint32_t lo = c << offset;
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = window16(hi, mid, lo, stream_end<K2>(j) - stream_end<K2>(0));
}

// Two codebook values from two 16-bit states, as a half2 in a uint32 (the first state in the low half).
// One fp16 rounding each: __hadd2 of two exact fp16 halves (3INST, MCG), or one fp16 FMA (MUL1).
template <int CB>
__device__ __forceinline__ uint32_t decode2(uint32_t s0, uint32_t s1) {
    if constexpr (CB == CB_MUL1) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t h0 = __dp4a(x0, 0x01010101u, 0x6400u);    // fp16 bits of 1024 + the byte sum
        const uint32_t h1 = __dp4a(x1, 0x01010101u, 0x6400u);
        const uint32_t hh = (h0 & 0xffffu) | (h1 << 16);
        const half2 r = __hfma2(*reinterpret_cast<const half2*>(&hh), __half2half2(__ushort_as_half(0x1eee)),
                                __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<const uint32_t*>(&r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == CB_MCG) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        const uint32_t lo = __byte_perm(x0, x1, 0x5410);
        const uint32_t hi = __byte_perm(x0, x1, 0x7632);
        const half2 r = __hadd2(*reinterpret_cast<const half2*>(&lo), *reinterpret_cast<const half2*>(&hi));
        return *reinterpret_cast<const uint32_t*>(&r);
    }
}

// This lane's 8 values of a tile as the B fragments of the tile's two n8 halves (columns 0-7 in b0, 8-15 in b1).
template <int K2, int CB>
__device__ __forceinline__ void decode_lane(const uint32_t (&w)[lane_words<K2>()], int lane, uint32_t (&b0)[2],
                                            uint32_t (&b1)[2]) {
    uint32_t s[8];
    lane_states<K2>(w, lane, s);
    b0[0] = decode2<CB>(s[0], s[1]);
    b0[1] = decode2<CB>(s[2], s[3]);
    b1[0] = decode2<CB>(s[4], s[5]);
    b1[1] = decode2<CB>(s[6], s[7]);
}

// Row and column in the 16x16 tile (row = k, column = n) of this lane's value j (0..7).
__host__ __device__ constexpr int value_row(int lane, int j) { return 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1); }
__host__ __device__ constexpr int value_col(int lane, int j) { return (lane >> 2) + 8 * (j >> 2); }

// mma.m16n8k16, fp16 inputs, fp32 accumulators: d += a @ b.
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Unscaled Walsh-Hadamard transform of 128 fp32 values, 4 a lane (element 4 lane + j in v[j]), in a fixed
// butterfly order (multiply by 1/sqrt(128) afterwards). Stage by stage it is the textbook in-place FWHT at
// strides 1, 2, 4, ..., 64 with (a + b, a - b) pairs, which is what the CPU mirror reproduces bit for bit.
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128), f32 bits 0x3DB504F3

}  // namespace avarok_exl3
