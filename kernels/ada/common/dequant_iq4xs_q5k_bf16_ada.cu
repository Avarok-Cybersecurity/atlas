// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from llama.cpp (ggml-org/llama.cpp), MIT License, Copyright (c) 2023-2026 The ggml
// authors: ggml/src/ggml-common.h (block_iq4_xs, block_iq4_nl, block_q5_K, kvalues_iq4nl),
// ggml/src/ggml-quants.c (dequantize_row_iq4_xs, dequantize_row_iq4_nl, dequantize_row_q5_K) and
// ggml/src/ggml-cuda/convert.cu (dequantize_block_iq4_xs). Changes: one CUDA block per GGUF block with
// the element loop strided over threads, bf16 output (round to nearest even), block strides passed as
// arguments, the style of Atlas's kernels/gb10/common/dequant_gguf_bf16.cu, by Atlas contributors.
//
// Responsibility: GPU dequant of the GGUF types the gb10 dequant file does not cover and the keep-packed
// 4-bit route needs on Ada (port lane L19, row `gguf_native_iq4`):
//   * IQ4_XS (ggml id 23): the `W4Linear::dequant_bf16` fallback and oracle of a packed IQ4_XS weight, and
//     the load-time expansion of an IQ4_XS tensor the route does not keep packed;
//   * IQ4_NL (id 20): its 32-weight sibling, for completeness of the IQ4 family;
//   * Q5_K (id 13): the load-time expansion of the Q5_K tensors a mixed ("UD") IQ4_XS file carries.
// Raw little-endian block bytes in, contiguous bf16 [n_blocks * QK] out.
//
// BIT EXACT with the CPU oracle (crates/spark-runtime/src/weights/gguf/dequant_cpu/iq4.rs and blocks.rs):
// the same products in the same order, no fused multiply-add (`--fmad=false` from common/KERNEL.toml), and
// the same round-to-nearest-even bf16 store. The microtest (examples/iq4xs_gemv_microtest.rs) compares the
// two bit for bit.
//
// Entry points (module `dequant_iq4xs_q5k_bf16_ada`), grid (n_blocks, 1, 1), block (256, 1, 1):
//   dequant_iq4_xs_to_bf16_ada(blocks, out, n_blocks, block_bytes = 136)
//   dequant_iq4_nl_to_bf16_ada(blocks, out, n_blocks, block_bytes = 18)
//   dequant_q5_k_to_bf16_ada  (blocks, out, n_blocks, block_bytes = 176)

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// The IQ4_NL / IQ4_XS non-linear code book (ggml-common.h `kvalues_iq4nl`).
__constant__ signed char atlas_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// Little-endian IEEE fp16 -> f32 (exact).
__device__ __forceinline__ float atlas_dq_rd_f16(const unsigned char* p) {
    const unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(bits));
}

// ---- IQ4_XS : { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128] }, QK=256, 136 B ------------------------
// Eight 32-weight sub-blocks ib; 6-bit scale ls = low nibble of scales_l[ib/2] (high nibble for odd ib)
// | two bits of scales_h at 2*ib, << 4. dl = d * (ls - 32); sub-block element j < 16 is the low nibble of
// qs[16*ib + j], element j >= 16 the high nibble of qs[16*ib + j - 16]; value = dl * kvalues[nibble].
extern "C" __global__ void dequant_iq4_xs_to_bf16_ada(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int block_bytes)        // 136
{
    const unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = blocks + (unsigned long long)b * block_bytes;
    const float d = atlas_dq_rd_f16(blk);
    const unsigned int scales_h = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
    const unsigned char* scales_l = blk + 4;
    const unsigned char* qs = blk + 8;
    __nv_bfloat16* o = out + (unsigned long long)b * 256u;

    for (unsigned int y = threadIdx.x; y < 256u; y += blockDim.x) {
        const unsigned int ib = y >> 5;          // sub-block 0..7
        const unsigned int j  = y & 31u;         // element in the sub-block
        const int ls = (int)(((scales_l[ib >> 1] >> (4u * (ib & 1u))) & 0x0Fu) |
                             (((scales_h >> (2u * ib)) & 3u) << 4));
        const float dl = d * (float)(ls - 32);
        const unsigned char byte = qs[16u * ib + (j & 15u)];
        const unsigned int nib = j < 16u ? (byte & 0x0Fu) : (byte >> 4);
        o[y] = __float2bfloat16(dl * (float)atlas_kvalues_iq4nl[nib]);
    }
}

// ---- IQ4_NL : { f16 d; u8 qs[16] }, QK=32, 18 B --------------------------------------------------------
// value j < 16 = d * kvalues[qs[j] & 15], j >= 16 = d * kvalues[qs[j - 16] >> 4].
extern "C" __global__ void dequant_iq4_nl_to_bf16_ada(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int block_bytes)        // 18
{
    const unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = blocks + (unsigned long long)b * block_bytes;
    const float d = atlas_dq_rd_f16(blk);
    const unsigned char* qs = blk + 2;
    __nv_bfloat16* o = out + (unsigned long long)b * 32u;
    for (unsigned int j = threadIdx.x; j < 32u; j += blockDim.x) {
        const unsigned char byte = qs[j & 15u];
        const unsigned int nib = j < 16u ? (byte & 0x0Fu) : (byte >> 4);
        o[j] = __float2bfloat16(d * (float)atlas_kvalues_iq4nl[nib]);
    }
}

// Q4_K / Q5_K 6-bit packed scale + min (ggml get_scale_min_k4).
__device__ __forceinline__ void atlas_dq_scale_min_k4(
    int j, const unsigned char* q, unsigned char* sc, unsigned char* mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

// ---- Q5_K : { f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128] }, QK=256, 176 B --------------------
// Four chunks c of 64: elements 0..31 take the low nibbles of qs[32c..32c+32] with scale/min 2c and high
// bit (qh[l] >> 2c) & 1, elements 32..63 the high nibbles with scale/min 2c+1 and bit 2c+1.
// value = (d * sc) * (q + 16 * bit) - (dmin * m).
extern "C" __global__ void dequant_q5_k_to_bf16_ada(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int block_bytes)        // 176
{
    const unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = blocks + (unsigned long long)b * block_bytes;
    const float d    = atlas_dq_rd_f16(blk);
    const float dmin = atlas_dq_rd_f16(blk + 2);
    const unsigned char* scales = blk + 4;
    const unsigned char* qh     = blk + 16;
    const unsigned char* qs     = blk + 48;
    __nv_bfloat16* o = out + (unsigned long long)b * 256u;

    for (unsigned int y = threadIdx.x; y < 256u; y += blockDim.x) {
        const unsigned int c    = y >> 6;          // chunk 0..3
        const unsigned int half = (y >> 5) & 1u;   // 0 = low nibble, 1 = high nibble
        const unsigned int l    = y & 31u;
        const int is = (int)(2u * c + half);
        unsigned char sc, mn;
        atlas_dq_scale_min_k4(is, scales, &sc, &mn);
        const float d1 = d * (float)sc;
        const float m1 = dmin * (float)mn;
        const unsigned char byte = qs[c * 32u + l];
        const int lo = (int)(half ? (byte >> 4) : (byte & 0x0F));
        const int hi = ((qh[l] >> (2u * c + half)) & 1u) ? 16 : 0;
        o[y] = __float2bfloat16(d1 * (float)(lo + hi) - m1);
    }
}
