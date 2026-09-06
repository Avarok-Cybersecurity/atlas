// SPDX-License-Identifier: AGPL-3.0-only

// embed_from_argmax: GPU-side token embedding after argmax.
//
// Reads the argmax result (u32 token ID) from device memory and copies
// the corresponding row from the BF16 embedding table to the output buffer.
// Eliminates the D2H sync that was required when the CPU read the token ID.
//
// Also writes the token ID to a secondary output for deferred CPU readback.
//
// Grid: (ceil(hidden_size/256), 1, 1)  Block: (256, 1, 1)
// 256 threads × 4 bf16 per thread = 1024 elements per iteration.
// hidden_size=8192 → 32 blocks × 256 threads, each loads 1 element.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void embed_from_argmax(
    const unsigned int* __restrict__ argmax_result,   // [1] token ID on GPU
    const __nv_bfloat16* __restrict__ embed_table,    // [vocab_size, hidden_size]
    __nv_bfloat16* __restrict__ output,               // [hidden_size]
    unsigned int* __restrict__ token_id_out,           // [1] copy of token ID for deferred readback
    unsigned int hidden_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Thread 0 of block 0 writes the token ID for deferred CPU readback
    if (idx == 0) {
        token_id_out[0] = argmax_result[0];
    }
    if (idx < hidden_size) {
        unsigned int token_id = argmax_result[0];
        output[idx] = embed_table[token_id * hidden_size + idx];
    }
}

// Batched embedding: gather N rows from the embedding table.
//
// token_ids[i] → embed_table[token_ids[i], :] → output[i, :]
// Replaces N individual D2D copies with a single kernel launch.
//
// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void batched_embed(
    const unsigned int* __restrict__ token_ids,        // [num_tokens] on device
    const __nv_bfloat16* __restrict__ embed_table,     // [vocab_size, hidden_size]
    __nv_bfloat16* __restrict__ output,                // [num_tokens, hidden_size]
    unsigned int hidden_size
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int token_id = token_ids[token_idx];
    const __nv_bfloat16* src = embed_table + (unsigned long long)token_id * hidden_size;
    __nv_bfloat16* dst = output + (unsigned long long)token_idx * hidden_size;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// FP32 output variant: reads BF16 embedding, writes FP32 to the residual stream.
extern "C" __global__ void batched_embed_f32(
    const unsigned int* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embed_table,     // [vocab_size, hidden_size] BF16
    float* __restrict__ output,                         // [num_tokens, hidden_size] FP32
    unsigned int hidden_size
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int token_id = token_ids[token_idx];
    const __nv_bfloat16* src = embed_table + (unsigned long long)token_id * hidden_size;
    float* dst = output + (unsigned long long)token_idx * hidden_size;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = __bfloat162float(src[i]);
    }
}

// FP32 output variant of embed_from_argmax for decode with FP32 residual stream.
extern "C" __global__ void embed_from_argmax_f32(
    const unsigned int* __restrict__ argmax_result,
    const __nv_bfloat16* __restrict__ embed_table,
    float* __restrict__ output,                          // FP32
    unsigned int* __restrict__ token_id_out,
    unsigned int hidden_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        token_id_out[0] = argmax_result[0];
    }
    if (idx < hidden_size) {
        unsigned int token_id = argmax_result[0];
        output[idx] = __bfloat162float(embed_table[token_id * hidden_size + idx]);
    }
}


// FP8-table batched embedding gather: rows stored as FP8 E4M3 BYTES with a
// per-row f32 dequant scale (exactly the `quantize_bf16_to_fp8` output
// layout from dense_gemv_fp8w.cu). Built for the n-gram embedding tables
// (LongCat / Qwen3.8-Flash-Next family) whose ~10M-row tables are
// FP8-quantized at load to halve their footprint; the gather dequantizes
// on read and hands BF16 rows to the projection GEMM.
//
// DECODE CONVENTION: software E4M3 decode on raw bytes, copied from
// dense_gemv_fp8w.cu's scl_fp8 — the hardware `__nv_fp8_e4m3` cast is
// non-standard under SCALE/gfx1151 and would mismatch the encoder there.
//
// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
__device__ __forceinline__ float ngram_fp8_decode(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)                  v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                          v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
extern "C" __global__ void batched_embed_fp8(
    const unsigned int* __restrict__ token_ids,   // [num_tokens] on device
    const unsigned char* __restrict__ embed_table, // [rows, hidden_size] FP8 bytes
    const float* __restrict__ row_scale,          // [rows] f32 dequant scale
    __nv_bfloat16* __restrict__ output,           // [num_tokens, hidden_size]
    unsigned int hidden_size
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int row = token_ids[token_idx];
    const unsigned char* src = embed_table + (unsigned long long)row * hidden_size;
    const float scale = row_scale[row];
    __nv_bfloat16* dst = output + (unsigned long long)token_idx * hidden_size;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = __float2bfloat16(ngram_fp8_decode(src[i]) * scale);
    }
}

// EXL3 `exl3_ngram_trellis` batched embedding gather: rows stored as
// (1 + ROW_DIM*K/16) little-endian uint16 words — word 0 is the fp16 row
// scale's bit pattern, the rest a ROW_DIM*K-bit tail-biting trellis ring
// where stream bits [i*K, (i+1)*K) are the LOW K bits of position i's
// 16-bit mul1-codebook state (LSB-first within each u16, ascending words —
// NOTE: different bit order from the 16x16-tile format in
// exl3_reconstruct.cu). state_i bit m lives at stream bit
// ((i - m/K) mod ROW_DIM)*K + (m%K). Decode (per upstream
// exllamav3 ngram_codec.py, f32 math):
//
//     row[i] = decode_mul1(state_i) * scale + head_bias[head][i]
//
// The gather's row order is [tokens, num_heads] contiguous, so
// head = row_index % num_heads. The arena holds RAW faulted rows (the
// NgramRowCache is byte-agnostic); this kernel is the decode.
//
// Grid: (num_rows, 1, 1)  Block: (192, 1, 1) — ROW_DIM=160 active lanes.
#define NGRAM_ROW_DIM 160

__device__ __forceinline__ float exl3_ngram_mul1_decode(unsigned int state)
{
    // Identical math to decode_3inst<2> in exl3_reconstruct.cu (mul1
    // codebook): scramble, dp4a byte-sum into the f16 window 1024..2047,
    // one fused multiply-add. Kept self-contained — this module loads
    // standalone.
    unsigned int x = state * 0x83DCD12Du;
    unsigned int sum = __dp4a(x, 0x01010101u, 0x6400u);
    const __half k_inv = __ushort_as_half(0x1eee);  //  0.00677 = 1/147.7
    const __half k_bias = __ushort_as_half(0xc931); // -10.39
    __half h = __ushort_as_half((unsigned short) sum);
    return __half2float(__hfma(h, k_inv, k_bias));
}

extern "C" __global__ void batched_embed_exl3(
    const unsigned int* __restrict__ slot_ids,     // [num_rows] arena slots
    const unsigned short* __restrict__ arena,      // slots x (1 + NGRAM_ROW_DIM*K/16) u16
    const __half* __restrict__ head_bias,          // [num_heads, NGRAM_ROW_DIM] f16
    __nv_bfloat16* __restrict__ output,            // [num_rows, NGRAM_ROW_DIM]
    unsigned int num_heads,
    unsigned int k_bits
) {
    const unsigned int row_idx = blockIdx.x;
    const unsigned int head = row_idx % num_heads;
    const unsigned int words_per_row = 1 + NGRAM_ROW_DIM * k_bits / 16;
    const unsigned int slot = slot_ids[row_idx];
    const unsigned short* src = arena + (unsigned long long) slot * words_per_row;

    // Row words to shared memory: the ring extraction reads scattered bits.
    __shared__ unsigned short s_row[1 + NGRAM_ROW_DIM * 8 / 16]; // K<=8 max
    for (unsigned int i = threadIdx.x; i < words_per_row; i += blockDim.x)
        s_row[i] = src[i];
    __syncthreads();

    const unsigned int i = threadIdx.x;
    if (i >= NGRAM_ROW_DIM) return;

    const float scale = __half2float(__ushort_as_half(s_row[0]));
    const unsigned short* stream = s_row + 1;
    const unsigned int total_bits = NGRAM_ROW_DIM * k_bits;

    unsigned int state = 0;
    #pragma unroll 4
    for (unsigned int m = 0; m < 16; m++) {
        unsigned int pos = (i + NGRAM_ROW_DIM - m / k_bits) % NGRAM_ROW_DIM;
        unsigned int bit = pos * k_bits + m % k_bits;
        bit %= total_bits; // defensive; pos*K + m%K < total_bits by construction
        unsigned int b = (stream[bit / 16] >> (bit % 16)) & 1u;
        state |= b << m;
    }

    float v = exl3_ngram_mul1_decode(state) * scale
        + __half2float(head_bias[(unsigned long long) head * NGRAM_ROW_DIM + i]);
    output[(unsigned long long) row_idx * NGRAM_ROW_DIM + i] = __float2bfloat16(v);
}
