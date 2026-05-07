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
