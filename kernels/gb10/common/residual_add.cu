// SPDX-License-Identifier: AGPL-3.0-only

// BF16 residual addition: residual[i] += src[i]
//
// Used in the transformer loop after attention/SSM and FFN blocks.
// Operates in-place on the residual stream.

#include <cuda_bf16.h>

extern "C" __global__ void bf16_residual_add(
    __nv_bfloat16* __restrict__ residual,  // [n] — modified in-place
    const __nv_bfloat16* __restrict__ src,  // [n]
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float r = __bfloat162float(residual[i]);
        float s = __bfloat162float(src[i]);
        residual[i] = __float2bfloat16(r + s);
    }
}

// BF16 → FP32 element-wise conversion.
// Used to promote embedding output (BF16) to FP32 residual stream.
extern "C" __global__ void bf16_to_f32(
    const __nv_bfloat16* __restrict__ src,  // [n] BF16
    float* __restrict__ dst,                 // [n] FP32
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __bfloat162float(src[i]);
    }
}

// Separate-input SiLU multiplication: output = SiLU(gate) * up.
//
// Unlike fused_silu_mul in dense_gemm_bf16.cu which expects interleaved
// gate_up, this takes separate gate and up buffers. Used by MoE shared
// expert which has separate gate_proj and up_proj.
extern "C" __global__ void silu_mul_separate(
    const __nv_bfloat16* __restrict__ gate,   // [n]
    const __nv_bfloat16* __restrict__ up,     // [n]
    __nv_bfloat16* __restrict__ output,        // [n]
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        // SiLU(x) = x * sigmoid(x)
        float silu_g = g / (1.0f + expf(-g));
        output[i] = __float2bfloat16(silu_g * u);
    }
}

// Scaled accumulate: output[i] += scale * src[i] (BF16 with f32 scale).
//
// Used for MoE routing: accumulate weighted expert outputs.
extern "C" __global__ void bf16_scaled_add(
    __nv_bfloat16* __restrict__ output,    // [n] — modified in-place
    const __nv_bfloat16* __restrict__ src,  // [n]
    float scale,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float o = __bfloat162float(output[i]);
        float s = __bfloat162float(src[i]);
        output[i] = __float2bfloat16(o + scale * s);
    }
}

// Sigmoid-gated blend: output = sigmoid_gate * src + output (BF16).
//
// Used for MoE shared expert: blend shared expert output with routed output.
// sigmoid_gate is a single f32 value applied to all elements.
extern "C" __global__ void bf16_sigmoid_blend(
    __nv_bfloat16* __restrict__ output,    // [n] — routed output, modified in-place
    const __nv_bfloat16* __restrict__ src,  // [n] — shared expert output
    float sigmoid_gate,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float o = __bfloat162float(output[i]);
        float s = __bfloat162float(src[i]);
        output[i] = __float2bfloat16(o + sigmoid_gate * s);
    }
}

// Element-wise sigmoid gate: output[i] = input[i] * sigmoid(gate[i]).
//
// Used for gated attention: attn_output = attn_output * sigmoid(q_gate).
// gate and input are separate BF16 vectors of the same size.
extern "C" __global__ void sigmoid_gate_mul(
    const __nv_bfloat16* __restrict__ input,   // [n]
    const __nv_bfloat16* __restrict__ gate,    // [n]
    __nv_bfloat16* __restrict__ output,        // [n]
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(gate[i]);
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = __float2bfloat16(x * sigmoid_g);
    }
}

// Batched sigmoid gate multiply across multiple tokens.
// Replaces per-token kernel launches with a single launch for all tokens.
//
// input is contiguous [num_tokens, dim]. gate is strided [num_tokens, gate_stride]
// (gate data starts at gate pointer, with gate_stride elements between tokens).
// output[t * dim + d] = input[t * dim + d] * sigmoid(gate[t * gate_stride + d])
//
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void sigmoid_gate_mul_batched(
    const __nv_bfloat16* __restrict__ input,   // [num_tokens, dim] contiguous
    const __nv_bfloat16* __restrict__ gate,    // [num_tokens, gate_stride] strided
    __nv_bfloat16* __restrict__ output,        // [num_tokens, dim] contiguous
    unsigned int dim,
    unsigned int gate_stride,
    unsigned int total_elements               // num_tokens * dim
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        unsigned int t = i / dim;
        unsigned int d = i % dim;
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(gate[t * gate_stride + d]);
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = __float2bfloat16(x * sigmoid_g);
    }
}

// BF16 concatenation: out[0..N] = a[0..N], out[N..2N] = b[0..N].
//
// Used by MTP head to concatenate normed embedding + normed hidden state
// before the FC projection. Both inputs are [N] BF16, output is [2N] BF16.
extern "C" __global__ void bf16_concat(
    const __nv_bfloat16* __restrict__ a,      // [N]
    const __nv_bfloat16* __restrict__ b,      // [N]
    __nv_bfloat16* __restrict__ out,          // [2N]
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = a[i];
        out[N + i] = b[i];
    }
}

// Device-pointer sigmoid blend: reads gate scalar from device BF16 pointer.
//
// output[i] += sigmoid(bf16_to_f32(*gate_ptr)) * src[i]
//
// Eliminates the D2H + CPU sigmoid for shared expert gate.
// Each block reads the single BF16 scalar and computes sigmoid in shared mem.
// Per-head sigmoid gate multiply with broadcast over head_dim.
// Step 3.7 attention gate: g_proj produces one scalar per head,
// which is sigmoid-gated and broadcast-multiplied across head_dim.
//
// input/output: [num_tokens, nq * hd] contiguous BF16
// gate: [num_tokens, nq] contiguous BF16 (one value per head)
//
// output[t, h, d] = input[t, h, d] * sigmoid(gate[t, h])
//
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void sigmoid_gate_mul_head_broadcast(
    const __nv_bfloat16* __restrict__ input,   // [num_tokens, nq * hd]
    const __nv_bfloat16* __restrict__ gate,    // [num_tokens, nq]
    __nv_bfloat16* __restrict__ output,        // [num_tokens, nq * hd]
    unsigned int nq,
    unsigned int hd,
    unsigned int total_elements               // num_tokens * nq * hd
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        unsigned int dim = nq * hd;
        unsigned int t = i / dim;
        unsigned int within_token = i % dim;
        unsigned int head_idx = within_token / hd;
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(gate[t * nq + head_idx]);
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = __float2bfloat16(x * sigmoid_g);
    }
}

extern "C" __global__ void bf16_sigmoid_blend_device(
    __nv_bfloat16* __restrict__ output,             // [n] — routed output, modified in-place
    const __nv_bfloat16* __restrict__ src,          // [n] — shared expert output
    const __nv_bfloat16* __restrict__ gate_ptr,     // device pointer to single BF16 scalar
    unsigned int n
) {
    __shared__ float sigmoid_val;
    if (threadIdx.x == 0) {
        float g = __bfloat162float(*gate_ptr);
        sigmoid_val = 1.0f / (1.0f + expf(-g));
    }
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float o = __bfloat162float(output[i]);
        float s = __bfloat162float(src[i]);
        output[i] = __float2bfloat16(o + sigmoid_val * s);
    }
}
