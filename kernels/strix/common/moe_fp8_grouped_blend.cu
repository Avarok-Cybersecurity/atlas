// SPDX-License-Identifier: AGPL-3.0-only

// Atlas cross-row grouped FP8 MoE decode — final weighted sum + sigmoid blend.
// Companion of moe_shared_expert_fused_fp8_grouped.cu (separate module so each
// file stays under the source-size cap; this one needs no E4M3 LUT).
//
// Grid: (ceil(hidden/256), num_tokens, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

#define WARP_SIZE 32

// Weighted sum + sigmoid blend over SORTED-position expert rows. Same math and
// order as moe_weighted_sum_blend (moe_expert_gemv.cu): slot order 0..top_k-1,
// then the sigmoid-gated shared expert, one BF16 rounding at the end. The only
// difference is the `token_to_perm` indirection into the sorted down output.
extern "C" __global__ void moe_weighted_sum_blend_fp8_grouped(
    __nv_bfloat16* __restrict__ output,              // [num_tokens, hidden] BF16
    const __nv_bfloat16* __restrict__ expert_out,    // [num_tokens*top_k, hidden] BF16, SORTED rows
    const float* __restrict__ expert_weights,         // [num_tokens*top_k] f32, slot-major
    const int* __restrict__ token_to_perm,            // [num_tokens*top_k] slot -> sorted pos
    const __nv_bfloat16* __restrict__ shared_out,    // [num_tokens, hidden] BF16
    const __nv_bfloat16* __restrict__ input,         // [num_tokens, K] BF16 (MoE input)
    const __nv_bfloat16* __restrict__ gate_weight,   // [K] BF16 shared gate, or NULL
    unsigned int hidden,
    unsigned int top_k,
    unsigned int K
) {
    const unsigned int token = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;

    const __nv_bfloat16* my_input = input + (unsigned long long)token * K;
    const float* my_weights = expert_weights + token * top_k;
    const int* my_perm = token_to_perm + token * top_k;
    const __nv_bfloat16* my_shared_out = shared_out + (unsigned long long)token * hidden;
    __nv_bfloat16* my_output = output + (unsigned long long)token * hidden;

    __shared__ float s_warp_sums[8];
    __shared__ float sigmoid_val;

    if (gate_weight == 0) {
        if (tid == 0) sigmoid_val = 1.0f;
        __syncthreads();
    } else {
        float dot_acc = 0.0f;
        unsigned int K8 = K / 8;
        for (unsigned int k8 = tid; k8 < K8; k8 += 256) {
            uint4 a_data = ((const uint4*)my_input)[k8];
            uint4 w_data = ((const uint4*)gate_weight)[k8];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            const unsigned int w_raw[4] = {w_data.x, w_data.y, w_data.z, w_data.w};

            #pragma unroll
            for (int b = 0; b < 4; b++) {
                __nv_bfloat16 a_lo, a_hi, w_lo, w_hi;
                *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
                *(unsigned short*)&w_lo = (unsigned short)(w_raw[b] & 0xFFFF);
                *(unsigned short*)&w_hi = (unsigned short)(w_raw[b] >> 16);
                dot_acc += __bfloat162float(a_lo) * __bfloat162float(w_lo);
                dot_acc += __bfloat162float(a_hi) * __bfloat162float(w_hi);
            }
        }
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            dot_acc += __shfl_down_sync(0xFFFFFFFF, dot_acc, offset);
        }
        if (lane == 0) s_warp_sums[warp_id] = dot_acc;
        __syncthreads();
        if (tid == 0) {
            float gate_scalar = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; w++) gate_scalar += s_warp_sums[w];
            sigmoid_val = 1.0f / (1.0f + __expf(-gate_scalar));
        }
        __syncthreads();
    }

    unsigned int j = blockIdx.x * blockDim.x + tid;
    if (j >= hidden) return;

    float acc = 0.0f;
    for (unsigned int e = 0; e < top_k; e++) {
        const unsigned long long pos = (unsigned long long)my_perm[e];
        acc += my_weights[e] * __bfloat162float(expert_out[pos * hidden + j]);
    }
    acc += sigmoid_val * __bfloat162float(my_shared_out[j]);
    my_output[j] = __float2bfloat16(acc);
}
