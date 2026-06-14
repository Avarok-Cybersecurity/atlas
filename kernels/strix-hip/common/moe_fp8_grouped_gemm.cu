// SPDX-License-Identifier: AGPL-3.0-only

// MoE FP8 grouped GEMM — HIP/gfx1151 COMPILE STUB.
//
// The dense Qwen3.6-27B model (the gfx1151 target) does NOT dispatch any MoE
// grouped-GEMM kernel. These entry points exist only so the kernel registry
// links. Bodies are intentionally empty (early return, outputs left untouched);
// signatures are byte-identical to the NVIDIA source so the registry builds.
// If a future MoE model targets gfx1151, port the NVIDIA mma.sync bodies to
// WMMA using the w8a16_gemm.cu / w4a16_gemm.cu idiom (the dequant + smem-tile +
// __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 pattern).

#include <cuda_bf16.h>

extern "C" __global__ void moe_fp8_grouped_gemm(
    const __nv_bfloat16* __restrict__ A,                   // [total_tokens, K] BF16
    const unsigned long long* __restrict__ B_weight_ptrs,   // [num_experts] → [N, K] FP8
    const unsigned long long* __restrict__ B_scale_ptrs,    // [num_experts] → [N/128, K/128] BF16
    __nv_bfloat16* __restrict__ C,                         // [total_expanded, N] BF16
    const int* __restrict__ expert_offsets,                 // [num_experts + 1]
    const int* __restrict__ sorted_token_ids,              // [total_expanded]
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    // Compile stub: not dispatched by the dense model. No-op.
    (void)A; (void)B_weight_ptrs; (void)B_scale_ptrs; (void)C;
    (void)expert_offsets; (void)sorted_token_ids; (void)num_experts; (void)N; (void)K;
}

extern "C" __global__ void moe_fp8_grouped_gemm_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_weight_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    // Compile stub: not dispatched by the dense model. No-op.
    (void)A; (void)B_weight_ptrs; (void)B_scale_ptrs; (void)C;
    (void)expert_offsets; (void)sorted_token_ids; (void)num_experts; (void)N; (void)K;
}
