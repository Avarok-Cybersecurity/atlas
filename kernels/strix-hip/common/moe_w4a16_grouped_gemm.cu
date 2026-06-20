// SPDX-License-Identifier: AGPL-3.0-only

// MoE W4A16 (NVFP4) grouped GEMM — HIP/gfx1151 COMPILE STUB.
//
// The dense Qwen3.6-27B model (the gfx1151 target) does NOT dispatch any MoE
// grouped-GEMM kernel. These entry points exist only so the kernel registry
// links. Bodies are intentionally empty (early return, outputs left untouched);
// signatures are byte-identical to the NVIDIA source so the registry builds.
// A real port would mirror w4a16_gemm.cu (NVFP4 nibble dequant + WMMA).
//
// NOTE: a separate, fully-ported MoE W4A16 GEMM already exists at
// /workspace/atlas-port-work/ported/moe_w4a16_grouped_gemm.cu (the
// qwen3.6-27b/nvfp4 model variant). This common/ stub is only for the dense
// build's registry and is never executed.

#include <cuda_bf16.h>

extern "C" __global__ void moe_w4a16_grouped_gemm(
    const __nv_bfloat16* __restrict__ A,        // [total_tokens, K]
    const unsigned char* __restrict__ B_packed,  // [num_experts, K, N/2] packed FP4
    const unsigned char* __restrict__ B_scale,   // [num_experts, K/GROUP, N] FP8 scales
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [total_tokens, N]
    const int* __restrict__ expert_offsets,       // [num_experts + 1]
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    // Compile stub: not dispatched by the dense model. No-op.
    (void)A; (void)B_packed; (void)B_scale; (void)scale2; (void)C;
    (void)expert_offsets; (void)num_experts; (void)N; (void)K;
}

extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable(
    const __nv_bfloat16* __restrict__ A,           // [num_tokens, K]
    const unsigned long long* __restrict__ B_packed_ptrs, // [num_experts]
    const unsigned long long* __restrict__ B_scale_ptrs,  // [num_experts]
    const float* __restrict__ scale2_vals,         // [num_experts]
    __nv_bfloat16* __restrict__ C,                  // [total_expanded, N_out]
    const int* __restrict__ expert_offsets,          // [num_experts + 1]
    const int* __restrict__ sorted_token_ids,       // [total_expanded]
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    // Compile stub: not dispatched by the dense model. No-op.
    (void)A; (void)B_packed_ptrs; (void)B_scale_ptrs; (void)scale2_vals; (void)C;
    (void)expert_offsets; (void)sorted_token_ids; (void)num_experts; (void)N; (void)K;
}

extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    // Compile stub: not dispatched by the dense model. No-op.
    (void)A; (void)B_packed_ptrs; (void)B_scale_ptrs; (void)scale2_vals; (void)C;
    (void)expert_offsets; (void)sorted_token_ids; (void)num_experts; (void)N; (void)K;
}
