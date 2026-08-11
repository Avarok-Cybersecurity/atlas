// SPDX-License-Identifier: AGPL-3.0-only

// FP32-output twin of `gdn_verify_fused_conv_kn` (issue #435 route (a)).
//
// The exact-verify arm must feed the GDN recurrence the SAME conv values the
// sequential decode path produces — and sequential decode runs the FP32-output
// conv (`causal_conv1d_update_l2norm_f32`). The BF16 fused verify conv
// truncates every Q/K/V element before the recurrence, which is the DOMINANT
// term of the spec-on/spec-off divergence (h-state relL2 ~8.6e-4 after one
// K=4 window on the 27B GDN shapes). This twin is the common fused kernel
// with `output` widened to float and the final `__float2bfloat16` removed —
// accumulation is already FP32 in both, so the FP32 output is FREE.
//
// BIT-EXACTNESS: same contract as the parent (see gdn_verify_fused_conv_kn.cu
// header): built with --fmad=false, conv dot product / SiLU / L2-norm
// reduction preserve the EXACT accumulation order of
// `causal_conv1d_update_l2norm_f32`, so per-position outputs are byte-equal
// to the per-token FP32 conv loop, and the inline conv-state snapshots are
// byte-equal to the copy_d2d they replace.
//
// `output_stride` is in FP32 elements (the parent's is BF16 elements).
//
// MODEL-SPECIFIC ON PURPOSE (shadow-first rule): qwen3.6-27b/nvfp4 only for
// now. Targets without it fall back to the per-token
// `causal_conv1d_update_l2norm_f32` loop + conv-state copy_d2d — same bits,
// more launches. Promote to common/ only after the bitwise gate passes on
// every inheriting model.
//
// Grid: (ceil(dim/256), 1, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void gdn_verify_fused_conv_kn_f32(
    float* __restrict__ conv_state,              // [dim, d_conv] FP32 (in/out)
    const __nv_bfloat16* __restrict__ new_input, // [K, input_stride] BF16
    const __nv_bfloat16* __restrict__ weight,    // [dim, d_conv] BF16
    float* __restrict__ output,                  // [K, output_stride] FP32
    float* __restrict__ conv_state_inter,        // [K, inter_stride] FP32 (out, per-token snapshots)
    unsigned int num_tokens,     // K
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,    // channels 0..qk_channels-1 get L2 normalized
    unsigned int head_dim,       // L2 norm group size (128)
    unsigned int input_stride,   // BF16 elems between positions in new_input
    unsigned int output_stride,  // FP32 elems between positions in output
    unsigned int inter_stride,   // FP32 elems between snapshots in conv_state_inter
    float l2_eps
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);
    const bool valid = (ch < dim);

    // ── Load this channel's d_conv-element sliding window into registers ──
    float win[8]; // d_conv <= 8
    if (valid) {
        const float* state = conv_state + ch * d_conv;
        for (unsigned int i = 0; i < d_conv; i++) win[i] = state[i];
    }

    const __nv_bfloat16* w = valid ? (weight + ch * d_conv) : nullptr;
    float wcoef[8];
    if (valid) {
        for (unsigned int k = 0; k < d_conv; k++) wcoef[k] = (float)w[k];
    }

    __shared__ float warp_sums[8];

    // Process the positions sequentially in registers. L2-norm needs
    // __syncthreads per position, so the loop body mirrors the single-token
    // kernel exactly.
    for (unsigned int t = 0; t < num_tokens; t++) {
        float silu = 0.0f;
        if (valid) {
            // Shift window left, append this position's input (== global path).
            for (unsigned int i = 0; i < d_conv - 1; i++) win[i] = win[i + 1];
            win[d_conv - 1] = (float)new_input[t * input_stride + ch];
            // bias == nullptr in production conv1d_update_l2norm.
            float acc = 0.0f;
            for (unsigned int k = 0; k < d_conv; k++) acc += win[k] * wcoef[k];
            float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
            silu = acc * sigmoid_acc;
        }

        if (block_needs_l2) {
            float sq = valid ? (silu * silu) : 0.0f;
            const unsigned int warp_id = tid / 32;
            const unsigned int lane = tid % 32;
            for (int offset = 16; offset >= 1; offset >>= 1)
                sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
            if (lane == 0) warp_sums[warp_id] = sq;
            __syncthreads();
            const unsigned int head_in_block = tid / head_dim;
            const unsigned int base_warp = head_in_block * (head_dim / 32);
            if (tid == 0 || tid == head_dim) {
                float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                            + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
                warp_sums[base_warp] = rsqrtf(total + l2_eps);
            }
            __syncthreads();
            if (valid) silu *= warp_sums[base_warp];
            // Close the loop-carried window on warp_sums: the next iteration's
            // lane-0 partial-sum write must not overtake this read.
            __syncthreads();
        }

        if (valid) output[t * output_stride + ch] = silu;  // FP32 — no truncation

        // Snapshot this position's conv-state (rollback intermediate t).
        if (valid) {
            float* snap = conv_state_inter + t * inter_stride + ch * d_conv;
            for (unsigned int i = 0; i < d_conv; i++) snap[i] = win[i];
        }
    }

    // Commit final (post last-position) sliding window to conv_state.
    if (valid) {
        float* state = conv_state + ch * d_conv;
        for (unsigned int i = 0; i < d_conv; i++) state[i] = win[i];
    }
}
