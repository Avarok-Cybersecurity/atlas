// SPDX-License-Identifier: AGPL-3.0-only

// Atlas fused generic-K MTP/DFlash-verify conv1d+L2norm — the K=17 (γ=16)
// generalization of gdn_verify_fused_conv_k2.
//
// The K=17 DFlash verify arm currently runs the conv epilogue as a PER-TOKEN
// loop: `causal_conv1d_update_l2norm` is launched 17 TIMES with a full
// conv-state D2D copy after every launch for rollback — 34 serialized ops per
// SSM layer per verify step, each at single-token occupancy and each blocked
// on the in-place conv_state update of the previous one. This kernel replaces
// the whole sequence with ONE launch:
//
//   gdn_verify_fused_conv_kn — advances the conv sliding window for ALL
//     num_tokens positions in registers (no global state re-read between
//     positions), writes conv output for every position, and writes every
//     per-token conv-state rollback snapshot inline to a strided
//     intermediates array (conv_state_inter + t*inter_stride). The committed
//     (post final-position) window is left in conv_state AND duplicated as
//     the last snapshot, so the caller needs NO copy_d2d at all.
//
// The per-token "sequential dependency" is an artifact of updating conv_state
// in place: the window after token t is just the last d_conv inputs ending at
// t, so every position is computable from registers in one pass — exactly
// what batched prefill conv already exploits.
//
// BIT-EXACTNESS: builds with --fmad=false (KERNEL.toml). The conv dot
// product, SiLU, and L2-norm reduction preserve the EXACT accumulation order
// of `causal_conv1d_update_l2norm`, so outputs are byte-identical to the
// per-token path (same property proven for the K=2 twin by
// gdn_verify_fused_microtest, cos == 1.0).
//
// NOTE vs the K=2 twin: a third __syncthreads() is added after the L2 apply.
// The K=2 kernel reads warp_sums[base_warp] after its second barrier and the
// NEXT iteration's lane-0 write to the same slot has no intervening barrier —
// a loop-carried WAR hazard that gets 16 more chances to bite at K=17.
// Barriers don't change arithmetic, so bit-exactness is unaffected.

#include <cuda_bf16.h>

// ============================================================
// Fused conv1d + SiLU + L2-norm for ALL K verify positions, one launch.
//
// Grid: (ceil(dim/256), 1, 1)   Block: (256, 1, 1)
// Each thread owns one channel `ch` and processes positions 0..num_tokens-1,
// keeping the d_conv sliding window in registers between positions. After
// each position t the window is snapshotted to conv_state_inter +
// t*inter_stride (the rollback intermediate for accept-length t+1). The
// committed (post final-position) state is left in conv_state.
// ============================================================
extern "C" __global__ void gdn_verify_fused_conv_kn(
    float* __restrict__ conv_state,              // [dim, d_conv] FP32 (in/out)
    const __nv_bfloat16* __restrict__ new_input, // [K, input_stride] BF16
    const __nv_bfloat16* __restrict__ weight,    // [dim, d_conv] BF16
    __nv_bfloat16* __restrict__ output,          // [K, output_stride] BF16
    float* __restrict__ conv_state_inter,        // [K, inter_stride] FP32 (out, per-token snapshots)
    unsigned int num_tokens,     // K (17 for DFlash γ=16 verify)
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,    // channels 0..qk_channels-1 get L2 normalized
    unsigned int head_dim,       // L2 norm group size (128)
    unsigned int input_stride,   // BF16 elems between positions in new_input
    unsigned int output_stride,  // BF16 elems between positions in output
    unsigned int inter_stride,   // FP32 elems between snapshots in conv_state_inter
    float l2_eps
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);
    const bool valid = (ch < dim);

    // ── Load this channel's d_conv-element sliding window into registers ──
    // d_conv is small (4); a fixed-size register window matches the global
    // shift loop in causal_conv1d_update_l2norm exactly.
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

        if (valid) output[t * output_stride + ch] = __float2bfloat16(silu);

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
