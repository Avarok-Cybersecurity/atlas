// SPDX-License-Identifier: AGPL-3.0-only
//
// Atlas fused causal-conv1d update + SiLU + per-head L2-norm —
// MSL port of `kernels/gb10/common/causal_conv1d.cu::
// causal_conv1d_update_l2norm`. Same name + arg layout so the
// existing spark-model `ops::conv1d_update_l2norm` orchestration
// dispatches through the metal backend.
//
// Per channel `ch` and batch `b`:
//   1. shift conv_state[ch][0..d_conv-1] left by 1, append new_input[ch]
//   2. acc = sum_k(conv_state[ch][k] * weight[ch][k])    (no bias here —
//                                                         caller passes
//                                                         null and bias
//                                                         path is dropped)
//   3. silu = acc * sigmoid(acc)
//   4. for channels 0..qk_channels: per-head L2-normalize silu
//      (head_dim threads cooperate: sum_sq → rsqrt → multiply)
//   5. output[ch] = bf16(silu)
//
// V channels (qk_channels..dim) get SiLU only (no L2 norm).
//
// State is FP32 (vs BF16) — prevents recurrent precision drift past
// 8K tokens that BF16 truncation introduces.
//
// Layout:
//   conv_state : float  [batch, dim, d_conv]                (in/out)
//   new_input  : bfloat [batch, dim]
//   weight     : bfloat [dim, d_conv]
//   output     : bfloat [batch, dim]
//
// Block: [block_x, 1, 1]; Grid: (dim / block_x, batch, 1).
// `block_x` MUST be a multiple of `head_dim` and ≥ 2 *
// `head_dim` so the per-head L2 reduction can fit in one block.

#include <metal_stdlib>
using namespace metal;

constant uint MAX_HEADS_PER_BLOCK = 4;
constant uint MAX_SIMDGROUPS_LN = 16;

kernel void causal_conv1d_update_l2norm(
    device float        *conv_state [[buffer(0)]],
    device const bfloat *new_input  [[buffer(1)]],
    device const bfloat *weight     [[buffer(2)]],
    device bfloat       *output     [[buffer(3)]],
    constant uint  &batch        [[buffer(4)]],
    constant uint  &dim          [[buffer(5)]],
    constant uint  &d_conv       [[buffer(6)]],
    constant uint  &qk_channels  [[buffer(7)]],
    constant uint  &head_dim     [[buffer(8)]],
    constant float &l2_eps       [[buffer(9)]],
    uint  tg_idx        [[threadgroup_position_in_grid]],
    uint  tid           [[thread_position_in_threadgroup]],
    uint  tg_size       [[threads_per_threadgroup]],
    uint  simd_lane     [[thread_index_in_simdgroup]],
    uint  simd_grp      [[simdgroup_index_in_threadgroup]])
{
    // Flat 1-D grid: caller dispatches `(dim / block_x) * batch`
    // threadgroups; we decode (block_x_idx, b) here.
    uint blocks_per_batch = (dim + tg_size - 1) / tg_size;
    uint block_x_idx = tg_idx % blocks_per_batch;
    uint b           = tg_idx / blocks_per_batch;

    uint block_start = block_x_idx * tg_size;
    uint ch = block_start + tid;
    bool block_needs_l2 = (block_start < qk_channels);

    // Threadgroup memory must be at function scope, not inside an
    // `if` block — Metal's spec is stricter than CUDA's __shared__.
    threadgroup float partial[MAX_SIMDGROUPS_LN];
    threadgroup float head_inv_norm[MAX_HEADS_PER_BLOCK];

    bool valid = (ch < dim && b < batch);
    float silu = 0.0f;

    // ── 1. Conv1d update + SiLU ─────────────────────────────────
    if (valid) {
        device float *state = conv_state + (b * dim + ch) * d_conv;
        for (uint i = 0; i + 1 < d_conv; ++i) {
            state[i] = state[i + 1];
        }
        state[d_conv - 1] = float(new_input[b * dim + ch]);

        device const bfloat *w = weight + ch * d_conv;
        float acc = 0.0f;
        for (uint k = 0; k < d_conv; ++k) {
            acc += state[k] * float(w[k]);
        }
        float sig = 1.0f / (1.0f + exp(-acc));
        silu = acc * sig;
    }

    // ── 2. Per-head L2 norm for Q+K channels ────────────────────
    // Each "head" group is `head_dim` consecutive threads. block_x
    // is a multiple of head_dim, so heads don't span block boundaries.
    if (block_needs_l2) {
        float sq = valid ? (silu * silu) : 0.0f;

        // simdgroup-level reduction (32-wide on Apple Silicon).
        float simd_sum_v = simd_sum(sq);
        if (simd_lane == 0) {
            partial[simd_grp] = simd_sum_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cross-simdgroup reduction within each head group.
        // simdgroups per head = head_dim / 32 (assuming head_dim ≥ 32).
        uint simds_per_head = head_dim / 32u;
        if (simds_per_head == 0u) simds_per_head = 1u;

        uint head_in_block = tid / head_dim;
        uint pos_in_head = tid % head_dim;
        if (pos_in_head == 0 && head_in_block < MAX_HEADS_PER_BLOCK) {
            float total = 0.0f;
            uint base_simd = head_in_block * simds_per_head;
            for (uint i = 0; i < simds_per_head; ++i) {
                total += partial[base_simd + i];
            }
            head_inv_norm[head_in_block] = rsqrt(total + l2_eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            silu *= head_inv_norm[head_in_block];
        }
    }

    if (valid) {
        output[b * dim + ch] = bfloat(silu);
    }
}
