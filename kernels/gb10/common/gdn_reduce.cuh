// SPDX-License-Identifier: AGPL-3.0-only
//
// Shared reduction primitives for Gated Delta Net (GDN) kernels.
//
// Purpose: guarantee bit-identical FP32 reduction order across the per-token
// baseline (`gated_delta_rule.cu`) and the WY-chunkwise fast paths
// (`gated_delta_rule_wy{,3,4}.cu`). BF16 is non-associative, so a different
// FP reduction tree produces a few-ulp drift in the final `gdn_out` that
// flips the argmax on boundary tokens during MTP verify — the 80B-MTP fib
// `return b → return a/return n` bug reported in pass-6/pass-7/pass-8.
//
// Both the baseline and the fast paths must call the exact same primitives
// declared here so the compiler emits the identical PTX reduction sequence
// in every kernel.
//
// Declared `static __device__ __forceinline__` to give internal linkage per
// translation unit — no ODR violations when included from multiple .cu
// files.

#ifndef ATLAS_GDN_REDUCE_CUH
#define ATLAS_GDN_REDUCE_CUH

// ── Warp-level tree reduction (32 lanes) ──
// Pattern: __shfl_down_sync with offsets 16,8,4,2,1. Matches
// `gated_delta_rule.cu` line 173-177 (baseline).
static __device__ __forceinline__ float atlas_warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val,  8);
    val += __shfl_down_sync(0xFFFFFFFF, val,  4);
    val += __shfl_down_sync(0xFFFFFFFF, val,  2);
    val += __shfl_down_sync(0xFFFFFFFF, val,  1);
    return val;
}

// ── Block-level tree reduction for 128-thread blocks (4 warps) ──
//
// Matches the per-token baseline at `gated_delta_rule.cu:179-193` exactly:
//   1. Each warp reduces its 32 lanes via `atlas_warp_reduce_sum`.
//   2. Lane 0 of each warp writes to smem_warp[warp_id] (4 slots).
//   3. Lanes 0..3 of warp 0 do a shuffle-based tree: each lane loads its slot,
//      then `s += __shfl_down(s, 2)` followed by `s += __shfl_down(s, 1)`.
//      On lane 0 this yields `(a+c) + (b+d)` where a,b,c,d = smem_warp[0..3].
//      Thread 0 stores the result back to smem_warp[0] (sentinel) and the
//      whole block reads `smem_warp[0]` after a second __syncthreads().
//
// This is bit-identical to the baseline when called with the same FP32
// inputs and the same warp count (4). BF16 non-associativity is thereby
// eliminated across the per-token kernel and the wy{2,3,4} fast paths.
//
// `smem_warp` must have at least 4 slots and is reused as the return buffer.
static __device__ __forceinline__ float atlas_block_reduce_sum(
    float val,
    float* smem_warp,
    unsigned int tid
) {
    val = atlas_warp_reduce_sum(val);
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid & 31;
    if (lane_id == 0) smem_warp[warp_id] = val;
    __syncthreads();
    if (tid < 4) {
        float s = smem_warp[tid];
        s += __shfl_down_sync(0xf, s, 2);
        s += __shfl_down_sync(0xf, s, 1);
        if (tid == 0) smem_warp[0] = s;
    }
    __syncthreads();
    return smem_warp[0];
}

#endif // ATLAS_GDN_REDUCE_CUH
