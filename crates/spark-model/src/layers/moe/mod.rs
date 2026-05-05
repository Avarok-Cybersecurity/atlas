// SPDX-License-Identifier: AGPL-3.0-only

//! MoE (Mixture of Experts) FFN component.
//!
//! Batched expert dispatch: top-K experts run in 2 fused kernel launches
//! (gate+up, silu+down) instead of 10 × 5 individual launches. Expert indices
//! and weights stay on device — zero D2H synchronization.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{ExpertWeight, Fp8ExpertWeight, Fp8Weight, MoeWeights, QuantizedWeight};

/// Device-side pointer table for one projection across all experts.
///
/// Enables GPU-side expert dispatch: the batched GEMV kernel reads
/// expert_id from device memory, then indexes these tables to find
/// the correct weight pointers — no CPU involvement needed.
pub(crate) struct ExpertPtrTable {
    /// [num_experts] u64 device pointers to each expert's B_packed.
    pub(crate) packed_ptrs: DevicePtr,
    /// [num_experts] u64 device pointers to each expert's B_scale.
    pub(crate) scale_ptrs: DevicePtr,
    /// [num_experts] f32 per-expert scale2 values.
    pub(crate) scale2_vals: DevicePtr,
}

/// Device-side pointer table for FP8 expert dispatch (one projection).
///
/// FP8 experts use 2 pointer arrays (weight + block_scale) instead of
/// NVFP4's 3 (packed + scale + scale2). The fused FP8 MoE kernel indexes
/// these tables by expert_id to load the correct FP8 weight matrix.
pub(crate) struct Fp8ExpertPtrTable {
    /// [num_experts] u64 device pointers to each expert's FP8 weight.
    pub(crate) weight_ptrs: DevicePtr,
    /// [num_experts] u64 device pointers to each expert's block scales.
    pub(crate) scale_ptrs: DevicePtr,
}

/// Unified expert pointer table for any quantization format.
///
/// Replaces the separate `ExpertPtrTable` (NVFP4) and `Fp8ExpertPtrTable` (FP8)
/// with a single enum. The MoE forward path matches on this to select the
/// correct fused kernel (moe_shared_expert_fused vs moe_shared_expert_fused_fp8).
#[allow(dead_code)]
pub(crate) enum ExpertPtrSet {
    /// NVFP4: 3 pointer arrays (packed_ptrs, scale_ptrs, per-expert scale2 f32).
    Nvfp4 {
        packed_ptrs: DevicePtr,
        scale_ptrs: DevicePtr,
        scale2_vals: DevicePtr,
    },
    /// FP8: 2 pointer arrays (weight_ptrs, block_scale_ptrs).
    Fp8 {
        weight_ptrs: DevicePtr,
        scale_ptrs: DevicePtr,
    },
}

/// MoE feed-forward network component.
///
/// Not a `TransformerLayer` — used as a component inside layers
/// for the FFN/MoE block after post-attention norm.
#[allow(dead_code)]
pub struct MoeLayer {
    pub weights: MoeWeights,
    // NVFP4-quantized gate weight (quarters bandwidth for routing)
    gate_nvfp4: Option<QuantizedWeight>,
    /// Pre-expert norm: applied to input AFTER routing but BEFORE expert dispatch.
    /// Gemma-4 26B: router sees raw residual, experts see pre_feedforward_layernorm_2(residual).
    pub pre_expert_norm: Option<crate::weight_map::DenseWeight>,
    pre_expert_norm_k: spark_runtime::gpu::KernelHandle,
    dense_gemv: KernelHandle,
    w4a16_gemv: KernelHandle,
    w4a16_gemm: KernelHandle,
    dense_gemm: KernelHandle,
    moe_expert_gate_up_shared: KernelHandle,
    moe_expert_silu_down_shared: KernelHandle,
    moe_topk: KernelHandle,
    moe_weighted_sum_blend: KernelHandle,
    residual_add: KernelHandle,
    moe_topk_batched: KernelHandle,
    // K=2 fused MoE kernel handles
    moe_expert_gate_up_shared_batch2: KernelHandle,
    moe_expert_silu_down_shared_batch2: KernelHandle,
    moe_weighted_sum_blend_batch2: KernelHandle,
    w4a16_gemv_batch2: KernelHandle,
    // K=3 fused MoE kernel handles
    moe_expert_gate_up_shared_batch3: KernelHandle,
    moe_expert_silu_down_shared_batch3: KernelHandle,
    moe_weighted_sum_blend_batch3: KernelHandle,
    w4a16_gemv_batch3: KernelHandle,
    // Sorted/grouped prefill path
    moe_sort_by_expert: KernelHandle,
    moe_sorted_gate_up: KernelHandle,
    moe_sorted_silu_down: KernelHandle,
    moe_grouped_gemm: KernelHandle,
    moe_silu_mul: KernelHandle,
    /// Activation kernel for sorted/unfused path. SiLU by default, GeGLU for Gemma-4.
    moe_act_mul: KernelHandle,
    /// When true, decode uses the sorted prefill path (avoids fused SiLU kernels).
    gelu_activation: bool,
    moe_unpermute_reduce: KernelHandle,
    moe_batched_blend: KernelHandle,
    /// Pointer tables for batched expert dispatch.
    gate_ptrs: ExpertPtrTable,
    up_ptrs: ExpertPtrTable,
    down_ptrs: ExpertPtrTable,
    /// Transposed pointer tables for coalesced prefill GEMM.
    gate_ptrs_t: Option<ExpertPtrTable>,
    up_ptrs_t: Option<ExpertPtrTable>,
    down_ptrs_t: Option<ExpertPtrTable>,
    /// Lazy down_proj transpose scratch — populated at the start of each
    /// prefill call when the persistent transpose pass couldn't fit
    /// down_proj. Decode keeps using `down_ptrs` (untransposed); prefill
    /// uses `down_ptrs_t` pointing into this scratch. Shared across all
    /// MoE layers (the same scratch is overwritten layer-by-layer during
    /// the sequential forward).
    ///
    /// `down_t_scratch_packed`: contiguous `[num_experts × N × K/2]` bytes.
    /// `down_t_scratch_scale`:  contiguous `[num_experts × N × K/16]` bytes.
    /// Both `None` when the persistent transpose pass already covered
    /// down (full-fits path) or when the layer doesn't need scratch
    /// transpose (FP8 experts, etc.).
    down_t_scratch_packed: Option<DevicePtr>,
    down_t_scratch_scale: Option<DevicePtr>,
    /// Kernel handle for the batched per-expert uint8 transpose.
    moe_transpose_u8_batched_k: KernelHandle,
    // ── Phase 8a transposed-layout decode kernels (unified-layout MoE).
    // Loaded eagerly at construction. Currently NOT wired into the
    // dispatch — Phase 8a part 3/3 will route decode through these once
    // the weight loader produces transposed-only pointer tables.
    moe_expert_gate_up_shared_t_k: KernelHandle,
    moe_expert_silu_down_shared_t_k: KernelHandle,
    moe_expert_gate_up_shared_batch2_t_k: KernelHandle,
    moe_expert_silu_down_shared_batch2_t_k: KernelHandle,
    moe_expert_gate_up_shared_batch3_t_k: KernelHandle,
    moe_expert_silu_down_shared_batch3_t_k: KernelHandle,
    moe_expert_gate_up_shared_fp8_t_k: KernelHandle,
    moe_expert_silu_down_shared_fp8_t_k: KernelHandle,
    moe_expert_gate_up_shared_fp8_batch2_t_k: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch2_t_k: KernelHandle,
    moe_expert_gate_up_shared_fp8_batch3_t_k: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch3_t_k: KernelHandle,
    /// `ATLAS_UNIFIED_MOE_LAYOUT=1` opts in to the unified-layout decode
    /// path: gate/up/down all use transposed `[K/2, N]` layout, decode
    /// dispatches to `moe_expert_*_shared_t` kernels. Default off — the
    /// dispatch falls through to the original `[N, K/2]` kernels.
    /// Resolved once at construction.
    unified_layout: bool,
    /// `ATLAS_NVFP4_GATE_UP_M128=1` opts in to the M=128 fused gate+up
    /// kernel (Block D #3, Avarok tile-shape rewrite). Halves block count
    /// at large prefill — better SM amortization on GB10's 25-SM budget.
    /// Currently only minimax-m2-229b ships the kernel; other models keep
    /// `moe_fused_gate_up_t_k64_m128 == KernelHandle(0)` and dispatch
    /// falls through to the M=64 path even when the env var is set.
    nvfp4_gate_up_m128: bool,
    /// `ATLAS_HYBRID_MOE_LAYOUT=1` opts in to the hybrid-layout path:
    /// keep BOTH original `[N, K/2]` weights (for decode + MTP verify) AND
    /// transposed `[K/2, N]` weights (for prefill). Doubles MoE-weight
    /// memory but recovers the ~15 % decode regression that pure unified
    /// layout suffers from. Resolved once at construction; mutually
    /// exclusive with `unified_layout` at the dispatch level (hybrid wins
    /// on decode paths since it preserves untransposed warp-reduction
    /// parallelism).
    hybrid_layout: bool,
    /// Transposed shared expert weights for prefill.
    shared_gate_t: Option<QuantizedWeight>,
    shared_up_t: Option<QuantizedWeight>,
    shared_down_t: Option<QuantizedWeight>,
    moe_grouped_gemm_t: KernelHandle,
    moe_grouped_gemm_t_k64: KernelHandle,
    moe_fused_gate_up_t: KernelHandle,
    moe_fused_gate_up_t_k64: KernelHandle,
    /// M=128 variant of the K64 fused gate+up kernel (Block D #3, Avarok
    /// tile-shape rewrite). Loaded with `try_kernel` — falls back to
    /// `KernelHandle(0)` on models that don't ship the kernel; dispatch
    /// gates on `nvfp4_gate_up_m128` AND handle non-zero.
    moe_fused_gate_up_t_k64_m128: KernelHandle,
    moe_fp8_grouped_gemm_t: KernelHandle,
    w4a16_gemm_t: KernelHandle,
    bf16_to_fp8_k: KernelHandle,
    /// Pre-dequanted FP8 weights for zero-overhead prefill GEMMs.
    gate_fp8: Option<DevicePtr>,
    shared_gate_fp8: Option<DevicePtr>,
    shared_up_fp8: Option<DevicePtr>,
    shared_down_fp8: Option<DevicePtr>,
    fp8_gemm_k: KernelHandle,
    /// Secondary CUDA stream for overlapping shared expert with routed experts.
    prefill_stream: u64,
    /// Event pair for stream synchronization (input_ready, shared_done).
    event_a: u64,
    event_b: u64,
    // ── Sigmoid + correction-bias routing (DeepSeek-V3 / MiniMax-M2 style) ──
    /// Device pointer to `[num_experts]` correction bias. Populated from
    /// `MoeWeights.correction_bias` in `new()` when the loader sets it.
    /// `None` = Atlas's default softmax path. When `Some`, every top-k
    /// dispatch site branches to `moe_topk_sigmoid` with this bias arg.
    correction_bias_dev: Option<DevicePtr>,
    /// Handle to `moe_topk_sigmoid` kernel. Lazy-loaded in `new()` even
    /// when bias is `None` (harmless if kernel isn't used).
    moe_topk_sigmoid_k: KernelHandle,
    /// Batched variant for prefill / MTP-verify (one block per token).
    /// Loaded via `try_kernel` — returns KernelHandle(0) on models whose
    /// KERNEL.toml doesn't register the sigmoid kernels (e.g. Mistral).
    /// Never dispatched on those paths because `correction_bias_dev` is
    /// `None` there.
    moe_topk_sigmoid_batched_k: KernelHandle,
    // FP8 fused MoE kernels (used when experts are FP8)
    moe_expert_gate_up_shared_fp8: KernelHandle,
    moe_expert_silu_down_shared_fp8: KernelHandle,
    // FP8 batch2/3 fused MoE kernels (for MTP K=2/K=3 verify)
    moe_expert_gate_up_shared_fp8_batch2: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch2: KernelHandle,
    moe_weighted_sum_blend_fp8_batch2: KernelHandle,
    moe_expert_gate_up_shared_fp8_batch3: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch3: KernelHandle,
    moe_weighted_sum_blend_fp8_batch3: KernelHandle,
    // FP8 grouped GEMM for sorted MoE prefill
    moe_fp8_grouped_gemm_k: KernelHandle,
    // FP8 grouped GEMM v2 — coalesced B/A load thread-remap. Opt-in via
    // ATLAS_FP8_MOE_COALESCED=1 env var. Kernel handle may be 0 on older
    // images (then dispatch falls back to v1). Same signature as v1.
    moe_fp8_grouped_gemm_v2_k: KernelHandle,
    // Resolved once per layer from ATLAS_FP8_MOE_COALESCED env var.
    fp8_moe_coalesced_enabled: bool,
    w8a16_gemm_k: KernelHandle, // for shared expert FP8 prefill
    // Fused gate GEMV + topK softmax (saves 1 kernel launch per layer)
    moe_gate_topk_fused_k: KernelHandle,
    // FP8 expert pointer tables (None when experts are NVFP4)
    fp8_gate_weight_ptrs: Option<Fp8ExpertPtrTable>,
    fp8_up_weight_ptrs: Option<Fp8ExpertPtrTable>,
    fp8_down_weight_ptrs: Option<Fp8ExpertPtrTable>,
    // FP8 shared expert weights (None when shared expert is NVFP4)
    fp8_shared_expert: Option<Fp8ExpertWeight>,
    // Phase 2.7 Tier C — Frankenstein dispatch flag.
    // True when this layer's index is in `config.dflash_capture_layers`.
    // When the env var `ATLAS_FRANKENSTEIN_DECODE_VIA_PREFILL=1` is set,
    // `forward()` (single-token decode) will route through `forward_prefill`
    // (tensor-core grouped GEMM kernel) on this layer only, so the captured
    // hidden states use a different numerical recipe than the scalar GEMV
    // path. Used to test whether the kernel choice is the dominant cause
    // of low DFlash drafter acceptance on FP4/FP8 targets.
    pub is_dflash_capture_layer: bool,
}

// ── Sub-files (split for ≤500 LoC) ────────────────────────────────────────
mod forward;
mod forward_batched;
mod forward_ep;
mod forward_k2;
mod forward_k3;
mod forward_phase;
mod forward_prefill;
mod forward_prefill_fp8;
mod forward_prefill_phase;
mod helpers_a;
mod helpers_b;
mod helpers_c;
mod init;

/// Build a device-side pointer table from pre-transposed QuantizedWeight vec.
fn build_ptr_table_from_qw(
    weights: &[QuantizedWeight],
    gpu: &dyn GpuBackend,
) -> Result<ExpertPtrTable> {
    let n = weights.len();
    let packed_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight_scale.0.to_le_bytes())
        .collect();
    let scale2_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight_scale_2.to_le_bytes())
        .collect();

    let packed_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&packed_bytes, packed_ptrs)?;
    let scale_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;
    let scale2_vals = gpu.alloc(n * 4)?;
    gpu.copy_h2d(&scale2_bytes, scale2_vals)?;

    Ok(ExpertPtrTable {
        packed_ptrs,
        scale_ptrs,
        scale2_vals,
    })
}

/// Build a device-side pointer table for one projection across all experts.
fn build_ptr_table(
    experts: &[ExpertWeight],
    proj: impl Fn(&ExpertWeight) -> &crate::weight_map::QuantizedWeight,
    gpu: &dyn GpuBackend,
) -> Result<ExpertPtrTable> {
    let n = experts.len();

    // Build host-side arrays
    let packed_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight_scale.0.to_le_bytes())
        .collect();
    let scale2_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight_scale_2.to_le_bytes())
        .collect();

    // Upload to device
    let packed_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&packed_bytes, packed_ptrs)?;

    let scale_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;

    let scale2_vals = gpu.alloc(n * 4)?;
    gpu.copy_h2d(&scale2_bytes, scale2_vals)?;

    Ok(ExpertPtrTable {
        packed_ptrs,
        scale_ptrs,
        scale2_vals,
    })
}

/// Build a device-side FP8 pointer table for one projection across all experts.
///
/// FP8 experts store 2 arrays (weight + block_scale) per projection,
/// vs NVFP4's 3 (packed + scale + scale2).
fn build_fp8_ptr_table(
    experts: &[Fp8ExpertWeight],
    proj: impl Fn(&Fp8ExpertWeight) -> &Fp8Weight,
    gpu: &dyn GpuBackend,
) -> Result<Fp8ExpertPtrTable> {
    let n = experts.len();

    let weight_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).row_scale.0.to_le_bytes())
        .collect();

    let weight_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&weight_bytes, weight_ptrs)?;

    let scale_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;

    Ok(Fp8ExpertPtrTable {
        weight_ptrs,
        scale_ptrs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn test_moe_kernel_loading() {
        let gpu = MockGpuBackend::new();
        assert!(gpu.kernel("gemv", "dense_gemv_bf16").is_ok());
        assert!(gpu.kernel("w4a16_gemv", "w4a16_gemv").is_ok());
        assert!(gpu.kernel("moe_topk", "moe_topk_softmax").is_ok());
        assert!(
            gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_gate_up")
                .is_ok()
        );
        assert!(
            gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_gate_up_2x")
                .is_ok()
        );
        assert!(
            gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_silu_down")
                .is_ok()
        );
        assert!(
            gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_silu_down_2x")
                .is_ok()
        );
        assert!(
            gpu.kernel("moe_shared_expert_fused", "moe_expert_gate_up_shared")
                .is_ok()
        );
        assert!(
            gpu.kernel("moe_shared_expert_fused", "moe_expert_silu_down_shared")
                .is_ok()
        );
        assert!(
            gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")
                .is_ok()
        );
        // K=2 batch dispatch
        assert!(gpu.kernel("moe_topk", "moe_topk_softmax_batched").is_ok());
    }
}
