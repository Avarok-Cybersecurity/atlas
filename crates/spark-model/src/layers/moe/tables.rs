// SPDX-License-Identifier: AGPL-3.0-only

//! How an expert's weights are addressed on the device.
//!
//! Split from `moe/mod.rs` on the 500-line cap. These four types are the
//! layer's vocabulary rather than its behaviour: three descriptions of where
//! expert weights live, and the enum that lets the forward path pick a fused
//! kernel by matching on the quantisation it actually landed in — instead of
//! inferring it, which is how a shared expert exempted from quantisation ends
//! up silently read as though it were not.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use crate::weight_map::DenseWeight;

/// Device-side pointer table for one projection across all experts.
///
/// Enables GPU-side expert dispatch: the batched GEMV kernel reads
/// expert_id from device memory, then indexes these tables to find
/// the correct weight pointers — no CPU involvement needed.
pub(crate) struct ExpertPtrTable {
    /// `[num_experts]` u64 device pointers to each expert's B_packed.
    pub(crate) packed_ptrs: DevicePtr,
    /// `[num_experts]` u64 device pointers to each expert's B_scale.
    pub(crate) scale_ptrs: DevicePtr,
    /// `[num_experts]` f32 per-expert scale2 values.
    pub(crate) scale2_vals: DevicePtr,
}

/// Device-side pointer table for FP8 expert dispatch (one projection).
///
/// FP8 experts use 2 pointer arrays (weight + block_scale) instead of
/// NVFP4's 3 (packed + scale + scale2). The fused FP8 MoE kernel indexes
/// these tables by expert_id to load the correct FP8 weight matrix.
pub(crate) struct Fp8ExpertPtrTable {
    /// `[num_experts]` u64 device pointers to each expert's FP8 weight.
    pub(crate) weight_ptrs: DevicePtr,
    /// `[num_experts]` u64 device pointers to each expert's block scales.
    pub(crate) scale_ptrs: DevicePtr,
}

/// Checkpoint-native BF16 weights for a shared expert.
///
/// This is intentionally independent of routed-expert precision. Models such
/// as Laguna ship NVFP4 routed experts but explicitly exempt the shared expert
/// from quantization, so coupling these pointers to the all-BF16 routed path
/// silently changes model numerics.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Bf16SharedExpert {
    // `pub(super)`, not private: these were reachable from every sibling while
    // the type lived in `mod.rs`, and the split must not narrow that.
    pub(super) gate_proj: DenseWeight,
    pub(super) up_proj: DenseWeight,
    pub(super) down_proj: DenseWeight,
}

impl Bf16SharedExpert {
    pub(super) fn new(
        gate_proj: DenseWeight,
        up_proj: DenseWeight,
        down_proj: DenseWeight,
    ) -> Result<Self> {
        anyhow::ensure!(
            !gate_proj.weight.is_null() && !up_proj.weight.is_null() && !down_proj.weight.is_null(),
            "BF16 shared expert requires non-null gate/up/down weights"
        );
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
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
