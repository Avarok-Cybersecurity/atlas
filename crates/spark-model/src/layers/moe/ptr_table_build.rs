// SPDX-License-Identifier: AGPL-3.0-only

//! Device-side expert pointer-table builders (NVFP4 / BF16 / FP8).
//!
//! Extracted from `mod.rs` (Wave: ARM-2 native-MXFP4) to keep it under the
//! 500-LoC cap. One device pointer array per projection across all experts,
//! consumed by the batched/grouped MoE GEMMs. Re-exported from `mod.rs`
//! (`pub(crate) use ptr_table_build::*`), so all call sites are unchanged.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{ExpertPtrTable, Fp8ExpertPtrTable};
use crate::weight_map::{DenseWeight, ExpertWeight, Fp8ExpertWeight, Fp8Weight, QuantizedWeight};

/// Build a device-side pointer table from pre-transposed QuantizedWeight vec.
pub(crate) fn build_ptr_table_from_qw(
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
pub(crate) fn build_ptr_table(
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
/// Build a device-side BF16 pointer table for one projection across all
/// experts. Used by the FP8-dequant-to-BF16 MoE path; one device pointer
/// per expert pointing at that expert's `[N, K]` BF16 weight buffer.
pub(crate) fn build_bf16_ptr_table(
    experts: &[DenseWeight],
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let n = experts.len();
    let weight_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| e.weight.0.to_le_bytes())
        .collect();
    let ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&weight_bytes, ptrs)?;
    Ok(ptrs)
}

pub(crate) fn build_fp8_ptr_table(
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
