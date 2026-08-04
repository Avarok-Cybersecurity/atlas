// SPDX-License-Identifier: AGPL-3.0-only

//! Device pointer tables for the per-expert NVFP4 weights consumed by the
//! grouped MoE GEMMs. Split from `nemotron_moe.rs` (500-LoC cap).

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::ExpertPtrTable;
use crate::weight_map::{NemotronExpertWeight, QuantizedWeight};

pub(super) fn build_ptr_table_from_weights(
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

pub(super) fn build_ptr_table(
    experts: &[NemotronExpertWeight],
    proj: impl Fn(&NemotronExpertWeight) -> &QuantizedWeight,
    gpu: &dyn GpuBackend,
) -> Result<ExpertPtrTable> {
    let n = experts.len();

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
