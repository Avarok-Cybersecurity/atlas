// SPDX-License-Identifier: AGPL-3.0-only

//! BF16 GDN output projection shared by verification dispatch and its GPU gate.
use crate::layers::ops;
use crate::weight_map::DenseWeight;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

#[allow(clippy::too_many_arguments)]
pub(super) fn project(
    gpu: &dyn GpuBackend,
    gemv: KernelHandle,
    gemm: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    rows: usize,
    hidden: usize,
    value_dim: usize,
    row_exact: bool,
    stream: u64,
) -> Result<()> {
    // Verification must reproduce the serial decode reduction order.
    // A K-row GEMM can round differently even with identical BF16 inputs.
    if row_exact {
        for row in 0..rows {
            ops::dense_gemv(
                gpu,
                gemv,
                input.offset(row * value_dim * 2),
                weight,
                output.offset(row * hidden * 2),
                hidden as u32,
                value_dim as u32,
                stream,
            )?;
        }
        return Ok(());
    }
    ops::dense_gemm(
        gpu,
        gemm,
        input,
        weight,
        output,
        rows as u32,
        hidden as u32,
        value_dim as u32,
        stream,
    )
}

#[cfg(all(test, feature = "cuda"))]
#[path = "bf16_verify_out_tests.rs"]
mod tests;
