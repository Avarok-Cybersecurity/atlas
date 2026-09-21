// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in K3 dense/shared MLP using original resident weights and FP32 I/O.
//! Only input and final output cross the host boundary; no weight duplication.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::WeightDtype;

use super::bound::K3BoundLayer;

fn dtype_tag(dtype: WeightDtype) -> Result<u32> {
    match dtype {
        WeightDtype::FP32 => Ok(0),
        WeightDtype::BF16 => Ok(1),
        other => anyhow::bail!("K3 dense GPU: unsupported resident weight dtype {other:?}"),
    }
}

struct Scratch<'a> {
    gpu: &'a dyn GpuBackend,
    pointers: Vec<DevicePtr>,
}
impl Scratch<'_> {
    fn alloc(&mut self, floats: usize) -> Result<DevicePtr> {
        let bytes = floats
            .checked_mul(4)
            .context("K3 dense scratch size overflow")?;
        let p = self.gpu.alloc(bytes)?;
        self.pointers.push(p);
        Ok(p)
    }
}
impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        for p in self.pointers.drain(..).rev() {
            let _ = self.gpu.free(p);
        }
    }
}

/// Dense and shared expert matrices are already TP-sharded in the loader.
#[allow(clippy::too_many_arguments)]
pub fn launch_dense_mlp(
    layer: &K3BoundLayer,
    gpu: &dyn GpuBackend,
    x: &[f32],
    hidden: usize,
    inter: usize,
    beta: f32,
    linear_beta: f32,
    stream: u64,
) -> Result<Vec<f32>> {
    ensure!(
        hidden > 0 && inter > 0 && x.len() == hidden,
        "K3 dense GPU: invalid activation geometry"
    );
    ensure!(
        beta.is_finite() && beta > 0.0 && linear_beta.is_finite() && linear_beta > 0.0,
        "K3 dense GPU: invalid SiTU coefficients"
    );
    let elements = hidden
        .checked_mul(inter)
        .context("K3 dense weight size overflow")?;
    ensure!(
        elements <= u32::MAX as usize,
        "K3 dense GPU: matrix exceeds 32-bit kernel indexing"
    );
    let prefix = match layer.spec.mlp {
        avarok_core::kimi_k3::MlpKind::Dense => ".mlp.",
        avarok_core::kimi_k3::MlpKind::LatentMoe => ".block_sparse_moe.shared_experts.",
    };
    let find = |role: &str| -> Result<_> {
        let suffix = format!("{prefix}{role}_proj.weight");
        let (weight, meta) = layer
            .weights
            .iter()
            .zip(&layer.weight_meta)
            .find(|(_, meta)| meta.name.ends_with(&suffix))
            .with_context(|| format!("K3 dense GPU: resident {suffix} missing"))?;
        ensure!(
            meta.numel == elements,
            "K3 dense GPU: {} has {} values, expected {elements}",
            meta.name,
            meta.numel
        );
        Ok((weight.weight, dtype_tag(meta.dtype)?))
    };
    let (gate, gate_dtype) = find("gate")?;
    let (up, up_dtype) = find("up")?;
    let (down, down_dtype) = find("down")?;
    ensure!(
        gate_dtype == up_dtype,
        "K3 dense GPU: gate/up dtypes differ"
    );
    launch_dense_matrices(
        gpu,
        x,
        gate,
        up,
        down,
        gate_dtype,
        down_dtype,
        hidden,
        inter,
        beta,
        linear_beta,
        stream,
    )
}

/// Explicit resident matrix ABI, also used by real-device numerical tests.
#[allow(clippy::too_many_arguments)]
pub fn launch_dense_matrices(
    gpu: &dyn GpuBackend,
    x: &[f32],
    gate: DevicePtr,
    up: DevicePtr,
    down: DevicePtr,
    gate_dtype: u32,
    down_dtype: u32,
    hidden: usize,
    inter: usize,
    beta: f32,
    linear_beta: f32,
    stream: u64,
) -> Result<Vec<f32>> {
    ensure!(
        hidden > 0 && inter > 0 && x.len() == hidden,
        "K3 dense GPU: invalid dimensions"
    );
    ensure!(
        hidden
            .checked_mul(inter)
            .is_some_and(|n| n <= u32::MAX as usize),
        "K3 dense GPU: index overflow"
    );
    ensure!(
        gate_dtype <= 1 && down_dtype <= 1,
        "K3 dense GPU: invalid dtype tag"
    );
    ensure!(
        beta.is_finite() && beta > 0.0 && linear_beta.is_finite() && linear_beta > 0.0,
        "K3 dense GPU: invalid activation coefficients"
    );
    let gate_kernel = gpu.kernel("dense_f32io", "k3_dense_gate_up_situ_f32io")?;
    let down_kernel = gpu.kernel("dense_f32io", "k3_dense_down_f32io")?;
    let mut scratch = Scratch {
        gpu,
        pointers: Vec::new(),
    };
    let input = scratch.alloc(hidden)?;
    let mid = scratch.alloc(inter)?;
    let output = scratch.alloc(hidden)?;
    gpu.copy_h2d(
        &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        input,
    )?;
    KernelLaunch::new(gpu, gate_kernel)
        .grid([div_ceil(inter as u32, 4), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(mid)
        .arg_u32(inter as u32)
        .arg_u32(hidden as u32)
        .arg_u32(gate_dtype)
        .arg_f32(beta)
        .arg_f32(linear_beta)
        .launch(stream)?;
    KernelLaunch::new(gpu, down_kernel)
        .grid([div_ceil(hidden as u32, 4), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(mid)
        .arg_ptr(down)
        .arg_ptr(output)
        .arg_u32(hidden as u32)
        .arg_u32(inter as u32)
        .arg_u32(down_dtype)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut raw = vec![0; hidden * 4];
    gpu.copy_d2h(output, &mut raw)?;
    let result: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        result.iter().all(|v| v.is_finite()),
        "K3 dense GPU: nonfinite output"
    );
    Ok(result)
}
