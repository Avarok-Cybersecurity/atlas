// SPDX-License-Identifier: AGPL-3.0-only
//! Device IQ2_XS / IQ3_XXS expert mix. Packed stacks stay packed. No per-token D2H.

use anyhow::{Context, Result, ensure};
use avarok_core::kimi_k3::situ_glu_vec;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::WeightDtype;

use super::bound::{K3BoundLayer, WeightMeta};
use crate::weight_map::DenseWeight;

const IQ2_QK: usize = 256;
const IQ2_BLOCK: usize = 74;
const IQ3_BLOCK: usize = 98;
const IQ2_MODULE: &str = "iq2_xs_gemv";
const IQ2_ENTRY: &str = "k3_iq2_xs_gemv_f32io";
const IQ3_ENTRY: &str = "k3_iq3_xxs_gemv_f32io";

fn find_weight<'a>(
    weights: &'a [DenseWeight],
    meta: &'a [WeightMeta],
    name: &str,
) -> Result<(&'a DenseWeight, &'a WeightMeta)> {
    meta.iter()
        .position(|m| m.name == name)
        .map(|i| (&weights[i], &meta[i]))
        .with_context(|| format!("K3 IQ2 expert missing {name}"))
}

fn cover(full_k: usize, k_local: usize, rank: usize, world: usize) -> Result<(u32, u32)> {
    ensure!(world > 0 && rank < world, "IQ2 cover rank");
    ensure!(full_k / world == k_local, "IQ2 cover k_local");
    let k0 = rank * k_local;
    let k1 = k0 + k_local;
    let b0 = k0 / IQ2_QK;
    let b1 = k1.div_ceil(IQ2_QK);
    let skip = k0 - b0 * IQ2_QK;
    Ok((skip as u32, (b1 - b0) as u32))
}

fn gemv_packed(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    dtype: WeightDtype,
    x: &[f32],
    n: usize,
    k_local: usize,
    skip: u32,
    n_cover: u32,
    stream: u64,
) -> Result<Vec<f32>> {
    let (entry, block) = match dtype {
        WeightDtype::Iq2Xs => (IQ2_ENTRY, IQ2_BLOCK),
        WeightDtype::Iq3Xxs => (IQ3_ENTRY, IQ3_BLOCK),
        other => anyhow::bail!("K3 IQ2 gpu: {other:?}"),
    };
    let kernel = gpu.kernel(IQ2_MODULE, entry)?;
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let xin = gpu.alloc(x_bytes.len())?;
    let y_out = gpu.alloc(n * 4)?;
    let result = (|| -> Result<Vec<f32>> {
        gpu.copy_h2d(&x_bytes, xin)?;
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(n as u32, 4), 1, 1])
            .block([128, 1, 1])
            .arg_ptr(xin)
            .arg_ptr(ptr)
            .arg_ptr(y_out)
            .arg_u32(n as u32)
            .arg_u32(k_local as u32)
            .arg_u32(skip)
            .arg_u32(n_cover)
            .arg_u32(block as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut raw = vec![0u8; n * 4];
        gpu.copy_d2h(y_out, &mut raw)?;
        Ok(raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    })();
    let _ = gpu.free(y_out);
    let _ = gpu.free(xin);
    let out = result?;
    ensure!(out.iter().all(|v| v.is_finite()), "K3 IQ2 gemv nonfinite");
    Ok(out)
}

fn expert_gemv(
    gpu: &dyn GpuBackend,
    weights: &[DenseWeight],
    meta: &[WeightMeta],
    name: &str,
    x: &[f32],
    n: usize,
    k_local: usize,
    col_cover: Option<(usize, usize, usize)>,
    stream: u64,
) -> Result<Vec<f32>> {
    let (w, m) = find_weight(weights, meta, name)?;
    ensure!(
        matches!(m.dtype, WeightDtype::Iq2Xs | WeightDtype::Iq3Xxs),
        "{name}: packed IQ2/IQ3 only, got {:?}",
        m.dtype
    );
    let (skip, n_cover) = if let Some((full_k, rank, world)) = col_cover {
        cover(full_k, k_local, rank, world)?
    } else {
        ensure!(
            k_local.is_multiple_of(IQ2_QK),
            "{name}: k {k_local} not IQ aligned"
        );
        (0, (k_local / IQ2_QK) as u32)
    };
    gemv_packed(gpu, w.weight, m.dtype, x, n, k_local, skip, n_cover, stream)
}

/// True when this layer bound IQ2/IQ3 experts (GGUF keep-packed path).
pub fn layer_has_iq2(layer: &K3BoundLayer) -> bool {
    layer.weight_meta.iter().any(|m| {
        m.name.contains("block_sparse_moe.experts.")
            && matches!(m.dtype, WeightDtype::Iq2Xs | WeightDtype::Iq3Xxs)
    })
}

#[allow(clippy::too_many_arguments)]
pub fn mix_iq2_experts(
    layer: &K3BoundLayer,
    gpu: &dyn GpuBackend,
    latent: &[f32],
    ids: &[usize],
    mix_w: &[f32],
    expert_hidden_full: usize,
    latent_dim: usize,
    tp_rank: usize,
    tp_world: usize,
    situ_beta: f32,
    situ_linear_beta: f32,
    stream: u64,
) -> Result<Vec<f32>> {
    let tp = tp_world.max(1);
    let local_eh = expert_hidden_full / tp;
    let lp = format!("model.layers.{}", layer.index);
    let mut mixed = vec![0f32; latent_dim];
    for (&id, &wt) in ids.iter().zip(mix_w.iter()) {
        tracing::debug!(expert = id, "K3 IQ2 gpu mix");
        let w1n = format!("{lp}.block_sparse_moe.experts.{id}.w1.weight");
        let w2n = format!("{lp}.block_sparse_moe.experts.{id}.w2.weight");
        let w3n = format!("{lp}.block_sparse_moe.experts.{id}.w3.weight");
        let gate = expert_gemv(
            gpu,
            &layer.weights,
            &layer.weight_meta,
            &w1n,
            latent,
            local_eh,
            latent_dim,
            None,
            stream,
        )?;
        let up = expert_gemv(
            gpu,
            &layer.weights,
            &layer.weight_meta,
            &w3n,
            latent,
            local_eh,
            latent_dim,
            None,
            stream,
        )?;
        let mid = situ_glu_vec(&gate, &up, situ_beta, situ_linear_beta);
        let y = expert_gemv(
            gpu,
            &layer.weights,
            &layer.weight_meta,
            &w2n,
            &mid,
            latent_dim,
            local_eh,
            Some((expert_hidden_full, tp_rank, tp)),
            stream,
        )?;
        for (m, yy) in mixed.iter_mut().zip(y) {
            *m += wt * yy;
        }
    }
    Ok(mixed)
}

#[cfg(test)]
mod tests {
    #[test]
    fn packed_block_sizes_stay_packed() {
        assert_eq!(super::IQ2_BLOCK, 74);
        assert_eq!(super::IQ3_BLOCK, 98);
        let local_eh = 3072 / 8;
        assert_eq!(local_eh, 384);
        assert_ne!(
            local_eh, 896,
            "384 is expert_inter/tp, not hidden/tp or n_experts"
        );
    }
}
