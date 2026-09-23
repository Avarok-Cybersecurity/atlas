// SPDX-License-Identifier: AGPL-3.0-only
//! On-demand IQ2_XS expert dequant for the K3 host MoE mix callback.

use anyhow::{Context, Result, ensure};
use avarok_core::kimi_k3::latent_moe::expert_situ;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightDtype;
use spark_runtime::weights::dequant_cpu::{iq2_xs, iq3_xxs};

use super::bound::{K3BoundLayer, WeightMeta};
use crate::weight_map::DenseWeight;

const IQ2_QK: usize = 256;
const IQ2_BLOCK: usize = 74;

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

fn d2h_bytes(gpu: &dyn GpuBackend, ptr: DevicePtr, nbytes: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; nbytes];
    gpu.copy_d2h(ptr, &mut buf)?;
    Ok(buf)
}

fn dequant_expert_iq2(
    gpu: &dyn GpuBackend,
    weights: &[DenseWeight],
    meta: &[WeightMeta],
    name: &str,
    out_rows: usize,
    inn: usize,
    col_cover: Option<(usize, usize, usize)>,
) -> Result<Vec<f32>> {
    let (w, m) = find_weight(weights, meta, name)?;
    ensure!(
        matches!(m.dtype, WeightDtype::Iq2Xs | WeightDtype::Iq3Xxs),
        "{name}: expected Iq2Xs/Iq3Xxs, got {:?}",
        m.dtype
    );
    ensure!(
        m.numel == out_rows * inn,
        "{name}: numel {} != {out_rows}*{inn}",
        m.numel
    );
    let block = match m.dtype {
        WeightDtype::Iq2Xs => IQ2_BLOCK,
        WeightDtype::Iq3Xxs => 98,
        _ => unreachable!(),
    };
    let mut f32s = vec![0f32; out_rows * inn];
    if let Some((full_k, rank, world)) = col_cover {
        ensure!(inn == full_k / world.max(1), "{name}: inn/cover mismatch");
        let k0 = rank * inn;
        let k1 = k0 + inn;
        let b0 = k0 / IQ2_QK;
        let b1 = k1.div_ceil(IQ2_QK);
        let n_cover = b1 - b0;
        let nbytes = out_rows * n_cover * block;
        let raw = d2h_bytes(gpu, w.weight, nbytes)?;
        match m.dtype {
            WeightDtype::Iq2Xs => iq2_xs::dequant_iq2_xs_column_cover(
                &raw, out_rows, full_k, inn, rank, world, &mut f32s,
            )?,
            WeightDtype::Iq3Xxs => iq3_xxs::dequant_iq3_xxs_column_cover(
                &raw, out_rows, full_k, inn, rank, world, &mut f32s,
            )?,
            _ => unreachable!(),
        }
    } else {
        ensure!(
            inn.is_multiple_of(IQ2_QK),
            "{name}: inn={inn} not IQ aligned"
        );
        let nbytes = out_rows * (inn / IQ2_QK) * block;
        let raw = d2h_bytes(gpu, w.weight, nbytes)?;
        match m.dtype {
            WeightDtype::Iq2Xs => iq2_xs::dequant_iq2_xs_tensor(&raw, out_rows, inn, &mut f32s)?,
            WeightDtype::Iq3Xxs => iq3_xxs::dequant_iq3_xxs_tensor(&raw, out_rows, inn, &mut f32s)?,
            _ => unreachable!(),
        }
    }
    Ok(f32s)
}

/// True when this layer bound IQ2 experts (GGUF keep-packed path).
pub fn layer_has_iq2(layer: &K3BoundLayer) -> bool {
    layer.weight_meta.iter().any(|m| {
        m.name.contains("block_sparse_moe.experts.")
            && matches!(m.dtype, WeightDtype::Iq2Xs | WeightDtype::Iq3Xxs)
    })
}

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
) -> Result<Vec<f32>> {
    let tp = tp_world.max(1);
    let local_eh = expert_hidden_full / tp;
    let lp = format!("model.layers.{}", layer.index);
    let mut mixed = vec![0f32; latent_dim];
    for (&id, &wt) in ids.iter().zip(mix_w.iter()) {
        let w1n = format!("{lp}.block_sparse_moe.experts.{id}.w1.weight");
        let w2n = format!("{lp}.block_sparse_moe.experts.{id}.w2.weight");
        let w3n = format!("{lp}.block_sparse_moe.experts.{id}.w3.weight");
        let w1 = dequant_expert_iq2(
            gpu,
            &layer.weights,
            &layer.weight_meta,
            &w1n,
            local_eh,
            latent_dim,
            None,
        )?;
        let w3 = dequant_expert_iq2(
            gpu,
            &layer.weights,
            &layer.weight_meta,
            &w3n,
            local_eh,
            latent_dim,
            None,
        )?;
        let w2 = dequant_expert_iq2(
            gpu,
            &layer.weights,
            &layer.weight_meta,
            &w2n,
            latent_dim,
            local_eh,
            Some((expert_hidden_full, tp_rank, tp)),
        )?;
        let y = expert_situ(
            latent,
            &w1,
            &w2,
            &w3,
            latent_dim,
            local_eh,
            situ_beta,
            situ_linear_beta,
        );
        for (m, yy) in mixed.iter_mut().zip(y) {
            *m += wt * yy;
        }
    }
    Ok(mixed)
}
