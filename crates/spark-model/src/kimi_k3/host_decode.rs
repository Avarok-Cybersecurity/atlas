// SPDX-License-Identifier: AGPL-3.0-only

//! Copy-out / CPU mixer+MLP+AttnRes / copy-in for [`super::bound::K3BoundLayer`].
//!
//! Explicit **CPU fallback GPU wrapper**. Not CUDA KDA.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use atlas_core::kimi_k3::{
    Ablation, AttnResStream, K3CpuLayer, K3LayerCtx, assemble_layer, forward_one_layer,
};
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightDtype;

use super::bound::K3BoundLayer;
use super::state::K3CpuFallbackState;
use crate::layer::{ForwardContext, LayerState};

impl K3BoundLayer {
    pub(super) fn decode_cpu_fallback(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        seq_len: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = ctx.config.hidden_size;
        let st = state
            .as_any_mut()
            .downcast_mut::<K3CpuFallbackState>()
            .context("K3 decode: expected K3CpuFallbackState (uses_ssm_pool=false)")?;
        let ablation = Ablation::from_env();
        let layer = self.host_layer(gpu)?;
        let lctx = K3LayerCtx {
            kda: &self.shared.kda,
            mla: &self.shared.mla,
            moe: &self.shared.moe,
            situ_beta: self.shared.graph.situ_beta,
            situ_linear_beta: self.shared.graph.situ_linear_beta,
            hidden: self.shared.graph.hidden,
            dense_intermediate: self.shared.config.intermediate_size,
            eps: ctx.config.rms_norm_eps as f32,
            rope_theta: ctx.config.rope_theta as f32,
        };

        let key = if residual.is_null() { hidden } else { residual };
        {
            let mut hub = self.shared.attnres.lock();
            if self.index == 0 {
                let hidden_f32 = hidden_to_f32(gpu, hidden, h, stream)?;
                let mut s = AttnResStream::new(h, self.shared.graph.attn_res_block_size);
                s.partial.clone_from(&hidden_f32);
                hub.insert(key, s);
            } else {
                gpu.synchronize(stream)?;
            }
            let stream_res = hub.get_mut(&key).with_context(|| {
                format!(
                    "K3 AttnRes missing at layer {} (layer 0 must run)",
                    self.index
                )
            })?;
            // `seq_len` is the 0-based *position* (`TransformerLayer::decode`).
            // Prefill already walks tokens in `prefill_default` (one decode per
            // token, KDA/MLA step once). Do not treat this as packed N — looping
            // `seq_len` times would step KDA N times on one hidden row.
            forward_one_layer(&lctx, layer, seq_len, &mut st.cache, stream_res, ablation);
        }

        let n_layers = self.shared.graph.layers.len();
        let out = if self.index + 1 == n_layers {
            let (proj, norm) = self.output_res(gpu)?;
            let mut hub = self.shared.attnres.lock();
            let stream_res = hub
                .remove(&key)
                .context("K3 AttnRes missing at last layer")?;
            stream_res.mix(proj, norm, lctx.eps, ablation.attnres_mix)
        } else {
            let hub = self.shared.attnres.lock();
            hub.get(&key)
                .map(|s| s.partial.clone())
                .context("K3 AttnRes missing after mixer")?
        };
        f32_to_hidden(gpu, hidden, &out, stream)?;
        Ok(())
    }

    fn host_layer(&self, gpu: &dyn GpuBackend) -> Result<&K3CpuLayer> {
        if let Some(l) = self.host.get() {
            return Ok(l);
        }
        let bound = bind_layer(self, gpu)?;
        let _ = self.host.set(bound);
        self.host.get().context("K3 host layer OnceLock")
    }

    fn output_res(&self, gpu: &dyn GpuBackend) -> Result<(&[f32], &[f32])> {
        if self.shared.output_host.get().is_none() {
            let (dt, n) = self.shared.output_res_proj_meta;
            let proj = copy_weight_f32(gpu, self.shared.output_res_proj.weight, dt, n)?;
            let (dt, n) = self.shared.output_res_norm_meta;
            let norm = copy_weight_f32(gpu, self.shared.output_res_norm.weight, dt, n)?;
            let _ = self.shared.output_host.set((proj, norm));
        }
        let pair = self
            .shared
            .output_host
            .get()
            .context("K3 output_attn_res host")?;
        Ok((&pair.0, &pair.1))
    }
}

fn bind_layer(layer: &K3BoundLayer, gpu: &dyn GpuBackend) -> Result<K3CpuLayer> {
    let mut got = HashMap::new();
    for (w, meta) in layer.weights.iter().zip(&layer.weight_meta) {
        got.insert(
            meta.name.clone(),
            copy_weight_f32(gpu, w.weight, meta.dtype, meta.numel)?,
        );
    }
    assemble_layer(
        &layer.shared.config.weight_prefix,
        &layer.spec,
        &layer.shared.config,
        &layer.shared.moe,
        &mut got,
    )
}

fn hidden_to_f32(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize, stream: u64) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h_on_stream(ptr, &mut raw, stream)?;
    Ok(raw
        .chunks_exact(2)
        .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect())
}

fn f32_to_hidden(gpu: &dyn GpuBackend, ptr: DevicePtr, v: &[f32], stream: u64) -> Result<()> {
    let raw: Vec<u8> = v
        .iter()
        .flat_map(|&f| bf16::from_f32(f).to_le_bytes())
        .collect();
    gpu.copy_h2d_async(&raw, ptr, stream)?;
    gpu.synchronize(stream)?;
    Ok(())
}

fn copy_weight_f32(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    dtype: WeightDtype,
    numel: usize,
) -> Result<Vec<f32>> {
    match dtype {
        WeightDtype::FP32 => {
            let mut b = vec![0u8; numel * 4];
            gpu.copy_d2h(ptr, &mut b)?;
            Ok(b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        WeightDtype::BF16 => {
            let mut b = vec![0u8; numel * 2];
            gpu.copy_d2h(ptr, &mut b)?;
            Ok(b.chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        other => bail!("K3 CPU fallback GPU wrapper: unsupported weight dtype {other:?}"),
    }
}
