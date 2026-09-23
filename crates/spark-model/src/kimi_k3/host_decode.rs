// SPDX-License-Identifier: AGPL-3.0-only

//! Copy-out / mixer+MLP+AttnRes / copy-in for [`super::bound::K3BoundLayer`].
//!
//! `MixerKind::Kda` runs conv+recurrent via [`launch_k3_kda_decode_token`]
//! unless `K3_CUDA_KDA=0`. `MixerKind::Mla` runs rope+SDPA+gate via
//! [`launch_k3_mla_decode_token_on_device`] unless `K3_CUDA_MLA=0`. Packed
//! `MlpKind::LatentMoe` experts launch [`launch_k3_latent_moe_experts`].

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use avarok_core::kimi_k3::{
    Ablation, AttnResStream, HiddenReduce, K3CpuLayer, K3LayerCtx, MixerKind, MlpKind,
    assemble_layer, forward_one_layer_with_cores, kda_decode_token, mix_routed_experts,
    mla_decode_token,
};
use half::bf16;
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightDtype;

use super::bound::K3BoundLayer;
use super::kda_cuda::{K3KdaDecodeKernels, launch_k3_kda_decode_token_on_device};
use super::mla_cuda::{K3MlaDecodeKernels, launch_k3_mla_decode_token_on_device};
use super::moe_cuda::{
    E8M0_ENTRY, K3MoeGemmKernels, MODULE as MOE_MODULE, launch_k3_latent_moe_experts,
};
use super::state::K3CpuFallbackState;
use crate::layer::{ForwardContext, LayerState};

impl K3BoundLayer {
    /// Serve path passes [`cuda_kda_enabled`] / [`cuda_mla_enabled`]. Tests
    /// pass the flags explicitly.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_host(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        seq_len: usize,
        ctx: &ForwardContext,
        stream: u64,
        want_cuda_kda: bool,
        want_cuda_mla: bool,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = ctx.config.hidden_size;
        let st = state
            .as_any_mut()
            .downcast_mut::<K3CpuFallbackState>()
            .context("K3 decode: expected K3CpuFallbackState (uses_ssm_pool=false)")?;
        let ablation = Ablation::from_env();
        if !self.mxfp4_experts.is_empty() && self.spec.mlp == MlpKind::LatentMoe {
            let _ = self.moe_kernels(gpu)?;
        }
        let layer = self.host_layer(gpu)?;
        let tp = ctx.config.tp_world_size.max(1);
        let comm = ctx.comm;
        let do_reduce = |v: &mut [f32]| tp_allreduce(gpu, comm, hidden, h, tp, v, stream);
        let reduce_ref: HiddenReduce<'_> = &do_reduce;
        let default_gpu = !cfg!(test);
        let parse_flag = |name: &str, default: bool| -> anyhow::Result<bool> {
            match std::env::var(name).as_deref() {
                Ok("0") => Ok(false),
                Ok("1") => Ok(true),
                Err(std::env::VarError::NotPresent) => Ok(default),
                _ => anyhow::bail!("{name} requires explicit 0 or 1"),
            }
        };
        let use_dense = parse_flag("K3_CUDA_DENSE", default_gpu)?;
        let use_gpu_gemv = parse_flag("K3_CUDA_GEMV", default_gpu)?;
        let use_iq2_gpu = parse_flag("K3_CUDA_IQ2", default_gpu)?;
        let gpu_gemv_core = |op: &str, x: &[f32], n: usize, k: usize| {
            super::gpu_gemv::launch(self, gpu, op, x, n, k, stream)
        };
        let dense_core = |_weights: &avarok_core::kimi_k3::cpu_weights::DenseMlp,
                          x: &[f32],
                          hidden: usize,
                          inter: usize,
                          beta: f32,
                          linear_beta: f32| {
            super::dense_cuda::launch_dense_mlp(
                self,
                gpu,
                x,
                hidden,
                inter,
                beta,
                linear_beta,
                stream,
            )
        };
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
            reduce_hidden: Some(reduce_ref),
            dense_mlp: if use_dense { Some(&dense_core) } else { None },
            gpu_gemv: if use_gpu_gemv {
                Some(&gpu_gemv_core)
            } else {
                None
            },
            shared_intermediate: ctx.config.shared_expert_intermediate_size,
            tp_rank: ctx.config.tp_rank,
            tp_world: ctx.config.tp_world_size.max(1),
        };
        let use_cuda_kda = want_cuda_kda && self.spec.mixer == MixerKind::Kda;
        let use_cuda_mla = want_cuda_mla && self.spec.mixer == MixerKind::Mla;
        let use_cuda_moe = !self.mxfp4_experts.is_empty() && self.spec.mlp == MlpKind::LatentMoe;

        let key = if residual.is_null() { hidden } else { residual };
        let result = (|| -> Result<()> {
            if use_cuda_kda {
                st.ensure_device_kda(gpu)?;
            } else if let (Some(device), avarok_core::kimi_k3::LayerCache::Kda(host)) =
                (&st.device_kda, &mut st.cache)
            {
                // Preserve continuation if the explicit CPU escape is selected.
                gpu.synchronize(stream)?;
                device.download(gpu, host)?;
                st.release(gpu)?;
            }
            if use_cuda_mla {
                let host_seq = match &st.cache {
                    avarok_core::kimi_k3::LayerCache::Mla(kv) => kv.seq_len,
                    _ => 0,
                };
                let cap = ctx.config.serve_max_seq_len;
                ensure!(
                    cap > 0,
                    "K3 CUDA MLA: set serve_max_seq_len (from --max-seq-len) to bound resident KV"
                );
                st.ensure_device_mla(gpu, &self.shared.mla, cap.max(host_seq.max(1)))?;
            } else if let (Some(device), avarok_core::kimi_k3::LayerCache::Mla(host)) =
                (&st.device_mla, &mut st.cache)
            {
                gpu.synchronize(stream)?;
                device.download(gpu, host)?;
                st.release(gpu)?;
            }
            let device_kda = st.device_kda.as_ref();
            let mut device_mla = st.device_mla.as_mut();
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
                let kda_k = if use_cuda_kda {
                    Some(self.kda_kernels(gpu)?)
                } else {
                    None
                };
                let mla_k = if use_cuda_mla {
                    Some(self.mla_kernels(gpu)?)
                } else {
                    None
                };
                let moe_k = if use_cuda_moe {
                    Some(self.moe_kernels(gpu)?)
                } else {
                    None
                };
                forward_one_layer_with_cores(
                    &lctx,
                    layer,
                    seq_len,
                    &mut st.cache,
                    stream_res,
                    ablation,
                    |x, w, g, b, cfg, kst| {
                        if let Some(k) = kda_k {
                            launch_k3_kda_decode_token_on_device(
                                gpu,
                                &k,
                                x,
                                w,
                                g,
                                b,
                                cfg,
                                device_kda.context("K3 CUDA KDA resident state missing")?,
                                stream,
                            )
                        } else {
                            Ok(kda_decode_token(x, w, g, b, cfg, kst))
                        }
                    },
                    |q, k, v, g, kv, cfg, pos, theta| {
                        if let Some(kern) = mla_k {
                            let device = device_mla
                                .as_mut()
                                .context("K3 CUDA MLA resident KV missing")?;
                            device.validate_cfg(cfg)?;
                            launch_k3_mla_decode_token_on_device(
                                gpu, &kern, q, k, v, g, device, cfg, pos, theta, stream,
                            )
                        } else {
                            Ok(mla_decode_token(q, k, v, g, kv, cfg, pos, theta))
                        }
                    },
                    |wts, latent, ids, mix_w, cfg| {
                        let mut mixed = if let Some(k) = moe_k {
                            launch_k3_latent_moe_experts(
                                gpu,
                                &k,
                                &self.mxfp4_experts,
                                latent,
                                ids,
                                mix_w,
                                cfg,
                                stream,
                            )?
                        } else if super::iq2_cuda::layer_has_iq2(self) && use_iq2_gpu {
                            super::iq2_cuda::mix_iq2_experts(
                                self,
                                gpu,
                                latent,
                                ids,
                                mix_w,
                                cfg.expert_hidden,
                                cfg.latent,
                                ctx.config.tp_rank,
                                ctx.config.tp_world_size.max(1),
                                cfg.situ_beta,
                                cfg.situ_linear_beta,
                                stream,
                            )?
                        } else if super::iq2_moe::layer_has_iq2(self) {
                            super::iq2_moe::mix_iq2_experts(
                                self,
                                gpu,
                                latent,
                                ids,
                                mix_w,
                                cfg.expert_hidden,
                                cfg.latent,
                                ctx.config.tp_rank,
                                ctx.config.tp_world_size.max(1),
                                cfg.situ_beta,
                                cfg.situ_linear_beta,
                            )?
                        } else {
                            mix_routed_experts(latent, ids, mix_w, &wts.experts, cfg)
                        };
                        // Expert w2 is row-parallel: allreduce latent before RMSNorm+up.
                        reduce_ref(&mut mixed)?;
                        Ok(mixed)
                    },
                )?;
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
        })();
        if result.is_err() {
            // Layer 0 inserts; last layer removes on success. A CUDA fail
            // mid-stack would otherwise leak until the next seq's layer 0
            // overwrites. Single-seq decode bounds the map; still drop.
            self.shared.attnres.lock().remove(&key);
        }
        result
    }

    fn kda_kernels(&self, gpu: &dyn GpuBackend) -> Result<K3KdaDecodeKernels> {
        if let Some(&k) = self.shared.kda_kernels.get() {
            return Ok(k);
        }
        let k = K3KdaDecodeKernels::resolve(gpu).context(
            "K3 CUDA KDA: kda_decode PTX missing (LinearAttention default). \
             Set K3_CUDA_KDA=0 for host kda_decode_token",
        )?;
        tracing::info!("K3 LinearAttention decode via CUDA kda_decode");
        Ok(*self.shared.kda_kernels.get_or_init(|| k))
    }

    fn mla_kernels(&self, gpu: &dyn GpuBackend) -> Result<K3MlaDecodeKernels> {
        if let Some(&k) = self.shared.mla_kernels.get() {
            return Ok(k);
        }
        let k = K3MlaDecodeKernels::resolve(gpu).context(
            "K3 CUDA MLA: mla_decode PTX missing (FullAttention default). \
             Set K3_CUDA_MLA=0 for host mla_decode_token",
        )?;
        tracing::info!("K3 FullAttention decode via CUDA mla_decode");
        Ok(*self.shared.mla_kernels.get_or_init(|| k))
    }

    fn moe_kernels(&self, gpu: &dyn GpuBackend) -> Result<K3MoeGemmKernels> {
        if let Some(&k) = self.shared.moe_kernels.get() {
            return Ok(k);
        }
        let k = K3MoeGemmKernels::resolve(gpu).with_context(|| {
            format!(
                "K3 packed LatentMoE: {MOE_MODULE}::{E8M0_ENTRY} missing; packed experts \
                 cannot silently run host F32"
            )
        })?;
        tracing::info!("K3 LatentMoE packed experts via CUDA moe_w4a16_grouped_gemm_ptrtable_e8m0");
        Ok(*self.shared.moe_kernels.get_or_init(|| k))
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

/// Row-parallel mixer `o_proj` / dense MLP `down` / expert `w2` leave a
/// partial sum. NCCL all-reduce is BF16 (same width as the GPU hidden).
fn tp_allreduce(
    gpu: &dyn GpuBackend,
    comm: Option<&dyn CommBackend>,
    scratch: DevicePtr,
    hidden: usize,
    tp: usize,
    v: &mut [f32],
    stream: u64,
) -> Result<()> {
    if tp <= 1 || v.is_empty() {
        return Ok(());
    }
    let comm = comm.context("K3 TP: CommBackend required when tp_size > 1")?;
    ensure!(
        v.len() <= hidden,
        "K3 TP allreduce len {} > hidden {hidden}",
        v.len()
    );
    f32_to_hidden(gpu, scratch, v, stream)?;
    comm.all_reduce_async(scratch.0, v.len() * 2, stream)?;
    gpu.synchronize(stream)?;
    let got = hidden_to_f32(gpu, scratch, v.len(), stream)?;
    v.copy_from_slice(&got);
    Ok(())
}

fn bind_layer(layer: &K3BoundLayer, gpu: &dyn GpuBackend) -> Result<K3CpuLayer> {
    let mut got = HashMap::new();
    for (w, meta) in layer.weights.iter().zip(&layer.weight_meta) {
        // IQ2 experts stay packed on GPU; assemble_moe leaves empty slots and
        // the mix callback dequants top-k on demand.
        if matches!(
            meta.dtype,
            WeightDtype::Iq2Xs | WeightDtype::Q8_0 | WeightDtype::Iq3Xxs
        ) {
            continue;
        }
        let gpu_linears = match std::env::var("K3_CUDA_GEMV").as_deref() {
            Ok("0") => false,
            Ok("1") => true,
            Err(std::env::VarError::NotPresent) => !cfg!(test),
            _ => anyhow::bail!("K3_CUDA_GEMV requires explicit 0 or 1"),
        };
        if gpu_linears
            && meta.dtype == WeightDtype::BF16
            && !super::gpu_gemv::host_keep_bf16(&meta.name)
        {
            got.insert(meta.name.clone(), Vec::new());
            continue;
        }
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
