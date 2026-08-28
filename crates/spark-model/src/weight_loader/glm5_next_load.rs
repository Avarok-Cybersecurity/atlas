// SPDX-License-Identifier: AGPL-3.0-only

//! `Glm5NextWeightLoader` — assembles the 45-layer GLM-5.3 text stack from a `WeightStore`.
//!
//! Everything it calls already existed and was gated: `bind_kda_weights`, `build_dsa_weights`,
//! `glm5next_mlp::build`. This is the wiring, plus the one thing wiring must do that the pieces
//! cannot — decide, per layer, WHICH pieces.
//!
//! # Where the classification comes from
//!
//! Not from tensor names, and not from modular arithmetic. `Glm5NextTextSkeleton::from_config`
//! derives the mixer and MLP kind of all 45 layers from the checkpoint's own
//! `linear_attn_config` index lists and `first_k_dense_replace`, cross-checked against the
//! textual arrays, and refuses anything it was not taught. This loader iterates that.
//!
//! # TP, on every half
//!
//! DSA shards through `DsaTpPlan`, the MLP through `Glm5NextMlpConfig`, and KDA through
//! [`KdaShardedSource`] — an adapter that slices the host bytes **before** the proven
//! `bind_kda_weights` sees them, so TP=1 and TP=2 take the identical binder code path.
//!
//! 🪤 Both mixers end in a **row-parallel** `o_proj`, so the attention output is a partial sum
//! at TP>1 and `Glm5NextLayer::mixer_all_reduce` reduces it before the mHC highway sees it.
//! Half-applying the sharding — the state before this was wired — meant every rank computed a
//! WHOLE KDA block and the all-reduce double-counted it: no crash, no shape error.

use anyhow::{Context, Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::glm5next_dsa::build::build_dsa_weights;
use crate::layers::glm5next_dsa::layer::{Glm5NextDsaLayer, Glm5NextDsaLayerKernels};
use crate::layers::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use crate::layers::glm5next_kda::binding::{
    KdaDtype, KdaTensorSource, RawTensor, bind_kda_weights,
};
use crate::layers::glm5next_kda::tp::KdaTpPlan;
use crate::layers::glm5next_kda::tp_bind::KdaShardedSource;
use crate::layers::glm5next_kda::{Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer};
use crate::layers::glm5next_layer::{Glm5NextLayer, Glm5NextMhc, Glm5NextMixer, Glm5NextMlpSite};
use crate::layers::glm5next_mlp::weights::{Glm5NextExpertWeights, Nvfp4Proj};
use crate::layers::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels, build as mlp_build};
use crate::layers::glm5next_skeleton::{Glm5NextTextSkeleton, Mixer, Mlp};
use crate::layers::ops::{Glm5NextMhcKernels, Glm5NextMhcSiteWeights, mix_hc};
use crate::weight_map::DenseWeight;

pub struct Glm5NextWeightLoader;

/// A `[layer]`-relative tensor name, fully qualified for this checkpoint.
///
/// 🪤 GLM-5.3 nests the text stack under `model.language_model.`, not `model.`. And it does NOT
/// use `mtp.0.*` — the MTP block is `layers.45`.
fn qualify(layer: usize, leaf: &str) -> String {
    format!("model.language_model.layers.{layer}.{leaf}")
}

/// Read a device tensor back as host bytes.
fn host_bytes(gpu: &dyn GpuBackend, t: &WeightTensor) -> Result<Vec<u8>> {
    let mut b = vec![0u8; t.byte_size()];
    gpu.copy_d2h(t.ptr, &mut b)?;
    Ok(b)
}

/// Read a device tensor back as host `f32`, whatever width it is stored at.
///
/// 🪤 The dtype is read off the tensor, never assumed. `hc_*_fn` is BF16 on disk while the kernel
/// wants F32, and `weight_scale_2` is F32 — this is the #341/#347 dtype-mismatch class, and
/// the shapes never say so.
fn host_f32(gpu: &dyn GpuBackend, t: &WeightTensor, what: &str) -> Result<Vec<f32>> {
    let b = host_bytes(gpu, t)?;
    match t.dtype {
        WeightDtype::BF16 => Ok(b
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        WeightDtype::FP32 => Ok(b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        other => bail!(
            "{what}: dtype {other:?} cannot be read as f32 without a conversion this loader refuses to guess"
        ),
    }
}

fn upload_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

/// One layer's slice of the store, presented to the KDA binder as layer-relative names.
struct LayerSource {
    names: Vec<String>,
    tensors: std::collections::BTreeMap<String, (WeightDtype, Vec<usize>, Vec<u8>)>,
}

impl LayerSource {
    fn collect(gpu: &dyn GpuBackend, store: &WeightStore, layer: usize) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.");
        let mut names = Vec::new();
        let mut tensors = std::collections::BTreeMap::new();
        let rels: Vec<String> = store
            .names()
            .filter_map(|n| n.strip_prefix(&prefix).map(|r| r.to_string()))
            .collect();
        for rel in rels {
            let rel = rel.as_str();
            // The routed experts are the bulk of a layer and are bound zero-copy from their
            // device pointers; pulling them to the host here would move gigabytes for nothing.
            if rel.starts_with("mlp.experts.") {
                names.push(rel.to_string());
                continue;
            }
            names.push(rel.to_string());
            let t = store.get(&format!("{prefix}{rel}"))?;
            tensors.insert(
                rel.to_string(),
                (t.dtype, t.shape.clone(), host_bytes(gpu, t)?),
            );
        }
        Ok(Self { names, tensors })
    }

    fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let (dtype, _, bytes) = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        match dtype {
            WeightDtype::BF16 => Ok(bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect()),
            WeightDtype::FP32 => Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            other => bail!("{name}: dtype {other:?} is not a plain float tensor"),
        }
    }
}

impl KdaTensorSource for LayerSource {
    fn get(&self, name: &str) -> Option<RawTensor<'_>> {
        let (dtype, shape, bytes) = self.tensors.get(name)?;
        // 🪤 A KDA block is entirely BF16 except `A_log`/`dt_bias`, which are F32. Anything
        // else here is not a KDA tensor, and the binder must see the absence rather than a
        // coerced dtype — it refuses on a dtype mismatch precisely because a cast would
        // change the numerics.
        let dtype = match dtype {
            WeightDtype::BF16 => KdaDtype::Bf16,
            WeightDtype::FP32 => KdaDtype::F32,
            _ => return None,
        };
        Some(RawTensor {
            dtype,
            shape: shape.clone(),
            bytes,
        })
    }
    fn names(&self) -> Vec<String> {
        self.names.clone()
    }
}

/// Bind the two mHC sites of one layer.
///
/// 🪤 `hc_*_fn` is **BF16 on disk and F32 at the kernel**; `base`/`scale` are already F32. All
/// three are uploaded as F32 here. Passing the on-disk BF16 straight through is exactly the
/// defect class that produced #341 and #347.
fn bind_mhc_site(
    gpu: &dyn GpuBackend,
    src: &LayerSource,
    site: &str,
    hc_mult: usize,
    hidden: usize,
) -> Result<Glm5NextMhcSiteWeights> {
    let f = src.f32(&format!("hc_{site}_fn"))?;
    let want = mix_hc(hc_mult) * hc_mult * hidden;
    if f.len() != want {
        bail!(
            "hc_{site}_fn has {} elements, expected mix_hc({hc_mult}) * {hc_mult} * {hidden} = {want}",
            f.len()
        );
    }
    let base = src.f32(&format!("hc_{site}_base"))?;
    if base.len() != mix_hc(hc_mult) {
        bail!(
            "hc_{site}_base has {} entries, expected mix_hc({hc_mult}) = {}",
            base.len(),
            mix_hc(hc_mult)
        );
    }
    let scale = src.f32(&format!("hc_{site}_scale"))?;
    if scale.len() != 3 {
        bail!(
            "hc_{site}_scale has {} entries, expected 3 (pre, post, comb)",
            scale.len()
        );
    }
    Ok(Glm5NextMhcSiteWeights {
        hc_fn: upload_f32(gpu, &f)?,
        hc_scale: upload_f32(gpu, &scale)?,
        hc_base: upload_f32(gpu, &base)?,
    })
}

/// One routed expert, bound straight off the checkpoint's device pointers.
fn bind_expert(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    id: usize,
) -> Result<Glm5NextExpertWeights> {
    let proj = |p: &str| -> Result<Nvfp4Proj> {
        let base = format!("mlp.experts.{id}.{p}");
        let packed = store.get(&qualify(layer, &format!("{base}.weight")))?;
        let scale = store.get(&qualify(layer, &format!("{base}.weight_scale")))?;
        let s2 = store.get(&qualify(layer, &format!("{base}.weight_scale_2")))?;
        if packed.dtype != WeightDtype::UInt8 {
            bail!(
                "{base}.weight is {:?}, expected packed U8 NVFP4",
                packed.dtype
            );
        }
        let s2 = host_f32(gpu, s2, &format!("{base}.weight_scale_2"))?;
        let [s2] = s2[..] else {
            bail!("{base}.weight_scale_2 is not a scalar");
        };
        Ok(Nvfp4Proj {
            packed: packed.ptr,
            scale: scale.ptr,
            scale_2: s2,
        })
    };
    Ok(Glm5NextExpertWeights {
        gate_proj: proj("gate_proj")?,
        up_proj: proj("up_proj")?,
        down_proj: proj("down_proj")?,
    })
}

fn dense(store: &WeightStore, name: &str) -> Result<DenseWeight> {
    Ok(DenseWeight {
        weight: store.get(name)?.ptr,
    })
}

impl ModelWeightLoader for Glm5NextWeightLoader {
    /// All three halves shard: DSA by head, KDA by head/channel, the MLP by width (TP) and by
    /// expert set (EP).
    fn supports_tp(&self) -> bool {
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let skeleton = Glm5NextTextSkeleton::from_config(config)?;
        // 🪤 `l2_eps` and `chunk` are NOT config keys — `l2_eps` is FLA's `1/sqrt(sum + eps)`
        // convention and `chunk` is a prefill tiling width whose results are identical over
        // 2..32. Everything else comes off the checkpoint, including `gate_lower_bound`, which
        // the parser now refuses to default.
        let kda_cfg = Glm5NextKdaConfig {
            hidden: config.hidden_size,
            heads: config.linear_num_value_heads,
            head_dim: config.linear_value_head_dim,
            conv_kernel: config.linear_conv_kernel_dim,
            gate_lower_bound: config.linear_gate_lower_bound,
            rms_norm_eps: config.rms_norm_eps as f32,
            l2_eps: 1e-6,
            chunk: 32,
        };
        kda_cfg.validate()?;
        // 🪤 `gate_rank` is not a config key — it is `f_a_proj`'s row count, read off the
        // checkpoint (128 on GLM-5.3). Reading it from layer 0 rather than assuming it means a
        // checkpoint revision that changes the gate bottleneck fails loudly at load.
        let gate_rank = {
            let n = qualify(0, "self_attn.f_a_proj.weight");
            let t = store.get(&n).with_context(|| {
                format!("glm5_next: {n} is needed to size the KDA gate bottleneck")
            })?;
            *t.shape.first().context("f_a_proj has no rows")?
        };
        let kda_plan = KdaTpPlan::from_config(config, gate_rank)?;
        let dsa_cfg = Glm5NextDsaConfig::from_config(config)?;
        let mlp_cfg = Glm5NextMlpConfig::from_config(config)?;

        let kda_kernels = Glm5NextKdaKernels::resolve(gpu)?;
        let dsa_kernels = Glm5NextDsaKernels::resolve(gpu)?;
        let dsa_layer_kernels = Glm5NextDsaLayerKernels::resolve(gpu)?;
        let mlp_kernels = Glm5NextMlpKernels::resolve(gpu)?;
        let mhc_kernels_probe = Glm5NextMhcKernels::resolve(gpu)?;
        let rms_norm_k = gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;

        // 🪤 All 34 KDA blocks have identical geometry, so ONE workspace serves them all.
        // Sized for a single token: prefill runs token-by-token through `Glm5NextLayer::prefill`
        // (the mHC highway forces per-token anyway), so the chunked KDA prefill is unused.
        let kda_ws = std::sync::Arc::new(crate::layers::glm5next_kda::Glm5NextKdaWorkspace::new(
            gpu, &kda_cfg, 1,
        )?);

        let dsa_plan = crate::layers::glm5next_dsa::tp::DsaTpPlan::new(
            config.tp_rank,
            config.tp_world_size.max(1),
            &dsa_cfg,
        )?;
        let last = skeleton.layers.len() - 1;
        let mut out: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(skeleton.layers.len());

        for sl in &skeleton.layers {
            let idx = sl.index;
            let src = LayerSource::collect(gpu, store, idx)
                .with_context(|| format!("glm5_next: collecting layer {idx}"))?;

            let mixer = match sl.mixer {
                Mixer::Kda => {
                    // The adapter yields THIS RANK's slice with local shapes; the binder
                    // validates against the (already local) config exactly as at TP=1.
                    let sharded = KdaShardedSource::new(&src, &kda_plan)?;
                    let (w, _report) = bind_kda_weights(gpu, &kda_cfg, idx, &sharded)?;
                    Glm5NextMixer::Kda {
                        layer: Box::new(Glm5NextKdaLayer::new(idx, kda_cfg, w, kda_kernels)?),
                        ws: kda_ws.clone(),
                        cfg: kda_cfg,
                    }
                }
                Mixer::Dsa => {
                    let load = |n: &str| src.f32(n);
                    let w = build_dsa_weights(gpu, &dsa_cfg, &dsa_plan, &load)?;
                    Glm5NextMixer::Dsa(Box::new(Glm5NextDsaLayer {
                        cfg: dsa_cfg,
                        weights: w,
                        kernels: dsa_layer_kernels,
                        select_kernels: dsa_kernels,
                        decode_kernel:
                            crate::layers::glm5next_dsa::attend::Glm5NextDsaDecodeKernel::resolve(
                                gpu,
                            )?,
                        workspace: crate::layers::glm5next_dsa::layer::Glm5NextDsaWorkspace::new(
                            gpu, &dsa_cfg,
                        )?,
                        layer_idx: idx,
                        rms_eps: config.rms_norm_eps as f32,
                        kv_scale: 1.0,
                    }))
                }
            };

            let load = |n: &str| src.f32(n);
            let mlp = match sl.mlp {
                Mlp::Dense => Glm5NextMlpSite::Dense(mlp_build::build_dense_mlp(
                    gpu,
                    &mlp_cfg,
                    config.tp_rank,
                    config.intermediate_size,
                    "mlp",
                    &load,
                )?),
                Mlp::RoutedMoe => {
                    let expert = |id: usize| bind_expert(gpu, store, idx, id);
                    Glm5NextMlpSite::Moe(Box::new(mlp_build::build_moe(
                        gpu,
                        &mlp_cfg,
                        config.tp_rank,
                        config.shared_expert_intermediate_size,
                        &load,
                        &expert,
                    )?))
                }
            };

            let mhc = if sl.hyper_connection {
                Some(Glm5NextMhc {
                    kernels: mhc_kernels_probe,
                    attn: bind_mhc_site(gpu, &src, "attn", config.hc_mult, config.hidden_size)?,
                    ffn: bind_mhc_site(gpu, &src, "ffn", config.hc_mult, config.hidden_size)?,
                    hc_mult: config.hc_mult,
                    sinkhorn_iters: config.hc_sinkhorn_iters,
                    hc_eps: config.hc_eps,
                })
            } else {
                None
            };

            out.push(Box::new(Glm5NextLayer {
                layer_idx: idx,
                mixer,
                mlp,
                mlp_cfg,
                mlp_kernels,
                mlp_ws: crate::layers::glm5next_mlp::forward::Glm5NextMlpWorkspace::new(
                    gpu, &mlp_cfg,
                )?,
                mhc,
                input_norm: upload_f32_as_bf16(gpu, &src.f32("input_layernorm.weight")?)?,
                post_attn_norm: upload_f32_as_bf16(
                    gpu,
                    &src.f32("post_attention_layernorm.weight")?,
                )?,
                rms_norm_k,
                rms_eps: config.rms_norm_eps as f32,
                hidden: config.hidden_size,
                mixer_all_reduce: match sl.mixer {
                    Mixer::Kda => kda_plan.needs_output_all_reduce(),
                    Mixer::Dsa => dsa_plan.needs_output_all_reduce(),
                },
                is_first: idx == 0,
                is_last: idx == last,
            }));
        }
        Ok(out)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.language_model.embed_tokens.weight")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.language_model.norm.weight")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "lm_head.weight")
    }

    /// MTP is deliberately out of scope for this slice. `None` = "no speculative head", which
    /// the scheduler already handles; it is not a silent skip of something wired.
    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::weight_loader::MtpWeights>> {
        Ok(None)
    }
}

fn upload_f32_as_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}
