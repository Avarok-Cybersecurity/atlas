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
use avarok_core::config::ModelConfig;
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
use crate::layers::ops::{Glm5NextMhcKernels, Glm5NextMhcSiteWeights, MHC_MIX_MAX_TOKENS, mix_hc};
use crate::weight_map::DenseWeight;

#[cfg(test)]
mod defer_hook_tests;
#[cfg(test)]
mod export_layout_tests;
mod nvfp4_dequant;
mod nvfp4_quant;
#[cfg(test)]
mod plan_cast_tests;
mod plan_dtype;

pub struct Glm5NextWeightLoader;

/// A `[layer]`-relative tensor name, fully qualified for this checkpoint.
///
/// 🪤 GLM-5.3 nests the text stack under `model.language_model.`, not `model.`. And it does NOT
/// use `mtp.0.*` — the MTP block is `layers.45`.
fn qualify(layer: usize, leaf: &str) -> String {
    format!("model.language_model.layers.{layer}.{leaf}")
}

/// Is this store tensor one the binders have already re-uploaded a copy of?
///
/// `load_layers` pulls every non-expert layer tensor to the host
/// (`LayerSource::collect`) and the binders upload fresh device buffers — a TP
/// shard for KDA/DSA, a dtype-converted copy for mHC. The store's originals are
/// dead from that moment, and on GB10 they are 15.7 GB of unified memory the KV
/// cache never gets. Measured 2026-08-28: the first 2-node bring-up died with
/// "No memory left for KV cache" at 112.8 GB resident against a 99.64 GB load.
///
/// 🪤 Two things must NOT match:
/// * `mlp.experts.*` — bound **zero-copy** from these very pointers
///   (`bind_expert`). Freeing them is a use-after-free with no diagnostic.
/// * `layers.{num_layers}` — GLM's MTP/draft block sits one past the skeleton
///   (`layers.45` at `num_hidden_layers = 45`) and is read by
///   `load_mtp_weights_multi`, not by `load_layers`.
fn is_reuploaded(name: &str, num_layers: usize) -> bool {
    let Some(rest) = name.strip_prefix("model.language_model.layers.") else {
        return false;
    };
    let Some((idx, rel)) = rest.split_once('.') else {
        return false;
    };
    let Ok(idx) = idx.parse::<usize>() else {
        return false;
    };
    idx < num_layers && !rel.starts_with("mlp.experts.")
}

/// Is this store tensor a routed-expert projection [`bind_expert`] QUANTISED
/// instead of binding zero-copy?
///
/// The discriminator is the on-disk dtype and nothing else. A packed U8 expert
/// IS the kernel's operand and must never be freed — that is the
/// use-after-free [`is_reuploaded`] exists to avoid. A BF16 one cannot be: no
/// GLM routed-expert kernel reads BF16, so the only thing that ever touched it
/// was the quantiser, and the NVFP4 result it produced lives in the layer.
///
/// 🪤 This can only be acted on AFTER `load_glm5next_mtp_module` has bound
/// `layers.{num_hidden_layers}` — which `factory::build` guarantees by calling
/// `prune_after_load` last, and which is the same ordering `is_reuploaded`'s
/// "the MTP block is kept" rule already depends on.
///
/// On both disk loaders this now matches NOTHING, because
/// [`Glm5NextWeightLoader::defer_predicate`] keeps those tensors off the device
/// in the first place. It stays for the paths that have no defer hook — the
/// RDMA weight peer, and any future loader that fills a store directly — where
/// a BF16 expert still arrives resident and still has to be freed.
fn is_quantized_expert_weight(name: &str, dtype: WeightDtype) -> bool {
    dtype == WeightDtype::BF16
        && name.starts_with("model.language_model.layers.")
        && name.contains(".mlp.experts.")
        && name.ends_with("_proj.weight")
}

/// Is this a routed-expert projection of the MTP block that the export left at
/// full width — the family [`Glm5NextWeightLoader::defer_predicate`] keeps off
/// the device entirely?
///
/// Both halves of the key matter:
///
/// * **the LAYER** is `config.num_hidden_layers`, never a literal. GLM puts the
///   MTP block one past the text stack, and the text stack's own experts are
///   packed U8 in both exports — deferring one of those would withhold the
///   kernel's actual operand.
/// * **the DTYPE** is BF16. `LibertAIDAI/GLM-5.3-Flash-NVFP4` quantises this
///   block like every other, so on that checkpoint this is false for every
///   tensor and the loader defers nothing at all.
///
/// 🪤 `mlp.experts.` and not `experts.`: the SHARED expert
/// (`mlp.shared_experts.{gate,up,down}_proj.weight`) is a plain float tensor in
/// both exports, is read through [`LayerSource`], and must stay resident.
fn is_full_width_mtp_expert(name: &str, dtype: WeightDtype, num_layers: usize) -> bool {
    if dtype != WeightDtype::BF16 || !name.ends_with("_proj.weight") {
        return false;
    }
    let Some(rest) = name.strip_prefix("model.language_model.layers.") else {
        return false;
    };
    let Some((idx, rel)) = rest.split_once('.') else {
        return false;
    };
    idx.parse::<usize>().ok() == Some(num_layers) && rel.starts_with("mlp.experts.")
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
pub(super) struct LayerSource {
    names: Vec<String>,
    tensors: std::collections::BTreeMap<String, (WeightDtype, Vec<usize>, Vec<u8>)>,
    /// Plan-dtype copies of the tensors whose on-disk width is not the width the
    /// attention binders bind at — see [`plan_dtype`]. Consulted by the RAW path
    /// ([`KdaTensorSource::get`]) and by nothing else: `f32` keeps reading the
    /// checkpoint's own bytes, so the DSA absorb math and the F32 `ape` upload
    /// still see every bit the export carries.
    ///
    /// **Empty for `LibertAIDAI/GLM-5.3-Flash-NVFP4`**, whose attention stack is
    /// BF16 throughout — that checkpoint allocates nothing here and takes the
    /// same code path it always did.
    plan_cast: std::collections::BTreeMap<String, (WeightDtype, Vec<u8>)>,
}

impl LayerSource {
    pub(super) fn collect(gpu: &dyn GpuBackend, store: &WeightStore, layer: usize) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.");
        let mut names = Vec::new();
        let mut tensors = std::collections::BTreeMap::new();
        let mut plan_cast = std::collections::BTreeMap::new();
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
            let bytes = host_bytes(gpu, t)?;
            // 🪤 Storage width is an EXPORT choice, not a model change. NVIDIA's
            // `nvidia/GLM-5.3-Flash-NVFP4` writes the non-quantised attention
            // tensors at F32 where `LibertAIDAI/GLM-5.3-Flash-NVFP4` writes
            // BF16 — same numbers, twice the bytes. Materialise the plan-dtype
            // copy the RAW path needs, and only for the tensors that need it.
            if let Some(cast) = plan_dtype::cast_to_plan_dtype(rel, t.dtype, &bytes)? {
                plan_cast.insert(rel.to_string(), cast);
            }
            tensors.insert(rel.to_string(), (t.dtype, t.shape.clone(), bytes));
        }
        Ok(Self {
            names,
            tensors,
            plan_cast,
        })
    }

    pub(super) fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let (dtype, shape, bytes) = self
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
            // 🪤 QUANTISATION is an EXPORT choice too, not a model change.
            // NVIDIA's `nvidia/GLM-5.3-Flash-NVFP4` quantises the three dense
            // MLP layers `LibertAIDAI/GLM-5.3-Flash-NVFP4` leaves BF16. The
            // dense MLP builder and its kernels are BF16-only by contract, so
            // the packed codes are unpacked HERE and nothing downstream moves.
            WeightDtype::UInt8 => self.dequant_packed_nvfp4(name, shape, bytes),
            other => bail!("{name}: dtype {other:?} is not a plain float tensor"),
        }
    }

    /// One packed-NVFP4 tensor of this layer as `f32`, read with its own
    /// `.weight_scale` / `.weight_scale_2` siblings.
    ///
    /// Reached only from the `UInt8` arm of [`Self::f32`], so a checkpoint
    /// whose dense MLP is BF16 never enters it.
    ///
    /// 🪤 The siblings are looked up by NAME, not assumed: a `.weight` without
    /// them is a format this loader has not been taught (compressed-tensors
    /// spells them `weight_packed` / `weight_global_scale` and stores the
    /// RECIPROCAL global scale), and guessing would apply the wrong
    /// convention with no error.
    fn dequant_packed_nvfp4(&self, name: &str, shape: &[usize], bytes: &[u8]) -> Result<Vec<f32>> {
        let base = name.strip_suffix(".weight").with_context(|| {
            format!("{name}: packed NVFP4 must be a `.weight`, with scale siblings beside it")
        })?;
        let (scale_dtype, _, scale_bytes) = self
            .tensors
            .get(&format!("{base}.weight_scale"))
            .with_context(|| format!("{name} is packed NVFP4 but {base}.weight_scale is absent"))?;
        if *scale_dtype != WeightDtype::FP8E4M3 {
            bail!("{base}.weight_scale is {scale_dtype:?}, expected F8_E4M3 block scales");
        }
        let s2 = self.f32(&format!("{base}.weight_scale_2"))?;
        let [scale_2] = s2[..] else {
            bail!("{base}.weight_scale_2 is not a scalar");
        };
        nvfp4_dequant::dequant_nvfp4_to_f32(name, bytes, shape, scale_bytes, scale_2)
    }
}

impl KdaTensorSource for LayerSource {
    fn get(&self, name: &str) -> Option<RawTensor<'_>> {
        let (dtype, shape, bytes) = match self.plan_cast.get(name) {
            // The export stored this one at a width the plan does not bind at;
            // hand over the plan-dtype copy. Absent for every checkpoint whose
            // attention stack is already at the plan's widths.
            Some((dtype, bytes)) => (dtype, &self.tensors.get(name)?.1, bytes),
            None => {
                let (dtype, shape, bytes) = self.tensors.get(name)?;
                (dtype, shape, bytes)
            }
        };
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
        // 🔴 BF16, because that is what the checkpoint stores (`[24, 16384]`, BF16 in the
        // safetensors header). Uploading it as F32 doubled `hc_mix`'s traffic — 1.57 MB
        // instead of 0.79 MB per site, 90 sites per token — for values that were already
        // exactly BF16. `glm5next_hc_mix_bf16` reads it at that width and is bit-identical.
        hc_fn: upload_f32_as_bf16(gpu, &f)?,
        hc_fn_bf16: true,
        hc_scale: upload_f32(gpu, &scale)?,
        hc_base: upload_f32(gpu, &base)?,
        // `hc_mix` -> `hc_finish` handoff. Per site so the layer's two sites cannot alias.
        mix: gpu.alloc(MHC_MIX_MAX_TOKENS * mix_hc(hc_mult) * 4)?,
    })
}

/// One routed expert, bound straight off the checkpoint's device pointers —
/// or quantised to the same operand triple when the export left it full width.
///
/// 🪤 The arm is chosen by the `.weight` tensor's ON-DISK dtype, never by a
/// flag. `LibertAIDAI/GLM-5.3-Flash-NVFP4`'s experts are U8 everywhere, so that
/// checkpoint takes the zero-copy arm for every expert of every layer exactly
/// as it did before the BF16 arm existed.
///
/// Three arms, in the order they are tried:
///
/// 1. **deferred** — the weight loader honoured
///    [`Glm5NextWeightLoader::defer_predicate`] and left it on disk. Read the
///    host bytes, quantise, upload the NVFP4. Nothing full-width ever touches
///    the device. This is the arm the official export takes.
/// 2. **U8 resident** — the checkpoint already carries the triple; bind it
///    where it lies, zero copy. The community export takes this arm.
/// 3. **BF16 resident** — a store filled by something with no defer hook (the
///    RDMA weight peer). Read it back off the device and quantise;
///    `prune_after_load` frees the source.
pub(super) fn bind_expert(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    id: usize,
) -> Result<Glm5NextExpertWeights> {
    let proj = |p: &str| -> Result<Nvfp4Proj> {
        let base = format!("mlp.experts.{id}.{p}");
        let wname = qualify(layer, &format!("{base}.weight"));
        // Deferred: never uploaded, at this loader's own request. Checked
        // FIRST, because the store has no tensor under this name at all.
        if let Some(d) = store.deferred(&wname) {
            return quantize_deferred_expert_proj(gpu, store, d, &base);
        }
        let w = store.get(&wname)?;
        match w.dtype {
            // The checkpoint already carries the triple: bind it where it lies.
            WeightDtype::UInt8 => {
                let scale = store.get(&qualify(layer, &format!("{base}.weight_scale")))?;
                let s2 = store.get(&qualify(layer, &format!("{base}.weight_scale_2")))?;
                let s2 = host_f32(gpu, s2, &format!("{base}.weight_scale_2"))?;
                let [s2] = s2[..] else {
                    bail!("{base}.weight_scale_2 is not a scalar");
                };
                Ok(Nvfp4Proj {
                    packed: w.ptr,
                    scale: scale.ptr,
                    scale_2: s2,
                })
            }
            // 🪤 NVIDIA's official export leaves the MTP block's experts BF16
            // with no scales at all. There is no BF16 routed-expert forward to
            // fall back to, so the triple is MADE here — see [`nvfp4_quant`].
            WeightDtype::BF16 => quantize_expert_proj(gpu, store, w, &base),
            other => bail!("{base}.weight is {other:?}, expected packed U8 NVFP4 or BF16"),
        }
    };
    Ok(Glm5NextExpertWeights {
        gate_proj: proj("gate_proj")?,
        up_proj: proj("up_proj")?,
        down_proj: proj("down_proj")?,
    })
}

/// Quantise one full-width expert projection the loader left ON DISK.
///
/// This is the arm that makes the official export fit. The BF16 bytes are read
/// from the shard into host memory, quantised, and only the ~3.6x smaller
/// NVFP4 triple is uploaded — so at no point in the 45-layer build is a
/// full-width expert resident. Measured 2026-09-21, before this arm existed:
/// the fast loader swept the 432 BF16 tensors of one EP=2 rank (~7.25 GB) to
/// the device and they stayed there until `prune_after_load`, which killed
/// rank 0 at layer 38 on a 32K boot. See `spark_runtime::weights::deferred`.
///
/// Only the experts this rank owns are read: `build_moe` calls
/// [`bind_expert`] over `local_expert_range()` alone, and the loader's own EP
/// rule never deferred a remote expert to begin with.
///
/// 🪤 One host buffer at a time. `read_host_bytes` + the `f32` expansion are
/// ~40 MB together for a `[2048, 4096]` BF16 projection, and both are dropped
/// before the next projection is read — on a unified-memory box the host
/// working set IS the device working set.
fn quantize_deferred_expert_proj(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    d: &spark_runtime::weights::DeferredTensor,
    base: &str,
) -> Result<Nvfp4Proj> {
    let [rows, cols] = d.shape[..] else {
        bail!(
            "{base}.weight must be 2-D to quantise, got shape {:?}",
            d.shape
        );
    };
    if d.dtype != WeightDtype::BF16 {
        bail!(
            "{base}.weight was deferred as {:?}; only a full-width BF16 expert is \
             quantised at load",
            d.dtype
        );
    }
    let bytes = d
        .read_host_bytes()
        .with_context(|| format!("{base}.weight: reading the deferred expert from its shard"))?;
    let values: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    drop(bytes);
    quantize_and_upload(gpu, store, base, &values, rows, cols)
}

/// Quantise one full-width BF16 expert projection that IS resident.
///
/// Reached only by a store no defer hook filled (the RDMA weight peer). The
/// BF16 source is dead afterwards and is released by
/// [`Glm5NextWeightLoader::prune_after_load`], which is the only place that
/// can, because every binder takes `&WeightStore`.
fn quantize_expert_proj(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    w: &WeightTensor,
    base: &str,
) -> Result<Nvfp4Proj> {
    let [rows, cols] = w.shape[..] else {
        bail!(
            "{base}.weight must be 2-D to quantise, got shape {:?}",
            w.shape
        );
    };
    let values = host_f32(gpu, w, &format!("{base}.weight"))?;
    quantize_and_upload(gpu, store, base, &values, rows, cols)
}

/// The tail both quantise arms share: encode, upload, adopt.
///
/// 🪤 Both buffers are ADOPTED by the store's derived-weight ledger. A
/// `gpu.alloc` that lives in a layer struct has no owner otherwise: teardown's
/// `sweep_unreleased` reclaims it, but only after the backend is gone, so the
/// bytes are unreclaimable for the life of the model (see
/// `spark_runtime::weights::derived`).
fn quantize_and_upload(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    base: &str,
    values: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Nvfp4Proj> {
    let blob = nvfp4_quant::quantize_to_nvfp4(base, values, rows, cols)?;
    let packed = upload_bytes(gpu, store, &blob.packed)?;
    let scale = upload_bytes(gpu, store, &blob.scales)?;
    Ok(Nvfp4Proj {
        packed,
        scale,
        scale_2: blob.scale_2,
    })
}

/// Label for every buffer [`quantize_expert_proj`] adopts, so the residency
/// report can attribute them as one line rather than per expert.
const QUANTIZED_EXPERT_LABEL: &str = "glm5_next routed expert, NVFP4 at load";

fn upload_bytes(gpu: &dyn GpuBackend, store: &WeightStore, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(b, p)?;
    store.derived().adopt(QUANTIZED_EXPERT_LABEL, p, b.len());
    Ok(p)
}

fn dense(store: &WeightStore, name: &str) -> Result<DenseWeight> {
    Ok(DenseWeight {
        weight: store.get(name)?.ptr,
    })
}

impl ModelWeightLoader for Glm5NextWeightLoader {
    /// Text-only port. `weight_loader/glm5_next.rs` classifies `model.visual.*`
    /// as `TensorRole::Vision` and excludes it from `is_required()`; nothing in
    /// this loader binds it. Saying so here keeps the tower off the GPU in the
    /// first place — on the LibertAIDAI NVFP4 checkpoint that is 1.05 GiB per
    /// rank, sitting between `--speculative --num-drafts 2` and a serve that
    /// fits (measured 2026-08-29: K=3 at 32 K needs 13.58 GiB against 12.07 free).
    fn binds_vision_encoder(&self) -> bool {
        false
    }

    /// Keep the MTP block's full-width routed experts off the device.
    ///
    /// `nvidia/GLM-5.3-Flash-NVFP4` ships `layers.{num_hidden_layers}.mlp.
    /// experts.E.{gate,up,down}_proj.weight` as BF16 `[2048, 4096]` against a
    /// w4a16-only forward. `bind_expert` reads each one ONCE and replaces it
    /// with an NVFP4 triple ~3.6x smaller, so uploading the originals is pure
    /// transient — and on GB10's unified memory the transient is the problem:
    /// measured 2026-09-21 at EP=2, ~7.25 GB per rank resident from the fast
    /// loader's sweep until `prune_after_load`, i.e. across the whole 45-layer
    /// build. That survived a 2048-token boot (host floor 1.74 GB) and killed
    /// rank 0 at layer 38 on the 32K qualification shape (935 MB against a
    /// 1200 MB floor), BEFORE the MTP bind could free anything.
    ///
    /// Deferring them makes the transient structurally impossible: the BF16
    /// stays in the page cache and `bind_expert` reads it from there.
    ///
    /// 🪤 `None` would be wrong ONLY as a silent default — the predicate is
    /// dtype-keyed, so returning it on `LibertAIDAI/GLM-5.3-Flash-NVFP4`
    /// (packed U8 experts everywhere) defers nothing and that checkpoint loads
    /// byte-identically. The layer index comes from the config, never a
    /// literal: `layers.45` is `num_hidden_layers`, not a property of GLM.
    fn defer_predicate(&self, config: &ModelConfig) -> Option<spark_runtime::weights::DeferHook> {
        let num_layers = config.num_hidden_layers;
        Some(std::sync::Arc::new(
            move |name: &str, dtype: WeightDtype| is_full_width_mtp_expert(name, dtype, num_layers),
        ))
    }

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
        // Only the MTP layer's plain residual path uses this; a text layer's residual lives in
        // the mHC highway. `try_kernel` so a target without it still serves the text stack.
        let add_k = crate::layers::try_kernel(gpu, "bf16_add", "bf16_add_inplace");

        // 🪤 All 34 KDA blocks have identical geometry, so ONE workspace serves them all.
        //
        // Sized for the widest speculative verify rather than one token. Prefill still runs
        // token-by-token through `Glm5NextLayer::prefill` (the mHC highway forces per-token
        // anyway), but `Glm5NextLayer::forward_k` sweeps the KDA weights ONCE for all K rows of
        // a verify and needs `[K, ...]` scratch to do it. The cap is the batched GEMV kernel's
        // own `MAX_M`: past it `ops::dense_mm_bf16` falls back to the tile GEMM, which is not
        // bit-identical to the serial decode a verify must reproduce.
        //
        // Cost is a few MB for the whole model: the FP32 buffers are already sized to
        // `t_pad = ceil(t / chunk) * chunk = 32` at t = 1, so only the BF16 `[t, *]` scratch
        // grows.
        // 🔴 The workspaces must also hold a batched PREFILL sub-chunk, which is wider than any
        // verify (ANOMALIES A65 — `Glm5NextLayer::prefill` hands `forward_k` `PREFILL_ROWS`
        // rows). Sizing to the verify width alone made `forward_k` bail the moment prefill
        // used it. Cost is per-layer scratch that scales with rows, not with context.
        // 🪤 `prefill_rows()` too, not just the constant: `AVAROK_GLM_PREFILL_ROWS` can widen the
        // sub-chunk at launch, and a workspace built for the default would make `forward_k` bail
        // the first time the A/B lever was actually used.
        let verify_k = (crate::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize)
            .max(crate::layers::glm5next_layer::PREFILL_ROWS)
            .max(crate::layers::glm5next_layer::prefill_rows());
        let kda_ws = std::sync::Arc::new(crate::layers::glm5next_kda::Glm5NextKdaWorkspace::new(
            gpu, &kda_cfg, verify_k,
        )?);

        // 🔴 ONE MLP scratch for the whole stack, allocated HERE — at load, before the layer
        // loop and long before the KV pool is sized (A59: a pool allocated after KV sizing is a
        // pool the KV sizing did not know about). The KDA workspace above has been shared since
        // it was written; the MLP one was private per layer, and at a wide prefill sub-chunk
        // that is what dominated the LOAD-time host-memory dip in ANOMALIES A124/A127 —
        // `mlp_ws_per_layer_bytes * 45`, ≈2.1 GB at 256 rows and ≈9.5 GB at 1024.
        // Correctness: the scratch never carries state between calls (see the field doc on
        // `Glm5NextLayer::mlp_ws`), and the whole stack runs on one stream.
        let mlp_ws_bytes =
            crate::layers::glm5next_mlp::forward::mlp_ws_total_bytes(&mlp_cfg, verify_k);
        let shared_mlp_ws = if crate::layers::glm5next_mlp::forward::mlp_ws_shared() {
            tracing::info!(
                "GLM MLP workspace: SHARED, 1 x {:.1} MB for {} layers at {verify_k} rows \
                 (per-layer would be {:.1} MB)",
                mlp_ws_bytes as f64 / 1e6,
                skeleton.layers.len(),
                (mlp_ws_bytes * skeleton.layers.len()) as f64 / 1e6,
            );
            Some(std::sync::Arc::new(
                crate::layers::glm5next_mlp::forward::Glm5NextMlpWorkspace::new(
                    gpu, &mlp_cfg, verify_k,
                )?,
            ))
        } else {
            tracing::warn!(
                "GLM MLP workspace: PER-LAYER, {} x {:.1} MB at {verify_k} rows",
                skeleton.layers.len(),
                mlp_ws_bytes as f64 / 1e6,
            );
            None
        };

        let dsa_plan = crate::layers::glm5next_dsa::tp::DsaTpPlan::new(
            config.tp_rank,
            config.tp_world_size.max(1),
            &dsa_cfg,
        )?;
        let last = skeleton.layers.len() - 1;
        let mut out: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(skeleton.layers.len());

        // 🪤 The KV pool is sized to `num_attention_layers()` (11 on GLM-5.3 — the
        // sparse blocks), so a DSA layer must address it by its ordinal among
        // KV-consuming layers, NOT by its index in the 45-layer model stack. The
        // 34 KDA blocks carry recurrent state and take no pool slot.
        let mut attn_layer_idx = 0usize;

        for sl in &skeleton.layers {
            let idx = sl.index;
            let t_layer = std::time::Instant::now();
            let src = LayerSource::collect(gpu, store, idx)
                .with_context(|| format!("glm5_next: collecting layer {idx}"))?;
            let t_collect = t_layer.elapsed();

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
                        // ON by default since A55 was closed (the `weights_proj` overrun
                        // fix). Kill switch `AVAROK_GLM_DSA_ALLOC_PER_STEP=1` restores the
                        // per-step `gpu.alloc` + `gpu.free`.
                        persist_bt: std::env::var("AVAROK_GLM_DSA_ALLOC_PER_STEP").as_deref()
                            != Ok("1"),
                        cfg: dsa_cfg,
                        weights: w,
                        kernels: dsa_layer_kernels,
                        select_kernels: dsa_kernels,
                        decode_kernel:
                            crate::layers::glm5next_dsa::attend::Glm5NextDsaDecodeKernel::resolve(
                                gpu,
                            )?,
                        workspace: crate::layers::glm5next_dsa::layer::Glm5NextDsaWorkspace::new(
                            gpu, &dsa_cfg, verify_k,
                        )?,
                        layer_idx: idx,
                        attn_layer_idx: {
                            let a = attn_layer_idx;
                            attn_layer_idx += 1;
                            a
                        },
                        rms_eps: config.rms_norm_eps as f32,
                        kv_scale: 1.0,
                    }))
                }
            };

            let t_mixer = t_layer.elapsed();

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

            let t_mlp = t_layer.elapsed();
            // Splits the unattributed post-load window by ARM. A dense layer has
            // neither a DSA mixer nor a MoE arm; a KDA+MoE layer has one; a
            // DSA+MoE layer has both — three shapes, so the byte-proportional
            // host round trip and the non-proportional per-arm work (absorb_q,
            // the expert sync storm) stop being collinear and can be separated.
            tracing::info!(
                "glm5_next layer {idx} built: collect {:.2}s mixer {:.2}s mlp {:.2}s \
                 (mixer={:?} mlp={:?})",
                t_collect.as_secs_f64(),
                (t_mixer - t_collect).as_secs_f64(),
                (t_mlp - t_mixer).as_secs_f64(),
                sl.mixer,
                sl.mlp,
            );

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
                mlp_ws: match &shared_mlp_ws {
                    Some(ws) => ws.clone(),
                    None => std::sync::Arc::new(
                        crate::layers::glm5next_mlp::forward::Glm5NextMlpWorkspace::new(
                            gpu, &mlp_cfg, verify_k,
                        )?,
                    ),
                },
                mhc,
                input_norm: upload_f32_as_bf16(gpu, &src.f32("input_layernorm.weight")?)?,
                post_attn_norm: upload_f32_as_bf16(
                    gpu,
                    &src.f32("post_attention_layernorm.weight")?,
                )?,
                rms_norm_k,
                add_k,
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

    /// Drop the store's copy of everything `load_layers` re-uploaded.
    fn prune_after_load(
        &self,
        store: &mut WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let n = config.num_hidden_layers;
        let (count, bytes) = store.free_matching(gpu, |name| is_reuploaded(name, n))?;
        tracing::info!(
            "glm5_next: released {count} store tensors ({:.2} GB) already re-uploaded by the \
             binders; routed experts and the MTP block kept",
            bytes as f64 / 1e9,
        );
        // Full-width BF16 routed experts (NVIDIA's official export leaves the
        // MTP block's that way) were re-encoded by `bind_expert`, so they are
        // dead the same way every other re-uploaded tensor is.
        //
        // Empty on both disk loaders: `defer_predicate` keeps that family off
        // the device entirely, which is the whole point — freeing 7 GB here is
        // 45 layers too late. This sweep remains for a store filled by
        // something with no defer hook (the RDMA weight peer).
        let quantized: std::collections::BTreeSet<String> = store
            .names()
            .filter(|n| {
                store
                    .get(n)
                    .is_ok_and(|t| is_quantized_expert_weight(n, t.dtype))
            })
            .map(str::to_string)
            .collect();
        if !quantized.is_empty() {
            let (qcount, qbytes) = store.free_matching(gpu, |name| quantized.contains(name))?;
            tracing::info!(
                "glm5_next: released {qcount} full-width BF16 routed-expert tensors \
                 ({:.2} GB) quantised to NVFP4 at bind time",
                qbytes as f64 / 1e9,
            );
        }
        Ok(())
    }
}

pub(super) fn upload_f32_as_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

#[cfg(test)]
mod prune_tests {
    use super::is_reuploaded;

    #[test]
    fn prunes_only_the_reuploaded_layer_tensors() {
        let n = 45;
        // Re-uploaded by the binders -> free.
        assert!(is_reuploaded(
            "model.language_model.layers.0.self_attn.q_proj.weight",
            n
        ));
        assert!(is_reuploaded(
            "model.language_model.layers.44.mlp.gate.weight",
            n
        ));
        assert!(is_reuploaded(
            "model.language_model.layers.3.hc_attn_fn.weight",
            n
        ));
        // Bound zero-copy from the store -> use-after-free if freed.
        assert!(!is_reuploaded(
            "model.language_model.layers.7.mlp.experts.12.down_proj.weight",
            n
        ));
        // MTP block, one past the skeleton -> read by load_mtp_weights_multi.
        assert!(!is_reuploaded(
            "model.language_model.layers.45.self_attn.q_proj.weight",
            n
        ));
        // Not a layer tensor at all.
        assert!(!is_reuploaded(
            "model.language_model.embed_tokens.weight",
            n
        ));
        assert!(!is_reuploaded("lm_head.weight", n));
        // Malformed / non-numeric index is never a match.
        assert!(!is_reuploaded("model.language_model.layers.x.foo", n));
    }
}

/// [`LayerSource::collect`], named for the MTP loader's call site.
pub(super) fn layer_source(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
) -> Result<LayerSource> {
    LayerSource::collect(gpu, store, layer)
}

/// [`bind_expert`], named for the MTP loader's call site.
pub(super) fn bind_expert_at(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    id: usize,
) -> Result<Glm5NextExpertWeights> {
    bind_expert(gpu, store, layer, id)
}

/// [`upload_f32_as_bf16`], named for the MTP loader's call site.
pub(super) fn upload_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    upload_f32_as_bf16(gpu, v)
}

#[cfg(test)]
mod vision_capability_tests {
    use super::Glm5NextWeightLoader;
    use crate::weight_loader::ModelWeightLoader;

    #[test]
    fn glm5_next_declares_itself_text_only() {
        // Mutation gate: flipping this to `true` re-loads 1.05 GiB/rank of
        // vision tower that nothing binds, and K=3 stops fitting at 32 K.
        assert!(
            !Glm5NextWeightLoader.binds_vision_encoder(),
            "GLM-5.3's port binds no vision encoder; saying otherwise makes the \
             weight loader read the tower into unified memory for nothing"
        );
    }

    #[test]
    fn a_multimodal_loader_still_declares_true_by_default() {
        // The trait default must stay "load everything" — a loader that never
        // overrides this must never lose weights.
        assert!(crate::weight_loader::qwen35::Qwen35WeightLoader.binds_vision_encoder());
    }
}
