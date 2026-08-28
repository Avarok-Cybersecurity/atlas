// SPDX-License-Identifier: AGPL-3.0-only

//! Per-layer attach helpers (QSA indexer, PLE) + the final-norm
//! placeholder story, split from `qwen4_exp.rs` for the ≤500 LoC cap.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::{mixer_prefix, ones_norm};
use crate::layer::TransformerLayer;
use crate::weight_map::DenseWeight;

/// QSA indexer attach for a full-attention layer (#753 phase G).
/// `ATLAS_QSA_DISABLE=1` skips (decode then runs DENSE past the budget,
/// which is NOT the reference model); layers without the tensor are
/// silently non-QSA.
pub(super) fn attach_qsa(
    layer: &mut Box<dyn TransformerLayer>,
    i: usize,
    lp: &str,
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    // QSA indexer on the 12 full-attention layers (#753 phase G).
    // Decode-side selection; inert below the budget by arithmetic.
    // ATLAS_QSA_DISABLE=1 skips the attach for A/B — decode then runs
    // DENSE past the budget, which is NOT the reference model.
    if config.index_topk > 0
        && std::env::var("ATLAS_QSA_DISABLE").as_deref() != Ok("1")
        && store.contains(&format!("{lp}.self_attn.indexer.index_qk_proj.weight"))
    {
        // Already device-resident in the store; the indexer holds the
        // pointers (the store keeps the weights alive for the model's
        // lifetime, same as every other layer weight).
        let up = |name: &str| -> Result<spark_runtime::gpu::DevicePtr> {
            Ok(store
                .get(&format!("{lp}.self_attn.indexer.{name}"))
                .with_context(|| format!("qwen4_exp layer {i}: indexer {name}"))?
                .ptr)
        };
        let qsa = crate::layers::qsa::QsaIndexer::new(
            up("index_qk_proj.weight")?,
            up("q_layernorm.weight")?,
            up("k_layernorm.weight")?,
            config.index_n_heads,
            config.index_head_dim,
            config.index_compress_ratio,
            config.index_topk,
            (config.head_dim as f64 * config.partial_rotary_factor) as usize,
            config.rope_theta as f32,
            config.rms_norm_eps as f32,
            config.hidden_size,
            config.num_key_value_heads,
            config.head_dim,
            gpu,
        )
        .with_context(|| format!("qwen4_exp layer {i}: QSA indexer"))?;
        let any = layer
            .as_any_mut()
            .ok_or_else(|| anyhow::anyhow!("qwen4_exp layer {i}: no as_any_mut for QSA"))?;
        any.downcast_mut::<crate::layers::Qwen3AttentionLayer>()
            .ok_or_else(|| {
                anyhow::anyhow!("qwen4_exp layer {i}: indexer on a non-attention layer")
            })?
            .set_qsa(qsa);
    }
    Ok(())
}

/// Hand a loaded PLE layer to its GDN host layer.
pub(super) fn attach_ple(
    layer: &mut Box<dyn TransformerLayer>,
    i: usize,
    p: crate::layers::ple::PleLayer,
) -> Result<()> {
    let any = layer
        .as_any_mut()
        .ok_or_else(|| anyhow::anyhow!("qwen4_exp layer {i}: no as_any_mut for PLE"))?;
    let l = any
        .downcast_mut::<crate::layers::Qwen3SsmLayer>()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "qwen4_exp layer {i} carries PLE but is not a GDN layer; \
                 the injection has nowhere to go"
            )
        })?;
    l.set_ple(p);
    Ok(())
}

pub(super) fn final_norm_placeholder(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let mixer = mixer_prefix(config);
    anyhow::ensure!(
        store.contains(&format!("{mixer}.hc_norm.weight")),
        "qwen4_exp: no `{mixer}.hc_norm.weight` — this architecture is \
         supposed to keep its final normalization in the hyper-connection \
         mixer, and it is not there. Refusing rather than guessing."
    );
    tracing::warn!(
        "qwen4_exp: final norm is a PLACEHOLDER. The real one is \
         `{mixer}.hc_norm` [{}], applied while collapsing the {} residual \
         streams — that is mHC work, not a final-norm substitution.",
        config.hc_mult * config.hidden_size,
        config.hc_mult,
    );
    ones_norm(config.hidden_size, gpu)
}

/// Attach both hyper-connection sites to a freshly built layer.
///
/// `set_hc_weights` lives on `Qwen3AttentionLayer`, but `load_layers` hands
/// back `Box<dyn TransformerLayer>`, so the concrete type has to be recovered.
/// A failure here is a hard error rather than a skip: a layer that silently
/// keeps no mHC weights would run attention on an unmixed stream and produce
/// plausible, wrong activations.
pub(super) fn attach_hc(
    layer: &mut Box<dyn TransformerLayer>,
    idx: usize,
    attn: crate::layers::qwen3_attention::HcSiteWeights,
    ffn: crate::layers::qwen3_attention::HcSiteWeights,
    head: Option<crate::layers::qwen3_attention::HcHeadWeights>,
    config: &ModelConfig,
) -> Result<()> {
    use crate::layers::qwen3_attention::HcWeights;
    // Hard error, never a skip. A layer that quietly kept no mHC weights
    // would run attention on a stream it never mixed and emit plausible,
    // wrong activations — with nothing in the log.
    let any = layer.as_any_mut().ok_or_else(|| {
        anyhow::anyhow!("qwen4_exp layer {idx}: no as_any_mut, cannot attach mHC weights")
    })?;
    // TWO concrete layer types carry mHC here: the 12 full-attention layers
    // are `Qwen3AttentionLayer`, the 36 GDN layers are `Qwen3SsmLayer`.
    // DeepSeek-V4 only ever needed the first, which is why the second had to
    // learn `set_hc_weights`.
    let w = HcWeights {
        attn,
        ffn,
        head,
        hc_mult: config.hc_mult,
        sinkhorn_iters: 0,
        hc_eps: config.rms_norm_eps as f32,
        // MODEL layer indices, not attention-layer ones. With a 3:1
        // GDN:attention interleave, model layer 0 is GDN and the last model
        // layer (47) is attention — so the layer that seeds the highway and
        // the layer that collapses it are DIFFERENT concrete types, and
        // neither is identified by `attn_layer_idx`.
        is_first_model_layer: idx == 0,
        is_last_model_layer: idx + 1 == config.num_hidden_layers,
    };
    if let Some(l) = any.downcast_mut::<crate::layers::Qwen3AttentionLayer>() {
        l.set_hc_weights(w);
        return Ok(());
    }
    if let Some(l) = any.downcast_mut::<crate::layers::Qwen3SsmLayer>() {
        l.set_hc_weights(w);
        return Ok(());
    }
    anyhow::bail!(
        "qwen4_exp layer {idx}: mHC weights have nowhere to go — the layer is \
         neither Qwen3AttentionLayer nor Qwen3SsmLayer"
    )
}
