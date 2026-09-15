// SPDX-License-Identifier: AGPL-3.0-only

//! The multi-hyperconnection weights: two sites per layer plus the
//! model-level mixer.
//!
//! Shapes, from the checkpoint (`hc_count = 4`, `hidden = 2560`, so
//! `hc_hidden = 10240`; `hc_lowrank = 320`):
//!
//! ```text
//! {lp}.attn_hyper_connection.hc_norm.weight                [10240]
//! {lp}.attn_hyper_connection.input_mix_weight_down.weight  [320, 10240]
//! {lp}.attn_hyper_connection.input_mix_weight_up.weight    [10240, 320]
//! {lp}.attn_hyper_connection.block_inject_weight.weight    [4, 10240]
//! {lp}.mlp_hyper_connection.*                              …same four
//! model.language_model.hyper_connection_mixer.*            …first THREE only
//! ```
//!
//! The model-level mixer has **no `block_inject_weight`** — it is built
//! `use_combine=False` in the reference and only collapses. It is also the
//! model's FINAL NORMALIZATION, which is why this checkpoint ships no
//! `model.norm.weight`.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::WeightStore;

use crate::layers::qwen3_attention::{HcHeadWeights, HcLowRank, HcSiteWeights};
use crate::weight_map::dense;

/// One hyper-connection site. `with_inject` is false only for the
/// model-level mixer.
fn load_site(
    store: &WeightStore,
    prefix: &str,
    rank: usize,
    with_inject: bool,
) -> Result<HcLowRank> {
    let g = |name: &str| -> Result<DevicePtr> {
        dense(store, &format!("{prefix}.{name}.weight"))
            .map(|d| d.weight)
            .with_context(|| format!("qwen4_exp mHC: {prefix}.{name}.weight"))
    };
    Ok(HcLowRank {
        norm_w: g("hc_norm")?,
        down_w: g("input_mix_weight_down")?,
        up_w: g("input_mix_weight_up")?,
        inject_w: if with_inject {
            g("block_inject_weight")?
        } else {
            DevicePtr::NULL
        },
        rank,
    })
}

/// The Sinkhorn fields are NULL on this path: `lowrank.is_some()` is what
/// selects the kernel, and leaving these dangling would be a live footgun if
/// a future dispatch site forgot to branch.
fn site(lowrank: HcLowRank) -> HcSiteWeights {
    HcSiteWeights {
        hc_fn: DevicePtr::NULL,
        hc_base: DevicePtr::NULL,
        hc_scale: DevicePtr::NULL,
        lowrank: Some(lowrank),
    }
}

/// Both per-layer sites for one decoder layer.
pub(super) fn load_layer_sites(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
) -> Result<(HcSiteWeights, HcSiteWeights)> {
    let rank = config.hc_lowrank;
    let attn = load_site(store, &format!("{lp}.attn_hyper_connection"), rank, true)?;
    let ffn = load_site(store, &format!("{lp}.mlp_hyper_connection"), rank, true)?;
    Ok((site(attn), site(ffn)))
}

/// A collapse-only mixer at an ARBITRARY prefix.
///
/// The main model's lives at `{weight_prefix}.hyper_connection_mixer`; the MTP
/// block carries its own at `mtp.hyper_connection_mixer`, same three tensors,
/// same shapes, still no `block_inject_weight`. `load_site` stays private —
/// this is the only new door, so nothing outside this module can construct a
/// site with the wrong `with_inject`.
pub(super) fn load_head_at(
    store: &WeightStore,
    prefix: &str,
    rank: usize,
) -> Result<HcHeadWeights> {
    let lowrank = load_site(store, prefix, rank, false)?;
    Ok(HcHeadWeights {
        hc_fn: DevicePtr::NULL,
        hc_base: DevicePtr::NULL,
        hc_scale: DevicePtr::NULL,
        lowrank: Some(lowrank),
    })
}

/// The model-level mixer, replicated onto every layer but consumed only by
/// the last one.
pub(super) fn load_head(store: &WeightStore, config: &ModelConfig) -> Result<HcHeadWeights> {
    load_head_at(
        store,
        &format!("{}.hyper_connection_mixer", super::embed_prefix(config)),
        config.hc_lowrank,
    )
}
