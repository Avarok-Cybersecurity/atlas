// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint-level detection of MTP / next-token-prediction weights.
//!
//! Atlas binds MTP through three unrelated loader paths — the Qwen-shaped
//! `MtpWeights` vec, DeepSeek-V4's `mtp.0.*` module, and GLM-5.3's
//! `layers.{num_hidden_layers}` block — but the "did the user get what they
//! asked for" check read only the first of them. On GLM-5.3 that produced
//!
//! ```text
//! GLM-5.3 MTP draft module loaded (layers.45)
//! `--speculative` was requested but no MTP weights were loaded for this model
//! ```
//!
//! in the same startup log: the module HAD loaded, from
//! `model.language_model.layers.45.*`. This module is the layout-aware
//! predicate the check should have been asking, kept separate from any one
//! loader so a new architecture only has to be named here once.

use atlas_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;

/// The MTP weight layout a checkpoint ships, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpLayout {
    /// `mtp.*` — Qwen3.5 (`mtp.fc.weight`, `mtp.layers.0.*`) and the
    /// DeepSeek `mtp.0.*` multi-module spelling.
    MtpPrefix,
    /// Transformer layer(s) one past the skeleton: GLM-5.3's
    /// `model.language_model.layers.45.*` at `num_hidden_layers = 45`, and
    /// the DeepSeek-V3 `model.layers.61.*` nextn block.
    ExtraLayer { first: usize, count: usize },
}

/// Layer index of a `*.layers.N.*` / `layers.N.*` tensor name.
///
/// Same acceptance as `preflight::validate_layer_coverage`: any prefix
/// (`model.`, `model.language_model.`, `backbone.`) or none at all
/// (Mistral consolidated checkpoints).
fn layer_index(name: &str) -> Option<usize> {
    let tail = if let Some(pos) = name.find(".layers.") {
        &name[pos + ".layers.".len()..]
    } else {
        name.strip_prefix("layers.")?
    };
    let end = tail.find('.')?;
    tail[..end].parse().ok()
}

/// Detect the MTP layout from raw tensor names.
///
/// `mtp.*` wins when both are present — that is the layout the generic
/// `load_mtp_weights_multi` path binds.
pub fn detect<'a>(
    names: impl Iterator<Item = &'a str>,
    num_hidden_layers: usize,
) -> Option<MtpLayout> {
    let mut extras: Vec<usize> = Vec::new();
    let mut has_prefix = false;
    for n in names {
        if n.starts_with("mtp.") {
            has_prefix = true;
        } else if let Some(idx) = layer_index(n)
            && idx >= num_hidden_layers
        {
            extras.push(idx);
        }
    }
    if has_prefix {
        return Some(MtpLayout::MtpPrefix);
    }
    if extras.is_empty() {
        return None;
    }
    extras.sort_unstable();
    extras.dedup();
    Some(MtpLayout::ExtraLayer {
        first: extras[0],
        count: extras.len(),
    })
}

/// [`detect`] over a loaded [`WeightStore`].
///
/// 🪤 On an EP worker the store has already been filtered by expert index,
/// but never by layer — the MTP block's non-expert tensors are present on
/// every rank, so this answers the same on rank 0 and rank 1.
pub fn detect_in_store(store: &WeightStore, config: &ModelConfig) -> Option<MtpLayout> {
    detect(store.names(), config.num_hidden_layers)
}

#[cfg(test)]
mod tests {
    use super::{MtpLayout, detect, layer_index};

    #[test]
    fn glm5_next_layer_45_block_is_detected() {
        // The regression this module exists for: GLM-5.3 nests the text stack
        // under `model.language_model.` and puts MTP at layers.45.
        let names = [
            "model.language_model.layers.44.self_attn.q_proj.weight",
            "model.language_model.layers.45.eh_proj.weight",
            "model.language_model.layers.45.shared_head.norm.weight",
            "model.language_model.layers.45.mlp.experts.7.down_proj.weight",
            "lm_head.weight",
        ];
        assert_eq!(
            detect(names.iter().copied(), 45),
            Some(MtpLayout::ExtraLayer {
                first: 45,
                count: 1
            }),
        );
    }

    #[test]
    fn qwen_mtp_prefix_still_detected() {
        let names = ["model.layers.0.self_attn.q_proj.weight", "mtp.fc.weight"];
        assert_eq!(
            detect(names.iter().copied(), 48),
            Some(MtpLayout::MtpPrefix)
        );
    }

    #[test]
    fn deepseek_multi_module_prefix_still_detected() {
        let names = ["mtp.0.self_attn.q_proj.weight", "mtp.1.eh_proj.weight"];
        assert_eq!(
            detect(names.iter().copied(), 61),
            Some(MtpLayout::MtpPrefix)
        );
    }

    #[test]
    fn deepseek_v3_nextn_extra_layer_detected() {
        let names = [
            "model.layers.60.mlp.gate.weight",
            "model.layers.61.eh_proj.weight",
        ];
        assert_eq!(
            detect(names.iter().copied(), 61),
            Some(MtpLayout::ExtraLayer {
                first: 61,
                count: 1
            }),
        );
    }

    #[test]
    fn plain_checkpoint_has_no_mtp() {
        let names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.47.mlp.down_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ];
        assert_eq!(detect(names.iter().copied(), 48), None);
    }

    #[test]
    fn vision_tower_blocks_are_not_mistaken_for_mtp() {
        // `model.visual.blocks.N.*` has no `.layers.` segment; a checkpoint
        // whose vision depth exceeds num_hidden_layers must NOT read as MTP.
        let names = [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.visual.blocks.23.attn.proj.weight",
            "model.visual.merger.proj.weight",
        ];
        assert_eq!(detect(names.iter().copied(), 45), None);
    }

    #[test]
    fn multi_extra_layers_are_counted() {
        let names = [
            "model.layers.61.eh_proj.weight",
            "model.layers.62.eh_proj.weight",
        ];
        assert_eq!(
            detect(names.iter().copied(), 61),
            Some(MtpLayout::ExtraLayer {
                first: 61,
                count: 2
            }),
        );
    }

    #[test]
    fn unprefixed_mistral_layer_names_parse() {
        assert_eq!(layer_index("layers.3.attention.wq.weight"), Some(3));
        assert_eq!(
            layer_index("model.language_model.layers.45.eh_proj.weight"),
            Some(45)
        );
        assert_eq!(layer_index("lm_head.weight"), None);
        assert_eq!(layer_index("model.visual.blocks.2.attn.proj.weight"), None);
    }
}
