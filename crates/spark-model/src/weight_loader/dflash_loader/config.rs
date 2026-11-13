// SPDX-License-Identifier: AGPL-3.0-only

//! Drafter HF `config.json` schema: [`DflashConfig`] plus the nested
//! [`DflashRopeScaling`] / [`DflashSubConfig`] blocks and their serde
//! defaults. Split out of `dflash_loader.rs` to stay under the 500-LoC
//! cap; field semantics are documented on the types themselves.

use serde::Deserialize;

/// Drafter HF `config.json` (subset Atlas consumes). Mirrors
/// `z-lab/Qwen3.6-35B-A3B-DFlash/config.json` field names verbatim so
/// `serde_json::from_str` works directly on the raw file.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub draft_vocab_size: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Block size γ. Qwen3.6-DFlash ships `block_size: 16`.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// DFlash-specific nested config object.
    #[serde(default)]
    pub dflash_config: Option<DflashSubConfig>,
    /// Drafter base RoPE θ. Defaults to 10M (matches Qwen3.6-DFlash).
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// HF-style `rope_scaling` block. `None` ⇒ plain RoPE (the v2 2026-04-27
    /// Qwen3.6-DFlash drafter ships `rope_scaling: null`). When present and
    /// `rope_type == "yarn"`, the drafter's YaRN parameters are used to
    /// build the inv_freq table at construction time.
    /// `alias = "rope_parameters"`: newer transformers releases (RadixArk
    /// DSpark, incoai DFlash2) ship the block under that key — without the
    /// alias Atlas silently drops the scaling (the RadixArk config.json had
    /// to be hand-patched before this).
    #[serde(default, alias = "rope_parameters")]
    pub rope_scaling: Option<DflashRopeScaling>,
    /// DSpark Markov head rank. `0` (default) ⇒ plain DFlash drafter with no
    /// Markov head. RadixArk `Qwen3.8-27B-DSpark` ships `markov_rank: 256`
    /// top-level (SpecForge `DSparkConfig` convention — see the checkpoint's
    /// `dspark.py`: DSpark fields are declared as top-level config attrs).
    #[serde(default)]
    pub markov_rank: usize,
    /// DSpark Markov head flavor. Only `"vanilla"` (low-rank learned bigram
    /// bias) is defined by the reference; anything else is rejected at load.
    #[serde(default)]
    pub markov_head_type: Option<String>,
    /// DSpark confidence head (`AcceptRatePredictor`): a `Linear(input, 1)`
    /// predicting per-draft-position acceptance probability, used for
    /// adaptive block length. Loaded when present; consumed by the dynamic-K
    /// scheduling phase (the Markov fixup works without it).
    #[serde(default)]
    pub enable_confidence_head: bool,
    /// When true the confidence head's input is `[hidden ‖ markov_embed]`
    /// (input_dim = hidden_size + markov_rank); when false, hidden only.
    #[serde(default = "default_true")]
    pub confidence_head_with_markov: bool,
    /// HF `sliding_window` (tokens). `z-lab/Qwen3.5-4B-DFlash` ships 4096
    /// (trained with sliding-window attention for long context). `None` on the
    /// older Qwen3.6 drafters, which attend the full prefix.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// HF `use_sliding_window`: whether `layer_types` entries of
    /// `"sliding_attention"` really are windowed. Absent = false.
    #[serde(default)]
    pub use_sliding_window: bool,
    /// HF per-layer attention kind, one entry per drafter layer
    /// (`"sliding_attention"` or `"full_attention"`). The 4B drafter ships five
    /// sliding layers and a full-attention last layer. Empty = all layers
    /// share one kind (read it through [`DflashConfig::layer_is_sliding`]).
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// Number of layers of the TARGET the drafter was trained against
    /// (`num_target_layers`; 32 for Qwen3.5-4B). `None` = not stated.
    #[serde(default)]
    pub num_target_layers: Option<usize>,
}

fn default_true() -> bool {
    true
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}

/// Subset of HF `rope_scaling` block consumed by Atlas. Mirrors the field
/// names in `transformers`' Qwen3 config so `serde_json::from_str` works
/// directly on the drafter's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashRopeScaling {
    /// Currently only `"yarn"` is recognised; anything else falls back to
    /// plain RoPE with a warning logged at construction time.
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<f32>,
    /// `rope_theta` as newer transformers releases state it, INSIDE the
    /// `rope_parameters` block (the 4B drafter ships no top-level
    /// `rope_theta`). Folded into [`DflashConfig::rope_theta`] by [`parse_normalized`]
    /// when the top level does not state one.
    #[serde(default)]
    pub rope_theta: Option<f32>,
}

fn default_block_size() -> usize {
    16
}

/// Nested `dflash_config` block in the drafter's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashSubConfig {
    /// Token id used to fill the γ "to-be-predicted" positions during draft
    /// inference. `248070` for Qwen3.6-DFlash.
    pub mask_token_id: u32,
    /// Target-model layer indices to capture intermediate hidden states from.
    /// `[1, 10, 19, 28, 37]` for Qwen3.6-35B-A3B-DFlash. Order matters:
    /// shallow-to-deep concatenation is what `fc` expects.
    pub target_layer_ids: Vec<usize>,
    /// Draft flavor tag. `"dspark"` marks a SpecForge-lineage drafter, whose
    /// row convention is SHIFTED (row j's output = token at position j+1; the
    /// anchor row's output is draft #1) versus the z-lab convention Atlas's
    /// forward was built for (row j predicts at j, row 0 = echo). The runtime
    /// keys the draft-vector rotation off this tag. DFlash2 checkpoints have
    /// no tag and keep the z-lab convention (verified against z-lab
    /// `dflash/model.py::dflash_generate` — anchor-row output discarded,
    /// mask rows fill in place).
    #[serde(default)]
    pub projector_type: Option<String>,

    // ── DFlash2 fields (incoai/z-lab `DFlash2DraftModel`); absent = DFlash1 ──
    /// Two-tap dynamic conv kernel size (2 for DFlash2).
    #[serde(default)]
    pub conv_kernel_size: usize,
    /// Channels per conv group (16 for DFlash2 → hidden/16 groups).
    #[serde(default)]
    pub conv_group_size: usize,
    /// Selector codebook rank (256 for DFlash2).
    #[serde(default)]
    pub selector_rank: usize,
    /// Candidates kept per position for the selector walk (16 for DFlash2).
    #[serde(default)]
    pub selector_top_k: usize,
    /// Block size γ as the drafter was TRAINED, when the checkpoint states it
    /// here. DFlash2 ships `block_size: 8` inside `dflash_config`; DFlash1
    /// ships it top-level only. `None` = not stated, fall back to the
    /// top-level field. Read through [`DflashConfig::effective_block_size`].
    #[serde(default)]
    pub block_size: Option<usize>,
}

impl DflashConfig {
    /// Resolved block size γ: the drafter's own trained value when the
    /// checkpoint states it, else the top-level field.
    ///
    /// The top-level `block_size` defaults to 16, and serde fills that default
    /// happily for a checkpoint that never mentioned it — so a DFlash2 drafter
    /// trained at 8 comes up as 16 unless the sub-config is consulted first.
    /// That is not a cosmetic mismatch: the serve then runs num_drafts=15
    /// against an 8-block drafter, which measured 0% accept on EVERY verify
    /// step, and sizes the drafter's per-sequence pools for twice the block it
    /// will ever use. `--dflash-gamma` still overrides both.
    pub fn effective_block_size(&self) -> usize {
        self.dflash_config
            .as_ref()
            .and_then(|c| c.block_size)
            .unwrap_or(self.block_size)
    }

    /// Number of target layers whose hidden states feed `fc` (8 for the 4B
    /// drafter, 5 for the Qwen3.6 drafters). 0 when `dflash_config` is absent.
    pub fn tap_count(&self) -> usize {
        self.dflash_config
            .as_ref()
            .map_or(0, |c| c.target_layer_ids.len())
    }

    /// `fc` input width for a target of hidden size `target_hidden`:
    /// `tap_count * target_hidden`.
    pub fn fc_in_features(&self, target_hidden: usize) -> usize {
        self.tap_count() * target_hidden
    }

    /// Whether drafter layer `layer` attends through a sliding window.
    /// Only when the checkpoint says `use_sliding_window` AND lists the layer
    /// as `"sliding_attention"`; every older drafter answers false.
    pub fn layer_is_sliding(&self, layer: usize) -> bool {
        self.use_sliding_window
            && self.sliding_window.is_some_and(|w| w > 0)
            && self
                .layer_types
                .get(layer)
                .is_some_and(|t| t == "sliding_attention")
    }

    /// Count of sliding layers among the drafter's `num_hidden_layers`.
    pub fn sliding_layer_count(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.layer_is_sliding(i))
            .count()
    }

    /// Fold the HF `rope_parameters` spelling into the plain-RoPE fields the
    /// head reads. Two cases, both pure bookkeeping:
    ///
    /// * the block states `rope_theta` and the top level does not
    ///   (`top_level_theta_stated == false`): the block's value wins over the
    ///   10M serde default, which is a coincidence for the 4B drafter and a bug
    ///   for any drafter with another theta;
    /// * `rope_type == "default"` (transformers' name for plain RoPE): the
    ///   block is dropped, so the head takes its plain-RoPE arm without the
    ///   "doesn't recognise rope_type" warning. The inverse-frequency table is
    ///   the same one either way.
    pub fn normalize_rope(&mut self, top_level_theta_stated: bool) {
        let Some(scaling) = self.rope_scaling.as_ref() else {
            return;
        };
        if !top_level_theta_stated && let Some(theta) = scaling.rope_theta {
            self.rope_theta = theta;
        }
        if scaling.rope_type.as_deref() == Some("default") {
            self.rope_scaling = None;
        }
    }

    /// Check that this drafter was built for a target with the given
    /// dimensions, before any weight is touched. A wrong pairing does not
    /// fail loudly later: the drafter shares the target's embedding and LM
    /// head and reads the target's hidden states through `fc`, so a mismatch
    /// is a GEMM over the wrong width or fluent-looking garbage drafts.
    ///
    /// * the capture ids are non-empty, strictly increasing and inside the
    ///   target's layer range;
    /// * `num_target_layers`, when the checkpoint states it, equals the
    ///   target's layer count (32 for Qwen3.5-4B and `holo-3.1-4b`);
    /// * the drafter's `vocab_size` equals the target's (shared embed/LM head);
    /// * `mask_token_id` is inside the vocabulary.
    pub fn validate_for_target(
        &self,
        target_hidden: usize,
        target_layers: usize,
        target_vocab: usize,
    ) -> anyhow::Result<()> {
        let Some(sub) = self.dflash_config.as_ref() else {
            anyhow::bail!("DFlash drafter config.json has no `dflash_config` block");
        };
        anyhow::ensure!(
            target_hidden > 0,
            "DFlash: the target reports hidden_size 0"
        );
        anyhow::ensure!(
            !sub.target_layer_ids.is_empty(),
            "DFlash drafter declares no target_layer_ids"
        );
        for pair in sub.target_layer_ids.windows(2) {
            anyhow::ensure!(
                pair[0] < pair[1],
                "DFlash drafter target_layer_ids {:?} are not strictly increasing (fc expects \
                 shallow-to-deep order)",
                sub.target_layer_ids
            );
        }
        if let Some(&deepest) = sub.target_layer_ids.last() {
            anyhow::ensure!(
                deepest < target_layers,
                "DFlash drafter taps target layer {deepest} but the target has only \
                 {target_layers} layers: this drafter was built for a different model"
            );
        }
        if let Some(trained) = self.num_target_layers {
            anyhow::ensure!(
                trained == target_layers,
                "DFlash drafter was trained against a {trained}-layer target but this target has \
                 {target_layers} layers"
            );
        }
        // `vocab_size` only: a drafter with a smaller `draft_vocab_size` and a
        // `d2t` remap states the target's vocabulary in `vocab_size` and is
        // handled (or refused) by the loader, not judged here.
        anyhow::ensure!(
            self.vocab_size == target_vocab,
            "DFlash drafter vocabulary ({}) differs from the target's {target_vocab}; the \
             drafter shares the target's embedding and LM head, so ids would not line up",
            self.vocab_size,
        );
        anyhow::ensure!(
            (sub.mask_token_id as usize) < self.vocab_size,
            "DFlash drafter mask_token_id {} is outside its vocabulary of {}",
            sub.mask_token_id,
            self.vocab_size
        );
        Ok(())
    }
}

/// Parse a drafter `config.json` and normalize the HF spellings the head does
/// not read directly (see [`DflashConfig::normalize_rope`]). The only entry
/// point `parse_dflash_config` uses.
pub fn parse_normalized(json: &str) -> serde_json::Result<DflashConfig> {
    let mut cfg: DflashConfig = serde_json::from_str(json)?;
    let top_level_theta_stated = serde_json::from_str::<serde_json::Value>(json)
        .ok()
        .is_some_and(|v| v.get("rope_theta").is_some());
    cfg.normalize_rope(top_level_theta_stated);
    Ok(cfg)
}
