// SPDX-License-Identifier: AGPL-3.0-only

//! Drafter `config.json` types.
//!
//! Split out of `dflash_loader.rs` to keep that file under the 500-line cap:
//! these are the pure `serde` shapes for the checkpoint's config, with no
//! loading logic. The weight structures and the loader itself stay in the
//! parent module.

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
}
