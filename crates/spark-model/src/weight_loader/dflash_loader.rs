// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash drafter weight loader.
//!
//! Loads `z-lab/Qwen3.6-{27B,35B-A3B}-DFlash`-style drafter checkpoints into
//! the typed [`DflashWeights`] structure consumed by
//! [`crate::layers::BlockDiffusionDraftHead`]. The drafter is a small
//! Qwen3-architecture transformer (8 layers, hidden=2048, GQA 32:4) with
//! these distinctive parts vs. a vanilla Qwen3:
//!
//!  * `model.fc` — `[len(target_layer_ids) * target_hidden, draft_hidden]`
//!    BF16 projection that maps the stack of captured target hidden states
//!    into the drafter's input space.
//!  * `model.hidden_norm` — RMSNorm applied to the projected target context
//!    before mixing with token embeddings.
//!  * `lm_head` — drafter ships its own (NOT tied to target's), so
//!    `tie_word_embeddings=false`.
//!  * Optional `d2t` — draft-vocab → target-vocab id remap (absent when
//!    drafter shares vocab with target, as in Qwen3.6-35B-A3B-DFlash where
//!    both = 248320).
//!  * Special `mask_token_id` (`248070` for Qwen3.6-DFlash) used for the γ
//!    "to-be-predicted" positions in block diffusion.
//!
//! Under TP the drafter is **not sharded** — it's small (~1–2 GB BF16),
//! every rank loads the full set. Mirrors the existing MTP-under-TP pattern
//! (`MTP loads ALL experts on every rank — no EP all_reduce needed`).

use anyhow::{Context, Result};
use serde::Deserialize;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::weight_map::{DenseWeight, dense};

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
    /// transformers-5.x home for rope settings (DFlash2 ships
    /// `rope_parameters: {rope_theta, rope_type}` and NO top-level
    /// `rope_theta`). When present, its theta wins over the field above.
    #[serde(default)]
    pub rope_parameters: Option<DflashRopeParameters>,
    /// HF-style `rope_scaling` block. `None` ⇒ plain RoPE (the v2 2026-04-27
    /// Qwen3.6-DFlash drafter ships `rope_scaling: null`). When present and
    /// `rope_type == "yarn"`, the drafter's YaRN parameters are used to
    /// build the inv_freq table at construction time.
    #[serde(default)]
    pub rope_scaling: Option<DflashRopeScaling>,
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}

/// transformers-5.x `rope_parameters` block (subset).
#[derive(Debug, Clone, Deserialize)]
pub struct DflashRopeParameters {
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rope_type: Option<String>,
}

impl DflashConfig {
    /// Resolved RoPE θ: `rope_parameters.rope_theta` (transformers 5.x /
    /// DFlash2) wins over the legacy top-level field.
    pub fn effective_rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.rope_theta)
            .unwrap_or(self.rope_theta)
    }
    /// Resolved block size γ: DFlash2 ships it INSIDE `dflash_config`
    /// (`block_size: 8`); DFlash1 ships it top-level. The top-level default
    /// (16) must not shadow a sub-config value.
    pub fn effective_block_size(&self) -> usize {
        self.dflash_config
            .as_ref()
            .and_then(|c| c.block_size)
            .unwrap_or(self.block_size)
    }
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

    // ── DFlash2 (incoai/Qwen3.8-27B-DFlash2, z-lab `DFlash2DraftModel`) ──
    // All four ship together; absence of all four = DFlash1. Field names
    // mirror the checkpoint's `dflash_config` block verbatim.
    /// Causal-conv tap count. `2` for Qwen3.8-27B-DFlash2.
    #[serde(default)]
    pub conv_kernel_size: Option<usize>,
    /// Channels sharing one dynamic kernel scalar. `16` (⇒ 320 groups at
    /// hidden 5120) for Qwen3.8-27B-DFlash2.
    #[serde(default)]
    pub conv_group_size: Option<usize>,
    /// Rank of the bilinear pair-scoring codebooks. `256`.
    #[serde(default)]
    pub selector_rank: Option<usize>,
    /// Candidates retained per draft position before the selector walk. `16`.
    #[serde(default)]
    pub selector_top_k: Option<usize>,
    /// DFlash2 ships block size γ HERE (8), not top-level.
    #[serde(default)]
    pub block_size: Option<usize>,
}

impl DflashSubConfig {
    /// True when the checkpoint declares the DFlash2 selector+conv config.
    pub fn is_dflash2(&self) -> bool {
        self.selector_rank.is_some() && self.conv_kernel_size.is_some()
    }
}

/// Raw weight bundle for the DFlash drafter, post-load.
///
/// Verified against `z-lab/Qwen3.6-35B-A3B-DFlash` (commit 42d3b34, May 2026):
/// the checkpoint ships 91 BF16 tensors — `fc.weight`, `hidden_norm.weight`,
/// `norm.weight`, plus 11 weights per drafter layer × 8 layers. **No
/// `embed_tokens` or `lm_head` are in the checkpoint** — the drafter shares
/// the target's embedding and LM head at construction time. This matches the
/// vLLM PR #40898 flow: when those keys are absent, vLLM's `AutoWeightsLoader`
/// adds them to `skip_substrs`, leaving the runtime to slot in the target's
/// pointers.
#[allow(dead_code)]
pub struct DflashWeights {
    pub config: DflashConfig,

    /// `[draft_hidden, len(target_layer_ids) * target_hidden]`.
    /// Qwen3.6-35B-A3B-DFlash: `[2048, 10240]`.
    pub fc: DenseWeight,
    /// `[draft_hidden]` — RMSNorm applied to the projected target context
    /// before mixing with token embeddings.
    pub hidden_norm: DenseWeight,
    /// `[draft_hidden]` — final RMSNorm before LM head.
    pub norm: DenseWeight,

    pub layers: Vec<DflashLayerWeights>,

    /// Present iff the drafter has a draft-id → target-id mapping (i.e.
    /// `draft_vocab_size != target_vocab_size`). Absent for
    /// Qwen3.6-35B-A3B-DFlash (both vocabs = 248320).
    pub draft_id_to_target_id: Option<Vec<i64>>,

    /// DFlash2 only: the candidate-path selector.
    pub candidate_selector: Option<DflashSelectorWeights>,
}

/// DFlash2 `GroupedDynamicCausalConv` weights for ONE sublayer site
/// (attention or MLP). Reference: z-lab/dflash `model.py`.
///
/// `base_kernel` is `[2, kernel_size, hidden]` — index 0 is the "prepare"
/// kernel (conv on the normed sublayer INPUT), index 1 the "finish" kernel
/// (conv on the sublayer OUTPUT, pre-residual). `kernel_projection` is
/// `[2 * kernel_size * groups, hidden]`; ONE projection of the normed input
/// yields the per-position dynamic kernels for BOTH stages.
#[allow(dead_code)]
pub struct DflashConvWeights {
    pub base_kernel: DenseWeight,
    pub kernel_projection: DenseWeight,
}

/// DFlash2 `CandidateSelector` weights. Codebooks are `[vocab, rank]` and
/// ship WITHOUT a `.weight` suffix (the reference loader remaps them — see
/// `DFlash2DraftModel::from_pretrained`'s `key_mapping`).
#[allow(dead_code)]
pub struct DflashSelectorWeights {
    /// `[rank, hidden]` — projects the drafter's final hidden to rank space.
    pub hidden_projection: DenseWeight,
    /// `[vocab, rank]` — row for the PRECEDING (chosen) token.
    pub predecessor_codebook: DenseWeight,
    /// `[vocab, rank]` — row for each CANDIDATE successor.
    pub successor_codebook: DenseWeight,
}

/// Per-drafter-layer raw weights (BF16). Same shape across all 8 layers.
#[allow(dead_code)]
pub struct DflashLayerWeights {
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
    /// DFlash2 only: dynamic causal conv around the attention sublayer.
    pub attention_conv: Option<DflashConvWeights>,
    /// DFlash2 only: dynamic causal conv around the MLP sublayer.
    pub mlp_conv: Option<DflashConvWeights>,
}

/// Probe a [`WeightStore`] for the presence of DFlash drafter weights.
/// Returns true if the store contains the unique `fc.weight` tensor that
/// DFlash drafters ship — a lightweight detection that doesn't load any
/// data. Both bare-key and `model.`-prefixed layouts are accepted; the
/// canonical `z-lab/Qwen3.6-{27B,35B-A3B}-DFlash` checkpoints ship the
/// bare layout (verified against commit 42d3b34, May 2026).
pub fn store_has_dflash_weights(store: &WeightStore) -> bool {
    store.contains("fc.weight") || store.contains("model.fc.weight")
}

/// Parse a DFlash drafter's `config.json` into a [`DflashConfig`]. Used by
/// `main.rs` after fetching the drafter's HF metadata to size the runtime
/// `BlockDiffusionDraftHead` (layer count, head_dim, vocab_size, the
/// `target_layer_ids` capture indices).
pub fn parse_dflash_config(json: &str) -> Result<DflashConfig> {
    serde_json::from_str(json).context("Parsing DFlash drafter config.json")
}

/// Load DFlash drafter weights from a separate [`WeightStore`] pointing at
/// the drafter checkpoint.
///
/// The drafter ships its weights at the **root** of the safetensors file
/// (no `model.` prefix), in the same naming convention as a vanilla Qwen3
/// transformer minus `embed_tokens` and `lm_head`. Atlas's runtime fills
/// those two from the *target* model's embedding / LM head at construction
/// time — exactly mirroring vLLM's "absent in checkpoint → skip_substrs →
/// share with parent" flow.
///
/// The probed key list (verified against `z-lab/Qwen3.6-35B-A3B-DFlash`):
///
/// ```text
///   fc.weight                                              [H, 5*H_target]
///   hidden_norm.weight                                     [H]
///   norm.weight                                            [H]
///   layers.{0..L-1}.input_layernorm.weight                 [H]
///   layers.{0..L-1}.post_attention_layernorm.weight        [H]
///   layers.{0..L-1}.self_attn.q_proj.weight                [Q*Hd, H]
///   layers.{0..L-1}.self_attn.k_proj.weight                [Kv*Hd, H]
///   layers.{0..L-1}.self_attn.v_proj.weight                [Kv*Hd, H]
///   layers.{0..L-1}.self_attn.o_proj.weight                [H, Q*Hd]
///   layers.{0..L-1}.self_attn.q_norm.weight                [Hd]
///   layers.{0..L-1}.self_attn.k_norm.weight                [Hd]
///   layers.{0..L-1}.mlp.gate_proj.weight                   [I, H]
///   layers.{0..L-1}.mlp.up_proj.weight                     [I, H]
///   layers.{0..L-1}.mlp.down_proj.weight                   [H, I]
/// ```
///
/// where `H=2048`, `H_target=2048`, `Q=32`, `Kv=4`, `Hd=128`, `I=6144`,
/// `L=8` for Qwen3.6-35B-A3B-DFlash.
///
/// Under TP the drafter is replicated, not sharded — `tp_size>1` produces
/// the same per-rank result as `tp_size=1`. Memory cost: ~948 MB BF16
/// per rank, trivially below the 119 GB GB10 budget.
pub fn load_dflash_weights(
    drafter_store: &WeightStore,
    drafter_config: &DflashConfig,
    _gpu: &dyn GpuBackend,
    _tp_size: usize,
) -> Result<Option<DflashWeights>> {
    if !store_has_dflash_weights(drafter_store) {
        tracing::debug!("DFlash drafter store has no `fc.weight` — skipping");
        return Ok(None);
    }

    // Detect bare vs. `model.`-prefixed layout. `z-lab` checkpoints use
    // bare; we accept either to be robust against a hypothetical re-upload
    // that uses the prefixed layout.
    let prefix = if drafter_store.contains("model.fc.weight") {
        "model."
    } else {
        ""
    };

    let fc = dense(drafter_store, &format!("{prefix}fc.weight"))
        .context("DFlash drafter: load fc.weight")?;
    let hidden_norm = dense(drafter_store, &format!("{prefix}hidden_norm.weight"))
        .context("DFlash drafter: load hidden_norm.weight")?;
    let norm = dense(drafter_store, &format!("{prefix}norm.weight"))
        .context("DFlash drafter: load norm.weight")?;

    let layer_count = drafter_config.num_hidden_layers;
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        let lp = format!("{prefix}layers.{i}");
        let layer = DflashLayerWeights {
            input_layernorm: dense(drafter_store, &format!("{lp}.input_layernorm.weight"))?,
            post_attention_layernorm: dense(
                drafter_store,
                &format!("{lp}.post_attention_layernorm.weight"),
            )?,
            q_proj: dense(drafter_store, &format!("{lp}.self_attn.q_proj.weight"))?,
            k_proj: dense(drafter_store, &format!("{lp}.self_attn.k_proj.weight"))?,
            v_proj: dense(drafter_store, &format!("{lp}.self_attn.v_proj.weight"))?,
            o_proj: dense(drafter_store, &format!("{lp}.self_attn.o_proj.weight"))?,
            q_norm: dense(drafter_store, &format!("{lp}.self_attn.q_norm.weight"))?,
            k_norm: dense(drafter_store, &format!("{lp}.self_attn.k_norm.weight"))?,
            gate_proj: dense(drafter_store, &format!("{lp}.mlp.gate_proj.weight"))?,
            up_proj: dense(drafter_store, &format!("{lp}.mlp.up_proj.weight"))?,
            down_proj: dense(drafter_store, &format!("{lp}.mlp.down_proj.weight"))?,
            // DFlash2: per-sublayer dynamic causal convs. Probed per layer so
            // a DFlash1 checkpoint (no conv tensors) loads exactly as before.
            attention_conv: load_conv(drafter_store, &format!("{lp}.attention_conv"))?,
            mlp_conv: load_conv(drafter_store, &format!("{lp}.mlp_conv"))?,
        };
        layers.push(layer);
    }

    // `d2t` (draft-id → target-id) is absent from Qwen3.6-DFlash because
    // both vocabs are 248320. If a future drafter ships a smaller vocab
    // (vLLM supports this via `draft_vocab_size`), the int64 mapping table
    // would land here. Probing first to keep this loader compatible.
    // DFlash2 candidate selector. The codebooks ship WITHOUT a `.weight`
    // suffix (reference `from_pretrained` remaps them; we address them by
    // their on-disk names directly).
    let candidate_selector = if drafter_store.contains(&format!(
        "{prefix}candidate_selector.hidden_projection.weight"
    )) {
        Some(DflashSelectorWeights {
            hidden_projection: dense(
                drafter_store,
                &format!("{prefix}candidate_selector.hidden_projection.weight"),
            )
            .context("DFlash2: load candidate_selector.hidden_projection")?,
            predecessor_codebook: dense(
                drafter_store,
                &format!("{prefix}candidate_selector.predecessor_codebook"),
            )
            .context("DFlash2: load predecessor_codebook")?,
            successor_codebook: dense(
                drafter_store,
                &format!("{prefix}candidate_selector.successor_codebook"),
            )
            .context("DFlash2: load successor_codebook")?,
        })
    } else {
        None
    };
    if candidate_selector.is_some()
        != drafter_config
            .dflash_config
            .as_ref()
            .is_some_and(|c| c.is_dflash2())
    {
        anyhow::bail!(
            "DFlash2 mismatch: config declares selector_rank/conv_kernel_size = {:?} but \
             candidate_selector tensors present = {} — checkpoint and config disagree",
            drafter_config
                .dflash_config
                .as_ref()
                .map(|c| c.is_dflash2()),
            candidate_selector.is_some(),
        );
    }

    let draft_id_to_target_id = if drafter_store.contains(&format!("{prefix}d2t"))
        || drafter_store.contains(&format!("{prefix}draft_id_to_target_id"))
    {
        // Mapping is loaded into device memory by upstream paths — for now
        // we just record presence. Phase 2.5 will copy it to a host Vec<i64>
        // when the head needs it for logit remapping.
        tracing::warn!(
            "DFlash drafter has draft-id→target-id mapping; remapping path is not yet wired (Phase 2.5 follow-up)"
        );
        Some(Vec::new())
    } else {
        None
    };

    tracing::info!(
        "DFlash drafter loaded: {} layers, hidden={}, vocab={}, γ={}, target_layers={:?}",
        layers.len(),
        drafter_config.hidden_size,
        drafter_config.vocab_size,
        drafter_config.effective_block_size(),
        drafter_config
            .dflash_config
            .as_ref()
            .map(|c| c.target_layer_ids.as_slice())
            .unwrap_or(&[]),
    );

    Ok(Some(DflashWeights {
        config: drafter_config.clone(),
        fc,
        hidden_norm,
        norm,
        layers,
        draft_id_to_target_id,
        candidate_selector,
    }))
}

/// Probe-and-load one DFlash2 `GroupedDynamicCausalConv` site
/// (`layers.N.attention_conv` / `layers.N.mlp_conv`). Returns `Ok(None)`
/// when the site is absent (DFlash1 checkpoints).
fn load_conv(store: &WeightStore, site: &str) -> Result<Option<DflashConvWeights>> {
    if !store.contains(&format!("{site}.base_kernel")) {
        return Ok(None);
    }
    Ok(Some(DflashConvWeights {
        base_kernel: dense(store, &format!("{site}.base_kernel"))
            .with_context(|| format!("DFlash2: load {site}.base_kernel"))?,
        kernel_projection: dense(store, &format!("{site}.kernel_projection.weight"))
            .with_context(|| format!("DFlash2: load {site}.kernel_projection"))?,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke-test the DFlash drafter `config.json` parser against the live
    /// `z-lab/Qwen3.6-35B-A3B-DFlash` checkpoint downloaded into the user's
    /// HF cache. Skipped when the cache directory isn't populated — keeps
    /// CI hermetic. Asserts the locked drafter dimensions: 8 layers,
    /// hidden=2048, vocab=248320, γ=16, mask=248070, layer_ids=[1,10,19,28,37].
    #[test]
    fn parse_qwen3_6_35b_dflash_config() {
        const SNAP: &str = "/workspace/.cache/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/42d3b34d588423cdae7ba8f53a8cf7789346a719/config.json";
        let json = match std::fs::read_to_string(SNAP) {
            Ok(s) => s,
            Err(_) => {
                tracing::warn!("Skipping: drafter snapshot not in cache");
                return;
            }
        };
        let config = parse_dflash_config(&json).expect("parse drafter config");
        assert_eq!(config.num_hidden_layers, 8);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.intermediate_size, 6144);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 248320);
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.block_size, 16);
        let sub = config.dflash_config.expect("dflash_config present");
        assert_eq!(sub.mask_token_id, 248070);
        assert_eq!(sub.target_layer_ids, vec![1, 10, 19, 28, 37]);
    }
}
