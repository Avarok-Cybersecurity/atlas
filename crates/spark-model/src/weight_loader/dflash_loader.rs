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

/// Load a drafter GEMM weight that may be NVFP4-packed.
///
/// ModelOpt NVFP4 drafter exports (NVIDIA-Nemotron-3.5-Lightning DFlash:
/// `fc` + every attention/MLP projection) ship U8 `[n, k/2]` packed nibbles
/// with F8_E4M3 `[n, k/16]` per-block scales and an F32 `weight_scale_2` —
/// under the SAME tensor names a BF16 checkpoint uses. Loading them as BF16
/// made every GEMM read 4x past the allocation (the Lightning bring-up's
/// CUDA 700, fault grid [21,12,1] = the fc [2688, 16128] GEMM). Detect the
/// packed form by dtype and GPU-dequantize to BF16; BF16 tensors pass
/// through untouched, so existing drafters are byte-identical.
fn dense_bf16_or_nvfp4(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let t = store.get(name)?;
    if t.dtype == spark_runtime::weights::WeightDtype::UInt8 {
        let prefix = name.strip_suffix(".weight").unwrap_or(name);
        anyhow::ensure!(
            t.shape.len() == 2,
            "NVFP4-packed drafter tensor {name} has shape {:?} (want 2-D)",
            t.shape
        );
        let (n, half_k) = (t.shape[0], t.shape[1]);
        tracing::info!(
            "DFlash drafter: dequantizing NVFP4 {name} [{n}, {}] -> BF16",
            half_k * 2
        );
        return crate::weight_map::dequant_nvfp4_to_bf16(store, prefix, n, half_k * 2, gpu);
    }
    Ok(DenseWeight { weight: t.ptr })
}

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
    #[serde(default)]
    pub rope_scaling: Option<DflashRopeScaling>,
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

    let fc = dense_bf16_or_nvfp4(drafter_store, &format!("{prefix}fc.weight"), _gpu)
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
            q_proj: dense_bf16_or_nvfp4(drafter_store, &format!("{lp}.self_attn.q_proj.weight"), _gpu)?,
            k_proj: dense_bf16_or_nvfp4(drafter_store, &format!("{lp}.self_attn.k_proj.weight"), _gpu)?,
            v_proj: dense_bf16_or_nvfp4(drafter_store, &format!("{lp}.self_attn.v_proj.weight"), _gpu)?,
            o_proj: dense_bf16_or_nvfp4(drafter_store, &format!("{lp}.self_attn.o_proj.weight"), _gpu)?,
            q_norm: dense(drafter_store, &format!("{lp}.self_attn.q_norm.weight"))?,
            k_norm: dense(drafter_store, &format!("{lp}.self_attn.k_norm.weight"))?,
            gate_proj: dense_bf16_or_nvfp4(drafter_store, &format!("{lp}.mlp.gate_proj.weight"), _gpu)?,
            up_proj: dense_bf16_or_nvfp4(drafter_store, &format!("{lp}.mlp.up_proj.weight"), _gpu)?,
            down_proj: dense_bf16_or_nvfp4(drafter_store, &format!("{lp}.mlp.down_proj.weight"), _gpu)?,
        };
        layers.push(layer);
    }

    // `d2t` (draft-id → target-id) is absent from Qwen3.6-DFlash because
    // both vocabs are 248320. If a future drafter ships a smaller vocab
    // (vLLM supports this via `draft_vocab_size`), the int64 mapping table
    // would land here. Probing first to keep this loader compatible.
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
        drafter_config.block_size,
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
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHIPPED_CONFIG: &str = r#"{
        "hidden_size": 2048,
        "num_hidden_layers": 8,
        "intermediate_size": 6144,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 248320,
        "draft_vocab_size": 248320,
        "tie_word_embeddings": false,
        "block_size": 16,
        "rope_theta": 10000000.0,
        "rope_scaling": null,
        "dflash_config": {
            "mask_token_id": 248070,
            "target_layer_ids": [1, 10, 19, 28, 37]
        }
    }"#;

    #[test]
    fn shipped_qwen3_6_fields_reach_the_runtime_config() {
        let config = parse_dflash_config(SHIPPED_CONFIG).expect("parse drafter config");
        assert_eq!(config.num_hidden_layers, 8);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.intermediate_size, 6144);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 248320);
        assert_eq!(config.draft_vocab_size, Some(248320));
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.block_size, 16);
        assert_eq!(config.rope_theta, 10_000_000.0);
        assert!(config.rope_scaling.is_none());
        let sub = config.dflash_config.expect("dflash_config present");
        assert_eq!(sub.mask_token_id, 248070);
        assert_eq!(sub.target_layer_ids, vec![1, 10, 19, 28, 37]);
    }

    #[test]
    fn omitted_optional_fields_use_runtime_defaults() {
        let config = parse_dflash_config(
            r#"{
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "vocab_size": 256
            }"#,
        )
        .unwrap();

        assert_eq!(config.block_size, 16);
        assert_eq!(config.rope_theta, 10_000_000.0);
        assert!(!config.tie_word_embeddings);
        assert!(config.draft_vocab_size.is_none());
        assert!(config.dflash_config.is_none());
        assert!(config.rope_scaling.is_none());
    }

    #[test]
    fn malformed_runtime_field_is_rejected_with_parser_context() {
        let malformed =
            SHIPPED_CONFIG.replace("\"mask_token_id\": 248070", "\"mask_token_id\": \"bad\"");
        let error = parse_dflash_config(&malformed).unwrap_err();
        assert_eq!(error.to_string(), "Parsing DFlash drafter config.json");
        assert!(format!("{error:#}").contains("invalid type: string \"bad\""));
    }
}
