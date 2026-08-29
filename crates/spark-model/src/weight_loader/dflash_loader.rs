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
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::weight_map::{DenseWeight, dense};

mod config;
pub use config::*;

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

    // ── DSpark heads (optional; RadixArk Qwen3.8-27B-DSpark ships all 4) ──
    /// Markov head `markov_w1`: `[vocab, markov_rank]` BF16 embedding table
    /// (prev-token → latent). Present iff `config.markov_rank > 0` and the
    /// checkpoint carries the tensor.
    pub markov_w1: Option<DenseWeight>,
    /// Markov head `markov_w2`: `[vocab, markov_rank]` BF16
    /// (`nn.Linear(rank, vocab, bias=False).weight`, i.e. `[N, K]` for the
    /// GEMV convention). Projects the latent back to a full-vocab bias.
    pub markov_w2: Option<DenseWeight>,
    /// Confidence head weight: `[1, hidden(+rank)]` BF16.
    pub confidence_proj: Option<DenseWeight>,
    /// Confidence head bias: `[1]` BF16.
    pub confidence_bias: Option<DenseWeight>,

    // ── DFlash2 candidate selector (None on DFlash1/DSpark drafters) ──
    /// `candidate_selector.predecessor_codebook` `[vocab, selector_rank]` BF16.
    pub selector_pred: Option<DenseWeight>,
    /// `candidate_selector.successor_codebook` `[vocab, selector_rank]` BF16.
    pub selector_succ: Option<DenseWeight>,
    /// `candidate_selector.hidden_projection.weight` `[selector_rank, hidden]`.
    pub selector_hidden_proj: Option<DenseWeight>,
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

    // ── DFlash2 grouped dynamic causal convs (None on DFlash1 drafters) ──
    /// `attention_conv.base_kernel` `[2, kernel_size, hidden]` BF16 —
    /// static tap weights (index 0 = prepare/pre-sublayer, 1 = finish/post).
    pub attention_conv_base: Option<DenseWeight>,
    /// `attention_conv.kernel_projection.weight`
    /// `[2 * kernel_size * groups, hidden]` BF16 — dynamic tap generator.
    pub attention_conv_proj: Option<DenseWeight>,
    /// `mlp_conv.base_kernel`, same shape as attention_conv_base.
    pub mlp_conv_base: Option<DenseWeight>,
    /// `mlp_conv.kernel_projection.weight`, same shape as attention_conv_proj.
    pub mlp_conv_proj: Option<DenseWeight>,
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

    // DFlash2 detection: conv/selector dims declared in dflash_config AND the
    // layer-0 conv tensor present. All four families load per layer or none.
    let dflash2_conv = drafter_config
        .dflash_config
        .as_ref()
        .map(|c| c.conv_kernel_size > 0 && c.conv_group_size > 0)
        .unwrap_or(false)
        && drafter_store.contains(&format!("{prefix}layers.0.attention_conv.base_kernel"));

    let layer_count = drafter_config.num_hidden_layers;
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        let lp = format!("{prefix}layers.{i}");
        let (attention_conv_base, attention_conv_proj, mlp_conv_base, mlp_conv_proj) =
            if dflash2_conv {
                (
                    Some(dense(
                        drafter_store,
                        &format!("{lp}.attention_conv.base_kernel"),
                    )?),
                    Some(dense(
                        drafter_store,
                        &format!("{lp}.attention_conv.kernel_projection.weight"),
                    )?),
                    Some(dense(drafter_store, &format!("{lp}.mlp_conv.base_kernel"))?),
                    Some(dense(
                        drafter_store,
                        &format!("{lp}.mlp_conv.kernel_projection.weight"),
                    )?),
                )
            } else {
                (None, None, None, None)
            };
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
            attention_conv_base,
            attention_conv_proj,
            mlp_conv_base,
            mlp_conv_proj,
        };
        layers.push(layer);
    }

    // DFlash2 candidate selector. NOTE: the two codebooks ship with NO
    // `.weight` suffix (raw nn.Parameter-style keys — the z-lab loader
    // key-maps them; verified against the incoai safetensors header).
    let selector_key = format!("{prefix}candidate_selector.predecessor_codebook");
    let (selector_pred, selector_succ, selector_hidden_proj) = if drafter_config
        .dflash_config
        .as_ref()
        .map(|c| c.selector_rank > 0 && c.selector_top_k > 0)
        .unwrap_or(false)
        && drafter_store.contains(&selector_key)
    {
        (
            Some(
                dense(drafter_store, &selector_key)
                    .context("DFlash2: load candidate_selector.predecessor_codebook")?,
            ),
            Some(
                dense(
                    drafter_store,
                    &format!("{prefix}candidate_selector.successor_codebook"),
                )
                .context("DFlash2: load candidate_selector.successor_codebook")?,
            ),
            Some(
                dense(
                    drafter_store,
                    &format!("{prefix}candidate_selector.hidden_projection.weight"),
                )
                .context("DFlash2: load candidate_selector.hidden_projection.weight")?,
            ),
        )
    } else {
        (None, None, None)
    };
    if dflash2_conv || selector_pred.is_some() {
        tracing::info!(
            "DFlash2 heads loaded: convs={} (k={}, group={}), selector={} (rank={}, top_k={})",
            dflash2_conv,
            drafter_config
                .dflash_config
                .as_ref()
                .map(|c| c.conv_kernel_size)
                .unwrap_or(0),
            drafter_config
                .dflash_config
                .as_ref()
                .map(|c| c.conv_group_size)
                .unwrap_or(0),
            selector_pred.is_some(),
            drafter_config
                .dflash_config
                .as_ref()
                .map(|c| c.selector_rank)
                .unwrap_or(0),
            drafter_config
                .dflash_config
                .as_ref()
                .map(|c| c.selector_top_k)
                .unwrap_or(0),
        );
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

    // ── DSpark heads (optional) ──────────────────────────────────────
    // Tensor names verified against RadixArk/Qwen3.8-27B-DSpark
    // (model.safetensors, 62 tensors): `markov_head.markov_w1.weight`
    // [248320, 256], `markov_head.markov_w2.weight` [248320, 256],
    // `confidence_head.proj.weight` [1, 5376], `confidence_head.proj.bias`
    // [1] — all BF16, bare layout (same prefix convention as fc.weight).
    let markov_key = format!("{prefix}markov_head.markov_w1.weight");
    let (markov_w1, markov_w2) =
        if drafter_config.markov_rank > 0 && drafter_store.contains(&markov_key) {
            if let Some(kind) = drafter_config.markov_head_type.as_deref()
                && kind != "vanilla"
            {
                anyhow::bail!(
                    "DSpark drafter declares markov_head_type={kind:?}; only \"vanilla\" \
                 (low-rank bigram bias) is defined by the reference implementation"
                );
            }
            let w1 = dense(drafter_store, &markov_key)
                .context("DSpark drafter: load markov_head.markov_w1.weight")?;
            let w2 = dense(
                drafter_store,
                &format!("{prefix}markov_head.markov_w2.weight"),
            )
            .context("DSpark drafter: load markov_head.markov_w2.weight")?;
            (Some(w1), Some(w2))
        } else {
            if drafter_config.markov_rank > 0 {
                tracing::warn!(
                    "DSpark drafter config declares markov_rank={} but the checkpoint has \
                 no {markov_key} — running as plain DFlash (Markov bias disabled)",
                    drafter_config.markov_rank,
                );
            }
            (None, None)
        };
    let conf_key = format!("{prefix}confidence_head.proj.weight");
    let (confidence_proj, confidence_bias) =
        if drafter_config.enable_confidence_head && drafter_store.contains(&conf_key) {
            let w = dense(drafter_store, &conf_key)
                .context("DSpark drafter: load confidence_head.proj.weight")?;
            let b = dense(drafter_store, &format!("{prefix}confidence_head.proj.bias"))
                .context("DSpark drafter: load confidence_head.proj.bias")?;
            (Some(w), Some(b))
        } else {
            (None, None)
        };
    if markov_w1.is_some() || confidence_proj.is_some() {
        tracing::info!(
            "DSpark heads loaded: markov={} (rank={}), confidence={} (with_markov={})",
            markov_w1.is_some(),
            drafter_config.markov_rank,
            confidence_proj.is_some(),
            drafter_config.confidence_head_with_markov,
        );
    }

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
        markov_w1,
        markov_w2,
        confidence_proj,
        confidence_bias,
        selector_pred,
        selector_succ,
        selector_hidden_proj,
    }))
}

#[cfg(test)]
#[path = "dflash_loader/loader_tests.rs"]
mod loader_tests;
