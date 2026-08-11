// SPDX-License-Identifier: AGPL-3.0-only

//! Token overlay (Feature 2): PEFT `trainable_tokens` / `modules_to_save`
//! vocab-extension and embed/lm_head row replacement, ported from the NLLB
//! `token_adapter` overlay to the main decoder (`TransformerModel`) path.
//!
//! This module is the PURE, GPU-free half: on-disk tensor classification
//! ([`classify_overlay_key`]), the collection struct the loader fills
//! ([`OverlayTensors`]), and the row-selection math ([`clamp_trainable_to_vocab`],
//! [`build_override_set`], [`override_source`]) that decides which vocab rows an
//! adapter overrides and where each row's replacement bytes come from. The GPU
//! materialization (row-diff kernel, compact-row copy, device tables) lives in
//! [`super::overlay_build`]; the forward hooks live in `model/token_overlay.rs`.
//!
//! Three on-disk mechanisms feed one runtime overlay:
//! - `trainable_tokens`: `…token_adapter.base_layer.weight [R,h]` bf16 +
//!   `…token_adapter.trainable_tokens_delta [T,h]` f32 (full row replacement,
//!   PEFT `index_copy` semantics: `E[idx[k]] = delta[k]`, NOT `base+delta`).
//! - `modules_to_save[embed_tokens|lm_head]`: a full `…embed_tokens.weight`
//!   `[vocab,h]` replacement (same builder, no delta — the row-diff finds the
//!   changed rows).
//! - `lora_embedding_A/B`: classic low-rank embed LoRA — Tier-2, classified
//!   here so the loader can NAME-reject it rather than mis-route it.

use anyhow::{Result, bail};

/// Rows whose max abs difference from the served embed table exceeds this are
/// treated as "overridden" by a `modules_to_save`/baked base_layer. Clears
/// bf16 rounding noise (matching rows land ≤0.05; a real differing row ≥1.3).
pub const ROWDIFF_THRESH: f32 = 0.1;

/// Which tied embedding an overlay tensor belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OverlayModule {
    /// Input embedding table (`embed_tokens` / NLLB `shared`).
    EmbedTokens,
    /// Output projection (`lm_head`). Distinct buffer when untied.
    LmHead,
}

/// The role an overlay tensor plays in [`build_overlay`](super::overlay_build::build_overlay).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayTensorKind {
    /// `token_adapter.base_layer.weight` — the adapter's own `[R,h]` table.
    Base,
    /// `token_adapter.trainable_tokens_delta` — `[T,h]` replacement rows.
    Delta,
    /// `modules_to_save` full weight (`embed_tokens.weight` / `lm_head.weight`).
    FullSave,
    /// Classic low-rank embedding LoRA A factor (Tier-2, load-rejected).
    LoraEmbedA,
    /// Classic low-rank embedding LoRA B factor (Tier-2, load-rejected).
    LoraEmbedB,
}

/// A classified overlay tensor: its target embedding and its role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverlayTensor {
    pub module: OverlayModule,
    pub kind: OverlayTensorKind,
}

/// Classify a PEFT adapter tensor as a token-overlay tensor, or `None` if it is
/// an ordinary `lora_A/lora_B` weight (which [`super::classify_key`] handles).
///
/// Prefix-agnostic beyond the PEFT wrapper, matching `classify_key`: it accepts
/// both `…model.embed_tokens.…` and the multimodal `…model.language_model.
/// embed_tokens.…` spellings. Suffix order matters — the `token_adapter` /
/// `lora_embedding` forms are checked before the bare `.weight`
/// (`modules_to_save`) form so `…token_adapter.base_layer.weight` is not
/// mis-read as a full save.
pub fn classify_overlay_key(key: &str) -> Option<OverlayTensor> {
    let stripped = key.strip_prefix("base_model.model.")?;
    // Ordinary LoRA weights are never overlays.
    if stripped.ends_with(".lora_A.weight") || stripped.ends_with(".lora_B.weight") {
        return None;
    }
    let module = overlay_module_of(stripped)?;
    let kind = if stripped.ends_with(".token_adapter.base_layer.weight") {
        OverlayTensorKind::Base
    } else if stripped.ends_with(".token_adapter.trainable_tokens_delta") {
        OverlayTensorKind::Delta
    } else if stripped.ends_with(".lora_embedding_A") {
        OverlayTensorKind::LoraEmbedA
    } else if stripped.ends_with(".lora_embedding_B") {
        OverlayTensorKind::LoraEmbedB
    } else if is_full_module_weight(stripped, module) {
        OverlayTensorKind::FullSave
    } else {
        return None;
    };
    Some(OverlayTensor { module, kind })
}

/// Identify the tied-embedding module a (prefix-stripped) key targets. `lm_head`
/// wins over `embed_tokens`/`shared` because the two never co-occur in one key.
fn overlay_module_of(stripped: &str) -> Option<OverlayModule> {
    if stripped.contains("lm_head") {
        Some(OverlayModule::LmHead)
    } else if stripped.contains("embed_tokens") || stripped.contains(".shared.") {
        Some(OverlayModule::EmbedTokens)
    } else {
        None
    }
}

/// True when `stripped` is exactly the module's own top-level weight
/// (`…embed_tokens.weight` / `lm_head.weight`) — a `modules_to_save` full
/// replacement — and NOT a per-layer weight that merely mentions the module.
fn is_full_module_weight(stripped: &str, module: OverlayModule) -> bool {
    // A modules_to_save tensor sits above the layer stack.
    if stripped.contains(".layers.") {
        return false;
    }
    let leaf_owner = match module {
        OverlayModule::EmbedTokens => "embed_tokens.weight",
        OverlayModule::LmHead => "lm_head.weight",
    };
    stripped.ends_with(leaf_owner)
}

/// Tensor names the loader has collected for one adapter, partitioned by
/// (module, role). `Option<String>` = the safetensors key, filled at most once.
#[derive(Debug, Default, Clone)]
pub struct OverlayTensors {
    pub embed_base: Option<String>,
    pub embed_delta: Option<String>,
    pub embed_full: Option<String>,
    pub lmhead_base: Option<String>,
    pub lmhead_delta: Option<String>,
    pub lmhead_full: Option<String>,
    /// Any classic `lora_embedding_A/B` tensor seen (Tier-2 — load-rejected).
    pub lora_embedding_seen: bool,
}

impl OverlayTensors {
    /// Record one classified overlay tensor. Duplicate roles are a hard error
    /// (never silently overwrite — an ambiguous adapter must fail loudly).
    pub fn insert(&mut self, t: OverlayTensor, name: &str) -> Result<()> {
        use OverlayModule::*;
        use OverlayTensorKind::*;
        let slot = match (t.module, t.kind) {
            (_, LoraEmbedA) | (_, LoraEmbedB) => {
                self.lora_embedding_seen = true;
                return Ok(());
            }
            (EmbedTokens, Base) => &mut self.embed_base,
            (EmbedTokens, Delta) => &mut self.embed_delta,
            (EmbedTokens, FullSave) => &mut self.embed_full,
            (LmHead, Base) => &mut self.lmhead_base,
            (LmHead, Delta) => &mut self.lmhead_delta,
            (LmHead, FullSave) => &mut self.lmhead_full,
        };
        if slot.is_some() {
            bail!(
                "REJECT[duplicate-overlay-tensor]: two tensors map to {:?}/{:?}",
                t.module,
                t.kind
            );
        }
        *slot = Some(name.to_string());
        Ok(())
    }

    /// Any overlay tensor at all was collected.
    pub fn is_empty(&self) -> bool {
        self.embed_base.is_none()
            && self.embed_delta.is_none()
            && self.embed_full.is_none()
            && self.lmhead_base.is_none()
            && self.lmhead_delta.is_none()
            && self.lmhead_full.is_none()
            && !self.lora_embedding_seen
    }
}

/// Feature 2 load gate, called from the loader once overlay tensors are
/// collected. The device-side overlay apply is now WIRED (Stage-1
/// [`super::overlay_build::stage_overlay_raw`] upload → Stage-2
/// [`super::overlay_build::build_overlay`] row-diff/compact → the
/// `embed_tokens` / `lm_head` forward hooks in `crate::model::token_overlay`),
/// so `trainable_tokens` / `modules_to_save` `{embed_tokens, lm_head}` tensors
/// are LOADED rather than rejected.
///
/// The ONLY remaining reject here is the classic low-rank embedding LoRA
/// (`lora_embedding_A/B`): a distinct Tier-2 mechanism with no kernel yet, named
/// so the adapter fails loudly rather than being silently mis-applied.
pub fn reject_pending_overlay(overlay: &OverlayTensors) -> Result<()> {
    if overlay.lora_embedding_seen {
        bail!(
            "REJECT[lora-embedding-unimplemented]: classic low-rank embedding LoRA \
             (lora_embedding_A/B) is not yet supported on the decoder path"
        );
    }
    Ok(())
}

/// Clamp `trainable` ids to the served vocab, preserving list order (the delta
/// tensor's rows align positionally to it).
///
/// Returns `(kept_ids, skipped_extension_count)`:
/// - `idx >= r` → hard error (id outside the adapter's own `[R,h]` embedding).
/// - `vocab <= idx < r` → vocab-extension token the served tokenizer can't emit;
///   dropped and counted (caller warns).
/// - `idx < vocab` → kept, but a kept id smaller than a previously-kept id is a
///   hard error: PEFT appends extension tokens as the largest indices with delta
///   rows in the same order, so the kept prefix must stay positionally aligned to
///   the delta rows after the extension tail is dropped.
pub fn clamp_trainable_to_vocab(
    trainable: &[u32],
    r: usize,
    vocab: usize,
) -> Result<(Vec<u32>, usize)> {
    let mut kept = Vec::new();
    let mut skipped = 0usize;
    let mut last_kept: Option<u32> = None;
    for &idx in trainable {
        let i = idx as usize;
        if i >= r {
            bail!("REJECT[trainable-index-out-of-adapter]: id {idx} >= adapter embedding rows {r}");
        }
        if i >= vocab {
            skipped += 1;
            continue;
        }
        if let Some(prev) = last_kept
            && idx < prev
        {
            bail!(
                "REJECT[trainable-order]: kept id {idx} follows larger kept id {prev}; \
                 PEFT trainable-token order must be ascending in the served-vocab prefix"
            );
        }
        last_kept = Some(idx);
        kept.push(idx);
    }
    Ok((kept, skipped))
}

/// Union of (rows that differ from the served base) and (trainable ids), sorted
/// ascending and deduped. This is the final set of vocab rows the overlay
/// replaces. `row_diff[i]` = row `i` of the adapter base differs from served.
pub fn build_override_set(row_diff: &[bool], trainable: &[u32]) -> Vec<u32> {
    let mut ids: Vec<u32> = row_diff
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| d.then_some(i as u32))
        .collect();
    ids.extend_from_slice(trainable);
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Where an overridden id's replacement row comes from: `Some(k)` = trainable
/// delta row `k` (delta WINS when an id is both trainable and baked-different);
/// `None` = the adapter's baked `base_layer[id]`.
pub fn override_source(id: u32, trainable: &[u32]) -> Option<usize> {
    trainable.iter().position(|&t| t == id)
}

#[cfg(test)]
#[path = "overlay_tests.rs"]
mod tests;
