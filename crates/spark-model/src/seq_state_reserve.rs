// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence device state the serve must RESERVE, computed before the model exists.
//!
//! # Why this is config-keyed and not a layer-trait sum
//!
//! `preflight_reserve(args, config, free_mem)` runs long before `build_model`, and the model
//! builder *consumes* the reserve it returns. There is no layer list, no proposer, and no
//! `GpuBackend` at that point, so a per-sequence term cannot be `Σ_layers layer.some_method()`.
//! Every sibling reserve term (`ssm_reserve::*`) is a free function over config-derived scalars
//! for exactly this reason; this module follows that precedent.
//!
//! # What is charged, and what deliberately is not
//!
//! Charged — device memory a sequence OWNS, that no other reserve term covers:
//!   * the target stack's DSA indexer caches (`Glm5NextDsaState`), one per text DSA layer;
//!   * the draft proposer's own per-sequence state (`Glm5NextMtpHead::alloc_state`).
//!
//! The proposer is a SEPARATE OWNER — it is not a `TransformerLayer` and never appears in the
//! layer list, so a layer-side sum cannot see it. It is returned separately and must not be
//! folded into the target-layer term.
//!
//! NOT charged, each because something else already accounts for it:
//!   * KDA recurrent + conv state — pool-owned, covered by `ssm_reserve::ssm_pool_reserve_bytes`
//!     (and `meta.rs` never calls `alloc_state` for a pool-backed mixer);
//!   * the paged KV pool — it *is* the KV budget this reserve is subtracted from; charging it
//!     here would be circular;
//!   * the buffer arena — sized by `max_batch_tokens`, not per sequence (`buffer_arena_bytes`);
//!   * SSM snapshot / replay-ring / h-stage — already reserve terms, and the snapshot term
//!     already scales by `max_batch_size`;
//!   * CUDA-graph exec memory — per SLOT, not per sequence, and LRU-bounded;
//!   * pad-row dummy states — transient per step, and GLM declines every batched path.
//!
//! 🔴 The indexer cache is REPLICATED across ranks. EP does not halve it, so this returns
//! per-rank bytes directly and callers must not divide again.

use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig};

use crate::layers::glm5next_dsa::state::{dsa_capacity, indexer_state_bytes};
use crate::layers::glm5next_skeleton::{Glm5NextTextSkeleton, Mixer};

/// Per-sequence, per-rank device state, split by OWNER.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PerSequenceState {
    /// Indexer caches owned by the target stack's DSA layers.
    pub target_layers: usize,
    /// State owned by the draft proposer (a `DraftProposer`, never a `TransformerLayer`).
    pub proposer: usize,
}

impl PerSequenceState {
    pub fn total(&self) -> usize {
        self.target_layers + self.proposer
    }

    /// What the reserve must hold for `max_batch_size` concurrent sequences.
    /// Multiplied here, once, so no caller can apply it twice.
    pub fn for_batch(&self, max_batch_size: usize) -> usize {
        self.total() * max_batch_size.max(1)
    }
}

/// Per-sequence owned device state for `config`, at a context of `max_seq_len` tokens.
///
/// Returns zeros for any model that owns no per-sequence device state outside the pools —
/// which is every model except GLM-5.3 today. `spec_on` gates the proposer term: with
/// speculation off no proposer is constructed and no proposer state is ever allocated.
pub fn per_sequence_state_bytes(
    config: &ModelConfig,
    max_seq_len: usize,
    spec_on: bool,
) -> Result<PerSequenceState> {
    if config.model_type != "glm5_next" {
        return Ok(PerSequenceState::default());
    }
    let capacity = dsa_capacity(max_seq_len, config.index_kpool);
    let per_layer = indexer_state_bytes(capacity, config.index_head_dim);

    // The skeleton is the SSOT for which layers carry a DSA mixer, and it builds from the
    // config alone — no weights, no checkpoint, no GPU. `state_budget().dsa_indexer_per_token`
    // is the same arithmetic per token; we multiply by the POOL-ROUNDED capacity rather than a
    // raw token count, which is what keeps the reserve equal to the allocation (`dsa_capacity`).
    let skeleton = Glm5NextTextSkeleton::from_config(config)?;
    let dsa_layers = skeleton
        .layers
        .iter()
        .filter(|l| l.mixer == Mixer::Dsa)
        .count();

    let target_layers = dsa_layers * per_layer;

    // The GLM MTP head allocates, per sequence: one DSA indexer block for its own drafter
    // layer, plus five small fixed buffers. `block_table` is a host `Vec<u32>`, not device
    // memory, and the drafter's private KV pool is a construction-time singleton — neither
    // belongs here. Mirrors `Glm5NextMtpHead::alloc_state`.
    let proposer = if spec_on && config.mtp_layer_types.contains(&LayerType::SparseAttention) {
        per_layer                              // drafter DSA indexer block, one layer
            + 2 * config.hidden_size * 2       // concat
            + config.hidden_size * 2           // x
            + config.vocab_size * 2            // logits
            + 4                                // arg
            + 16 // head_xchg
    } else {
        0
    };

    Ok(PerSequenceState {
        target_layers,
        proposer,
    })
}

#[cfg(test)]
#[path = "seq_state_reserve_tests.rs"]
mod tests;
