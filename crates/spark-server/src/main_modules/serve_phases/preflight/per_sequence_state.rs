// SPDX-License-Identifier: AGPL-3.0-only

//! The per-sequence owned-state reserve term (C3).
//!
//! A sibling of `preflight.rs` (which sits at the 500-line cap), following the
//! `ssm_h_fp16.rs` precedent: the term and its explanation live here, and `preflight.rs`
//! gains only the call and the addend.
//!
//! # Why this term exists
//!
//! `inference_reserve` covered the SSM pools, the h-stage, the replay ring, the snapshot
//! region, the GDN two-phase buffer and a flat `cuda_headroom` — and nothing at all for
//! device memory a SEQUENCE owns. At `--max-batch-size 1` that hid inside `cuda_headroom`.
//! Stage 0 made concurrency legal and a C=3 arm served byte-identically, so the engine now
//! admits three sequences whose owned state nothing budgets.
//!
//! # Two owners, two addends
//!
//! The target stack's DSA indexer caches are owned by `TransformerLayer`s. The drafter's
//! per-sequence state is owned by a `DraftProposer`, which is never in the layer list — a
//! layer-side sum cannot see it. They are returned and logged separately so a boot log says
//! which owner a byte belongs to.

use atlas_core::config::ModelConfig;
use spark_model::seq_state_reserve::per_sequence_state_bytes;

use crate::cli;

/// Per-sequence owned device state, per rank, already multiplied by `--max-batch-size`.
///
/// Logs the split by owner and returns the total charge for `inference_reserve`.
///
/// 🔴 Multiplication by `max_batch_size` happens HERE and nowhere else, so a caller cannot
/// apply it twice. 🔴 The indexer is replicated across ranks — EP does not halve it — so
/// nothing here divides by a world size.
pub(super) fn per_sequence_reserve(args: &cli::ServeArgs, config: &ModelConfig) -> usize {
    let spec_on = args.speculative || args.self_speculative || args.dflash;
    let per_seq = per_sequence_state_bytes(config, args.max_seq_len, spec_on).unwrap_or_default();
    let charge = per_seq.for_batch(args.max_batch_size);
    if charge > 0 {
        tracing::info!(
            "Per-sequence state reserve: {} MB = {} seq x ({} MB target DSA layers + {} MB \
             proposer). Owned per sequence, replicated per rank (EP does not shard the \
             indexer); previously covered only by cuda_headroom.",
            charge / (1024 * 1024),
            args.max_batch_size.max(1),
            per_seq.target_layers / (1024 * 1024),
            per_seq.proposer / (1024 * 1024),
        );
    }
    charge
}

#[cfg(test)]
#[path = "per_sequence_state_tests.rs"]
mod tests;
