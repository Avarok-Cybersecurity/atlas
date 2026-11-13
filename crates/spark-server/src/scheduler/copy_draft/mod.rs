// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0. Source: src/models/qwen3_5/program/speculative/prompt_lookup.{h,cpp} (see the child modules for line ranges). Changes: translated from C++ to Rust and wired into Atlas's MTP rounds, by Atlas contributors.
// Portions copied and adapted from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce, zig/src/core/copy_index.zig:1-80. Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written before TensorFold 0.6.0 is also available under the MIT License. See COPY-DRAFT-ATTRIBUTION.md. Changes for Atlas: see `index.rs` and `legacy.rs`.
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the merged copy drafter (port sprint lane L13): one
//! per-sequence index over committed tokens feeding three plans, plus the
//! scheduler entry points that apply them.
//!
//! Sources merged into one design (master plan 3.7.4): Cinference's prompt
//! lookup (hashed suffix windows, period continuation, the decayed per-window
//! estimate and its lookup rounds), TensorFold's copy index (the "latest
//! earlier occurrence" contract, chain fill of the MTP chain, the legacy
//! n-gram index) and the KVMem description (16-token windows, backward match
//! extension, the source cursor and the copy-versus-MTP yield gate).
//!
//! Plans (`scheduler::spec_round_plan::RoundPlan`):
//! * `CopyChain`: the MTP drafter's chain (K = 2, 3 or 4) is replaced by the
//!   copy continuation of the same width when the source agrees on at least
//!   `min_match` tokens and the yield gate says the copy pays. Serial rounds
//!   go through [`plan_serial`], batched rows through [`substitute_batched`]
//!   (one call line in `verify_k4_batch_step.rs`, lane L10).
//! * `LookupRound`: a single sequence whose estimate is confident (>= 0.96)
//!   verifies up to `max_len` copy tokens alone through the existing
//!   variable-width verify (`step_verify_dflash`), discarding the drafter's
//!   chain for that round.
//! * `Mtp`: today's round.
//!
//! Exactness: the target verifies every plan with the same pre-sample
//! pipeline, so outputs are the target's own; a copy only changes which
//! tokens are offered. Grammar-constrained sequences never take a copy plan,
//! and a request with `seq.port.no_drafts` or `seq.disable_mtp` takes no
//! speculative step at all.
//!
//! With `copy_draft = off` (every target's default) nothing here runs: the
//! per-sequence state stays an empty `Option`, [`plan_serial`] and
//! [`substitute_batched`] return at their first line, and the legacy
//! `--ngram-speculative` mode keeps today's scan byte for byte.

use super::*;

pub(crate) mod config;
pub(crate) mod estimator;
pub(crate) mod index;
pub(crate) mod legacy;
pub(crate) mod metrics;
pub(crate) mod policy;

#[cfg(test)]
mod estimator_tests;
#[cfg(test)]
mod index_tests;
#[cfg(test)]
mod legacy_tests;
#[cfg(test)]
mod tests;

pub(crate) use config::{active, apply_args, config, validate_args};
pub(crate) use metrics::render_prometheus;

use crate::scheduler::spec_round_plan::RoundPlan;
use policy::{CopyState, Plan, RowFacts};

/// The copy drafter's per-sequence state (`SeqPortState::copy`). Empty, and
/// never allocated, while the drafter is off.
#[derive(Default)]
pub struct SeqCopyState {
    state: Option<Box<CopyState>>,
}

impl SeqCopyState {
    fn get(&mut self) -> &mut CopyState {
        self.state.get_or_insert_with(Box::default)
    }

    /// True once the drafter has planned for this sequence.
    pub fn engaged(&self) -> bool {
        self.state.is_some()
    }
}

/// The request asked for no speculative step (lane L05's per-request
/// `atlas_speculative: false`). Both switches are set from it:
/// `seq.port.no_drafts` (lost on a swap-in until lane L27's hunk lands) and
/// the older `seq.disable_mtp` (kept across swap-out and swap-in), so a
/// sequence is out when either is set. No copy plan is ever chosen for it.
#[inline]
pub(in crate::scheduler) fn no_drafts(a: &ActiveSeq) -> bool {
    a.disable_mtp || a.port.no_drafts
}

fn facts<'a>(
    model: &dyn Model,
    a: &'a ActiveSeq,
    drafts: &'a [u32],
    lookup_ok: bool,
) -> RowFacts<'a> {
    RowFacts {
        head: &a.seq.tokens,
        last: a.last_token,
        no_drafts: no_drafts(a),
        grammar: a.grammar_state.is_some(),
        mtp_drafts: drafts,
        lookup_ok,
        capacity: model.mtp_slot_draft_capacity(a.seq.slot_idx),
        remaining: a.remaining,
    }
}

/// Plan a serial (per-sequence) round over the MTP chain `drafts`.
///
/// `n_active` is the scheduler's active count: lookup rounds are
/// single-sequence in this sprint. `raw_argmax` is the DFlash mode, whose
/// drafter context commits are ordered against its own chain, so it never
/// takes a copy plan. Returns `Mtp { drafts: drafts.len() }` when nothing
/// changes (always, with the drafter off).
pub(in crate::scheduler) fn plan_serial(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    n_active: usize,
    raw_argmax: bool,
) -> RoundPlan {
    let unchanged = RoundPlan::Mtp {
        drafts: drafts.len(),
    };
    if !active() || raw_argmax || drafts.is_empty() {
        return unchanged;
    }
    let lookup_ok = n_active == 1 && a.top_logprobs.is_none() && !model.is_ep();
    let cfg = config();
    let plan = {
        let mut cs = std::mem::take(&mut a.port.copy);
        let p = policy::plan(cs.get(), cfg, &facts(model, a, drafts, lookup_ok));
        a.port.copy = cs;
        p
    };
    match plan {
        Plan::NoSpec => RoundPlan::Mtp { drafts: 0 },
        Plan::Mtp => unchanged,
        Plan::Chain(tokens) => RoundPlan::CopyChain { tokens },
        Plan::Lookup(tokens) => RoundPlan::LookupRound { tokens },
    }
}

/// Batched rows (3.7.4): replace, in place, the chain of every row whose
/// copy plan wins, and return how many were replaced.
///
/// `drafts[i]` is row `i`'s freshly proposed chain; when `drafts` is empty
/// the rows' own `pending_drafts` are read and written instead. Only
/// `CopyChain` applies here (same width, so the batch shape and its graph
/// key are unchanged); a replaced row's `pending_draft_conf` is cleared, so
/// D-Cut reads it as "not measured" and keeps its full depth. Call it after
/// the confidences are assigned. A no-op with the drafter off.
pub(in crate::scheduler) fn substitute_batched(
    batch: &mut [&mut ActiveSeq],
    drafts: &mut [Vec<u32>],
) -> usize {
    if !active() {
        return 0;
    }
    let cfg = config();
    if !cfg.mode.chains() {
        return 0;
    }
    let mut replaced = 0usize;
    for (i, a) in batch.iter_mut().enumerate() {
        let a: &mut ActiveSeq = a;
        let current: Vec<u32> = match drafts.get(i) {
            Some(d) => d.clone(),
            None => a.pending_drafts.clone(),
        };
        if current.is_empty() {
            continue;
        }
        let mut cs = std::mem::take(&mut a.port.copy);
        let p = {
            let f = RowFacts {
                head: &a.seq.tokens,
                last: a.last_token,
                no_drafts: no_drafts(a),
                grammar: a.grammar_state.is_some(),
                mtp_drafts: &current,
                lookup_ok: false,
                capacity: usize::MAX,
                remaining: a.remaining,
            };
            policy::plan(cs.get(), cfg, &f)
        };
        a.port.copy = cs;
        if let Plan::Chain(tokens) = p {
            match drafts.get_mut(i) {
                Some(d) => *d = tokens,
                None => a.pending_drafts = tokens,
            }
            a.pending_draft_conf.clear();
            replaced += 1;
        }
    }
    replaced
}
