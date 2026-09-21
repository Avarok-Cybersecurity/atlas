// SPDX-License-Identifier: AGPL-3.0-only

//! Whole-registry lookups over the compiled targets, split out of `lib.rs` at
//! the 500-LoC cap. Exact piecewise move — no logic changed.
//!
//! Distinct from [`super::resolve`], which picks ONE target for a given
//! `(model_type, hidden_size)` and needs a tie-break; these two just enumerate
//! or substring-match, and neither can fail.

use super::{TargetPtxSet, all_ptx_sets};

/// All compiled kernel targets and their PTX module sets.
///
/// Returns one entry per target compiled at build time.
/// Single-target builds return one entry; wildcard builds return all.
pub fn available_targets() -> Vec<TargetPtxSet> {
    all_ptx_sets()
}

/// Find the PTX module set for a target whose model name contains `needle`.
///
/// Returns `None` if no compiled target matches.
pub fn ptx_for_model(needle: &str) -> Option<TargetPtxSet> {
    all_ptx_sets()
        .into_iter()
        .find(|t| t.target.model.contains(needle))
}
