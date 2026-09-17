// SPDX-License-Identifier: AGPL-3.0-only

//! Whole-registry lookups over the compiled targets, split out of `lib.rs` at
//! the 500-LoC cap. Exact piecewise move — no logic changed.
//!
//! Distinct from [`super::resolve`], which picks ONE target for a given
//! `(model_type, hidden_size)` and needs a tie-break; these two just enumerate
//! or substring-match, and neither can fail.

use super::{ServePreset, TargetPtxSet, all_ptx_sets};

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

/// The serve preset called `name` (case-insensitive), with the kernel-target
/// directory name that declares it.
///
/// `None` when no compiled target declares such a preset — which is also the
/// answer for every HF id and path, so a caller can ask first and fall through
/// to ordinary checkpoint resolution. Uniqueness across targets is enforced by
/// `build.rs`, so the first hit is the only hit.
pub fn preset_named(name: &str) -> Option<(&'static str, &'static ServePreset)> {
    all_ptx_sets().into_iter().find_map(|t| {
        t.serve_presets
            .iter()
            .find(|p| p.name.eq_ignore_ascii_case(name))
            .map(|p| (t.target.model, p))
    })
}
