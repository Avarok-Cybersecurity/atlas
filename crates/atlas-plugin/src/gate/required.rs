// SPDX-License-Identifier: AGPL-3.0-only

//! What a PR owes: `path_derived ∪ intent_derived`.
//!
//! [`super::pr_taxonomy`] documented this union as the thing that makes a
//! language model safe near a merge gate, and then nothing computed it. The
//! intent half had **zero non-test callers**; the only consumer was a jq
//! reimplementation of `benches_for` in `ci.yml`. That is the same shape as the
//! bug already found inside `pr_taxonomy`: two implementations of one function,
//! the Rust half failing in the removing direction.
//!
//! # ★ The union is very nearly a no-op today, and pretending otherwise helps
//! nobody
//!
//! Two independent reasons, both currently true:
//!
//! 1. `pr_taxonomy::validate` rejects any `_benches` id that is not in
//!    [`super::coverage::REQUIRED`], so `intent ⊆ REQUIRED` for *any* tree.
//! 2. [`super::coverage::PERF_PATHS`] contains a bare `"crates"`, so any code
//!    change already invalidates all five gates.
//!
//! So for a code PR the union adds nothing, and `benches_may_only_add` — the
//! property the design rests on — is true and *vacuous*. It is insurance for a
//! world that does not exist yet.
//!
//! The one class where it bites today is real and measured: **`recipes/` and
//! `docker/` invalidate nothing**. A recipe change that alters serve flags
//! genuinely moves decode wall, `by_path` is empty, and `performance/decode`
//! adds the legs that would otherwise never run. `the_live_case_is_recipes`
//! pins exactly that, and `intent_is_redundant_for_a_crates_change` pins the
//! vacuity so it cannot quietly stop being true.
//!
//! # Why a UNION over classifications, not the newest one
//!
//! The classifier is not stable. Three live runs on one PR produced `tooling`,
//! `performance`, `tooling`. A gate that changes its mind between re-runs is
//! worse than no gate — so every category ever recorded for a head sha counts,
//! and the ledger being grow-only and deduplicated-on-read makes that cheap.
//! Unioning is monotone, replay-stable, and fails in the adding direction,
//! which is the same footing as everything else here.

use std::collections::BTreeSet;

use super::pr_taxonomy::{Node, benches_for};

/// Both halves, kept apart on purpose.
///
/// The telemetry table has to be able to say *why* a gate is required. Collapse
/// this to one set and "intent added this one" becomes invisible — which is how
/// the last coverage gap survived as long as it did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RequiredSet {
    /// Gates the changed paths invalidate. Stands entirely on its own; nothing
    /// in the intent half can shrink it.
    pub by_path: BTreeSet<String>,
    /// Gates the classified intent implies, unioned over every classification
    /// recorded for this head.
    pub by_intent: BTreeSet<String>,
}

impl RequiredSet {
    /// Everything the PR owes.
    pub fn union(&self) -> BTreeSet<String> {
        self.by_path.union(&self.by_intent).cloned().collect()
    }

    /// What intent added that the paths did not already require — the only part
    /// worth a line in the telemetry table, and empty in the vacuous case.
    pub fn intent_only(&self) -> BTreeSet<String> {
        self.by_intent.difference(&self.by_path).cloned().collect()
    }
}

/// Compute both halves.
///
/// `categories` is every descended path recorded for this head sha, not the
/// newest — see the module docs. An empty slice is the honest representation of
/// "not classified", and yields an empty intent half rather than a guess.
pub fn required_for(
    changed: &[String],
    categories: &[Vec<String>],
    roots: &[Node],
) -> RequiredSet {
    let by_path = super::coverage::invalidated_by(changed.iter().map(String::as_str))
        .into_iter()
        .map(str::to_string)
        .collect();
    let mut by_intent = BTreeSet::new();
    for category in categories {
        by_intent.extend(benches_for(roots, category));
    }
    RequiredSet { by_path, by_intent }
}

/// Parse `performance/decode` into `["performance", "decode"]`.
///
/// Empty segments are dropped rather than descended into: a trailing slash or a
/// `//` is a formatting slip, and `benches_for` would simply stop at the empty
/// segment, silently truncating the path and *removing* benches.
pub fn parse_category(value: &str) -> Vec<String> {
    value
        .split('/')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
#[path = "required_tests.rs"]
mod required_tests;
