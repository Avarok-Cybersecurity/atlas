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
//! # ★ Where the union actually bites, corrected
//!
//! An earlier version of this comment claimed the union was very nearly a
//! no-op, for two reasons. The first stands: `pr_taxonomy::validate` rejects
//! any `_benches` id outside [`super::coverage::REQUIRED`], so
//! `intent ⊆ REQUIRED` for any tree.
//!
//! **The second was wrong.** It read: "`PERF_PATHS` contains a bare `crates`,
//! so any code change already invalidates all five gates." It does not.
//! `GATE_MACHINERY` excludes the whole `crates/atlas-plugin/src/gate` prefix
//! from **every** gate, and each benchmark driver is excluded from the other
//! gates — so plenty of `crates/` paths invalidate nothing at all and intent is
//! their only source of coverage. The union is live inside `crates/` today; it
//! is not waiting on the closure-hash work.
//!
//! It also cited `recipes/` as the live case. **This repo tracks no `recipes/`
//! files** — they live in the separate `atlas-recipes` repo, and
//! `invalidating_paths` diffs *this* one, so that path can never appear in a
//! diff here. The reachable classes are `docker/`, `docs/`, `.github/`,
//! `scripts/`, `bench/`, `kernels/**/BENCH.toml`, and the excluded `crates/`
//! paths above. `intent_adds_where_the_paths_are_silent` and
//! `crates_paths_split_into_fully_covered_and_not_covered_at_all` pin those.
//!
//! ★★ **The union is NOT the loop set.** [`super::check::check_gates`] iterates
//! the five-element `REQUIRED_GATES` constant unconditionally, and
//! `union() ⊊ REQUIRED_GATES` for most real PRs. Swapping the constant for the
//! union would *reduce* coverage — an unclassified docs PR would go from five
//! gates checked to none. The add-only property holds against `by_path`; it
//! says nothing about the constant. Whatever consumes this must keep the
//! constant as the loop set and use the union to ESCALATE — to widen what
//! invalidates a standing record — never to select what gets checked.
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
pub fn required_for(changed: &[String], categories: &[Vec<String>], roots: &[Node]) -> RequiredSet {
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
