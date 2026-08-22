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
//! so any code change already invalidates all ten gates." It does not.
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
//! the ten-element `REQUIRED_GATES` constant unconditionally, and
//! `union() ⊊ REQUIRED_GATES` for most real PRs. Swapping the constant for the
//! union would *reduce* coverage — an unclassified docs PR would go from ten
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

// ── Where the intent half came from ────────────────────────────────────────

/// An abstention is not an empty answer, and the two must never render alike.
///
/// `required_for(changed, &[], roots)` and `required_for(changed, cats, &[])`
/// both yield an empty intent half. One means "nobody classified this PR",
/// which is honest; the other means "the taxonomy would not parse", which is a
/// repo defect. Collapsing them is how a loud failure becomes "implies
/// nothing" — the same collapse the jq walk made in CI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntentSource {
    /// No `--pr` supplied — a local run or a push build. Not evaluated.
    NotRequested,
    /// The ledger holds no countable classification for this PR. This is the
    /// steady state until the harvester runs, and it is not an error.
    NotRecorded { ledger: std::path::PathBuf },
    /// The ledger or the taxonomy could not be read. NEVER silently mapped to
    /// an empty set.
    Degraded { reason: String },
    Recorded {
        /// Every descended path recorded for this PR, deduplicated.
        categories: Vec<Vec<String>>,
        /// `error`/`abstain` rows. Counted, never treated as intent: an
        /// endpoint outage must not read as a confident classification, and
        /// the day the fallback root gains `_benches` it would otherwise
        /// manufacture GPU spend out of a 429.
        skipped: usize,
    },
}

/// Both halves plus the provenance of the second.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequiredReport {
    pub set: RequiredSet,
    pub source: IntentSource,
}

/// Read the classifications recorded for `pr`.
///
/// ★ **Every Category row in the file counts, with NO `head_sha` filter.** The
/// module docs above say "recorded for this head sha"; as an implementation
/// instruction that is wrong and would zero the intent half forever. The line
/// recording head X is written by CI and committed as a LATER commit, so it
/// cannot be present in the tree AT head X — exactly the argument `check.rs`
/// makes for why a gate record can never be written at its own commit. A
/// head-filtered read returns the empty set every time, and would look like a
/// working feature that simply never fires.
///
/// Unioning across a PR's older heads is safe by the same monotonicity that
/// makes unioning across re-runs safe: it can only add.
pub fn intent_source(root: &std::path::Path, pr: Option<u64>) -> IntentSource {
    let Some(pr) = pr else {
        return IntentSource::NotRequested;
    };
    let ledger = atlas_governance::ledger::path_for(root, pr);
    if !ledger.exists() {
        return IntentSource::NotRecorded { ledger };
    }
    let journey = match atlas_governance::ledger::read_all(&ledger) {
        Ok(j) => j.deduplicated(),
        // `read_all` hard-errors on a malformed line, which is right for an
        // auditor. For an advisory consumer, one corrupt byte must not fail the
        // job — but it must not read as "no intent" either.
        Err(e) => {
            return IntentSource::Degraded {
                reason: format!("{}: {e:#}", ledger.display()),
            };
        }
    };

    let (mut categories, mut skipped) = (Vec::new(), 0usize);
    for event in &journey.events {
        let atlas_governance::event::EventKind::Category { value, status } = &event.kind else {
            continue;
        };
        // `ok` and `partial` are opinions; `partial`'s matched prefix carries
        // real ancestor `_benches` by the union rule. `abstain`/`error` are not.
        if status != "ok" && status != "partial" {
            skipped += 1;
            continue;
        }
        let segments = parse_category(value);
        if !segments.is_empty() && !categories.contains(&segments) {
            categories.push(segments);
        }
    }
    if categories.is_empty() {
        return IntentSource::NotRecorded { ledger };
    }
    IntentSource::Recorded {
        categories,
        skipped,
    }
}

/// Assemble the report. Pure: all I/O happened in [`intent_source`] and the
/// caller's taxonomy load, so every branch here is testable without a
/// filesystem.
pub fn report(changed: &[String], source: IntentSource, roots: &[Node]) -> RequiredReport {
    let categories: &[Vec<String>] = match &source {
        IntentSource::Recorded { categories, .. } => categories,
        _ => &[],
    };
    RequiredReport {
        set: required_for(changed, categories, roots),
        source,
    }
}
