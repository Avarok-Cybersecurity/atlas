// SPDX-License-Identifier: AGPL-3.0-only

//! Turning the box's roster into the rounds this run intends to cover.
//!
//! This is the module the coverage guarantee lives in, so it is worth being
//! precise about the three states a checkpoint can be in. They are NOT
//! interchangeable, and the whole gate rests on the third never reading as the
//! second:
//!
//! | state | meaning | verdict |
//! |---|---|---|
//! | **planned** | the box can serve it, the run will try | must produce a result |
//! | **skipped** | the box cannot serve it at all (no weights / no kernels) | excluded, and SAID |
//! | **failed to boot** | planned, tried, never came up | **FAIL** |
//!
//! `tests/run_all_models.py` writes a `_manifest.json` before any container
//! starts for exactly this reason: a model that crashes at boot writes no
//! result file, and a gate that globs result files scores the survivors and
//! reports green. Here the plan plays the manifest's part — it is built before
//! the first boot and it is what the scoring iterates, so an absent outcome is
//! a failure rather than an absence.

pub use super::host::Absence;
use super::host::ServeCandidate;

/// Why a candidate is not being run. `None` on a planned round.
pub type Skip = Option<Absence>;

/// One model×quant the run intends to boot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Round {
    pub model: String,
    pub quant: String,
    /// Set when the round is not planned; carries the reason.
    pub skipped: Skip,
    /// Filtered out by the operator's `include` pattern. Distinct from
    /// `skipped`, and DISJOINT from it: the box could serve this one, the run
    /// just was not asked to. A checkpoint that is both unservable and outside
    /// the filter is only ever counted as unservable — double-counting it
    /// makes "3 skipped · 3 filtered" out of two checkpoints.
    pub excluded: bool,
}

impl Round {
    pub fn is_planned(&self) -> bool {
        self.skipped.is_none() && !self.excluded
    }

    /// Label used in the table and in a failure line.
    pub fn label(&self) -> String {
        match self.quant.trim() {
            "" | "-" => self.model.clone(),
            q => format!("{} · {q}", self.model),
        }
    }
}

/// The whole roster, classified. Skipped and excluded rounds are KEPT — the
/// count of what was not run is part of the result, not something dropped on
/// the floor.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    pub rounds: Vec<Round>,
}

impl Plan {
    /// Classify `roster` under an operator filter.
    ///
    /// `include` is a case-insensitive substring of the model id; empty means
    /// everything the box can serve. Sorted by model id so two runs on the same
    /// box produce the same round order and are readable side by side.
    pub fn build(roster: &[ServeCandidate], include: &str) -> Self {
        let needle = include.trim().to_lowercase();
        let mut rounds: Vec<Round> = roster
            .iter()
            .map(|c| Round {
                model: c.model.clone(),
                quant: c.quant.clone(),
                skipped: c.absent,
                excluded: c.absent.is_none()
                    && !needle.is_empty()
                    && !c.model.to_lowercase().contains(&needle),
            })
            .collect();
        rounds.sort_by(|a, b| a.model.cmp(&b.model).then(a.quant.cmp(&b.quant)));
        Self { rounds }
    }

    /// The rounds that will actually be booted, in order.
    ///
    /// This iterator is the SSOT for the roster: the cursor walks it and the
    /// scoring walks it, so the two cannot disagree about what was planned.
    pub fn planned(&self) -> impl Iterator<Item = &Round> {
        self.rounds.iter().filter(|r| r.is_planned())
    }

    pub fn planned_count(&self) -> usize {
        self.planned().count()
    }

    /// Checkpoints the box cannot serve, each with its reason. Reported, never
    /// silently dropped: "12 of 20 verified" means nothing without them.
    ///
    /// Yields the reason alongside rather than leaving the caller to unwrap an
    /// `Option` this filter has already proved is `Some` — an `unwrap_or` at
    /// the call site is a fallback for a case that cannot happen, which reads
    /// as if it can.
    pub fn skipped(&self) -> impl Iterator<Item = (&Round, Absence)> {
        self.rounds
            .iter()
            .filter_map(|r| r.skipped.map(|why| (r, why)))
    }

    pub fn excluded_count(&self) -> usize {
        self.rounds.iter().filter(|r| r.excluded).count()
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
