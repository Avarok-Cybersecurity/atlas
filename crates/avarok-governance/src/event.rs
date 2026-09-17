// SPDX-License-Identifier: AGPL-3.0-only

//! What a journey is made of.

use serde::{Deserialize, Serialize};

/// The verdict a gate reached, when the event carries one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Pass,
    Fail,
    /// The gate was open — no record covered the commit. Distinct from `Fail`,
    /// which means a run happened and its numbers were out of bounds. Reporting
    /// them as one thing loses the difference between "we have not measured
    /// this" and "we measured it and it regressed".
    Missing,
}

/// What happened.
///
/// ★ The lifecycle vocabulary is not invented here. `CONTRIBUTING.md` already
/// defines the pull request as an eight-state machine — branch, open, edit,
/// checks, gates, cla, review, merge — each state naming its exit condition and
/// the command that proves it. [`EventKind::State`] records transitions through
/// *that* machine rather than a parallel one, so the ledger and the contributor
/// documentation cannot drift into describing different processes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventKind {
    /// A transition in the documented lifecycle.
    State { to: String },
    /// A gate was evaluated. `invalidated_by` names the paths that re-opened
    /// it, when it was open — the same list the CLI prints, kept so the reason
    /// survives after the console scrollback is gone.
    Gate {
        id: String,
        verdict: Verdict,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        invalidated_by: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// The advisory classifier's opinion.
    ///
    /// ★ Recorded, never acted upon. It is here so the observe-only rollout has
    /// something to audit — "did the category ever disagree with the floor, and
    /// in which direction?" is answerable only if the disagreements were
    /// written down. `status` distinguishes an answer from an abstention, since
    /// a run where the endpoint was down must not read as a confident verdict.
    Category { value: String, status: String },
    /// A benchmark run produced numbers.
    Measurement {
        benchmark: String,
        #[serde(default)]
        metrics: std::collections::BTreeMap<String, f64>,
    },
}

/// One line of a journey.
///
/// `(head_sha, run_id, attempt, kind)` is the identity used for deduplication.
/// `at` is deliberately excluded from it: the same logical event replayed by a
/// re-run must collapse rather than accumulate, and a timestamp would defeat
/// that. Keeping the timestamp as data rather than identity is what makes the
/// set grow-only in the useful sense.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub pr: u64,
    pub head_sha: String,
    /// The CI run that observed this. Empty for local invocations.
    #[serde(default)]
    pub run_id: String,
    /// Which attempt of that run. A re-run after a flake is a different
    /// attempt, not an overwrite — losing the first attempt would hide exactly
    /// the flakiness worth seeing.
    #[serde(default)]
    pub attempt: u32,
    /// Unix seconds. Data, not identity.
    pub at: u64,
    #[serde(flatten)]
    pub kind: EventKind,
}

impl Event {
    /// The deduplication key: everything except the timestamp.
    pub fn identity(&self) -> String {
        // Serialising the kind rather than hand-formatting each variant means a
        // new variant cannot silently collide with an existing one by
        // forgetting to include its own fields.
        let kind = serde_json::to_string(&self.kind).unwrap_or_default();
        format!(
            "{}|{}|{}|{}",
            self.head_sha, self.run_id, self.attempt, kind
        )
    }

    /// The node label this event contributes to the materialised graph.
    pub fn node_label(&self) -> &'static str {
        match self.kind {
            EventKind::State { .. } => "state",
            EventKind::Gate { .. } => "gate",
            EventKind::Category { .. } => "category",
            EventKind::Measurement { .. } => "measurement",
        }
    }
}
