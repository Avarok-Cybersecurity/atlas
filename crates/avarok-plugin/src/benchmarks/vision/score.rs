// SPDX-License-Identifier: AGPL-3.0-only

//! Turning outcomes into a verdict.
//!
//! Three states, not two. A run where every capability probe passed AND the
//! no-image control also "passed" is not a pass — it is **VACUOUS**, and
//! reporting it as green is the specific failure this benchmark exists to
//! avoid. "What colour is this image?" has a confident answer available from
//! language priors alone, so a server that silently stopped splicing vision
//! embeddings would sail through a naive suite.

use std::fmt;

/// How one geometry cell came out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeomCell {
    /// Reported vision tokens matched the model exactly.
    Match {
        fixture: &'static str,
        tokens: usize,
    },
    /// Reported a different count. The engine's preprocessing changed.
    Mismatch {
        fixture: &'static str,
        want: usize,
        got: usize,
    },
    /// Could not be asserted — e.g. the image exceeds the server's encoder
    /// capacity, so a failure here says nothing about correctness.
    Unmeasured { fixture: &'static str, why: String },
    /// The request itself failed.
    Error { fixture: &'static str, msg: String },
}

/// How one capability probe came out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeCell {
    Pass { id: &'static str },
    Fail { id: &'static str, reply: String },
    Error { id: &'static str, msg: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    /// The capability probes cannot be trusted: the control answered as though
    /// it had seen an image it was never sent.
    Vacuous,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Vacuous => "VACUOUS",
        })
    }
}

/// Does a reply satisfy a probe's expectations?
///
/// Case-insensitive substring matching. Deliberately crude: the probes are
/// written so a correct answer cannot avoid the words, which is a property of
/// the probe rather than of the matcher.
pub fn reply_matches(reply: &str, want_all: &[&str], want_none: &[&str]) -> bool {
    let hay = reply.to_lowercase();
    want_all
        .iter()
        .all(|term| crate::benchmarks::first_standalone_term(&hay, term).is_some())
        && !want_none
            .iter()
            .any(|term| crate::benchmarks::first_standalone_term(&hay, term).is_some())
}

/// Fold the legs into one verdict.
///
/// The control is evaluated FIRST and can veto everything: if it did not hold,
/// the capability results carry no information whatever they say. Geometry is
/// independent of that — it measures token counts, which language priors
/// cannot fake — so a mismatch there fails the run on its own.
pub fn verdict(geom: &[GeomCell], probes: &[ProbeCell], control_held: bool) -> Verdict {
    let geom_bad = geom
        .iter()
        .any(|c| matches!(c, GeomCell::Mismatch { .. }) || matches!(c, GeomCell::Error { .. }));
    let probes_bad = probes.iter().any(|c| !matches!(c, ProbeCell::Pass { .. }));

    if !control_held {
        // Vacuous even when everything else is green — ESPECIALLY then. An
        // all-green run whose control also passed is the exact shape of a
        // server answering from priors.
        return Verdict::Vacuous;
    }
    if geom_bad || probes_bad {
        return Verdict::Fail;
    }
    Verdict::Pass
}

/// Fold the runtime integrity legs into the geometry/capability verdict.
/// A broken control remains `Vacuous`; the extra legs cannot rehabilitate it.
pub fn with_runtime_checks(
    base: Verdict,
    integrity_failed: bool,
    concurrency_clean: bool,
) -> Verdict {
    match base {
        Verdict::Pass if integrity_failed || !concurrency_clean => Verdict::Fail,
        other => other,
    }
}

/// How many geometry cells actually asserted something.
///
/// Reported alongside the verdict because a run where every cell came back
/// `Unmeasured` is green and worthless; the count is what tells a reader
/// which it was.
pub fn asserted_cells(geom: &[GeomCell]) -> usize {
    geom.iter()
        .filter(|c| matches!(c, GeomCell::Match { .. } | GeomCell::Mismatch { .. }))
        .count()
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod score_tests;
