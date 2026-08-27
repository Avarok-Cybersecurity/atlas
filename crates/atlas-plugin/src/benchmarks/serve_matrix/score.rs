// SPDX-License-Identifier: AGPL-3.0-only

//! The verdict — a port of `tests/gate_results.py`.
//!
//! Pure over the round outcomes: no endpoint, no clock, no filesystem, so the
//! bars can be tested without a GPU (SBIO). Every bar is an explicit constant
//! with its rationale beside it (PCND) — there is no "looks close enough", and
//! no implicit "coverage was probably fine".

use super::plan::Plan;

/// Coherence probes that must pass. The Python bar was 2-of-3 to tolerate its
/// one temperature>0 creative probe occasionally missing; every probe here is
/// greedy, so with fewer probes than this the bar becomes "all of them" rather
/// than vacuous.
pub const COHERENCE_MIN_PASS: usize = 2;

/// Fraction below a blessed tok/s that counts as a regression.
pub const TPS_TOLERANCE: f64 = 0.10;

/// One probe's answer.
///
/// `NotApplicable` exists for the tool-call probe and is not a softer `Fail`:
/// a model whose parser this build has no support for scores every tool test
/// N/A, and failing the gate on that would block the matrix on a known gap. A
/// model whose parser IS wired up and still never produced a call is a `Fail`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Signal {
    Pass,
    Fail(String),
    NotApplicable(String),
    /// The probe was not run (turned off, or the round ended first). The
    /// default, so a half-built `Signals` can never read as a pass.
    #[default]
    NotRun,
}

impl Signal {
    pub fn is_fail(&self) -> bool {
        matches!(self, Signal::Fail(_))
    }

    /// Short cell text.
    pub fn text(&self) -> &str {
        match self {
            Signal::Pass => "PASS",
            Signal::Fail(_) => "FAIL",
            Signal::NotApplicable(_) => "N/A",
            Signal::NotRun => "—",
        }
    }

    /// The detail behind a non-pass, for the log line.
    pub fn detail(&self) -> Option<&str> {
        match self {
            Signal::Fail(d) | Signal::NotApplicable(d) => Some(d),
            _ => None,
        }
    }
}

/// What one booted round measured.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Signals {
    /// Is the endpoint serving the checkpoint this round loaded? See
    /// [`super::probes::Coherence::identity`] — a failed swap that restored the
    /// previous model still answers every question correctly.
    pub identity: Signal,
    pub coherence_pass: usize,
    pub coherence_total: usize,
    pub codegen: Signal,
    pub tool_call: Signal,
    /// Reported, deliberately NOT gated — `gate_results.py` never scored the
    /// long-context leg, and quietly adding a bar would make this run's PASS
    /// incomparable with every recorded one.
    pub long_ctx: Signal,
    /// Decode tokens/sec, `None` when the endpoint delivered the reply in one
    /// SSE delta and there is no inter-token interval to time.
    pub tps: Option<f64>,
}

/// What became of a planned round.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// Came up and was probed.
    Probed(Box<Signals>),
    /// Planned, attempted, never answered. **Never a skip.**
    BootFailed(String),
    /// Planned but the run ended before reaching it.
    NotReached,
}

/// One row of the matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundResult {
    pub label: String,
    pub outcome: Outcome,
    /// Blessed tok/s for this label, if one was recorded on this box.
    pub baseline_tps: Option<f64>,
}

impl RoundResult {
    pub fn signals(&self) -> Option<&Signals> {
        match &self.outcome {
            Outcome::Probed(s) => Some(s),
            _ => None,
        }
    }

    /// The bars this round FAILED. Empty means verified.
    pub fn bars(&self) -> Vec<String> {
        let signals = match &self.outcome {
            // The two shapes of "no result". Worded apart because they need
            // different fixes, and because "did not boot" is the sentence the
            // Python gate exists to be able to print.
            Outcome::BootFailed(why) => return vec![format!("did-not-boot ({why})")],
            Outcome::NotReached => return vec!["no-result".into()],
            Outcome::Probed(s) => s,
        };
        let mut fails = Vec::new();
        // Every bar below is FAIL-CLOSED: a probe that did not run is a
        // failure, not a pass. `Signals::default()` is all-`NotRun`, and a
        // default that scored clean would mean any future conditional probe —
        // or any early return between booting and probing — silently produced
        // a verified round.
        for (name, signal, allows_not_applicable) in [
            ("wrong-model", &signals.identity, false),
            ("codegen", &signals.codegen, false),
            ("tool_call", &signals.tool_call, true),
        ] {
            match signal {
                Signal::Pass => {}
                Signal::NotApplicable(_) if allows_not_applicable => {}
                Signal::NotApplicable(_) => fails.push(format!("{name}(not-applicable)")),
                Signal::Fail(_) => fails.push(name.to_string()),
                Signal::NotRun => fails.push(format!("{name}(not-probed)")),
            }
        }
        let coherence_bar = COHERENCE_MIN_PASS.min(signals.coherence_total);
        if signals.coherence_total == 0
            || signals.coherence_pass < coherence_bar
            || signals.coherence_pass > signals.coherence_total
        {
            fails.push(format!(
                "coherence({}/{})",
                signals.coherence_pass, signals.coherence_total
            ));
        }
        if let Some(tps) = signals.tps {
            if !tps.is_finite() {
                fails.push("tps(non-finite)".into());
            } else if tps <= 0.0 {
                fails.push("tps(0)".into());
            } else if let Some(base) = self
                .baseline_tps
                .filter(|baseline| baseline.is_finite() && *baseline > 0.0)
            {
                let floor = base * (1.0 - TPS_TOLERANCE);
                if tps < floor {
                    fails.push(format!("tps({tps:.1}<{floor:.1})"));
                }
            }
        }
        fails
    }

    /// The honest one-liner about the throughput bar.
    ///
    /// `tests/baselines/` is empty repo-wide, so the Python gate's tps check
    /// has never once compared against anything — every green it produced was
    /// liveness only. Saying "no baseline" is the difference between reporting
    /// that and implying a check passed.
    pub fn tps_note(&self) -> Option<&'static str> {
        let tps = self.signals()?.tps?;
        if tps.is_finite()
            && tps > 0.0
            && self
                .baseline_tps
                .filter(|baseline| baseline.is_finite() && *baseline > 0.0)
                .is_none()
        {
            return Some("no baseline — liveness only");
        }
        None
    }
}

/// The whole matrix, scored.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tally {
    pub verified: usize,
    pub planned: usize,
    pub skipped: usize,
    pub excluded: usize,
    /// `(label, bars)` for every round below bar.
    pub failures: Vec<(String, Vec<String>)>,
}

impl Tally {
    pub fn passed(&self) -> bool {
        self.planned > 0 && self.failures.is_empty()
    }
}

/// Score every PLANNED round against its result.
///
/// Iterates the PLAN, not the results — that inversion is the coverage
/// guarantee. A planned round with no result is scored as a failure, so a
/// checkpoint that crashed at boot cannot vanish from the denominator.
pub fn tally(plan: &Plan, results: &[RoundResult]) -> Tally {
    let mut out = Tally {
        planned: plan.planned_count(),
        skipped: plan.skipped().count(),
        excluded: plan.excluded_count(),
        ..Tally::default()
    };
    for round in plan.planned() {
        let label = round.label();
        let bars = match results.iter().find(|r| r.label == label) {
            Some(r) => r.bars(),
            None => vec!["no-result".into()],
        };
        if bars.is_empty() {
            out.verified += 1;
        } else {
            out.failures.push((label, bars));
        }
    }
    out
}

/// The sentence under the verdict. Names the measured state against the bar —
/// a bare FAIL sends the reader back to the raw log.
pub fn verdict_text(tally: &Tally, plan: &Plan) -> String {
    let coverage = format!(
        "{}/{} planned checkpoints verified",
        tally.verified, tally.planned
    );
    let mut extra = Vec::new();
    if tally.skipped > 0 {
        let names: Vec<String> = plan
            .skipped()
            .take(3)
            .map(|(r, why)| format!("{} ({})", r.model, why.reason()))
            .collect();
        extra.push(format!(
            "{} not runnable on this box: {}{}",
            tally.skipped,
            names.join(", "),
            if tally.skipped > 3 { ", …" } else { "" }
        ));
    }
    if tally.excluded > 0 {
        extra.push(format!("{} outside the filter", tally.excluded));
    }
    let tail = if extra.is_empty() {
        String::new()
    } else {
        format!(" · {}", extra.join(" · "))
    };
    if tally.failures.is_empty() {
        return format!("{coverage}{tail}");
    }
    let detail: Vec<String> = tally
        .failures
        .iter()
        .map(|(label, bars)| format!("{label}: {}", bars.join(", ")))
        .collect();
    format!(
        "{coverage}{tail} — {} below bar: {}",
        tally.failures.len(),
        detail.join(" · ")
    )
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod tests;
