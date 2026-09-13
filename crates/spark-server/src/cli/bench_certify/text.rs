// SPDX-License-Identifier: AGPL-3.0-only

//! The human rendering of a campaign: plan, progress and summary lines.

use super::plan::{self, Unit};
use super::runner::RunOutcome;
use super::state;

pub(in crate::cli::bench_certify) fn human(secs: u64) -> String {
    if secs >= 3600 {
        format!("{:.1} h", secs as f64 / 3600.0)
    } else if secs >= 60 {
        format!("{} min", secs / 60)
    } else {
        format!("{secs} s")
    }
}

pub(in crate::cli::bench_certify) fn describe(o: &RunOutcome) -> String {
    match o {
        RunOutcome::Passed { record } => format!("PASS ({})", record.display()),
        RunOutcome::MemberDone { record } => format!("shard done ({})", record.display()),
        RunOutcome::VerdictFail { reason, .. } => format!("FAIL — {reason}"),
        RunOutcome::Harness { reason, .. } => format!("harness — {reason}"),
        RunOutcome::TimedOut => "timed out".into(),
        RunOutcome::Cancelled => "cancelled".into(),
    }
}

pub(super) fn print_plan(
    anchor: &str,
    hardware: &str,
    guard_ref: Option<&str>,
    gates: &[&str],
    units: &[Unit],
    serial: u64,
) {
    eprintln!(
        "certify: anchor {anchor} · hardware {hardware} · guard {}",
        guard_ref.unwrap_or("(none)")
    );
    if gates.is_empty() {
        eprintln!("certify: every required gate already passes at {anchor}");
        return;
    }
    eprintln!(
        "certify: {} gate(s) open → {} unit(s), ~{} serial:",
        gates.len(),
        units.len(),
        human(serial)
    );
    for u in units {
        eprintln!(
            "  {:<28} {:>8}  {:?}{}",
            u.id,
            human(u.secs()),
            u.class,
            match u.estimate {
                plan::Estimate::Declared(_) => "",
                plan::Estimate::Measured { .. } => "  (measured)",
            }
        );
    }
}

pub(super) fn print_summary(s: &state::Summary) {
    eprintln!();
    eprintln!(
        "certify: {} passed, {} shard(s) done, {} failed, {} skipped{}",
        s.passed.len(),
        s.member_done.len(),
        s.failed.len(),
        s.skipped.len(),
        s.aborted
            .as_deref()
            .map_or(String::new(), |a| format!(" — ABORTED: {a}"))
    );
    for (id, why) in &s.failed {
        eprintln!("  failed  {id}: {why}");
    }
}

#[derive(serde::Serialize)]
pub(super) struct SummaryJson {
    passed: Vec<&'static str>,
    member_done: Vec<&'static str>,
    failed: Vec<(&'static str, String)>,
    skipped: Vec<(&'static str, String)>,
    aborted: Option<String>,
}

impl From<&state::Summary> for SummaryJson {
    fn from(s: &state::Summary) -> Self {
        Self {
            passed: s.passed.clone(),
            member_done: s.member_done.clone(),
            failed: s.failed.clone(),
            skipped: s.skipped.clone(),
            aborted: s.aborted.clone(),
        }
    }
}
