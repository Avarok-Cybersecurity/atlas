// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the `/metrics` text of the swap timer (port sprint lane L41):
//! `atlas_model_swap_seconds{phase}`, `atlas_model_swap_total{outcome}`,
//! `atlas_model_swap_estimate_seconds{model}` and `atlas_model_resident_bytes`.
//! Every value is zero with the `model_swap_timing` row off.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use super::atlas::round_ms;
use super::{Outcome, Phase, STATS, Snapshot, SwapStats, enabled};
use crate::api::port_metrics as pm;

/// Escape a Prometheus label value.
pub fn escape_label(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    for c in v.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out
}

/// Append the L41 series to the `/metrics` text (one call line in
/// `port_metrics::render`).
pub fn render_prometheus(out: &mut String) {
    let live = u64::try_from(spark_runtime::run_metrics::ledger_bytes().0).unwrap_or(u64::MAX);
    render_with(out, &STATS, enabled(), live);
}

/// [`render_prometheus`] over explicit inputs (pure, so a test grades the
/// text). With `on == false` every value is zero and no model label appears.
pub fn render_with(out: &mut String, stats: &SwapStats, on: bool, resident_bytes: u64) {
    let snap = if on {
        stats.snapshot()
    } else {
        Snapshot::default()
    };
    let name = "atlas_model_swap_seconds";
    let _ = writeln!(
        out,
        "# HELP {name} Hot-swap time by phase (drain, teardown, load, total), seconds\n\
         # TYPE {name} summary"
    );
    for p in Phase::ALL {
        let (count, sum) = snap.phases[p as usize];
        let _ = writeln!(out, "{name}_sum{{phase=\"{}\"}} {sum:.6}", p.label());
        let _ = writeln!(out, "{name}_count{{phase=\"{}\"}} {count}", p.label());
    }
    let labels: Vec<String> = Outcome::ALL
        .iter()
        .map(|o| format!("outcome=\"{}\"", o.label()))
        .collect();
    let series: Vec<(&str, u64)> = labels
        .iter()
        .map(String::as_str)
        .zip(snap.outcomes)
        .collect();
    pm::labeled_counter(
        out,
        "atlas_model_swap_total",
        "Hot-swaps by outcome (ok, restored, failed, refused)",
        &series,
    );
    let estimates = if on {
        stats.estimates()
    } else {
        BTreeMap::new()
    };
    let labels: Vec<String> = estimates
        .keys()
        .map(|m| format!("model=\"{}\"", escape_label(m)))
        .collect();
    let series: Vec<(&str, f64)> = labels
        .iter()
        .map(String::as_str)
        .zip(estimates.values().map(|s| round_ms(*s)))
        .collect();
    pm::labeled(
        out,
        "atlas_model_swap_estimate_seconds",
        "Median of the last timed swaps to the model, seconds",
        &series,
    );
    pm::gauge(
        out,
        "atlas_model_resident_bytes",
        "Device bytes live in the allocation ledger for the loaded model",
        if on { resident_bytes } else { 0 },
    );
}
