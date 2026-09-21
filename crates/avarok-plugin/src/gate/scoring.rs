// SPDX-License-Identifier: AGPL-3.0-only

//! Scoring one record against one baseline — the pure half of the gate check.
//!
//! Split from [`super::check`] (exact piecewise copy) at the 500-line
//! boundary. Nothing here reads the filesystem or git: `compare` judges one
//! metric against one bound, and `check_record` resolves the record's own
//! (hardware, checkpoint) pair and scores every bound in that entry.

use super::record::{GateBaseline, GateRecord};

/// One metric's comparison, or why it cannot be judged.
pub enum Comparison {
    Pass,
    Fail(String),
    Skip(String),
}

/// Compare one recorded metric against its bound.
pub fn compare(name: &str, value: f64, bound: &super::record::Bound) -> Comparison {
    let noise = bound.noise.unwrap_or(0.0);
    match (bound.min, bound.max) {
        (Some(min), None) if value + noise >= min => Comparison::Pass,
        (Some(min), None) => Comparison::Fail(format!(
            "{name} {value:.2} is below the floor {min:.2} (noise {noise:.2})"
        )),
        (None, Some(max)) if value - noise <= max => Comparison::Pass,
        (None, Some(max)) => Comparison::Fail(format!(
            "{name} {value:.2} is above the ceiling {max:.2} (noise {noise:.2})"
        )),
        // BOTH bounds: a range, or — when they are equal — an EXACT pin.
        //
        // ★ This arm was missing, and a two-sided bound fell through to
        // "malformed". That is fail-closed, so nothing was scored leniently,
        // but it made an exact pin unusable: the gate failed every time and
        // blamed the baseline's syntax rather than the measurement. The BFCL
        // draw size is pinned this way (n=995 / n=1004), because a draw that
        // silently changes size produces a plausible score against thresholds
        // that no longer apply.
        (Some(min), Some(max)) if value + noise >= min && value - noise <= max => Comparison::Pass,
        (Some(min), Some(max)) if (min - max).abs() < f64::EPSILON => Comparison::Fail(format!(
            "{name} is {value:.0}, but this gate is pinned to exactly {min:.0} — \
             the run measured something other than what the baseline describes"
        )),
        (Some(min), Some(max)) => Comparison::Fail(format!(
            "{name} {value:.2} is outside [{min:.2}, {max:.2}] (noise {noise:.2})"
        )),
        (None, None) => Comparison::Skip(format!("{name} has no bound")),
    }
}

/// Check one record against its baseline. `None` means every checkable metric
/// passed; `Some` carries the list of failures. A record whose model does not
/// match the baseline's is a hard failure — comparing gate numbers across
/// checkpoints manufactures results.
pub fn check_record(record: &GateRecord, baseline: &GateBaseline) -> Option<Vec<String>> {
    // The record names both axes: which box served it, and which checkpoint.
    // Score it against THAT pair's thresholds or not at all — a TTFT ceiling
    // from another box, or a BFCL floor from another checkpoint, is not a
    // lenient comparison, it is a meaningless one.
    let hardware = record.hardware.gate_key();
    let entry = match baseline.resolve(&hardware, Some(&record.target_model)) {
        Ok((_, entry)) => entry,
        Err(e) => return Some(vec![format!("{e:#}")]),
    };
    // ★ An entry with no thresholds must not read as "everything passed". The
    // loop below is a no-op over an empty map, so without this the strictest
    // possible verdict — Pass, unconditionally, whatever the run measured —
    // would be produced by the WEAKEST possible baseline. A gate with nothing
    // to enforce has not been passed; it has not been defined.
    if entry.metrics.is_empty() {
        return Some(vec![format!(
            "the baseline entry for {} on {hardware} declares no thresholds — \
             there is nothing here for this run to have passed",
            record.target_model
        )]);
    }
    let mut problems = Vec::new();
    // Baseline-declared serve pins must be ON the record, at the pinned value.
    // BENCH.toml is outside the closure hash, so a pin-only edit would
    // otherwise leave an old record — measured under a different serve config —
    // reading green for a config it never ran.
    for (k, want) in &entry.serve_overrides {
        match record.serve_overrides.get(k) {
            Some(got) if got == want => {}
            Some(got) => problems.push(format!(
                "serve override {k}={got} does not match the baseline pin {k}={want}"
            )),
            None => problems.push(format!(
                "serve override {k}={want} is pinned on the baseline but missing from the record"
            )),
        }
    }
    for (k, got) in &record.serve_overrides {
        if !entry.serve_overrides.contains_key(k) {
            problems.push(format!(
                "serve override {k}={got} is present on the record but not pinned by the baseline"
            ));
        }
    }
    // Baseline-declared PARAM pins must be on the record too, at the pinned
    // value — the same argument as the serve pins one loop up: BENCH.toml is
    // outside the closure hash, so pinning a gate to a calibrated instrument
    // (a specific concurrency ladder, prompt size, output budget) must not
    // leave an old record — measured on the schema default — reading green
    // for an instrument it never ran. Compared whitespace-insensitively:
    // records render int lists as "1, 4, 8, 16" while pins are typed
    // "1,4,8,16", and both name the same value.
    let normalize = |s: &str| s.split(',').map(str::trim).collect::<Vec<_>>().join(",");
    for (k, want) in &entry.param_overrides {
        match record.params.get(k) {
            Some(got) if normalize(got) == normalize(want) => {}
            Some(got) => problems.push(format!(
                "param {k}={got} does not match the baseline pin {k}={want} — the run \
                 measured a different instrument than the one these thresholds describe"
            )),
            None => problems.push(format!(
                "param {k}={want} is pinned on the baseline but missing from the record"
            )),
        }
    }
    for (name, bound) in &entry.metrics {
        let Some(value) = record.metrics.get(name) else {
            problems.push(format!("{name}: missing from the record"));
            continue;
        };
        match compare(name, *value, bound) {
            Comparison::Pass => {}
            Comparison::Fail(reason) => problems.push(reason),
            Comparison::Skip(reason) => problems.push(reason),
        }
    }
    if problems.is_empty() {
        None
    } else {
        Some(problems)
    }
}
