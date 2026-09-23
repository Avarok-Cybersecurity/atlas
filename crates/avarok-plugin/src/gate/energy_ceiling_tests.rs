// SPDX-License-Identifier: AGPL-3.0-only

//! The per-rung J/token CEILING as the gate judge applies it
//! (`scoring::check_record`), on the key the concurrency sweep writes
//! (`c{C}_gpu_rail_joules_per_token`, `hardware::energy::metrics`).
//!
//! Three verdicts matter and each is proven here: above the ceiling FAILS,
//! at or below it PASSES, and an ABSENT value FAILS as missing — energy that
//! was not measured must be visible, never a silent pass. The control pair
//! proves the failure is the ceiling's doing: the same record against the
//! same baseline minus that one bound passes.

use super::tests::{MODEL, SHA, baseline_for, hw, run_record};
use super::*;
use crate::result::Verdict;
use std::collections::BTreeMap;

const KEY: &str = "c8_gpu_rail_joules_per_token";
/// The dense gate's C=8 ceiling as committed (0.3933 J/tok on dgx2 x 1.10,
/// rounded up to two significant figures).
const CEILING: f64 = 0.44;

fn baseline(with_ceiling: bool) -> GateBaseline {
    let mut metrics = BTreeMap::new();
    // A second bound so the entry is never empty without the ceiling — an
    // empty entry is refused outright, which would fake the control.
    metrics.insert(
        "c8_aggregate_tok_s".to_string(),
        Bound {
            min: Some(110.0),
            ..Bound::default()
        },
    );
    if with_ceiling {
        metrics.insert(
            KEY.to_string(),
            Bound {
                max: Some(CEILING),
                ..Bound::default()
            },
        );
    }
    baseline_for(MODEL, metrics)
}

fn record(j_per_tok: Option<f64>) -> GateRecord {
    let mut m = BTreeMap::new();
    m.insert("c8_aggregate_tok_s".to_string(), 127.8);
    if let Some(v) = j_per_tok {
        m.insert(KEY.to_string(), v);
    }
    GateRecord::from_run(
        &run_record(m, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
    )
    .unwrap()
}

#[test]
fn a_rung_above_its_joule_ceiling_fails_and_names_the_ceiling() {
    let problems = check_record(&record(Some(0.50)), &baseline(true)).expect("must fail");
    assert_eq!(
        problems,
        vec![format!("{KEY} 0.50 is above the ceiling 0.44 (noise 0.00)")]
    );
}

#[test]
fn a_rung_at_or_below_its_joule_ceiling_passes() {
    // dgx3 at 2731f279a read 0.3499 at C=8; the ceiling itself is inclusive.
    assert_eq!(check_record(&record(Some(0.3499)), &baseline(true)), None);
    assert_eq!(check_record(&record(Some(CEILING)), &baseline(true)), None);
}

/// MUTATION CONTROL for the failing case: delete the ceiling and the very
/// record that failed above passes. So the red verdict is caused by the `max`
/// bound, not by anything else in the fixture — remove the ceiling from
/// BENCH.toml and `a_rung_above_its_joule_ceiling_fails…`'s premise is gone.
#[test]
fn without_the_ceiling_the_same_over_budget_record_passes() {
    assert_eq!(check_record(&record(Some(0.50)), &baseline(false)), None);
    assert!(check_record(&record(Some(0.50)), &baseline(true)).is_some());
}

/// Energy not measured (no sampler, a remote endpoint, a zero-token window —
/// `joules_per_token` then writes no key) must FAIL the ceiling, visibly.
#[test]
fn an_absent_joule_reading_fails_as_missing_never_passes() {
    assert_eq!(
        check_record(&record(None), &baseline(true)),
        Some(vec![format!("{KEY}: missing from the record")])
    );
}
