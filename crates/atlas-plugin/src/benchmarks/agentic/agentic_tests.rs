// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn the_prompt_is_the_harness_prompt() {
    // Guards against a well-meaning reword: a different prompt is a
    // different benchmark and its numbers are not comparable.
    assert!(PROMPT.starts_with("Please create a pure rust Axum project"));
    assert!(PROMPT.contains("ATLAS_HARNESS_PORT"));
    assert!(PROMPT.contains("Add tests, run them and prove all tests pass"));
    assert!(PROMPT.contains("fuser -k"));
}

#[test]
fn it_requires_confirmation_because_it_runs_shell() {
    const { assert!(DESCRIPTOR.needs_confirmation) };
}

#[test]
fn defaults_are_the_gate_a_tier() {
    let b = AgenticWebserver::default();
    let v = ParamValues::defaults(&b.parameters());
    assert_eq!(v.usize("iterations").unwrap(), 10);
    assert_eq!(v.float("wall_budget_s").unwrap(), 1000.0);
}

fn with_rows(rows: Vec<IterationRow>, budget: f64) -> AgenticWebserver {
    AgenticWebserver {
        iterations: rows.len(),
        wall_budget_s: budget,
        rows,
        ..Default::default()
    }
}

fn row(ok: bool, steps_ok: bool, wall: f64) -> IterationRow {
    IterationRow {
        index: 0,
        wall_s: wall,
        webserver_ok: ok,
        directions: score::Directions {
            steps: score::REQUIRED_STEPS
                .iter()
                .map(|n| (*n, steps_ok))
                .collect(),
        },
        turns: 3,
        tool_calls: 9,
        note: String::new(),
    }
}

#[test]
fn all_three_conditions_must_hold_to_pass() {
    let pass = with_rows(vec![row(true, true, 100.0), row(true, true, 100.0)], 1300.0);
    assert_eq!(pass.verdict().kind, crate::result::VerdictKind::Pass);

    let ws = with_rows(vec![row(false, true, 100.0)], 1300.0);
    assert!(ws.verdict().reason.contains("webserver_ok 0/1"));

    let fd = with_rows(vec![row(true, false, 100.0)], 1300.0);
    assert!(fd.verdict().reason.contains("followed_directions 0/1"));

    let slow = with_rows(vec![row(true, true, 2000.0)], 1300.0);
    assert!(slow.verdict().reason.contains("Σwall"));
}

/// The reference dense-27B tier (2026-08-14, dgx2, main 680b3a568, N=10):
/// webserver_ok 10/10, followed_directions 10/10, per-run walls below,
/// Σ 1925.1 s. Measured, it FAILED — but only the 35B-calibrated 1000 s
/// budget, which is the miscomparison model variants exist to remove. Under
/// the dense variant's own committed ceiling (2500 s, the value
/// `--pull-request-gate`/the TUI derive from its BENCH.toml) the same tier is
/// a PASS. Both directions pinned with the real numbers, so neither the
/// budget nor the derivation can drift without this noticing.
#[test]
fn the_measured_dense_tier_passes_its_own_budget_and_fails_the_35bs() {
    const DENSE_TIER_WALLS: [f64; 10] = [
        156.0, 187.3, 274.2, 144.7, 230.5, 243.3, 205.2, 117.3, 185.2, 181.4,
    ];
    let rows = || {
        DENSE_TIER_WALLS
            .iter()
            .map(|w| row(true, true, *w))
            .collect::<Vec<IterationRow>>()
    };

    let under_35b_budget = with_rows(rows(), 1000.0);
    let v = under_35b_budget.verdict();
    assert_eq!(v.kind, crate::result::VerdictKind::Fail);
    assert!(
        v.reason.contains("Σwall 1925s > 1000s"),
        "wall is the ONLY failure: {}",
        v.reason
    );
    assert!(
        !v.reason.contains("webserver_ok") && !v.reason.contains("followed_directions"),
        "correctness was perfect: {}",
        v.reason
    );

    let under_own_budget = with_rows(rows(), 2500.0);
    assert_eq!(
        under_own_budget.verdict().kind,
        crate::result::VerdictKind::Pass
    );
}

#[test]
fn a_failing_verdict_lists_every_reason_not_just_the_first() {
    let bad = with_rows(vec![row(false, false, 9000.0)], 1300.0);
    let reason = bad.verdict().reason;
    assert!(reason.contains("webserver_ok") && reason.contains("followed_directions"));
    assert!(reason.contains("Σwall"), "{reason}");
}

/// ★ A gate that cannot say WHICH directive failed cannot be fixed. The names
/// lived in `Directions::steps` all along; only the count was ever surfaced,
/// and the 2026-08-09 investigation into an intermittent 9/10 had to be
/// reconstructed from a leftover file in /tmp four hours later.
#[test]
fn a_failed_iteration_names_the_directives_it_missed() {
    let d = super::score::Directions {
        steps: vec![
            ("built", true),
            ("ran", true),
            ("pinged", false),
            ("tore_down", false),
        ],
    };
    assert_eq!(d.met(), 2);
    assert!(!d.overall());
    assert_eq!(
        d.missing(),
        vec!["pinged", "tore_down"],
        "missing() must name them, in declaration order"
    );

    // A fully-evidenced iteration owes no names — an empty list here is what
    // keeps the note clean on the passing path.
    let ok = super::score::Directions {
        steps: vec![("built", true), ("ran", true)],
    };
    assert!(ok.missing().is_empty());
    assert!(ok.overall());
}
