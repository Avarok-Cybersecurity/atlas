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
    assert_eq!(v.float("wall_budget_s").unwrap(), 1300.0);
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

#[test]
fn a_failing_verdict_lists_every_reason_not_just_the_first() {
    let bad = with_rows(vec![row(false, false, 9000.0)], 1300.0);
    let reason = bad.verdict().reason;
    assert!(reason.contains("webserver_ok") && reason.contains("followed_directions"));
    assert!(reason.contains("Σwall"), "{reason}");
}
