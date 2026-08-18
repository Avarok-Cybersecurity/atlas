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
    assert_eq!(v.float("s_per_turn_budget").unwrap(), 8.5);
}

/// `s_per_turn_budget` is deliberately wide open here (PCND: stated, not
/// implied). These fixtures pin the CORRECTNESS halves and the Σwall bound,
/// and their synthetic 3-turn rows carry no meaningful speed; a real budget
/// would fail them for the wrong reason and stop them testing what they name.
/// The speed bound gets its own fixtures, on measured tiers, below.
fn with_rows(rows: Vec<IterationRow>, budget: f64) -> AgenticWebserver {
    with_budgets(rows, budget, f64::INFINITY)
}

fn with_budgets(rows: Vec<IterationRow>, budget: f64, s_per_turn: f64) -> AgenticWebserver {
    AgenticWebserver {
        iterations: rows.len(),
        wall_budget_s: budget,
        s_per_turn_budget: s_per_turn,
        rows,
        ..Default::default()
    }
}

/// One row carrying a whole tier's totals. Both bounds are aggregates over the
/// tier (Σwall, and Σwall÷Σturns), so a tier is fully determined by its two
/// sums — splitting them across ten rows would add fixture noise, not coverage.
fn tier(wall: f64, turns: usize) -> IterationRow {
    IterationRow {
        turns,
        ..row(true, true, wall)
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

/// The five 10/10 + 10/10 tiers measured on the 35B flagship (2026-08-17/18),
/// every one of them a CORRECT run of code that shipped:
///
/// | box  | Σwall  | Σturns | s/turn | old Σ≤1000 | new s/turn≤8.5 |
/// |------|--------|--------|--------|------------|----------------|
/// | dgx1 |  774 s |    113 |  6.85  | pass       | pass           |
/// | dgx1 |  813 s |    115 |  7.07  | pass       | pass           |
/// | dgx1 |  860 s |    126 |  6.83  | pass       | pass           |
/// | dgx1 | 1039 s |    144 |  7.22  | **FAIL**   | pass           |
/// | dgx2 | 1019 s |    166 |  6.14  | **FAIL**   | pass           |
///
/// The last two are the point of this change. Both were 10/10 on both
/// correctness halves; the 1039 s tier ran on the SAME box, the SAME binary and
/// the SAME night as the 774 s one — a 34% swing with the code held constant —
/// and the wall bound ranked them backwards, failing dgx2's 6.14 s/turn (the
/// FASTEST tier ever measured here) while passing dgx1's 7.07.
#[test]
fn every_measured_correct_tier_passes_the_speed_bound() {
    const MEASURED: [(f64, usize); 5] = [
        (774.0, 113),
        (813.0, 115),
        (860.0, 126),
        (1039.0, 144),
        (1019.0, 166),
    ];
    for (wall, turns) in MEASURED {
        let v = with_budgets(vec![tier(wall, turns)], 1300.0, 8.5).verdict();
        assert_eq!(
            v.kind,
            crate::result::VerdictKind::Pass,
            "measured-correct tier {wall}s/{turns} turns must pass: {}",
            v.reason
        );
    }
    // ...and the two that the OLD 1000 s bound rejected really were rejected,
    // so this test proves a behaviour change rather than restating the status quo.
    for (wall, turns) in [(1039.0, 144), (1019.0, 166)] {
        let v = with_budgets(vec![tier(wall, turns)], 1000.0, 8.5).verdict();
        assert_eq!(v.kind, crate::result::VerdictKind::Fail);
        assert!(v.reason.contains("Σwall"), "{}", v.reason);
    }
}

/// The bound has to still BITE, or it is decoration. A 20% per-turn decode
/// regression on the worst measured draw (7.22 -> 8.66) must fail even though
/// its Σwall (1247 s) stays under the 1300 s degeneracy bound — which is
/// exactly the class of regression the old wall-only gate could not see.
#[test]
fn a_real_per_turn_regression_fails_while_wall_stays_in_budget() {
    let regressed = with_budgets(vec![tier(1247.0, 144)], 1300.0, 8.5);
    let v = regressed.verdict();
    assert_eq!(v.kind, crate::result::VerdictKind::Fail);
    assert!(
        v.reason.contains("8.66s/turn > 8.50s/turn"),
        "speed must be the named failure: {}",
        v.reason
    );
    assert!(
        !v.reason.contains("Σwall"),
        "wall must NOT fire — that is the gap being closed: {}",
        v.reason
    );
}

/// Σwall survives as a DEGENERACY bound: an agent that wanders through far more
/// turns than the work needs is fast per turn and still failing. Without this,
/// dropping the wall bound in favour of s/turn would open a real hole.
#[test]
fn wall_still_catches_turn_degeneracy_that_is_fast_per_turn() {
    let wandering = with_budgets(vec![tier(2000.0, 400)], 1300.0, 8.5);
    let v = wandering.verdict();
    assert_eq!(v.kind, crate::result::VerdictKind::Fail);
    assert!(v.reason.contains("Σwall 2000s > 1300s"), "{}", v.reason);
    assert!(
        !v.reason.contains("s/turn >"),
        "5.00 s/turn is healthy; only the wall is wrong: {}",
        v.reason
    );
}

/// A zero-turn tier must not manufacture a speed number. `metrics()` omits the
/// key entirely (a 0.0 would read to `check_record` as the best speed ever
/// recorded, and an infinity would double-report a failure the correctness
/// halves already own).
#[test]
fn a_zero_turn_tier_reports_no_speed_at_all() {
    let empty = with_budgets(vec![tier(120.0, 0)], 1300.0, 8.5);
    assert!(!empty.metrics().contains_key("s_per_turn"));
    assert_eq!(empty.metrics()["sum_turns"], 0.0);
    // The speed bound stays silent; webserver_ok/followed_directions do the failing.
    assert!(!empty.verdict().reason.contains("s/turn >"));
}

/// The record must carry the DENOMINATOR, not just the ratio. Every wall
/// anomaly this campaign chased was undiagnosable from the artifact because
/// `sum_turns` was collected per iteration and then dropped on the floor.
#[test]
fn the_record_carries_turns_so_a_wall_anomaly_is_diagnosable_after_the_fact() {
    let m = with_budgets(vec![tier(500.0, 60), tier(274.0, 53)], 1300.0, 8.5).metrics();
    assert_eq!(m["sum_wall_s"], 774.0);
    assert_eq!(m["sum_turns"], 113.0);
    assert!((m["s_per_turn"] - 774.0 / 113.0).abs() < 1e-9);
}
