// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn the_prompt_is_the_harness_prompt() {
    // A different prompt is a different benchmark and its numbers are not
    // comparable. Compare the whole authoritative shell assignment, not a few
    // fragments that allow an unobserved rewording to survive.
    let harness = include_str!("../../../../../bench/fp8_dgx2_drift/harness/run_tier.sh");
    let assignment = harness
        .lines()
        .find(|line| line.starts_with("PROMPT='"))
        .expect("run_tier.sh must define PROMPT");
    let expected = assignment
        .strip_prefix("PROMPT='")
        .and_then(|prompt| prompt.strip_suffix('\''))
        .expect("PROMPT must remain a single-quoted shell assignment");
    assert_eq!(PROMPT, expected);
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
    assert_eq!(v.usize("max_turns").unwrap(), 40);
    assert_eq!(v.usize("command_timeout_s").unwrap(), 180);
    assert_eq!(v.usize("build_timeout_s").unwrap(), 600);
    assert_eq!(v.usize("serve_timeout_s").unwrap(), 30);
    assert_eq!(v.usize("max_tokens").unwrap(), 8192);
    assert_eq!(v.usize("request_timeout_s").unwrap(), 900);
    assert_eq!(v.float("wall_budget_s").unwrap(), 1000.0);
    // 0.0 = NON-GATING. A schema speed default cannot be right for both the
    // 35B MoE (6.8 s/turn) and the dense 27B (18-40), so variants opt IN by
    // committing a measured bound; see the ParamSpec note.
    assert_eq!(v.float("s_per_turn_budget").unwrap(), 0.0);
}

/// Speed is left NON-GATING here (0.0), which is the shipped default and a
/// legal `ParamKind::Float { min: 0.0 }` value — not the out-of-range
/// `f64::INFINITY` an earlier revision used to switch the bound off. These
/// fixtures pin the CORRECTNESS halves and the Σwall bound; the speed bound
/// gets its own fixtures, on measured tiers, below.
fn with_rows(rows: Vec<IterationRow>, budget: f64) -> AgenticWebserver {
    with_budgets(rows, budget, 0.0)
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

/// One row carrying a whole tier's totals. The two BOUNDS are aggregates over
/// the tier (Σwall, and agent-Σwall÷Σturns), so for them a tier is fully
/// determined by its sums, and splitting across ten rows would add fixture
/// noise rather than coverage.
///
/// ★ It is NOT a whole gate fixture. `metrics()["iterations"]` is `rows.len()`,
/// and both agentic BENCH.toml entries pin `iterations` to exactly 10, so a
/// one-row tier would be rejected by `check_record` even when every bound here
/// passes. Anything testing the RECORD rather than the verdict needs ten rows.
fn tier(wall: f64, turns: usize) -> IterationRow {
    IterationRow {
        turns,
        ..row(true, true, wall)
    }
}

/// A tier whose agent-only wall differs from its total — the scorer's share.
fn tier_split(total: f64, agent: f64, turns: usize) -> IterationRow {
    IterationRow {
        agent_wall_s: agent,
        ..tier(total, turns)
    }
}

fn row(ok: bool, steps_ok: bool, wall: f64) -> IterationRow {
    IterationRow {
        index: 0,
        wall_s: wall,
        // Equal by default: fixtures that care about the scorer's share set it.
        agent_wall_s: wall,
        webserver_ok: ok,
        directions: score::Directions {
            steps: score::REQUIRED_STEPS
                .iter()
                .map(|n| (*n, steps_ok))
                .collect(),
        },
        turns: 3,
        tool_calls: 9,
        completion_tokens: 300,
        note: String::new(),
    }
}

fn committed_agentic_max(model: &str, checkpoint: &str, metric: &str) -> f64 {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace layout")
        .to_path_buf();
    let (_, committed) = crate::gate::bench::load_all(&root)
        .expect("committed BENCH.toml files must load")
        .into_iter()
        .find(|(target, entry)| {
            target.hardware == "gb10"
                && target.model == model
                && entry.gate == "agentic-webserver"
                && entry.checkpoint == checkpoint
        })
        .unwrap_or_else(|| panic!("the {model}/{checkpoint} agentic gate must be committed"));
    committed
        .metrics
        .expect("a measured agentic gate must declare bounds")[metric]
        .max
        .unwrap_or_else(|| panic!("{metric} must have a maximum"))
}

#[test]
fn all_three_conditions_must_hold_to_pass() {
    let pass = with_rows(vec![row(true, true, 100.0), row(true, true, 100.0)], 1300.0);
    assert_eq!(pass.verdict().kind, crate::result::VerdictKind::Pass);

    let ws = with_rows(vec![row(false, true, 100.0)], 1300.0);
    assert_eq!(ws.verdict().kind, crate::result::VerdictKind::Fail);
    assert!(ws.verdict().reason.contains("webserver_ok 0/1"));

    let fd = with_rows(vec![row(true, false, 100.0)], 1300.0);
    assert_eq!(fd.verdict().kind, crate::result::VerdictKind::Fail);
    assert!(fd.verdict().reason.contains("followed_directions 0/1"));

    let slow = with_rows(vec![row(true, true, 2000.0)], 1300.0);
    assert_eq!(slow.verdict().kind, crate::result::VerdictKind::Fail);
    assert!(slow.verdict().reason.contains("Σwall"));
}

/// The reference dense-27B tier (2026-08-14, dgx2, main 680b3a568, N=10):
/// webserver_ok 10/10, followed_directions 10/10, per-run walls below,
/// Σ 1925.1 s. Measured, it FAILED — but only the 35B-calibrated 1000 s
/// budget, which is the miscomparison model variants exist to remove. Under
/// the dense variant's own committed ceiling (5000 s, the value
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

    let dense_wall_budget =
        committed_agentic_max("qwen3.8-27b", "unsloth/Qwen3.8-27B-NVFP4", "sum_wall_s");
    assert_eq!(
        dense_wall_budget, 5000.0,
        "the documented dense wall bound drifted"
    );
    let under_own_budget = with_rows(rows(), dense_wall_budget);
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
/// | box  | Σwall  | Σturns | s/turn | old Σ≤1000 | committed 1800 + 8.5 |
/// |------|--------|--------|--------|------------|----------------------|
/// | dgx1 |  774 s |    113 |  6.85  | pass       | pass                 |
/// | dgx1 |  813 s |    115 |  7.07  | pass       | pass                 |
/// | dgx1 |  860 s |    126 |  6.83  | pass       | pass                 |
/// | dgx1 | 1039 s |    144 |  7.22  | **FAIL**   | pass                 |
/// | dgx2 | 1019 s |    166 |  6.14  | **FAIL**   | pass                 |
///
/// The last two are the point of this change. Both were 10/10 on both
/// correctness halves; the 1039 s tier ran on the SAME box, the SAME binary and
/// the SAME night as the 774 s one — a 34% swing with the code held constant —
/// and the wall bound ranked them backwards, failing dgx2's 6.14 s/turn (the
/// FASTEST tier ever measured here) while passing dgx1's 7.07.
///
/// The budgets here are the 35B's COMMITTED BENCH.toml numbers, not invented
/// ones: `sum_wall_s max = 1800`, `s_per_turn max = 8.5`. That matters because
/// the gate substitutes those, never the schema defaults — an earlier revision
/// of this test asserted 1300, a value the gate cannot produce.
#[test]
fn every_measured_correct_tier_passes_the_committed_35b_bounds() {
    let wall_budget =
        committed_agentic_max("qwen3.6-35b-a3b", "Qwen/Qwen3.6-35B-A3B-FP8", "sum_wall_s");
    let speed_budget =
        committed_agentic_max("qwen3.6-35b-a3b", "Qwen/Qwen3.6-35B-A3B-FP8", "s_per_turn");
    assert_eq!(wall_budget, 1800.0, "the documented blowup bound drifted");
    assert_eq!(speed_budget, 8.5, "the documented speed bound drifted");

    const MEASURED: [(f64, usize); 5] = [
        (774.0, 113),
        (813.0, 115),
        (860.0, 126),
        (1039.0, 144),
        (1019.0, 166),
    ];
    for (wall, turns) in MEASURED {
        let v = with_budgets(vec![tier(wall, turns)], wall_budget, speed_budget).verdict();
        assert_eq!(
            v.kind,
            crate::result::VerdictKind::Pass,
            "measured-correct tier {wall}s/{turns} turns must pass: {}",
            v.reason
        );
    }
    // ...and the two that the OLD 1000 s bound rejected really were rejected,
    // so this proves a behaviour change rather than restating the status quo.
    for (wall, turns) in [(1039.0, 144), (1019.0, 166)] {
        let v = with_budgets(vec![tier(wall, turns)], 1000.0, speed_budget).verdict();
        assert_eq!(v.kind, crate::result::VerdictKind::Fail);
        assert!(v.reason.contains("Σwall"), "{}", v.reason);
    }
}

/// WHY the schema speed default is 0.0 and not the 35B's 8.5.
///
/// The dense 27B's own BENCH.toml records its per-turn cost as 18 s/turn on the
/// pre-branch nightly and ~40 s/turn on the current serve, against the 35B's
/// 6.8. Its reference tier — Σ 1925 s, 10/10 + 10/10, a HEALTHY run — is
/// therefore 4x the 35B's bound. Had `s_per_turn_budget` shipped defaulting to
/// 8.5, every dense agentic run would have failed on a number measured from a
/// different model, and `check_record` would not even have shown why: the dense
/// entry commits no `s_per_turn` bound, so the failure would surface only as an
/// unexplained rejected verdict.
#[test]
fn the_35b_speed_bound_would_fail_the_healthy_dense_tier_hence_non_gating() {
    // 1925 s over ~107 turns, the dense reference tier's own numbers.
    let dense = || vec![tier(1925.0, 107)];

    let with_the_35bs_bound = with_budgets(dense(), 5000.0, 8.5).verdict();
    assert_eq!(with_the_35bs_bound.kind, crate::result::VerdictKind::Fail);
    assert!(
        with_the_35bs_bound
            .reason
            .contains("17.991s/turn > 8.500s/turn"),
        "the healthy dense tier fails a bound drawn from another model: {}",
        with_the_35bs_bound.reason
    );

    // Shipped behaviour: no committed bound -> 0.0 -> speed is not gated.
    let v = with_budgets(dense(), 5000.0, 0.0).verdict();
    assert_eq!(v.kind, crate::result::VerdictKind::Pass, "{}", v.reason);
    assert!(
        v.reason.contains("(unbounded)"),
        "a pass must SAY speed went unchecked rather than imply it passed: {}",
        v.reason
    );
}

/// The speed bound divides the AGENT's seconds, not the tier's total. The
/// scorer's `cargo build --release` is a per-ITERATION cost, so charging it to
/// a per-TURN ratio adds a term that shrinks as turns grow — a long trajectory
/// would look faster per turn purely for amortising the build. With a 10 s
/// scorer per iteration that artefact is ~0.28 s/turn between the 113- and
/// 166-turn ends of the measured range, which is 72% of the entire 0.39 s/turn
/// spread the 8.5 headroom is drawn against.
#[test]
fn the_speed_bound_excludes_the_scorers_build_from_the_numerator() {
    // Total 874 s, of which 100 s is scorer; 113 turns.
    let b = with_budgets(vec![tier_split(874.0, 774.0, 113)], 1800.0, 7.0);
    let m = b.metrics();
    assert_eq!(m["sum_wall_s"], 874.0);
    assert_eq!(m["sum_agent_wall_s"], 774.0);
    assert!((m["s_per_turn"] - 774.0 / 113.0).abs() < 1e-9);

    // 6.85 agent-only passes a 7.0 bound; 7.73 total-wall would have failed it.
    assert_eq!(b.verdict().kind, crate::result::VerdictKind::Pass);
    // ...and the same tier scored on TOTAL wall would have failed the same
    // bound, which is the whole point of splitting the two numerators.
    let charged_the_scorer = with_budgets(vec![tier(874.0, 113)], 1800.0, 7.0);
    assert_eq!(
        charged_the_scorer.verdict().kind,
        crate::result::VerdictKind::Fail
    );
}

/// The bound has to still BITE, or it is decoration. A 20% per-turn decode
/// regression on the worst measured draw (7.22 -> 8.66) must fail even though
/// its Σwall (1247 s) stays under the 1300 s degeneracy bound — which is
/// exactly the class of regression the old wall-only gate could not see.
#[test]
fn a_real_per_turn_regression_fails_while_wall_stays_in_budget() {
    let regressed = with_budgets(vec![tier(1247.0, 144)], 1800.0, 8.5);
    let v = regressed.verdict();
    assert_eq!(v.kind, crate::result::VerdictKind::Fail);
    assert!(
        v.reason.contains("8.660s/turn > 8.500s/turn"),
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
    // 220 turns at 8.30 s/turn = 1826 s. Deliberately INSIDE the reachable
    // envelope — 220 against the 400-turn hard cap (max_turns 40 x 10) and only
    // 33% above the 166-turn worst measured tier — rather than standing on the
    // cap, which would require all ten iterations to exhaust max_turns while
    // still scoring 10/10 and would prove the bound only in a corner nobody
    // reaches.
    let wandering = with_budgets(vec![tier(1826.0, 220)], 1800.0, 8.5);
    let v = wandering.verdict();
    assert_eq!(v.kind, crate::result::VerdictKind::Fail);
    assert!(v.reason.contains("Σwall 1826s > 1800s"), "{}", v.reason);
    assert!(
        !v.reason.contains("s/turn >"),
        "8.30 s/turn is inside the speed bound; only the wall is wrong: {}",
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
    // This fixture keeps correctness true to isolate speed omission: those
    // independent failure partitions are owned by `all_three_conditions`.
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
