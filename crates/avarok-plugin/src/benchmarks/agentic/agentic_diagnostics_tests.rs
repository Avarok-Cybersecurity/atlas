// SPDX-License-Identifier: AGPL-3.0-only

//! The trajectory diagnostics `metrics()` records (#1159), and the property
//! their addition must keep: no verdict moves. Split from `agentic_tests.rs`
//! at the 500-line cap; the tier fixtures stay there and are shared.

use super::tests::{tier, with_budgets, with_rows};
use super::*;

/// How each trajectory ENDED, in the record — the observation #1159 found
/// computed and thrown away. `hit_turn_cap >= 1` says an iteration ran out of
/// its turn budget; `= 0` beside a missing step-suffix says the loop believed
/// the model was done. Six failing records could not tell those apart.
#[test]
fn the_record_says_how_each_trajectory_ended() {
    let mut capped = tier(300.0, 12);
    capped.hit_turn_cap = true;
    capped.truncated_turns = 1;
    let mut degenerate = tier(400.0, 17);
    degenerate.unparsed_call_turns = 3;
    degenerate.truncated_turns = 2;
    let m = with_rows(vec![capped, degenerate, tier(200.0, 9)], 1300.0).metrics();
    assert_eq!(m["sum_hit_turn_cap"], 1.0);
    assert_eq!(m["sum_truncated_turns"], 3.0);
    assert_eq!(m["sum_unparsed_call_turns"], 3.0);
    assert_eq!(m["max_iter_turns"], 17.0);
    // A clean tier records zeros — measured, not missing — and an EMPTY tier
    // has no maximum to report, so it reports none rather than 0.
    let clean = with_rows(vec![tier(100.0, 5)], 1300.0).metrics();
    assert_eq!(clean["sum_hit_turn_cap"], 0.0);
    assert_eq!(clean["max_iter_turns"], 5.0);
    let empty = with_rows(vec![], 1300.0).metrics();
    assert_eq!(empty["sum_hit_turn_cap"], 0.0);
    assert!(!empty.contains_key("max_iter_turns"));
}

/// Every diagnostic key this change adds, in one place — the set the
/// invariance proof strips and the bound guard forbids.
const DIAGNOSTIC_KEYS: [&str; 4] = [
    "sum_hit_turn_cap",
    "sum_truncated_turns",
    "sum_unparsed_call_turns",
    "max_iter_turns",
];

fn agentic_gate_record(rows: Vec<IterationRow>) -> crate::gate::GateRecord {
    let bench = with_budgets(rows, 1800.0, 8.5);
    let frame = crate::result::BenchmarkResult::completed("done", std::time::Duration::ZERO)
        .with_metrics(bench.metrics())
        .with_verdict(bench.verdict());
    let run = crate::history::RunRecord {
        schema: 1,
        run_id: "run-1".to_string(),
        benchmark_id: DESCRIPTOR.id.to_string(),
        benchmark_name: DESCRIPTOR.name.to_string(),
        recorded_at: 1_785_891_382,
        serve_overrides: Default::default(),
        target_url: String::new(),
        target_model: "Qwen/Qwen3.6-35B-A3B-FP8".to_string(),
        params: Default::default(),
        source: crate::RunSource::Cli,
        atlas_version: "test".to_string(),
        frame,
    };
    let hw = crate::hardware::Hardware {
        gpu: "NVIDIA GB10".to_string(),
        ..Default::default()
    };
    crate::gate::GateRecord::from_run(&run, hw, "b72dad1893".into(), Vec::new(), None).unwrap()
}

/// The property this change must keep: adding metric keys changes no verdict.
/// Proven three ways — the tier verdict ignores the counters, `check_record`
/// against the COMMITTED 35B baseline scores a record identically with and
/// without the keys, and no committed BENCH.toml bounds any of them. If that
/// last assertion ever fails, a disclosure has become a gate: that is a
/// benchmark-definition change and needs its own stack and a re-measured bar.
#[test]
fn trajectory_diagnostics_are_recorded_and_never_gated() {
    // 1. The tier verdict reads none of the counters.
    // Builders rather than values: `IterationRow` is deliberately not `Clone`.
    // ★ 77.4 x 10 (Sigma 774 s) -> 57.3 x 10 (Sigma 573 s) on 2026-09-21, with the
    // ceiling re-cut 1800 -> 700. The fixture is incidental to what this test
    // proves — it only has to be a clean tier that PASSES the committed bounds,
    // or the `check_record` comparison below cannot bite — but 774 stopped
    // being one. 573 is the MEAN of the post-2026-09-12 regime the new ceiling
    // was cut against (n=21, 521-647 s), and 57.3/9 = 6.37 s/turn sits inside
    // the 6.14-7.22 band every measured correct tier has occupied, so the
    // fixture is still a real shape rather than a number chosen to pass.
    let clean = || {
        (0..10)
            .map(|_| tier(57.3, 9))
            .collect::<Vec<IterationRow>>()
    };
    let noisy = || {
        (0..10)
            .map(|i| {
                let mut r = tier(57.3, 9);
                r.hit_turn_cap = i % 2 == 0;
                r.truncated_turns = i;
                r.unparsed_call_turns = 10 - i;
                r
            })
            .collect::<Vec<IterationRow>>()
    };
    let (a, b) = (
        with_budgets(clean(), 1800.0, 8.5).verdict(),
        with_budgets(noisy(), 1800.0, 8.5).verdict(),
    );
    assert_eq!((a.kind, &a.reason), (b.kind, &b.reason));
    assert_eq!(a.kind, crate::result::VerdictKind::Pass, "{}", a.reason);

    // 2. The committed baseline scores the record the same with and without.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace layout")
        .to_path_buf();
    let baseline = crate::gate::bench::baseline_for(&root, DESCRIPTOR.id)
        .expect("the committed agentic-webserver baseline must load");
    for rows in [clean(), noisy()] {
        let with = agentic_gate_record(rows);
        for key in DIAGNOSTIC_KEYS {
            assert!(with.metrics.contains_key(key), "{key} must be recorded");
        }
        let mut without = with.clone();
        for key in DIAGNOSTIC_KEYS {
            without.metrics.remove(key);
        }
        // Compared BEFORE the mutation below on the same two records.
        assert_eq!(
            crate::gate::check_record(&with, &baseline),
            None,
            "a clean 10/10 tier must pass the committed bounds for this comparison to bite"
        );
        assert_eq!(
            crate::gate::check_record(&with, &baseline),
            crate::gate::check_record(&without, &baseline)
        );
        // ...and on the failing side too, so equality is not "both pass".
        let mut with = with;
        with.metrics.insert("followed_directions".into(), 9.0);
        without.metrics.insert("followed_directions".into(), 9.0);
        let failing = crate::gate::check_record(&with, &baseline);
        assert!(failing.is_some());
        assert_eq!(failing, crate::gate::check_record(&without, &baseline));
    }

    // 3. No committed bound names a diagnostic.
    let mut checked = 0;
    for (target, entry) in crate::gate::bench::load_all(&root).expect("BENCH.toml files load") {
        if entry.gate != DESCRIPTOR.id {
            continue;
        }
        checked += 1;
        for key in DIAGNOSTIC_KEYS {
            assert!(
                !entry.metrics.as_ref().is_some_and(|m| m.contains_key(key)),
                "{}/{} bounds `{key}`: a trajectory diagnostic has become a gate, which is \
                 a benchmark-definition change and needs its own stack and a re-measured bar",
                target.hardware,
                target.model
            );
        }
    }
    assert!(
        checked >= 2,
        "expected the 35B and dense agentic entries, saw {checked}"
    );
}
