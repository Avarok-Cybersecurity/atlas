// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use super::*;
use crate::artifacts::ArtifactStore;
use crate::plugin::TargetEndpoint;
use crate::result::{CellStyle, Stat, Verdict, VerdictKind};

fn stat_projection(stats: &[Stat]) -> Vec<(&str, &str, &str, CellStyle)> {
    stats
        .iter()
        .map(|stat| {
            (
                stat.label.as_str(),
                stat.value.as_str(),
                stat.unit.as_str(),
                stat.style,
            )
        })
        .collect()
}

fn gate(mode: Mode, root: &str) -> TtftGate {
    let mut g = TtftGate::new(mode);
    let (tx, rx) = std::sync::mpsc::channel();
    // Keep the receiver alive for the test's lifetime so `emit` does not fail.
    std::mem::forget(rx);
    let dir = std::env::temp_dir().join(format!("atlas-ttft-{root}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    g.handle = Some(PluginHandle::new(
        1,
        TargetEndpoint::local(8888, "test-model"),
        ArtifactStore::with_root(dir),
        tx,
        Arc::new(AtomicBool::new(false)),
    ));
    g.started = Some(Instant::now());
    let v = ParamValues::defaults(&g.parameters());
    g.configure(&v).unwrap();
    g
}

#[test]
fn the_two_gates_are_distinct_benchmarks_with_distinct_baselines() {
    assert_ne!(WARM_DESCRIPTOR.id, COLD_DESCRIPTOR.id);
    assert_eq!(TtftGate::new(Mode::Warm).descriptor().id, "ttft-warm-gate");
    assert_eq!(TtftGate::new(Mode::Cold).descriptor().id, "ttft-cold-gate");
}

#[test]
fn defaults_are_gate_c_thresholds() {
    let g = TtftGate::new(Mode::Warm);
    let v = ParamValues::defaults(&g.parameters());
    assert_eq!(v.float("median_limit_pct").unwrap(), 3.0);
    assert_eq!(v.float("p90_limit_pct").unwrap(), 5.0);
    assert_eq!(v.int_list("prompt_lengths").unwrap(), &[256, 1024, 4096]);
    assert_eq!(v.usize("repeats").unwrap(), 12);
    assert!(v.bool("update_baseline").unwrap());
    assert_eq!(v.usize("request_timeout_s").unwrap(), 300);
}

#[test]
fn three_samples_use_the_true_median_not_nearest_rank_p50() {
    assert_eq!(
        ttft_stats(&[3.0, 1.0, 2.0]),
        (Some(2.0), Some(3.0), Some(3.0))
    );
}

#[test]
fn only_finite_positive_ttft_is_measurement_evidence() {
    assert!(valid_ttft_ms(1.0));
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(!valid_ttft_ms(invalid), "value={invalid:?}");
    }
}

#[test]
fn reconfiguring_restarts_the_endpoint_probe_and_clears_rows() {
    let mut g = gate(Mode::Warm, "reconfigure");
    g.probed = true;
    g.cursor = 9;
    g.rows.push(LengthRow {
        prompt_tokens: 256,
        samples: vec![1.0],
        cached_tokens: 128,
    });

    let mut values = ParamValues::defaults(&g.parameters());
    values.set("prompt_lengths", ParamValue::IntList(vec![16]));
    values.set("repeats", ParamValue::Int(1));
    values.set("median_limit_pct", ParamValue::Float(1.0));
    values.set("p90_limit_pct", ParamValue::Float(2.0));
    values.set("update_baseline", ParamValue::Bool(false));
    values.set("request_timeout_s", ParamValue::Int(10));
    g.configure(&values).unwrap();

    assert_eq!(g.lengths, [16]);
    assert_eq!(g.repeats, 1);
    assert_eq!(g.median_limit_pct, 1.0);
    assert_eq!(g.p90_limit_pct, 2.0);
    assert!(!g.update_baseline);
    assert_eq!(g.timeout, Duration::from_secs(10));
    assert!(!g.probed);
    assert_eq!(g.cursor, 0);
    assert!(g.rows.is_empty());
}

#[test]
fn without_a_baseline_the_verdict_is_info_not_pass() {
    let g = gate(Mode::Warm, "nobase");
    let (verdict, summary) = g.verdict(Some(800.0), Some(950.0));
    assert_eq!(verdict.kind, VerdictKind::Info);
    assert_eq!(
        verdict.reason,
        "no baseline on this box yet — this run is recorded as the baseline"
    );
    assert_eq!(
        stat_projection(&summary),
        [
            ("Median TTFT", "800.0", "ms", CellStyle::Accent),
            ("p90 TTFT", "950.0", "ms", CellStyle::Neutral),
            ("Baseline", "none", "", CellStyle::Dim),
        ]
    );
}

#[test]
fn a_regression_past_the_median_limit_fails() {
    let g = gate(Mode::Warm, "regress");
    let store = g.handle().unwrap().artifacts().clone();
    let mut m = std::collections::BTreeMap::new();
    m.insert("median_ms".to_string(), 800.0);
    m.insert("p90_ms".to_string(), 900.0);
    baseline::save(
        &store,
        WARM_DESCRIPTOR.id,
        "http://127.0.0.1:8888",
        "test-model",
        m,
    )
    .unwrap();

    // Exact limits are inclusive.
    let (ok, _) = g.verdict(Some(824.0), Some(945.0));
    assert_eq!(ok.kind, VerdictKind::Pass, "{}", ok.reason);
    assert_eq!(
        ok.reason,
        "median +3.0% (limit +3.0%) · p90 +5.0% (limit +5.0%)"
    );

    // +3.1% median is past the limit and far above the absolute noise floor.
    let (bad, _) = g.verdict(Some(824.8), Some(945.0));
    assert_eq!(bad.kind, VerdictKind::Fail);
    assert_eq!(
        bad.reason,
        "REGRESSED — median +3.1% (limit +3.0%) · p90 +5.0% (limit +5.0%)"
    );

    // p90 alone can fail it too.
    let (bad90, _) = g.verdict(Some(800.0), Some(954.0));
    assert_eq!(bad90.kind, VerdictKind::Fail);
    assert_eq!(
        bad90.reason,
        "REGRESSED — median +0.0% (limit +3.0%) · p90 +6.0% (limit +5.0%)"
    );
}

#[test]
fn a_baseline_from_another_target_reports_instead_of_gating() {
    let g = gate(Mode::Warm, "othertarget");
    let store = g.handle().unwrap().artifacts().clone();
    let mut m = std::collections::BTreeMap::new();
    m.insert("median_ms".to_string(), 100.0);
    baseline::save(
        &store,
        WARM_DESCRIPTOR.id,
        "http://other-box:8888",
        "test-model",
        m,
    )
    .unwrap();
    // 8x worse than the stored number, but the stored number is from a
    // different box: comparing it would be the exact "manufactured win/loss"
    // trap, so this must not gate.
    let (v, _) = g.verdict(Some(800.0), Some(900.0));
    assert_eq!(v.kind, VerdictKind::Info);
    assert_eq!(
        v.reason,
        "baseline was recorded against http://other-box:8888 / test-model — not comparable, reporting only"
    );
}

#[test]
fn missing_current_measurements_cannot_pass() {
    let g = gate(Mode::Warm, "missing-current");
    let store = g.handle().unwrap().artifacts().clone();
    baseline::save(
        &store,
        WARM_DESCRIPTOR.id,
        "http://127.0.0.1:8888",
        "test-model",
        std::collections::BTreeMap::from([("median_ms".into(), 800.0), ("p90_ms".into(), 900.0)]),
    )
    .unwrap();
    for (median, p90) in [
        (None, Some(900.0)),
        (Some(800.0), None),
        (Some(0.0), Some(900.0)),
    ] {
        let (verdict, _) = g.verdict(median, p90);
        assert_eq!(verdict.kind, VerdictKind::Fail);
        assert_eq!(
            verdict.reason,
            "run produced no usable median and p90 TTFT measurements"
        );
    }
}

#[test]
fn an_incomplete_same_box_baseline_is_not_partially_gated() {
    let g = gate(Mode::Warm, "incomplete-baseline");
    let store = g.handle().unwrap().artifacts().clone();
    baseline::save(
        &store,
        WARM_DESCRIPTOR.id,
        "http://127.0.0.1:8888",
        "test-model",
        std::collections::BTreeMap::from([("median_ms".into(), 800.0)]),
    )
    .unwrap();
    let (verdict, summary) = g.verdict(Some(800.0), Some(900.0));
    assert_eq!(verdict.kind, VerdictKind::Info);
    assert_eq!(
        verdict.reason,
        "same-box baseline is missing usable median_ms or p90_ms — not comparable, reporting only"
    );
    assert_eq!(summary.last().expect("baseline tile").value, "incomplete");
    assert_eq!(
        summary.last().expect("baseline tile").style,
        CellStyle::Warn
    );
}

#[test]
fn the_absolute_noise_floor_is_strict() {
    let g = gate(Mode::Warm, "noise-floor");
    let store = g.handle().unwrap().artifacts().clone();
    baseline::save(
        &store,
        WARM_DESCRIPTOR.id,
        "http://127.0.0.1:8888",
        "test-model",
        std::collections::BTreeMap::from([("median_ms".into(), 30.0), ("p90_ms".into(), 30.0)]),
    )
    .unwrap();

    let (at_floor, _) = g.verdict(Some(32.0), Some(32.0));
    assert_eq!(at_floor.kind, VerdictKind::Pass);
    let (past_floor, _) = g.verdict(Some(32.1), Some(32.1));
    assert_eq!(past_floor.kind, VerdictKind::Fail);
}

/// A failing run must not overwrite the baseline it just failed against.
///
/// ★ The stored baseline is what the NEXT run is compared to. Saving it
/// unconditionally meant a regression became the new bar: run once and FAIL at
/// +10%, run the identical build again and it is 0% against its own regressed
/// number — PASS, with a gate record to prove it. The percentage guard would
/// then only ever catch the FIRST run after a regression landed, and a re-run
/// (which a stochastic gate invites) launders it away.
#[test]
fn a_failing_run_does_not_become_the_new_baseline() {
    let g = gate(Mode::Warm, "nolaunder");
    assert!(!g.should_store(&Verdict::fail("REGRESSED — median +10.0%")));
    // The two cases that MUST still store: a clean pass, and the first run on a
    // box, which has no baseline to compare against and exists to create one.
    assert!(g.should_store(&Verdict::pass("median +0.1%")));
    assert!(g.should_store(&Verdict::info("no baseline on this box yet")));

    // …and the opt-out still wins over all of them.
    let mut off = gate(Mode::Warm, "nolaunder-off");
    let mut v = ParamValues::defaults(&off.parameters());
    v.set("update_baseline", ParamValue::Bool(false));
    off.configure(&v).unwrap();
    assert!(!off.should_store(&Verdict::pass("median +0.1%")));
}

#[test]
fn warm_reuses_one_tag_per_length_and_cold_never_repeats_one() {
    // The whole cold/warm distinction is the prefix_tag, so pin it directly.
    let warm_a = sample_prefix_tag(Mode::Warm, 1024, 0, 1);
    let warm_b = sample_prefix_tag(Mode::Warm, 1024, 11, 2);
    assert_eq!(warm_a, "warm-1024");
    assert_eq!(warm_a, warm_b);
    let cold_a = sample_prefix_tag(Mode::Cold, 1024, 0, 1);
    let cold_b = sample_prefix_tag(Mode::Cold, 1024, 0, 2);
    let cold_c = sample_prefix_tag(Mode::Cold, 1024, 1, 1);
    assert_ne!(cold_a, cold_b);
    assert_ne!(cold_a, cold_c);
}
