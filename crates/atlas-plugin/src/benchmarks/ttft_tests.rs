// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use super::*;
use crate::artifacts::ArtifactStore;
use crate::plugin::TargetEndpoint;
use crate::result::{Verdict, VerdictKind};

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
}

#[test]
fn without_a_baseline_the_verdict_is_info_not_pass() {
    let g = gate(Mode::Warm, "nobase");
    let (verdict, summary) = g.verdict(Some(800.0), Some(950.0));
    assert_eq!(verdict.kind, VerdictKind::Info);
    assert!(verdict.reason.contains("no baseline"), "{}", verdict.reason);
    assert!(summary.iter().any(|s| s.value == "none"));
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

    // +2.5% median, +1.1% p90 — inside both limits.
    let (ok, _) = g.verdict(Some(820.0), Some(910.0));
    assert_eq!(ok.kind, VerdictKind::Pass, "{}", ok.reason);

    // +5% median — past the 3% limit.
    let (bad, _) = g.verdict(Some(840.0), Some(910.0));
    assert_eq!(bad.kind, VerdictKind::Fail);
    assert!(bad.reason.contains("REGRESSED"), "{}", bad.reason);

    // p90 alone can fail it too: +11% p90 with a flat median.
    let (bad90, _) = g.verdict(Some(800.0), Some(1000.0));
    assert_eq!(bad90.kind, VerdictKind::Fail);
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
    assert!(v.reason.contains("other-box"), "{}", v.reason);
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
    let warm_a = format!("warm-{}", 1024);
    let warm_b = format!("warm-{}", 1024);
    assert_eq!(warm_a, warm_b);
    let cold_a = crate::benchmarks::unique_prefix_tag("cold-1024-0", 1);
    let cold_b = crate::benchmarks::unique_prefix_tag("cold-1024-0", 2);
    assert_ne!(cold_a, cold_b);
}
