// SPDX-License-Identifier: AGPL-3.0-only

//! The two-phase report: what it holds, and what it refuses to claim.

use super::*;
use crate::hardware::policy::{Decision, Validity};
use crate::hardware::state::{GpuComputeApp, ThrottleActive, ThrottleCounters};

fn state(sw_thermal_us: u64, apps: usize) -> HardwareState {
    HardwareState {
        captured_at: 1_000,
        machine: super::super::state::MachineIdentity {
            hostname: Some("dgx2".into()),
            gpu: Some("NVIDIA GB10".into()),
            ..Default::default()
        },
        throttle_active: ThrottleActive {
            sw_thermal: Some(false),
            hw_thermal: Some(false),
            hw_power_brake: Some(false),
            sw_power_cap: Some(true),
        },
        throttle_counters: ThrottleCounters {
            sw_thermal_us: Some(sw_thermal_us),
            hw_thermal_us: Some(0),
            hw_power_brake_us: Some(0),
            ..ThrottleCounters::default()
        },
        gpu_compute_apps: Some(
            (0..apps)
                .map(|i| GpuComputeApp {
                    pid: 1_000 + i as u32,
                    name: "spark".into(),
                    used_mib: Some(1),
                })
                .collect(),
        ),
        ..HardwareState::default()
    }
}

#[test]
fn opening_stamps_the_perf_class_and_the_precheck() {
    let r = HardwareStateReport::opened(Sensitivity::Speed, state(0, 1));
    assert_eq!(r.perf_class, "gb10@dgx2");
    assert_eq!(r.precheck.decision, Decision::Proceed);
    assert!(!r.refuses());
    assert!(r.after.is_none() && r.delta.is_none() && r.postcheck.is_none());
}

/// ★ A run that never closed is UNMEASURED, not invalid. Marking a harness
/// failure as a hardware fault blames the box for the wrong thing.
#[test]
fn an_unclosed_report_is_not_invalid() {
    let r = HardwareStateReport::opened(Sensitivity::Speed, state(0, 1));
    assert!(!r.invalidated());
}

#[test]
fn closing_computes_the_delta_and_the_validity() {
    let mut r = HardwareStateReport::opened(Sensitivity::Speed, state(0, 1));
    let mut after = state(12_000_000, 1);
    after.captured_at = 1_692;
    r.close(after);
    assert_eq!(r.delta.as_ref().unwrap().sw_thermal_us, Some(12_000_000));
    assert_eq!(r.postcheck.as_ref().unwrap().validity, Validity::Invalid);
    assert!(r.invalidated());
}

#[test]
fn a_clean_run_closes_valid() {
    let mut r = HardwareStateReport::opened(Sensitivity::Speed, state(0, 1));
    let mut after = state(0, 1);
    after.captured_at = 1_692;
    r.close(after);
    assert_eq!(r.postcheck.as_ref().unwrap().validity, Validity::Valid);
    assert!(!r.invalidated());
}

#[test]
fn concerns_come_out_in_phase_order() {
    let mut r = HardwareStateReport::opened(Sensitivity::Speed, state(0, 3));
    assert!(r.refuses());
    let pre = r.concerns().len();
    assert!(pre > 0);
    let mut after = state(12_000_000, 3);
    after.captured_at = 1_692;
    r.close(after);
    let all = r.concerns();
    let expected: Vec<&str> = r
        .precheck
        .concerns
        .iter()
        .chain(r.postcheck.as_ref().unwrap().concerns.iter())
        .map(String::as_str)
        .collect();
    assert!(expected.len() > pre);
    assert_eq!(
        all, expected,
        "preflight concerns must precede post-run concerns without loss"
    );
}

/// The report is what lands in `.benchmarks/<id>/<date>-<sha>.json`, so it has
/// to survive the trip verbatim — before, after, delta and both verdicts.
#[test]
fn the_report_round_trips_through_the_record_json() {
    let mut r = HardwareStateReport::opened(Sensitivity::Speed, state(0, 1));
    let mut after = state(5, 1);
    after.captured_at = 1_692;
    r.close(after);
    let json = serde_json::to_string(&r).unwrap();
    let back: HardwareStateReport = serde_json::from_str(&json).unwrap();
    assert_eq!(r, back);
    // Field names a future reader will grep for.
    for key in [
        "before",
        "after",
        "delta",
        "precheck",
        "postcheck",
        "perf_class",
    ] {
        assert!(json.contains(key), "{key} missing from {json}");
    }
}
