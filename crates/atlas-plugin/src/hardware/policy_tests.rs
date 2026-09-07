// SPDX-License-Identifier: AGPL-3.0-only

//! The verdict policy, exercised as what it is: a pure function of
//! `(sensitivity, state, options)`. No env, no GPU, no I/O.
//!
//! The load-bearing property, asserted from several directions below: an
//! UNKNOWN input never produces a healthy verdict. `Decision::Proceed` and
//! `Validity::Valid` are reachable only when every gating input was readable
//! AND healthy. Anything else is at worst `Warn`/`Unknown` — recorded, and
//! never mistakable for a pass.

use super::*;
use crate::hardware::state::{GpuComputeApp, ThermalZone, ThrottleActive};

fn options() -> PolicyOptions {
    PolicyOptions::default()
}

/// dgx1: cool, unthrottled, one model resident.
fn healthy() -> HardwareState {
    HardwareState {
        gpu_temp_c: Some(55.0),
        chassis_temps_c: Some(vec![ThermalZone {
            name: "acpitz".into(),
            temp_c: 65.0,
        }]),
        throttle_active: ThrottleActive {
            sw_power_cap: Some(true),
            sw_thermal: Some(false),
            hw_thermal: Some(false),
            hw_power_brake: Some(false),
        },
        gpu_compute_apps: Some(vec![GpuComputeApp {
            pid: 2_392_883,
            name: "spark".into(),
            used_mib: Some(109_281),
        }]),
        ..HardwareState::default()
    }
}

fn app(pid: u32) -> GpuComputeApp {
    GpuComputeApp {
        pid,
        name: "spark".into(),
        used_mib: Some(87_014),
    }
}

fn delta(sw_us: Option<u64>, hw_us: Option<u64>) -> HardwareStateDelta {
    HardwareStateDelta {
        elapsed_s: Some(692),
        sw_thermal_us: sw_us,
        hw_thermal_us: hw_us,
        hw_power_brake_us: Some(0),
        ..HardwareStateDelta::default()
    }
}

#[test]
fn a_healthy_box_proceeds_with_nothing_to_say() {
    let p = precheck(Sensitivity::Speed, &healthy(), options());
    assert_eq!(p.decision, Decision::Proceed);
    assert!(p.concerns.is_empty(), "{:?}", p.concerns);
}

/// SW power cap is Active on the HEALTHY box for 16,130 s of an 11.2-day
/// uptime. Treating it as a thermal fault would refuse every run.
#[test]
fn an_active_power_cap_is_not_a_reason_to_refuse() {
    assert_eq!(healthy().throttle_active.sw_power_cap, Some(true));
    assert_eq!(
        precheck(Sensitivity::Speed, &healthy(), options()).decision,
        Decision::Proceed
    );
}

#[test]
fn a_speed_gate_refuses_a_box_that_is_thermally_throttling_right_now() {
    let mut s = healthy();
    s.throttle_active.hw_thermal = Some(true);
    let p = precheck(Sensitivity::Speed, &s, options());
    assert_eq!(p.decision, Decision::Refuse);
    assert!(p.concerns.iter().any(|c| c.contains("ACTIVE")), "{p:?}");
}

/// ★ The 2026-08-15 leftover: a killed `spark serve` left its driver process
/// holding 87 GB, so the box hosted two compute apps and the next run starved.
#[test]
fn a_speed_gate_refuses_a_box_with_a_second_gpu_process() {
    let mut s = healthy();
    s.gpu_compute_apps = Some(vec![app(2_392_883), app(2_118_440)]);
    let p = precheck(Sensitivity::Speed, &s, options());
    assert_eq!(p.decision, Decision::Refuse);
    assert!(
        p.concerns.iter().any(|c| c.contains("2 GPU compute")),
        "{p:?}"
    );
}

/// ★ A correctness gate is RECORDED and PROCEEDS. Accuracy is not thermally
/// sensitive, and blocking a 3.5-hour BFCL run because the chassis is warm
/// would stop correctness work for a reason that cannot reach the number.
#[test]
fn a_correctness_gate_never_refuses_but_still_records_everything() {
    let mut s = healthy();
    s.throttle_active.hw_thermal = Some(true);
    s.gpu_compute_apps = Some(vec![app(1), app(2), app(3)]);
    let speed = precheck(Sensitivity::Speed, &s, options());
    let correctness = precheck(Sensitivity::Correctness, &s, options());
    assert_eq!(speed.decision, Decision::Refuse);
    assert_eq!(correctness.decision, Decision::Warn);
    for concern in &speed.concerns {
        assert!(
            correctness.concerns.contains(concern),
            "correctness dropped {concern:?}"
        );
    }
}

/// ★ Unknown is not a pass. Every gating input that could not be read has to
/// leave the verdict at Warn, so a box that cannot describe itself never looks
/// like a box that described itself as healthy.
#[test]
fn an_unreadable_field_never_yields_a_healthy_verdict() {
    let cases: [(&str, HardwareState); 3] = [
        ("nothing readable at all", HardwareState::default()),
        (
            "throttle reasons unreadable",
            HardwareState {
                throttle_active: ThrottleActive::default(),
                ..healthy()
            },
        ),
        (
            "compute apps unlistable",
            HardwareState {
                gpu_compute_apps: None,
                ..healthy()
            },
        ),
    ];
    for (label, state) in cases {
        for sensitivity in [Sensitivity::Speed, Sensitivity::Correctness] {
            let p = precheck(sensitivity, &state, options());
            assert_ne!(p.decision, Decision::Proceed, "{label} / {sensitivity:?}");
            assert!(!p.concerns.is_empty(), "{label} said nothing");
        }
    }
}

/// An unreadable box is WARNED about, not refused: a benchmark must stay
/// runnable on a machine that lacks the reporting tools.
#[test]
fn an_unreadable_box_warns_rather_than_blocking_the_suite() {
    let p = precheck(Sensitivity::Speed, &HardwareState::default(), options());
    assert_eq!(p.decision, Decision::Warn);
}

/// Absolute temperature ships DISABLED — one healthy box and one degraded box
/// do not locate a knee. Recorded either way, which is how the third point
/// gets collected.
#[test]
fn absolute_temperature_is_recorded_by_default_and_gates_only_on_opt_in() {
    let mut s = healthy();
    s.gpu_temp_c = Some(89.0);
    s.chassis_temps_c = Some(vec![ThermalZone {
        name: "acpitz".into(),
        temp_c: 89.0,
    }]);

    let off = precheck(Sensitivity::Speed, &s, options());
    assert_eq!(off.decision, Decision::Warn);
    assert_eq!(off.concerns.len(), 2, "{:?}", off.concerns);

    let on = precheck(
        Sensitivity::Speed,
        &s,
        PolicyOptions {
            absolute_temp_gate: true,
            ..options()
        },
    );
    assert_eq!(on.decision, Decision::Refuse);
    assert_eq!(
        on.concerns, off.concerns,
        "the opt-in changes only the level"
    );
}

#[test]
fn the_documented_ceilings_are_the_measured_ones() {
    // dgx1 idled 52-66 C, dgx2 sat at 70-89 C. Both constants live between
    // those two clusters; pinned so a silent edit shows up as a test change.
    assert_eq!(GPU_TEMP_CEILING_C, 75.0);
    assert_eq!(CHASSIS_TEMP_CEILING_C, 80.0);
}

/// ★ The kill switch suppresses the REFUSAL and nothing else. It says loudly
/// that it was used, and it does not delete a single concern.
#[test]
fn the_kill_switch_downgrades_the_refusal_and_announces_itself() {
    let mut s = healthy();
    s.throttle_active.hw_thermal = Some(true);
    let blocked = precheck(Sensitivity::Speed, &s, options());
    let overridden = precheck(
        Sensitivity::Speed,
        &s,
        PolicyOptions {
            kill_switch: true,
            ..options()
        },
    );
    assert_eq!(blocked.decision, Decision::Refuse);
    assert_eq!(overridden.decision, Decision::Warn);
    for concern in &blocked.concerns {
        assert!(overridden.concerns.contains(concern));
    }
    assert!(
        overridden
            .concerns
            .iter()
            .any(|c| c.contains(KILL_SWITCH_ENV) && c.contains("SUPPRESSED")),
        "{overridden:?}"
    );
}

/// The kill switch must not be able to turn a warning into a pass.
#[test]
fn the_kill_switch_cannot_manufacture_a_proceed() {
    let opts = PolicyOptions {
        kill_switch: true,
        ..options()
    };
    let p = precheck(Sensitivity::Speed, &HardwareState::default(), opts);
    assert_eq!(p.decision, Decision::Warn);
}

#[test]
fn a_run_that_did_not_throttle_is_valid() {
    let p = postcheck(Sensitivity::Speed, &delta(Some(0), Some(0)), options());
    assert_eq!(p.validity, Validity::Valid);
}

/// ★ The measurement-validity question. The box started perfect and throttled
/// at minute 40; only the delta can say so.
#[test]
fn a_run_during_which_a_thermal_counter_advanced_is_invalid() {
    let p = postcheck(
        Sensitivity::Speed,
        &delta(Some(12_000_000), Some(0)),
        options(),
    );
    assert_eq!(p.validity, Validity::Invalid);
    assert!(
        p.concerns.iter().any(|c| c.contains("not comparable")),
        "{p:?}"
    );
    assert!(p.concerns.iter().any(|c| c.contains("1.73%")), "{p:?}");
}

/// Even the last-resort HW path on its own invalidates: dgx2 spent 228 s
/// there against dgx1's 0.7 s.
#[test]
fn hw_thermal_alone_invalidates() {
    let p = postcheck(Sensitivity::Speed, &delta(Some(0), Some(1)), options());
    assert_eq!(p.validity, Validity::Invalid);
}

/// ★ Unreadable counters are UNKNOWN, never Valid.
#[test]
fn unreadable_counters_leave_the_run_unknown_not_valid() {
    let p = postcheck(
        Sensitivity::Speed,
        &HardwareStateDelta::default(),
        options(),
    );
    assert_eq!(p.validity, Validity::Unknown);
    assert!(p.concerns.iter().any(|c| c.contains("not known")), "{p:?}");
}

#[test]
fn partially_unreadable_zero_counters_leave_the_run_unknown() {
    let p = postcheck(
        Sensitivity::Speed,
        &HardwareStateDelta {
            elapsed_s: Some(692),
            sw_thermal_us: Some(0),
            hw_thermal_us: None,
            hw_power_brake_us: None,
            ..HardwareStateDelta::default()
        },
        options(),
    );
    assert_eq!(p.validity, Validity::Unknown);
    assert_eq!(
        p.concerns,
        [
            "throttle counters were unreadable on at least one capture — this run is not known to have been unthrottled"
        ]
    );
}

/// A correctness number is never invalidated by thermals — and the concerns
/// are still recorded beside it.
#[test]
fn a_correctness_run_is_never_invalidated_but_still_reports() {
    let d = delta(Some(12_000_000), Some(500_000));
    let p = postcheck(Sensitivity::Correctness, &d, options());
    assert_eq!(p.validity, Validity::NotApplicable);
    assert!(p.concerns.iter().any(|c| c.contains("throttled")), "{p:?}");
}

/// The kill switch lets a run START. It does not make a throttled run valid —
/// an operator may decide to measure on a hot box; nobody gets to decide the
/// record should say the box was cool.
#[test]
fn the_kill_switch_cannot_validate_a_throttled_run() {
    let p = postcheck(
        Sensitivity::Speed,
        &delta(Some(12_000_000), Some(0)),
        PolicyOptions {
            kill_switch: true,
            ..options()
        },
    );
    assert_eq!(p.validity, Validity::Invalid);
}

#[test]
fn verdicts_round_trip_through_json_for_the_record() {
    let p = precheck(Sensitivity::Speed, &healthy(), options());
    let back: Precheck = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
    assert_eq!(p, back);
    let q = postcheck(Sensitivity::Speed, &delta(Some(1), None), options());
    let back: Postcheck = serde_json::from_str(&serde_json::to_string(&q).unwrap()).unwrap();
    assert_eq!(q, back);
}

/// Nothing is on by default. A gate that switches itself on because an env var
/// happens to be set to the empty string is a gate nobody asked for.
#[test]
fn no_option_is_on_unless_it_was_asked_for() {
    assert_eq!(
        PolicyOptions::default(),
        PolicyOptions {
            kill_switch: false,
            absolute_temp_gate: false
        }
    );
}

/// ★ The registry is the SSOT for which benchmarks are speed-sensitive: the
/// bit lives on the descriptor, so adding a benchmark is a compile error until
/// its author answers the question. This pins the answers the incident report
/// named, so a later edit to one of them is a visible test change rather than
/// a silent loss of gating.
#[test]
fn the_registry_classifies_every_benchmark_the_incident_named() {
    let expected = [
        ("quick-speed-bench", Sensitivity::Speed),
        ("decode-floor", Sensitivity::Speed),
        ("concurrency-sweep", Sensitivity::Speed),
        ("ttft-warm-gate", Sensitivity::Speed),
        ("ttft-cold-gate", Sensitivity::Speed),
        // Mixed, and classified by the half thermals can corrupt: its
        // `wall_budget_s` Sigma-wall bound is the number that was retracted.
        ("agentic-webserver", Sensitivity::Speed),
        ("bfcl-subset", Sensitivity::Correctness),
        ("bfcl-subset-echolp", Sensitivity::Correctness),
        ("bfcl-full", Sensitivity::Correctness),
        ("ssm-state-poisoning-gate", Sensitivity::Correctness),
        ("vision-fidelity", Sensitivity::Correctness),
        ("video-fidelity", Sensitivity::Correctness),
        ("cross-contamination", Sensitivity::Correctness),
    ];
    for (id, sensitivity) in expected {
        let d = crate::registry::find(id).unwrap_or_else(|| panic!("{id} left the registry"));
        assert_eq!(d.sensitivity, sensitivity, "{id}");
    }
    // Every registered benchmark is covered by the policy, whether or not it
    // is pinned above — the field is not optional.
    for d in crate::registry::all() {
        let _: Sensitivity = d.sensitivity;
    }
}

/// ★ The record must not quote an understated throttle as a figure.
///
/// `hardware_state.postcheck.concerns` is persisted with the gate record and is
/// the text a human reads when deciding whether a speed number is comparable.
/// With one counter unreadable the percentage is a floor, and the counter list
/// used to print `sw 0 µs` for a counter that was never read — a measured zero
/// invented out of an absent reading, in the one paragraph whose job is to say
/// how bad the throttle was.
#[test]
fn an_unreadable_counter_is_never_printed_as_a_measured_zero() {
    let p = postcheck(
        Sensitivity::Speed,
        &delta(None, Some(12_000_000)),
        options(),
    );
    assert_eq!(p.validity, Validity::Invalid);
    let joined = p.concerns.join("\n");
    assert!(
        joined.contains("sw unreadable"),
        "an unread counter must say so: {joined}"
    );
    assert!(
        !joined.contains("sw 0 µs"),
        "a never-read counter was recorded as a measured zero: {joined}"
    );
    assert!(
        joined.contains("AT LEAST 1.73%"),
        "a partial sum is a floor, not a figure: {joined}"
    );
    // The complete case keeps the plain wording, so the hedge is information
    // rather than boilerplate on every record.
    let full = postcheck(
        Sensitivity::Speed,
        &delta(Some(0), Some(12_000_000)),
        options(),
    );
    let full = full.concerns.join("\n");
    assert!(full.contains("throttled for 1.73% of the run"), "{full}");
    assert!(!full.contains("AT LEAST"), "{full}");
    assert!(
        full.contains("sw 0 µs"),
        "a real zero still reads as zero: {full}"
    );
}
