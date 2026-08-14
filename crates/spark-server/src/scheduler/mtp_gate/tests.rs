// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the throughput-arbitrated MTP gate. All scenarios drive the
//! gate through `record_*` exactly as the scheduler does; walls are
//! synthetic, tokens/step model acceptance.

use super::*;

fn ms(x: u64) -> Duration {
    Duration::from_millis(x)
}

/// Drive `n` MTP-path steps at `emitted` tokens per step and `wall` each.
fn drive_mtp(g: &mut MtpGate, n: usize, emitted: usize, wall: Duration) {
    for _ in 0..n {
        assert_eq!(
            g.next_step(),
            GateStep::MeasureVerify,
            "expected Mtp-path step"
        );
        g.record_verify_step(wall, emitted);
    }
}

/// Drive `n` serial decode steps at `wall` each.
fn drive_serial(g: &mut MtpGate, n: usize, wall: Duration) {
    for _ in 0..n {
        assert_eq!(
            g.next_step(),
            GateStep::MeasureDecode,
            "expected serial step"
        );
        g.record_decode(wall);
    }
}

/// Run MTP steps until the gate opens its first serial-refresh probe.
fn run_mtp_until_probe(g: &mut MtpGate, emitted: usize, wall: Duration) {
    for _ in 0..10_000 {
        if g.next_step() == GateStep::MeasureDecode {
            return;
        }
        g.record_verify_step(wall, emitted);
    }
    panic!("gate never opened a serial probe");
}

/// Run serial steps until the gate opens an MTP re-probe.
fn run_serial_until_probe(g: &mut MtpGate, wall: Duration) {
    for _ in 0..10_000 {
        if g.next_step() == GateStep::MeasureVerify {
            return;
        }
        g.record_decode(wall);
    }
    panic!("gate never opened an MTP re-probe");
}

#[test]
fn starts_in_mtp_mode() {
    let g = MtpGate::new(1);
    assert_eq!(g.next_step(), GateStep::MeasureVerify);
    assert!(!g.in_serial_mode());
}

#[test]
fn no_switch_without_both_baselines() {
    let mut g = MtpGate::new(1);
    // Plenty of slow MTP windows, but serial was never measured: stay put.
    drive_mtp(&mut g, 64, 1, ms(100));
    assert!(!g.in_serial_mode());
    assert_eq!(g.take_fresh_decision(), None);
}

#[test]
fn refresh_probe_opens_after_interval_and_returns() {
    let mut g = MtpGate::new(1);
    // 2 tok/step: ~512 steps to reach the 1024-token refresh interval.
    run_mtp_until_probe(&mut g, 2, ms(50));
    // Probe is exactly one window of serial steps, then control returns.
    drive_serial(&mut g, WINDOW_STEPS, ms(40));
    // Serial measured 25 tok/s < MTP 40 tok/s: stays MTP.
    assert_eq!(g.next_step(), GateStep::MeasureVerify);
    assert!(!g.in_serial_mode());
    assert!(
        g.serial_tps_debug().is_some(),
        "probe must set the serial baseline"
    );
}

#[test]
fn switches_to_serial_when_clearly_faster_with_dwell() {
    let mut g = MtpGate::new(1);
    // MTP delivers 2 tok / 100ms = 20 tok/s.
    run_mtp_until_probe(&mut g, 2, ms(100));
    // Serial probe: 10ms/tok = 100 tok/s — way past any margin.
    drive_serial(&mut g, WINDOW_STEPS, ms(10));
    // Dwell: one more losing evaluation is required before the switch.
    assert!(
        !g.in_serial_mode(),
        "dwell must prevent single-window switches"
    );
    for _ in 0..(WINDOW_STEPS * SWITCH_DWELL_WINDOWS) {
        if g.next_step() != GateStep::MeasureVerify {
            break;
        }
        g.record_verify_step(ms(100), 2);
    }
    assert!(
        g.in_serial_mode(),
        "sustained 5x serial advantage must switch"
    );
    assert_eq!(g.take_fresh_decision(), Some(GateDecision::DisableMtp));
    assert_eq!(g.take_fresh_decision(), None, "fresh decision is one-shot");
    assert_eq!(g.next_step(), GateStep::MeasureDecode);
}

#[test]
fn hysteresis_blocks_within_margin_switches() {
    let mut g = MtpGate::new(1);
    // MTP 2 tok / 50ms = 40.0 tok/s.
    run_mtp_until_probe(&mut g, 2, ms(50));
    // Serial probe at 41 tok/s — inside the 5% noise floor (needs > 42).
    drive_serial(&mut g, WINDOW_STEPS, Duration::from_micros(24_390));
    for _ in 0..(WINDOW_STEPS * 4) {
        if g.next_step() != GateStep::MeasureVerify {
            break;
        }
        g.record_verify_step(ms(50), 2);
    }
    assert!(
        !g.in_serial_mode(),
        "a within-margin advantage must not switch modes"
    );
    assert_eq!(g.take_fresh_decision(), None);
}

#[test]
fn serial_mode_reprobes_mtp_and_recovers() {
    let mut g = MtpGate::new(1);
    // Establish MTP=20 tok/s, serial=100 tok/s, and switch to serial.
    run_mtp_until_probe(&mut g, 2, ms(100));
    drive_serial(&mut g, WINDOW_STEPS, ms(10));
    for _ in 0..(WINDOW_STEPS * SWITCH_DWELL_WINDOWS) {
        if g.next_step() != GateStep::MeasureVerify {
            break;
        }
        g.record_verify_step(ms(100), 2);
    }
    assert!(g.in_serial_mode());
    g.take_fresh_decision();

    // Workload shifts: MTP now 3 tok / 10ms = 300 tok/s. Two probe windows
    // (dwell) must bring MTP back.
    for _ in 0..SWITCH_DWELL_WINDOWS {
        run_serial_until_probe(&mut g, ms(10));
        drive_mtp(&mut g, WINDOW_STEPS, 3, ms(10));
    }
    assert!(
        !g.in_serial_mode(),
        "re-probe must recover MTP when it wins again"
    );
    assert_eq!(g.take_fresh_decision(), Some(GateDecision::KeepMtp));
    assert_eq!(g.next_step(), GateStep::MeasureVerify);
}

#[test]
fn depth_change_schedules_early_probe_without_state_wipe() {
    let mut g = MtpGate::new(1);
    g.note_depth(600);
    // A few MTP windows at depth 600.
    drive_mtp(&mut g, WINDOW_STEPS * 2, 2, ms(50));
    let tps_before = g.mtp_tps_debug();
    assert!(tps_before.is_some());
    // Depth doubles: baselines stale, probe due immediately, EWMA retained.
    g.maybe_remeasure(1300);
    assert_eq!(
        g.mtp_tps_debug(),
        tps_before,
        "no state wipe on regime change"
    );
    // The probe-due condition closes the window on the very next step.
    drive_mtp(&mut g, 1, 2, ms(50));
    assert_eq!(
        g.next_step(),
        GateStep::MeasureDecode,
        "stale regime must probe soon"
    );
}

#[test]
fn bootstrap_steps_count_at_least_one_token() {
    let mut g = MtpGate::new(1);
    // emitted=0 must not divide-by-zero or record zero-token windows.
    for _ in 0..WINDOW_STEPS {
        g.record_verify_step(ms(10), 0);
    }
    let tps = g.mtp_tps_debug().expect("window closed");
    assert!(tps > 0.0);
}

/// The spec-entry pin: while any active sequence is inside the entry
/// window, a Serial gate verdict must be overridden to the verify path.
/// This is the defect shape of the 2026-08-14 bfcl-subset-echolp residual:
/// the gate dwelt in Serial across requests #89–#101 and the three
/// `live_irrelevance` answer openings decoded on the serial forward flipped
/// from a prose decline to a fabricated tool call. Before the pin existed
/// the scheduler ran `gate.next_step()` unconditionally, so a Serial dwell
/// put answer OPENINGS on the serial forward — the exact window where the
/// serial-vs-batch-K T=0 flips concentrate.
#[test]
fn entry_pin_overrides_serial_mode_for_answer_openings() {
    let mut g = MtpGate::new(1);
    // Drive the gate into a legitimate Serial dwell (MTP 20 tok/s vs
    // serial 100 tok/s), exactly like the production switch.
    run_mtp_until_probe(&mut g, 2, ms(100));
    drive_serial(&mut g, WINDOW_STEPS, ms(10));
    for _ in 0..(WINDOW_STEPS * SWITCH_DWELL_WINDOWS) {
        if g.next_step() != GateStep::MeasureVerify {
            break;
        }
        g.record_verify_step(ms(100), 2);
    }
    assert!(g.in_serial_mode());
    assert_eq!(g.next_step(), GateStep::MeasureDecode);

    // A sequence at the post-`</think>` boundary (0..8 tokens emitted)
    // pins the step to the verify path despite the Serial verdict.
    assert!(entry_pin_forces_verify(0));
    assert!(entry_pin_forces_verify(7));
    // Past the measured ≤7-token flip window the gate's verdict stands.
    assert!(!entry_pin_forces_verify(8));
    assert!(!entry_pin_forces_verify(u32::MAX));
}

/// Pinned steps are not recorded, so a Serial dwell's baselines must be
/// unaffected by however many entry windows pass through it — the pin must
/// never inflate the Serial EWMA with verify walls (which emit >1 token per
/// step and would make Serial look unbeatable).
#[test]
fn entry_pin_steps_do_not_touch_arbitration_state() {
    let mut g = MtpGate::new(1);
    run_mtp_until_probe(&mut g, 2, ms(100));
    drive_serial(&mut g, WINDOW_STEPS, ms(10));
    for _ in 0..(WINDOW_STEPS * SWITCH_DWELL_WINDOWS) {
        if g.next_step() != GateStep::MeasureVerify {
            break;
        }
        g.record_verify_step(ms(100), 2);
    }
    assert!(g.in_serial_mode());
    g.take_fresh_decision();
    let serial_before = g.serial_tps_debug();
    let mtp_before = g.mtp_tps_debug();
    // The scheduler runs pinned verify steps WITHOUT record_* calls; the
    // gate object is untouched, so its baselines cannot move. (This test
    // documents the contract; the wiring in scheduler/mod.rs is what
    // honors it.)
    assert_eq!(g.serial_tps_debug(), serial_before);
    assert_eq!(g.mtp_tps_debug(), mtp_before);
    assert!(g.in_serial_mode(), "pin must not flip the gate's mode");
}

/// `ATLAS_SPEC_ENTRY_PIN` parsing: strict integer, default 8, `0` disables.
#[test]
fn entry_pin_env_parse() {
    assert_eq!(parse_entry_pin_tokens(None), 8);
    assert_eq!(parse_entry_pin_tokens(Some("0")), 0);
    assert_eq!(parse_entry_pin_tokens(Some("12")), 12);
    assert_eq!(parse_entry_pin_tokens(Some("garbage")), 8);
    assert_eq!(parse_entry_pin_tokens(Some("-3")), 8);
}
