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

// ---------------------------------------------------------------------------
// Regression: partial windows must never replace a baseline.
//
// Production failure (2026-07-20, W4A4 27B @16k): `maybe_remeasure` marks BOTH
// modes stale AND forces `tokens_since_event` to the probe interval. The very
// next step therefore satisfies the early-close condition in `record_step`, so
// `close_window` runs with `win_steps == 1`; because `stale` is set, `replace`
// is true and that ONE step became the mode's whole baseline. The step that
// lands there is an MTP bootstrap (1 token for a full step's wall), so the gate
// recorded MTP at 6.8 tok/s against a true 18.3-18.4 and switched to Serial for
// the rest of the session.
//
// Walls below are the measured ones: MTP 2 tok / 109ms = 18.35 tok/s steady,
// bootstrap 1 tok / 147ms = 6.80 tok/s, serial 1 tok / 82ms = 12.20 tok/s.
// ---------------------------------------------------------------------------

const MTP_STEADY_WALL: Duration = Duration::from_millis(109); // 2 tok => 18.35 tok/s
const BOOTSTRAP_WALL: Duration = Duration::from_millis(147); //  1 tok =>  6.80 tok/s
const SERIAL_WALL: Duration = Duration::from_millis(82); //  1 tok => 12.20 tok/s

#[test]
fn stale_partial_window_does_not_replace_baseline() {
    let mut g = MtpGate::new(2);
    g.note_depth(600);
    drive_mtp(&mut g, WINDOW_STEPS * 2, 2, MTP_STEADY_WALL);
    let good = g.mtp_tps_debug().expect("baseline established");
    assert!(
        (good - 18.35).abs() < 0.1,
        "precondition: steady MTP baseline, got {good}"
    );

    // Depth regime changes -> both baselines stale, probe forced due.
    g.maybe_remeasure(2400);
    // The single bootstrap-heavy step that the early-close path turns into a
    // whole "window". This is the poison pill.
    drive_mtp(&mut g, 1, 1, BOOTSTRAP_WALL);

    let after = g.mtp_tps_debug().expect("baseline must survive");
    assert!(
        (after - good).abs() < 1e-9,
        "a 1-step window replaced the MTP baseline: {good} -> {after} \
         (the 6.8-vs-18.3 production bug)"
    );
}

#[test]
fn stale_replace_still_happens_on_the_next_full_window() {
    // Discarding the partial window must not lose the regime change: the
    // NEXT full window still REPLACES (not blends), which is the whole point
    // of the stale flag.
    let mut g = MtpGate::new(2);
    g.note_depth(600);
    drive_mtp(&mut g, WINDOW_STEPS * 2, 2, MTP_STEADY_WALL);
    g.maybe_remeasure(2400);
    drive_mtp(&mut g, 1, 1, BOOTSTRAP_WALL); // discarded

    // The regime change schedules an off-mode probe, so the gate asks for a
    // serial window next; serve it before MTP resumes.
    drive_serial(&mut g, WINDOW_STEPS, SERIAL_WALL);

    // A full window at a genuinely different rate: 2 tok / 200ms = 10 tok/s.
    drive_mtp(&mut g, WINDOW_STEPS, 2, Duration::from_millis(200));
    let tps = g.mtp_tps_debug().expect("updated");
    assert!(
        (tps - 10.0).abs() < 0.05,
        "stale flag must survive a discarded window and force a REPLACE \
         (expected 10.0 tok/s, blended would be ~15.8); got {tps}"
    );
}

#[test]
fn regime_change_does_not_flip_a_winning_mtp_to_serial() {
    // End-to-end shape of the production failure: MTP is genuinely faster
    // (18.35 vs 12.20) and must still be the mode after a depth-regime change.
    let mut g = MtpGate::new(2);
    g.note_depth(600);

    // Establish both baselines: MTP windows, then the scheduled serial probe.
    run_mtp_until_probe(&mut g, 2, MTP_STEADY_WALL);
    drive_serial(&mut g, WINDOW_STEPS, SERIAL_WALL);
    assert!(!g.in_serial_mode(), "precondition: MTP is the faster mode");

    // Depth regime change, then the bootstrap step, then the forced probe.
    g.maybe_remeasure(2400);
    drive_mtp(&mut g, 1, 1, BOOTSTRAP_WALL);
    drive_serial(&mut g, WINDOW_STEPS, SERIAL_WALL);
    // And MTP continues at its true rate.
    drive_mtp(&mut g, WINDOW_STEPS * 2, 2, MTP_STEADY_WALL);

    let mtp = g.mtp_tps_debug().expect("mtp baseline");
    assert!(
        mtp > 15.0,
        "MTP baseline must reflect its true 18.35 tok/s, got {mtp}"
    );
    assert!(
        !g.in_serial_mode(),
        "gate switched to Serial while MTP was 1.5x faster (mtp={mtp}, \
         serial={:?}) — the 2026-07-20 regression",
        g.serial_tps_debug()
    );
}
