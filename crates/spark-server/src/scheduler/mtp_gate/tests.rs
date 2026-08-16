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
        g.record_verify_step(wall, emitted, 1);
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
        g.record_decode(wall, 1);
    }
}

/// Run MTP steps until the gate opens its first serial-refresh probe.
fn run_mtp_until_probe(g: &mut MtpGate, emitted: usize, wall: Duration) {
    for _ in 0..10_000 {
        if g.next_step() == GateStep::MeasureDecode {
            return;
        }
        g.record_verify_step(wall, emitted, 1);
    }
    panic!("gate never opened a serial probe");
}

/// Run serial steps until the gate opens an MTP re-probe.
fn run_serial_until_probe(g: &mut MtpGate, wall: Duration) {
    for _ in 0..10_000 {
        if g.next_step() == GateStep::MeasureVerify {
            return;
        }
        g.record_decode(wall, 1);
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
        g.record_verify_step(ms(100), 2, 1);
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
        g.record_verify_step(ms(50), 2, 1);
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
        g.record_verify_step(ms(100), 2, 1);
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
    assert!(g.maybe_remeasure(1300));
    assert_eq!(g.regime_reprobe_count(), 1);
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
    // Extend: the early probe is ONE window. A 300–1000 token think
    // (3.8 budget / #517-shaped length) must not be dominated by serial.
    // Probe serial at 20 tok/s vs MTP 40 — stay MTP after the window.
    drive_serial(&mut g, WINDOW_STEPS, ms(50));
    assert_eq!(g.next_step(), GateStep::MeasureVerify);
    assert!(!g.in_serial_mode());
    let mut serial_steps = WINDOW_STEPS;
    let mut mtp_steps = WINDOW_STEPS * 2 + 1;
    let mut tokens = (WINDOW_STEPS * 2 + 1) * 2 + WINDOW_STEPS;
    while tokens < 1000 {
        match g.next_step() {
            GateStep::MeasureVerify => {
                g.record_verify_step(ms(50), 2, 1);
                mtp_steps += 1;
                tokens += 2;
            }
            GateStep::MeasureDecode => {
                g.record_decode(ms(50), 1);
                serial_steps += 1;
                tokens += 1;
            }
        }
    }
    let serial_frac = serial_steps as f64 / (serial_steps + mtp_steps) as f64;
    assert!(
        serial_steps <= WINDOW_STEPS,
        "regime change must not add serial beyond the one-window probe ({serial_steps})"
    );
    assert!(
        serial_frac < 0.05,
        "serial must not dominate a 1000-token turn (frac={serial_frac:.3})"
    );
    assert!(!g.in_serial_mode());
}

#[test]
fn bootstrap_steps_count_at_least_one_token() {
    let mut g = MtpGate::new(1);
    // emitted=0 must not divide-by-zero or record zero-token windows.
    for _ in 0..WINDOW_STEPS {
        g.record_verify_step(ms(10), 0, 1);
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
        g.record_verify_step(ms(100), 2, 1);
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
        g.record_verify_step(ms(100), 2, 1);
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

#[test]
fn stale_other_baseline_cannot_steal_mode() {
    let mut g = MtpGate::new(1);
    run_mtp_until_probe(&mut g, 2, ms(50));
    drive_serial(&mut g, WINDOW_STEPS, ms(40));
    assert!(!g.in_serial_mode());
    assert!(g.serial_tps_debug().is_some());
    // Regime change: serial EWMA is stale. Degrade MTP to 10 tok/s —
    // faster than switching onto the stale 25 tok/s serial baseline.
    assert!(g.maybe_remeasure(2000));
    for _ in 0..(WINDOW_STEPS * SWITCH_DWELL_WINDOWS * 2) {
        if g.next_step() != GateStep::MeasureVerify {
            break;
        }
        g.record_verify_step(ms(200), 2, 1);
    }
    assert!(
        !g.in_serial_mode(),
        "a stale serial baseline must not win a switch after a depth-regime change"
    );
    assert_eq!(g.take_fresh_decision(), None);
}

#[test]
fn standard_mtp_stays_serial_in_think() {
    assert!(!spec_dispatch_eligible(
        true, 0, 0, false, false, false, 0, false
    ));
    assert!(!spec_dispatch_eligible(
        true, 0, 50, false, false, false, 0, false
    ));
    assert!(spec_dispatch_eligible(
        false, 0, 50, false, false, false, 0, false
    ));
}

#[test]
fn standard_mtp_spec_think_opts_in() {
    assert!(spec_dispatch_eligible(
        true, 0, 50, false, false, true, 0, false
    ));
}

#[test]
fn dflash_raw_argmax_stays_serial_in_think() {
    assert!(!spec_dispatch_eligible(
        true, 0, 50, false, false, false, 0, true
    ));
    assert!(spec_dispatch_eligible(
        false, 0, 50, false, false, false, 0, true
    ));
}

#[test]
fn dflash_spec_think_opts_in() {
    assert!(spec_dispatch_eligible(
        true, 0, 0, false, false, true, 0, true
    ));
}

/// 3.8 `max_thinking_budget = 2048` can cross the floor twice.
/// Each crossing may open the shipped one-window probe. Serial must
/// stay a measurement phase, not the mode.
#[test]
fn think_budget_2048_two_crossings_do_not_dump_serial() {
    let mut g = MtpGate::new(1);
    let mut depth = 64usize;
    g.note_depth(depth);
    let mut serial_steps = 0usize;
    let mut mtp_steps = 0usize;
    let mut tokens = 0usize;
    while tokens < 2048 {
        g.note_depth(depth);
        let _ = g.maybe_remeasure(depth);
        match g.next_step() {
            GateStep::MeasureVerify => {
                g.record_verify_step(ms(50), 2, 1);
                mtp_steps += 1;
                tokens += 2;
                depth += 2;
            }
            GateStep::MeasureDecode => {
                g.record_decode(ms(50), 1);
                serial_steps += 1;
                tokens += 1;
                depth += 1;
            }
        }
    }
    let serial_frac = serial_steps as f64 / (serial_steps + mtp_steps) as f64;
    assert_eq!(g.regime_reprobe_count(), 2);
    assert!(
        serial_steps <= WINDOW_STEPS * 3,
        "at most one-window probe per crossing plus shipped refresh, got {serial_steps}"
    );
    assert!(
        serial_frac < 0.08,
        "serial must not dominate a 2048-token think (frac={serial_frac:.3})"
    );
    assert!(!g.in_serial_mode());
}

// ---- batch-width accounting (2026-08-15 decode-side fix) --------------------

/// A plain decode step over n sequences emits n tokens. Before the fix,
/// `record_decode` charged 1 token regardless of width, so a serial probe at
/// batch 4 read ~4× slower than delivered while the verify side was already
/// multi-seq summed — DisableMtp was unreachable exactly under concurrency.
#[test]
fn decode_steps_charge_the_batch_width() {
    let mut g = MtpGate::new(1);
    run_mtp_until_probe(&mut g, 2, ms(50));
    // Serial probe at batch width 4: 4 tokens per 10ms step = 400 tok/s.
    // (The width-1 → width-4 regime change lands on the probe's first,
    // still-empty window, so all WINDOW_STEPS steps measure in-regime.)
    for _ in 0..WINDOW_STEPS {
        assert_eq!(g.next_step(), GateStep::MeasureDecode);
        g.record_decode(ms(10), 4);
    }
    let serial = g.serial_tps_debug().expect("probe closed a window");
    assert!(
        (serial - 400.0).abs() < 1.0,
        "serial EWMA must read the delivered 400 tok/s, not the pre-fix \
         one-token 100 tok/s (got {serial:.1})"
    );
}

/// Drain-tail graph borrowing (spark-model `graph_borrow.rs`) may replay a
/// WIDER captured CUDA graph for a shrinking batch — the wall then includes
/// pad-lane compute. The arbiter stays honest only if the scheduler keeps
/// charging steps at the ACTIVE width (tokens actually delivered), never at
/// any padded width: the model layer hides padding entirely, so the call
/// sites must pass `active.len()`.
///
/// PROVEN BY: rewriting the decode charge in `scheduler/mod.rs` to anything
/// other than `active.len()` turns this red.
#[test]
fn arbiter_charges_active_width_never_a_padded_width() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/scheduler/mod.rs"),
    )
    .unwrap();
    assert!(
        src.contains("gate.record_decode(t0.elapsed(), active.len())"),
        "decode steps must be charged at the active batch width"
    );
    for call in src.split("gate.record_decode(").skip(1) {
        let args = call.split(';').next().unwrap_or("");
        assert!(
            args.contains("active.len()"),
            "every decode charge must use active.len(), got `record_decode({args})`"
        );
    }
}

/// A width-regime change is a depth-regime change's twin: EWMAs measured at
/// batch 1 must not arbitrate against windows measured at batch 8. The
/// mixed-width partial window is discarded, and — like a depth change — the
/// probe cadence is pulled forward, so the first in-regime step closes a
/// one-step window that REPLACES the current mode's EWMA and the very next
/// step probes the other mode at the new width. Arbitration therefore only
/// ever compares two same-regime numbers.
#[test]
fn width_regime_change_stales_and_remeasures_in_the_new_regime() {
    let mut g = MtpGate::new(1);
    // Both baselines at width 1; gate stays Mtp (40 vs 25 tok/s).
    run_mtp_until_probe(&mut g, 2, ms(50));
    drive_serial(&mut g, WINDOW_STEPS, ms(40));
    assert!(!g.in_serial_mode());
    // Half a window at width 1, then the batch widens into bucket 8.
    drive_mtp(&mut g, WINDOW_STEPS / 2, 2, ms(50));
    g.record_verify_step(ms(5), 6, 8);
    assert!(
        g.serial.stale,
        "off-mode baseline is stale after the change"
    );
    let mtp = g.mtp_tps_debug().expect("one-step window closed");
    assert!(
        (mtp - 1200.0).abs() < 1.0,
        "Mtp EWMA must be REPLACED by the in-regime window (6 tok / 5 ms = \
         1200 tok/s), not blended with the width-1 40 tok/s (got {mtp:.0})"
    );
    // The early probe opens immediately, measuring serial at the new width.
    assert_eq!(
        g.next_step(),
        GateStep::MeasureDecode,
        "probe pulled forward"
    );
    for _ in 0..WINDOW_STEPS {
        g.record_decode(ms(10), 8); // 800 tok/s in-regime
    }
    assert!(
        !g.serial.stale,
        "probe re-measured serial in the new regime"
    );
    // 800 < 1200: stays Mtp. Had the stale width-1 numbers survived, serial
    // (probe window vs the old 40 tok/s EWMA) would have looked 20x faster.
    assert!(!g.in_serial_mode());
    assert_eq!(g.take_fresh_decision(), None);
}

/// Per-step width churn INSIDE a power-of-two bucket is scheduler noise
/// (join/leave of one sequence), not a regime change — it must blend into
/// the EWMA, not thrash the baselines stale on every step.
#[test]
fn width_jitter_inside_a_bucket_does_not_stale() {
    let mut g = MtpGate::new(1);
    // Widths 5..=8 share bucket 8.
    for w in [5usize, 6, 7, 8, 6, 5, 7] {
        g.record_verify_step(ms(10), 2 * w, w);
    }
    assert!(
        !g.mtp.stale && !g.serial.stale,
        "same-bucket churn is noise"
    );
    assert_eq!(g.win_steps, 7, "nothing discarded inside the bucket");
    // Dropping to bucket 4 IS a regime change: partial window discarded,
    // the step re-measures Mtp in-regime (probe-forward closes it), and the
    // never-measured serial side is left stale until its probe.
    g.record_verify_step(ms(10), 8, 4);
    assert!(g.serial.stale && !g.mtp.stale);
    assert_eq!(
        g.win_steps, 0,
        "mixed-width window discarded, fresh one closed"
    );
    assert_eq!(
        g.next_step(),
        GateStep::MeasureDecode,
        "probe pulled forward"
    );
}
