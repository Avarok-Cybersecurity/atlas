// SPDX-License-Identifier: AGPL-3.0-only

//! The per-rung INSTRUMENT keys the concurrency sweep publishes: the
//! speculation arm (#835), both ITL clocks, the arrival-gap jitter
//! distribution, and the GPU-rail energy window.
//!
//! Split from `concurrency_tests.rs` for the 500-LoC cap when the ITL and
//! energy keys landed. Exact piecewise copy — no test changed in the move.
//! It is a CHILD of that module rather than a sibling so the fixture
//! builders (`configured`, `row`, `evidence`, `evidence_with_arm`) stay
//! single-sourced; a sibling would have to duplicate them.

use super::*;

// ── THE ARM PIN (#835) ───────────────────────────────────────────────────
//
// `c2_aggregate_tok_s` is trimodal (~30.6 / ~27.5 / ~23.5 tok/s, nothing
// between) because the C=2 cell's ~640 measured tokens contain only one to
// three MTP-gate arbitrations, so its outcome is all-MTP, mixed, or
// all-serial. The committed floor sits in the empty gap, which makes it a
// mode detector rather than a regression detector.
//
// ★ CORRECTED after a real campaign. The arm was first wired as a COMPARABILITY
// class — a serial cell excluded from scoring, and any such cell failing the run
// INCONCLUSIVE. That was wrong, and only a live gate run showed it: at wide batch
// the MTP gate drops speculation ON PURPOSE, so C=8 upward legitimately run
// serial, and the floors for those rungs were calibrated on runs that did
// exactly that. The pin dropped five of eight cells, made `peak_aggregate_tok_s`
// read 49.5 (from C=4) instead of ~115 (from C=64), and failed a gate with nine
// consecutive passing records.
//
// The arm is now PUBLISHED and never gated on: the benchmark cannot know which
// arm should have run, and asserting otherwise is a claim it cannot support.
// What #835 needs is that a human reading a low C=2 can see whether the serial
// arm explains it — and a metric on the record does that.

#[test]
fn a_serial_arm_cell_is_reported_but_still_scored() {
    // accept_len == 1.00 exactly: every emitted token cost one step.
    let serial = row(
        2,
        23.5,
        Some(2000.0),
        vec![evidence_with_arm(320, Some(0)); 2],
        320,
    );
    assert_eq!(serial.accept_len(), Some(1.0));
    assert!(serial.arm_is_not_mtp(), "accept_len 1.00 is the serial arm");
    // ★ THE CORRECTION. It stays comparable. At wide batch the gate drops
    // speculation deliberately and those floors were calibrated that way, so
    // excluding a serial cell throws away a measurement the floor expects.
    assert!(
        serial.comparable(),
        "a serial cell is still comparable to a floor calibrated on serial"
    );
    assert!(!serial.vacuous);
    assert!(!serial.cache_uncontrolled);
}

#[test]
fn an_mtp_arm_cell_is_comparable() {
    let mtp = row(2, 30.6, Some(2000.0), vec![evidence(320); 2], 320);
    let a = mtp.accept_len().expect("accept_len derivable");
    assert!(
        a > 1.5,
        "the MTP arm sits well above the 1.5 threshold, got {a}"
    );
    assert!(!mtp.arm_is_not_mtp());
    assert!(mtp.comparable());
}

#[test]
fn a_mixed_cell_reports_the_minimum_arm_and_is_still_scored() {
    // One request served serial, one speculative. The MINIMUM governs the
    // reported figure, so a cell that touched the serial arm at all says so —
    // that is the signal a human needs to explain a low reading.
    let mixed = row(
        2,
        27.5,
        Some(2000.0),
        vec![evidence_with_arm(320, Some(0)), evidence(320)],
        320,
    );
    assert_eq!(mixed.accept_len(), Some(1.0), "the minimum, not the mean");
    assert!(
        mixed.comparable(),
        "reporting the arm must not remove the cell from scoring"
    );
}

#[test]
fn a_missing_accept_field_is_not_read_as_serial() {
    // ★ THE TRAP. `None` means the server did not report the field. Treating
    // it as 1.00 would convert a missing instrument into a verdict about the
    // engine — and every pre-existing recorded sweep has no accept field.
    let unknown = row(
        2,
        27.5,
        Some(2000.0),
        vec![evidence_with_arm(320, None); 2],
        320,
    );
    assert_eq!(unknown.accept_len(), None);
    assert!(
        !unknown.arm_is_not_mtp(),
        "an unreported arm is unknown, not serial"
    );
    assert!(
        unknown.comparable(),
        "a sweep from before this instrument existed must stay comparable"
    );
}

#[test]
fn a_corrupt_accept_count_does_not_divide_by_zero() {
    // accepted >= completion is impossible on the wire but must not panic or
    // produce a negative/infinite accept depth if it ever appears.
    let corrupt = row(
        2,
        27.5,
        Some(2000.0),
        vec![evidence_with_arm(320, Some(320)); 2],
        320,
    );
    assert_eq!(corrupt.accept_len(), None);
    assert!(!corrupt.arm_is_not_mtp());
}

// ---- instrument keys: both ITL clocks, jitter, energy --------------------

/// Every rung carries its client-clock and server-clock ITL under the
/// repo's existing `tpot` name (no parallel `itl_*` key), the pooled
/// arrival-gap distribution, and the batch's joules WITH the tokens they
/// were spent on. A clock the server did not report is absent, not zero.
#[test]
fn metrics_map_carries_both_itl_clocks_jitter_and_energy_per_rung() {
    let osl = 128;
    let mut b = configured(vec![4], vec![512]);
    b.osl = osl;
    let mut r = row(4, 100.0, Some(150.0), vec![evidence(128); 4], osl);
    r.tpot = Percentiles {
        p50: Some(31.0),
        p90: Some(35.0),
        p99: Some(40.0),
    };
    r.server_tpot = Percentiles {
        p50: Some(29.5),
        p90: Some(33.0),
        p99: Some(38.0),
    };
    let mut gaps = GapSample::default();
    for g in [30.0, 30.0, 30.0, 31.0, 30.0, 30.0, 30.0, 30.0, 30.0, 180.0] {
        gaps.push(g);
    }
    r.gaps = gaps.stats();
    r.energy = Some(EnergyWindow {
        window_s: 12.0,
        samples: 48,
        energy_j: 720.0,
        mean_power_w: 60.0,
        max_power_w: 66.0,
        sw_power_cap_frac: Some(1.0),
        hw_power_brake_frac: Some(0.0),
    });
    b.rows.push(r);
    let m = b.metrics();
    assert_eq!(m.get("c4_tpot_p50_ms"), Some(&31.0));
    assert_eq!(m.get("c4_tpot_p90_ms"), Some(&35.0));
    assert_eq!(m.get("c4_server_tpot_p50_ms"), Some(&29.5));
    assert_eq!(m.get("c4_server_tpot_p90_ms"), Some(&33.0));
    assert!(
        !m.keys().any(|k| k.contains("itl")),
        "no parallel itl_* key: {m:?}"
    );
    assert_eq!(m.get("c4_arrival_gap_count"), Some(&10.0));
    assert_eq!(m.get("c4_arrival_gap_max_ms"), Some(&180.0));
    assert_eq!(m.get("c4_arrival_gap_p50_ms"), Some(&30.0));
    assert_eq!(m.get("c4_arrival_gap_p99_ms"), Some(&180.0));
    assert_eq!(
        m.get("c4_stability"),
        Some(&5.0),
        "lower is better; a stall raises it"
    );
    assert!(m.contains_key("c4_arrival_gap_cv"));
    assert_eq!(m.get("c4_gpu_rail_energy_j"), Some(&720.0));
    assert_eq!(m.get("c4_gpu_rail_power_samples"), Some(&48.0));
    assert_eq!(m.get("c4_gpu_rail_energy_window_tokens"), Some(&512.0));
    assert_eq!(m.get("c4_gpu_rail_sw_power_cap_frac"), Some(&1.0));
    // No idle baseline was taken → no above-idle key, never a zero.
    assert!(!m.contains_key("c4_gpu_rail_energy_above_idle_j"));
    assert!(!m.contains_key("gpu_rail_idle_power_w"));

    // A rung measured against an older server: the client clock is there,
    // the server clock and the instruments are simply absent.
    let mut old = configured(vec![2], vec![512]);
    old.osl = osl;
    let mut r = row(2, 30.0, Some(120.0), vec![evidence(128); 2], osl);
    r.tpot = Percentiles {
        p50: Some(31.0),
        p90: Some(35.0),
        p99: Some(40.0),
    };
    old.rows.push(r);
    let m = old.metrics();
    assert_eq!(m.get("c2_tpot_p50_ms"), Some(&31.0));
    assert!(!m.contains_key("c2_server_tpot_p50_ms"));
    assert!(
        !m.keys()
            .any(|k| k.contains("arrival_gap") || k.contains("gpu_rail"))
    );
}
