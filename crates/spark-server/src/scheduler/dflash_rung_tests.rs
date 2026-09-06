// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! Unit tests for the DFlash gamma resolver. The 2026-09-03 receipts are the
//! anchor: the rule must land every measured winner. The pure rules are
//! tested with explicit [`Rungs`] (no environment); the controller tests
//! drive the REAL `configure_with` / `drafts_for` / `observe_step` path and
//! serialize on one lock because the controller is process-global.

use std::sync::Mutex;

use super::*;

/// The controller is one static; tests that touch it run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn r10() -> Rungs {
    Rungs::defaults(10)
}

#[test]
fn concurrency_ladder_lands_the_records() {
    let r = r10();
    // C=16 prose 187.7 and code 269.7 are both K=4 (write-on-accept).
    assert_eq!(k_for(16, true, &r), 4);
    assert_eq!(k_for(16, false, &r), 4);
    assert_eq!(k_for(2, true, &r), 4);
    // C=1 code 66.8 is the full block; C=1 prose 27.0 is K=5.
    assert_eq!(k_for(1, true, &r), 10);
    assert_eq!(k_for(1, false, &r), 5);
}

#[test]
fn widths_never_exceed_the_cap() {
    // A head sized for K=4 (record launch) cannot be asked for K=5 or K=10.
    let r4 = Rungs::defaults(4);
    assert_eq!(k_for(1, true, &r4), 4);
    assert_eq!(k_for(1, false, &r4), 4);
    assert_eq!(k_for(16, true, &Rungs::defaults(3)), 3);
}

/// Question 3 of the review: with the write-on-accept kernel off
/// (`ATLAS_NO_GDN_WOA=1`) there is no receipt for a K=4 parent-kernel serve,
/// so the C>=2 rung falls back to the cap; C=1 is unaffected.
#[test]
fn no_woa_pins_the_multi_rung_at_the_cap() {
    let r = Rungs::from_env(10, false);
    assert_eq!(r.multi, 10);
    assert_eq!(r.narrow, Rungs::defaults(10).narrow);
    assert_eq!(r.wide, 10);
    assert_eq!(k_for(16, true, &r), 10);
    assert_eq!(k_for(1, false, &r), 5);
}

#[test]
fn hysteresis_separates_prose_from_code() {
    let r = r10();
    // Measured p1 on the narrow rung: prose ~0.76 (49/64), code ~0.97 (62/64);
    // on the wide rung code drops to ~0.875 (56/64) and prose to 0.56..0.76.
    assert!(!next_wide(true, 40, &r), "prose must leave wide");
    assert!(next_wide(false, 62, &r), "code must enter wide");
    assert!(next_wide(true, 56, &r), "wide-rung code must hold wide");
    // Dead band: 49..=59 holds whichever state it is in (prose at 0.76 does
    // not leave from 49 alone; it leaves on its low windows).
    assert!(next_wide(true, 49, &r));
    assert!(!next_wide(false, 49, &r));
    assert!(next_wide(true, 58, &r));
    assert!(!next_wide(false, 58, &r));
    // Edges.
    assert!(next_wide(false, 60, &r));
    assert!(!next_wide(true, 46, &r));
    assert!(next_wide(true, 47, &r));
}

#[test]
fn shift_register_is_exactly_one_window() {
    let mut r = 0u64;
    for _ in 0..WINDOW {
        r = shift_in(r, true);
    }
    assert_eq!(r.count_ones(), WINDOW);
    // The next step evicts the oldest bit: still exactly WINDOW bits.
    r = shift_in(r, false);
    assert_eq!(r.count_ones(), WINDOW - 1);
}

/// Deterministic Bernoulli stream (LCG) at hit rate `p`, `steps` long.
fn stream(p: f64, steps: usize, seed: u64) -> Vec<bool> {
    let mut x = seed;
    (0..steps)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 11) as f64 / (1u64 << 53) as f64) < p
        })
        .collect()
}

/// Drive the REAL controller (cap 10, defaults) from `start_wide` over a
/// C=1 stream: each step is one `drafts_for(1, 9)` dispatch followed by the
/// verify's `observe_step`, exactly the scheduler's order. Returns (flips,
/// final wide). Caller holds `SERIAL`.
fn simulate(start_wide: bool, hits: &[bool]) -> (u64, bool) {
    configure_with(10, false, r10());
    assert!(armed());
    if !start_wide {
        // Walk the controller into the narrow state the way production
        // would: one window of misses after the dwell.
        for _ in 0..(2 * WINDOW) {
            drafts_for(1, 9);
            observe_step(false);
        }
        assert_eq!(drafts_for(1, 9), 4, "setup did not reach narrow");
    }
    let before = flips();
    for &h in hits {
        drafts_for(1, 9);
        observe_step(h);
    }
    let wide = drafts_for(1, 9) == 9;
    (flips() - before, wide)
}

/// Flips allowed for 40k steady wide-rung-code steps (p1 0.875): the sized
/// rate is ~2.5 false drops per 40k steps, each paired with a re-entry once
/// the register clears 60 again, so ~5 flips expected (30 seed sets: mean
/// 4.7, sd 2.7, max 10). 16 is four sigma over the mean.
const CODE_WIDE_FLIP_BOUND: u64 = 16;

#[test]
fn steady_workloads_never_flip_and_transitions_do() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // 20 seeds x 2000 steps (~17 Volvo requests each) of steady prose from
    // the narrow state, and steady code from wide. Sized for 0.11 / 0.01
    // spurious flips per 2000 steps: the 40k-step totals must stay in the
    // single digits (expected ~2 and ~0).
    let prose: u64 = (1..=20u64)
        .map(|s| simulate(false, &stream(0.76, 2000, s)).0)
        .sum();
    let code: u64 = (1..=20u64)
        .map(|s| simulate(true, &stream(0.97, 2000, s)).0)
        .sum();
    // Code as the WIDE rung actually sees it (p1 ~0.875, 2026-09-04): the
    // leave line sits 3.8 sigma under 56/64, Bin(64, 0.875) <= 46 is 6.5e-4
    // per sample. Each false drop is followed by a re-entry once the register
    // clears 60 again, so flips come in pairs; see the bound below.
    let code_wide: u64 = (1..=20u64)
        .map(|s| simulate(true, &stream(0.875, 2000, s)).0)
        .sum();
    assert!(
        prose <= 6,
        "steady prose flipped {prose} times in 40k steps"
    );
    assert!(code <= 2, "steady code flipped {code} times in 40k steps");
    assert!(
        code_wide <= CODE_WIDE_FLIP_BOUND,
        "steady wide-rung code (0.875) flipped {code_wide} times in 40k steps (bound {CODE_WIDE_FLIP_BOUND})"
    );
    // A real transition (prose then code, and back) must END in the right
    // state every time, and the flip total over 20 seeds x 2 directions must
    // be the 40 real flips plus at most the sized false-alarm allowance
    // (the same 0.11 per 2000 steps: expected ~1 extra pair in 16k steps).
    let mut total = 0u64;
    for seed in 1..=20u64 {
        let mut s = stream(0.76, 400, seed);
        s.extend(stream(0.97, 400, seed + 100));
        let (flips, wide) = simulate(false, &s);
        assert!(
            wide && flips >= 1,
            "prose->code seed {seed}: flips={flips} wide={wide}"
        );
        total += flips;
        let mut s = stream(0.97, 400, seed);
        s.extend(stream(0.76, 400, seed + 100));
        let (flips, wide) = simulate(true, &s);
        assert!(
            !wide && flips >= 1,
            "code->prose seed {seed}: flips={flips} wide={wide}"
        );
        total += flips;
    }
    assert!(total <= 46, "transitions: {total} flips for 40 real ones");
    configure_with(10, true, r10());
}

/// The dwell is enforced by the real controller: after a switch, a full
/// window of the opposite signal must pass before the next decision.
#[test]
fn dwell_holds_one_full_window_after_a_switch() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    configure_with(10, false, r10());
    // Wide at start; misses for one dwell -> no decision yet, then narrow.
    for i in 0..WINDOW {
        drafts_for(1, 9);
        observe_step(false);
        assert_eq!(
            drafts_for(1, 9),
            if i + 1 < WINDOW { 9 } else { 4 },
            "step {i}"
        );
    }
    // Now hits: nothing may flip inside the dwell, even at 64/64.
    let before = flips();
    for i in 0..(WINDOW - 1) {
        drafts_for(1, 9);
        observe_step(true);
        assert_eq!(drafts_for(1, 9), 4, "flipped inside the dwell at step {i}");
    }
    drafts_for(1, 9);
    observe_step(true);
    assert_eq!(drafts_for(1, 9), 9, "a full window of hits must go wide");
    assert_eq!(flips() - before, 1);
    configure_with(10, true, r10());
}

/// The observer scores C=1 steps only: a C>=2 dispatch between two C=1
/// steps is not a sample.
#[test]
fn observer_ignores_multi_sequence_steps() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    configure_with(10, false, r10());
    for _ in 0..(4 * WINDOW) {
        drafts_for(16, 9);
        observe_step(false);
    }
    assert_eq!(
        drafts_for(1, 9),
        9,
        "C=16 misses must not move the C=1 state"
    );
    configure_with(10, true, r10());
}

#[test]
fn pinned_returns_num_drafts_unchanged() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // An explicit --dflash-gamma pins the serve: the record launches must
    // reproduce byte-for-byte.
    configure_with(10, true, r10());
    assert!(!armed());
    assert_eq!(drafts_for(16, 9), 9);
    assert_eq!(drafts_for(1, 9), 9);
    // Unpinned: C=16 -> 3 drafts (K=4), C=1 wide -> 9 (K=10).
    configure_with(10, false, r10());
    assert!(armed());
    assert_eq!(drafts_for(16, 9), 3);
    assert_eq!(drafts_for(1, 9), 9);
    configure_with(10, true, r10());
}
