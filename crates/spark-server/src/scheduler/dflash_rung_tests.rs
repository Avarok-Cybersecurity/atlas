// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! Unit tests for the DFlash gamma resolver. The 2026-09-03 receipts are the
//! anchor: the rule must land every measured winner. Pure functions only; the
//! shared controller is exercised through `configure` + `drafts_for` once.

use super::*;

#[test]
fn concurrency_ladder_lands_the_records() {
    // C=16 prose 187.7 and code 269.7 are both K=4 (write-on-accept).
    assert_eq!(k_for(16, 10, true), 4);
    assert_eq!(k_for(16, 10, false), 4);
    assert_eq!(k_for(2, 10, true), 4);
    // C=1 code 66.8 is the full block; C=1 prose 27.0 is K=5.
    assert_eq!(k_for(1, 10, true), 10);
    assert_eq!(k_for(1, 10, false), 5);
}

#[test]
fn widths_never_exceed_the_cap() {
    // A head sized for K=4 (record launch) cannot be asked for K=5 or K=10.
    assert_eq!(k_for(1, 4, true), 4);
    assert_eq!(k_for(1, 4, false), 4);
    assert_eq!(k_for(16, 3, true), 3);
}

#[test]
fn hysteresis_separates_prose_from_code() {
    // Measured p1: prose ~0.76 (49/64), code ~0.97 (62/64).
    assert!(!next_wide(true, 49), "prose must leave wide");
    assert!(next_wide(false, 62), "code must enter wide");
    // Dead band: 58/64 holds whichever state it is in.
    assert!(next_wide(true, 58));
    assert!(!next_wide(false, 58));
    // Edges.
    assert!(next_wide(false, 60));
    assert!(!next_wide(true, 55));
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

/// Run the pure rule over a stream from a given state; returns (flips, final).
fn simulate(start_wide: bool, hits: &[bool]) -> (u32, bool) {
    let (mut r, mut wide, mut flips, mut last) = (0u64, start_wide, 0u32, 0i64);
    for (i, &h) in hits.iter().enumerate() {
        r = shift_in(r, h);
        let tick = i as i64 + 1;
        if tick - last < WINDOW as i64 {
            continue;
        }
        let now = next_wide(wide, r.count_ones());
        if now != wide {
            wide = now;
            flips += 1;
            last = tick;
        }
    }
    (flips, wide)
}

#[test]
fn steady_workloads_never_flip_and_transitions_do() {
    // 20 seeds x 2000 steps (~17 Volvo requests each) of steady prose from
    // the narrow state, and steady code from wide. Sized for 0.11 / 0.01
    // spurious flips per 2000 steps: the 40k-step totals must stay in the
    // single digits (expected ~2 and ~0).
    let prose: u32 = (1..=20u64)
        .map(|s| simulate(false, &stream(0.76, 2000, s)).0)
        .sum();
    let code: u32 = (1..=20u64)
        .map(|s| simulate(true, &stream(0.97, 2000, s)).0)
        .sum();
    assert!(
        prose <= 6,
        "steady prose flipped {prose} times in 40k steps"
    );
    assert!(code <= 2, "steady code flipped {code} times in 40k steps");
    // A real transition (prose then code, and back) must END in the right
    // state every time, and the flip total over 20 seeds x 2 directions must
    // be the 40 real flips plus at most the sized false-alarm allowance
    // (the same 0.11 per 2000 steps: expected ~1 extra pair in 16k steps).
    let mut total = 0u32;
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
}

#[test]
fn pinned_returns_num_drafts_unchanged() {
    // An explicit --dflash-gamma pins the serve: the record launches must
    // reproduce byte-for-byte.
    configure(10, true);
    assert!(!armed());
    assert_eq!(drafts_for(16, 9), 9);
    assert_eq!(drafts_for(1, 9), 9);
    // Unpinned: C=16 -> 3 drafts (K=4), C=1 wide -> 9 (K=10).
    configure(10, false);
    assert!(armed());
    assert_eq!(drafts_for(16, 9), 3);
    assert_eq!(drafts_for(1, 9), 9);
    configure(10, true);
}
