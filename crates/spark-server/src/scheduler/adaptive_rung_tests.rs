// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the wave-28 accept-rate-aware n=16 rung.
//!
//! The wave-27 measurements are the ANCHOR: the decision rule must classify
//! every one of them the way the box did. Only ONE test drives the shared
//! controller state (`converges_and_holds_without_oscillating`); the rest are
//! pure so they can run in any order on any thread.

use super::*;

/// The four wave-27 telemetry points, `(label, p1, p2_cond, measured winner
/// is depth)`. Prose is measured at every length as a LOSS or a tie for
/// `k=2`; tool-shaped is a +7.9% WIN for it.
const W27: [(&str, f64, f64, bool); 4] = [
    ("decode_short L=64", 0.840, 0.542, false),
    ("decode_short L=128", 0.773, 0.539, false),
    ("decode_short L=1024", 0.732, 0.567, false),
    ("tool-shaped natural EOS", 0.917, 0.877, true),
];

#[test]
fn token_ratio_reproduces_wave_27() {
    // The formula must land on the ratios the box measured (1.186-1.229 on
    // prose, 1.424 on tool-shaped) within the telemetry's own rounding.
    assert!((token_ratio(0.917, 0.877) - 1.424).abs() < 0.01);
    assert!((token_ratio(0.732, 0.567) - 1.229).abs() < 0.015);
    // Degenerate inputs never claim depth pays.
    assert_eq!(token_ratio(0.0, 0.9), 1.0);
    assert_eq!(token_ratio(f64::NAN, 0.9), 1.0);
    // p2_cond == 0 -> depth buys exactly nothing beyond the first token.
    assert_eq!(token_ratio(0.9, 0.0), 1.0);
}

#[test]
fn p2_cond_inverts_mean_na() {
    // mean_na = p1 + p1*p2_cond, so the inverse must round-trip.
    let (p1, p2) = (0.917, 0.877);
    let mean_na = p1 + p1 * p2;
    assert!((p2_cond_from(p1, mean_na).unwrap() - p2).abs() < 1e-9);
    // A k=1 flush reports mean_na == p1 -> no second-token evidence at all.
    assert_eq!(p2_cond_from(p1, p1).unwrap(), 0.0);
    // Out-of-range telemetry clamps instead of producing a nonsense ratio.
    assert_eq!(p2_cond_from(0.5, 2.0).unwrap(), 1.0);
    assert_eq!(p2_cond_from(0.0, 0.5), None);
}

#[test]
fn rule_classifies_every_wave_27_point_the_way_the_box_did() {
    for (label, p1, p2, depth_wins) in W27 {
        let tr = token_ratio(p1, p2);
        // From BOTH states, since the dead band is only 0.02 wide and every
        // wave-27 point must sit OUTSIDE it — a point inside the band would
        // make the rung a function of its own history, not of the traffic.
        assert!(
            !(LEAVE..ENTER).contains(&tr),
            "{label}: token_ratio {tr:.4} lands INSIDE the dead band"
        );
        assert_eq!(next_state(false, tr), depth_wins, "{label} from k=1");
        assert_eq!(next_state(true, tr), depth_wins, "{label} from k=2");
    }
}

#[test]
fn dead_band_holds_state_and_is_asymmetric() {
    // Strictly inside the band: whatever state we are in, we stay in it.
    let mid = (ENTER + LEAVE) / 2.0;
    assert!(
        !next_state(false, mid),
        "must not enter depth on weak evidence"
    );
    assert!(
        next_state(true, mid),
        "must not leave depth on weak evidence"
    );
    // Enter is the higher bar (depth is EARNED, surrendered more readily).
    const { assert!(ENTER > LEAVE) };
    assert!(next_state(false, ENTER));
    assert!(!next_state(true, LEAVE - 1e-9));
}

#[test]
fn scope_is_the_n16_rung_only() {
    // n<=8 keeps the 8:3 rung and n>16 keeps 32:1 — wave 28 does not touch
    // either, so no width outside 9..=16 may be lifted by the controller.
    for n in [1usize, 4, 8] {
        assert_eq!(drafts_for(n, 3), 3, "n={n} must keep the 8:3 rung");
    }
    for n in [17usize, 24, 32, 64] {
        assert_eq!(drafts_for(n, 3), 1, "n={n} must keep the 32:1 rung");
    }
    // --num-drafts stays the ceiling at every width, adaptation included.
    assert_eq!(drafts_for(16, 1), 1);
    assert_eq!(drafts_for(16, 0), 0);
}

#[test]
fn converges_and_holds_without_oscillating() {
    // The ONLY test that drives the shared controller. Flush granularity is
    // one `mtp_accept_debug` PERIOD (128 verifies ~ 8 steps at n=16 ~ 1.16 s).
    let flush = |p1: f64, p2: f64, k: usize| {
        let mean_na = if k >= 2 { p1 + p1 * p2 } else { p1 };
        observe(16, k, p1, mean_na);
    };

    // 1. Cold start PROBES: p2_cond is unobservable at k=1, so a fresh serve
    //    must measure it rather than guess.
    assert_eq!(drafts_for(16, 3), 2, "cold start must probe at depth");

    // 2. Prose: the probe scores a losing token ratio -> settle at k=1 in ONE
    //    flush, and the probe self-terminates.
    let (_, pp1, pp2, _) = W27[2];
    flush(pp1, pp2, 2);
    assert_eq!(
        drafts_for(16, 3),
        1,
        "prose must settle at k=1 after one probe"
    );

    // 3. ★ It must then STOP PROBING. A probe costs a measured 1.74 s of
    //    graph re-capture, so a controller that keeps probing loses ~11% on
    //    prose (timer at 12.5% duty) or ~4.8% (fast-EWMA trigger at 0.06).
    //    Replayed against the MEASURED per-flush p1 noise at n=16 — sigma =
    //    0.053, deterministic pseudo-random so the test cannot flake — the
    //    slow trigger must fire ZERO probes.
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut noise = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        // Sum of 3 uniforms -> approximately normal, scaled to sigma 0.053.
        let u = |b: u32| ((rng >> b) & 0xFFFF) as f64 / 65535.0 - 0.5;
        (u(0) + u(16) + u(32)) * 0.106
    };
    for t in 0..200 {
        flush((pp1 + noise()).clamp(0.0, 1.0), 0.0, 1);
        assert_eq!(
            drafts_for(16, 3),
            1,
            "spurious probe at tick {t} on steady p1"
        );
    }

    // 4. Traffic turns tool-shaped. p1 is observed FREE and jumps ~0.17, which
    //    is the trigger to spend a probe — within a flush or two, not a
    //    2048-flush backstop.
    let (_, tp1, tp2, _) = W27[3];
    let mut ticks_to_probe = 0;
    while drafts_for(16, 3) != 2 && ticks_to_probe < 20 {
        flush(tp1, 0.0, 1);
        ticks_to_probe += 1;
    }
    // ~5 flushes ~ 6 s: the slow EWMA's price for not firing on jitter, and
    // still inside a single agentic burst.
    assert!(
        ticks_to_probe <= 8,
        "p1 jump took {ticks_to_probe} flushes to trigger a probe"
    );

    // 5. The probe scores a winning token ratio -> depth latches at once.
    flush(tp1, tp2, 2);
    assert!(
        CTL.at_depth.load(Ordering::Relaxed),
        "tool traffic never reached k=2"
    );

    // 6. Depth HOLDS on tool traffic — every flush is a depth flush now, so
    //    p2_cond is observed continuously and for free, and no probe is ever
    //    needed again while the regime lasts.
    let before = CTL.flips.load(Ordering::Relaxed);
    for t in 0..60 {
        assert_eq!(drafts_for(16, 3), 2, "tool oscillated at tick {t}");
        flush(tp1, tp2, 2);
    }
    assert_eq!(CTL.flips.load(Ordering::Relaxed), before, "rung oscillated");

    // 7. Back to prose: depth is surrendered within the EWMA window, with no
    //    probe required (we are already at depth, so the evidence is free).
    let mut ticks = 0;
    while CTL.at_depth.load(Ordering::Relaxed) && ticks < 20 {
        flush(pp1, pp2, 2);
        ticks += 1;
    }
    assert!(!CTL.at_depth.load(Ordering::Relaxed), "never left depth");
    assert!(ticks <= 4, "took {ticks} flushes to leave depth");
}

/// The WIDTH half of the regime (wave 47): one flip per real transition, and
/// none for a width that stays on the same side of the cap. Pure — it drives
/// its own statics and never touches the depth controller.
#[test]
fn width_regime_flips_once_per_transition() {
    let base = WIDTH_FLIPS.load(Ordering::Relaxed);
    // A ladder walked up: engaged through the cap, then disengaged. ONE flip,
    // no matter how many steps are taken at each width.
    for _ in 0..50 {
        note_width_regime(16, true);
    }
    assert_eq!(
        WIDTH_FLIPS.load(Ordering::Relaxed),
        base,
        "engaged->engaged flipped"
    );
    for _ in 0..50 {
        note_width_regime(64, false);
    }
    assert_eq!(
        WIDTH_FLIPS.load(Ordering::Relaxed),
        base + 1,
        "crossing the cap must count exactly one flip"
    );
    // And back down.
    note_width_regime(32, true);
    assert_eq!(WIDTH_FLIPS.load(Ordering::Relaxed), base + 2);
    assert!(WIDTH_ENGAGED.load(Ordering::Relaxed));
}
