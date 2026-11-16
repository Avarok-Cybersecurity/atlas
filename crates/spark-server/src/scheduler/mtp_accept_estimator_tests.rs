// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for [`super`] (`mtp_accept_estimator`). A sibling file via `#[path]`,
//! like `mtp_dcut_tests.rs`, so the estimator stays under the 500-line cap.

use super::*;

/// Deterministic xorshift64* stream in `[0, 1)`; the tests need no `rand`.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        (v >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn feature_is_the_log_odds_of_the_top1_probability() {
    assert!(feature(0.5f32.ln()).abs() < 1e-6, "q=0.5 has log-odds 0");
    let f9 = feature(0.9f32.ln());
    assert!((f9 - 9.0f64.ln()).abs() < 1e-5, "q=0.9 -> ln 9, got {f9}");
    let f1 = feature(0.1f32.ln());
    assert!((f1 + 9.0f64.ln()).abs() < 1e-5, "q=0.1 -> -ln 9, got {f1}");
    // Monotone in the probability.
    let probs = [0.01f32, 0.2, 0.5, 0.8, 0.99, 0.999_999];
    let fs: Vec<f64> = probs.iter().map(|p| feature(p.ln())).collect();
    assert!(fs.windows(2).all(|w| w[0] < w[1]), "{fs:?}");
}

#[test]
fn feature_clamps_and_survives_degenerate_input() {
    assert_eq!(feature(0.0), MAX_LOG_ODDS, "q=1 saturates high");
    assert_eq!(
        feature(f32::NEG_INFINITY),
        -MAX_LOG_ODDS,
        "q=0 saturates low"
    );
    assert_eq!(feature(f32::NAN), 0.0, "NaN carries no information");
    // A tiny negative log-probability keeps its complement mass (no 1 - exp).
    let near_one = feature(-1e-12);
    assert!(near_one > 20.0 && near_one <= MAX_LOG_ODDS, "{near_one}");
    // A positive "log-probability" is a caller bug; it reads as certain.
    assert_eq!(feature(0.5), MAX_LOG_ODDS);
}

#[test]
fn survival_is_a_non_increasing_prefix_product() {
    let e = AcceptEstimator::new();
    let conf = [-0.05f32, -0.4, -1.5, -0.01];
    let s = e.survival(&conf, 4);
    assert_eq!(s.len(), 4);
    assert!(s.windows(2).all(|w| w[1] <= w[0]), "{s:?}");
    assert!(s.iter().all(|&x| x > 0.0 && x < 1.0), "{s:?}");
    // The first entry is the calibrated p of position 0 itself.
    assert!((s[0] - e.predict(feature(conf[0]), 0)).abs() < 1e-12);
}

#[test]
fn an_unmeasured_position_counts_as_certain() {
    let e = AcceptEstimator::new();
    let s = e.survival(&[], 3);
    assert_eq!(s, vec![1.0, 1.0, 1.0]);
    let part = e.survival(&[-0.2], 3);
    assert!(part[0] < 1.0);
    assert_eq!(part[1], part[0], "no measurement leaves survival unchanged");
    assert_eq!(part[2], part[0]);
}

#[test]
fn observe_pass_grades_the_accepted_prefix_and_the_first_rejection() {
    let conf = [-0.1f32, -0.2, -0.3];
    // Accepted 1 of 3: position 0 graded 1, position 1 graded 0, position 2 unreached.
    let mut e = AcceptEstimator::new();
    e.observe_pass(&conf, 1);
    assert_eq!(e.observed, 2);
    assert_eq!(e.counts[0], 1.0);
    assert_eq!(e.counts[1], 1.0);
    assert_eq!(e.counts[2], 0.0);
    // Everything accepted: all three graded, no rejection to grade.
    let mut e = AcceptEstimator::new();
    e.observe_pass(&conf, 3);
    assert_eq!(e.observed, 3);
    // Nothing accepted: only position 0 (the rejection) is graded.
    let mut e = AcceptEstimator::new();
    e.observe_pass(&conf, 0);
    assert_eq!(e.observed, 1);
    // An empty pass grades nothing; an over-large accepted count is clamped.
    let mut e = AcceptEstimator::new();
    e.observe_pass(&[], 5);
    e.observe_pass(&conf, 99);
    assert_eq!(e.observed, 3);
}

#[test]
fn a_refit_on_empty_statistics_changes_nothing() {
    let mut e = AcceptEstimator::new();
    e.refit();
    assert_eq!(e.slope, INIT_SLOPE);
    assert!(e.bias.iter().all(|&b| b == INIT_BIAS));
    assert_eq!(e.refits, 1);
}

#[test]
fn tick_refits_every_interval_and_clears_the_statistics() {
    let mut e = AcceptEstimator::new();
    e.observe(0, 1.0, true);
    for i in 1..REFIT_INTERVAL {
        assert!(!e.tick(), "tick {i} must not refit");
    }
    assert!(e.tick(), "the interval-th tick refits");
    assert_eq!(e.refits, 1);
    assert_eq!(e.counts[0], 0.0, "statistics are cleared by a refit");
    assert!(!e.tick(), "the interval restarts after a refit");
}

#[test]
fn irls_recovers_known_coefficients_from_bernoulli_data() {
    let (w_true, b_true) = (1.3f64, [-0.5f64, 0.2, 0.6]);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let per_pos = 30_000;
    let mut data: Vec<(usize, f64, bool)> = Vec::new();
    for (pos, &b) in b_true.iter().enumerate() {
        for _ in 0..per_pos {
            let f = rng.unit() * 6.0 - 3.0;
            let p = 1.0 / (1.0 + (-(w_true * f + b)).exp());
            data.push((pos, f, rng.unit() < p));
        }
    }
    let mut e = AcceptEstimator::new();
    // Each refit is ONE damped Newton step; re-feeding the same sample converges
    // like Newton-Raphson on the full-batch log-likelihood.
    for _ in 0..15 {
        for &(pos, f, y) in &data {
            e.observe(pos, f, y);
        }
        e.refit();
    }
    assert!(
        (e.slope - w_true).abs() < 0.05,
        "slope {} vs {w_true}",
        e.slope
    );
    for (k, &b) in b_true.iter().enumerate() {
        assert!(
            (e.bias[k] - b).abs() < 0.05,
            "bias[{k}] {} vs {b}",
            e.bias[k]
        );
    }
    // A position that never saw data keeps the cold-start intercept.
    assert_eq!(e.bias[5], INIT_BIAS);
}

#[test]
fn a_step_that_would_flip_the_slope_negative_is_dropped() {
    let mut e = AcceptEstimator::new();
    // Confident drafts rejected, doubtful drafts accepted: the likelihood wants
    // a negative slope, which the model forbids (confidence must not LOWER p).
    for _ in 0..4_000 {
        e.observe(0, 5.0, false);
        e.observe(0, -5.0, true);
    }
    e.refit();
    assert!(e.slope >= 0.0, "slope {}", e.slope);
    assert!(e.slope.is_finite() && e.bias.iter().all(|b| b.is_finite()));
}

#[test]
fn damping_makes_a_data_poor_position_move_slowly() {
    let mut few = AcceptEstimator::new();
    few.observe(0, 0.0, true);
    few.refit();
    let mut many = AcceptEstimator::new();
    for _ in 0..5_000 {
        many.observe(0, 0.0, true);
    }
    many.refit();
    let moved_few = few.bias[0] - INIT_BIAS;
    let moved_many = many.bias[0] - INIT_BIAS;
    assert!(
        moved_few > 0.0 && moved_many > moved_few,
        "{moved_few} {moved_many}"
    );
    // One observation carries 1/51 of a full step.
    assert!(moved_few < 0.1, "{moved_few}");
}
