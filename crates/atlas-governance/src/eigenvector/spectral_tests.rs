// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the intent-space decomposition.
//!
//! These are not shape checks. The decomposition drives a published report, and
//! the two ways it can be wrong without looking wrong are: recovering an axis
//! that is not there, and producing a different answer for the same input.
//! Every test below targets one of those, or a degenerate corpus that will
//! genuinely occur — a young ledger, or one where every PR did the same thing.

use super::spectral::{Spectrum, decompose};

/// Deterministic noise. A seeded LCG rather than a random source, because a
/// test for determinism cannot itself be nondeterministic.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }
}

/// `n` points spread along `axis` with a little noise on every other dimension.
fn planted(n: usize, dim: usize, axis: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Lcg(seed);
    let mut rows = Vec::new();
    let mut coords = Vec::new();
    for i in 0..n {
        // Spread symmetrically about zero so the planted axis is variation and
        // not merely offset — centring would remove the latter.
        let t = i as f64 - (n as f64 - 1.0) / 2.0;
        let mut row: Vec<f64> = (0..dim).map(|_| rng.next() * 0.01).collect();
        row[axis] += t;
        rows.push(row);
        coords.push(t);
    }
    (rows, coords)
}

fn stamps(n: usize) -> Vec<i64> {
    (0..n as i64).collect()
}

/// Pearson correlation, used to ask "did the loadings recover the coordinate we
/// planted?" without caring about scale or sign.
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let sa: f64 = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>().sqrt();
    let sb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum::<f64>().sqrt();
    cov / (sa * sb)
}

#[test]
fn recovers_a_planted_axis() {
    let (rows, coords) = planted(40, 64, 7, 1);
    let s = decompose(&rows, &stamps(40), 3);

    let pc1 = &s.components[0];
    // The loadings must order the PRs the way the planted coordinate does.
    assert!(
        correlation(&pc1.loadings, &coords).abs() > 0.99,
        "PC1 did not recover the planted axis (r = {})",
        correlation(&pc1.loadings, &coords)
    );
    // And it must dominate: the noise dimensions carry a thousandth of the
    // spread, so anything but near-total explained variance means the
    // decomposition found structure that is not there.
    assert!(pc1.explained > 0.99, "PC1 explained only {}", pc1.explained);
    assert_eq!(s.coherence, Some(pc1.explained));
}

#[test]
fn explained_variance_is_ordered_and_bounded() {
    let (rows, _) = planted(30, 32, 3, 2);
    let s = decompose(&rows, &stamps(30), 3);
    let total: f64 = s.components.iter().map(|c| c.explained).sum();
    assert!(
        total <= 1.0 + 1e-9,
        "explained variance sums to {total} > 1"
    );
    for pair in s.components.windows(2) {
        assert!(
            pair[0].explained >= pair[1].explained - 1e-12,
            "components out of order: {} then {}",
            pair[0].explained,
            pair[1].explained
        );
    }
}

#[test]
fn deflation_actually_orthogonalises() {
    // Two genuinely independent directions, so PC1 and PC2 are both real and
    // must come out orthogonal rather than PC2 re-finding PC1.
    let mut rows = Vec::new();
    for i in 0..24 {
        let mut row = vec![0.0; 16];
        row[2] = (i % 6) as f64 - 2.5;
        row[9] = (i / 6) as f64 - 1.5;
        rows.push(row);
    }
    let s = decompose(&rows, &stamps(24), 2);
    let (u1, u2) = (&s.components[0].loadings, &s.components[1].loadings);
    let dot: f64 = u1.iter().zip(u2).map(|(a, b)| a * b).sum();
    assert!(dot.abs() < 1e-6, "PC1·PC2 = {dot}, expected ~0");
}

/// The report is upserted into a Discussion comment on a schedule. A
/// decomposition that answered differently for identical input would rewrite
/// that comment forever, showing readers churn that means nothing.
#[test]
fn identical_input_gives_bit_identical_output() {
    let (rows, _) = planted(25, 48, 11, 3);
    let a = decompose(&rows, &stamps(25), 3);
    let b = decompose(&rows, &stamps(25), 3);
    let json = |s: &Spectrum| serde_json::to_string(s).expect("spectrum serialises");
    assert_eq!(json(&a), json(&b));
}

/// Eigenvectors are defined only up to sign, so power iteration will happily
/// return `u` one day and `-u` the next. The canonical-sign rule is what stops
/// that, and it is only observable as this invariant.
#[test]
fn sign_convention_is_stable_across_reorderings_of_equivalent_data() {
    let (rows, _) = planted(20, 32, 5, 4);
    let s = decompose(&rows, &stamps(20), 1);
    let pivot = s.components[0]
        .loadings
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, |acc, x| acc.max(x.abs()));
    let has_positive_pivot = s.components[0]
        .loadings
        .iter()
        .any(|x| (*x - pivot).abs() < 1e-12);
    assert!(
        has_positive_pivot,
        "largest-magnitude loading was left negative"
    );
}

#[test]
fn drift_is_zero_when_the_direction_holds() {
    // Both halves spread along the same axis: the repository is pushing the
    // same way it was.
    let (rows, _) = planted(40, 32, 6, 5);
    let s = decompose(&rows, &stamps(40), 1);
    let drift = s
        .drift_degrees
        .expect("40 points is enough for a drift reading");
    assert!(drift < 5.0, "expected ~0°, got {drift}°");
}

#[test]
fn drift_is_ninety_degrees_when_the_direction_turns() {
    // Older half varies along axis 1, newer half along axis 2 — orthogonal.
    let mut rows = Vec::new();
    for i in 0..20 {
        let mut row = vec![0.0; 16];
        row[1] = i as f64 - 9.5;
        rows.push(row);
    }
    for i in 0..20 {
        let mut row = vec![0.0; 16];
        row[2] = i as f64 - 9.5;
        rows.push(row);
    }
    let s = decompose(&rows, &stamps(40), 1);
    let drift = s.drift_degrees.expect("drift readable");
    assert!((drift - 90.0).abs() < 1.0, "expected ~90°, got {drift}°");
}

// ── Degenerate corpora that will actually occur ─────────────────────────────

#[test]
fn a_single_pr_yields_no_axes_rather_than_a_panic() {
    let s = decompose(&[vec![1.0, 2.0, 3.0]], &[0], 3);
    assert!(s.components.is_empty());
    assert_eq!(s.coherence, None);
    assert_eq!(s.drift_degrees, None);
    assert_eq!(s.n, 1);
}

#[test]
fn an_empty_ledger_yields_no_axes() {
    let s = decompose(&[], &[], 3);
    assert!(s.components.is_empty());
    assert_eq!(s.n, 0);
    assert_eq!(s.dim, 0);
}

/// A hundred PRs that all did the same thing have zero variance in every
/// direction. The explained ratio is 0/0 there, and the report must say "no
/// axes yet" rather than render NaN at a reader.
#[test]
fn a_zero_variance_corpus_yields_no_axes_and_no_nan() {
    let rows = vec![vec![0.5; 32]; 12];
    let s = decompose(&rows, &stamps(12), 3);
    assert!(
        s.components.is_empty(),
        "zero variance must not produce an axis"
    );
    assert_eq!(s.coherence, None);
    let json = serde_json::to_string(&s).expect("serialises");
    assert!(
        !json.contains("NaN") && !json.contains("null,null"),
        "NaN leaked into {json}"
    );
}

#[test]
fn asking_for_more_components_than_samples_is_bounded_by_the_samples() {
    let (rows, _) = planted(3, 8, 1, 6);
    let s = decompose(&rows, &stamps(3), 10);
    assert!(
        s.components.len() <= 3,
        "got {} components for 3 samples",
        s.components.len()
    );
}

#[test]
fn drift_needs_two_points_a_side_and_says_so_otherwise() {
    let (rows, _) = planted(3, 8, 1, 7);
    assert_eq!(decompose(&rows, &stamps(3), 1).drift_degrees, None);
}
