// SPDX-License-Identifier: AGPL-3.0-only

//! EAS-1.0 conformance.
//!
//! The five invariants below are the specification's test vectors, not
//! examples: an implementation that fails any of them is producing a
//! different number under the same name, which for a metric meant to be
//! compared across models and labs is the whole failure mode.
//!
//! The two that matter most are the ends of the scale. Chance must score
//! 0.00000 — an uncorrected plug-in estimator scores well above zero on
//! independent routing, and a standard whose floor drifts with the expert
//! count cannot be compared across models. A router that determines the
//! category must score 1.00000, which is what forces the null out of the
//! DENOMINATOR as well as the numerator.

use super::*;
use crate::benchmarks::expert_categories::aggregate::Accumulator;
use crate::benchmarks::expert_categories::usage::{Activation, LayerActivation};

/// `(layer, [(expert, count, mass)])` — one layer of a fixture prompt.
type LayerFixture = (usize, Vec<(u32, u32, f64)>);

const PERMS: usize = 200;
const SEED: u64 = 0xA71A5;

/// Build one prompt's activation: `(layer, [(expert, count, mass)])`.
fn act(layers: Vec<LayerFixture>) -> Activation {
    let routed: u64 = layers
        .iter()
        .map(|(_, e)| e.iter().map(|(_, c, _)| u64::from(*c)).sum::<u64>())
        .sum::<u64>()
        / 2;
    Activation {
        top_k: 2,
        num_experts: 16,
        tokens_routed: routed.max(1),
        unattributed_rows: 0,
        layers: layers
            .into_iter()
            .map(|(layer, experts)| LayerActivation { layer, experts })
            .collect(),
    }
}

/// `k` categories x `per_cat` prompts, each category routing to its own
/// disjoint pair of experts in every layer: expert identity determines the
/// category exactly.
fn deterministic(k: usize, per_cat: usize, layers: usize) -> Accumulator {
    let mut a = Accumulator::new();
    for c in 0..k {
        for _ in 0..per_cat {
            let ls = (0..layers)
                .map(|l| {
                    let base = (c * 2) as u32;
                    (l, vec![(base, 4u32, 0.6), (base + 1, 4u32, 0.4)])
                })
                .collect();
            a.feed(&format!("cat{c}"), &act(ls)).unwrap();
        }
    }
    a
}

// ------------------------------------------------- invariant (ii): ceiling

#[test]
fn a_router_that_determines_the_category_scores_one() {
    // The top of the scale. If the null were subtracted from the numerator
    // only, this would land below 1 and no model could ever reach the target
    // the metric is supposed to define.
    let a = deterministic(4, 25, 3);
    let e = compute(&a, PERMS, SEED).expect("scoreable");
    assert!(
        e.eas >= 0.999,
        "disjoint per-category experts must score ~1, got {:.5}",
        e.eas
    );
    // Per category the bar is 0.99, not 0.999: each category's null KL is
    // averaged over B draws of ITS OWN rows, so it is noisier than the pooled
    // Î⁰ that sets the global figure. The spec's 0.999 invariant is on EAS.
    for (c, s) in e.per_category.iter().enumerate() {
        assert!(*s >= 0.99, "category {c} scored {s:.5}");
    }
    for (l, s) in &e.per_layer {
        assert!(*s >= 0.999, "layer {l} scored {s:.5}");
    }
}

// -------------------------------------------------- invariant (i): floor

#[test]
fn routing_independent_of_category_scores_zero() {
    // The bottom of the scale, and the reason the estimator is not a plain
    // I/H ratio: every prompt routes identically regardless of its label, so
    // the plug-in mutual information is pure finite-sample noise. Chance must
    // read 0.00000, not "small".
    let mut a = Accumulator::new();
    for c in 0..4 {
        for i in 0..25 {
            // Same expert pair for every category; only the split varies, and
            // it varies by prompt index, not by label.
            let w = 0.5 + (i as f64) * 0.01;
            let ls = (0..3)
                .map(|l| (l, vec![(1u32, 4u32, w), (2u32, 4u32, 1.0 - w)]))
                .collect();
            a.feed(&format!("cat{c}"), &act(ls)).unwrap();
        }
    }
    let e = compute(&a, PERMS, SEED).expect("scoreable");
    assert!(
        e.eas <= 1e-6,
        "category-independent routing must score 0.00000, got {:.7}",
        e.eas
    );
}

#[test]
fn shuffling_the_labels_destroys_the_score() {
    // The same routing, relabelled at random, must fall to chance. This is
    // the null the estimator subtracts, applied as an end-to-end check: if
    // the correction were wrong in either direction, the shuffled score would
    // not land at zero while the true one stays at one.
    let real = compute(&deterministic(4, 25, 3), PERMS, SEED).expect("scoreable");
    assert!(real.eas >= 0.999);

    let mut a = Accumulator::new();
    for i in 0..100 {
        let c = i % 4; // routing block cycles fast
        // Label cycles SLOWLY, so every label sees all four routing blocks
        // equally. `(i * 7) % 4` would not do: 7 is coprime with 4, making the
        // label a bijection of the block and the two perfectly correlated.
        let label = (i / 4) % 4;
        let base = (c * 2) as u32;
        let ls = (0..3)
            .map(|l| (l, vec![(base, 4u32, 0.6), (base + 1, 4u32, 0.4)]))
            .collect();
        a.feed(&format!("cat{label}"), &act(ls)).unwrap();
    }
    let shuffled = compute(&a, PERMS, SEED).expect("scoreable");
    assert!(
        shuffled.eas <= 0.05,
        "shuffled labels must fall to chance, got {:.5}",
        shuffled.eas
    );
}

// ------------------------------------------- invariant (iii): collapse

#[test]
fn a_layer_where_one_expert_dominates_every_category_scores_zero() {
    // Router collapse is not alignment. This is the case that decides the
    // normalizer: dividing by min(H(C), H(E)) would score this layer HIGH,
    // because H(E) collapses with it.
    let mut a = Accumulator::new();
    for c in 0..4 {
        for _ in 0..25 {
            a.feed(
                &format!("cat{c}"),
                // Layer 0 collapsed onto expert 3 for everyone; layer 1 is
                // genuinely category-specific.
                &act(vec![
                    (0, vec![(3u32, 8u32, 1.0)]),
                    (1, vec![((c * 2) as u32, 8u32, 1.0)]),
                ]),
            )
            .unwrap();
        }
    }
    let e = compute(&a, PERMS, SEED).expect("scoreable");
    let collapsed = e.per_layer.iter().find(|(l, _)| *l == 0).unwrap().1;
    let specific = e.per_layer.iter().find(|(l, _)| *l == 1).unwrap().1;
    assert!(collapsed <= 1e-6, "collapsed layer scored {collapsed:.7}");
    assert!(specific >= 0.999, "specific layer scored {specific:.5}");
    assert!(
        e.layers_at_chance.contains(&0),
        "a collapsed layer must be reported as at-chance"
    );
}

// ------------------------------------------- invariants (iv) and (v)

#[test]
fn a_dead_expert_does_not_move_the_score() {
    // An expert nobody routes to carries no information. If it moved the
    // number, the score would depend on the checkpoint's expert COUNT rather
    // than on its routing, and two models with different expert counts could
    // not be compared.
    let base = compute(&deterministic(4, 25, 3), PERMS, SEED).expect("scoreable");

    let mut a = Accumulator::new();
    for c in 0..4 {
        for _ in 0..25 {
            let ls = (0..3)
                .map(|l| {
                    let b = (c * 2) as u32;
                    // Expert 15 present in the id space with zero mass.
                    (
                        l,
                        vec![(b, 4u32, 0.6), (b + 1, 4u32, 0.4), (15u32, 0u32, 0.0)],
                    )
                })
                .collect();
            a.feed(&format!("cat{c}"), &act(ls)).unwrap();
        }
    }
    let with_dead = compute(&a, PERMS, SEED).expect("scoreable");
    assert!(
        (base.eas - with_dead.eas).abs() < 1e-9,
        "dead expert moved the score: {:.9} vs {:.9}",
        base.eas,
        with_dead.eas
    );
}

#[test]
fn scaling_every_mass_does_not_move_the_score() {
    // The score is over DISTRIBUTIONS. If it were not scale-invariant, a
    // longer prompt would count as better-aligned than a short one saying the
    // same thing about routing.
    let base = compute(&deterministic(4, 25, 3), PERMS, SEED).expect("scoreable");

    let mut a = Accumulator::new();
    for c in 0..4 {
        for _ in 0..25 {
            let ls = (0..3)
                .map(|l| {
                    let b = (c * 2) as u32;
                    (l, vec![(b, 4u32, 6.0), (b + 1, 4u32, 4.0)])
                })
                .collect();
            a.feed(&format!("cat{c}"), &act(ls)).unwrap();
        }
    }
    let scaled = compute(&a, PERMS, SEED).expect("scoreable");
    assert!(
        (base.eas - scaled.eas).abs() < 1e-9,
        "10x mass moved the score: {:.9} vs {:.9}",
        base.eas,
        scaled.eas
    );
}

// ------------------------------------------------------------ properties

#[test]
fn the_same_seed_reproduces_the_same_number() {
    // A permutation null nobody can reproduce makes the score unauditable.
    let a = deterministic(3, 20, 2);
    let x = compute(&a, 64, 12345).unwrap();
    let y = compute(&a, 64, 12345).unwrap();
    assert_eq!(x.eas, y.eas);
    assert_eq!(x.per_category, y.per_category);
}

#[test]
fn partial_alignment_lands_between_the_ends() {
    // A metric that only distinguished its two extremes would be useless for
    // the thing it is for — telling models apart.
    let mut a = Accumulator::new();
    for c in 0..4 {
        for i in 0..25 {
            // Half of each category's mass goes to a shared expert, half to
            // its own: real but incomplete separation.
            let own = (c * 2) as u32;
            let ls = (0..3)
                .map(|l| {
                    (
                        l,
                        vec![(own, 4u32, 0.5), (14u32, 4u32, 0.5 + (i % 3) as f64 * 0.01)],
                    )
                })
                .collect();
            a.feed(&format!("cat{c}"), &act(ls)).unwrap();
        }
    }
    let e = compute(&a, PERMS, SEED).unwrap();
    assert!(
        e.eas > 0.05 && e.eas < 0.95,
        "partial alignment should land strictly between the ends, got {:.5}",
        e.eas
    );
}

#[test]
fn counts_and_mass_are_scored_separately() {
    // Categories that SELECT the same experts and differ only in weighting:
    // the mass score should see the difference, the count score should not.
    // That gap is the diagnostic the report exists to surface, because expert
    // dropping acts on selection, not on weight.
    let mut a = Accumulator::new();
    for c in 0..4 {
        for _ in 0..25 {
            // Every category selects experts 0..3 (identical counts), but the
            // mass concentrates on a different one of them per category.
            let experts: Vec<(u32, u32, f64)> = (0..4u32)
                .map(|e| (e, 4u32, if e as usize == c { 0.7 } else { 0.1 }))
                .collect();
            let ls = (0..3).map(|l| (l, experts.clone())).collect();
            a.feed(&format!("cat{c}"), &act(ls)).unwrap();
        }
    }
    let e = compute(&a, PERMS, SEED).unwrap();
    // 0.7 on the category's own expert against a 0.25 uniform marginal is
    // KL = 0.7·ln(2.8) + 3·0.1·ln(0.4) = 0.4458 nats, and H(C) = ln 4, so the
    // separation this construction actually carries is 0.4458/1.3863 = 0.32.
    // Asserting a bigger number would be asserting against the arithmetic.
    assert!(
        (0.30..0.35).contains(&e.eas),
        "mass separation should be ~0.32 for this construction: {:.5}",
        e.eas
    );
    assert!(
        e.eas_count <= 1e-6,
        "identical selection counts carry no category information: {:.7}",
        e.eas_count
    );
}

#[test]
fn a_single_category_is_unscoreable_rather_than_perfect() {
    // With one category there is no uncertainty to resolve: Ĥ(C) = 0 and the
    // ratio is 0/0. Returning 1.0 ("perfectly aligned!") would be the most
    // flattering possible reading of no information at all.
    let a = deterministic(1, 20, 2);
    assert!(compute(&a, 32, SEED).is_none());
}
