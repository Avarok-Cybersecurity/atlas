// SPDX-License-Identifier: AGPL-3.0-only

//! The selection rule and the accumulation that feeds it.
//!
//! These decide which experts a category gets loaded with, so the cases are
//! chosen to pin the ordering (mass, not frequency) and the boundary
//! behaviour of coverage — the two places where a wrong answer still looks
//! entirely reasonable.

use super::*;

/// `(layer, [(expert, count, mass)])` — one layer of a fixture response.
type LayerFixture = (usize, Vec<(u32, u32, f64)>);
use crate::benchmarks::expert_categories::usage::{Activation, LayerActivation};

fn act(layers: Vec<LayerFixture>, tokens: u64) -> Activation {
    Activation {
        top_k: 2,
        num_experts: 8,
        tokens_routed: tokens,
        unattributed_rows: 0,
        layers: layers
            .into_iter()
            .map(|(layer, experts)| LayerActivation { layer, experts })
            .collect(),
    }
}

// ---------------------------------------------------------------- Path A

#[test]
fn sums_mass_and_counts_across_prompts() {
    let mut a = Accumulator::new();
    a.feed("python", &act(vec![(0, vec![(1, 2, 1.0), (3, 1, 0.5)])], 2))
        .unwrap();
    a.feed(
        "python",
        &act(vec![(0, vec![(1, 1, 0.5), (5, 1, 0.25)])], 1),
    )
    .unwrap();

    let mass = a.layer_mass("python");
    assert_eq!(mass[0].0, 0);
    assert_eq!(mass[0].1, vec![(1, 3, 1.5), (3, 1, 0.5), (5, 1, 0.25)]);
    assert_eq!(a.totals("python").prompts, 2);
    assert_eq!(a.totals("python").tokens_routed, 3);
}

#[test]
fn budget_keeps_the_smallest_set_covering_the_target_mass() {
    let mut a = Accumulator::new();
    // Total 1.0; 0.6 + 0.3 = 0.9 covers 90%, so expert 3 (0.1) drops.
    a.feed(
        "python",
        &act(vec![(0, vec![(1, 1, 0.6), (2, 1, 0.3), (3, 1, 0.1)])], 1),
    )
    .unwrap();
    let b = &a.budgets(0.9)[0];
    assert_eq!(b.layers, vec![(0usize, vec![1u32, 2u32])]);
    assert_eq!(b.coverage, 0.9);
}

#[test]
fn budgeted_ids_are_ascending_not_mass_ordered() {
    // budget_experts returns descending by weight; the table is written
    // ascending so two runs of the same measurement diff cleanly.
    let mut a = Accumulator::new();
    a.feed("python", &act(vec![(0, vec![(1, 1, 0.1), (5, 1, 0.9)])], 1))
        .unwrap();
    let b = &a.budgets(1.0)[0];
    assert_eq!(b.layers[0].1, vec![1, 5], "ids must be ascending");
}

// ---------------------------------------------------------------- Path B

#[test]
fn mass_not_frequency_decides_what_is_kept() {
    // The rule this benchmark exists to apply. Expert 1 is chosen five times
    // at trivial weight; expert 2 once at dominant weight. A
    // frequency-ranked selection keeps the wrong one, and the resulting
    // serve would load an expert the layer barely uses while dropping the
    // one it is mostly made of.
    let mut a = Accumulator::new();
    a.feed(
        "python",
        &act(vec![(0, vec![(1, 5, 0.10), (2, 1, 0.90)])], 3),
    )
    .unwrap();
    let b = &a.budgets(0.8)[0];
    assert_eq!(b.layers[0].1, vec![2], "the high-mass expert must survive");
}

#[test]
fn a_layer_with_no_mass_is_omitted_not_emitted_empty() {
    // An empty id list in the table would tell boot-time loading to load NO
    // experts for that layer — every token routed there would hit an
    // unloaded expert. Absent means "this category has nothing to say about
    // this layer"; empty would mean something much worse.
    let mut a = Accumulator::new();
    a.feed(
        "python",
        &act(vec![(0, vec![(1, 1, 0.0)]), (1, vec![(2, 1, 1.0)])], 2),
    )
    .unwrap();
    let b = &a.budgets(0.9)[0];
    assert_eq!(
        b.layers.iter().map(|(l, _)| *l).collect::<Vec<_>>(),
        vec![1],
        "the zero-mass layer must be omitted entirely"
    );
}

#[test]
fn full_coverage_keeps_every_expert_that_routed() {
    let mut a = Accumulator::new();
    a.feed(
        "python",
        &act(vec![(0, vec![(1, 1, 0.5), (2, 1, 0.3), (3, 1, 0.2)])], 1),
    )
    .unwrap();
    assert_eq!(a.budgets(1.0)[0].layers[0].1, vec![1, 2, 3]);
}

// ---------------------------------------------------------------- Path C

#[test]
fn a_changed_routing_geometry_mid_run_is_refused() {
    // The served model changed under the benchmark. Summing mass across two
    // expert spaces would produce a table describing neither.
    let mut a = Accumulator::new();
    a.feed("python", &act(vec![(0, vec![(1, 2, 1.0)])], 1))
        .unwrap();
    let mut other = act(vec![(0, vec![(1, 2, 1.0)])], 1);
    other.num_experts = 256;
    let err = a.feed("python", &other).unwrap_err().to_string();
    assert!(err.contains("routing geometry changed"), "got: {err}");
}

#[test]
fn jaccard_separates_identical_from_disjoint_categories() {
    // The number that says whether categorization is possible at all: two
    // categories routing identically cannot be given different expert sets.
    let mut a = Accumulator::new();
    a.feed("python", &act(vec![(0, vec![(1, 1, 1.0)])], 1))
        .unwrap();
    a.feed("rust", &act(vec![(0, vec![(1, 1, 1.0)])], 1))
        .unwrap();
    a.feed("french", &act(vec![(0, vec![(5, 1, 1.0)])], 1))
        .unwrap();
    let b = a.budgets(1.0);
    let get = |n: &str| b.iter().find(|c| c.category == n).unwrap().clone();

    assert_eq!(mean_jaccard(&get("python"), &get("rust")), 1.0);
    assert_eq!(mean_jaccard(&get("python"), &get("french")), 0.0);
}

#[test]
fn jaccard_of_categories_sharing_no_layer_is_zero_not_a_divide_by_zero() {
    let mut a = Accumulator::new();
    a.feed("python", &act(vec![(0, vec![(1, 1, 1.0)])], 1))
        .unwrap();
    a.feed("rust", &act(vec![(7, vec![(1, 1, 1.0)])], 1))
        .unwrap();
    let b = a.budgets(1.0);
    let get = |n: &str| b.iter().find(|c| c.category == n).unwrap().clone();
    assert_eq!(mean_jaccard(&get("python"), &get("rust")), 0.0);
}
