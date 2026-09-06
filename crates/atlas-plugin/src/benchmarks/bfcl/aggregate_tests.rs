// SPDX-License-Identifier: AGPL-3.0-only
//! The aggregator must reproduce `score.py` exactly, and must be immune to how
//! the samples were split across shards.
//!
//! The second property is the one the whole four-way split rests on. If it does
//! not hold, sharding silently changes the gate's number and the shards must not
//! ship.

use super::aggregate::{Tally, aggregate, union};
use std::collections::BTreeMap;

fn t(hits: u64, n: u64) -> Tally {
    Tally { hits, n }
}

fn map(pairs: &[(&str, u64, u64)]) -> BTreeMap<String, Tally> {
    pairs
        .iter()
        .map(|(k, h, n)| ((*k).to_string(), t(*h, *n)))
        .collect()
}

/// The 12-sample fixture `provision.rs` already pins against the real
/// `score.py`: non_live 25.0, live 75.0, hallucination 50.0, normalized 50.0,
/// overall 66.67.
///
/// Its per-subset tallies, from the same table:
///   simple_python 2/2, simple_java 0/1, multiple 0/1,
///   live_simple 3/3, live_multiple 0/1,
///   irrelevance 3/3, live_irrelevance 0/1
///
/// It is the right fixture precisely because normalized (50.00) and overall
/// (66.67) DIVERGE by 16.67 points on the same data — a flat mean cannot produce
/// both, so only correct weighting reproduces the pair.
fn reference() -> BTreeMap<String, Tally> {
    map(&[
        ("simple_python", 2, 2),
        ("simple_java", 0, 1),
        ("multiple", 0, 1),
        ("live_simple", 3, 3),
        ("live_multiple", 0, 1),
        ("irrelevance", 3, 3),
        ("live_irrelevance", 0, 1),
    ])
}

#[test]
fn the_aggregator_reproduces_the_reference_scores() {
    let a = aggregate(&reference());
    assert_eq!(a.total_samples, 12);
    assert_eq!(a.category_scores.get("non_live"), Some(&25.0), "{a:?}");
    assert_eq!(a.category_scores.get("live"), Some(&75.0), "{a:?}");
    assert_eq!(a.category_scores.get("hallucination"), Some(&50.0), "{a:?}");
    assert_eq!(a.normalized_single_turn_score, 50.0, "{a:?}");
    assert_eq!(a.overall_accuracy, 66.67, "{a:?}");
}

/// THE PROPERTY THE SPLIT DEPENDS ON. Any partition of the same rows must
/// produce the identical aggregate — that is what makes four shards equal one
/// run. Three different splits, including a deliberately LOPSIDED one, because
/// an equal split would also pass under a naive mean-of-means and would prove
/// nothing.
#[test]
fn any_partition_of_the_same_rows_gives_the_identical_aggregate() {
    let whole = aggregate(&reference());

    // Even-ish split.
    let even = union(&[
        map(&[
            ("simple_python", 1, 1),
            ("live_simple", 2, 2),
            ("irrelevance", 2, 2),
        ]),
        map(&[
            ("simple_python", 1, 1),
            ("simple_java", 0, 1),
            ("multiple", 0, 1),
            ("live_simple", 1, 1),
            ("live_multiple", 0, 1),
            ("irrelevance", 1, 1),
            ("live_irrelevance", 0, 1),
        ]),
    ]);
    assert_eq!(aggregate(&even), whole, "even split diverged");

    // Lopsided: one shard holds a single sample, and several subsets are
    // ABSENT from it entirely — the case that reweights a category if you
    // average scores instead of counts.
    let lopsided = union(&[
        map(&[("simple_python", 1, 1)]),
        map(&[
            ("simple_python", 1, 1),
            ("simple_java", 0, 1),
            ("multiple", 0, 1),
            ("live_simple", 3, 3),
            ("live_multiple", 0, 1),
            ("irrelevance", 3, 3),
            ("live_irrelevance", 0, 1),
        ]),
    ]);
    assert_eq!(aggregate(&lopsided), whole, "lopsided split diverged");

    // Four ways, the shape actually shipped.
    let four = union(&[
        map(&[("simple_python", 1, 1), ("live_simple", 1, 1)]),
        map(&[("simple_python", 1, 1), ("live_simple", 1, 1)]),
        map(&[
            ("simple_java", 0, 1),
            ("live_simple", 1, 1),
            ("irrelevance", 2, 2),
        ]),
        map(&[
            ("multiple", 0, 1),
            ("live_multiple", 0, 1),
            ("irrelevance", 1, 1),
            ("live_irrelevance", 0, 1),
        ]),
    ]);
    assert_eq!(aggregate(&four), whole, "four-way split diverged");
}

/// And the negative: averaging the SHARD SCORES rather than the counts really
/// does give a different answer. Without this, the test above could be passing
/// for the trivial reason that this data is insensitive to weighting.
#[test]
fn averaging_shard_scores_would_have_been_wrong() {
    let a = map(&[("simple_python", 1, 1)]);
    let b = map(&[
        ("simple_python", 1, 1),
        ("simple_java", 0, 1),
        ("multiple", 0, 1),
        ("live_simple", 3, 3),
        ("live_multiple", 0, 1),
        ("irrelevance", 3, 3),
        ("live_irrelevance", 0, 1),
    ]);
    let correct = aggregate(&union(&[a.clone(), b.clone()]));
    let naive_normalized = (aggregate(&a).normalized_single_turn_score
        + aggregate(&b).normalized_single_turn_score)
        / 2.0;
    assert_ne!(
        naive_normalized, correct.normalized_single_turn_score,
        "if these matched, this fixture could not detect a wrong aggregation"
    );
}

/// A category with no samples at all must DROP OUT, changing normalized's
/// divisor — `score.py` does `if not present: continue`, and a shard-shaped
/// view that silently kept a zero would score differently.
#[test]
fn an_absent_category_drops_out_rather_than_scoring_zero() {
    let no_hallucination = map(&[("simple_python", 1, 2), ("live_simple", 1, 2)]);
    let a = aggregate(&no_hallucination);
    assert!(
        !a.category_scores.contains_key("hallucination"),
        "{:?}",
        a.category_scores
    );
    // mean of the two present categories (50, 50), not of three including a 0.
    assert_eq!(a.normalized_single_turn_score, 50.0, "{a:?}");
}

/// The three `simple_*` subsets collapse to ONE term inside non_live, so a
/// subset with 1 sample weighs as much as one with 200. Pinned because it is
/// the least intuitive part of `score.py` and the easiest to "simplify" away.
#[test]
fn the_three_simple_subsets_collapse_to_a_single_non_live_term() {
    // simple_python 0/100, simple_java 1/1, simple_javascript 1/1 -> simple term
    // = mean(0, 1, 1) = 0.667, NOT 2/102.
    let m = map(&[
        ("simple_python", 0, 100),
        ("simple_java", 1, 1),
        ("simple_javascript", 1, 1),
    ]);
    let a = aggregate(&m);
    let nl = a.category_scores["non_live"];
    assert!(
        (nl - 66.67).abs() < 0.01,
        "expected the unweighted collapse (66.67), got {nl}"
    );
}

/// live is sample-weighted, which is the flat mean over live samples — the
/// opposite convention to hallucination, on purpose.
#[test]
fn live_is_sample_weighted_and_hallucination_is_not() {
    let live = map(&[("live_simple", 0, 100), ("live_parallel", 1, 1)]);
    let a = aggregate(&live);
    assert!(
        (a.category_scores["live"] - 0.99).abs() < 0.01,
        "sample-weighted: {a:?}"
    );

    let hall = map(&[("irrelevance", 0, 100), ("live_irrelevance", 1, 1)]);
    let b = aggregate(&hall);
    assert_eq!(
        b.category_scores["hallucination"], 50.0,
        "unweighted: {b:?}"
    );
}

#[test]
fn an_empty_tally_set_is_all_zero_rather_than_a_panic() {
    let a = aggregate(&BTreeMap::new());
    assert_eq!(a.total_samples, 0);
    assert_eq!(a.overall_accuracy, 0.0);
    assert_eq!(a.normalized_single_turn_score, 0.0);
}
