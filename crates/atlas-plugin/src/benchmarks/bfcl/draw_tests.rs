// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// The real BFCL v4 single-turn per-subset counts.
///
/// These pin the draw: the golden config has to produce n = 995 from THESE
/// numbers, and it only does so because `live_relevance` (16) is excluded by
/// the category selection. If bfcl-eval ever ships different counts, the
/// benchmark reports the n it actually drew and this test is what says the
/// arithmetic still matches the reference.
fn real_totals() -> BTreeMap<String, usize> {
    super::reference_subset_totals()
}

#[test]
fn the_golden_draw_is_exactly_995() {
    let p = plan(&DrawSpec::golden(), &real_totals());
    assert_eq!(
        total(&p),
        995,
        "golden draw must be the MLPerf n=995: {p:?}"
    );
}

/// The echolp draw has to produce n = 1004 from the same totals, or the 35B
/// gate's baseline (84.66 / 83.32, measured on that draw) does not describe it.
///
/// A draw we did not reproduce is exactly as dangerous as scoring one draw
/// against another's threshold: it yields a plausible number that means nothing.
#[test]
fn the_echolp_draw_is_exactly_1004() {
    let p = plan(&DrawSpec::echolp(), &real_totals());
    assert_eq!(
        total(&p),
        1004,
        "echolp draw must be n=1004 or its baseline does not apply: {p:?}"
    );
}

/// The two draws must differ in COMPOSITION, not merely in size — that
/// difference is why each needs its own baseline.
#[test]
fn golden_and_echolp_are_different_draws() {
    let g: BTreeMap<String, usize> = plan(&DrawSpec::golden(), &real_totals())
        .into_iter()
        .collect();
    let e: BTreeMap<String, usize> = plan(&DrawSpec::echolp(), &real_totals())
        .into_iter()
        .collect();
    assert_ne!(g, e, "the draws must not collapse onto each other");
    // live is weighted 23% vs 10%, so echolp takes strictly more live_multiple.
    assert!(
        e["live_multiple"] > g["live_multiple"],
        "echolp must weight `live` more heavily: {} vs {}",
        e["live_multiple"],
        g["live_multiple"]
    );
}

#[test]
fn the_golden_per_subset_counts_match_the_reference_rule() {
    let p: BTreeMap<String, usize> = plan(&DrawSpec::golden(), &real_totals())
        .into_iter()
        .collect();
    let expected: BTreeMap<String, usize> = [
        // hallucination @10%: int(240*.10)=24, int(884*.10)=88
        ("irrelevance", 24),
        ("live_irrelevance", 88),
        // live @10%: int(1053*.10)=105, int(258*.10)=25
        ("live_multiple", 105),
        // floor 25 takes these whole rather than collapsing them to 1 and 2
        ("live_parallel", 16),
        ("live_parallel_multiple", 24),
        ("live_simple", 25),
        // non_live @62%
        ("multiple", 124),
        ("parallel", 124),
        ("parallel_multiple", 124),
        ("simple_java", 62),
        ("simple_javascript", 31),
        ("simple_python", 248),
    ]
    .into_iter()
    .map(|(subset, count)| (subset.to_string(), count))
    .collect();
    assert_eq!(p, expected);
}

#[test]
fn live_relevance_is_excluded_by_the_category_selection() {
    let p: BTreeMap<String, usize> = plan(&DrawSpec::golden(), &real_totals())
        .into_iter()
        .collect();
    assert!(
        !p.contains_key("live_relevance"),
        "live_relevance belongs to no scored category; including it makes n=1011, not 995"
    );
    assert_eq!(category_of("live_relevance"), None);
}

#[test]
fn the_full_draw_keeps_the_golden_composition() {
    let p = plan(&DrawSpec::full(), &real_totals());
    // Every sample of the three scored categories: 3641 total minus the 16
    // uncategorised live_relevance rows.
    assert_eq!(total(&p), 3625);
    assert!(!p.iter().any(|(s, _)| s == "live_relevance"));
}

#[test]
fn an_empty_category_selection_takes_everything_including_live_relevance() {
    let spec = DrawSpec {
        categories: Vec::new(),
        category_pct: BTreeMap::new(),
        subset_floor: None,
    };
    assert_eq!(total(&plan(&spec, &real_totals())), 3641);
}

#[test]
fn a_subset_never_collapses_to_zero() {
    let spec = DrawSpec {
        categories: vec!["non_live".into()],
        category_pct: [("non_live".to_string(), 0.5)].into_iter().collect(),
        subset_floor: None,
    };
    // int(50 * 0.005) = 0, floored up to 1 by the reference's max(1, …).
    assert_eq!(spec.take_count("simple_javascript", 50), 1);
}

#[test]
fn the_floor_beats_the_percentage() {
    let spec = DrawSpec::golden();
    assert_eq!(spec.take_count("live_parallel", 16), 16);
    // One over the floor and the percentage applies again.
    assert_eq!(spec.take_count("live_parallel", 26), 2);
}

#[test]
fn every_subset_maps_to_a_category_except_live_relevance() {
    let uncategorised: Vec<&str> = SINGLE_TURN_SUBSETS
        .iter()
        .copied()
        .filter(|s| category_of(s).is_none())
        .collect();
    assert_eq!(uncategorised, vec!["live_relevance"]);
}

/// The PARAMETER DEFAULTS must reproduce the pinned draw, not just the
/// `DrawSpec` constants.
///
/// ★ This is the test that was missing, and its absence cost a 3.5-hour run.
/// `configure` rebuilds the spec from parameter defaults, so `DrawSpec::echolp()`
/// being correct proves nothing about what a default run actually draws. The
/// echolp variant shipped with `subset_floor` defaulting to 0 while its spec
/// says 25, which takes `live_parallel` (16 rows) and `live_parallel_multiple`
/// (24) by percentage instead of whole and yields n=972, not 1004 — a
/// plausible score against a baseline for a different draw.
#[test]
fn the_parameter_defaults_reproduce_each_pinned_draw() {
    use crate::benchmark::Benchmark as _;
    use crate::benchmarks::bfcl::{Bfcl, Variant};

    for variant in [Variant::Subset, Variant::SubsetEcholp] {
        // Read from the variant rather than restated here: `expected_samples`
        // is what warns mid-run and what the committed baseline is checked
        // against, so a fourth copy of "995" would be a fourth thing to keep
        // in step.
        let want = variant
            .expected_samples()
            .expect("a gated variant is pinned");
        let mut b = Bfcl::new(variant);
        let defaults = crate::params::ParamValues::defaults(&b.parameters());
        b.configure(&defaults).expect("defaults must validate");
        let n = total(&plan(&b.spec, &real_totals()));
        assert_eq!(
            n, want,
            "{variant:?}: a DEFAULT run draws n={n}, but this draw is pinned at {want}. \
             Its baseline does not apply to n={n}."
        );
    }
}
