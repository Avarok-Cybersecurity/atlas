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
    [
        ("irrelevance", 240),
        ("live_irrelevance", 884),
        ("live_multiple", 1053),
        ("live_parallel", 16),
        ("live_parallel_multiple", 24),
        ("live_relevance", 16),
        ("live_simple", 258),
        ("multiple", 200),
        ("parallel", 200),
        ("parallel_multiple", 200),
        ("simple_java", 100),
        ("simple_javascript", 50),
        ("simple_python", 400),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
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

#[test]
fn the_golden_per_subset_counts_match_the_reference_rule() {
    let p: BTreeMap<String, usize> = plan(&DrawSpec::golden(), &real_totals())
        .into_iter()
        .collect();
    // hallucination @10%: int(240*.10)=24, int(884*.10)=88
    assert_eq!(p["irrelevance"], 24);
    assert_eq!(p["live_irrelevance"], 88);
    // live @10%: int(1053*.10)=105, int(258*.10)=25
    assert_eq!(p["live_multiple"], 105);
    assert_eq!(p["live_simple"], 25);
    // floor 25 takes these whole rather than collapsing them to 1 and 2
    assert_eq!(p["live_parallel"], 16);
    assert_eq!(p["live_parallel_multiple"], 24);
    // non_live @62%: int(400*.62)=248, int(200*.62)=124, int(100*.62)=62, int(50*.62)=31
    assert_eq!(p["simple_python"], 248);
    assert_eq!(p["multiple"], 124);
    assert_eq!(p["simple_java"], 62);
    assert_eq!(p["simple_javascript"], 31);
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
