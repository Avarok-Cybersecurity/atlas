// SPDX-License-Identifier: AGPL-3.0-only

//! The union's tests do two jobs. Most pin behaviour. Two pin *facts about the
//! current system* that the design is waiting to stop being true —
//! `intent_is_redundant_for_a_crates_change` and its recipes counterpart. When
//! those fail, the union has become load-bearing, and that is progress.

use super::*;

fn real_taxonomy() -> Vec<Node> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    super::super::pr_taxonomy::load(&root).expect("the shipped taxonomy loads")
}

fn cat(s: &str) -> Vec<String> {
    parse_category(s)
}

// ── The live case ──────────────────────────────────────────────────────────

/// ★ **The one place the union is not a no-op today.** `recipes/` is outside
/// `PERF_PATHS`, so a recipe change invalidates nothing — yet a recipe sets the
/// serve flags, and those provably move decode wall (the whole GB10 ladder is
/// flags). Intent is the only thing that can ask for a leg here.
#[test]
fn the_live_case_is_recipes() {
    let roots = real_taxonomy();
    let changed = vec!["recipes/gb10/qwen3.6-27b.yaml".to_string()];

    let got = required_for(&changed, &[cat("performance/decode")], &roots);

    assert!(
        got.by_path.is_empty(),
        "recipes/ is outside PERF_PATHS; if this fails the floor grew and the \
         live case moved — re-derive which paths are still uncovered: {:?}",
        got.by_path
    );
    assert_eq!(
        got.intent_only(),
        ["agentic-webserver", "bfcl-subset", "ttft-warm-gate"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        "performance (agentic) + performance/decode (bfcl + warm TTFT) should \
         all be added by intent, since paths ask for nothing"
    );
}

/// The same change with no classification owes nothing — and that is correct,
/// not a hole to paper over. A guessed category would be worse than none.
#[test]
fn an_unclassified_change_gets_no_invented_intent() {
    let roots = real_taxonomy();
    let got = required_for(&["recipes/gb10/x.yaml".to_string()], &[], &roots);
    assert!(got.by_intent.is_empty());
    assert!(got.union().is_empty());
}

// ── The vacuity tripwires ──────────────────────────────────────────────────

/// ★ **Pins that the union currently adds NOTHING to a code PR**, for the two
/// independent reasons in the module docs: `_benches ⊆ REQUIRED`, and
/// `PERF_PATHS` contains a bare `"crates"`.
///
/// **If this test fails, that is very likely GOOD NEWS** — it means `by_path`
/// narrowed (the closure-hash work landing) and intent can now genuinely add a
/// leg. Do not "fix" it by widening paths again; re-read the union's docs,
/// confirm the narrowing was intended, and update this test to describe the new
/// world.
#[test]
fn intent_is_redundant_for_a_crates_change() {
    let roots = real_taxonomy();
    let changed = vec!["crates/spark-server/src/scheduler/mod.rs".to_string()];

    let got = required_for(&changed, &[cat("performance/scheduling")], &roots);

    assert_eq!(
        got.by_path.len(),
        super::super::coverage::REQUIRED.len(),
        "a bare `crates` in PERF_PATHS means every code change owes every gate"
    );
    assert!(
        got.intent_only().is_empty(),
        "the union is vacuous for code PRs today; intent added {:?} — if that \
         is real, read this test's doc comment before changing it",
        got.intent_only()
    );
}

// ── The safety property, now over the real union ───────────────────────────

/// `pr_taxonomy::benches_may_only_add` proves `benches_for` is monotone along a
/// path. That is *not* the same claim as this one, which is the one the gate
/// actually depends on: whatever intent says, the path-derived floor survives
/// intact.
#[test]
fn intent_can_never_remove_a_path_derived_gate() {
    let roots = real_taxonomy();
    let changed = vec!["kernels/gb10/common/paged_decode_attn_fp8.cu".to_string()];
    let floor = required_for(&changed, &[], &roots).by_path;
    assert!(!floor.is_empty(), "a kernels/ change must owe something");

    // Every category in the tree, including the ones that declare nothing.
    for category in [
        "documentation/reference",
        "infrastructure/ci",
        "unknown",
        "correctness/kv-cache",
        "a-category-that-was-renamed",
    ] {
        let got = required_for(&changed, &[cat(category)], &roots);
        assert!(
            floor.is_subset(&got.union()),
            "classifying as {category} DROPPED {:?}",
            floor.difference(&got.union()).collect::<Vec<_>>()
        );
    }
}

// ── Classifier instability ─────────────────────────────────────────────────

/// ★ Three live runs on one PR produced `tooling`, `performance`, `tooling`.
/// A gate whose demands change between re-runs is worse than no gate, so every
/// recorded classification counts and the result is their union — monotone and
/// replay-stable, in the adding direction.
#[test]
fn disagreeing_classifications_union_rather_than_last_wins() {
    let roots = real_taxonomy();
    let changed = vec!["recipes/x.yaml".to_string()];

    let a = required_for(&changed, &[cat("correctness/kv-cache")], &roots);
    let b = required_for(&changed, &[cat("performance/decode")], &roots);
    let both = required_for(
        &changed,
        &[cat("correctness/kv-cache"), cat("performance/decode")],
        &roots,
    );

    assert!(a.by_intent.is_subset(&both.by_intent));
    assert!(b.by_intent.is_subset(&both.by_intent));
    assert!(
        both.by_intent.len() > a.by_intent.len(),
        "two disagreeing classifications must ask for MORE than either alone"
    );
    // Order must not matter, or a re-run could read differently.
    let reversed = required_for(
        &changed,
        &[cat("performance/decode"), cat("correctness/kv-cache")],
        &roots,
    );
    assert_eq!(both, reversed);
}

// ── Parsing ────────────────────────────────────────────────────────────────

/// An empty segment would stop `benches_for`'s walk early and silently TRUNCATE
/// the path — removing benches, from a stray slash.
#[test]
fn empty_segments_are_dropped_not_descended_into() {
    assert_eq!(parse_category("performance//decode"), ["performance", "decode"]);
    assert_eq!(parse_category("performance/decode/"), ["performance", "decode"]);
    assert_eq!(parse_category(" performance / decode "), ["performance", "decode"]);
    assert!(parse_category("").is_empty());
    assert!(parse_category("///").is_empty());
}

/// A truncating parse is not hypothetical — prove the failure it prevents.
#[test]
fn a_truncated_path_would_lose_benches() {
    let roots = real_taxonomy();
    let full = super::super::pr_taxonomy::benches_for(&roots, &cat("performance/decode"));
    let truncated =
        super::super::pr_taxonomy::benches_for(&roots, &["performance".to_string(), String::new()]);
    assert!(
        truncated.len() < full.len(),
        "an empty segment must actually cost benches, or this guard is theatre"
    );
}
