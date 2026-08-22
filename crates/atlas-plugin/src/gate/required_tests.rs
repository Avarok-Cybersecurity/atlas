// SPDX-License-Identifier: AGPL-3.0-only

//! The union's tests do two jobs. Most pin behaviour.
//! `crates_paths_split_into_fully_covered_and_not_covered_at_all` instead pins
//! a *fact about the current system* — that ordinary engine code owes every
//! gate while the gate's own machinery owes none. When its first half fails,
//! `by_path` has narrowed and the union has become load-bearing, which is
//! progress rather than a regression to undo.
//!
//! ★ Every changed path used here must be one THIS repo can actually produce.
//! An earlier version pinned `recipes/…`, which lives in a different
//! repository — a test over an unreachable input, proving nothing while reading
//! as coverage.

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

/// ★ **The live cases, and they must be paths this repo can actually produce.**
///
/// This test previously used `recipes/gb10/…yaml` and an audit caught it:
/// **this repo tracks zero `recipes/` files** — they live in the separate
/// `atlas-recipes` repo (`ci.yml`), and `invalidating_paths` diffs *this* one.
/// So the single case pinned as "the" live case was an input the gate can never
/// observe. A test over an unreachable input proves nothing while reading as
/// coverage, which is the same failure this whole module exists to prevent.
///
/// These are reachable, verified by `git ls-files`.
#[test]
fn intent_adds_where_the_paths_are_silent() {
    let roots = real_taxonomy();
    for changed in [
        "docker/gb10/Dockerfile",
        "scripts/mlperf-edge/kl_coherence_gate.py",
        "bench/bench_isl_osl.py",
        "kernels/gb10/qwen3.6-27b/BENCH.toml",
    ] {
        let got = required_for(&[changed.to_string()], &[cat("performance/decode")], &roots);
        assert!(
            got.by_path.is_empty(),
            "{changed} was expected off the invalidation floor, got {:?}",
            got.by_path
        );
        assert_eq!(
            got.intent_only(),
            [
                "agentic-webserver",
                "bfcl-subset",
                "decode-floor",
                "ttft-warm-gate"
            ]
            .iter()
            .map(|s| s.to_string())
            .collect::<BTreeSet<String>>(),
            "{changed}: intent should supply all four, since paths supply none \
             (decode-floor joined the leaf in the 2026-08-16 fill)"
        );
    }

    // ★ The promoted gates are reachable through intent too. `concurrency-sweep`
    // graduated to REQUIRED on 2026-08-15, and the 2026-08-16 fill put it on
    // `performance/scheduling` — so a diff off the invalidation floor entirely
    // (docs/) classified as a scheduling change now owes the concurrency curve,
    // the exact "10% at C=32, flat at C=1" regression paths cannot see.
    let promoted = required_for(
        &["docs/adr/0011-ep-batched-decode-optimization.md".to_string()],
        &[cat("performance/scheduling")],
        &roots,
    );
    assert!(
        promoted.by_path.is_empty(),
        "a docs diff must stay off the floor, got {:?}",
        promoted.by_path
    );
    assert!(
        promoted.intent_only().contains("concurrency-sweep"),
        "performance/scheduling must add the promoted concurrency-sweep gate; \
         intent supplied {:?}",
        promoted.intent_only()
    );
}

/// ★ **The mutation four rounds of mutation-testing missed.**
///
/// Replacing `union()`'s body with `self.by_path.clone()` — deleting the intent
/// half from the only accessor a gate would ever consume — passed every other
/// test in this file. `intent_adds_where_the_paths_are_silent` asserts
/// `intent_only()`, and `intent_can_never_remove_a_path_derived_gate` asserts
/// only `floor ⊆ union`, which `by_path.clone()` satisfies trivially.
///
/// Nothing asserted that the union is ever STRICTLY larger than `by_path`.
#[test]
fn union_actually_includes_the_intent_half() {
    let roots = real_taxonomy();
    let got = required_for(
        &["docker/gb10/Dockerfile".to_string()],
        &[cat("performance/decode")],
        &roots,
    );
    assert!(
        got.union().len() > got.by_path.len(),
        "union() must be STRICTLY larger than by_path when intent adds; \
         got union={:?} by_path={:?}",
        got.union(),
        got.by_path
    );
    for bench in &got.by_intent {
        assert!(
            got.union().contains(bench),
            "union() dropped the intent-derived {bench:?}"
        );
    }
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

/// ★ **`crates/` does NOT uniformly owe every gate, and I claimed it did.**
///
/// The module docs originally argued the union was vacuous because
/// "`PERF_PATHS` contains a bare `crates`, so any code change already
/// invalidates all ten gates". An audit refuted it: `GATE_MACHINERY` excludes
/// the whole `crates/atlas-plugin/src/gate` prefix from **all ten** gates, so
/// paths under it invalidate nothing and intent is the only source of coverage
/// there. The original test passed only because it happened to pick
/// `spark-server/scheduler`, one of the paths where the claim does hold.
///
/// Both halves are pinned here so the distinction cannot quietly collapse
/// again. If the first case ever stops owing all ten, `by_path` narrowed —
/// likely the closure-hash work landing, which is GOOD NEWS. Do not "fix" it by
/// widening paths; confirm the narrowing was intended and update this test.
#[test]
fn crates_paths_split_into_fully_covered_and_not_covered_at_all() {
    let roots = real_taxonomy();

    // Ordinary engine code: every gate, and intent adds nothing.
    let engine = required_for(
        &["crates/spark-server/src/scheduler/mod.rs".to_string()],
        &[cat("performance/scheduling")],
        &roots,
    );
    assert_eq!(
        engine.by_path.len(),
        super::super::coverage::REQUIRED.len(),
        "an ordinary crates/ path should owe every gate"
    );
    assert!(
        engine.intent_only().is_empty(),
        "intent should be redundant here; it added {:?}",
        engine.intent_only()
    );

    // The gate's own machinery: excluded from all ten by GATE_MACHINERY, so
    // the union is LIVE inside crates/ — not waiting on the closure hash.
    let machinery = required_for(
        &["crates/atlas-plugin/src/gate/telemetry.rs".to_string()],
        &[cat("performance/scheduling")],
        &roots,
    );
    assert!(
        machinery.by_path.is_empty(),
        "GATE_MACHINERY excludes crates/atlas-plugin/src/gate from every gate; \
         got {:?}",
        machinery.by_path
    );
    assert!(
        !machinery.intent_only().is_empty(),
        "with no path-derived coverage, intent must be the thing that supplies it"
    );
}

/// An empty taxonomy silently yields no intent. The natural wiring —
/// `load(root).unwrap_or_default()` — would therefore convert a LOUD taxonomy
/// parse failure (`load` hard-bails on a malformed `_benches`) into "this PR
/// implies nothing", which is a removal wearing the costume of an answer.
///
/// This pins the hazard so the wiring is written to distinguish "not
/// classified" from "could not read the taxonomy".
#[test]
fn an_empty_taxonomy_yields_no_intent_and_must_not_be_mistaken_for_an_answer() {
    let got = required_for(
        &["docker/gb10/Dockerfile".to_string()],
        &[cat("performance/decode")],
        &[],
    );
    assert!(got.by_intent.is_empty());
    assert!(got.union().is_empty());
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
    assert_eq!(
        parse_category("performance//decode"),
        ["performance", "decode"]
    );
    assert_eq!(
        parse_category("performance/decode/"),
        ["performance", "decode"]
    );
    assert_eq!(
        parse_category(" performance / decode "),
        ["performance", "decode"]
    );
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

// ── IntentSource: an abstention is not an empty answer ──────────────────────

fn ledger_dir() -> super::super::tests::tempdir::Dir {
    super::super::tests::tempdir::Dir::new()
}

fn write_events(root: &std::path::Path, pr: u64, rows: &[(&str, &str, &str)]) {
    let path = atlas_governance::ledger::path_for(root, pr);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    for (i, (run, value, status)) in rows.iter().enumerate() {
        let e = atlas_governance::event::Event {
            pr,
            head_sha: "deadbeef".into(),
            run_id: (*run).into(),
            attempt: 1,
            at: 1_786_280_000 + i as u64,
            kind: atlas_governance::event::EventKind::Category {
                value: (*value).into(),
                status: (*status).into(),
            },
        };
        atlas_governance::ledger::append(&path, &e).unwrap();
    }
}

#[test]
fn no_pr_is_not_requested_and_no_pr_is_not_a_missing_ledger() {
    let d = ledger_dir();
    assert_eq!(intent_source(d.path(), None), IntentSource::NotRequested);
    assert!(matches!(
        intent_source(d.path(), Some(7)),
        IntentSource::NotRecorded { .. }
    ));
}

/// ★ The one that would silently kill the feature. The ledger line for head X
/// is committed as a LATER commit, so it can never be in the tree at head X.
/// A read that filtered on `head_sha` would therefore return empty ALWAYS —
/// a working-looking feature that never fires. This pins that every row counts.
#[test]
fn every_recorded_category_counts_regardless_of_head_sha() {
    let d = ledger_dir();
    write_events(
        d.path(),
        7,
        &[
            ("100", "performance/decode", "ok"),
            ("101", "correctness/kv-cache", "ok"),
        ],
    );
    let IntentSource::Recorded { categories, .. } = intent_source(d.path(), Some(7)) else {
        panic!("expected Recorded");
    };
    assert_eq!(categories.len(), 2, "both rows must count: {categories:?}");
}

/// An outage is not a classification. The day the fallback root gains
/// `_benches`, treating an `error` row as intent would turn a 429 into a GPU
/// bill.
#[test]
fn error_and_abstain_rows_are_counted_but_never_treated_as_intent() {
    let d = ledger_dir();
    write_events(
        d.path(),
        7,
        &[
            ("100", "performance/decode", "ok"),
            ("101", "unknown", "abstain"),
            ("102", "unknown", "error"),
            ("103", "performance", "partial"),
        ],
    );
    let IntentSource::Recorded {
        categories,
        skipped,
    } = intent_source(d.path(), Some(7))
    else {
        panic!("expected Recorded");
    };
    assert_eq!(skipped, 2, "abstain + error");
    assert!(
        categories.contains(&vec!["performance".to_string()]),
        "a `partial` row is real intent — its matched prefix carries ancestor \
         _benches by the union rule: {categories:?}"
    );
}

/// A ledger holding only abstentions is NOT recorded — it must not read as a
/// confident empty classification.
#[test]
fn a_ledger_of_only_abstentions_reads_as_not_recorded() {
    let d = ledger_dir();
    write_events(d.path(), 7, &[("100", "unknown", "error")]);
    assert!(matches!(
        intent_source(d.path(), Some(7)),
        IntentSource::NotRecorded { .. }
    ));
}

/// One corrupt byte must not fail an advisory consumer — and must not read as
/// "no intent" either.
#[test]
fn a_malformed_ledger_line_degrades_and_is_distinguishable_from_empty() {
    let d = ledger_dir();
    let path = atlas_governance::ledger::path_for(d.path(), 7);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "{\"pr\":7,\"head_sha\"\n").unwrap();
    let got = intent_source(d.path(), Some(7));
    assert!(
        matches!(got, IntentSource::Degraded { .. }),
        "got {got:?} — a corrupt ledger must be Degraded, never NotRecorded"
    );
}

/// `report` must carry the provenance through, and a non-Recorded source must
/// contribute nothing to the intent half.
#[test]
fn only_a_recorded_source_contributes_intent() {
    let roots = real_taxonomy();
    let changed = vec!["docker/gb10/Dockerfile".to_string()];
    for source in [
        IntentSource::NotRequested,
        IntentSource::NotRecorded { ledger: "x".into() },
        IntentSource::Degraded {
            reason: "boom".into(),
        },
    ] {
        let r = report(&changed, source.clone(), &roots);
        assert!(r.set.by_intent.is_empty(), "{source:?} contributed intent");
        assert_eq!(r.source, source, "provenance must survive");
    }
    let r = report(
        &changed,
        IntentSource::Recorded {
            categories: vec![cat("performance/decode")],
            skipped: 0,
        },
        &roots,
    );
    assert!(!r.set.by_intent.is_empty());
}
