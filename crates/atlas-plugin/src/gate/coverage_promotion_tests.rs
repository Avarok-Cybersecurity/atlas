// SPDX-License-Identifier: AGPL-3.0-only

//! Promotion-candidate debt, and the boundary entries the intent half needs.
//!
//! Split out of `coverage_map_tests.rs` when that file crossed the repo's
//! 500-line ceiling. Same subject, different question: `coverage_map_tests`
//! asks what the REQUIRED gates invalidate; this asks what a gate that is not
//! required yet would still have wanted to see.

use super::coverage;

// ── Promotion debt: "ungated" must never read as "unaffected" ───────────────

/// A candidate naming a benchmark nobody runs would produce a debt row that can
/// never be discharged — worse than no row, because it trains people to ignore
/// the column.
#[test]
fn every_promotion_candidate_is_a_registered_benchmark() {
    let known: std::collections::BTreeSet<&str> =
        crate::registry::all().iter().map(|d| d.id).collect();
    assert_eq!(
        coverage::PROMOTION_CANDIDATES
            .iter()
            .map(|gate| gate.id)
            .collect::<Vec<_>>(),
        ["cross-contamination"],
        "promotion tracking must not pass vacuously or gain an unreviewed candidate"
    );
    for gate in coverage::PROMOTION_CANDIDATES {
        assert!(
            known.contains(gate.id),
            "{} is a promotion candidate but not a registered benchmark",
            gate.id
        );
        assert!(
            !coverage::REQUIRED.iter().any(|r| r.id == gate.id),
            "{} is BOTH required and a promotion candidate — it cannot be owed \
             and excused at once",
            gate.id
        );
    }
}

/// ★ The mechanism, proven against a synthetic candidate rather than waiting
/// for `memory-convergence` to exist. Without this the list is empty, every
/// assertion is vacuous, and the feature would ship untested — the exact shape
/// of the dead code this campaign keeps finding.
#[test]
fn a_candidate_accrues_debt_exactly_where_its_coverage_says() {
    // Mirrors what a real candidate looks like: excused from the gate dir,
    // owed for engine code.
    let candidate = coverage::GateCoverage {
        id: "synthetic-candidate",
        excludes: &[],
    };
    let owed = |p: &str| coverage::invalidates(&candidate, p);

    assert!(
        owed("crates/spark-server/src/scheduler/mod.rs"),
        "engine code must accrue debt"
    );
    assert!(
        owed("kernels/gb10/common/paged_decode_attn_fp8.cu"),
        "kernel code must accrue debt"
    );
    assert!(
        !owed("docs/adr/0014-pr-intent-taxonomy-and-the-required-union.md"),
        "docs must not"
    );
    assert!(!owed("site/index.html"), "site must not");
}

/// ★ The debt column is LIVE: `cross-contamination` is registered as a
/// candidate, so a merge touching engine code owes it (positive) and a merge
/// touching nothing on the boundary owes nothing (negative). This replaces the
/// empty-list placeholder test, per that test's own instruction.
#[test]
fn the_contamination_candidate_accrues_debt_for_engine_changes() {
    assert!(
        coverage::PROMOTION_CANDIDATES
            .iter()
            .any(|g| g.id == "cross-contamination"),
        "the cross-contamination candidate must be registered"
    );
    let owed = coverage::promotion_debt(["crates/spark-server/src/scheduler/mod.rs"]);
    assert_eq!(
        owed,
        ["cross-contamination"],
        "a scheduler change is exactly the kind of edit that can cross-wire \
         concurrent requests; it must accrue debt, got {owed:?}"
    );
    assert!(
        coverage::promotion_debt(["docs/adr/README.md", "site/index.html"]).is_empty(),
        "off-boundary paths must owe nothing"
    );
}

/// A change to the candidate's OWN driver re-opens the candidate — its
/// exclusion list must never contain its own directory, or improving the
/// detector would silently excuse re-proving it.
#[test]
fn the_candidate_is_owed_for_its_own_driver_and_not_for_other_drivers() {
    let owed =
        coverage::promotion_debt(["crates/atlas-plugin/src/benchmarks/contamination/driver.rs"]);
    assert_eq!(owed, ["cross-contamination"]);
    assert!(
        coverage::promotion_debt(["crates/atlas-plugin/src/benchmarks/ttft/descriptors.rs"])
            .is_empty(),
        "another benchmark's driver cannot change what this detector measures"
    );
}

/// ★ PROMOTED 2026-08-15: `decode-floor` and `concurrency-sweep` graduated
/// from this list to [`coverage::REQUIRED`] once their calibration
/// preconditions were met (12-run sigma set for the floor; a measured n=3
/// ladder on the pinned instrument for the sweep). Promotion must not have
/// weakened anything: what used to accrue DEBT now INVALIDATES — engine and
/// kernel paths, each gate's own driver (the pins are the benchmark, and the
/// decode-floor driver is a directory since the C1 split), and the usage
/// plumbing the accept pin reads. This audit narrows only the flat concurrency
/// driver; the decode-floor driver's existing coverage is unchanged here.
#[test]
fn the_promoted_gates_invalidate_where_they_used_to_accrue_debt() {
    for id in ["decode-floor", "concurrency-sweep"] {
        assert!(
            coverage::REQUIRED.iter().any(|g| g.id == id),
            "{id} must be REQUIRED after promotion"
        );
        assert!(
            !coverage::PROMOTION_CANDIDATES.iter().any(|g| g.id == id),
            "{id} must have left the candidate list — owed and excused at once is a contradiction"
        );
        assert!(
            !coverage::NOT_REQUIRED.iter().any(|(n, _)| *n == id),
            "{id} must not be excused any more"
        );
    }
    for path in [
        "crates/spark-server/src/scheduler/mod.rs",
        "kernels/gb10/common/paged_decode_attn_fp8.cu",
        "crates/spark-server/src/openai/encode_stream.rs",
        "crates/atlas-plugin/src/benchmarks/decode_floor/mod.rs",
    ] {
        let hit = coverage::invalidated_by([path]);
        assert!(
            hit.contains(&"decode-floor") && hit.contains(&"concurrency-sweep"),
            "{path} must invalidate both promoted gates: {hit:?}"
        );
    }
    for path in [
        "crates/atlas-plugin/src/benchmarks/concurrency.rs",
        "crates/atlas-plugin/src/benchmarks/concurrency_verdict.rs",
    ] {
        assert_eq!(
            coverage::invalidated_by([path]),
            ["concurrency-sweep"],
            "the flat concurrency driver belongs to the concurrency instrument only: {path}"
        );
    }
    let hit = coverage::invalidated_by(["crates/atlas-plugin/src/benchmarks/bfcl/report.rs"]);
    assert!(
        !hit.contains(&"decode-floor") && !hit.contains(&"concurrency-sweep"),
        "the BFCL driver can change neither the decode rate nor the ladder: {hit:?}"
    );
}

/// Candidate exclusions are held to the same bar as required-gate exclusions:
/// a written rationale, a prefix that exists, and a prefix that is actually on
/// the boundary (an off-boundary exclusion is a rule with no effect).
#[test]
fn candidate_exclusions_meet_the_required_gate_bar() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate")
        .to_path_buf();
    assert!(!coverage::PROMOTION_CANDIDATES.is_empty());
    for gate in coverage::PROMOTION_CANDIDATES {
        for ex in gate.excludes {
            assert!(
                ex.rationale.trim().len() > 20,
                "{} excludes {} with no real rationale",
                gate.id,
                ex.prefix
            );
            assert!(
                root.join(ex.prefix).exists(),
                "{} excludes {}, which does not exist",
                gate.id,
                ex.prefix
            );
            assert!(
                coverage::on_boundary(ex.prefix),
                "{} excludes {}, which is off the boundary — the rule does nothing",
                gate.id,
                ex.prefix
            );
        }
    }
}

/// ★ The intent half's coverage policy lives OUTSIDE `PERF_PATHS`, so before it
/// joined [`BOUNDARY_FILES`] a PR could delete every `_benches` line in
/// `.github/pr-taxonomy.json` and invalidate NOTHING — silently shrinking what
/// intent adds. That is the lock-whose-key-is-kept-inside-it shape this list
/// exists to close, left unapplied to the half added later.
///
/// This also pins the mechanism the entry depends on: `invalidates` consults
/// `BOUNDARY_FILES` BEFORE `on_boundary`, so an off-`PERF_PATHS` entry works.
/// If that order were ever flipped, the entry would silently stop doing
/// anything and this test is the only thing that would notice.
#[test]
fn the_taxonomy_and_the_union_are_on_the_boundary() {
    for path in [
        ".github/pr-taxonomy.json",
        "crates/atlas-plugin/src/gate/required.rs",
    ] {
        assert_eq!(
            coverage::invalidated_by([path]),
            super::REQUIRED_GATES,
            "{path} decides what the gate requires; it must re-open EVERY gate"
        );
    }

    // ★ And this is WHY the taxonomy entry is load-bearing rather than
    // decorative: it is not under any PERF_PATH, so `on_boundary` is false and
    // the ONLY thing that catches it is `invalidates`' boundary-file check
    // running FIRST. Flip that order and the entry silently stops working.
    assert!(
        !coverage::on_boundary(".github/pr-taxonomy.json"),
        "the taxonomy is off PERF_PATHS — if that changes, the assertion above \
         starts passing for a different reason than the one documented"
    );
    // `required.rs` is under `crates`, so it would invalidate anyway. Its
    // BOUNDARY_FILES entry is what makes it invalidate even for gates whose
    // GATE_MACHINERY exclusion would otherwise forgive the whole gate dir.
    assert!(coverage::on_boundary(
        "crates/atlas-plugin/src/gate/required.rs"
    ));
}
