// SPDX-License-Identifier: AGPL-3.0-only

//! A PR whose changed files could not be read must never render as a PR that
//! changes nothing.
//!
//! The collector calls `pulls/N/files` once per PR. That call can fail — a
//! 404 on a PR that vanished mid-run, a 502, a token that lost a scope — and
//! it used to fall through to `changed_paths: []` with no warning. Everything
//! downstream reads an empty list as a measurement: zero targets, zero owners,
//! zero promotion debt, zero collisions, and first place in the merge order
//! (fewest partners, fewest targets, smallest diff). An API error therefore
//! published the single most reassuring record this view can produce, and did
//! it into a comment on the tracking issue.
//!
//! These tests pin the opposite: unknown is maximal, and every row it produces
//! says it is assumed.

use super::*;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace layout")
        .to_path_buf()
}

fn blind(number: u64) -> PrFacts {
    PrFacts {
        number,
        title: format!("pr {number}"),
        author: "someone".into(),
        draft: false,
        merged: false,
        // Exactly the shape the failure path produces: empty, but flagged.
        changed_paths: Vec::new(),
        paths_unknown: true,
    }
}

fn known(number: u64, paths: &[&str]) -> PrFacts {
    PrFacts {
        number,
        title: format!("pr {number}"),
        author: "someone".into(),
        draft: false,
        merged: false,
        changed_paths: paths.iter().map(|s| s.to_string()).collect(),
        paths_unknown: false,
    }
}

const FLAGSHIP: &str = "kernels/gb10/qwen3.6-27b/nvfp4/w4a4_gemm.cu";

/// The load-bearing assertion: unknown paths are the MAXIMUM blast radius.
/// Before the fix this view had `targets == {}` and `whole_repo == false`.
#[test]
fn an_unreadable_diff_re_opens_every_target_not_none() {
    let root = repo_root();
    let v = &views(&root, &[blind(1)])[0];
    assert!(v.whole_repo, "unknown paths must be treated as whole-repo");
    assert_eq!(
        v.targets,
        taxon::walk(&root).into_iter().collect::<BTreeSet<_>>(),
        "an unreadable diff must re-open every target, not zero"
    );
    // An empty path list joins against no coverage table, so the debt is the
    // whole candidate list, not the empty one an absent join would yield.
    assert_eq!(
        v.promotion_debt,
        coverage::PROMOTION_CANDIDATES
            .iter()
            .map(|g| g.id)
            .collect::<Vec<_>>(),
        "unknown paths owe every promotion candidate"
    );
    // And a PR with genuinely no matching paths still owes nothing — the two
    // cases must not have collapsed into one.
    let clean = &views(&root, &[known(2, &["docs/adr/README.md"])])[0];
    assert_eq!(clean.promotion_debt, Vec::<&str>::new());
}

/// Maximal radius is only honest if the reader can tell it is assumed. Every
/// surface the unknown PR reaches must say so.
#[test]
fn the_comment_labels_every_assumed_row_as_assumed() {
    let root = repo_root();
    let body = render(&root, &[blind(7), known(8, &[FLAGSHIP])]);
    assert!(
        body.contains("⚠ **Changed files unavailable for #7.**"),
        "the banner must name the affected PRs: {body}"
    );
    let row = body
        .lines()
        .find(|l| l.starts_with("| #7"))
        .expect("the blind PR still has a row");
    assert!(
        row.contains("ALL — **assumed**, changed files unreadable"),
        "the targets cell must not read as a measurement: {row}"
    );
    assert!(
        row.contains("| unknown |"),
        "no matched owner and no paths to match are different cells: {row}"
    );
    let debt = body
        .lines()
        .find(|l| l.starts_with("| #7 |"))
        .expect("the blind PR appears in the debt table");
    assert!(
        debt.contains("**assumed** (paths unreadable)"),
        "assumed debt must not be quoted as owed: {debt}"
    );
    // The readable PR's rows stay unqualified — the labelling is targeted, not
    // a blanket disclaimer that would make every row equally unbelievable.
    let good = body
        .lines()
        .find(|l| l.starts_with("| #8"))
        .expect("the readable PR has a row");
    assert!(!good.contains("assumed"), "{good}");
}

/// The worst single consequence of the old behaviour: an empty diff sorts to
/// the FRONT of the merge order, so an API error published "merge next: #7".
#[test]
fn an_unreadable_diff_is_never_recommended_as_merge_next() {
    let root = repo_root();
    let vs = views(&root, &[blind(7), known(8, &[FLAGSHIP])]);
    assert_eq!(
        order::merge_order(&vs),
        vec![8],
        "an unrankable PR must not be ranked at all, least of all first"
    );
    let body = render(&root, &[blind(7), known(8, &[FLAGSHIP])]);
    assert!(body.contains("**Merge next: #8**"), "{body}");
    assert!(
        body.contains("**Cannot rank #7:**"),
        "exclusion from the order must be stated, not silent: {body}"
    );
}

/// Dropping the record was the other candidate fix. It is not what we do, and
/// the reason is testable: a blind PR must still appear in the collision table
/// it may well be colliding with.
#[test]
fn a_blind_pr_still_collides_with_everything() {
    let root = repo_root();
    let vs = views(&root, &[blind(7), known(8, &[FLAGSHIP])]);
    let c = collisions(&vs);
    assert!(
        !c.is_empty(),
        "a maximal-radius PR contends with every gated PR"
    );
    for (target, prs) in &c {
        assert!(prs.contains(&7), "#7 missing from `{target}`: {prs:?}");
    }
}

/// When the ONLY open PR is blind, "nothing to order" would be false. The two
/// empty-order cases must read differently.
#[test]
fn an_all_blind_run_does_not_claim_there_is_nothing_to_merge() {
    let root = repo_root();
    let body = render(&root, &[blind(7)]);
    assert!(
        !body.contains("Nothing to merge: no open, non-draft PRs."),
        "there IS an open PR; the run just cannot see it: {body}"
    );
    assert!(body.contains("**No recommendation.**"), "{body}");
    assert!(body.contains("came back unreadable"), "{body}");
}

/// The field is `#[serde(default)]`, so the collector's older records still
/// parse — but a record that carries the flag must not lose it in transit.
#[test]
fn the_marker_survives_the_json_the_workflow_actually_writes() {
    let facts: Vec<PrFacts> = serde_json::from_str(
        r#"[{"number":7,"title":"t","author":"a","draft":false,"merged":false,
             "changed_paths":[],"paths_unknown":true},
            {"number":8,"title":"t","author":"a","draft":false,"merged":false,
             "changed_paths":[]}]"#,
    )
    .expect("the workflow's shape parses");
    assert!(facts[0].paths_unknown, "the flag must round-trip");
    assert!(
        !facts[1].paths_unknown,
        "absent means known-and-empty, which is what pre-fix records mean"
    );
}
