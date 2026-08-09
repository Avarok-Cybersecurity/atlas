// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace layout")
        .to_path_buf()
}

fn pr(number: u64, paths: &[&str]) -> PrFacts {
    PrFacts {
        number,
        title: format!("pr {number}"),
        author: "someone".into(),
        draft: false,
        changed_paths: paths.iter().map(|s| s.to_string()).collect(),
    }
}

const COMMON: &str = "kernels/gb10/common/paged_decode_attn_fp8.cu";
const FLAGSHIP: &str = "kernels/gb10/qwen3.6-27b/nvfp4/w4a4_gemm.cu";

// ---------------------------------------------------------------------------
// The view
// ---------------------------------------------------------------------------

/// A shared kernel is inherited by every target on that hardware, so it must
/// show up as the wide blast radius it is.
#[test]
fn a_common_kernel_change_reopens_every_target_on_that_hardware() {
    let root = repo_root();
    let v = &views(&root, &[pr(1, &[COMMON])])[0];
    let gb10 = taxon::walk(&root)
        .into_iter()
        .filter(|t| t.hardware == "gb10")
        .count();
    assert_eq!(v.targets.len(), gb10, "all gb10 targets");
    assert!(!v.whole_repo, "a kernels-only diff is not whole-repo");
}

#[test]
fn a_model_specific_change_reopens_only_that_model() {
    let root = repo_root();
    let v = &views(&root, &[pr(2, &[FLAGSHIP])])[0];
    assert_eq!(v.targets.len(), 1);
    assert_eq!(v.targets.iter().next().unwrap().model, "qwen3.6-27b");
}

/// ★ A diff that reaches outside `kernels/` re-opens everything, and must be
/// reported that way rather than as the handful of kernel targets it also
/// happens to touch. Reporting the small number would be the fail-open.
#[test]
fn a_diff_reaching_outside_kernels_is_marked_whole_repo() {
    let root = repo_root();
    let v = &views(
        &root,
        &[pr(3, &[FLAGSHIP, "crates/spark-model/src/lib.rs"])],
    )[0];
    assert!(v.whole_repo);
    let body = render(
        &root,
        &[pr(3, &[FLAGSHIP, "crates/spark-model/src/lib.rs"])],
    );
    assert!(
        body.contains("ALL (diff reaches outside kernels/)"),
        "the table must say ALL, not 1: {body}"
    );
}

#[test]
fn codeowners_are_resolved_from_the_changed_paths() {
    let root = repo_root();
    let v = &views(&root, &[pr(4, &["crates/spark-model/src/lib.rs"])])[0];
    assert!(!v.owners.is_empty(), "spark-model has owners in CODEOWNERS");
}

// ---------------------------------------------------------------------------
// Collisions — the reason this exists
// ---------------------------------------------------------------------------

/// ★ Two PRs on one target are each green against a baseline the other moves.
#[test]
fn two_prs_touching_one_target_collide() {
    let root = repo_root();
    let v = views(&root, &[pr(1, &[FLAGSHIP]), pr(2, &[FLAGSHIP])]);
    let c = collisions(&v);
    assert_eq!(c.len(), 1);
    assert_eq!(c["gb10/qwen3.6-27b/nvfp4"], vec![1, 2]);
}

#[test]
fn prs_on_different_targets_do_not_collide() {
    let root = repo_root();
    let v = views(
        &root,
        &[
            pr(1, &[FLAGSHIP]),
            pr(2, &["kernels/gb10/qwen3.6-35b-a3b/nvfp4/x.cu"]),
        ],
    );
    assert!(collisions(&v).is_empty());
}

/// A shared-kernel PR collides with every model-specific PR on that hardware —
/// which is exactly the situation a merge queue cannot see on its own.
#[test]
fn a_shared_kernel_pr_collides_with_every_model_pr_beneath_it() {
    let root = repo_root();
    let v = views(&root, &[pr(1, &[COMMON]), pr(2, &[FLAGSHIP])]);
    let c = collisions(&v);
    assert!(
        c.contains_key("gb10/qwen3.6-27b/nvfp4"),
        "the shared change and the model change meet on the flagship: {c:?}"
    );
}

// ---------------------------------------------------------------------------
// Order
// ---------------------------------------------------------------------------

#[test]
fn the_narrowest_pr_is_suggested_first() {
    let root = repo_root();
    let v = views(&root, &[pr(1, &[COMMON]), pr(2, &[FLAGSHIP])]);
    assert_eq!(merge_order(&v), vec![2, 1], "1 target before 22");
}

/// The order must be total and reproducible, or the comment churns on every
/// run and readers stop trusting it.
#[test]
fn the_order_is_deterministic_under_input_permutation() {
    let root = repo_root();
    let a = views(&root, &[pr(7, &[FLAGSHIP]), pr(3, &[FLAGSHIP])]);
    let b = views(&root, &[pr(3, &[FLAGSHIP]), pr(7, &[FLAGSHIP])]);
    assert_eq!(merge_order(&a), merge_order(&b));
    assert_eq!(merge_order(&a), vec![3, 7], "ties break by PR number");
}

// ---------------------------------------------------------------------------
// The comment body
// ---------------------------------------------------------------------------

/// ★ Every target is listed, always. Listing only affected ones would convert
/// "ungated" into "unaffected" by omission.
#[test]
fn every_target_appears_even_when_no_pr_touches_it() {
    let root = repo_root();
    let body = render(&root, &[pr(1, &[FLAGSHIP])]);
    for target in taxon::walk(&root) {
        assert!(
            body.contains(&format!("`{target}`")),
            "{target} missing from the target table"
        );
    }
}

/// The markers are what let the workflow rewrite one comment instead of
/// appending a new one every run.
#[test]
fn the_body_is_delimited_so_it_can_be_rewritten_in_place() {
    let root = repo_root();
    let body = render(&root, &[pr(1, &[FLAGSHIP])]);
    assert!(body.starts_with(MARKER_START));
    assert!(body.trim_end().ends_with(MARKER_END));
    assert_eq!(body.matches(MARKER_START).count(), 1);
}

#[test]
fn an_empty_pr_list_still_renders_a_valid_body() {
    let root = repo_root();
    let body = render(&root, &[]);
    assert!(body.contains("No open pull requests"));
    assert!(body.trim_end().ends_with(MARKER_END));
}

/// ★ A PR title is attacker-controlled text landing in a markdown table. A `|`
/// would break the table; a newline would break the row.
#[test]
fn pr_titles_cannot_break_the_table() {
    let root = repo_root();
    let hostile = PrFacts {
        number: 9,
        title: "evil | row\ninjection".into(),
        author: "x".into(),
        draft: false,
        changed_paths: vec![FLAGSHIP.to_string()],
    };
    let body = render(&root, &[hostile]);
    let row = body
        .lines()
        .find(|l| l.starts_with("| #9"))
        .expect("the row rendered");
    assert!(row.contains("evil \\| row injection"), "{row}");
    assert_eq!(
        row.matches(" | ").count(),
        3,
        "still a four-column row: {row}"
    );
}

#[test]
fn a_draft_is_marked_as_one() {
    let root = repo_root();
    let mut facts = pr(5, &[FLAGSHIP]);
    facts.draft = true;
    assert!(render(&root, &[facts]).contains("#5 (draft)"));
}

/// The mermaid block must be well-formed even for a single PR, where there is
/// no edge to draw.
#[test]
fn the_mermaid_graph_is_valid_with_one_pr() {
    let root = repo_root();
    let body = render(&root, &[pr(1, &[FLAGSHIP])]);
    assert!(body.contains("```mermaid\ngraph LR\n"));
    assert!(body.contains("pr1["));
    assert!(!body.contains("--> pr1"), "no edge into the only node");
}
