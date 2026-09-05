// SPDX-License-Identifier: AGPL-3.0-only

use super::super::{PrFacts, render, views};
use super::*;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace layout")
        .to_path_buf()
}

fn pr(number: u64, paths: &[&str]) -> PrFacts {
    PrFacts {
        number,
        title: format!("pr {number}"),
        author: "someone".into(),
        draft: false,
        merged: false,
        paths_unknown: false,
        changed_paths: paths.iter().map(|s| s.to_string()).collect(),
    }
}

const FLAGSHIP: &str = "kernels/gb10/qwen3.6-27b/nvfp4/w4a4_gemm.cu";
const MOE: &str = "kernels/gb10/qwen3.6-35b-a3b/nvfp4/x.cu";

// ---------------------------------------------------------------------------
// The chart leads, and it is bounded
// ---------------------------------------------------------------------------

/// ★ The chart is the conclusion; it must be the first thing in the comment,
/// before the advisory line and before every section.
#[test]
fn the_chart_is_the_first_thing_in_the_comment() {
    let root = repo_root();
    let body = render(&root, &[pr(1, &[FLAGSHIP]), pr(2, &[MOE])]);
    assert!(
        body.starts_with(
            "<!-- atlas-pr-telemetry:start -->\n## Open-PR telemetry\n\n```mermaid\ngitGraph\n  \
             commit id: \"main\"\n"
        ),
        "only the replacement marker and title may precede the chart: {body}"
    );
}

/// ★ The bound actually truncates: past CHART_PR_BOUND open PRs, the chart
/// stops growing instead of shrinking below legibility.
#[test]
fn the_chart_is_bounded_to_chart_pr_bound() {
    let root = repo_root();
    let prs: Vec<PrFacts> = (101..=112).map(|n| pr(n, &[FLAGSHIP])).collect();
    let body = render(&root, &prs);
    assert_eq!(
        body.matches("  branch pr-").count(),
        CHART_PR_BOUND,
        "exactly the bound, not all 12"
    );
}

/// ★ Truncation is disclosed, never silent: "showing N of M" plus the names
/// of what was dropped. A chart that silently drops points is the same defect
/// class as a check that silently passes.
#[test]
fn truncation_is_disclosed_with_showing_n_of_m() {
    let root = repo_root();
    let prs: Vec<PrFacts> = (101..=112).map(|n| pr(n, &[FLAGSHIP])).collect();
    let body = render(&root, &prs);
    assert!(
        body.contains("Showing 10 of 12 open PRs"),
        "the caption must say showing N of M: {body}"
    );
    assert!(
        body.contains("Not charted: #111, #112."),
        "the dropped PRs must be named: {body}"
    );
}

/// The negative arm: under the bound nothing is dropped and the caption says
/// so — N equals M and no "Not charted" appears.
#[test]
fn a_small_input_is_not_truncated() {
    let root = repo_root();
    let body = render(&root, &[pr(1, &[FLAGSHIP]), pr(2, &[MOE])]);
    assert!(
        body.contains("Showing 2 of 2 open PRs, left to right in recommended merge order.\nOrder:"),
        "{body}"
    );
    assert!(!body.contains("Not charted"), "nothing was dropped");
}

/// ★ Zero orderable PRs must not emit a degenerate diagram (a gitGraph with
/// no branches renders as a lone dot that reads as broken).
#[test]
fn only_merged_prs_yield_a_note_not_a_degenerate_chart() {
    let root = repo_root();
    let mut merged = pr(1, &[FLAGSHIP]);
    merged.merged = true;
    let v = views(&root, &[merged]);
    assert_eq!(
        render_order_chart(&v),
        "_No open, non-draft PRs to order._\n"
    );
}

// ---------------------------------------------------------------------------
// The ordering rule
// ---------------------------------------------------------------------------

/// Rule leg 1: fewest conflict partners first. #3 contends with nobody, so it
/// leads even though every PR here re-opens exactly one target.
#[test]
fn the_uncontended_pr_is_recommended_first() {
    let root = repo_root();
    let v = views(
        &root,
        &[pr(1, &[FLAGSHIP]), pr(2, &[FLAGSHIP]), pr(3, &[MOE])],
    );
    assert_eq!(merge_order(&v), vec![3, 1, 2]);
}

/// Rule leg 2: a whole-repo diff re-opens every gate, so it contends with
/// everything and counts as the widest — it goes last.
#[test]
fn a_whole_repo_pr_orders_last() {
    let root = repo_root();
    let v = views(
        &root,
        &[
            pr(1, &["crates/spark-model/src/lib.rs"]),
            pr(2, &[FLAGSHIP]),
        ],
    );
    assert_eq!(
        merge_order(&v),
        vec![2, 1],
        "whole-repo after the narrow PR"
    );
}

/// ★ Merged PRs have landed and drafts cannot land; ranking either makes the
/// suggestion wrong on its face (the old order ranked both).
#[test]
fn merged_and_draft_prs_are_not_ranked_or_charted() {
    let root = repo_root();
    let mut merged = pr(1, &[FLAGSHIP]);
    merged.merged = true;
    let mut draft = pr(2, &[FLAGSHIP]);
    draft.draft = true;
    let open = pr(3, &[FLAGSHIP]);
    let v = views(&root, &[merged.clone(), draft.clone(), open.clone()]);
    assert_eq!(merge_order(&v), vec![3], "only the open, non-draft PR");
    let body = render(&root, &[merged, draft, open]);
    assert!(!body.contains("branch pr-1\n"), "merged not charted");
    assert!(!body.contains("branch pr-2\n"), "draft not charted");
    assert!(body.contains("branch pr-3\n"), "open PR charted");
}

/// Non-kernel diffs are whole-repo, so they contend with everything gated —
/// including each other. With partners tied, the smaller diff goes first and
/// the narrow kernel PR still leads on breadth.
#[test]
fn whole_repo_prs_contend_and_tie_break_by_size() {
    let root = repo_root();
    let v = views(
        &root,
        &[
            pr(1, &["docs/adr/README.md", "docs/adr/0002.md"]),
            pr(2, &["docs/other.md"]),
            pr(3, &[MOE]),
        ],
    );
    // All three contend pairwise (two whole-repo diffs, one gated kernel PR),
    // so partners tie at 2; #3 wins on breadth (1 target vs whole-repo), then
    // #2's one-path diff precedes #1's two.
    assert_eq!(merge_order(&v), vec![3, 2, 1]);
}

// ---------------------------------------------------------------------------
// The chart's syntax survives hostile input
// ---------------------------------------------------------------------------

/// gitGraph delimits commit ids with double quotes and has no escape, so a
/// quote in a PR title must fold to an apostrophe or the diagram dies and
/// GitHub degrades the whole block to a raw code fence.
#[test]
fn a_hostile_title_cannot_break_the_chart() {
    let root = repo_root();
    let mut hostile = pr(9, &[FLAGSHIP]);
    hostile.title = "evil \"quote\"\ninject".into();
    let v = views(&root, &[hostile]);
    assert_eq!(commit_label(&v[0]), "#9 evil 'quote' inject");
}

/// Long titles are clipped so ten branches stay legible side by side.
#[test]
fn a_long_title_is_clipped_in_the_chart() {
    let root = repo_root();
    let mut long = pr(9, &[FLAGSHIP]);
    long.title = "a".repeat(80);
    let v = views(&root, &[long]);
    assert_eq!(commit_label(&v[0]), format!("#9 {}…", "a".repeat(24)));
}

// ---------------------------------------------------------------------------
// Next steps
// ---------------------------------------------------------------------------

/// ★ The section is dynamic across a merge: while both PRs are open, the head
/// of the order is the recommendation; once one merges, the survivor is
/// flagged for re-gating because its baseline moved.
#[test]
fn next_steps_change_when_a_partner_merges() {
    let root = repo_root();
    let before = render(&root, &[pr(1, &[FLAGSHIP]), pr(2, &[FLAGSHIP])]);
    assert!(before.contains("**Merge next: #1**"), "{before}");
    assert!(
        !before.contains("Re-gate before merging"),
        "nothing merged yet, nothing to re-gate: {before}"
    );

    let mut merged = pr(2, &[FLAGSHIP]);
    merged.merged = true;
    let after = render(&root, &[pr(1, &[FLAGSHIP]), merged]);
    assert!(after.contains("**Merge next: #1**"), "{after}");
    assert!(
        after.contains("Re-gate before merging:** #1"),
        "the merged partner moved #1's baseline: {after}"
    );
}

/// A merged PR that shipped without a promotion-candidate gate is a concrete
/// next action, and the section says so.
#[test]
fn next_steps_surface_merged_promotion_debt() {
    let root = repo_root();
    let mut merged = pr(7, &["crates/spark-server/src/scheduler/mod.rs"]);
    merged.merged = true;
    let body = render(&root, &[pr(1, &[FLAGSHIP]), merged]);
    assert!(
        body.contains("Discharge promotion debt:** merged #7"),
        "the merged debtor must be named: {body}"
    );
}

/// ★ Honesty clause: the section must state what its inputs cannot see, so a
/// reader never mistakes "derived from paths" for "checked CI".
#[test]
fn next_steps_admit_what_they_cannot_know() {
    let root = repo_root();
    let v = views(&root, &[pr(1, &[FLAGSHIP])]);
    let body = render_next_steps(&v);
    assert!(
        body.ends_with(
            "\n_Derived only from changed paths and merge state. This section cannot see check \
             or required-context status, review state, true git mergeability, or whether one PR \
             unblocks another — those live on each PR._\n"
        ),
        "the complete limits must close the section: {body}"
    );
}

// ---------------------------------------------------------------------------
// The shared bound reaches the table cells
// ---------------------------------------------------------------------------

/// ★ The same X bounds every per-PR list: a cell with two hundred refs is as
/// unreadable as a two-hundred-node chart.
#[test]
fn cap_prs_caps_at_the_bound_and_discloses_the_rest() {
    let numbers: Vec<u64> = (1..=12).collect();
    let capped = cap_prs(&numbers);
    assert_eq!(capped, "#1, #2, #3, #4, #5, #6, #7, #8, #9, #10 … +2 more");
    assert_eq!(cap_prs(&[1, 2]), "#1, #2", "no noise under the bound");
}

/// The target table's cells go through the cap, so a shared-kernel storm
/// cannot turn the table into a wall of refs.
#[test]
fn target_table_cells_are_bounded() {
    let root = repo_root();
    let prs: Vec<PrFacts> = (101..=112).map(|n| pr(n, &[FLAGSHIP])).collect();
    let body = render(&root, &prs);
    let row = body
        .lines()
        .find(|l| l.starts_with("| `gb10/qwen3.6-27b/nvfp4` |"))
        .expect("the flagship target row rendered");
    assert!(row.contains("+2 more"), "cell capped and disclosed: {row}");
    assert_eq!(row.matches('#').count(), CHART_PR_BOUND, "{row}");
}
