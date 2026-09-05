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
        merged: false,
        paths_unknown: false,
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
    let gb10: BTreeSet<_> = taxon::walk(&root)
        .into_iter()
        .filter(|t| t.hardware == "gb10")
        .collect();
    assert_eq!(v.targets, gb10, "all and only gb10 targets");
    assert!(!v.whole_repo, "a kernels-only diff is not whole-repo");
}

#[test]
fn a_source_owner_change_reopens_the_owner_and_redirected_consumer() {
    let root = repo_root();
    let v = &views(&root, &[pr(2, &[FLAGSHIP])])[0];
    assert_eq!(
        v.targets,
        BTreeSet::from([
            taxon::Target {
                hardware: "gb10".into(),
                model: "qwen3.6-27b".into(),
                quant: "nvfp4".into(),
            },
            taxon::Target {
                hardware: "gb10".into(),
                model: "qwen3.8-27b".into(),
                quant: "nvfp4".into(),
            },
        ])
    );
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
    assert_eq!(
        v.targets,
        taxon::walk(&root).into_iter().collect(),
        "a whole-repository diff re-opens every target"
    );
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
    assert_eq!(v.owners, ["@SeedSource", "@rsafier", "@tbraun96"]);
}

// ---------------------------------------------------------------------------
// Collisions — the reason this exists
// ---------------------------------------------------------------------------

/// ★ Two PRs on one source owner collide on it and every redirected consumer.
#[test]
fn two_prs_touching_one_target_collide() {
    let root = repo_root();
    let v = views(&root, &[pr(1, &[FLAGSHIP]), pr(2, &[FLAGSHIP])]);
    let c = collisions(&v);
    assert_eq!(
        c,
        BTreeMap::from([
            ("gb10/qwen3.6-27b/nvfp4".into(), vec![1, 2]),
            ("gb10/qwen3.8-27b/nvfp4".into(), vec![1, 2]),
        ])
    );
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
    assert_eq!(
        c,
        BTreeMap::from([
            ("gb10/qwen3.6-27b/nvfp4".into(), vec![1, 2]),
            ("gb10/qwen3.8-27b/nvfp4".into(), vec![1, 2]),
        ]),
        "the shared change meets both the source owner and its consumer"
    );
}

#[test]
fn a_whole_repo_pr_collides_with_a_kernel_pr() {
    let root = repo_root();
    let v = views(
        &root,
        &[
            pr(1, &["crates/spark-model/src/lib.rs"]),
            pr(2, &[FLAGSHIP]),
        ],
    );
    assert_eq!(
        collisions(&v),
        BTreeMap::from([
            ("gb10/qwen3.6-27b/nvfp4".into(), vec![1, 2]),
            ("gb10/qwen3.8-27b/nvfp4".into(), vec![1, 2]),
        ])
    );
}

#[test]
fn a_merged_pr_does_not_remain_in_the_open_collision_map() {
    let root = repo_root();
    let mut merged = pr(1, &[FLAGSHIP]);
    merged.merged = true;
    let v = views(&root, &[merged, pr(2, &[FLAGSHIP])]);
    assert_eq!(collisions(&v), BTreeMap::new());
}

// ---------------------------------------------------------------------------
// Order
// ---------------------------------------------------------------------------

#[test]
fn the_narrowest_pr_is_suggested_first() {
    let root = repo_root();
    let v = views(&root, &[pr(1, &[COMMON]), pr(2, &[FLAGSHIP])]);
    assert_eq!(
        merge_order(&v),
        vec![2, 1],
        "2 targets before all gb10 targets"
    );
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
        let reopened = if matches!(
            target.to_string().as_str(),
            "gb10/qwen3.6-27b/nvfp4" | "gb10/qwen3.8-27b/nvfp4"
        ) {
            "#1"
        } else {
            "—"
        };
        assert!(body.contains(&format!("| `{target}` | {reopened} |\n")));
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
    assert_eq!(body.matches(MARKER_END).count(), 1);
}

#[test]
fn an_empty_pr_list_still_renders_a_valid_body() {
    let root = repo_root();
    assert_eq!(
        render(&root, &[]),
        format!("{MARKER_START}\n## Open-PR telemetry\n\n_No open pull requests._\n{MARKER_END}\n")
    );
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
        merged: false,
        paths_unknown: false,
        changed_paths: vec![FLAGSHIP.to_string()],
    };
    let body = render(&root, &[hostile]);
    let row = body
        .lines()
        .find(|l| l.starts_with("| #9"))
        .expect("the row rendered");
    assert_eq!(
        row,
        "| #9 evil \\| row injection | gb10 | 2 | @SeedSource @rsafier @tbraun96 |"
    );
}

#[test]
fn a_draft_is_marked_as_one() {
    let root = repo_root();
    let mut facts = pr(5, &[FLAGSHIP]);
    facts.draft = true;
    let row = render(&root, &[facts])
        .lines()
        .find(|line| line.starts_with("| #5"))
        .unwrap()
        .to_string();
    assert_eq!(
        row,
        "| #5 (draft) pr 5 | gb10 | 2 | @SeedSource @rsafier @tbraun96 |"
    );
}

/// ★ The debt section is rendered even when nothing is owed — that is the
/// whole mechanism. A section that appears only when non-empty cannot be
/// distinguished from a section nobody wired up, and "no row" then reads as
/// "no debt" whether or not the join ever ran.
#[test]
fn the_promotion_debt_section_is_always_rendered() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let prs = vec![super::PrFacts {
        number: 1,
        title: "a scheduler change".into(),
        author: "someone".into(),
        draft: false,
        merged: false,
        paths_unknown: false,
        changed_paths: vec!["crates/spark-server/src/scheduler/mod.rs".into()],
    }];
    let body = super::render(&root, &prs);
    assert!(
        body.contains(
            "### Promotion-candidate debt\n\nThese gates are NOT required, so these PRs can merge without them. Each row is coverage this repository chose not to buy — recorded so the choice stays visible rather than becoming an assumption.\n\n| PR | merged? | title | gates that wanted to run |\n|---|---|---|---|\n| #1 | not yet | a scheduler change | cross-contamination |\n"
        ),
        "the unconditional debt section must retain its policy, schema, and row: {body}"
    );
}

/// The PR's debt is computed from its own paths, so it stays correct as the
/// candidate list grows.
#[test]
fn debt_is_derived_from_the_prs_own_paths() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let prs = vec![
        super::PrFacts {
            number: 1,
            title: "docs".into(),
            author: "a".into(),
            draft: false,
            merged: false,
            paths_unknown: false,
            changed_paths: vec!["docs/adr/README.md".into()],
        },
        super::PrFacts {
            number: 2,
            title: "engine".into(),
            author: "b".into(),
            draft: false,
            merged: false,
            paths_unknown: false,
            changed_paths: vec!["crates/spark-server/src/scheduler/mod.rs".into()],
        },
    ];
    let views = super::views(&root, &prs);
    // The discrimination is real now that `cross-contamination` is a
    // candidate: the docs PR owes nothing, the engine PR owes the candidate.
    assert_eq!(views[0].promotion_debt, Vec::<&str>::new());
    assert_eq!(views[1].promotion_debt, vec!["cross-contamination"]);
}

/// A merged debt and an open debt are different things: one is coverage already
/// gone without, the other is still a warning. The column is what keeps them
/// from collapsing into one undifferentiated list.
#[test]
fn the_debt_table_distinguishes_merged_from_open() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let prs = vec![
        super::PrFacts {
            number: 1,
            title: "still open".into(),
            author: "a".into(),
            draft: false,
            merged: false,
            paths_unknown: false,
            changed_paths: vec!["crates/spark-server/src/scheduler/mod.rs".into()],
        },
        super::PrFacts {
            number: 2,
            title: "already landed".into(),
            author: "b".into(),
            draft: false,
            merged: true,
            paths_unknown: false,
            changed_paths: vec!["crates/spark-server/src/scheduler/mod.rs".into()],
        },
    ];
    let body = super::render(&root, &prs);
    assert!(
        body.contains(
            "| #1 | not yet | still open | cross-contamination |\n| #2 | **yes** | already landed | cross-contamination |\n"
        ),
        "open warning and accrued merged debt must remain distinct: {body}"
    );
}
