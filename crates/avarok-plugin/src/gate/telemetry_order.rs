// SPDX-License-Identifier: AGPL-3.0-only

//! The bounded renderings: the recommended merge order, the gitGraph that
//! draws it, and the "Recommended next steps" list.
//!
//! # Why `gitGraph` and not `graph LR`
//!
//! The thing being drawn IS a git timeline: a mainline with each PR branching
//! off and merging back in the recommended order. `gitGraph` has those
//! semantics natively — `main` is a real line, every PR is a real branch to
//! the side, and left-to-right position IS the order. A `graph LR` chain of
//! boxes has none of that; it was a list wearing a diagram costume.
//!
//! # Why everything here is bounded
//!
//! A chart with two hundred nodes is not a chart, it is noise that renders —
//! unreadable and therefore effectively invisible, which is the same defect
//! class as a check that silently passes. Every per-PR rendering in the
//! comment shares [`CHART_PR_BOUND`], and every truncation says "showing N of
//! M" plus what was dropped, so a bounded view can never be mistaken for a
//! complete one.

use super::PrView;

/// The one bound every chart and per-PR list in the comment shares.
///
/// Why 10: GitHub renders comment bodies ~900px wide and scales a mermaid
/// SVG to fit, and each `gitGraph` branch-and-merge lane costs roughly
/// 80-100px of track — past ten branches the whole chart shrinks below
/// legibility, which recreates the unreadable-chart defect the bound exists
/// to prevent. Ten is also about as many "next" items as a reader acts on.
/// One constant on purpose: every bounded rendering degrades at the same
/// point, and changing the budget is a one-line edit here, never a hunt for
/// magic numbers.
pub const CHART_PR_BOUND: usize = 10;

/// True when landing either PR disturbs the other: they share a gate target,
/// so each is measured against a baseline the other will move and whichever
/// lands second must re-gate. A whole-repo diff re-opens every gate, so it
/// contends with any PR that has one.
///
/// Shared-changed-file overlap (textual conflict risk) is deliberately NOT a
/// separate clause — it is subsumed: any diff reaching outside `kernels/` is
/// whole-repo and already contends with everything gated, and two kernel
/// diffs sharing a file share that file's target.
fn contend(a: &PrView, b: &PrView) -> bool {
    let a_gated = a.whole_repo || !a.targets.is_empty();
    let b_gated = b.whole_repo || !b.targets.is_empty();
    if (a.whole_repo && b_gated) || (b.whole_repo && a_gated) {
        return true;
    }
    !a.targets.is_disjoint(&b.targets)
}

/// The PRs a merge order can contain: open, non-draft, and with a diff this
/// run could actually read. A merged PR has already landed; a draft cannot
/// land. The old order ranked both, which made the suggestion wrong on its
/// face whenever the feed carried merged PRs.
///
/// ★ A PR whose changed files came back unreadable is NOT ranked. Ranking is
/// entirely a function of what a diff touches, so ordering one we cannot see
/// would be a recommendation with no input behind it — and with an empty path
/// list it would sort to the FRONT (zero partners, zero targets, zero files)
/// and be published as "merge next". [`super::render_next_steps`] names them
/// instead, so they are excluded loudly rather than silently.
fn orderable(views: &[PrView]) -> Vec<&PrView> {
    views
        .iter()
        .filter(|v| !v.facts.merged && !v.facts.draft && !v.facts.paths_unknown)
        .collect()
}

/// The recommended merge order over open, non-draft PRs.
///
/// The rule, in tie-break order — deliberately simple enough that a reader
/// can tell at a glance when it is wrong:
/// 1. fewest conflict partners first (see the private `contend` helper) —
///    landing uncontended work first invalidates nobody's baseline and defers
///    the contended cluster to be decided together;
/// 2. fewest targets re-opened (whole-repo counts as all of them) — the
///    cheapest to re-gate if the order changes underneath it;
/// 3. smallest diff by changed-path count — a size proxy;
/// 4. lowest PR number — oldest first, and it makes the order total, so the
///    comment does not churn between runs.
///
/// What the rule does NOT know (nothing in [`super::PrFacts`] carries it):
/// whether one PR unblocks another, required-context/check state, review
/// state, or true git mergeability. Those live on each PR.
pub fn merge_order(views: &[PrView]) -> Vec<u64> {
    let open = orderable(views);
    let mut ranked: Vec<(usize, usize, usize, u64)> = open
        .iter()
        .map(|v| {
            let partners = open
                .iter()
                .filter(|o| o.facts.number != v.facts.number && contend(v, o))
                .count();
            let breadth = if v.whole_repo {
                usize::MAX
            } else {
                v.targets.len()
            };
            (
                partners,
                breadth,
                v.facts.changed_paths.len(),
                v.facts.number,
            )
        })
        .collect();
    ranked.sort_unstable();
    ranked.into_iter().map(|(_, _, _, n)| n).collect()
}

/// A bounded `#1, #2, … +k more` list. Every per-PR cell in the comment goes
/// through this so no table cell can grow into an unreadable wall of refs.
pub fn cap_prs(numbers: &[u64]) -> String {
    let shown = numbers.len().min(CHART_PR_BOUND);
    let mut s = numbers[..shown]
        .iter()
        .map(|n| format!("#{n}"))
        .collect::<Vec<_>>()
        .join(", ");
    if numbers.len() > shown {
        s.push_str(&format!(" … +{} more", numbers.len() - shown));
    }
    s
}

/// The commit label for one PR inside the chart. `gitGraph` delimits ids with
/// double quotes and has no escape sequence, so quotes fold to apostrophes;
/// the title is clipped so ten branches stay legible side by side.
fn commit_label(v: &PrView) -> String {
    const TITLE_CLIP: usize = 24; // longer labels shrink the whole SVG to fit
    let clean = v.facts.title.replace('"', "'").replace('\n', " ");
    let clipped: String = clean.chars().take(TITLE_CLIP).collect();
    let ellipsis = if clean.chars().count() > TITLE_CLIP {
        "…"
    } else {
        ""
    };
    format!("#{} {}{}", v.facts.number, clipped, ellipsis)
}

/// The chart: `main` as a line, each recommended PR as a branch off it,
/// merged back in order. Bounded by [`CHART_PR_BOUND`]; the caption always
/// says "showing N of M", and anything past the bound is named in text so the
/// truncation is visible, never silent.
pub fn render_order_chart(views: &[PrView]) -> String {
    let order = merge_order(views);
    if order.is_empty() {
        // No fence at all: an empty gitGraph (a mainline with no branches)
        // renders as a lone dot, which reads as a broken chart.
        //
        // "nothing to order" and "everything was unreadable" are different
        // facts and must not share a sentence.
        if views
            .iter()
            .any(|v| !v.facts.merged && v.facts.paths_unknown)
        {
            return "_Nothing could be ordered: the changed files of the open PR(s) came \
                    back unreadable on this run._\n"
                .to_string();
        }
        return "_No open, non-draft PRs to order._\n".to_string();
    }
    let shown = order.len().min(CHART_PR_BOUND);
    let mut out = String::from("```mermaid\ngitGraph\n  commit id: \"main\"\n");
    for number in &order[..shown] {
        let v = views
            .iter()
            .find(|v| v.facts.number == *number)
            .expect("order is drawn from these views");
        out.push_str(&format!(
            "  branch pr-{number}\n  commit id: \"{}\"\n  checkout main\n  merge pr-{number}\n",
            commit_label(v)
        ));
    }
    out.push_str("```\n\n");
    out.push_str(&format!(
        "Showing {shown} of {} open PRs, left to right in recommended merge order.",
        order.len()
    ));
    if order.len() > shown {
        out.push_str(&format!(" Not charted: {}.", cap_prs(&order[shown..])));
    }
    out.push_str(
        "\nOrder: fewest conflict partners first (a partner shares a gate target; \
         a whole-repo diff contends with every gated PR), then fewest targets \
         re-opened, then smallest diff, then lowest PR number. Drafts and merged \
         PRs are not ranked.\n",
    );
    out
}

/// "Recommended next steps" — recomputed on every run, and the workflow now
/// runs on every push to `main`, so this updates the moment a merge lands.
///
/// Everything here is derived from changed paths plus the merged flag; the
/// closing line says out loud what those inputs cannot see, so the section
/// never pretends to more knowledge than it has.
pub fn render_next_steps(views: &[PrView]) -> String {
    let mut out = String::from("\n### Recommended next steps\n\n");
    let order = merge_order(views);
    match order.first() {
        None if views
            .iter()
            .any(|v| !v.facts.merged && v.facts.paths_unknown) =>
        {
            out.push_str(
                "- **No recommendation.** Every rankable PR's changed files came \
                       back unreadable on this run.\n",
            )
        }
        None => out.push_str("- Nothing to merge: no open, non-draft PRs.\n"),
        Some(head) => {
            let v = views.iter().find(|v| v.facts.number == *head).unwrap();
            out.push_str(&format!(
                "- **Merge next: #{head}** ({}) — least disruptive open PR under the \
                 order rule above.\n",
                super::escape(&v.facts.title)
            ));
        }
    }

    // Unrankable, and said so where the reader is deciding what to merge.
    let blind: Vec<u64> = views
        .iter()
        .filter(|v| !v.facts.merged && v.facts.paths_unknown)
        .map(|v| v.facts.number)
        .collect();
    if !blind.is_empty() {
        out.push_str(&format!(
            "- **Cannot rank {}:** their changed files came back unreadable, so this \
             run has no input to order them by. They are counted as touching \
             everything elsewhere in this comment; re-run before trusting the order.\n",
            cap_prs(&blind)
        ));
    }

    // Baselines already moved: open PRs whose targets a merged PR re-opened.
    let merged: Vec<&PrView> = views.iter().filter(|v| v.facts.merged).collect();
    let mut regate: Vec<u64> = Vec::new();
    for v in views.iter().filter(|v| !v.facts.merged) {
        if merged.iter().any(|m| contend(v, m)) {
            regate.push(v.facts.number);
        }
    }
    if !regate.is_empty() {
        out.push_str(&format!(
            "- **Re-gate before merging:** {} — a merged PR in this window moved a \
             baseline they were measured against.\n",
            cap_prs(&regate)
        ));
    }

    // Debt a merge left behind: coverage a candidate gate wanted and the code
    // shipped without. Discharging it is a concrete, derivable next action.
    let mut owing: Vec<u64> = merged
        .iter()
        .filter(|v| !v.promotion_debt.is_empty())
        .map(|v| v.facts.number)
        .collect();
    owing.sort_unstable();
    if !owing.is_empty() {
        out.push_str(&format!(
            "- **Discharge promotion debt:** merged {} shipped without a \
             promotion-candidate gate (see the debt table below).\n",
            cap_prs(&owing)
        ));
    }

    out.push_str(
        "\n_Derived only from changed paths and merge state. This section cannot \
         see check or required-context status, review state, true git \
         mergeability, or whether one PR unblocks another — those live on each \
         PR._\n",
    );
    out
}

#[cfg(test)]
#[path = "telemetry_order_tests.rs"]
mod telemetry_order_tests;
