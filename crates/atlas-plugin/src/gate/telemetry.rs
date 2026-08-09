// SPDX-License-Identifier: AGPL-3.0-only

//! The open-PR telemetry view: one comment, rewritten in place.
//!
//! # What it is for
//!
//! Each PR's own checks answer "is this one green?". Nothing answers "are these
//! seven green *together*" — two PRs touching one kernel target are each
//! measured against a baseline neither will hold once the other lands. This
//! renders the cross-PR view: which targets each PR re-opens, where they
//! collide, and an order that lands them without a collision.
//!
//! # Rendering is separate from fetching on purpose
//!
//! [`render`] is a pure function of [`PrFacts`] plus the tree. Everything that
//! talks to GitHub lives in the workflow, so the part with the judgement in it —
//! which targets, which order, who to mention — is unit-testable without a
//! network, a token, or a fixture repository.
//!
//! # It advises; it does not block
//!
//! Nothing here fails a check. A collision is a note for whoever merges, and the
//! CODEOWNERS mentions are a courtesy. The blocking decisions stay in
//! `check.rs`, where they are made against committed records rather than
//! against a model's or a heuristic's opinion.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::{codeowners, taxon};

/// The marker pair that makes the comment rewritable in place.
///
/// Without it the bot would append, and a week of appends is a comment nobody
/// reads. The workflow finds its own previous comment by this marker rather
/// than by tracking an id it would have to store somewhere.
pub const MARKER_START: &str = "<!-- atlas-pr-telemetry:start -->";
pub const MARKER_END: &str = "<!-- atlas-pr-telemetry:end -->";

/// What the workflow collects about one open PR.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PrFacts {
    pub number: u64,
    pub title: String,
    pub author: String,
    #[serde(default)]
    pub draft: bool,
    /// Repo-relative paths this PR changes.
    #[serde(default)]
    pub changed_paths: Vec<String>,
}

/// One PR's derived position in the taxonomy.
#[derive(Debug, Clone)]
pub struct PrView {
    pub facts: PrFacts,
    pub hardware: BTreeSet<String>,
    pub models: BTreeSet<(String, String)>,
    pub targets: BTreeSet<taxon::Target>,
    pub owners: Vec<String>,
    /// True when the diff reaches beyond `kernels/` and therefore re-opens
    /// every gate regardless of which targets it touches.
    pub whole_repo: bool,
}

/// Derive every PR's view. Pure: the tree supplies the taxonomy, nothing else.
pub fn views(root: &Path, prs: &[PrFacts]) -> Vec<PrView> {
    let rules = codeowners::load(root);
    prs.iter()
        .map(|facts| {
            let kernel_paths: Vec<String> = facts
                .changed_paths
                .iter()
                .filter(|p| taxon::hardware_of(p).is_some())
                .cloned()
                .collect();
            PrView {
                hardware: taxon::hardware_span(&kernel_paths),
                models: taxon::model_span(&kernel_paths),
                targets: taxon::affected(root, &kernel_paths),
                owners: codeowners::owners_for_paths(&rules, &facts.changed_paths),
                whole_repo: facts.changed_paths.len() > kernel_paths.len(),
                facts: facts.clone(),
            }
        })
        .collect()
}

/// Targets more than one open PR re-opens.
///
/// This is the whole reason the view exists: each of those PRs is measured
/// against a baseline the other will move, so whichever lands second is gated
/// on a number that no longer describes the tree.
pub fn collisions(views: &[PrView]) -> BTreeMap<String, Vec<u64>> {
    let mut by_target: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for view in views {
        for target in &view.targets {
            by_target
                .entry(target.to_string())
                .or_default()
                .push(view.facts.number);
        }
    }
    by_target.retain(|_, prs| prs.len() > 1);
    by_target
}

/// A merge order that avoids re-measuring the same target twice in a row.
///
/// Deliberately simple, and simple is the point: fewest targets first, ties
/// broken by PR number. It is a SUGGESTION printed for a human, not a scheduler
/// — a clever order that nobody can predict is worse than an obvious one, since
/// the reader has to be able to tell at a glance when it is wrong.
pub fn merge_order(views: &[PrView]) -> Vec<u64> {
    let mut ranked: Vec<(usize, bool, u64)> = views
        .iter()
        .map(|v| (v.targets.len(), v.whole_repo, v.facts.number))
        .collect();
    ranked.sort();
    ranked.into_iter().map(|(_, _, n)| n).collect()
}

fn escape(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

/// The comment body, between its markers.
pub fn render(root: &Path, prs: &[PrFacts]) -> String {
    let views = views(root, prs);
    let all_targets = taxon::walk(root);
    let mut out = String::new();

    out.push_str(MARKER_START);
    out.push_str("\n## Open-PR telemetry\n\n");
    out.push_str(
        "Advisory. Nothing here blocks a merge — the blocking checks live on each PR.\n\n",
    );

    if views.is_empty() {
        out.push_str("_No open pull requests._\n");
        out.push_str(MARKER_END);
        out.push('\n');
        return out;
    }

    // ── PRs, grouped by the hardware they touch ──
    out.push_str("### Pull requests\n\n");
    out.push_str("| PR | category | targets re-opened | codeowners |\n");
    out.push_str("|---|---|---|---|\n");
    let mut grouped: BTreeMap<String, Vec<&PrView>> = BTreeMap::new();
    for view in &views {
        let key = if view.hardware.is_empty() {
            "host / non-kernel".to_string()
        } else {
            view.hardware
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(" + ")
        };
        grouped.entry(key).or_default().push(view);
    }
    for (category, group) in &grouped {
        for view in group {
            let targets = if view.whole_repo {
                "ALL (diff reaches outside kernels/)".to_string()
            } else if view.targets.is_empty() {
                "none".to_string()
            } else {
                format!("{}", view.targets.len())
            };
            let owners = if view.owners.is_empty() {
                "—".to_string()
            } else {
                view.owners.join(" ")
            };
            out.push_str(&format!(
                "| #{} {}{} | {} | {} | {} |\n",
                view.facts.number,
                if view.facts.draft { "(draft) " } else { "" },
                escape(&view.facts.title),
                escape(category),
                targets,
                escape(&owners),
            ));
        }
    }

    // ── Collisions ──
    let collisions = collisions(&views);
    out.push_str("\n### Collisions\n\n");
    if collisions.is_empty() {
        out.push_str("None: no target is re-opened by more than one open PR.\n");
    } else {
        out.push_str(
            "Each PR below is measured against a baseline another open PR will \
             move. Whichever lands second needs re-gating.\n\n\
             | target | PRs |\n|---|---|\n",
        );
        for (target, prs) in &collisions {
            let list = prs
                .iter()
                .map(|n| format!("#{n}"))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("| `{target}` | {list} |\n"));
        }
    }

    // ── Suggested order ──
    out.push_str("\n### Suggested merge order\n\n```mermaid\ngraph LR\n");
    let order = merge_order(&views);
    for (i, number) in order.iter().enumerate() {
        let view = views.iter().find(|v| v.facts.number == *number);
        let count = view.map(|v| v.targets.len()).unwrap_or(0);
        out.push_str(&format!(
            "  pr{number}[\"#{number}<br/>{count} target(s)\"]\n"
        ));
        if i > 0 {
            out.push_str(&format!("  pr{} --> pr{number}\n", order[i - 1]));
        }
    }
    out.push_str("```\n\nFewest targets first — the cheapest to re-gate if the order changes.\n");

    // ── Every target, always ──
    out.push_str(&format!(
        "\n### Targets ({} total)\n\nEvery target is listed, including the ones no open PR \
         touches. Showing only the affected ones would silently turn *ungated* into \
         *unaffected*.\n\n| target | re-opened by |\n|---|---|\n",
        all_targets.len()
    ));
    for target in &all_targets {
        let key = target.to_string();
        let touching: Vec<String> = views
            .iter()
            .filter(|v| v.targets.contains(target))
            .map(|v| format!("#{}", v.facts.number))
            .collect();
        out.push_str(&format!(
            "| `{key}` | {} |\n",
            if touching.is_empty() {
                "—".to_string()
            } else {
                touching.join(", ")
            }
        ));
    }

    out.push_str(MARKER_END);
    out.push('\n');
    out
}

#[cfg(test)]
#[path = "telemetry_tests.rs"]
mod telemetry_tests;
