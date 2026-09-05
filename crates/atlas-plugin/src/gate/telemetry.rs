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

use super::{codeowners, coverage, taxon};

/// The bounded renderings: recommended order, the gitGraph, next steps.
#[path = "telemetry_order.rs"]
pub mod order;
pub use order::{CHART_PR_BOUND, merge_order};

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
    /// True once this PR has landed on the default branch.
    ///
    /// ★ Promotion debt outlives the merge. A PR that a promotion-candidate
    /// gate wanted to see, merged without it, is not a warning any more — it is
    /// coverage this repository has already gone without, and it stays on the
    /// books until a record discharges it. CLOSED PRs carry no debt: nothing
    /// shipped, so nothing is owed.
    #[serde(default)]
    pub merged: bool,
    /// Repo-relative paths this PR changes.
    ///
    /// Only meaningful when [`Self::paths_unknown`] is false. When the list
    /// could not be read it is empty, and empty-because-unknown is NOT the
    /// same fact as empty-because-nothing-changed.
    #[serde(default)]
    pub changed_paths: Vec<String>,
    /// True when the collector could not read this PR's file list at all.
    ///
    /// ★ The whole point of this view is blast radius, and understating it is
    /// the one direction it must not fail in. A failed `pulls/N/files` call
    /// used to fall through to `changed_paths: []`, which every consumer here
    /// reads as *this PR touches nothing*: no targets, no owners, no promotion
    /// debt, no collisions — a maximally reassuring record produced by an API
    /// error. This flag makes the absence of the input a first-class fact so
    /// the renderer can say "unknown" instead of "none".
    ///
    /// The marker is kept rather than dropping the record: a dropped PR is
    /// simply absent from the comment, and absent reads as *no such PR* — the
    /// same understatement by a different route, and one that also removes it
    /// from the collision table and the merge order it should be blocking.
    #[serde(default)]
    pub paths_unknown: bool,
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
    /// [`coverage::PROMOTION_CANDIDATES`] this PR's paths would have
    /// invalidated — gates that WANTED to run and were not required to.
    ///
    /// ★ This is debt, and it is rendered whether or not it is empty. Showing
    /// only the gates that ran silently converts "ungated" into "unaffected",
    /// which is fail-open by omission and exactly how a coverage gap becomes
    /// invisible. It needs no model: it is a join between changed paths and a
    /// coverage table.
    pub promotion_debt: Vec<&'static str>,
}

/// Derive every PR's view. Pure: the tree supplies the taxonomy, nothing else.
pub fn views(root: &Path, prs: &[PrFacts]) -> Vec<PrView> {
    let rules = codeowners::load(root);
    let all_targets: BTreeSet<taxon::Target> = taxon::walk(root).into_iter().collect();
    prs.iter()
        .map(|facts| {
            let kernel_paths: Vec<String> = facts
                .changed_paths
                .iter()
                .filter(|p| taxon::hardware_of(p).is_some())
                .cloned()
                .collect();
            // ★ Unknown paths are the MAXIMUM blast radius, never the empty
            // one. `whole_repo` is what makes a PR re-open every target and
            // contend with every other PR, so routing the unknown case through
            // it gives the conservative answer everywhere at once.
            let whole_repo = facts.paths_unknown || facts.changed_paths.len() > kernel_paths.len();
            PrView {
                hardware: taxon::hardware_span(&kernel_paths),
                models: taxon::model_span(&kernel_paths),
                targets: if whole_repo {
                    all_targets.clone()
                } else {
                    taxon::affected(root, &kernel_paths)
                },
                owners: codeowners::owners_for_paths(&rules, &facts.changed_paths),
                whole_repo,
                // Same rule for debt: with no paths there is no join to
                // perform, so the answer is every candidate, and the renderer
                // labels the row as assumed rather than measured.
                promotion_debt: if facts.paths_unknown {
                    coverage::PROMOTION_CANDIDATES
                        .iter()
                        .map(|gate| gate.id)
                        .collect()
                } else {
                    coverage::promotion_debt(facts.changed_paths.iter().map(String::as_str))
                },
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
    for view in views.iter().filter(|view| !view.facts.merged) {
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

    if views.is_empty() {
        out.push_str("_No open pull requests._\n");
        out.push_str(MARKER_END);
        out.push('\n');
        return out;
    }

    // ── The chart, first ──
    //
    // The one picture the comment exists for: `main` as a line, the
    // recommended merge order branching off it. Everything below is the
    // evidence; the chart is the conclusion, so it leads.
    out.push_str(&order::render_order_chart(&views));
    out.push_str(
        "\nAdvisory. Nothing here blocks a merge — the blocking checks live on each PR.\n",
    );

    // ── The one thing this view cannot be quiet about ──
    //
    // A PR whose file list the collector could not read is shown at MAXIMUM
    // blast radius below, not zero. Saying so out loud is the difference
    // between a conservative estimate and a fabricated measurement: every
    // "ALL" and every debt row those PRs contribute is an assumption, and a
    // reader who cannot tell which rows are assumed will quote them as facts.
    let blind: Vec<u64> = views
        .iter()
        .filter(|v| v.facts.paths_unknown)
        .map(|v| v.facts.number)
        .collect();
    if !blind.is_empty() {
        out.push_str(&format!(
            "\n> ⚠ **Changed files unavailable for {}.** The API call for their diffs \
             failed on this run, so nothing below measures them. They are counted at \
             MAXIMUM blast radius — every target, every promotion candidate — because \
             an unknown diff must never render as an empty one. Re-run this workflow \
             before using any row they appear in.\n",
            order::cap_prs(&blind)
        ));
    }

    out.push_str(&order::render_next_steps(&views));

    // ── PRs, grouped by the hardware they touch ──
    //
    // Open PRs only. A merged PR's remaining relevance is the debt it left,
    // which has its own ledger below — repeating it here buried the open work
    // under history and was most of why the comment grew unreadable.
    let merged_count = views.iter().filter(|v| v.facts.merged).count();
    out.push_str("\n### Pull requests\n\n");
    if merged_count > 0 {
        out.push_str(&format!(
            "_{merged_count} merged PR(s) tracked for debt only — see the ledger below._\n\n"
        ));
    }
    out.push_str("| PR | category | targets re-opened | codeowners |\n");
    out.push_str("|---|---|---|---|\n");
    let mut grouped: BTreeMap<String, Vec<&PrView>> = BTreeMap::new();
    for view in views.iter().filter(|v| !v.facts.merged) {
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
            let targets = if view.facts.paths_unknown {
                "ALL — **assumed**, changed files unreadable".to_string()
            } else if view.whole_repo {
                "ALL (diff reaches outside kernels/)".to_string()
            } else if view.targets.is_empty() {
                "none".to_string()
            } else {
                format!("{}", view.targets.len())
            };
            // "—" means CODEOWNERS matched nobody. With no paths there was
            // nothing to match, which is a different sentence.
            let owners = if view.facts.paths_unknown {
                "unknown".to_string()
            } else if view.owners.is_empty() {
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

    // ── Promotion debt ──
    //
    // ★ Rendered ALWAYS, empty or not. A gate that is not required yet still
    // has an opinion about which PRs it wanted to see; listing only the gates
    // that ran turns "ungated" into "unaffected" by omission, which is how a
    // coverage gap becomes invisible. Deterministic — a join between changed
    // paths and `coverage::PROMOTION_CANDIDATES`, no model involved.
    out.push_str("\n### Promotion-candidate debt\n\n");
    if coverage::PROMOTION_CANDIDATES.is_empty() {
        out.push_str(
            "No gates are on a promotion path, so nothing can be owed. When one \
             is registered (`coverage::PROMOTION_CANDIDATES`), every PR whose \
             paths it covers appears here until a record discharges it.\n",
        );
    } else {
        let owing: Vec<&PrView> = views
            .iter()
            .filter(|v| !v.promotion_debt.is_empty())
            .collect();
        out.push_str(
            "These gates are NOT required, so these PRs can merge without them. \
             Each row is coverage this repository chose not to buy — recorded so \
             the choice stays visible rather than becoming an assumption.\n\n",
        );
        out.push_str("| PR | merged? | title | gates that wanted to run |\n|---|---|---|---|\n");
        if owing.is_empty() {
            out.push_str("| — | — | _no tracked PR touches a promotion candidate's paths_ | — |\n");
        }
        for v in owing {
            out.push_str(&format!(
                "| #{} | {} | {} | {} |\n",
                v.facts.number,
                // Merged debt is ACCRUED — the coverage was skipped and the code
                // shipped. Open debt is still a warning. The column is what makes
                // those two different rows instead of one undifferentiated list.
                if v.facts.merged { "**yes**" } else { "not yet" },
                escape(&v.facts.title),
                if v.facts.paths_unknown {
                    format!(
                        "{} — **assumed** (paths unreadable)",
                        v.promotion_debt.join(", ")
                    )
                } else {
                    v.promotion_debt.join(", ")
                }
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
            // Cells share the chart's bound: a wall of two hundred refs is as
            // unreadable as a two-hundred-node chart, and `cap_prs` says how
            // many were dropped.
            out.push_str(&format!("| `{target}` | {} |\n", order::cap_prs(prs)));
        }
    }

    // ── Every target, always ──
    out.push_str(&format!(
        "\n### Targets ({} total)\n\nEvery target is listed, including the ones no open PR \
         touches. Showing only the affected ones would silently turn *ungated* into \
         *unaffected*.\n\n| target | re-opened by |\n|---|---|\n",
        all_targets.len()
    ));
    for target in &all_targets {
        let key = target.to_string();
        let touching: Vec<u64> = views
            .iter()
            .filter(|v| v.targets.contains(target))
            .map(|v| v.facts.number)
            .collect();
        out.push_str(&format!(
            "| `{key}` | {} |\n",
            if touching.is_empty() {
                "—".to_string()
            } else {
                order::cap_prs(&touching)
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

#[cfg(test)]
#[path = "telemetry_unknown_tests.rs"]
mod telemetry_unknown_tests;
