// SPDX-License-Identifier: AGPL-3.0-only

//! Validate a harvested ledger artifact and merge it into the journey ledger.
//!
//! ```text
//! ledger_harvest --root . --pr 433 --from downloaded/pr-433.jsonl
//! ```
//!
//! # Why a harvester exists at all
//!
//! The classify job appends a `Category` line and cannot persist it: it holds
//! `permissions: contents: read`, deliberately, because it consumes model
//! output derived from an attacker-authored PR title. Its own header promises
//! "a compromised model response has nowhere to go even in principle" — and
//! that promise *is* the missing write scope. Granting it `contents: write`
//! would delete the promise to fix the symptom.
//!
//! So the line leaves as a workflow artifact (no write scope needed, works from
//! forks) and a scheduled job running **default-branch code** harvests it. This
//! binary is the validating half of that job.
//!
//! # The trust boundary, stated exactly
//!
//! Artifact content is UNTRUSTED. It was produced by a job running the PR
//! author's workflow file, on the PR author's branch.
//!
//! ★ **`--pr` is supplied by the caller from the run's own API record, and any
//! event disagreeing with it is REJECTED.** This is the single load-bearing
//! check. Without it, PR #1 could upload an artifact claiming to be PR #2 and
//! write into another PR's journey — the artifact naming itself is not
//! evidence of anything.
//!
//! Everything else follows from "escalation is monotone-stricter": a forged
//! `Category` line for one's OWN pr can only make that PR owe MORE gates, never
//! fewer, because `benches_for` unions along a path and the required set is
//! `path_derived ∪ intent_derived`. The worst outcome is a self-inflicted GPU
//! bill.
//!
//! Also enforced, because "monotone-stricter" is only true of well-formed
//! input:
//!
//! * every line must parse as an [`Event`] — malformed input is rejected
//!   wholesale rather than skipped, so a truncated upload cannot silently
//!   contribute a prefix;
//! * only [`EventKind::Category`] is accepted here. `Gate` and `Measurement`
//!   are produced where those things happen (on a GPU box, committed beside the
//!   `.benchmarks/` record), and accepting them from an artifact would let a PR
//!   author assert its own gate verdicts;
//! * the category value must resolve in the taxonomy, or it is dropped with a
//!   warning. A path naming a segment nobody has ever defined is not an
//!   opinion, it is noise — and `pr_taxonomy::validate` already pins that every
//!   `_benches` id is a registered benchmark, so a resolvable path can only
//!   name real gates.
//!
//! Dedup is by [`Event::identity`], which excludes `at` — so re-harvesting the
//! same artifact converges instead of accumulating.

use anyhow::{Context, Result, bail};
use atlas_governance::event::{Event, EventKind};
use atlas_plugin::gate::{pr_taxonomy, required};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn require(name: &str) -> Result<String> {
    arg(name).with_context(|| format!("{name} is required"))
}

fn main() -> Result<()> {
    let root = std::path::PathBuf::from(arg("--root").unwrap_or_else(|| ".".into()));

    // PCND, and this one is the security boundary rather than a style rule.
    // There is no defensible default PR number: defaulting to the artifact's
    // own claim is precisely the attack.
    let pr: u64 = require("--pr")?
        .parse()
        .context("--pr must be a number, taken from the RUN's API record")?;
    let from = std::path::PathBuf::from(require("--from")?);

    let text =
        std::fs::read_to_string(&from).with_context(|| format!("reading {}", from.display()))?;

    // The taxonomy is loaded once, and a load failure is FATAL rather than
    // "accept everything": validating against an empty tree would drop every
    // category silently, which is a removal wearing the costume of a pass.
    let roots = pr_taxonomy::load(&root).context("loading the taxonomy to validate categories")?;

    let dest = atlas_governance::ledger::path_for(&root, pr);
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // Existing identities, so a re-harvest converges rather than accumulating.
    let seen: std::collections::BTreeSet<String> = if dest.exists() {
        atlas_governance::ledger::read_all(&dest)
            .with_context(|| format!("reading the existing ledger at {}", dest.display()))?
            .events
            .iter()
            .map(atlas_governance::event::Event::identity)
            .collect()
    } else {
        Default::default()
    };

    let (mut appended, mut skipped, mut dropped) = (0usize, 0usize, 0usize);
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let event: Event = serde_json::from_str(line)
            .with_context(|| format!("{}:{} is not a well-formed Event", from.display(), i + 1))?;

        // ★ The load-bearing check. The artifact does not get to say which PR
        // it belongs to.
        if event.pr != pr {
            bail!(
                "{}:{} claims pr={} but this artifact belongs to pr={pr} (per the run record). \
                 Refusing: an artifact naming its own PR is not evidence.",
                from.display(),
                i + 1,
                event.pr
            );
        }

        let label = event.node_label();
        let EventKind::Category { value, status } = &event.kind else {
            bail!(
                "{}:{} is a `{label}` event. Only `category` is harvested — Gate and Measurement \
                 are written where those things happen (beside the .benchmarks/ record, on the \
                 box that measured them), and accepting them from an artifact would let a PR \
                 assert its own gate verdicts.",
                from.display(),
                i + 1,
            );
        };

        // A category nobody can resolve is noise, not an opinion. Dropped with
        // a warning rather than fatal: a taxonomy rename must not wedge the
        // harvester on old artifacts.
        let segments = required::parse_category(value);
        let (_, matched) = pr_taxonomy::benches_for_matched(&roots, &segments);
        if !segments.is_empty() && matched == 0 {
            eprintln!(
                "warning: dropping {value:?} (status {status:?}) — no segment resolves in the \
                 taxonomy"
            );
            dropped += 1;
            continue;
        }

        if seen.contains(&event.identity()) {
            skipped += 1;
            continue;
        }
        atlas_governance::ledger::append(&dest, &event)?;
        appended += 1;
    }

    println!(
        "{}: appended {appended}, already present {skipped}, dropped {dropped}",
        dest.display()
    );
    Ok(())
}
