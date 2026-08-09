// SPDX-License-Identifier: AGPL-3.0-only

//! Append one event to a PR's journey ledger.
//!
//! `crates/atlas-governance` shipped with a full event model, a grow-only
//! `Journey`, deduplication, and 16 tests — and **zero writers**. Nothing had
//! ever appended a line. This is the first one.
//!
//! ```text
//! ledger_append category --root . --pr 433 --head-sha abc123 \
//!               --run-id 42 --attempt 1 --at 1786280000 \
//!               --value performance/decode --status ok
//! ```
//!
//! Writes `governance/pr-<n>.jsonl` under the repo root. The workflow commits
//! it; this binary does not touch git, for the same reason `pr_telemetry` does
//! not talk to GitHub — the part that decides something stays testable, and
//! the part that has credentials stays trivially reviewable.
//!
//! ★ WHY A CATEGORY EVENT IS WORTH STORING AT ALL. `EventKind::Category`'s own
//! doc says it is "recorded, never acted upon … so the observe-only rollout
//! has something to audit". Today that matters concretely: three live runs on
//! one PR produced `tooling`, `performance`, `tooling` from the flat
//! classifier while the descending one held `infrastructure/*` throughout.
//! Neither observation would exist if the disagreements had not been written
//! down.

use anyhow::{Context, Result, bail};
use atlas_governance::event::{Event, EventKind};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn require(name: &str) -> Result<String> {
    // PCND: no defaults. A ledger line that guessed its own PR number or sha
    // would be worse than no line — it would be evidence of the wrong thing.
    arg(name).with_context(|| format!("{name} is required"))
}

fn main() -> Result<()> {
    let root = std::path::PathBuf::from(arg("--root").unwrap_or_else(|| ".".into()));

    let pr: u64 = require("--pr")?.parse().context("--pr must be a number")?;
    let head_sha = require("--head-sha")?;
    let run_id = require("--run-id")?;
    let attempt: u32 = arg("--attempt")
        .unwrap_or_else(|| "1".into())
        .parse()
        .context("--attempt must be a number")?;

    // `at` is data, not identity — `Event::identity()` deliberately excludes it
    // so a re-run collapses instead of accumulating. Supplied by the caller
    // rather than read from the clock so a replay is reproducible.
    let at: u64 = require("--at")?
        .parse()
        .context("--at must be a unix time")?;

    let kind = match std::env::args().nth(1).as_deref() {
        Some("category") => EventKind::Category {
            value: require("--value")?,
            status: require("--status")?,
        },
        other => bail!(
            "unknown event kind {other:?}; this binary currently writes `category` only. \
             Gate and Measurement events are produced where those things happen, not here."
        ),
    };

    let event = Event {
        pr,
        head_sha,
        run_id,
        attempt,
        at,
        kind,
    };
    let path = atlas_governance::ledger::path_for(&root, pr);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    atlas_governance::ledger::append(&path, &event)?;
    println!("appended to {}", path.display());
    Ok(())
}
