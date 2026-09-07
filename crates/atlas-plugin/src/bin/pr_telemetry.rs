// SPDX-License-Identifier: AGPL-3.0-only

//! Render the open-PR telemetry comment body.
//!
//! Reads a JSON array of `PrFacts` on stdin, writes markdown to stdout. It does
//! not talk to GitHub: fetching and posting are the workflow's job, so the part
//! that decides anything stays unit-testable and this binary stays trivially
//! reviewable.
//!
//!     gh api ... | pr-telemetry > body.md
//!
//! Lives in `atlas-plugin` because that crate is host-only and CUDA-free, so CI
//! can build it on a runner with no toolchain and no GPU.

use std::io::Read;

use atlas_plugin::gate::telemetry::{PrFacts, render};

fn main() -> anyhow::Result<()> {
    let root = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    // ★ Never `unwrap_or_default()` here. A malformed or truncated feed would
    // become an empty Vec, `render` would emit "_No open pull requests._", and
    // the workflow would post that to the tracking issue as the state of the
    // repository — a parse failure published as a measurement of zero. Failing
    // the step leaves the previous comment standing, which is stale but true.
    let prs: Vec<PrFacts> = serde_json::from_str(input.trim())
        .map_err(|e| anyhow::anyhow!("the PR feed on stdin is not a JSON array of PrFacts: {e}"))?;

    print!("{}", render(&root, &prs));
    Ok(())
}
