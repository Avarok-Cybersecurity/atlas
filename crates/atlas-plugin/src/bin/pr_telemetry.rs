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
    let prs: Vec<PrFacts> = serde_json::from_str(input.trim()).unwrap_or_default();

    print!("{}", render(&root, &prs));
    Ok(())
}
