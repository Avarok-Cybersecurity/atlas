// SPDX-License-Identifier: AGPL-3.0-only

//! Decompose the PR intent ledger into its principal axes, and render the
//! result as the trajectory Discussion comment.
//!
//! Two modes, because a language model runs between them and the model call is
//! the workflow's job, not this binary's:
//!
//!     intent-eigenvector analyze < docs.json     > spectral.json
//!     # workflow asks a model to NAME the axes in spectral.json
//!     intent-eigenvector render  < merged.json   > body.md
//!
//! Neither mode talks to the network or the filesystem. Everything that decides
//! anything stays unit-testable, and the naming step is optional by
//! construction: `render` accepts input with no `naming` field at all and emits
//! the same report with its axes shown by their poles.
//!
//! Lives in `atlas-plugin` beside `pr_telemetry` because that crate is
//! host-only and CUDA-free, so CI builds it on a runner with no toolchain and
//! no GPU.

use std::io::Read;

use anyhow::{Context, bail};
use atlas_governance::eigenvector::{AnalyzeInput, RenderInput, analyze, render::render};

/// Three axes: the dominant direction plus the two largest tensions around it.
/// A fourth has never carried enough variance to be worth a reader's attention.
const AXES: usize = 3;

fn main() -> anyhow::Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_default();

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let input = input.trim();

    match mode.as_str() {
        "analyze" => {
            let parsed: AnalyzeInput =
                serde_json::from_str(input).context("stdin is not a valid AnalyzeInput")?;
            let analysis = analyze(&parsed, AXES);
            println!("{}", serde_json::to_string_pretty(&analysis)?);
        }
        "render" => {
            let parsed: RenderInput =
                serde_json::from_str(input).context("stdin is not a valid RenderInput")?;
            print!("{}", render(&parsed));
        }
        // PCND: no default mode. Defaulting to one of two behaviours that write
        // different things to stdout is how a workflow silently publishes JSON
        // into a comment.
        other => bail!("expected `analyze` or `render`, got `{other}`"),
    }
    Ok(())
}
