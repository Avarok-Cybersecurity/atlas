// SPDX-License-Identifier: AGPL-3.0-only

//! Print the benchmarks a classified PR intent implies.
//!
//! ```text
//! implied_benches --root . --path performance/decode
//! agentic-webserver
//! bfcl-subset
//! ttft-warm-gate
//! ```
//!
//! # Why this exists: there were TWO taxonomy walks, and they disagreed
//!
//! `ci.yml` walked `.github/pr-taxonomy.json` in jq to render the summary
//! table, duplicating [`atlas_plugin::gate::pr_taxonomy::benches_for`]. Run
//! against the same inputs they diverge on four shapes:
//!
//! ```text
//!   performance//decode        jq {agentic-webserver}   Rust 3 benches
//!   " performance / decode "   jq {}                    Rust 3 benches
//!   performance/_benches       jq exit 5                Rust {agentic-webserver}
//!   _doc                       jq exit 5                Rust {}
//! ```
//!
//! Three of those are jq REMOVING benchmarks the Rust requires — the same
//! removing-direction failure that `pr_taxonomy`'s strict `_benches` parser was
//! hardened against, one file over. jq also happily read a bare-string
//! `_benches` that `load` hard-rejects, so a taxonomy typo made the PR summary
//! advertise benchmarks the gate would refuse to load at all.
//!
//! ★ And the error rendering was worse than "(none)": the step runs under
//! `set -euo pipefail`, so a jq exit-5 inside `IMPLIED=$(…)` aborted the step
//! mid-table with stderr discarded. There was no failure rendering.
//!
//! # The contract, and why the three outcomes are separated
//!
//! | stream | meaning |
//! |---|---|
//! | stdout | one bench id per line, sorted; EMPTY means "implies nothing" |
//! | stderr | a stale-segment warning (exit 0), or the cause (exit != 0) |
//! | exit 0 | an answer was computed, possibly empty |
//! | exit 1 | no answer: unreadable/malformed taxonomy, or bad invocation |
//!
//! "Implies nothing" and "could not tell" must never render alike. That
//! collapse is what let a broken walk read as a clean one.
//!
//! A STALE SEGMENT WARNS RATHER THAN FAILS, deliberately. `benches_for`'s
//! contract is that a renamed category degrades to *fewer extra* benches and
//! never crashes, because it feeds a gate. This binary must not be stricter
//! than the gate it fronts, or the summary and the gate would disagree about
//! the same ledger line — which is the disease being cured.

use anyhow::{Context, Result, bail};
use atlas_plugin::gate::{pr_taxonomy, required};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() -> Result<()> {
    let root = std::path::PathBuf::from(arg("--root").unwrap_or_else(|| ".".into()));

    // PCND. `--root` defaults because "the repo I am standing in" is the only
    // sane reading (and `ledger_append` sets that precedent). `--path` does
    // not: there is no defensible default intent. An EMPTY value is accepted
    // and means "not classified" — distinct from the flag being absent, which
    // is a mistake.
    let path = arg("--path").context(
        "--path is required (pass an empty string for \"not classified\"). \
         There is no default intent: guessing one would attribute benchmarks \
         to a PR nobody classified.",
    )?;

    // NEVER `unwrap_or_default()` here. `load` hard-bails on a malformed
    // `_benches`, and swallowing that into an empty tree would turn a loud
    // parse failure into "this PR implies nothing" — a removal wearing the
    // costume of an answer.
    let roots = pr_taxonomy::load(&root)
        .with_context(|| format!("reading the taxonomy under {}", root.display()))?;

    let segments = required::parse_category(&path);
    let (benches, matched) = pr_taxonomy::benches_for_matched(&roots, &segments);

    if matched < segments.len() {
        // Not fatal — see the module docs. But it must be SAID: without this,
        // `performance/decodes` and `performance` are indistinguishable in the
        // output, and a renamed category silently costs coverage forever.
        eprintln!(
            "warning: segment {:?} is not in the taxonomy; matched {} of {} segments",
            segments[matched],
            matched,
            segments.len()
        );
    }

    for bench in &benches {
        println!("{bench}");
    }
    if benches.is_empty() && segments.is_empty() && !path.trim().is_empty() {
        // A non-empty --path that parsed to nothing is a caller bug, not an
        // intent: "///" is not a classification.
        bail!("--path {path:?} contains no usable segments");
    }
    Ok(())
}
