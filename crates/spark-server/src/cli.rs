// SPDX-License-Identifier: AGPL-3.0-only

//! CLI argument parsing.

use clap::Parser;

mod bench_args;
pub mod bench_card;
mod bench_gate_check;
mod bench_print;
pub mod bench_record;
mod bench_resolve;
pub mod bench_run;
mod bench_selfstart;
pub(crate) mod doctor;
pub(crate) mod flag_values;
pub(crate) mod manifest;
mod serve_args;
pub(crate) mod sync_recipes;
mod validate;
pub use bench_args::BenchmarkArgs;
pub use serve_args::{DEFAULT_KV_CACHE_DTYPE, DEFAULT_NUM_DRAFTS, ServeArgs};
pub use validate::validate_serve_args;

/// The user-facing release string, e.g. `1.0.0-beta-preview`.
///
/// Read from `Cargo.toml` rather than written out here, so the version a build
/// reports and the version it was packaged as cannot drift. Anything that needs
/// to record which Atlas produced an artifact should use this rather than
/// re-deriving it.
pub const ATLAS_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(
    name = "spark",
    version = ATLAS_VERSION,
    about = "Atlas Spark — pure Rust LLM inference server"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Start the inference server.
    Serve(ServeArgs),
    /// Run and inspect the benchmark suite, without the dashboard.
    Benchmark(BenchmarkArgs),
    /// Print the serve flag surface as JSON.
    ///
    /// Hidden because it is a build tool, not part of the supported CLI: it
    /// exists so downstream tooling can be GENERATED from clap rather than
    /// transcribed from it. `ServeArgs` still has no `Serialize` derive and
    /// this does not promise that any flag keeps its name — a rename shows up
    /// as a diff in whatever consumes the output.
    #[command(hide = true)]
    DumpServeOptions,
    /// Populate the local recipe index from the recipe repository.
    ///
    /// The index is what `benchmark run` resolves a recipe id against, and
    /// until this existed the only thing that wrote it was the TUI Library —
    /// so the error a headless box got was "open the TUI Library once to
    /// populate it", which is advice a CI runner, a container, or a machine
    /// reached over ssh cannot take. That is not a hint, it is a dead end.
    ///
    /// Deliberately explicit rather than an automatic fetch inside
    /// `benchmark run`: a benchmark that silently reaches the network mid-run
    /// is a benchmark whose result depends on something nobody declared.
    SyncRecipes,
    /// Report whether this box can run a benchmark, and say what to fix.
    ///
    /// Every finding here is a condition that has cost hours and used to
    /// present as the same symptom — `recipe "..." is not in the local index
    /// (0 cached)` — whatever the real cause was: an `~/.atlas` owned by another
    /// uid, a `sync-recipes` that was never run, or a signing identity minted
    /// into a scratch ATLAS_HOME whose key nobody committed.
    ///
    /// Exits non-zero when anything is wrong, so a provisioning script can gate
    /// on it.
    Doctor,
}

#[cfg(test)]
mod version_tests {
    use super::*;

    #[test]
    fn version_flag_reports_the_packaged_version() {
        // `--version` short-circuits parsing, so clap reports it as an "error"
        // whose kind is DisplayVersion and whose rendering is the output.
        let err = Cli::try_parse_from(["spark", "--version"]).expect_err("exits early");
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
        assert!(
            err.to_string().contains(ATLAS_VERSION),
            "`--version` printed {:?}, which does not carry {ATLAS_VERSION}",
            err.to_string()
        );
    }

    #[test]
    fn the_reported_version_is_the_cargo_version() {
        // The point of reading it from Cargo.toml: a release bump moves both or
        // neither. A literal here could silently disagree with the package.
        assert_eq!(ATLAS_VERSION, env!("CARGO_PKG_VERSION"));
        assert!(!ATLAS_VERSION.is_empty());
    }
}
