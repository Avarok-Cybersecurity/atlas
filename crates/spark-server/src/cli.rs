// SPDX-License-Identifier: AGPL-3.0-only

//! CLI argument parsing.

use clap::Parser;

mod bench_args;
mod bench_print;
pub mod bench_run;
mod bench_selfstart;
mod serve_args;
mod validate;
pub use bench_args::BenchmarkArgs;
pub use serve_args::ServeArgs;
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
