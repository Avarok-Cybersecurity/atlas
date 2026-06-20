// SPDX-License-Identifier: AGPL-3.0-only

//! CLI argument parsing.

use clap::Parser;

mod serve_args;
pub use serve_args::ServeArgs;

#[derive(Parser, Debug)]
#[command(name = "spark", about = "Atlas Spark — pure Rust LLM inference server")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Start the inference server.
    Serve(ServeArgs),
}
