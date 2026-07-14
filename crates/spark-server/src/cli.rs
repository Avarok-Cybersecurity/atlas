// SPDX-License-Identifier: AGPL-3.0-only

//! CLI argument parsing.

use clap::Parser;

mod serve_args;
mod validate;
pub use serve_args::ServeArgs;
pub use validate::validate_serve_args;

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
