// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

//! Atlas Spark — pure Rust LLM inference server.
//!
//! Startup sequence:
//! 1. Parse CLI args
//! 2. Load model config
//! 3. Initialize GPU backend (AtlasCudaBackend)
//! 4. Load model weights (SafetensorsLoader)
//! 5. Build model via factory
//! 6. Load tokenizer
//! 7. Spawn scheduler thread
//! 8. Start axum HTTP server

mod adaptive_sampler;
mod anthropic;
mod api;
mod auth;
mod citation;
mod cli;
mod conversation_store;
pub mod grammar;
mod halluc_probe;
mod hint_injector;
mod llmlingua;
mod lookback_lens;
mod loop_detector;
mod loop_simhash;
mod lqer;
mod main_modules;
pub mod metrics;
mod model_resolver;
mod moe_quality;
mod ngram;
mod observation_mask;
mod openai;
mod rate_limiter;
pub mod reasoning_parser;
mod refusal;
mod request_dumper;
mod response_store;
mod retrieval_heads;
mod scheduler;
mod scheduling_policy;
mod session_manager;
mod symbol_trie;
mod task_pin;
mod tokenizer;
mod tool_arg_dedup;
pub mod tool_parser;
mod tool_rag;
mod tool_salvage;

use anyhow::Result;
use clap::Parser;

use crate::cli::{Cli, Command};
use crate::main_modules::serve;

pub(crate) use crate::main_modules::AppState;

/// Re-export for convenience in api.rs / anthropic.rs.
pub type ModelBehavior = atlas_kernels::ModelBehavior;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Serve(args) => serve(args).await,
    }
}
