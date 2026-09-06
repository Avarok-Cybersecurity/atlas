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
mod citation_structured;
mod cli;
mod conversation_store;
mod disk_guard;
mod error_hints;
pub mod grammar;
mod halluc_probe;
mod hint_injector;
mod ids;
mod ir;
mod llmlingua;
mod lookback_lens;
mod loop_detector;
mod loop_simhash;
mod lqer;
mod main_modules;
pub mod metrics;
mod model_download;
mod model_resolver;
mod moe_quality;
mod ngram;
mod openai;
mod rate_limiter;
pub mod reasoning_parser;
pub mod recipe;
mod refusal;
mod request_dumper;
mod response_store;
mod retrieval_heads;
mod scheduler;
mod scheduling_policy;
mod session_manager;
mod symbol_trie;
mod tokenizer;
mod tool_arg_dedup;
pub mod tool_parser;
mod tool_rag;
mod tscg;
pub mod tui;

use anyhow::Result;

use crate::cli::Command;
use crate::main_modules::serve;

pub(crate) use crate::main_modules::AppState;

/// Re-export for convenience in api.rs / anthropic.rs.
pub type ModelBehavior = atlas_kernels::ModelBehavior;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse BEFORE subscriber install so the TUI gate can see `--no-tui`.
    // clap emits no tracing events, so plain-mode output is unchanged.
    //
    // A positional MODEL that names a serve preset (`spark serve
    // qwen3.8-flash-next-exl3`) is expanded HERE — flag defaults appended and
    // re-parsed, `ATLAS_*` defaults published — so the host, the dashboard and
    // the validator all see the configuration that will actually run. For
    // every other invocation this is `Cli::parse()`.
    let (cli, preset) = main_modules::serve_presets::parse_cli()?;

    // Answered before anything else initialises. This prints a document and
    // exits: no subscriber, no TUI, no GPU. A dashboard would take the
    // terminal and garble the only output the caller wants.
    if matches!(cli.command, Command::DumpServeOptions) {
        println!(
            "{}",
            serde_json::to_string_pretty(&cli::manifest::build())
                .expect("the manifest is plain data and always serialises")
        );
        return Ok(());
    }

    // Same treatment as `DumpServeOptions`: it talks to the network and prints
    // a line. No subscriber, no TUI, no GPU — and no reason to initialise a
    // dashboard for a command whose whole output is two lines of text.
    if matches!(cli.command, Command::SyncRecipes) {
        return cli::sync_recipes::run();
    }

    let no_tui = match &cli.command {
        // `--check-kernels` is a script's entry point too: it prints a report
        // and a JSON line on stdout and exits, so a dashboard would take the
        // terminal, garble both, and have nothing to show afterwards.
        Command::Serve(args) => args.no_tui || args.rank > 0 || args.check_kernels,
        // The benchmark subcommand is a script's entry point: always plain, so
        // nothing here reaches `tui::start` or takes the terminal.
        Command::Benchmark(_) => true,
        // Handled above; it never reaches here.
        Command::DumpServeOptions | Command::SyncRecipes => true,
    };

    let tui_channels = if tui::plain_mode(no_tui) {
        // The pre-TUI init, byte-for-byte: this exact fmt layout is the
        // contract every benchmark driver and gate script greps.
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .init();
        None
    } else {
        let (progress_tx, progress_rx) = std::sync::mpsc::channel();
        tui::init::install_tty_subscriber(progress_tx);
        Some(progress_rx)
    };

    // The preset's decisions were made before a subscriber existed; record
    // them now, first thing in the log, where a reproduction starts reading.
    if let Some(applied) = &preset {
        main_modules::serve_presets::log_applied(applied);
    }

    // Race the server against shutdown. No spawn: `serve()` is a real future that
    // yields while its blocking startup runs on the blocking pool, so pinning it
    // here is enough for `select!` to poll the other branch. (It would NOT be
    // enough if startup still blocked inside the future — `select!` chooses at
    // await points, it does not preempt.)
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<&'static str>();
    tui::shutdown::arm_startup_escape(shutdown_tx);
    let result = match cli.command {
        // Returned above, before anything initialised. Kept as an explicit arm
        // rather than a wildcard so a future subcommand cannot land here by
        // accident and silently do nothing.
        Command::DumpServeOptions | Command::SyncRecipes => {
            unreachable!("handled before initialisation")
        }
        Command::Benchmark(args) => {
            // No model load, so none of the startup-escape plumbing below
            // applies — `dispatch` installs its own Ctrl-C handling. Drop the
            // receiver explicitly: `let _ =` on a future silently discards it
            // without polling, which is a different thing and one clippy is
            // right to flag.
            drop(shutdown_rx);
            cli::bench_run::dispatch(args).await
        }
        Command::Serve(args) => {
            let serving = serve(args, tui_channels);
            tokio::pin!(serving);
            // Only a SEND means shutdown. The sender is parked for the life of the
            // process rather than dropped when startup ends, so the channel should
            // never close; this arm exists so that if one ever did, a closed
            // channel could not masquerade as a shutdown and kill a healthy server.
            let shutdown_signal = async {
                match shutdown_rx.await {
                    Ok(reason) => reason,
                    Err(_) => std::future::pending::<&'static str>().await,
                }
            };
            tokio::pin!(shutdown_signal);
            tokio::select! {
                res = &mut serving => res,
                reason = &mut shutdown_signal => {
                    // Cancelled before the server came up. Nothing is in flight
                    // and no client is connected, so there is nothing to drain —
                    // the startup task is abandoned where it stands.
                    tracing::info!(
                        "Shutdown requested ({reason}) during startup — exiting before the server came up"
                    );
                    // Cleanup that would otherwise run below, then exit without
                    // waiting on the runtime: a task parked inside a synchronous
                    // CUDA call cannot be aborted, and dropping the runtime would
                    // block on it — reintroducing the very wait this fixes.
                    tui::stop_and_join(std::time::Duration::from_secs(2));
                    tui::terminal_guard::restore();
                    tui::init::flush_tee();
                    // A fault can latch during startup (weight upload, warmup),
                    // so this exit needs the same status mapping as the one
                    // below — otherwise the escape hatch silently reports a
                    // poisoned context as a clean stop.
                    std::process::exit(atlas_core::fault::exit_code(
                        true,
                        atlas_core::fault::global().fault(),
                    ));
                }
            }
        }
    };
    // If serve() returned while the TUI owned the screen (startup error, clean
    // shutdown), stop the dashboard thread and wait for its TerminalGuard to
    // drop BEFORE the error prints — main's exit never runs another thread's
    // Drop, and a bare restore() races the thread's raw-mode entry when
    // serve() fails within milliseconds. restore() stays as the backstop.
    tui::stop_and_join(std::time::Duration::from_secs(2));
    tui::terminal_guard::restore();
    tui::init::flush_tee();

    // A GPU fault drains and returns `Ok` by the same path as `SIGTERM`
    // (issue #429), so without this the two are indistinguishable to a
    // supervisor and `restart: on-failure` leaves the endpoint down. Returning
    // `result` unchanged when healthy keeps every other exit byte-identical.
    match atlas_core::fault::global().fault() {
        Some(reason) => {
            if let Err(e) = &result {
                tracing::error!("{e:#}");
            }
            tracing::error!(
                "Exiting after a fatal GPU fault ({reason}). The CUDA context is \
                 destroyed and cannot be recovered in-process; restart the server."
            );
            std::process::exit(atlas_core::fault::exit_code(result.is_ok(), Some(reason)));
        }
        None => result,
    }
}

#[cfg(test)]
#[path = "main_exit_tests.rs"]
mod main_exit_tests;
