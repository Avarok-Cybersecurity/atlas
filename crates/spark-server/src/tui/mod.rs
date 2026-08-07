// SPDX-License-Identifier: AGPL-3.0-only

//! Atlas TUI — the ratatui dashboard for `spark serve`.
//!
//! Activation is strictly opt-out-safe: [`plain_mode`] must return `false`
//! before any TUI machinery is touched, and when it returns `true` the caller
//! (`main.rs`) installs the pre-TUI `tracing_subscriber::fmt().init()`
//! UNCHANGED — that plain-log format is a compatibility contract with every
//! benchmark driver and gate script that greps `docker logs`.
//!
//! Module map (each file ≤450 LoC for the CI cap):
//!   init            subscriber stack, SwitchableWriter, tee file, TUI_ACTIVE
//!   terminal_guard  raw-mode RAII + idempotent restore + panic hook
//!   shutdown        one shutdown path: signals, Ctrl+C-as-key, /quit
//!   log_ring        structured log capture + global ring for the log pane
//!   capture_layer   typed startup-progress event decoding
//!   progress        ProgressModel — phases/shards/layers/ETA state machine
//!   events          input/tick event loop on the dedicated "atlas-tui" thread
//!   events_rules    the loop's decisions as pure functions, so they are testable
//!   section         Section — the sidebar/nav SSOT
//!   app             App state + reducer (section, focus, per-tab state)
//!   app_quit        what `q` costs, and when it costs a second press
//!   bench_state     Benchmarks section state + the executor's channels
//!   bench_keys      Benchmarks key handling (list / params / run / history)
//!   theme           palette + shared styles (brand chevron colors)
//!   format          byte counts and scheduler enums, as the screen says them
//!   logo            header art + CLI flag badge derivation
//!   commands        Terminal tab slash-command parser/dispatch
//!   chat            Chat transcript state + delta reducer
//!   chat_stream     loopback SSE chat client for the served model
//!   chat_thinking   thinking: what the client asks for, and how it is drawn
//!   data/           pollers: metrics deltas, library scan, kernel rows
//!   render/         one file per section, pure App-state -> Frame
//!   worker          the one way to run work off the render thread
//!
//! # Where the sync/async boundary is
//!
//! **The render thread never polls a future. Its only contact with work
//! happening elsewhere is `try_recv` on a channel.** That single rule is what
//! keeps this tree comprehensible, and
//! `.github/workflows/tui-threading.yml` fails the build if it is broken. Two
//! consequences worth stating plainly, because both were once decided
//! case-by-case:
//!
//! **Which side does a piece of work belong on?** Ask what it needs, not what
//! is fashionable:
//!
//!   * ASYNC, on the SERVING runtime — work that is already async, or that
//!     talks to this server over its own HTTP API. `chat` (loopback SSE) and
//!     `bench_preflight` + `atlas-plugin`'s executor (benchmarks are async
//!     end to end). These take a `tokio::runtime::Handle` captured at start.
//!   * SYNC, on a named `std::thread` — everything with no runtime need:
//!     blocking HTTPS via `ureq` (recipe index, recipe dates, model downloads,
//!     freshness checks), filesystem walks (the HF cache scan), and CUDA (the
//!     model swap). None of these are faster async, and an async filesystem
//!     does not exist — `tokio::fs` is a thread pool with nicer syntax.
//!
//! **Both sides answer the same way**, which is why the render loop needs only
//! one shape: a `std::sync::mpsc::Sender`, drained with `try_recv` on the tick.
//! There is no second idiom to learn and no third place to look.
//!
//! The threads, all named so a stack dump during an incident is attributable:
//! `atlas-tui` (this render loop) · `atlas-recipes` · `atlas-recipe-date` ·
//! `atlas-libscan` · `atlas-download` · `atlas-freshness` · `atlas-swap`.
//!
//! The first five one-shot workers go through [`worker::spawn`], which owns the
//! part that is easy to forget: answering anyway when the thread will not
//! start, so a receiver cannot be polled forever. Two do not, for reasons:
//! `atlas-download` is a STREAMING producer (many progress messages, not one
//! result), and `atlas-swap` sends only on FAILURE — silence means "still
//! loading" — so an always-send helper would change what its silence means.

pub mod capture_layer;
pub mod clipboard;
pub mod init;
pub mod log_ring;
pub mod selection;
pub mod shutdown;
pub mod terminal_guard;

pub mod app;
mod app_input;
pub mod app_library;
pub mod app_quit;
pub mod app_scroll;
pub mod bench_host;
pub mod bench_keys;
pub mod bench_preflight;
pub mod bench_state;
pub mod commands;
pub mod download_state;
pub mod events;
pub mod events_rules;
pub mod format;
pub mod lib_dates;
pub mod lib_keys;
pub mod lib_scan;
pub mod lib_state;
pub mod logo;
pub mod progress;
pub mod section;
pub mod theme;
pub mod worker;

pub mod chat;
pub mod chat_stream;
pub mod chat_thinking;
pub mod data;
pub mod render;

use std::io::IsTerminal;

/// A live run's levers, published to the dashboard when the scheduler starts
/// so `/watchdog on|off` toggles the run's own flag instead of a process
/// global. Sent again on a hot-swap, replacing the previous run's handle.
pub type RunLevers = std::sync::Arc<crate::scheduler::levers::SchedLevers>;

/// Everything the dashboard needs a handle to in the LIVE run: the levers it
/// can toggle and the snapshot cell it polls. Published together because
/// they arrive together and a swap replaces both.
#[derive(Clone)]
pub struct RunHandles {
    pub levers: RunLevers,
    pub snapshot: std::sync::Arc<crate::scheduler::snapshot::SnapshotCell>,
}

/// Start the dashboard thread (head node, TTY mode only — the caller has
/// already gated on `plain_mode`). Captures the tokio runtime handle for the
/// chat client, and returns the sender the caller uses to publish each run's
/// levers. A failed spawn still returns a usable sender whose receiver is
/// gone: publishing becomes a no-op rather than a caller-side special case.
pub fn start(
    args: crate::cli::ServeArgs,
    progress_rx: std::sync::mpsc::Receiver<capture_layer::ProgressEvent>,
    host: std::sync::Arc<crate::main_modules::model_host::ModelHost>,
) -> std::sync::mpsc::Sender<RunHandles> {
    let (levers_tx, levers_rx) = std::sync::mpsc::channel::<RunHandles>();
    let runtime = tokio::runtime::Handle::current();
    // Claim the terminal for the dashboard HERE, on the caller's thread,
    // before the TUI thread exists.
    //
    // `SwitchableIo` sends log output to stdout while this is false, and it
    // used to be set inside the spawned thread's loop — so every line the main
    // thread logged between the spawn and that store went to stdout, and any
    // that landed after the thread had entered the alternate screen was
    // written INTO it. Two lines were enough: the screen scrolled by two rows,
    // ratatui's idea of where it had drawn no longer matched the terminal, and
    // its diff then repaired nothing. That is the ghost header sitting above a
    // live list on the very first screen a user sees.
    //
    // Setting it before the spawn closes the window: from here on the only
    // writer to this terminal is the dashboard.
    init::TUI_ACTIVE.store(true, std::sync::atomic::Ordering::SeqCst);
    match std::thread::Builder::new()
        .name("atlas-tui".into())
        .spawn(move || {
            let port = args.port;
            let model = args
                .model_name
                .clone()
                .or_else(|| args.model.clone())
                .unwrap_or_default();
            let cache_dir = args.cache_dir.clone();
            let mut app = app::App::new(args);
            // The Serve Matrix boots checkpoints through this process. Install
            // the seam before the section can start a run; without it the
            // benchmark refuses to load and says why, which is the correct
            // behaviour for a harness with no server, not for this one.
            atlas_plugin::benchmarks::serve_matrix::host::install(std::sync::Arc::new(
                bench_host::TuiServeHost::new(host.clone(), cache_dir),
            ));
            app.host = Some(host);
            app.chat.set_runtime(runtime.clone());
            // Benchmarks default to the server they are attached to. A store
            // that cannot be created (no HOME, read-only) is not fatal: the
            // section reports it when a run is started, rather than taking the
            // whole dashboard down at boot.
            match atlas_plugin::ArtifactStore::discover() {
                Ok(store) => app.bench.attach(
                    atlas_plugin::BenchmarkExecutor::new(runtime, store),
                    atlas_plugin::TargetEndpoint::local(port, model),
                ),
                Err(e) => tracing::warn!("benchmarks unavailable: {e:#}"),
            }
            events::run(app, progress_rx, levers_rx);
        }) {
        Ok(handle) => {
            *THREAD.lock() = Some(handle);
        }
        Err(e) => tracing::warn!("TUI thread failed to start: {e}"),
    }
    levers_tx
}

/// The dashboard thread's handle, for the exit-path join.
///
/// STATIC, DELIBERATELY — process lifecycle. There is exactly one dashboard
/// thread per process, and `stop_and_join` runs on the way out, from a path
/// that has no server state left to reach through. It holds a join handle,
/// not a value: nothing model-derived can go stale in it.
static THREAD: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>> =
    parking_lot::Mutex::new(None);

/// Exit path: ask the TUI loop to stop and wait (bounded) for it to drop its
/// TerminalGuard. Closes the startup race where `serve()` errors in the
/// milliseconds BEFORE the thread enters raw mode — a bare `restore()` then
/// runs as a no-op and the thread wrecks the terminal as the process dies.
/// After the join (or timeout), the idempotent `restore()` is the backstop.
pub fn stop_and_join(timeout: std::time::Duration) {
    let Some(handle) = THREAD.lock().take() else {
        return;
    };
    shutdown::request("process exit");
    let deadline = std::time::Instant::now() + timeout;
    while !handle.is_finished() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    if handle.is_finished() {
        let _ = handle.join();
    }
}

/// True when the TUI must NOT start and `main.rs` keeps the byte-identical
/// plain fmt subscriber.
///
/// Gates, in order: explicit `--no-tui`, `ATLAS_NO_TUI=1`, non-interactive
/// stdout OR stdin (docker `-t` without `-i` therefore stays plain), and
/// `TERM=dumb`. EP workers are additionally refused in `serve()` — belt and
/// braces, since rank isn't parsed yet when this runs.
pub fn plain_mode(no_tui_flag: bool) -> bool {
    if no_tui_flag {
        return true;
    }
    if std::env::var("ATLAS_NO_TUI").as_deref() == Ok("1") {
        return true;
    }
    if !std::io::stdout().is_terminal() || !std::io::stdin().is_terminal() {
        return true;
    }
    matches!(std::env::var("TERM").as_deref(), Ok("dumb") | Err(_))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_flag_always_wins() {
        // Regardless of environment, --no-tui forces plain mode.
        assert!(plain_mode(true));
    }

    #[test]
    fn piped_test_runner_is_plain() {
        // Under `cargo test` stdout is captured (not a TTY) => plain. This is
        // exactly the property the benchmark rigs rely on.
        assert!(plain_mode(false));
    }
}
