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
//!   app             App state + reducer (section, focus, per-tab state)
//!   theme           palette + shared styles (brand chevron colors)
//!   logo            header art + CLI flag badge derivation
//!   commands        Terminal tab slash-command parser/dispatch
//!   chat            loopback SSE chat client for the served model
//!   data/           pollers: metrics deltas, library scan, kernel rows
//!   render/         one file per section, pure App-state -> Frame

pub mod capture_layer;
pub mod init;
pub mod log_ring;
pub mod shutdown;
pub mod terminal_guard;

pub mod app;
pub mod commands;
pub mod events;
pub mod logo;
pub mod progress;
pub mod theme;

pub mod chat;
pub mod data;
pub mod render;

use std::io::IsTerminal;

/// Start the dashboard thread (head node, TTY mode only — the caller has
/// already gated on `plain_mode`). Captures the tokio runtime handle for the
/// chat client; everything else the TUI reads is process-global.
pub fn start(
    args: crate::cli::ServeArgs,
    progress_rx: std::sync::mpsc::Receiver<capture_layer::ProgressEvent>,
) {
    let runtime = tokio::runtime::Handle::current();
    match std::thread::Builder::new()
        .name("atlas-tui".into())
        .spawn(move || {
            let mut app = app::App::new(args);
            app.chat.set_runtime(runtime);
            events::run(app, progress_rx);
        }) {
        Ok(handle) => {
            *THREAD.lock() = Some(handle);
        }
        Err(e) => tracing::warn!("TUI thread failed to start: {e}"),
    }
}

/// The dashboard thread's handle, for the exit-path join.
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
