// SPDX-License-Identifier: AGPL-3.0-only

//! Raw-mode/alternate-screen lifecycle with crash safety.
//!
//! The terminal MUST be restored on every exit path — clean quit, detach,
//! `?`-panic on ANY thread (CUDA `.expect()`s in the scheduler thread
//! included), or the process unwinding out of `main`. Three layers:
//!
//!  1. [`TerminalGuard`] — RAII: enters raw mode + alternate screen + mouse
//!     capture on construction, restores on `Drop`.
//!  2. [`restore`] — idempotent (an `AtomicBool` guards double-restore), so
//!     the guard's Drop and the panic hook can both call it safely.
//!  3. A process-global panic hook (installed once, chained to the previous
//!     hook) that restores the terminal FIRST — so the panic message and
//!     backtrace print onto a sane screen — then dumps the newest log-ring
//!     lines to stderr and points at the tee file.
//!
//! SIGKILL cannot be caught; `reset`/`stty sane` is the documented recovery.

use std::io::Write;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};

/// True while raw mode + alt screen are active. `restore()` flips it false.
static TERMINAL_TAKEN: AtomicBool = AtomicBool::new(false);
/// Saved dup of the original stderr fd while it is redirected (-1 = not).
static ORIG_STDERR: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// Redirect fd 2 into the tee file while the TUI owns the screen. Ten-plus
/// `eprintln!` sites exist in spark-model/spark-runtime (plus anything a C
/// library prints); any one of them scribbles over the raw-mode frame. The
/// writes land in the tee file instead; `restore()` puts the real stderr back
/// BEFORE the panic hook prints, so panics stay visible on the terminal.
/// No-op off unix: `libc::dup`/`dup2` and `std::os::fd` do not exist there, and
/// the Windows console does not share the fd-2 aliasing this works around. The
/// TUI still runs; stray `eprintln!`s are simply not captured.
#[cfg(not(unix))]
fn redirect_stderr_to_tee() {}

#[cfg(not(unix))]
fn unredirect_stderr() {}

#[cfg(unix)]
fn redirect_stderr_to_tee() {
    if let Some(tee_fd) = super::init::tee_raw_fd() {
        // SAFETY: plain fd juggling on fds we own; dup/dup2 are async-signal-
        // safe and the saved fd is released in restore().
        unsafe {
            let orig = libc::dup(2);
            if orig >= 0 && libc::dup2(tee_fd, 2) >= 0 {
                ORIG_STDERR.store(orig, Ordering::SeqCst);
            } else if orig >= 0 {
                libc::close(orig);
            }
        }
    }
}

#[cfg(unix)]
fn unredirect_stderr() {
    let orig = ORIG_STDERR.swap(-1, Ordering::SeqCst);
    if orig >= 0 {
        // SAFETY: restoring the fd we saved above.
        unsafe {
            libc::dup2(orig, 2);
            libc::close(orig);
        }
    }
}

/// Snapshot fn the panic hook uses to dump recent log lines. Set by
/// `install_panic_hook`; kept as a plain fn pointer so this module does not
/// depend on the ring's type.
static RING_DUMP: OnceLock<fn(&mut dyn Write, usize)> = OnceLock::new();
/// Tee-file path, printed by the panic hook so operators know where the full
/// log went while the alt screen was eating stdout.
static TEE_PATH: OnceLock<String> = OnceLock::new();

/// Idempotently undo raw mode, mouse capture, and the alternate screen.
///
/// Safe to call from any thread, any number of times, including inside a
/// panic hook. Errors are deliberately ignored — there is no better recovery
/// than trying the next teardown step.
pub fn restore() {
    if !TERMINAL_TAKEN.swap(false, Ordering::SeqCst) {
        return;
    }
    unredirect_stderr();
    let _ = disable_raw_mode();
    let mut out = std::io::stdout();
    let _ = crossterm::execute!(out, DisableMouseCapture, LeaveAlternateScreen);
    let _ = crossterm::execute!(out, crossterm::cursor::Show);
    let _ = out.flush();
}

/// RAII terminal ownership for the TUI thread.
pub struct TerminalGuard;

impl TerminalGuard {
    /// Enter raw mode + alternate screen + mouse capture.
    pub fn enter() -> std::io::Result<Self> {
        enable_raw_mode()?;
        crossterm::execute!(std::io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
        TERMINAL_TAKEN.store(true, Ordering::SeqCst);
        redirect_stderr_to_tee();
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore();
    }
}

/// Install the chained panic hook. `ring_dump(w, n)` writes the newest `n`
/// captured log lines to `w`; `tee_path` is the always-on log file.
///
/// Must be called BEFORE `TerminalGuard::enter` so a panic during entry is
/// covered too. Installing more than once is a no-op.
pub fn install_panic_hook(ring_dump: fn(&mut dyn Write, usize), tee_path: &str) {
    if RING_DUMP.set(ring_dump).is_err() {
        return; // already installed
    }
    let _ = TEE_PATH.set(tee_path.to_string());
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // 1. Sane screen first, so everything below is actually visible.
        restore();
        // 2. Recent context: the last lines the operator saw in the TUI.
        let mut err = std::io::stderr();
        let _ = writeln!(err, "\n── atlas-tui: panic — last log lines ──");
        if let Some(dump) = RING_DUMP.get() {
            dump(&mut err, 50);
        }
        if let Some(p) = TEE_PATH.get() {
            let _ = writeln!(err, "── full log: {p} ──");
        }
        // 3. The original hook prints the panic message + backtrace.
        previous(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_is_idempotent_when_never_taken() {
        // Never entered: restore must be a no-op that doesn't touch the tty.
        assert!(!TERMINAL_TAKEN.load(Ordering::SeqCst));
        restore();
        restore();
        assert!(!TERMINAL_TAKEN.load(Ordering::SeqCst));
    }
}
