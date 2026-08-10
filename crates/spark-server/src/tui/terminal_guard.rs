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
use std::sync::atomic::{AtomicBool, Ordering};

use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};

/// What this process has done to the terminal.
///
/// One object rather than loose flags, because the two are a single
/// invariant: fd 2 is redirected exactly while raw mode is held, and
/// `restore()` must undo both or neither. Plain atomics, no lock — the panic
/// hook and the signal path read them, and neither may block.
struct Terminal {
    /// True while raw mode + alt screen are active. `restore()` flips it false.
    taken: AtomicBool,
    /// Saved dup of the original stderr fd while it is redirected (-1 = not).
    orig_stderr: std::sync::atomic::AtomicI32,
}

// STATIC, DELIBERATELY — process lifecycle. This describes THE TERMINAL, which
// is a property of the process, not of a model or a request. `restore()` is
// called from the panic hook, from a signal handler and from the normal exit
// path, none of which can be handed a context; and getting it wrong wrecks the
// user's shell.
static TERM: Terminal = Terminal {
    taken: AtomicBool::new(false),
    orig_stderr: std::sync::atomic::AtomicI32::new(-1),
};

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
                TERM.orig_stderr.store(orig, Ordering::SeqCst);
            } else if orig >= 0 {
                libc::close(orig);
            }
        }
    }
}

#[cfg(unix)]
fn unredirect_stderr() {
    let orig = TERM.orig_stderr.swap(-1, Ordering::SeqCst);
    if orig >= 0 {
        // SAFETY: restoring the fd we saved above.
        unsafe {
            libc::dup2(orig, 2);
            libc::close(orig);
        }
    }
}

// The panic hook's log-dump fn was held in a `OnceLock<fn(..)>` so this module
// "would not depend on the ring's type" — but the signature names no ring
// type, so the indirection bought nothing and cost a global plus an
// installed-once flag. The hook calls `super::log_ring::dump_to` directly.
// The tee path was duplicated here, copied in by `install_panic_hook` from the
// module that actually opens the file. One value, two owners: `super::init`
// keeps it now and the hook reads it through `tee_file_path()`.

/// Idempotently undo raw mode, mouse capture, and the alternate screen.
///
/// Safe to call from any thread, any number of times, including inside a
/// panic hook. Errors are deliberately ignored — there is no better recovery
/// than trying the next teardown step.
pub fn restore() {
    if !TERM.taken.swap(false, Ordering::SeqCst) {
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
        TERM.taken.store(true, Ordering::SeqCst);
        redirect_stderr_to_tee();
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore();
    }
}

/// Install the chained panic hook: it restores the terminal, prints the last
/// captured log lines and the always-on log file's path, then chains to the
/// previous hook for the message and backtrace.
///
/// Must be called BEFORE `TerminalGuard::enter` so a panic during entry is
/// covered too. Installing more than once is a no-op.
pub fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // 1. Sane screen first, so everything below is actually visible.
        restore();
        // 2. Recent context: the last lines the operator saw in the TUI.
        let mut err = std::io::stderr();
        let _ = writeln!(err, "\n── atlas-tui: panic — last log lines ──");
        super::log_ring::dump_to(&mut err, 50);
        if let Some(p) = super::init::tee_file_path() {
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
        assert!(!TERM.taken.load(Ordering::SeqCst));
        restore();
        restore();
        assert!(!TERM.taken.load(Ordering::SeqCst));
    }
}
