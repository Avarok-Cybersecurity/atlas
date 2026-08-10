// SPDX-License-Identifier: AGPL-3.0-only

//! The event loop's decisions, as values.
//!
//! `events::run` owns a real terminal — a `CrosstermBackend` over stdout, raw
//! mode, and a `poll`/`read` pair only a tty answers. None of it is reachable
//! from the `TestBackend` the rest of this tree's suite drives: the terminal
//! has to actually be there, or actually refuse.
//!
//! That is not a reason for the loop's RULES to live there too. Everything
//! below is a pure function of values the loop already holds, so the suite
//! reaches each one directly and the loop reads as the sequence of side
//! effects it is. The precedent is `events::on_mouse` / `events::sidebar_row`,
//! extracted for the same reason after a click on `Main ▸ Overview` selected
//! Stats.
//!
//! ★ The rule that was a live bug is deliberately NOT here. `TUI_ACTIVE` was
//! released by a single statement at the bottom of the loop that two early
//! `return`s jumped over, and no pure function fixes a missing call — see
//! [`super::init::ActiveClaim`], which fixes it by construction.

use std::sync::mpsc::Receiver;

use crossterm::event::KeyEventKind;

use super::section::Section;

/// Ticks between metrics samples: 1 Hz at the loop's 10 Hz tick.
pub const SAMPLE_EVERY: u32 = 10;

/// Should this key event reach the reducer?
///
/// Only `Release` is dropped, and the polarity is the point: `Repeat` is what
/// a HELD arrow key sends, so a `== Press` test would silently break scrolling
/// by holding a key, and a variant crossterm adds later should arrive at the
/// reducer rather than be swallowed by a filter that never heard of it.
///
/// Reachable only where the terminal reports key-up at all — the Windows
/// console, which emits a record for both edges natively, and the kitty
/// keyboard protocol under `REPORT_EVENT_TYPES`. Atlas pushes no keyboard
/// enhancement flags (see [`super::terminal_guard::TerminalGuard::enter`]), so
/// on a unix tty this guard never fires; on Windows it is the difference
/// between one keystroke doing one thing and every keystroke being applied
/// twice — a typed character doubled, `⇥` skipping a row.
pub fn key_is_actionable(kind: KeyEventKind) -> bool {
    kind != KeyEventKind::Release
}

/// The newest value on a channel, discarding everything older.
///
/// A hot-swap republishes [`super::RunHandles`], and the previous run's is not
/// merely stale: it names levers and a snapshot cell belonging to a model that
/// is no longer loaded, so toggling one would change a run nobody is watching.
/// The queue is therefore DRAINED rather than read one item per tick — the
/// dashboard has no use for any handle but the last.
pub fn newest<T>(rx: &Receiver<T>) -> Option<T> {
    rx.try_iter().last()
}

/// Is this the tick that samples metrics?
///
/// Counted in TICKS rather than wall clock, deliberately: the tick fires when
/// at least `TICK` has elapsed, so under load the loop falls behind and
/// sampling slows with it, instead of firing a catch-up burst of deltas over
/// intervals none of them actually covered.
///
/// The counter wraps, and wrapping is the right choice: `u32::MAX` is not a
/// multiple of ten, so a SATURATING counter would pin there and stop sampling
/// for good. Wrapping costs one short interval — six ticks instead of ten,
/// because zero is a multiple of everything — every 13.6 years of
/// uninterrupted uptime. Pinned below rather than left as a surprise.
pub fn samples_metrics(ticks: u32) -> bool {
    ticks.is_multiple_of(SAMPLE_EVERY)
}

/// What the Library has already done for itself.
///
/// Passed as values so [`tick_work`] is a function rather than a method on
/// `&mut App`: every field is something the loop can read before it decides.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LibraryPhase {
    /// Something changed the local cache since the last scan.
    pub dirty: bool,
    /// A background scan is running; its results are for the cache as it was
    /// when it STARTED.
    pub scan_in_flight: bool,
    /// The recipe store has been attached, so the one GitHub fetch has run.
    pub recipes_attached: bool,
    /// Attaching was tried and cannot succeed in this process.
    pub recipes_unavailable: bool,
}

/// What a tick owes the sections.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TickWork {
    /// Start a background scan of the local cache — and only then clear
    /// `App::library_dirty`.
    pub start_scan: bool,
    /// First entry into the Library: attach the recipe store and kick the one
    /// GitHub fetch it is allowed.
    pub attach_recipes: bool,
    /// Drain the Library's pollers (scan, index, recipe date).
    pub poll_library: bool,
    /// Re-read the run history — a no-op until a run invalidates it.
    pub load_history: bool,
}

/// The lazy-work rules, in one place instead of four `if`s down the tick.
pub fn tick_work(section: Section, lib: LibraryPhase) -> TickWork {
    let in_library = section == Section::Library;
    TickWork {
        // ★ `&& !scan_in_flight` is a FIX, not a transcription.
        //
        // `LibState::start_scan` is idempotent — a second call while one is in
        // flight returns without starting anything — but the loop cleared
        // `library_dirty` either way. So a download that finished DURING a
        // scan had its request dropped on the floor: the scan already running
        // was started before that checkpoint existed, its results do not
        // mention it, and nothing marks the Library dirty a second time. The
        // finished model then did not appear at all until some unrelated event
        // invalidated the list. Asking the question here keeps the flag set
        // until a scan that can actually see the change is the one that starts.
        start_scan: in_library && lib.dirty && !lib.scan_in_flight,
        // ★ ONCE, likewise a fix. `LibState::attached()` stays false when
        // `ArtifactStore::discover()` fails, and it fails only for reasons that
        // cannot change while the process runs (neither `ATLAS_HOME` nor `HOME`
        // is set, or `ATLAS_HOME` is empty). The condition was
        // `in_library && !attached`, so the failure path re-ran at the full
        // 10 Hz tick: one `warn!` every 100 ms into the tee file AND the log
        // pane — where 10 000 identical lines evict the entire ring in under
        // twenty minutes, taking the startup log with them — plus a full
        // catalogue rebuild each time, on the render thread.
        attach_recipes: in_library && !lib.recipes_attached && !lib.recipes_unavailable,
        poll_library: in_library,
        load_history: section == Section::Benchmarks,
    }
}

/// Why the event loop stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    /// The user asked to STOP THE SERVER: `q`, Ctrl+C, `/quit`.
    Quit,
    /// The dashboard is going away and the server keeps serving: `/detach`, or
    /// a terminal that stopped accepting draws.
    Detach,
    /// Something else already requested shutdown — a signal, the panic hook,
    /// the process exit path.
    ShuttingDown,
}

/// Why the loop should stop, or `None` to keep going.
///
/// One function for both questions because they were one decision read twice:
/// the loop broke on `shutdown::requested() || should_quit || detach` and then,
/// forty lines later, re-derived what that had meant from two of the three
/// flags. The third was missing from the second reading, so a `SIGTERM` — a
/// `docker stop`, or `tui::stop_and_join` on the way out of `main` — printed
/// "TUI detached — plain logs resume", which says the server is still up. It
/// was already draining.
///
/// `should_quit` outranks `shutdown_requested` because `/quit` and Ctrl+C set
/// both, and what they mean is the user's intent, not the flag they happened to
/// raise first.
pub fn exit_kind(should_quit: bool, detach: bool, shutdown_requested: bool) -> Option<Exit> {
    if should_quit {
        return Some(Exit::Quit);
    }
    if detach {
        return Some(Exit::Detach);
    }
    shutdown_requested.then_some(Exit::ShuttingDown)
}

#[cfg(test)]
#[path = "events_rules_tests.rs"]
mod tests;
