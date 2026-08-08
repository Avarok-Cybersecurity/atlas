// SPDX-License-Identifier: AGPL-3.0-only

//! The TUI event loop. Runs on a dedicated OS thread ("atlas-tui"),
//! synchronous crossterm polling + a 10 Hz render tick; tokio is never on the
//! render path. Mirrors the scheduler's dedicated-thread pattern.

use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use crossterm::event::{Event, MouseButton, MouseEventKind};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use super::app::{App, Section};
use super::capture_layer::ProgressEvent;
use super::events_rules::{Exit, LibraryPhase, exit_kind, key_is_actionable, newest, tick_work};
use super::init::{ActiveClaim, TUI_ACTIVE};
use super::terminal_guard::TerminalGuard;
use super::{events_rules, render, shutdown};

const TICK: Duration = Duration::from_millis(100);

pub fn run(
    mut app: App,
    progress_rx: Receiver<ProgressEvent>,
    levers_rx: Receiver<crate::tui::RunHandles>,
) {
    // Taken FIRST, so it outlives every `return` below. `tui::start` already
    // set the flag before this thread existed (see the comment there); this
    // takes ownership of giving it back, which the two bail-outs below used to
    // skip — see `init::ActiveClaim`.
    let claim = ActiveClaim::claim();
    super::terminal_guard::install_panic_hook();
    let guard = match TerminalGuard::enter() {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!("TUI unavailable ({e}); continuing with plain logs");
            return;
        }
    };
    // Already claimed in `tui::start`, before this thread existed — see the
    // comment there. Kept as a no-op assertion of the invariant rather than a
    // second place that decides it.
    debug_assert!(
        TUI_ACTIVE.load(Ordering::SeqCst),
        "start() claims the terminal"
    );
    let mut terminal = match Terminal::new(CrosstermBackend::new(std::io::stdout())) {
        Ok(t) => t,
        Err(e) => {
            drop(guard);
            tracing::warn!("TUI terminal init failed ({e}); plain logs");
            return;
        }
    };

    let mut last_tick = Instant::now();
    // Set when a drag ends; consumed after the next draw, which is the first
    // moment the selected text exists anywhere readable.
    let mut copy_after_draw = false;
    let mut copy_result: Option<(Result<usize, String>, bool)> = None;
    let mut clear_selection = false;
    let mut ticks: u32 = 0;

    // `break`s with the reason, so the classification below reads the same
    // decision the loop stopped on rather than re-deriving it from the flags.
    let exit = loop {
        // 1. Input (poll ≤50ms keeps both input latency and tick cadence).
        if crossterm::event::poll(Duration::from_millis(50)).unwrap_or(false) {
            match crossterm::event::read() {
                Ok(Event::Key(k)) if key_is_actionable(k.kind) => app.on_key(k),
                Ok(Event::Mouse(m)) => {
                    let size = terminal.size().ok();
                    // The copy cannot happen here: the text lives in the
                    // rendered frame, and between draws there ISN'T one —
                    // ratatui swaps and RESETS its buffers after each draw, so
                    // reading `current_buffer_mut()` now returns the blank
                    // frame about to be drawn into. Flag it and read the frame
                    // that `terminal.draw` hands back below.
                    if on_mouse(&mut app, m, size) == MouseOutcome::CopySelection {
                        copy_after_draw = true;
                    }
                }
                // ratatui diffs against the frame it last drew, so on a resize
                // the cells the OLD layout wrote and the NEW one does not
                // reach are never overwritten — they persist as fragments of a
                // previous frame. Observed on a pane that grew from 80x24: a
                // stale panel title reading "0 recipes never fetched" sat above
                // a list of 25. A full clear discards the diff baseline.
                Ok(Event::Resize(..)) => {
                    let _ = terminal.clear();
                }
                _ => {}
            }
        }
        // 2. Data ingress.
        // The newest published run wins — a hot-swap replaces the handle.
        if let Some(h) = newest(&levers_rx) {
            app.run = Some(h);
        }
        for ev in progress_rx.try_iter() {
            app.progress.apply(ev);
        }
        app.chat.pump();
        app.bench.pump();
        // Downloads are pumped UNCONDITIONALLY, not only in the Library: one
        // started there must still finish, and report, if the user navigates
        // away to watch the logs — the same argument as `poll_preflight`.
        if let Some(settled) = app.download.pump() {
            // Either outcome changed the cache: a finished download adds a
            // model, a stopped one changes the reported size.
            app.library_dirty = true;
            if let crate::tui::download_state::Settled::Finished(_) = settled {
                // A new model may satisfy a recipe that had none.
                app.repaint = true;
            }
        }
        if let Some((text, error)) = app.download.last_message.take() {
            app.toast(text, error);
        }
        // 3. Tick.
        if last_tick.elapsed() >= TICK {
            last_tick = Instant::now();
            ticks = ticks.wrapping_add(1);
            app.on_tick();
            if events_rules::samples_metrics(ticks) {
                app.stats.sample(app.run.as_ref());
            }
            // The Library reducer cannot reach `App`, so it raises a flag the
            // tick drains into the one `App` owns.
            if std::mem::take(&mut app.lib.mark_dirty) {
                app.library_dirty = true;
            }
            // Everything lazy on this tick decided in one place — see
            // `events_rules::tick_work` for why each condition reads as it
            // does, two of which are corrections rather than transcriptions.
            let work = tick_work(
                app.section,
                LibraryPhase {
                    dirty: app.library_dirty,
                    scan_in_flight: app.lib.scan_in_flight(),
                    recipes_attached: app.lib.attached(),
                    recipes_unavailable: app.lib.recipes_unavailable(),
                },
            );
            // The scan stats every blob directory, so it runs on its own thread
            // and `poll_scan` below only try_recvs. The flag is cleared HERE,
            // where a scan that can see the change is what starts.
            if work.start_scan {
                app.library_dirty = false;
                app.lib.start_scan(app.args.cache_dir.as_deref());
            }
            // Once only: a rescan must NOT re-trigger the GitHub fetch, which
            // is rate-limited and unrelated to what changed on disk.
            if work.attach_recipes {
                app.attach_recipes();
            }
            if work.poll_library {
                if let Some(found) = app.lib.poll_scan() {
                    app.library = found;
                    app.lib.rebuild(&app.library);
                }
                app.lib.poll(&app.library);
                app.lib.poll_date();
                // A recipe carrying no `metadata.updated` gets its date from
                // GitHub's commit history — but only the one being read, and
                // only once. `want_date_for` is a no-op unless all three of
                // those hold, so calling it every tick is free.
                if let Some(id) = app.lib.visible_recipe_id() {
                    app.lib.want_date_for(&id);
                }
            }
            // Run history: lazily too, and re-read after a run persists a frame
            // (`load_history` is a no-op until something invalidates it).
            if work.load_history {
                app.bench.load_history();
            }
            // Unconditional: a pre-flight started in Benchmarks must still
            // resolve if the user navigates away, or the run is stranded on a
            // check nobody is draining.
            app.bench.poll_preflight();
        }
        // 4. Render.
        // A wholesale layout change re-establishes the diff baseline; see
        // `App::repaint`.
        if std::mem::take(&mut app.repaint) {
            let _ = terminal.clear();
        }
        match terminal.draw(|f| render::draw(f, &app)) {
            Err(e) => {
                tracing::warn!("TUI draw error: {e}; detaching");
                // The terminal stopped accepting frames. The server is
                // untouched and keeps serving, which is exactly what a detach
                // is — so it leaves by the same door `/detach` does.
                break Exit::Detach;
            }
            Ok(frame) => {
                // `CompletedFrame` borrows the buffer that was just rendered —
                // the only place the on-screen TEXT exists, and the reason the
                // copy is deferred to here rather than done in the handler.
                if std::mem::take(&mut copy_after_draw)
                    && let Some(sel) = app.selection
                {
                    let text = super::selection::extract(frame.buffer, frame.area, &sel);
                    copy_result = Some((super::clipboard::copy(&text), text.is_empty()));
                    // Drop the selection now that it has been read. It has
                    // done its job, and a highlight that outlives the copy
                    // goes on painting reversed cells over whatever the user
                    // navigates to next — the coordinates are screen cells,
                    // so they mean something different on every screen.
                    clear_selection = true;
                }
            }
        }
        // Both of these need `&mut app`, which the frame borrow above forbids.
        if std::mem::take(&mut clear_selection) {
            app.selection = None;
        }
        if let Some((res, was_empty)) = copy_result.take() {
            match res {
                Ok(n) => app.toast(format!("Copied {n} characters to clipboard"), false),
                // A stray drag across blank space is a non-event, not an error
                // worth interrupting anyone about.
                Err(e) if was_empty => tracing::debug!("copy skipped: {e}"),
                Err(e) => app.toast(e, true),
            }
        }
        // 5. Exit conditions.
        if let Some(e) = exit_kind(app.should_quit, app.detach, shutdown::requested()) {
            break e;
        }
    };
    // Nothing is going to render the answer now, so stop asking for it. Eight
    // in-flight recipe fetches would otherwise run to the 20 s timeout while
    // the process is trying to exit.
    app.lib.cancel_refresh();
    // Restore the terminal FIRST, release the claim second: the old order let
    // go of the claim while raw mode and the alternate screen were still up, so
    // any line logged in that window was written into a screen ratatui still
    // believed it owned. The two lines below need stdout back, hence the
    // explicit drop rather than leaving it to the end of the function.
    drop(guard);
    drop(claim);
    let log = super::init::tee_file_path().unwrap_or("-");
    match exit {
        Exit::Quit => shutdown::request("TUI quit"),
        Exit::Detach => {
            tracing::info!("TUI detached — plain logs resume (full history: {log})")
        }
        // NOT "detached": the server this would tell the operator is still up
        // is already draining. Reached by SIGTERM (`docker stop`) and by
        // `tui::stop_and_join` on the way out of `main`.
        Exit::ShuttingDown => {
            tracing::info!("TUI closed — shutdown already under way (full history: {log})")
        }
    }
}

/// What the caller must do after a mouse event it cannot do itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MouseOutcome {
    None,
    /// A drag finished: read the selection out of the rendered frame and copy.
    CopySelection,
}

/// What a clicked sidebar row draws.
enum SidebarRow {
    /// Index into [`Section::ALL`].
    Section(usize),
    /// Subsection of the ACTIVE section, the only one drawing any.
    Sub(usize),
}

/// Map a visual sidebar row back to what is drawn on it.
///
/// The active section's subsection rows are inserted UNDERNEATH it, so they
/// shift every row below them — and they belong to the section above, not to
/// the section index they happen to sit at. Only the shift was handled, and
/// only for the rows below: a click on Main ▸ Overview selected Stats.
fn sidebar_row(active_idx: usize, subs: usize, visual: usize) -> SidebarRow {
    if visual <= active_idx {
        SidebarRow::Section(visual)
    } else if visual <= active_idx + subs {
        SidebarRow::Sub(visual - active_idx - 1)
    } else {
        SidebarRow::Section(visual - subs)
    }
}

fn on_mouse(
    app: &mut App,
    m: crossterm::event::MouseEvent,
    size: Option<ratatui::layout::Size>,
) -> MouseOutcome {
    let Some(size) = size else {
        return MouseOutcome::None;
    };
    // The renderer's own breakpoints, not a second copy of them — see
    // `render::Chrome`.
    let chrome = render::Chrome::of(size);
    match m.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            if m.column < chrome.sidebar_w && m.row >= chrome.header_h {
                let visual = (m.row - chrome.header_h) as usize;
                let active_idx = Section::ALL
                    .iter()
                    .position(|s| *s == app.section)
                    .unwrap_or(0);
                // `Section::subs` is the SSOT for what the sidebar draws and
                // what ⇥ stops on; deriving the mouse offset from anything else
                // is how a sixth section silently breaks clicking. A narrow
                // sidebar draws the icons only, so it offsets nothing.
                let subs = if chrome.full_sidebar() {
                    app.section.subs().len()
                } else {
                    0
                };
                match sidebar_row(active_idx, subs, visual) {
                    SidebarRow::Section(i) => app.sidebar_click(i),
                    SidebarRow::Sub(i) => app.sidebar_sub_click(i),
                }
                // A sidebar click is navigation, not the start of a drag.
                app.selection = None;
            } else {
                // Anywhere else, the button going down is a potential drag.
                // Nothing is copied until it actually moves.
                app.selection = Some(super::selection::Selection::new((m.column, m.row)));
            }
        }
        MouseEventKind::Drag(MouseButton::Left) => {
            if let Some(sel) = app.selection.as_mut() {
                sel.cursor = (m.column, m.row);
            }
        }
        MouseEventKind::Up(MouseButton::Left) => {
            // Copy on release, and only if the pointer moved: a plain click
            // would otherwise copy one character and raise a toast every time
            // anyone touched the dashboard.
            return match app.selection {
                Some(sel) if sel.is_drag() => MouseOutcome::CopySelection,
                _ => {
                    app.selection = None;
                    MouseOutcome::None
                }
            };
        }
        // Scrolling moves the content out from under the highlight, so the
        // same argument as a keystroke applies: the cells it covers no longer
        // hold the text that was chosen. (Mouse tests live in `events_tests`.)
        MouseEventKind::ScrollUp => {
            app.selection = None;
            app.scroll(-3);
        }
        MouseEventKind::ScrollDown => {
            app.selection = None;
            app.scroll(3);
        }
        _ => {}
    }
    MouseOutcome::None
}

#[cfg(test)]
#[path = "events_tests.rs"]
mod tests;
