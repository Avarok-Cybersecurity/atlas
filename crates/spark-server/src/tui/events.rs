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
use super::init::TUI_ACTIVE;
use super::terminal_guard::TerminalGuard;
use super::{render, shutdown};

const TICK: Duration = Duration::from_millis(100);
const SAMPLE_EVERY: u32 = 10; // 1 Hz metrics sampling at the 10 Hz tick

pub fn run(
    mut app: App,
    progress_rx: Receiver<ProgressEvent>,
    levers_rx: Receiver<crate::tui::RunHandles>,
) {
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

    loop {
        // 1. Input (poll ≤50ms keeps both input latency and tick cadence).
        if crossterm::event::poll(Duration::from_millis(50)).unwrap_or(false) {
            match crossterm::event::read() {
                Ok(Event::Key(k)) if k.kind != crossterm::event::KeyEventKind::Release => {
                    app.on_key(k)
                }
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
        if let Some(h) = levers_rx.try_iter().last() {
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
            if ticks.is_multiple_of(SAMPLE_EVERY) {
                app.stats.sample(app.run.as_ref());
            }
            // Library: scan the local cache lazily on entry, and again
            // whenever something changed it — a finished download must appear
            // without a restart. The scan now runs on its own thread
            // (`poll_scan` below only try_recvs), because it stats every blob
            // directory and was doing that on the render thread.
            // The reducer cannot reach `App`, so it raises a flag here.
            if std::mem::take(&mut app.lib.mark_dirty) {
                app.library_dirty = true;
            }
            if app.library_dirty && app.section == Section::Library {
                app.library_dirty = false;
                app.lib.start_scan(app.args.cache_dir.as_deref());
            }
            // The recipe half is genuinely once-only: a rescan must NOT
            // re-trigger a GitHub fetch, which is rate-limited and unrelated to
            // what changed on disk.
            if app.section == Section::Library && !app.lib.attached() {
                match atlas_plugin::ArtifactStore::discover() {
                    Ok(store) => {
                        app.lib.attach(store.root().to_path_buf(), &app.library);
                        app.lib.refresh();
                    }
                    // No HOME, or a read-only one: the local scan still renders.
                    Err(e) => {
                        tracing::warn!("recipes unavailable: {e:#}");
                        app.lib.rebuild(&app.library);
                    }
                }
            }
            if app.section == Section::Library {
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
            if app.section == Section::Benchmarks {
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
                break;
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
        if shutdown::requested() {
            break;
        }
        if app.should_quit || app.detach {
            break;
        }
    }
    // Nothing is going to render the answer now, so stop asking for it. Eight
    // in-flight recipe fetches would otherwise run to the 20 s timeout while
    // the process is trying to exit.
    app.lib.cancel_refresh();
    TUI_ACTIVE.store(false, Ordering::SeqCst);
    drop(guard); // restore terminal; logs fall back to stdout
    if app.should_quit && !app.detach {
        shutdown::request("TUI quit");
    } else {
        tracing::info!(
            "TUI detached — plain logs resume (full history: {})",
            super::init::tee_file_path().unwrap_or("-")
        );
    }
}

/// What the caller must do after a mouse event it cannot do itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MouseOutcome {
    None,
    /// A drag finished: read the selection out of the rendered frame and copy.
    CopySelection,
}

fn on_mouse(
    app: &mut App,
    m: crossterm::event::MouseEvent,
    size: Option<ratatui::layout::Size>,
) -> MouseOutcome {
    let Some(size) = size else {
        return MouseOutcome::None;
    };
    let header_h: u16 = if size.height >= 28 { 3 } else { 1 };
    let sidebar_w: u16 = if size.width >= 96 { 18 } else { 4 };
    match m.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            if m.column < sidebar_w && m.row >= header_h {
                // Sidebar rows include expanded subsection lines; map the
                // clicked visual row back to a section index conservatively
                // (subsections only render under the active section).
                let mut visual = (m.row - header_h) as usize;
                let active_idx = Section::ALL
                    .iter()
                    .position(|s| *s == app.section)
                    .unwrap_or(0);
                // `Section::subs` is the SSOT for what the sidebar draws and
                // what ⇥ stops on; deriving the mouse offset from anything else
                // is how a sixth section silently breaks clicking.
                let subs = app.section.subs().len();
                if visual > active_idx + subs {
                    visual -= subs;
                }
                app.sidebar_click(visual);
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
        // hold the text that was chosen.
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
