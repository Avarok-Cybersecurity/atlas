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
    TUI_ACTIVE.store(true, Ordering::SeqCst);
    let mut terminal = match Terminal::new(CrosstermBackend::new(std::io::stdout())) {
        Ok(t) => t,
        Err(e) => {
            drop(guard);
            tracing::warn!("TUI terminal init failed ({e}); plain logs");
            return;
        }
    };

    let mut last_tick = Instant::now();
    let mut ticks: u32 = 0;
    let mut library_scanned = false;

    loop {
        // 1. Input (poll ≤50ms keeps both input latency and tick cadence).
        if crossterm::event::poll(Duration::from_millis(50)).unwrap_or(false) {
            match crossterm::event::read() {
                Ok(Event::Key(k)) if k.kind != crossterm::event::KeyEventKind::Release => {
                    app.on_key(k)
                }
                Ok(Event::Mouse(m)) => on_mouse(&mut app, m, terminal.size().ok()),
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
        // 3. Tick.
        if last_tick.elapsed() >= TICK {
            last_tick = Instant::now();
            ticks = ticks.wrapping_add(1);
            app.on_tick();
            if ticks.is_multiple_of(SAMPLE_EVERY) {
                app.stats.sample(app.run.as_ref());
            }
            // Library: scan the local cache once, lazily, on first entry
            // (fs-only), attach the recipe cache, and kick one background
            // fetch. The fetch runs on its own std::thread; `poll` below only
            // ever does a try_recv, so a slow network cannot cost a frame.
            if !library_scanned && app.section == Section::Library {
                library_scanned = true;
                app.library = super::data::library::scan(app.args.cache_dir.as_deref());
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
                app.lib.poll(&app.library);
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
        if let Err(e) = terminal.draw(|f| render::draw(f, &app)) {
            tracing::warn!("TUI draw error: {e}; detaching");
            break;
        }
        // 5. Exit conditions.
        if shutdown::requested() {
            break;
        }
        if app.should_quit || app.detach {
            break;
        }
    }
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

fn on_mouse(app: &mut App, m: crossterm::event::MouseEvent, size: Option<ratatui::layout::Size>) {
    let Some(size) = size else { return };
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
            }
        }
        MouseEventKind::ScrollUp => {
            if app.section == Section::Main {
                let cur = app.log_scroll.unwrap_or(0);
                app.log_scroll = Some(cur + 3);
            }
        }
        MouseEventKind::ScrollDown => {
            if app.section == Section::Main {
                match app.log_scroll {
                    Some(n) if n > 3 => app.log_scroll = Some(n - 3),
                    _ => app.log_scroll = None,
                }
            }
        }
        _ => {}
    }
}
