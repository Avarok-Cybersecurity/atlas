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
use super::{log_ring, render, shutdown};

const TICK: Duration = Duration::from_millis(100);
const SAMPLE_EVERY: u32 = 10; // 1 Hz metrics sampling at the 10 Hz tick

pub fn run(mut app: App, progress_rx: Receiver<ProgressEvent>) {
    super::terminal_guard::install_panic_hook(
        log_ring::dump_to,
        super::init::tee_file_path().unwrap_or("(no tee file)"),
    );
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
                Ok(Event::Resize(..)) => {}
                _ => {}
            }
        }
        // 2. Data ingress.
        for ev in progress_rx.try_iter() {
            app.progress.apply(ev);
        }
        app.chat.pump();
        // 3. Tick.
        if last_tick.elapsed() >= TICK {
            last_tick = Instant::now();
            ticks = ticks.wrapping_add(1);
            app.on_tick();
            if ticks.is_multiple_of(SAMPLE_EVERY) {
                app.stats.sample();
            }
            // Library scan: once, lazily, after entering the tab (fs-only).
            if !library_scanned && app.section == Section::Library {
                library_scanned = true;
                app.library = super::data::library::scan(app.args.cache_dir.as_deref());
            }
        }
        // 4. Render.
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
                let subs = match app.section {
                    Section::Main | Section::Terminal => 2,
                    _ => 0,
                };
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
