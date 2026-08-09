// SPDX-License-Identifier: AGPL-3.0-only

//! One `TestBackend` harness for every render test in this tree.
//!
//! Shared rather than redefined per file because the assertions are all of the
//! form "some row contains this text": two harnesses that trim differently
//! disagree about where a row ends, and the disagreement only surfaces as a
//! test that passes on a bug.

use ratatui::Terminal;
use ratatui::backend::TestBackend;

use crate::tui::app::App;

/// The rendered frame, one `String` per row, trailing blanks trimmed.
pub(super) fn screen(app: &App, w: u16, h: u16) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("backend");
    terminal
        .draw(|f| crate::tui::render::draw(f, app))
        .expect("draw");
    let buf = terminal.backend().buffer();
    (0..h)
        .map(|y| {
            (0..w)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}

pub(super) fn has(rows: &[String], needle: &str) -> bool {
    rows.iter().any(|r| r.contains(needle))
}

/// Render one widget-drawing closure into a buffer of its own.
///
/// For the helpers that take a `Rect` rather than an `App` — a results table,
/// a stat tile row — which have no section of their own to be reached through.
pub(super) fn draw_into(
    w: u16,
    h: u16,
    f: impl FnOnce(&mut ratatui::Frame, ratatui::layout::Rect),
) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("backend");
    terminal
        .draw(|frame| {
            let area = frame.area();
            f(frame, area);
        })
        .expect("draw");
    let buf = terminal.backend().buffer();
    (0..h)
        .map(|y| {
            (0..w)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}
