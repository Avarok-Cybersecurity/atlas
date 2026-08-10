// SPDX-License-Identifier: AGPL-3.0-only

//! The two things drawn ON TOP of a finished frame: toasts and the key map.
//!
//! Split out of `render/mod.rs` at the 500-LoC cap. They belong together: both
//! are modal chrome that owns cells some other pane has already painted, so
//! both must `Clear` what they cover and both must decline to draw at all
//! rather than clip themselves into something unreadable.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};

use crate::tui::app::App;
use crate::tui::theme;

/// Toasts, drawn over whatever is underneath them.
///
/// They are boxed rather than being a single tinted line. A toast lands on top
/// of a dense pane — a table, a log, a progress bar — and a one-row strip with
/// only a background colour to separate it reads as part of that pane rather
/// than as a message about it. The border is in the toast's own accent colour
/// (green or red), which is also what tells you at a glance whether something
/// succeeded or failed.
pub(super) fn draw_toasts(f: &mut Frame, app: &App, content: Rect) {
    // +2 columns and +2 rows per toast for the border box.
    let width = 44.min(content.width.saturating_sub(2));
    let inner_w = width.saturating_sub(2) as usize;
    for (i, t) in app.toasts.iter().rev().take(3).enumerate() {
        // Three rows each now (box + a blank), so they stack without touching.
        let area = Rect {
            x: content.x + content.width.saturating_sub(width + 1),
            y: content.y + 1 + (i as u16) * 4,
            width,
            height: 3,
        };
        if area.bottom() > content.bottom() || width < 6 {
            // Not enough room to draw a box honestly; skip rather than clip a
            // border into something unreadable.
            continue;
        }
        let accent = if t.error {
            theme::error()
        } else {
            theme::brand_green()
        };
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(accent.add_modifier(Modifier::BOLD))
            .style(Style::default().bg(theme::BG_RAISED.color()));
        let inner = block.inner(area);
        // `Clear` over the WHOLE box, including the border cells: without it
        // the rounded corners sit on top of whatever glyph was underneath.
        f.render_widget(Clear, area);
        f.render_widget(block, area);
        let text = truncate_toast(&t.text, inner_w);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(text, theme::text())))
                .style(Style::default().bg(theme::BG_RAISED.color())),
            inner,
        );
    }
}

/// One line, ellipsised, so a long message cannot spill past the border it is
/// supposed to be inside.
pub(super) fn truncate_toast(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let n = text.chars().count();
    if n <= width {
        return text.to_string();
    }
    let head: String = text.chars().take(width.saturating_sub(1)).collect();
    format!("{head}…")
}

/// The key map, and the SSOT for how tall its modal has to be.
pub(super) const KEYS: [(&str, &str); 19] = [
    ("1-6", "jump to section (repeat cycles its subsections)"),
    (
        "Tab / Shift+Tab",
        "walk every sidebar row, subsections included",
    ),
    ("j/k ↑/↓", "move / scroll"),
    ("g / G", "top / bottom (follow)"),
    ("f", "log filter (Main)"),
    ("/", "search (Library)"),
    ("←/→ + Enter", "select node / detail (Network)"),
    ("Enter", "focus input (Terminal) / edit field (Benchmarks)"),
    ("s / c", "start / cancel the configured benchmark"),
    (
        "d",
        "Library: download / resume / update the selected model",
    ),
    ("x", "Library: stop the running download"),
    ("u", "Library: check the selected model for updates"),
    ("t / Ctrl+T", "Chat: ask for thinking — auto / off / on"),
    ("T / Alt+T", "Chat: reasoning collapsed / expanded / hidden"),
    ("Esc", "back / cancel"),
    ("Ctrl+C", "clean shutdown (drain + exit)"),
    // ★ NOT "quit TUI". `q` sets should_quit, and the loop then calls
    // shutdown::request -- it DRAINS AND STOPS THE SERVER, exactly like
    // Ctrl+C. Describing that as closing a window invites a stray keypress to
    // end a four-hour benchmark. The honest label is half the fix; the other
    // half is `App::work_in_flight`, which makes the press cost a confirmation
    // whenever there is something to lose.
    ("q", "shut down the server (drain + exit; confirms if busy)"),
    // ★ The one way to leave the dashboard WITHOUT stopping the server, and it
    // was reachable only by typing it into the Terminal tab and knowing it
    // existed. A user looking for "how do I get out of this" found `q` in this
    // list and nothing else -- so the destructive answer was the discoverable
    // one. It is a slash command rather than a key, and it is listed here
    // anyway, because this modal is where the question gets asked.
    (
        "/detach",
        "Terminal: leave the TUI, keep serving with plain logs",
    ),
    ("?", "this help"),
];

pub(super) fn draw_help(f: &mut Frame, area: Rect) {
    let w = 64.min(area.width.saturating_sub(4));
    // Sized to the LIST, not to a number someone typed once: the table has
    // grown three times since 18 was chosen, and every entry past the
    // sixteenth was silently below the modal's bottom border — a key nobody
    // could read is a key nobody has.
    let h = ((KEYS.len() + 2) as u16).min(area.height.saturating_sub(2));
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);
    let mut lines = Vec::with_capacity(KEYS.len());
    for (k, d) in KEYS {
        lines.push(Line::from(vec![
            Span::styled(format!("  {k:<16}"), theme::brand_cyan()),
            Span::styled(d.to_string(), theme::text2()),
        ]));
    }
    let block = Block::bordered()
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(theme::border(false))
        .title(Span::styled("─ KEYS ─", theme::text2()))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    f.render_widget(Paragraph::new(lines).block(block), modal);
}

/// Ask before `q` drains a server that is in the middle of something.
///
/// Deliberately modal and deliberately small: it names WHAT is in flight,
/// because "are you sure" answers nothing a user did not already know, and
/// the thing they need to weigh is whether the hours already spent matter
/// more than getting their prompt back.
pub(super) fn draw_quit_confirm(f: &mut Frame, app: &App, area: Rect) {
    let Some(what) = app.work_in_flight() else {
        return;
    };
    let lines = vec![
        Line::from(Span::styled(format!("  {what}."), theme::warn())),
        Line::from(Span::styled(
            "  Quitting drains it and stops the server.",
            theme::text(),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  q / y", theme::brand_cyan()),
            Span::styled("  quit anyway", theme::text2()),
            Span::styled("     any other key", theme::brand_cyan()),
            Span::styled("  stay", theme::text2()),
        ]),
    ];
    let w = 62.min(area.width.saturating_sub(4));
    let h = ((lines.len() + 2) as u16).min(area.height.saturating_sub(2));
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::warn())
        .title(Span::styled(
            "\u{2500} STOP THE SERVER? \u{2500}",
            theme::warn(),
        ))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    f.render_widget(Paragraph::new(lines).block(block), modal);
}
