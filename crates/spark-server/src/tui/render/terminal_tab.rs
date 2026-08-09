// SPDX-License-Identifier: AGPL-3.0-only

//! Terminal tab: Ops REPL + Chat, tab-switched. `❯` purple prompt, ghost-text
//! completion, role-guttered chat with streaming cursor.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use super::{chat_lines, panel};
use crate::tui::app::{App, Focus, TermSub};
use crate::tui::{commands, theme};

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(4)])
        .split(area);
    draw_tabs(f, app, rows[0]);
    match app.term_sub {
        TermSub::Ops => draw_ops(f, app, rows[1]),
        TermSub::Chat => draw_chat(f, app, rows[1]),
    }
}

fn draw_tabs(f: &mut Frame, app: &App, area: Rect) {
    let tab = |name: &str, active: bool| {
        if active {
            Span::styled(
                format!(" {name} "),
                theme::brand_cyan().add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )
        } else {
            Span::styled(format!(" {name} "), theme::text2())
        }
    };
    let line = Line::from(vec![
        tab("Ops", app.term_sub == TermSub::Ops),
        Span::styled("─", theme::dim()),
        tab("Chat", app.term_sub == TermSub::Chat),
        Span::styled("   (6 toggles)", theme::dim()),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn draw_ops(f: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .split(area);
    // Output stream.
    let out_block = panel(format!("OPS ─ {} lines ─", app.ops.output.len()), false);
    let inner = out_block.inner(rows[0]);
    f.render_widget(out_block, rows[0]);
    let visible = inner.height as usize;
    let lines: Vec<Line> = app
        .ops
        .output
        .iter()
        .rev()
        .take(visible)
        .rev()
        .map(|l| {
            if let Some(cmd) = l.strip_prefix("❯ ") {
                Line::from(vec![
                    Span::styled("❯ ", theme::brand_purple().add_modifier(Modifier::BOLD)),
                    Span::styled(cmd.to_string(), theme::text().add_modifier(Modifier::BOLD)),
                ])
            } else {
                Line::from(Span::styled(l.clone(), theme::text2()))
            }
        })
        .collect();
    f.render_widget(Paragraph::new(lines), inner);
    // Input with ghost completion.
    let focused = app.focus == Focus::Input;
    let in_block = panel("─".into(), focused);
    let in_inner = in_block.inner(rows[1]);
    f.render_widget(in_block, rows[1]);
    let mut spans = vec![
        Span::styled("❯ ", theme::brand_purple().add_modifier(Modifier::BOLD)),
        Span::styled(app.ops.input.clone(), theme::text()),
    ];
    if focused {
        if let Some(ghost) = commands::complete(&app.ops.input) {
            let rest = &ghost[app.ops.input.len()..];
            spans.push(Span::styled(rest.to_string(), theme::dim()));
            spans.push(Span::styled("  ⇥ accept", theme::dim()));
        } else {
            spans.push(Span::styled("▏", theme::brand_cyan()));
        }
    } else {
        spans.push(Span::styled("  (Enter to focus · /help)", theme::dim()));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), in_inner);
}

/// The input pane's key hints, which differ by focus because the bare-letter
/// toggles are text while the input box owns the keyboard.
fn chat_hints(focused: bool, wide: bool) -> String {
    match (focused, wide) {
        (true, true) => {
            "─ ⏎ send · \\+⏎ newline · Ctrl+T thinking · Alt+T reasoning · Esc cancel ─".into()
        }
        (true, false) => "─ ⏎ send · Esc cancel ─".into(),
        (false, true) => "─ ⏎ focus · t thinking · T reasoning ─".into(),
        (false, false) => "─ ⏎ focus ─".into(),
    }
}

fn draw_chat(f: &mut Frame, app: &App, area: Rect) {
    let input_h = (app.chat.input.lines().count().clamp(1, 5) + 2) as u16;
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(input_h)])
        .split(area);
    // Transcript. The thinking state rides the title: it is a property of the
    // conversation, and the user must never have to guess whether the toggle
    // is doing anything. In `Auto` it reports only what the last reply was
    // OBSERVED to do — never a prediction of what this model will choose.
    let wide = rows[0].width >= 76;
    let block = panel(
        format!(
            "CHAT ─ {} ─ {} ─{}",
            super::live_model_name(app),
            app.chat.think_req.chip(app.chat.observed_thinking, wide),
            match (app.chat.streaming, app.chat.scroll) {
                (_, Some(n)) => format!(" ↑{n} ─ End follows ─"),
                (true, None) => " streaming ─".to_string(),
                (false, None) => String::new(),
            }
        ),
        false,
    );
    let inner = block.inner(rows[0]);
    f.render_widget(block, rows[0]);
    // Body width = pane minus the 2-col gutter and the 1-col rule.
    let body_w = inner.width.saturating_sub(3) as usize;
    let tip = app.chat.transcript.len().saturating_sub(1);
    let mut lines: Vec<Line> = Vec::new();
    for (i, m) in app.chat.transcript.iter().enumerate() {
        let is_tip = app.chat.streaming && i == tip;
        lines.extend(chat_lines::message_lines(
            m,
            is_tip,
            app.chat.think_view,
            app.tick,
            body_w,
        ));
        lines.push(Line::default());
    }
    // `lines` is already in DISPLAY rows (see wrap_rows), so the tail slice is
    // exact and needs no Wrap. It used to slice on unwrapped logical lines and let
    // the widget wrap afterwards: one long reply is a single logical line, so the
    // slice kept everything, the wrapped result overflowed the pane, and the
    // streaming tip rendered below the visible area — the stream "stopped
    // following" precisely when a reply got long enough to matter.
    let h = inner.height as usize;
    let max_off = lines.len().saturating_sub(h);
    // `max_off` is already exactly the ceiling: scrolled back that far, the
    // OLDEST line is at the top and there is nothing above it.
    app.chat_scroll_max.set(max_off);
    let off = match app.chat.scroll {
        None => max_off,
        Some(n) => max_off.saturating_sub(n),
    };
    let shown: Vec<Line> = lines.into_iter().skip(off).take(h).collect();
    f.render_widget(Paragraph::new(shown), inner);
    // Input.
    let focused = app.focus == Focus::Input;
    let in_block = panel(chat_hints(focused, wide), focused);
    let in_inner = in_block.inner(rows[1]);
    f.render_widget(in_block, rows[1]);
    let mut text = app.chat.input.clone();
    if focused {
        text.push('▏');
    }
    f.render_widget(
        Paragraph::new(text)
            .style(theme::text())
            .wrap(Wrap { trim: false }),
        in_inner,
    );
}

#[cfg(test)]
#[path = "terminal_tab_tests.rs"]
mod tests;
