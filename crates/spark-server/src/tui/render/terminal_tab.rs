// SPDX-License-Identifier: AGPL-3.0-only

//! Terminal tab: Ops REPL + Chat, tab-switched. `❯` purple prompt, ghost-text
//! completion, role-guttered chat with streaming cursor.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::panel;
use crate::tui::app::{App, Focus, TermSub};
use crate::tui::chat::Role;
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
        Span::styled("   (5 toggles)", theme::dim()),
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

/// Word-wrap `text` into display rows of at most `width` columns.
///
/// The chat pane slices its viewport on these rows, so they must be what actually
/// renders — measured in display columns via `unicode-width`, not `str::len`, or
/// CJK and emoji replies would compute a tail that is short by a row per line.
/// A word longer than the pane is hard-split rather than allowed to overhang.
fn wrap_rows(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    let mut rows = Vec::new();
    for logical in text.split('\n') {
        let (mut cur, mut cur_w) = (String::new(), 0usize);
        for word in logical.split_inclusive(' ') {
            let w = UnicodeWidthStr::width(word);
            if cur_w + w > width && !cur.is_empty() {
                rows.push(std::mem::take(&mut cur));
                cur_w = 0;
            }
            if w > width {
                for ch in word.chars() {
                    let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
                    if cur_w + cw > width {
                        rows.push(std::mem::take(&mut cur));
                        cur_w = 0;
                    }
                    cur.push(ch);
                    cur_w += cw;
                }
            } else {
                cur.push_str(word);
                cur_w += w;
            }
        }
        rows.push(cur);
    }
    rows
}

fn draw_chat(f: &mut Frame, app: &App, area: Rect) {
    let input_h = (app.chat.input.lines().count().clamp(1, 5) + 2) as u16;
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(input_h)])
        .split(area);
    // Transcript.
    let block = panel(
        format!(
            "CHAT ─ {} ─{}",
            app.args
                .model_name
                .clone()
                .or_else(|| app.args.model.clone())
                .unwrap_or_default(),
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
    // Body width = pane minus the 2-col gutter and the 1-col model rule.
    let body_w = inner.width.saturating_sub(3) as usize;
    let mut lines: Vec<Line> = Vec::new();
    for m in &app.chat.transcript {
        let (gutter, gstyle) = match m.role {
            Role::User => ("❯ ", theme::brand_purple().add_modifier(Modifier::BOLD)),
            Role::Model => ("⬢ ", theme::brand_cyan()),
        };
        let body_style = match m.role {
            Role::User => theme::text(),
            Role::Model => theme::text().bg(theme::BG_PANEL.color()),
        };
        for (i, text_line) in wrap_rows(&m.text, body_w).iter().enumerate() {
            let g = if i == 0 {
                Span::styled(gutter, gstyle)
            } else {
                Span::styled("  ", Style::default())
            };
            let rule = if m.role == Role::Model {
                Span::styled("▏", theme::brand_cyan())
            } else {
                Span::raw("")
            };
            lines.push(Line::from(vec![
                g,
                rule,
                Span::styled(text_line.to_string(), body_style),
            ]));
        }
        // Streaming cursor at the tip of the live message.
        if m.role == Role::Model
            && app.chat.streaming
            && std::ptr::eq(m, app.chat.transcript.last().unwrap())
            && let Some(last) = lines.last_mut()
        {
            last.spans.push(Span::styled("▍", theme::brand_cyan()));
        }
        // Footer for completed model replies.
        if m.role == Role::Model && (m.ttft_ms.is_some() || m.tok_per_s.is_some()) {
            let footer = format!(
                "  ttft {} · {} · {} tok",
                m.ttft_ms
                    .map(|v| format!("{v:.0} ms"))
                    .unwrap_or_else(|| "—".into()),
                m.tok_per_s
                    .map(|v| format!("{v:.0} tok/s"))
                    .unwrap_or_else(|| "—".into()),
                m.tokens
            );
            lines.push(Line::from(Span::styled(footer, theme::dim())));
        }
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
    let off = match app.chat.scroll {
        None => max_off,
        Some(n) => max_off.saturating_sub(n),
    };
    let shown: Vec<Line> = lines.into_iter().skip(off).take(h).collect();
    f.render_widget(Paragraph::new(shown), inner);
    // Input.
    let focused = app.focus == Focus::Input;
    let in_block = panel(
        if focused {
            "─ Enter send · \\+Enter newline · Esc cancel ─".into()
        } else {
            "─ Enter to focus ─".into()
        },
        focused,
    );
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
