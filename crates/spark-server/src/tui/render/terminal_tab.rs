// SPDX-License-Identifier: AGPL-3.0-only

//! Terminal tab: Ops REPL + Chat, tab-switched. `❯` purple prompt, ghost-text
//! completion, role-guttered chat with streaming cursor.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

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
            if app.chat.streaming {
                " streaming ─"
            } else {
                ""
            }
        ),
        false,
    );
    let inner = block.inner(rows[0]);
    f.render_widget(block, rows[0]);
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
        for (i, text_line) in m.text.split('\n').enumerate() {
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
    let skip = lines.len().saturating_sub(inner.height as usize);
    let shown: Vec<Line> = lines.into_iter().skip(skip).collect();
    f.render_widget(Paragraph::new(shown).wrap(Wrap { trim: false }), inner);
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
