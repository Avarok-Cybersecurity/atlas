// SPDX-License-Identifier: AGPL-3.0-only

//! History: past runs, read back from `~/.atlas/runs`.
//!
//! A stored run is the same `BenchmarkResult` the live pane rendered, so the
//! table and the stat tiles come from the shared helpers — a past run and a
//! present one look identical, which is what makes them comparable at a glance.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::panel;
use super::{draw_stats, draw_table, verdict_line};
use crate::tui::app::App;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    if app.bench.history.is_empty() {
        let block = panel("HISTORY ─".into(), false);
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(
            Paragraph::new(vec![
                Line::default(),
                Line::from(Span::styled("  No runs recorded yet.", theme::text2())),
                Line::from(Span::styled(
                    "  Every completed run is written to ~/.atlas/runs and appears here.",
                    theme::dim(),
                )),
            ]),
            inner,
        );
        return;
    }

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(34), Constraint::Min(30)])
        .split(area);
    draw_list(f, app, cols[0]);
    draw_detail(f, app, cols[1]);
}

fn draw_list(f: &mut Frame, app: &App, area: Rect) {
    let block = panel(format!("RUNS ─ {} ─", app.bench.history.len()), true);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let visible = inner.height as usize;
    let offset = app
        .bench
        .history_row
        .saturating_sub(visible.saturating_sub(1));
    let mut lines: Vec<Line> = Vec::new();
    for (i, entry) in app
        .bench
        .history
        .iter()
        .enumerate()
        .skip(offset)
        .take(visible)
    {
        let selected = i == app.bench.history_row;
        let mark = match entry.frame.verdict.as_ref().map(|v| v.kind) {
            Some(atlas_plugin::VerdictKind::Pass) => Span::styled("✓", theme::brand_green()),
            Some(atlas_plugin::VerdictKind::Fail) => Span::styled("✗", theme::error()),
            _ => Span::styled("·", theme::dim()),
        };
        let mut line = Line::from(vec![
            Span::styled(if selected { "▌" } else { " " }, theme::brand_purple()),
            mark,
            Span::styled(
                format!(" {:<20}", entry.benchmark_id),
                if selected {
                    theme::text().add_modifier(Modifier::BOLD)
                } else {
                    theme::text2()
                },
            ),
            Span::styled(entry.age_text(), theme::dim()),
        ]);
        if selected {
            line = line.style(theme::selected());
        }
        lines.push(line);
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_detail(f: &mut Frame, app: &App, area: Rect) {
    let Some(entry) = app.bench.history.get(app.bench.history_row) else {
        return;
    };
    let frame = &entry.frame;
    let has_stats = !frame.summary.is_empty();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(if has_stats { 3 } else { 0 }),
            Constraint::Min(6),
            Constraint::Length(2),
        ])
        .split(area);
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                format!(" {} ", entry.benchmark_id),
                theme::text().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "· {} · {:.0}s · {}",
                    frame.phase,
                    frame.elapsed.as_secs_f64(),
                    entry.age_text()
                ),
                theme::text2(),
            ),
        ])),
        rows[0],
    );
    if has_stats {
        draw_stats(f, &frame.summary, rows[1]);
    }
    if let Some(table) = &frame.table {
        draw_table(f, table, 0, rows[2]);
    }
    if let Some(verdict) = &frame.verdict {
        f.render_widget(Paragraph::new(verdict_line(verdict)), rows[3]);
    }
}
