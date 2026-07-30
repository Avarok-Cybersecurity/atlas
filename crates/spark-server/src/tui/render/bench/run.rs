// SPDX-License-Identifier: AGPL-3.0-only

//! The live run pane: phase, progress, stat tiles, results table, log.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::{gradient_bar, panel};
use super::{draw_stats, draw_table, verdict_line};
use crate::tui::app::App;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let has_table = app
        .bench
        .frame
        .as_ref()
        .is_some_and(|fr| fr.table.is_some());
    let has_stats = app
        .bench
        .frame
        .as_ref()
        .is_some_and(|fr| !fr.summary.is_empty());
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),                             // header + progress
            Constraint::Length(if has_stats { 3 } else { 0 }), // stat tiles
            Constraint::Min(if has_table { 8 } else { 0 }),    // table
            Constraint::Length(2),                             // verdict
            Constraint::Length(8),                             // log
        ])
        .split(area);

    draw_header(f, app, rows[0]);
    if let Some(frame) = &app.bench.frame {
        if has_stats {
            draw_stats(f, &frame.summary, rows[1]);
        }
        if let Some(table) = &frame.table {
            draw_table(f, table, app.bench.table_scroll, rows[2]);
        }
        if let Some(verdict) = &frame.verdict {
            f.render_widget(Paragraph::new(verdict_line(verdict)), rows[3]);
        }
    }
    draw_log(f, app, rows[4]);
}

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let name = app.bench.descriptor().map(|d| d.name).unwrap_or("");
    let running = app.bench.is_running();
    let spinner = if running {
        theme::SPINNER[(app.tick as usize / 2) % theme::SPINNER.len()]
    } else {
        "●"
    };
    let spinner_style = if running {
        theme::brand_cyan()
    } else {
        theme::brand_green()
    };
    let head = Line::from(vec![
        Span::styled(format!(" {spinner} "), spinner_style),
        Span::styled(name.to_string(), theme::text().add_modifier(Modifier::BOLD)),
        Span::styled(format!("  {}", app.bench.status), theme::text2()),
        Span::styled(format!("   {}  ", app.bench.elapsed_text()), theme::dim()),
        Span::styled(app.bench.target.base_url.clone(), theme::dim()),
    ]);
    f.render_widget(Paragraph::new(head), Rect { height: 1, ..area });

    // A benchmark that cannot know its total (provisioning, scoring) reports
    // no progress; showing a full bar there would be a lie, so it stays a
    // caption.
    let bar_area = Rect {
        y: area.y + 1,
        height: 1,
        x: area.x + 1,
        width: area.width.saturating_sub(2),
    };
    match app.bench.progress {
        Some((done, total)) if total > 0 => {
            let frac = done as f64 / total as f64;
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(10), Constraint::Length(14)])
                .split(bar_area);
            f.render_widget(Paragraph::new(gradient_bar(frac, cols[0].width)), cols[0]);
            f.render_widget(
                Paragraph::new(Span::styled(
                    format!(" {done}/{total}  {:.0}%", frac * 100.0),
                    theme::text2(),
                )),
                cols[1],
            );
        }
        _ => f.render_widget(
            Paragraph::new(Span::styled(
                if running { "working…" } else { "idle" },
                theme::dim(),
            )),
            bar_area,
        ),
    }
    f.render_widget(
        Paragraph::new(Span::styled(
            if running {
                " c cancel · j/k scroll table · Esc back to suite"
            } else {
                " Esc back to suite · j/k scroll table"
            },
            theme::dim(),
        )),
        Rect {
            y: area.y + 2,
            height: 1,
            ..area
        },
    );
}

fn draw_log(f: &mut Frame, app: &App, area: Rect) {
    let block = panel(format!("LOG ─ {} lines ─", app.bench.log.len()), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let visible = inner.height as usize;
    let lines: Vec<Line> = app
        .bench
        .log
        .iter()
        .rev()
        .take(visible)
        .rev()
        .map(|line| {
            use atlas_plugin::LogLevel as L;
            let style = match line.level {
                L::Error => theme::error(),
                L::Warn => theme::warn(),
                L::Info => theme::text2(),
                L::Debug => theme::dim(),
            };
            Line::from(Span::styled(format!(" {}", line.text), style))
        })
        .collect();
    f.render_widget(Paragraph::new(lines), inner);
}
