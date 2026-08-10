// SPDX-License-Identifier: AGPL-3.0-only

//! Benchmarks section: Suite (list → parameters → live run) and History.
//!
//! One file per step; this one dispatches and owns the pieces they share —
//! the plugin-provenance block and the results table, which the live pane and
//! the History pane render identically because a stored frame and a live one
//! are the same type.

mod history;
mod list;
mod params;
mod run;

use ratatui::Frame;
use ratatui::layout::{Constraint, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Paragraph, Row, Table};

use super::panel;
use crate::tui::app::{App, BenchSub};
use crate::tui::bench_state::View;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    match app.bench_sub {
        BenchSub::History => history::draw(f, app, area),
        BenchSub::Suite => match app.bench.view {
            View::List => list::draw(f, app, area),
            View::Params => params::draw(f, app, area),
            View::Run => run::draw(f, app, area),
        },
    }
}

/// The `OFFICIAL` / `COMMUNITY` badge. A trust signal, so first-party is the
/// only thing that renders in brand green.
pub(super) fn origin_badge(meta: &atlas_plugin::PluginMetadata) -> Span<'static> {
    if meta.official {
        Span::styled(
            " OFFICIAL ",
            theme::brand_green().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        )
    } else {
        Span::styled(
            " COMMUNITY ",
            theme::warn().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        )
    }
}

/// Authorship and support links, as label/value rows.
pub(super) fn metadata_lines(meta: &atlas_plugin::PluginMetadata) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(vec![
        origin_badge(meta),
        Span::styled(format!("  v{}", meta.version), theme::text2()),
    ])];
    for (label, value) in meta.rows() {
        lines.push(Line::from(vec![
            Span::styled(format!(" {label:<13}"), theme::dim()),
            Span::styled(value.to_string(), theme::text2()),
        ]));
    }
    lines
}

/// The verdict banner. `Info` is deliberately not green: a benchmark that
/// measured without gating has not passed anything.
pub(super) fn verdict_line(verdict: &atlas_plugin::Verdict) -> Line<'static> {
    use atlas_plugin::VerdictKind as K;
    let (label, style) = match verdict.kind {
        K::Pass => (" PASS ", theme::brand_green()),
        K::Fail => (" FAIL ", theme::error()),
        K::Info => (" INFO ", theme::brand_cyan()),
    };
    Line::from(vec![
        Span::styled(
            label,
            style.add_modifier(Modifier::REVERSED | Modifier::BOLD),
        ),
        Span::styled(format!("  {}", verdict.reason), theme::text()),
    ])
}

/// The stat tile row above a results table.
pub(super) fn draw_stats(f: &mut Frame, stats: &[atlas_plugin::Stat], area: Rect) {
    if stats.is_empty() {
        return;
    }
    let widths: Vec<Constraint> =
        std::iter::repeat_n(Constraint::Ratio(1, stats.len() as u32), stats.len()).collect();
    let cols = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Horizontal)
        .constraints(widths)
        .split(area);
    for (stat, cell) in stats.iter().zip(cols.iter()) {
        let block = panel(format!("{} ─", stat.label.to_uppercase()), false);
        let inner = block.inner(*cell);
        f.render_widget(block, *cell);
        let line = Line::from(vec![
            Span::styled(
                format!(" {}", stat.value),
                theme::cell_style(stat.style).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!(" {}", stat.unit), theme::text2()),
        ]);
        f.render_widget(Paragraph::new(line), inner);
    }
}

/// A benchmark's results table. Shared by the live run and History, since a
/// stored frame is the same `BenchmarkResult` the run emitted.
pub(super) fn draw_table(
    f: &mut Frame,
    table: &atlas_plugin::ResultTable,
    scroll: usize,
    area: Rect,
) {
    let block = panel(
        format!("{} ─ {} rows ─", table.title, table.rows.len()),
        false,
    );
    let inner = block.inner(area);
    f.render_widget(block, area);
    let header = Row::new(
        table
            .columns
            .iter()
            .map(|c| Cell::from(c.title.clone()).style(theme::dim()))
            .collect::<Vec<_>>(),
    );
    // Rows scroll but the header does not; a 40-cell sweep is unreadable
    // without the column titles staying put.
    let visible = inner.height.saturating_sub(1) as usize;
    let max_scroll = table.rows.len().saturating_sub(visible);
    let rows: Vec<Row> = table
        .rows
        .iter()
        .skip(scroll.min(max_scroll))
        .take(visible)
        .map(|cells| {
            Row::new(
                cells
                    .iter()
                    .map(|c| Cell::from(c.text.clone()).style(theme::cell_style(c.style)))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let widths: Vec<Constraint> = table
        .columns
        .iter()
        .map(|c| Constraint::Length(c.width))
        .collect();
    f.render_widget(
        Table::new(rows, widths)
            .header(header)
            .column_spacing(1)
            .style(theme::text()),
        inner,
    );
}

#[cfg(test)]
#[path = "bench_tests.rs"]
mod tests;
