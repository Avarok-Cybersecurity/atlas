// SPDX-License-Identifier: AGPL-3.0-only

//! Main ▸ Kernels: the per-module kernel audit table.
//!
//! Split from `main_tab.rs` at the 500-LoC cap — a coherent unit on its own:
//! everything here renders the audit a model LOAD populates (`App::kernels`),
//! and nothing else in the Main tab reads it. Its tests stay in
//! `main_tab_tests.rs`, which drives the whole section through `render::draw`.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Row, Table};

use super::panel;
use crate::tui::app::App;
use crate::tui::theme;

pub(super) fn draw(f: &mut Frame, app: &App, area: Rect) {
    let Some(model) = &app.kernels else {
        let block = panel("KERNELS ─ waiting for startup ─".into(), false);
        f.render_widget(
            Paragraph::new(Span::styled(
                "  kernel audit runs at model load…",
                theme::dim(),
            ))
            .block(block),
            area,
        );
        return;
    };
    // Only the ACTIONABLE class alarms; expected-absent is declared with a reason.
    let missing = &model.missing_required;
    let mut constraints = vec![Constraint::Min(6)];
    if !missing.is_empty() {
        constraints.insert(0, Constraint::Length(missing.len().min(6) as u16 + 2));
    }
    let rows_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);
    let mut idx = 0;
    if !missing.is_empty() {
        let n = missing.len();
        let title = format!(
            "⚠ {n} UNRESOLVED ─ {} EXPECTED-ABSENT ─",
            model.missing_expected.len()
        );
        let block = panel(title, false).border_style(theme::warn());
        let lines: Vec<Line> = missing
            .iter()
            .take(6)
            .map(|m| {
                Line::from(Span::styled(
                    format!("  {}::{}  at {}", m.module, m.func, m.site),
                    theme::warn(),
                ))
            })
            .collect();
        f.render_widget(Paragraph::new(lines).block(block), rows_layout[idx]);
        idx += 1;
    }
    let filtered: Vec<_> = model
        .rows
        .iter()
        .filter(|r| app.kernel_filter.is_empty() || r.module.contains(&app.kernel_filter))
        .collect();
    // Two rows of chrome (header + border) come off the visible count; the
    // ceiling is what remains once a full screen is showing.
    let kernel_view = rows_layout[idx].height.saturating_sub(3) as usize;
    app.kernel_scroll_max
        .set(filtered.len().saturating_sub(kernel_view));
    let title = format!("KERNELS ─ {} modules ─", filtered.len());
    let header = Row::new(vec!["MODULE", "PTX-HASH", "RESOLUTION"])
        .style(theme::text2().add_modifier(Modifier::BOLD));
    let table_rows: Vec<Row> = filtered
        .iter()
        .skip(app.kernel_scroll)
        .map(|r| {
            let (res, style) = match r.resolution {
                Some(true) => ("used", theme::brand_cyan()),
                Some(false) => ("** lookup FAILED **", theme::error()),
                None => ("-", theme::dim()),
            };
            let row_style = if r.resolution.is_none() {
                theme::dim()
            } else {
                theme::text()
            };
            Row::new(vec![
                Span::styled(r.module.clone(), row_style),
                Span::styled(r.ptx_hash.clone(), theme::dim()),
                Span::styled(res, style),
            ])
        })
        .collect();
    let table = Table::new(
        table_rows,
        [
            Constraint::Length(34),
            Constraint::Length(14),
            Constraint::Min(10),
        ],
    )
    .header(header)
    .block(panel(title, true));
    f.render_widget(table, rows_layout[idx]);
}
