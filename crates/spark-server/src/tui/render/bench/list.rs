// SPDX-License-Identifier: AGPL-3.0-only

//! The suite list and its detail pane.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::panel;
use super::super::wrap;
use super::metadata_lines;
use crate::tui::app::App;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(area);
    draw_list(f, app, cols[0]);
    draw_detail(f, app, cols[1]);
}

fn draw_list(f: &mut Frame, app: &App, area: Rect) {
    let block = panel("SUITE ─".into(), true);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // ★ Scroll offset, in ENTRIES rather than lines: each benchmark occupies
    // ROWS_PER_ENTRY rows, so an 80x24 terminal shows about four of them.
    // Without this the list always rendered from index 0 and `j` past the
    // fourth moved a cursor nobody could see -- the detail pane changing was
    // the only clue. Both sibling lists (bench/history.rs, library/list.rs)
    // already compute one.
    const ROWS_PER_ENTRY: usize = 4;
    let visible = (inner.height as usize / ROWS_PER_ENTRY).max(1);
    let offset = app.bench.selected.saturating_sub(visible.saturating_sub(1));
    let mut lines: Vec<Line> = Vec::new();
    for (i, descriptor) in atlas_plugin::registry::all()
        .iter()
        .enumerate()
        .skip(offset)
    {
        let selected = i == app.bench.selected;
        let running = app.bench.running_id == Some(descriptor.id) && app.bench.is_running();
        let marker = if running {
            Span::styled(
                theme::SPINNER[(app.tick as usize / 2) % theme::SPINNER.len()],
                theme::brand_cyan(),
            )
        } else if selected {
            Span::styled("▌", theme::brand_purple())
        } else {
            Span::raw(" ")
        };
        let name_style = if selected {
            theme::text().add_modifier(Modifier::BOLD)
        } else {
            theme::text2()
        };
        let mut line = Line::from(vec![
            marker,
            Span::styled(format!(" {}", descriptor.name), name_style),
        ]);
        if selected {
            line = line.style(theme::selected());
        }
        lines.push(line);
        lines.push(Line::from(vec![
            Span::raw("   "),
            Span::styled(descriptor.summary.to_string(), theme::dim()),
        ]));
        lines.push(Line::from(vec![
            Span::raw("   "),
            Span::styled(descriptor.duration_hint.to_string(), theme::dim()),
            // The only place the shell-execution warning is visible before you
            // open the benchmark, so it goes in the list itself.
            if descriptor.needs_confirmation {
                Span::styled("  ⚠ runs shell", theme::warn())
            } else {
                Span::raw("")
            },
        ]));
        lines.push(Line::default());
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_detail(f: &mut Frame, app: &App, area: Rect) {
    let Some(descriptor) = app.bench.descriptor() else {
        return;
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(11)])
        .split(area);

    let block = panel(format!("{} ─", descriptor.name.to_uppercase()), false);
    let inner = block.inner(rows[0]);
    f.render_widget(block, rows[0]);
    let width = inner.width.saturating_sub(2) as usize;
    let mut lines = wrap(descriptor.detail, width, theme::text2());
    lines.push(Line::default());
    lines.push(Line::from(vec![
        Span::styled(" Parameters  ", theme::dim()),
        Span::styled(
            format!("{} editable", app.bench.specs.len()),
            theme::text2(),
        ),
    ]));
    // When the MEASUREMENT last changed — the fact that decides whether two
    // runs are comparable. Same question the Library answers for a recipe.
    lines.push(Line::from(vec![
        Span::styled(" Updated     ", theme::dim()),
        Span::styled(descriptor.updated.to_string(), theme::text2()),
    ]));
    lines.push(Line::from(vec![
        Span::styled(" Target      ", theme::dim()),
        Span::styled(app.bench.target.base_url.clone(), theme::brand_cyan()),
        Span::styled(format!("  {}", app.bench.target.model), theme::text2()),
    ]));
    lines.push(Line::default());
    lines.push(Line::from(Span::styled(
        " ⏎ configure and start",
        theme::brand_cyan(),
    )));
    f.render_widget(Paragraph::new(lines), inner);

    // Provenance: who wrote this and where to report it.
    let meta_block = panel("PLUGIN ─".into(), false);
    let meta_inner = meta_block.inner(rows[1]);
    f.render_widget(meta_block, rows[1]);
    f.render_widget(
        Paragraph::new(metadata_lines(app.bench.plugin_metadata())),
        meta_inner,
    );
}
