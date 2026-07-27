// SPDX-License-Identifier: AGPL-3.0-only

//! Library tab: master/detail over the locally cached HF models.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::panel;
use crate::tui::app::App;
use crate::tui::data::library::human_size;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);
    draw_list(f, app, cols[0]);
    draw_detail(f, app, cols[1]);
}

fn draw_list(f: &mut Frame, app: &App, area: Rect) {
    let entries = app.filtered_library();
    let search = if app.lib_filter_editing {
        format!(" search: {}▏", app.lib_filter)
    } else if !app.lib_filter.is_empty() {
        format!(" search: {}", app.lib_filter)
    } else {
        String::new()
    };
    let block = panel(
        format!("LIBRARY ─ {} models{search} ─", entries.len()),
        true,
    );
    let inner = block.inner(area);
    f.render_widget(block, area);
    let rows_visible = (inner.height / 2) as usize;
    let first = app
        .lib_selected
        .saturating_sub(rows_visible.saturating_sub(1));
    let mut lines: Vec<Line> = Vec::new();
    for (i, e) in entries.iter().enumerate().skip(first).take(rows_visible) {
        let selected = i == app.lib_selected;
        let bar = if selected {
            Span::styled("▌", theme::brand_purple())
        } else {
            Span::raw(" ")
        };
        let check = if e.has_weights {
            Span::styled("✓ ", theme::brand_green())
        } else {
            Span::styled("· ", theme::dim())
        };
        let name_style = if selected {
            theme::text().add_modifier(Modifier::BOLD)
        } else {
            theme::text()
        };
        let mut top = Line::from(vec![
            bar.clone(),
            check,
            Span::styled(e.id.clone(), name_style),
            Span::styled(format!("  {}", human_size(e.size_bytes)), theme::dim()),
        ]);
        let mut meta_spans = vec![
            bar,
            Span::raw("  "),
            Span::styled(
                format!("{} · {} · {}L", e.quant, e.model_type, e.layers),
                theme::dim(),
            ),
        ];
        if e.optimized {
            meta_spans.push(Span::styled(
                " ▐optimized▌",
                theme::brand_purple().bg(theme::BG_RAISED.color()),
            ));
        }
        let mut meta = Line::from(meta_spans);
        if selected {
            top = top.style(Style::default().bg(theme::BG_SELECTION.color()));
            meta = meta.style(Style::default().bg(theme::BG_SELECTION.color()));
        }
        lines.push(top);
        lines.push(meta);
    }
    if entries.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no cached models found",
            theme::dim(),
        )));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_detail(f: &mut Frame, app: &App, area: Rect) {
    let entries = app.filtered_library();
    let Some(e) = entries.get(app.lib_selected) else {
        f.render_widget(panel("MODEL ─".into(), false), area);
        return;
    };
    let block = panel(format!("MODEL ─ {} ─", e.id), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let kv = |k: &str, v: String| {
        Line::from(vec![
            Span::styled(format!(" {k:<14}"), theme::dim()),
            Span::styled(v, theme::text()),
        ])
    };
    let mut lines = vec![
        Line::default(),
        kv("size on disk", human_size(e.size_bytes)),
        kv("model_type", e.model_type.clone()),
        kv("quantization", e.quant.clone()),
        kv("layers", e.layers.to_string()),
        kv("hidden", e.hidden.to_string()),
        kv("heads", e.heads.to_string()),
    ];
    if e.experts > 0 {
        lines.push(kv("experts", e.experts.to_string()));
    }
    if e.context > 0 {
        lines.push(kv("context", format!("{} tok", e.context)));
    }
    lines.push(kv(
        "weights",
        if e.has_weights {
            "complete ✓".into()
        } else {
            "missing".into()
        },
    ));
    lines.push(kv(
        "kernels",
        if e.optimized {
            "optimized target compiled ✓".into()
        } else {
            "generic".into()
        },
    ));
    lines.push(Line::default());
    lines.push(Line::from(Span::styled(
        format!(" {}", e.snapshot_dir.display()),
        theme::dim(),
    )));
    f.render_widget(Paragraph::new(lines), inner);
}
