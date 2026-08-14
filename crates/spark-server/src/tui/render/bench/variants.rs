// SPDX-License-Identifier: AGPL-3.0-only

//! The model-variant step: which checkpoint the selected benchmark runs on.
//!
//! Deliberately the same shape as the Library's recipe cards — a list of
//! choices on the left, the selected one's measured rationale on the right —
//! because it answers the same question one level up: not "which serve config
//! for this model" but "which model for this measurement". The detail pane
//! shows the variant's committed thresholds and its `note`, which is where the
//! provenance of every number lives.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::{panel, wrap};
use crate::tui::app::App;
use crate::tui::bench_variants::VariantRow;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
        .split(area);
    draw_rows(f, app, cols[0]);
    draw_detail(f, app, cols[1]);
}

fn draw_rows(f: &mut Frame, app: &App, area: Rect) {
    let name = app
        .bench
        .descriptor()
        .map(|d| d.name.to_uppercase())
        .unwrap_or_default();
    let block = panel(
        format!(
            "{name}: {} MODEL VARIANT{} ─",
            app.bench.variants.len(),
            if app.bench.variants.len() == 1 {
                ""
            } else {
                "S"
            }
        ),
        true,
    );
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Entry-based scroll, same reasoning as the suite list: each variant
    // occupies ROWS_PER_ENTRY rows, and the cursor must stay visible.
    const ROWS_PER_ENTRY: usize = 4;
    let visible = (inner.height as usize / ROWS_PER_ENTRY).max(1);
    let offset = app
        .bench
        .variant_row
        .saturating_sub(visible.saturating_sub(1));
    let mut lines: Vec<Line> = Vec::new();
    for (i, row) in app.bench.variants.iter().enumerate().skip(offset) {
        let selected = i == app.bench.variant_row;
        let marker = if selected {
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
            Span::styled(format!(" {}", row.title), name_style),
        ]);
        if selected {
            line = line.style(theme::selected());
        }
        lines.push(line);
        lines.push(Line::from(vec![
            Span::raw("   "),
            Span::styled(row.checkpoint.clone(), theme::dim()),
        ]));
        lines.push(Line::from(chips(row)));
        lines.push(Line::default());
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// The facts that distinguish sibling variants, in the Library cards' chip
/// idiom: the gate's declared subject, the box class, the serving recipe.
fn chips(row: &VariantRow) -> Vec<Span<'static>> {
    let mut out = vec![Span::raw("   ")];
    let mut chip = |text: String, style: ratatui::style::Style| {
        out.push(Span::styled(
            format!(" {text} "),
            style.add_modifier(Modifier::REVERSED | Modifier::BOLD),
        ));
        out.push(Span::raw(" "));
    };
    if row.is_default {
        // The checkpoint the gate runs when none is named — the required
        // subject, so it leads and renders in brand green.
        chip("default".into(), theme::brand_green());
    }
    chip(row.hardware.clone(), theme::dim());
    if let Some(recipe) = &row.recipe {
        let stem = recipe.rsplit('/').next().unwrap_or(recipe);
        chip(stem.to_string(), theme::brand_cyan());
    }
    out
}

fn draw_detail(f: &mut Frame, app: &App, area: Rect) {
    let Some(row) = app.bench.variants.get(app.bench.variant_row) else {
        return;
    };
    let block = panel(format!("{} ─", row.checkpoint.to_uppercase()), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let width = inner.width.saturating_sub(2) as usize;

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled(" Thresholds  ", theme::dim()),
        Span::styled(
            "what a gate record of this variant must meet",
            theme::text2(),
        ),
    ]));
    for (metric, bound) in &row.metrics {
        let mut parts = Vec::new();
        if let (Some(min), Some(max)) = (bound.min, bound.max)
            && (min - max).abs() < f64::EPSILON
        {
            parts.push(format!("exactly {min}"));
        } else {
            if let Some(min) = bound.min {
                parts.push(format!("≥ {min}"));
            }
            if let Some(max) = bound.max {
                parts.push(format!("≤ {max}"));
            }
        }
        if let Some(noise) = bound.noise {
            parts.push(format!("±{noise} noise"));
        }
        lines.push(Line::from(vec![
            Span::styled(format!("   {metric}  "), theme::text2()),
            Span::styled(parts.join(", "), theme::brand_cyan()),
        ]));
    }
    if let Some(recipe) = &row.recipe {
        lines.push(Line::from(vec![
            Span::styled(" Serves via  ", theme::dim()),
            Span::styled(recipe.clone(), theme::text2()),
        ]));
    }
    lines.push(Line::default());
    // The note is the provenance: which run defined these numbers, on which
    // box, and what is provisional about them.
    lines.extend(wrap(&row.note, width, theme::text2()));
    f.render_widget(Paragraph::new(lines), inner);
}
