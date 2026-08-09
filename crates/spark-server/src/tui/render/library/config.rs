// SPDX-License-Identifier: AGPL-3.0-only

//! One recipe's settings, editable before launch.
//!
//! Deliberately the same shape as the benchmark parameter form — same purple
//! selection bar, same `⏎ edit` / `Esc cancel` contract, same one-help-line-at-
//! a-time rule. Two forms in one dashboard that behaved differently would make
//! both harder to learn.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::{panel, wrap};
use crate::tui::app::App;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let Some(recipe) = app.lib.config_recipe() else {
        f.render_widget(panel("SETTINGS ─".into(), true), area);
        return;
    };
    let edited = app.lib.overrides.len();
    let title = if edited == 0 {
        format!("{} ─ SETTINGS ─", recipe.id.to_uppercase())
    } else {
        format!(
            "{} ─ SETTINGS ─ {edited} changed ─",
            recipe.id.to_uppercase()
        )
    };
    let block = panel(title, true);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let width = inner.width.saturating_sub(4) as usize;

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled("  model  ", theme::dim()),
        Span::styled(recipe.model.clone(), theme::text()),
    ]));
    lines.push(Line::from(""));

    for (i, (key, value, changed)) in app.lib.config_rows().into_iter().enumerate() {
        let selected = i == app.lib.row;
        let editing = selected && app.lib.editing;
        let marker = if selected { "▌" } else { " " };
        // A changed row is marked in the gutter rather than by colour alone:
        // colour is also carrying "selected" here, and two meanings on one
        // channel is one too many.
        let change_mark = if changed { "•" } else { " " };
        let value_style = if editing {
            theme::brand_cyan().add_modifier(Modifier::BOLD)
        } else if changed {
            theme::brand_green()
        } else {
            theme::text()
        };
        let shown = if editing {
            format!("{}▏", app.lib.edit_buffer)
        } else {
            value.clone()
        };
        let mut line = Line::from(vec![
            Span::styled(marker, theme::brand_purple()),
            Span::styled(change_mark, theme::brand_green()),
            Span::styled(format!(" {key:<26}"), theme::text2()),
            Span::styled(shown, value_style),
        ]);
        if selected {
            line = line.style(theme::selected());
        }
        lines.push(line);

        // The error attaches to the row that caused it, like the benchmark form.
        if let Some(err) = app.lib.error.as_ref().filter(|_| selected && !editing) {
            lines.extend(wrap(&format!("  {err}"), width, theme::error()));
        }
    }

    lines.push(Line::from(""));
    match app.lib.preview_argv() {
        Some(argv) => {
            lines.push(Line::from(Span::styled(" COMMAND", theme::dim())));
            // Show what would actually run. A form that hides its output makes
            // the user guess whether an edit took effect.
            let rendered = argv.iter().skip(1).cloned().collect::<Vec<_>>().join(" ");
            lines.extend(wrap(&format!("spark {rendered}"), width, theme::text2()));
        }
        None => lines.push(Line::from(Span::styled(
            " this recipe cannot be launched from here",
            theme::warn(),
        ))),
    }

    f.render_widget(Paragraph::new(lines), inner);
}
