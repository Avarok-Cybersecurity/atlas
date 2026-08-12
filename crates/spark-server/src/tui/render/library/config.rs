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
use crate::tui::lib_fields;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let Some(recipe) = app.lib.config_recipe() else {
        f.render_widget(panel("SETTINGS ─".into(), true), area);
        return;
    };
    let edited = app.lib.overrides.len() + app.lib.removed.len();
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
    // A starting point stays marked in the form, not only on the card behind
    // it: this is the screen `s` launches from, and the last honest moment to
    // say the values below are copied, not measured. `warn()` is BOLD under
    // NO_COLOR, and the sentence itself is the colour-free signal.
    if let Some(provenance) = &recipe.starting_point {
        lines.push(Line::from(Span::styled(
            format!("  starting point — {provenance}; unverified on this model"),
            theme::warn(),
        )));
    }
    // Borrowed values get their own line BESIDE the starting-point one, not
    // instead of it: "this card is synthesized" and "these values came from a
    // donor" are different claims, and a borrow must not erase or overwrite
    // the first. Same NO_COLOR contract — `warn()` is BOLD, the words carry
    // the meaning.
    if let Some(borrowed) = &app.lib.borrowed {
        lines.push(Line::from(Span::styled(
            format!("  borrowed — values from {borrowed}; not a measurement for this model"),
            theme::warn(),
        )));
    }
    lines.push(Line::from(""));

    for (i, row) in app.lib.config_rows().into_iter().enumerate() {
        let selected = i == app.lib.row;
        let editing = selected && app.lib.editing && app.lib.pending_add.is_none();
        let marker = if selected { "▌" } else { " " };
        // Row state is marked in the gutter rather than by colour alone:
        // colour is also carrying "selected" here, and two meanings on one
        // channel is one too many. `✗` removed, `+` added, `•` changed — the
        // same glyphs under NO_COLOR.
        let (change_mark, mark_style) = if row.removed {
            ("✗", theme::dim())
        } else if row.added {
            ("+", theme::brand_green())
        } else if row.changed {
            ("•", theme::brand_green())
        } else {
            (" ", theme::dim())
        };
        let value_style = if editing {
            theme::brand_cyan().add_modifier(Modifier::BOLD)
        } else if row.removed {
            theme::dim().add_modifier(Modifier::DIM)
        } else if row.changed {
            theme::brand_green()
        } else {
            theme::text()
        };
        let shown = if editing {
            format!("{}▏", app.lib.edit_buffer)
        } else if row.removed {
            // A removed flag is NOT PASSED, and the honest value column is
            // what the server does about that. "removed" is the word that
            // survives NO_COLOR; the dim styling is only reinforcement.
            match lib_fields::spec_for_key(&row.key).and_then(|s| s.default.clone()) {
                Some(d) => format!("removed — server default {d}"),
                None => "removed — flag not passed".to_string(),
            }
        } else {
            row.value.clone()
        };
        let key_style = if row.removed {
            theme::dim().add_modifier(Modifier::DIM | Modifier::CROSSED_OUT)
        } else {
            theme::text2()
        };
        let mut line = Line::from(vec![
            Span::styled(marker, theme::brand_purple()),
            Span::styled(change_mark, mark_style),
            Span::styled(format!(" {:<26}", row.key), key_style),
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
    // A setting being ADDED that has no default yet: a synthetic row at the
    // bottom, gone without trace on Esc. It renders like any edited row so
    // the flow feels like editing, not like a second kind of form.
    if let (Some(key), true) = (&app.lib.pending_add, app.lib.editing) {
        lines.push(
            Line::from(vec![
                Span::styled("▌", theme::brand_purple()),
                Span::styled("+", theme::brand_green()),
                Span::styled(format!(" {key:<26}"), theme::text2()),
                Span::styled(
                    format!("{}▏", app.lib.edit_buffer),
                    theme::brand_cyan().add_modifier(Modifier::BOLD),
                ),
            ])
            .style(theme::selected()),
        );
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
