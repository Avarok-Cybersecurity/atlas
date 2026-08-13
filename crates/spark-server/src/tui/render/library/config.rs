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

    // Head, body, tail — windowed separately. The head (identity and
    // provenance) stays pinned: it is the last honest moment to say the
    // values below are copied, not measured. The COMMAND preview stays at
    // the bottom: it is the form's output. Only the row region scrolls.
    let mut head: Vec<Line> = Vec::new();
    head.push(Line::from(vec![
        Span::styled("  model  ", theme::dim()),
        Span::styled(recipe.model.clone(), theme::text()),
    ]));
    // A starting point stays marked in the form, not only on the card behind
    // it: this is the screen `s` launches from, and the last honest moment to
    // say the values below are copied, not measured. `warn()` is BOLD under
    // NO_COLOR, and the sentence itself is the colour-free signal.
    if let Some(provenance) = &recipe.starting_point {
        head.push(Line::from(Span::styled(
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
        head.push(Line::from(Span::styled(
            format!("  borrowed — values from {borrowed}; not a measurement for this model"),
            theme::warn(),
        )));
    }
    head.push(Line::from(""));

    // The row region, with the line index just past the selected row (and
    // its attached error) recorded as the scroll anchor.
    let mut body: Vec<Line> = Vec::new();
    let mut anchor_end = 0usize;
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
        body.push(line);

        // The error attaches to the row that caused it, like the benchmark form.
        if let Some(err) = app.lib.error.as_ref().filter(|_| selected && !editing) {
            body.extend(wrap(&format!("  {err}"), width, theme::error()));
        }
        if selected {
            anchor_end = body.len();
        }
    }
    // A setting being ADDED that has no default yet: a synthetic row at the
    // bottom, gone without trace on Esc. It renders like any edited row so
    // the flow feels like editing, not like a second kind of form.
    if let (Some(key), true) = (&app.lib.pending_add, app.lib.editing) {
        body.push(
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
        // The synthetic row is where the cursor effectively is.
        anchor_end = body.len();
    }

    let mut tail: Vec<Line> = vec![Line::from("")];
    match app.lib.preview_argv() {
        Some(argv) => {
            tail.push(Line::from(Span::styled(" COMMAND", theme::dim())));
            // Show what would actually run. A form that hides its output makes
            // the user guess whether an edit took effect.
            let rendered = argv.iter().skip(1).cloned().collect::<Vec<_>>().join(" ");
            tail.extend(wrap(&format!("spark {rendered}"), width, theme::text2()));
        }
        None => tail.push(Line::from(Span::styled(
            " this recipe cannot be launched from here",
            theme::warn(),
        ))),
    }

    // Cursor-follows-scroll on the row region — the same
    // `selected.saturating_sub(visible - 1)` idiom the list and cards use.
    // Without it, a recipe with ~16 defaults on a 24-row terminal clipped
    // its tail, and `select_row` after an add deliberately lands the cursor
    // at the END of the form — exactly where the clipping was: `j` kept
    // moving onto rows that were not on screen.
    let body_h = (inner.height as usize)
        .saturating_sub(head.len() + tail.len())
        .max(1);
    let off = anchor_end.saturating_sub(body_h);
    let mut lines = head;
    lines.extend(body.into_iter().skip(off).take(body_h));
    lines.extend(tail);
    f.render_widget(Paragraph::new(lines), inner);
}
