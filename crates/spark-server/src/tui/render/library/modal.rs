// SPDX-License-Identifier: AGPL-3.0-only

//! The Config form's pickers: the option list and the add-a-setting list.
//!
//! Grammar borrowed from the overlay confirmations (`draw_quit_confirm` and
//! kin): rounded border, panel background, `Clear` under the whole box, and a
//! decline-to-draw rule instead of clipping. What differs is the body — these
//! are selection LISTS, so they carry the form's own `▌` cursor bar and they
//! scroll, because the KV dtype set alone is sixteen rows and will not always
//! fit. Selection is purple with a bar GLYPH, and the current value carries a
//! `✓`, so both survive NO_COLOR where every hue flattens to Reset.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};

use crate::tui::app::App;
use crate::tui::lib_modal::ConfigModal;
use crate::tui::theme;

pub(super) fn draw(f: &mut Frame, app: &App, area: Rect) {
    let Some(modal) = &app.lib.modal else {
        return;
    };
    match modal {
        ConfigModal::Options {
            key,
            options,
            selected,
        } => {
            // The current effective value gets the ✓, not the cursor: the
            // cursor is where you are, the mark is what is set.
            let current = app
                .lib
                .config_rows()
                .into_iter()
                .find(|r| r.key == *key)
                .map(|r| r.value);
            let rows: Vec<(String, bool)> = options
                .iter()
                .map(|o| (o.clone(), current.as_deref() == Some(o)))
                .collect();
            let flag = key.replace('_', "-").to_uppercase();
            draw_list(f, area, &format!("─ {flag} ─"), &rows, *selected, 40);
        }
        ConfigModal::Add {
            fields,
            selected,
            help_scroll,
        } => {
            let rows: Vec<(String, bool)> = fields
                .iter()
                .map(|s| {
                    // Key, then the help line — the same two things `--help`
                    // leads with, because that is where the user will meet
                    // this flag again.
                    (format!("{:<26} {}", s.key, s.help), false)
                })
                .collect();
            draw_add(
                f,
                area,
                &rows,
                *selected,
                fields.get(*selected),
                *help_scroll,
            );
        }
        ConfigModal::Borrow { donors, selected } => {
            // The donor's MEASURED model beside its id, on every row: the
            // whole point of this picker is that these values belong to
            // another checkpoint, and the row is where that is decided.
            let rows: Vec<(String, bool)> = donors
                .iter()
                .map(|d| (format!("{:<40} measured on {}", d.id, d.model), false))
                .collect();
            // Wider than the other pickers: a row is a recipe id AND an HF
            // model id, and clipping the model is clipping the provenance.
            draw_list(f, area, "─ BORROW PARAMETERS FROM ─", &rows, *selected, 100);
        }
        ConfigModal::Preview {
            donors,
            donor,
            changes,
            scroll,
        } => {
            if let Some(d) = donors.get(*donor) {
                draw_preview(f, area, d, changes, *scroll);
            }
        }
    }
}

/// The borrow preview: exactly the rows applying the donor would change,
/// current value beside incoming. This box is the "seen it coming" step — a
/// borrow commits nothing until Enter is pressed HERE.
fn draw_preview(
    f: &mut Frame,
    area: Rect,
    donor: &crate::recipe::Recipe,
    changes: &[crate::tui::lib_borrow::BorrowChange],
    scroll: usize,
) {
    let w = 76.min(area.width.saturating_sub(4));
    // 2 border rows + 2 header rows + the change rows; cap and scroll.
    let h = ((changes.len() + 4) as u16).min(area.height.saturating_sub(2));
    if w < 10 || h < 5 {
        // Same rule as `draw_list`: a box that cannot hold its border and
        // header is not drawn at all rather than drawn wrong.
        return;
    }
    let visible = (h - 4) as usize;
    let top = scroll.min(changes.len().saturating_sub(visible));
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);

    let inner_w = w.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = Vec::with_capacity(visible + 2);
    // The header is the honesty: these values were measured on the donor's
    // model, and applying them here does not carry the measurement over.
    // `warn()` is BOLD under NO_COLOR; the sentence is the colour-free signal.
    lines.push(Line::from(Span::styled(
        clip(
            &format!("copied from {} — not measured on this model", donor.model),
            inner_w,
        ),
        theme::warn(),
    )));
    lines.push(Line::from(Span::styled(
        clip(
            &format!(
                "{} changes; settings the donor does not name keep their values",
                changes.len()
            ),
            inner_w,
        ),
        theme::dim(),
    )));
    for change in changes.iter().skip(top).take(visible) {
        // `from → to`, one glyph carrying the direction: the arrow (with the
        // column order) is what survives NO_COLOR, the hues only reinforce.
        let budget = inner_w.saturating_sub(28 + change.to.len() + 3);
        lines.push(Line::from(vec![
            Span::styled(format!(" {:<26} ", clip(&change.key, 26)), theme::text2()),
            Span::styled(clip(&change.from, budget.max(4)), theme::dim()),
            Span::styled(" → ", theme::dim()),
            Span::styled(change.to.clone(), theme::brand_green()),
        ]));
    }

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::border(true))
        .title(Span::styled(
            format!("─ BORROW: {} ─", donor.id.to_uppercase()),
            theme::title(true),
        ))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    let block = if changes.len() > visible {
        block.title_bottom(Span::styled(
            format!("─ {}/{} ─", top + 1, changes.len()),
            theme::dim(),
        ))
    } else {
        block
    };
    f.render_widget(Paragraph::new(lines).block(block), modal);
}

/// The add-picker: the flag list, and — when the terminal can afford it — a
/// side panel carrying the highlighted flag's FULL help, wrapped and
/// scrollable. The list row shows only clap's first line, and several of
/// these doc comments run to paragraphs; the panel is where the rest is
/// readable without leaving the picker.
fn draw_add(
    f: &mut Frame,
    area: Rect,
    rows: &[(String, bool)],
    selected: usize,
    spec: Option<&&'static crate::tui::lib_fields::FieldSpec>,
    help_scroll: usize,
) {
    // Panel text width + 2 border columns + 2 padding columns. Derived from
    // the same constant the key handler clamps against, so the two cannot
    // wrap at different widths.
    const PANEL_W: u16 = (crate::tui::lib_modal::HELP_PANEL_TEXT_W + 4) as u16;
    // Below ~50 list columns the key column alone eats the row; squeezing a
    // panel in beside it would leave two unreadable slivers. The fallback is
    // stated, not emergent: the panel is DROPPED whole and the picker is the
    // single list it always was, first help line on the row, ellipsised.
    let avail = area.width.saturating_sub(4);
    let (Some(spec), true) = (spec, avail >= 50 + PANEL_W) else {
        draw_list(f, area, "─ ADD A SETTING ─", rows, selected, 76);
        return;
    };
    let list_w = (avail - PANEL_W).min(76);
    let w = list_w + PANEL_W;
    let h = ((rows.len() + 2) as u16).min(area.height.saturating_sub(2));
    if h < 3 {
        return;
    }
    let x = area.x + (area.width - w) / 2;
    let y = area.y + (area.height - h) / 2;
    draw_list_at(
        f,
        Rect {
            x,
            y,
            width: list_w,
            height: h,
        },
        "─ ADD A SETTING ─",
        rows,
        selected,
    );
    draw_help_panel(
        f,
        Rect {
            x: x + list_w,
            y,
            width: PANEL_W,
            height: h,
        },
        spec,
        help_scroll,
    );
}

/// The full clap help for one flag, wrapped to the panel and scrolled to
/// `help_scroll` (already clamped by the key handler; re-clamped here because
/// the panel's height is only known now).
fn draw_help_panel(
    f: &mut Frame,
    panel: Rect,
    spec: &crate::tui::lib_fields::FieldSpec,
    help_scroll: usize,
) {
    let wrapped =
        crate::tui::format::wrap_help(&spec.help_full, crate::tui::lib_modal::HELP_PANEL_TEXT_W);
    let visible = panel.height.saturating_sub(2) as usize;
    let top = help_scroll.min(wrapped.len().saturating_sub(visible));
    let mut lines: Vec<Line> = wrapped
        .iter()
        .skip(top)
        .take(visible)
        .map(|l| Line::from(Span::styled(format!(" {l}"), theme::text2())))
        .collect();
    if lines.is_empty() {
        // An empty box reads as a rendering bug; the absence is the content.
        lines.push(Line::from(Span::styled(" no help text", theme::dim())));
    }
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::border(true))
        .title(Span::styled(
            clip(
                &format!("─ {} ─", spec.key.to_uppercase()),
                panel.width.saturating_sub(2) as usize,
            ),
            theme::title(true),
        ))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    // The binding lives in the border, beside the position it moves: the
    // footer names it too, but the panel is where the reader's eye is when
    // the question "how do I see the rest" arises. Only when there IS a rest.
    let block = if wrapped.len() > visible {
        block.title_bottom(Span::styled(
            format!("─ J/K {}/{} ─", top + 1, wrapped.len()),
            theme::dim(),
        ))
    } else {
        block
    };
    f.render_widget(Clear, panel);
    f.render_widget(Paragraph::new(lines).block(block), panel);
}

/// Truncate to `width` columns with an ellipsis that says so — a silently
/// clipped value reads as the whole value, which in THIS box means agreeing
/// to a setting the user never saw the end of.
fn clip(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        return s.to_string();
    }
    let mut out: String = s.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// One scrolling selection list, centred in `area`. `rows` are
/// (label, is_current).
fn draw_list(
    f: &mut Frame,
    area: Rect,
    title: &str,
    rows: &[(String, bool)],
    selected: usize,
    want_w: u16,
) {
    let w = want_w.min(area.width.saturating_sub(4));
    // +2 border rows; cap to the frame and scroll the rest.
    let h = ((rows.len() + 2) as u16).min(area.height.saturating_sub(2));
    if w < 10 || h < 3 {
        // Same rule as the toasts: a box that cannot hold its own border is
        // not drawn at all rather than drawn wrong.
        return;
    }
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    draw_list_at(f, modal, title, rows, selected);
}

/// [`draw_list`]'s body, at an exact rect — for `draw_add`, which places the
/// list beside its help panel rather than centred alone.
fn draw_list_at(f: &mut Frame, modal: Rect, title: &str, rows: &[(String, bool)], selected: usize) {
    let visible = modal.height.saturating_sub(2) as usize;
    // Keep the cursor inside the window, cursor-follows-scroll like every
    // list in the dashboard.
    let top = selected.saturating_sub(visible.saturating_sub(1));
    f.render_widget(Clear, modal);

    let inner_w = modal.width.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = Vec::with_capacity(visible);
    for (i, (label, is_current)) in rows.iter().enumerate().skip(top).take(visible) {
        let cursor = i == selected;
        let marker = if cursor { "▌" } else { " " };
        // ✓ before the text, not a colour on it: the mark must survive both
        // NO_COLOR and being under the selection bar.
        let mark = if *is_current { "✓ " } else { "  " };
        // Ellipsised, not silently cut: on a narrow terminal the clipped
        // half of a help line or a donor's model id has to announce itself.
        let text = clip(label, inner_w.saturating_sub(3));
        let mut line = Line::from(vec![
            Span::styled(marker, theme::brand_purple()),
            Span::styled(mark, theme::brand_green().add_modifier(Modifier::BOLD)),
            Span::styled(
                text,
                if cursor {
                    theme::text()
                } else {
                    theme::text2()
                },
            ),
        ]);
        if cursor {
            line = line.style(theme::selected());
        }
        lines.push(line);
    }

    // The position, in the bottom border, only when the list is clipped: a
    // list that fits needs no scrollbar and gets no noise.
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::border(true))
        .title(Span::styled(title.to_string(), theme::title(true)))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    let block = if rows.len() > visible {
        block.title_bottom(Span::styled(
            format!("─ {}/{} ─", selected + 1, rows.len()),
            theme::dim(),
        ))
    } else {
        block
    };
    f.render_widget(Paragraph::new(lines).block(block), modal);
}

#[cfg(test)]
#[path = "modal_tests.rs"]
mod tests;
