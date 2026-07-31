// SPDX-License-Identifier: AGPL-3.0-only

//! The joined recipe⋈local list, and its detail pane.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::{panel, wrap};
use crate::tui::app::App;
use crate::tui::data::catalogue::Entry;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
        .split(area);
    draw_list(f, app, cols[0]);
    draw_detail(f, app, cols[1]);
}

/// `▐recipe▌` and `▐optimized▌`. Two independent facts: a recipe with no
/// compiled kernel target still serves, on generic kernels.
fn badges(entry: &Entry) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    if let Some(r) = &entry.recipe {
        let (label, style) = if r.is_atlas() {
            (" recipe ", theme::brand_purple())
        } else {
            // Listed, never hidden — but it cannot be launched from here.
            (" vllm ", theme::dim())
        };
        out.push(Span::styled(
            label,
            style.add_modifier(Modifier::REVERSED | Modifier::BOLD),
        ));
        out.push(Span::raw(" "));
    }
    if entry.optimized() {
        out.push(Span::styled(
            " optimized ",
            theme::brand_cyan().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        ));
    }
    out
}

fn draw_list(f: &mut Frame, app: &App, area: Rect) {
    let rows = app.lib.visible();
    let search = if app.lib.filter_editing {
        format!(" search: {}▏", app.lib.filter)
    } else if !app.lib.filter.is_empty() {
        format!(" search: {}", app.lib.filter)
    } else {
        String::new()
    };
    // The freshness of the recipe list belongs in the title: it is context for
    // every row, not a property of any one of them.
    let status = if app.lib.fetching {
        format!(
            " {} fetching ─",
            theme::SPINNER[(app.tick as usize / 2) % theme::SPINNER.len()]
        )
    } else {
        format!(" recipes {} ─", app.lib.index.status_text())
    };
    // Kept short: a long title is the first thing to overflow on a narrow
    // terminal, and the separators are decoration, not information.
    let block = panel(format!("MODELS ─ {} ─{search}{status} ─", rows.len()), true);
    let inner = block.inner(area);
    f.render_widget(block, area);

    if rows.is_empty() {
        let hint = if app.lib.filter.is_empty() {
            "no models or recipes yet — press r to fetch recipes"
        } else {
            "nothing matches this search"
        };
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(format!(" {hint}"), theme::dim()))),
            inner,
        );
        return;
    }

    // Three lines per row; keep the selection on screen.
    let per_row = 3usize;
    let visible = (inner.height as usize / per_row).max(1);
    let first = app.lib.selected.saturating_sub(visible.saturating_sub(1));

    let mut lines: Vec<Line> = Vec::new();
    for (i, entry) in rows.iter().enumerate().skip(first).take(visible) {
        let selected = i == app.lib.selected;
        let bar = if selected {
            Span::styled("▌", theme::brand_purple())
        } else {
            Span::raw(" ")
        };
        // A checkmark means "the weights are here", nothing else.
        let mark = if entry.has_weights() {
            Span::styled("✓ ", theme::brand_green())
        } else if entry.local.is_some() {
            Span::styled("◐ ", theme::warn())
        } else {
            Span::styled("· ", theme::dim())
        };
        let name_style = if selected {
            theme::text().add_modifier(Modifier::BOLD)
        } else {
            theme::text()
        };
        let mut head = Line::from(vec![
            bar,
            mark,
            Span::styled(entry.model.clone(), name_style),
        ]);
        if selected {
            head = head.style(Style::default().bg(theme::BG_SELECTION.color()));
        }
        lines.push(head);

        let mut second = vec![Span::raw("   ")];
        second.extend(badges(entry));
        let subtitle = entry.subtitle();
        if !subtitle.is_empty() {
            second.push(Span::styled(format!(" {subtitle}"), theme::dim()));
        }
        lines.push(Line::from(second));
        lines.push(Line::from(vec![
            Span::raw("   "),
            Span::styled(entry.size_text(), theme::text2()),
            Span::styled(
                match &entry.recipe {
                    Some(r) => format!("  ·  {}", r.id),
                    None => "  ·  no recipe".into(),
                },
                theme::dim(),
            ),
        ]));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_detail(f: &mut Frame, app: &App, area: Rect) {
    let Some(entry) = app.lib.current() else {
        let block = panel("MODEL ─".into(), false);
        f.render_widget(block, area);
        return;
    };
    let block = panel(format!("{} ─", entry.model.to_uppercase()), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let width = inner.width.saturating_sub(2) as usize;

    let mut lines: Vec<Line> = Vec::new();
    match &entry.recipe {
        Some(recipe) => {
            lines.extend(wrap(&recipe.description, width, theme::text2()));
            lines.push(Line::from(""));
            for (label, value) in [
                ("recipe", recipe.id.clone()),
                ("maintainer", recipe.maintainer.clone()),
                ("quantization", recipe.quantization.clone()),
                ("kv cache", recipe.kv_dtype.clone()),
                ("container", recipe.container.clone()),
                (
                    "nodes",
                    if recipe.min_nodes > 1 {
                        format!("{} (multi-node)", recipe.min_nodes)
                    } else {
                        "1".into()
                    },
                ),
            ] {
                if value.is_empty() {
                    continue;
                }
                lines.push(kv(label, &value));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!(" SETTINGS  {} editable", recipe.defaults.len()),
                theme::dim(),
            )));
            // A preview, not the form: enough to judge the recipe without
            // opening it, capped so the pane stays readable.
            for (key, value) in recipe.defaults.iter().take(6) {
                lines.push(kv(key, value));
            }
            if recipe.defaults.len() > 6 {
                lines.push(Line::from(Span::styled(
                    format!("   … {} more", recipe.defaults.len() - 6),
                    theme::dim(),
                )));
            }
        }
        None => {
            lines.push(Line::from(Span::styled(
                "No recipe covers this checkpoint. It can still be served, but",
                theme::text2(),
            )));
            lines.push(Line::from(Span::styled(
                "you choose the flags yourself.",
                theme::text2(),
            )));
        }
    }

    if let Some(local) = &entry.local {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(" ON DISK", theme::dim())));
        lines.push(kv("size", &entry.size_text()));
        lines.push(kv("architecture", &local.model_type));
        lines.push(kv("layers", &local.layers.to_string()));
        lines.push(kv(
            "kernels",
            if local.optimized {
                "optimized"
            } else {
                "generic"
            },
        ));
    } else {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            " weights are not in the local cache",
            theme::warn(),
        )));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        match (entry.runnable_now(), entry.has_recipe()) {
            (true, _) => " ⏎ configure this recipe",
            (_, true) => " ⏎ configure  ·  weights must be downloaded first",
            _ => " no recipe to configure",
        },
        theme::brand_cyan(),
    )));
    f.render_widget(Paragraph::new(lines), inner);
}

fn kv(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {label:<14}"), theme::dim()),
        Span::styled(value.to_string(), theme::text()),
    ])
}
