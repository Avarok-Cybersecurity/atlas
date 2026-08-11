// SPDX-License-Identifier: AGPL-3.0-only

//! The recipe cards for one model.
//!
//! A model's recipes differ in quantization, context, speculation and topology —
//! a real choice, and one the list cannot present: repeating the model id once
//! per recipe with only a trailing stem to tell them apart puts the choice in
//! the wrong place and hides the reason for it.
//!
//! A card has room for the reason. Each recipe's `metadata.description` is the
//! measured rationale — which lm-head dtype was gated 10/10, why the KV cache is
//! bf16 — written by whoever ran the numbers. That is the content that decides
//! which card to pick, and this pane exists to show it.
//!
//! Deliberately the same shape as the website's model slider
//! (`site/src/lib/components/ModelSlider.svelte:107-137`): a family header over
//! a grid of subcards, each carrying name, chips, and identity.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::{panel, wrap};
use crate::recipe::Recipe;
use crate::tui::app::App;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
        .split(area);
    draw_cards(f, app, cols[0]);
    draw_detail(f, app, cols[1]);
}

/// `nvfp4` / `EP=2` / `27B` — the facts that distinguish sibling recipes.
fn chips(recipe: &Recipe) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    let mut chip = |text: String, style: Style| {
        out.push(Span::styled(
            format!(" {text} "),
            style.add_modifier(Modifier::REVERSED | Modifier::BOLD),
        ));
        out.push(Span::raw(" "));
    };
    if !recipe.quantization.is_empty() {
        chip(recipe.quantization.clone(), theme::brand_cyan());
    }
    // Topology is the one fact that changes what hardware you need, so it is
    // stated even when it is the unremarkable case.
    chip(
        if recipe.min_nodes > 1 {
            format!("EP={}", recipe.min_nodes)
        } else {
            "1 node".to_string()
        },
        if recipe.min_nodes > 1 {
            theme::warn()
        } else {
            theme::dim()
        },
    );
    if !recipe.model_params.is_empty() {
        chip(recipe.model_params.clone(), theme::brand_purple());
    }
    out
}

/// The recipe's own short name — the stem after `family/`.
fn stem(recipe: &Recipe) -> &str {
    recipe.id.rsplit('/').next().unwrap_or(&recipe.id)
}

fn draw_cards(f: &mut Frame, app: &App, area: Rect) {
    let recipes = app.lib.cards();
    let model = app
        .lib
        .current()
        .map(|e| e.model.clone())
        .unwrap_or_default();
    let block = panel(
        format!(
            "{} ─ {} recipe{} ─",
            model.to_uppercase(),
            recipes.len(),
            if recipes.len() == 1 { "" } else { "s" }
        ),
        true,
    );
    let inner = block.inner(area);
    f.render_widget(block, area);
    let width = inner.width.saturating_sub(4) as usize;

    // Four lines per card: title, chips, container, spacer.
    let per_card = 4usize;
    let visible = (inner.height as usize / per_card).max(1);
    let first = app.lib.card.saturating_sub(visible.saturating_sub(1));

    let mut lines: Vec<Line> = Vec::new();
    for (i, recipe) in recipes.iter().enumerate().skip(first).take(visible) {
        let selected = i == app.lib.card;
        let bar = if selected {
            Span::styled("▌", theme::brand_purple())
        } else {
            Span::raw(" ")
        };
        let title_style = if selected {
            theme::text().add_modifier(Modifier::BOLD)
        } else {
            theme::text()
        };
        // A recipe that cannot be launched from here says so on its face,
        // rather than only when the user presses Enter on it.
        let mut head = Line::from(vec![
            bar,
            Span::styled(format!(" {}", stem(recipe)), title_style),
            Span::styled(
                if recipe.is_atlas() {
                    String::new()
                } else {
                    format!("  ⊘ {}", recipe.runtime.as_deref().unwrap_or("non-atlas"))
                },
                theme::dim(),
            ),
        ]);
        if selected {
            head = head.style(theme::selected());
        }
        lines.push(head);

        let mut chip_line = vec![Span::raw("   ")];
        chip_line.extend(chips(recipe));
        lines.push(Line::from(chip_line));

        lines.push(Line::from(vec![
            Span::raw("   "),
            Span::styled(
                truncate(&recipe.container, width.saturating_sub(3)),
                theme::dim(),
            ),
        ]));
        lines.push(Line::from(""));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_detail(f: &mut Frame, app: &App, area: Rect) {
    let Some(recipe) = app.lib.selected_card() else {
        f.render_widget(panel("RECIPE ─".into(), false), area);
        return;
    };
    let block = panel(format!("{} ─", stem(recipe).to_uppercase()), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let width = inner.width.saturating_sub(2) as usize;

    // The description first and in full: it is the measured rationale, and the
    // reason this pane exists rather than a one-line row.
    let mut lines: Vec<Line> = wrap(&recipe.description, width, theme::text2());
    lines.push(Line::from(""));
    for (label, value) in [
        ("model", recipe.model.clone()),
        ("maintainer", recipe.maintainer.clone()),
        // Empty until a recipe carries `metadata.updated` or GitHub answers
        // with the file's last-commit date; the loop below skips empty values,
        // so an undated recipe draws no row rather than a blank one.
        ("updated", app.lib.date_text(recipe)),
        ("kv cache", recipe.kv_dtype.clone()),
        ("container", recipe.container.clone()),
    ] {
        if value.is_empty() {
            continue;
        }
        lines.push(Line::from(vec![
            Span::styled(format!("  {label:<12}"), theme::dim()),
            Span::styled(value, theme::text()),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!(" {} SETTINGS", recipe.defaults.len()),
        theme::dim(),
    )));
    for (key, value) in recipe.defaults.iter().take(8) {
        lines.push(Line::from(vec![
            Span::styled(format!("  {key:<24}"), theme::dim()),
            Span::styled(value.clone(), theme::text()),
        ]));
    }
    if recipe.defaults.len() > 8 {
        lines.push(Line::from(Span::styled(
            format!("   … {} more", recipe.defaults.len() - 8),
            theme::dim(),
        )));
    }
    lines.push(Line::from(""));
    // Say on the card what Enter will do, including when it will refuse. A
    // multi-node recipe reaching the form only to fail on a world-size check
    // sends the reader after the wrong fix.
    let launchable = recipe.is_atlas() && recipe.min_nodes <= 1;
    lines.push(Line::from(Span::styled(
        if launchable {
            " ⏎ configure and start".to_string()
        } else if !recipe.is_atlas() {
            " this runtime cannot be launched from here".to_string()
        } else {
            format!(
                " needs {} nodes — the dashboard starts single-node runs only",
                recipe.min_nodes
            )
        },
        if launchable {
            theme::brand_cyan()
        } else {
            theme::warn()
        },
    )));
    f.render_widget(Paragraph::new(lines), inner);
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max || max == 0 {
        return s.to_string();
    }
    s.chars().take(max.saturating_sub(1)).collect::<String>() + "…"
}
