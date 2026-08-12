// SPDX-License-Identifier: AGPL-3.0-only

//! The detail pane for the selected list row: the recipe summary, the on-disk
//! facts, the live download bar, and the per-row key hints.
//!
//! Split from `list.rs` at the 500-LoC cap. It is the right seam: the list is
//! the scannable half and this pane is the readable half, and the two share
//! only `App` and the selected `Entry`.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::{gradient_bar, panel, wrap};
use crate::tui::app::App;
use crate::tui::data::catalogue::Entry;
use crate::tui::theme;

pub(super) fn draw_detail(f: &mut Frame, app: &App, area: Rect) {
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
    match entry.primary() {
        Some(recipe) => {
            lines.extend(wrap(&recipe.description, width, theme::text2()));
            lines.push(Line::from(""));
            for (label, value) in [
                ("recipe", recipe.id.clone()),
                ("maintainer", recipe.maintainer.clone()),
                ("updated", app.lib.date_text(recipe)),
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
                match entry.recipes.len() {
                    1 => format!(" SETTINGS  {} editable", recipe.defaults.len()),
                    n => format!(" {n} RECIPES  ⏎ to choose"),
                },
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
                "No recipe covers this checkpoint. ⏎ offers starting points —",
                theme::text2(),
            )));
            lines.push(Line::from(Span::styled(
                "published recipes re-aimed at this model, or a blank config.",
                theme::text2(),
            )));
            lines.push(Line::from(Span::styled(
                "None of them is measured on this model; review before launch.",
                theme::warn(),
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

    // The download this model is in the middle of, as a real bar. The list
    // row's one-line progress survives narrow panes; this pane has the width
    // to answer "how far along, how fast, which file" in full.
    if let Some(job) = app.download.job.as_ref().filter(|j| j.repo == entry.model) {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(" DOWNLOADING", theme::dim())));
        let bar_w = inner.width.saturating_sub(12).clamp(8, 36);
        let mut bar = vec![Span::raw("  ")];
        match job.fraction() {
            Some(frac) => {
                bar.extend(gradient_bar(frac, bar_w).spans);
                bar.push(Span::styled(
                    format!(" {:>3.0}%", frac * 100.0),
                    theme::text().add_modifier(Modifier::BOLD),
                ));
            }
            // No sizes from the Hub: motion instead of a bar stuck at zero,
            // same rule as the list row.
            None => {
                let phase = (app.tick as usize / 2) % theme::SPINNER.len();
                bar.push(Span::styled(theme::SPINNER[phase], theme::brand_cyan()));
                bar.push(Span::styled(" resolving…", theme::text2()));
            }
        }
        lines.push(Line::from(bar));
        if job.total > 0 {
            let mut detail = vec![Span::styled(
                format!(
                    "  {} / {}",
                    crate::tui::format::bytes(job.done),
                    crate::tui::format::bytes(job.total)
                ),
                theme::text2(),
            )];
            // No rate while stopping: it is the one field that implies the
            // bytes are still moving, and the user just asked them not to.
            if job.rate_bps > 0.0 && !job.cancelling {
                detail.push(Span::styled(
                    format!("  {}", crate::tui::format::rate(job.rate_bps)),
                    theme::dim(),
                ));
            }
            lines.push(Line::from(detail));
        }
        if let Some((i, of, name)) = &job.file {
            lines.push(Line::from(Span::styled(
                format!("  file {i}/{of}  {name}"),
                theme::dim(),
            )));
        }
        if job.cancelling {
            lines.push(Line::from(Span::styled("  stopping…", theme::warn())));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        detail_footer(app, entry),
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
/// What the keys will do for THIS row, said on the row itself.
///
/// The alternative — one static hint listing every key — makes the reader work
/// out which of them apply, and `d` means three different things depending on
/// what is on disk.
fn detail_footer(app: &App, entry: &Entry) -> String {
    if app.download.is_downloading(&entry.model) {
        return " x stop the download  ·  ⏎ choose a recipe".into();
    }
    let stale = app
        .download
        .freshness
        .get(&entry.model)
        .is_some_and(|f| f.is_stale());
    match (
        entry.runnable_now(),
        entry.has_recipe(),
        entry.local.is_some(),
    ) {
        // Everything is here. Only mention updating if we KNOW it is behind.
        (true, _, _) if stale => " d update  ·  ⏎ choose a recipe".into(),
        (true, _, _) => " ⏎ choose a recipe  ·  u check for updates".into(),
        // On disk but unloadable: a previous download did not finish.
        (_, true, true) => " d resume the download  ·  ⏎ choose a recipe".into(),
        (_, true, false) => " d download the weights  ·  ⏎ choose a recipe".into(),
        _ => " d download the weights  ·  ⏎ pick a starting point".into(),
    }
}
