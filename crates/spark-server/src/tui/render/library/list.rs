// SPDX-License-Identifier: AGPL-3.0-only

//! The joined recipe⋈local list, and its detail pane.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
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
fn badges(app: &App, entry: &Entry) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    if let Some(r) = entry.primary() {
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
        out.push(Span::raw(" "));
    }
    // Last, so it is the first thing dropped when the pane is narrow, and only
    // for a CONFIRMED mismatch: an unreachable Hub reports `Unknown`, which
    // draws nothing. A stale badge shown because the network was down would be
    // a lie about the user's disk.
    match app.download.freshness.get(&entry.model) {
        Some(f) if f.is_stale() => out.push(Span::styled(
            " update ",
            theme::warn().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        )),
        _ if app.download.checking.as_deref() == Some(entry.model.as_str()) => {
            // Skeleton in the place the badge will occupy, so the row does not
            // reflow when the answer lands.
            out.push(Span::styled(" ░░░░░░ ", theme::dim()));
        }
        _ => {}
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

    // Why the recipe list is stale belongs on screen, not only in a log. The
    // title has room for "offline"; the cause is what the reader has to act on,
    // and it differs completely — no route needs a proxy, a 403 needs a wait.
    let mut header: Vec<Line> = Vec::new();
    if let Some(detail) = app.lib.index.offline_detail() {
        header.extend(wrap(
            &detail,
            inner.width.saturating_sub(2) as usize,
            theme::warn(),
        ));
        header.push(Line::from(""));
    }

    if rows.is_empty() {
        let hint = if app.lib.filter.is_empty() {
            "no models or recipes yet — press r to fetch recipes"
        } else {
            "nothing matches this search"
        };
        header.push(Line::from(Span::styled(format!(" {hint}"), theme::dim())));
        f.render_widget(Paragraph::new(header), inner);
        return;
    }

    // Three lines per row; keep the selection on screen.
    let per_row = 3usize;
    let visible = (inner.height as usize / per_row).max(1);
    let first = app.lib.selected.saturating_sub(visible.saturating_sub(1));

    let mut lines: Vec<Line> = header;
    for (i, entry) in rows.iter().enumerate().skip(first).take(visible) {
        let selected = i == app.lib.selected;
        let bar = if selected {
            Span::styled("▌", theme::brand_purple())
        } else {
            Span::raw(" ")
        };
        // A checkmark means "the weights are here", nothing else. A download
        // in flight owns the glyph while it runs.
        let mark = if app.download.is_downloading(&entry.model) {
            Span::styled("↓ ", theme::brand_cyan())
        } else if entry.has_weights() {
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
            head = head.style(theme::selected());
        }
        lines.push(head);

        let mut second = vec![Span::raw("   ")];
        second.extend(badges(app, entry));
        let subtitle = entry.subtitle();
        if !subtitle.is_empty() {
            second.push(Span::styled(format!(" {subtitle}"), theme::dim()));
        }
        lines.push(Line::from(second));
        // Line three is either the model's size, or — while it is being
        // fetched — that same line turned into a progress bar. Reusing the
        // line is deliberate: `per_row` stays 3, so no scroll arithmetic
        // changes, and the progress stays attached to the model it belongs to
        // instead of floating in a modal.
        lines.push(match progress_line(app, &entry.model, inner.width) {
            Some(l) => l,
            None => Line::from(vec![
                Span::raw("   "),
                Span::styled(entry.size_text(), theme::text2()),
                Span::styled(
                    match entry.recipes.len() {
                        0 => "  ·  no recipe".to_string(),
                        // One recipe still names itself; several become a count,
                        // because listing three stems is what the card view is for.
                        1 => format!("  ·  {}", entry.recipes[0].id),
                        n => format!("  ·  {n} recipes"),
                    },
                    theme::dim(),
                ),
            ]),
        });
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

/// The progress line for a model, if one is being downloaded.
///
/// Degrades right-to-left as the pane narrows — rate first, then the file
/// counter, then the byte pair — so the bar and percentage, which are the
/// parts that answer "is this still moving", survive longest.
fn progress_line(app: &App, model: &str, width: u16) -> Option<Line<'static>> {
    let job = app.download.job.as_ref().filter(|j| j.repo == model)?;
    let mut spans = vec![Span::raw("   ")];

    // A spinner ALWAYS, in front of whatever else this line says. On a 20 GB
    // model the first several minutes are under one percent, and a bar that
    // renders as twelve empty cells next to "0%" is indistinguishable from a
    // download that never started — which is exactly how this was first
    // reported. The spinner is the part that says "still moving" when the
    // numbers cannot yet.
    let phase = (app.tick as usize / 2) % theme::SPINNER.len();
    spans.push(Span::styled(theme::SPINNER[phase], theme::brand_cyan()));
    spans.push(Span::raw(" "));

    match job.fraction() {
        Some(f) => {
            const CELLS: usize = 12;
            // At least one cell once ANY bytes have moved: rounding 0.75% of
            // twelve cells to zero drew an empty bar for a download that was
            // running fine.
            let exact = (f * CELLS as f64).round() as usize;
            let filled = if job.done > 0 { exact.max(1) } else { exact };
            spans.push(Span::styled(
                "▓".repeat(filled.min(CELLS)),
                theme::brand_cyan(),
            ));
            spans.push(Span::styled(
                "░".repeat(CELLS.saturating_sub(filled)),
                theme::dim(),
            ));
            // One decimal below 10%, because "0%" for twenty minutes of real
            // progress reads as a stall. Whole numbers above that.
            let pct = f * 100.0;
            spans.push(Span::styled(
                if pct < 10.0 {
                    format!("  {pct:>4.1}%")
                } else {
                    format!("  {pct:>3.0}%")
                },
                theme::text(),
            ));
        }
        // The Hub did not report sizes, so there is no fraction to draw.
        None => spans.push(Span::styled("fetching", theme::text())),
    }

    if job.cancelling {
        // Cancellation is honoured within a chunk, but saying so beats a bar
        // that keeps moving after the user asked it to stop.
        spans.push(Span::styled("  stopping…", theme::warn()));
        return Some(Line::from(spans));
    }
    if width >= 60 && job.total > 0 {
        spans.push(Span::styled(
            format!("  {} / {}", gb(job.done), gb(job.total)),
            theme::text2(),
        ));
    }
    // Rate BEFORE the file counter, and at a much lower threshold. The list
    // pane is roughly half the content area, so at a 200-column terminal this
    // line gets ~85 columns — the old 96 threshold meant the rate, the one
    // field that proves bytes are moving, never appeared at any realistic
    // window size.
    if width >= 70 && job.rate_bps > 0.0 {
        spans.push(Span::styled(
            format!("  {}", crate::tui::format::rate(job.rate_bps)),
            theme::dim(),
        ));
    }
    if width >= 88
        && let Some((i, of, _)) = &job.file
    {
        spans.push(Span::styled(format!("  file {i}/{of}"), theme::dim()));
    }
    Some(Line::from(spans))
}

/// ★ This used to divide by 10⁹ while the Library card that replaces this row
/// on completion divided by 1024³ — so a checkpoint downloaded as "20.0 GB"
/// became "18.6 GB" the moment it finished, for no reason the user could see.
fn gb(bytes: u64) -> String {
    crate::tui::format::bytes(bytes)
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
        _ => " d download the weights".into(),
    }
}
