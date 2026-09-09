// SPDX-License-Identifier: AGPL-3.0-only

//! History: past runs, read back from `~/.atlas/runs`.
//!
//! A stored run is the same `BenchmarkResult` the live pane rendered, so the
//! table and the stat tiles come from the shared helpers — a past run and a
//! present one look identical.
//!
//! ★ WHICH IS EXACTLY THE PROBLEM THIS FILE NOW GUARDS. Looking identical is
//! what makes two runs LOOK comparable; it does not make them comparable. A
//! run measured against a different model, or with different parameters, is a
//! different measurement wearing the same clothes — and the numbers most
//! worth comparing are the ones most likely to be compared wrongly. So the
//! list draws a RULE wherever the comparison basis changes between adjacent
//! runs, and names what changed. Points within a band are like-for-like;
//! points across a rule are not.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::panel;
use super::{draw_stats, draw_table, verdict_line};
use crate::tui::app::App;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    if app.bench.history.is_empty() {
        let block = panel("HISTORY ─".into(), false);
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(
            Paragraph::new(vec![
                Line::default(),
                Line::from(Span::styled("  No runs recorded yet.", theme::text2())),
                Line::from(Span::styled(
                    "  Every completed run is written to ~/.atlas/runs and appears here.",
                    theme::dim(),
                )),
            ]),
            inner,
        );
        return;
    }

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(34), Constraint::Min(30)])
        .split(area);
    draw_list(f, app, cols[0]);
    draw_detail(f, app, cols[1]);
}

fn draw_list(f: &mut Frame, app: &App, area: Rect) {
    let block = panel(format!("RUNS ─ {} ─", app.bench.history.len()), true);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let visible = inner.height as usize;
    let offset = app
        .bench
        .history_row
        .saturating_sub(visible.saturating_sub(1));
    let mut lines: Vec<Line> = Vec::new();
    let mut prev_basis: Option<String> = None;
    for (i, entry) in app
        .bench
        .history
        .iter()
        .enumerate()
        .skip(offset)
        .take(visible)
    {
        // Draw a rule where the comparison basis changes. The basis is what
        // a number has to hold constant to be compared: the benchmark, the
        // model it ran against, and the parameters it ran with.
        //
        // LIMITATION, stated rather than left for a reader to discover:
        // `RunRecord` does not carry the SERVE overrides, only the benchmark
        // `params`. So a run that differs solely in its serve profile — the
        // `hermetic` case — is not yet detected here. Closing that needs the
        // overrides on the record; until then this rule catches the changes
        // it can see and silently misses that one.
        let basis = basis_of(&entry.target_model, &entry.params);
        if let Some(prev) = &prev_basis
            && *prev != basis
            && !lines.is_empty()
        {
            lines.push(Line::from(Span::styled(
                format!(" ┄┄ {basis} ┄┄"),
                theme::dim(),
            )));
        }
        prev_basis = Some(basis);
        let selected = i == app.bench.history_row;
        let mark = match entry.frame.verdict.as_ref().map(|v| v.kind) {
            Some(atlas_plugin::VerdictKind::Pass) => Span::styled("✓", theme::brand_green()),
            Some(atlas_plugin::VerdictKind::Fail) => Span::styled("✗", theme::error()),
            _ => Span::styled("·", theme::dim()),
        };
        let mut line = Line::from(vec![
            Span::styled(if selected { "▌" } else { " " }, theme::brand_purple()),
            mark,
            Span::styled(
                format!(" {:<20}", entry.benchmark_id),
                if selected {
                    theme::text().add_modifier(Modifier::BOLD)
                } else {
                    theme::text2()
                },
            ),
            Span::styled(entry.age_text(), theme::dim()),
        ]);
        if selected {
            line = line.style(theme::selected());
        }
        lines.push(line);
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// What a run must hold constant for its numbers to be compared with
/// another's. Two runs sharing this string are like-for-like.
///
/// Model first because it is the change that most often invalidates a
/// comparison silently — the same gate against a different checkpoint
/// produces a plausible number that means something else entirely.
///
/// Pure over the two things it reads rather than taking the whole record, so
/// a test can exercise it without constructing a `RunRecord` — the same
/// reason `option_b_from` and `parse_flag_default_on` are split from their
/// readers elsewhere in this tree.
pub(super) fn basis_of(
    target_model: &str,
    params: &std::collections::BTreeMap<String, String>,
) -> String {
    let model = target_model.rsplit('/').next().unwrap_or(target_model);
    // A stable digest of every parameter, not a subset: the record stores
    // EVERY parameter precisely so a comparison cannot be invalidated by one
    // nobody thought to list here.
    let params: String = params
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("{model} · {}", short_hash(&params))
}

/// A short, stable, human-ignorable tag for a parameter set. Not a security
/// hash — it only has to differ when the parameters differ.
fn short_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:04x}", (h ^ (h >> 32)) as u16)
}

fn draw_detail(f: &mut Frame, app: &App, area: Rect) {
    let Some(entry) = app.bench.history.get(app.bench.history_row) else {
        return;
    };
    let frame = &entry.frame;
    let has_stats = !frame.summary.is_empty();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(if has_stats { 3 } else { 0 }),
            Constraint::Min(6),
            Constraint::Length(2),
        ])
        .split(area);
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                format!(" {} ", entry.benchmark_id),
                theme::text().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "· {} · {:.0}s · {}",
                    frame.phase,
                    frame.elapsed.as_secs_f64(),
                    entry.age_text()
                ),
                theme::text2(),
            ),
        ])),
        rows[0],
    );
    if has_stats {
        draw_stats(f, &frame.summary, rows[1]);
    }
    if let Some(table) = &frame.table {
        // A stored 40-row sweep was readable only down to the pane height —
        // `scroll` was hardcoded 0 and History bound no key to move it.
        let max = draw_table(f, table, app.bench.history_table_scroll, rows[2]);
        app.bench.history_table_scroll_max.set(max);
    }
    if let Some(verdict) = &frame.verdict {
        f.render_widget(Paragraph::new(verdict_line(verdict)), rows[3]);
    }
}
