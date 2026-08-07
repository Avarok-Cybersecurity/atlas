// SPDX-License-Identifier: AGPL-3.0-only

//! The parameter form: every knob the benchmark exposes, editable before the
//! run, plus the endpoint it will be pointed at.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};

use super::super::{panel, wrap};
use crate::tui::app::App;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(4)])
        .split(area);
    draw_form(f, app, rows[0]);
    draw_help(f, app, rows[1]);
    if app.bench.confirm_open {
        draw_confirm(f, app, area);
    }
    // Drawn last so it sits above the consent gate: by the time the endpoint
    // is being checked, consent has already been given.
    if app.bench.preflight.is_some() {
        draw_preflight(f, app, area);
    }
}

fn draw_form(f: &mut Frame, app: &App, area: Rect) {
    let name = app.bench.descriptor().map(|d| d.name).unwrap_or("");
    let block = panel(format!("{} ─ PARAMETERS ─", name.to_uppercase()), true);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let label_w = 22usize;
    let mut lines: Vec<Line> = Vec::new();
    for row in 0..app.bench.row_count() {
        let (label, help, hint) = app.bench.row_meta(row);
        let selected = row == app.bench.row;
        let editing = selected && app.bench.editing;
        let value = app.bench.edit.get(row).cloned().unwrap_or_default();

        // The two endpoint rows are a different kind of thing from the
        // benchmark's own knobs, so they get a rule rather than blending in.
        if row == app.bench.specs.len() {
            lines.push(Line::from(Span::styled(
                format!(" {}", "─".repeat(inner.width.saturating_sub(2) as usize)),
                theme::dim(),
            )));
            lines.push(Line::from(Span::styled(" TARGET", theme::dim())));
        }

        let marker = if selected { "▌" } else { " " };
        let value_style = if editing {
            theme::brand_cyan().add_modifier(Modifier::BOLD)
        } else if app.bench.row_error(row).is_some() {
            theme::error()
        } else {
            theme::text()
        };
        let mut spans = vec![
            Span::styled(marker, theme::brand_purple()),
            Span::styled(format!(" {label:<label_w$}"), {
                if selected {
                    theme::text().add_modifier(Modifier::BOLD)
                } else {
                    theme::text2()
                }
            }),
            Span::styled(value.clone(), value_style),
        ];
        if editing {
            spans.push(Span::styled("▏", theme::brand_cyan()));
        }
        spans.push(Span::styled(format!("   {hint}"), theme::dim()));
        let mut line = Line::from(spans);
        if selected {
            line = line.style(theme::selected());
        }
        lines.push(line);

        // Help and validation attach to the selected row only — showing every
        // help line at once turns the form into a wall of grey.
        if selected {
            if let Some(err) = app.bench.row_error(row) {
                lines.push(Line::from(Span::styled(
                    format!("   {err}"),
                    theme::error(),
                )));
            } else {
                lines.push(Line::from(Span::styled(format!("   {help}"), theme::dim())));
            }
        }
    }

    // The probe is a run option rather than a form field, so it gets a status
    // line instead of a row — but it must be visible, or `p` is a secret.
    let (probe_text, probe_style) = match app.bench.coherence {
        atlas_plugin::CoherencePolicy::Probe => (
            " endpoint check  on — warns if the model answers unexpectedly",
            theme::dim(),
        ),
        atlas_plugin::CoherencePolicy::Skip => (
            " endpoint check  OFF — the model will not be asked anything",
            theme::warn(),
        ),
    };
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(probe_text, probe_style)));

    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_help(f: &mut Frame, app: &App, area: Rect) {
    let block = panel("─".into(), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let keys = if app.bench.editing {
        "⏎ commit · Esc cancel"
    } else {
        "j/k move · ⏎ edit · d defaults · p probe · s START · Esc back"
    };
    let mut lines = vec![Line::from(Span::styled(
        format!(" {keys}"),
        theme::brand_cyan(),
    ))];
    if !app.bench.errors.is_empty() {
        lines.push(Line::from(Span::styled(
            format!(
                " {} field(s) need fixing before this can start",
                app.bench.errors.len()
            ),
            theme::error(),
        )));
    } else if app.bench.is_running() {
        lines.push(Line::from(Span::styled(
            " a benchmark is already running — cancel it first",
            theme::warn(),
        )));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// The endpoint check, and the decision when it has something to report.
///
/// Two states in one modal because they are one moment for the user: the
/// spinner is indeterminate on purpose — the check is two completions against
/// a server whose speed is the thing being measured, so any percentage would
/// be invented.
fn draw_preflight(f: &mut Frame, app: &App, area: Rect) {
    use crate::tui::bench_preflight::Phase;
    let Some(pre) = &app.bench.preflight else {
        return;
    };
    let checking = pre.is_checking();
    let w = 72.min(area.width.saturating_sub(4));
    let h = if checking { 7 } else { 14 }.min(area.height.saturating_sub(2));
    let modal = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);
    let block = panel(
        if checking {
            "CHECKING THE ENDPOINT ─".into()
        } else {
            "BEFORE YOU START ─".to_string()
        },
        true,
    );
    let inner = block.inner(modal);
    f.render_widget(block, modal);
    let width = inner.width.saturating_sub(2) as usize;

    let mut lines: Vec<Line> = Vec::new();
    match &pre.phase {
        Phase::Checking => {
            lines.push(Line::from(vec![
                Span::styled(
                    format!(
                        " {} ",
                        theme::SPINNER[(app.tick as usize / 2) % theme::SPINNER.len()]
                    ),
                    theme::brand_cyan(),
                ),
                Span::styled("asking the model two known-answer questions", theme::text()),
            ]));
            lines.push(Line::default());
            lines.extend(wrap(
                &format!("{} · {}", app.bench.target.base_url, app.bench.target.model),
                width,
                theme::dim(),
            ));
            lines.push(Line::default());
            lines.push(Line::from(Span::styled(" Esc  cancel", theme::dim())));
        }
        Phase::Concern(text) => {
            lines.push(Line::from(Span::styled(
                " ⚠  this may not be the server you meant",
                theme::warn().add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::default());
            lines.extend(wrap(text, width, theme::text2()));
            lines.push(Line::default());
            lines.push(Line::from(vec![
                Span::styled(" p ", theme::brand_green().add_modifier(Modifier::REVERSED)),
                Span::styled(" run it anyway   ", theme::text()),
                Span::styled(" Esc ", theme::dim().add_modifier(Modifier::REVERSED)),
                Span::styled(" back to the form", theme::text2()),
            ]));
        }
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// The consent gate for the one benchmark that executes model-authored shell.
fn draw_confirm(f: &mut Frame, app: &App, area: Rect) {
    let w = 66.min(area.width.saturating_sub(4));
    let h = 11.min(area.height.saturating_sub(2));
    let modal = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);
    let block = panel("CONFIRM ─".into(), true);
    let inner = block.inner(modal);
    f.render_widget(block, modal);
    let mut lines = vec![Line::from(Span::styled(
        // Short enough to fit the modal's inner width at every terminal size
        // this renders at — a clipped warning reads as a rendering bug.
        format!(
            " {} runs model-written shell.",
            app.bench
                .descriptor()
                .map(|d| d.name)
                .unwrap_or("This benchmark")
        ),
        theme::warn().add_modifier(Modifier::BOLD),
    ))];
    lines.push(Line::default());
    lines.extend(wrap(
        "Commands run inside a fresh sandbox directory under ~/.atlas/runs, with a per-command \
         timeout and a capped turn count. They are not otherwise restricted: building and \
         running the code the model wrote is the measurement.",
        inner.width.saturating_sub(2) as usize,
        theme::text2(),
    ));
    lines.push(Line::default());
    lines.push(Line::from(vec![
        Span::styled(" y ", theme::brand_green().add_modifier(Modifier::REVERSED)),
        Span::styled(" start   ", theme::text()),
        Span::styled(" any other key ", theme::dim()),
        Span::styled(" cancel", theme::text2()),
    ]));
    f.render_widget(Paragraph::new(lines), inner);
}
