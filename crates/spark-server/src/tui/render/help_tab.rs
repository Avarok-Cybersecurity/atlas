// SPDX-License-Identifier: AGPL-3.0-only

//! The Help section: the static guide, and the Report Issue pipeline's five
//! screens. Pure `App` → `Frame`, like every sibling.
//!
//! NO_COLOR discipline: every state signal here has a colour-free twin — the
//! checkbox is `[x]`/`[ ]` glyphs, phase changes are words, the user code and
//! action keys are bold, success/failure carry `✓`/`✗` marks — so the section
//! is fully legible when `theme` collapses to `Color::Reset`.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::{panel, wrap};
use crate::tui::app::App;
use crate::tui::help_state::{ComposerField, HelpSub, ReportPhase};
use crate::tui::{report, theme};

pub(super) fn draw(f: &mut Frame, app: &App, area: Rect) {
    match app.help.sub {
        HelpSub::Guide => draw_guide(f, app, area),
        HelpSub::Report => draw_report(f, app, area),
    }
}

fn draw_guide(f: &mut Frame, _app: &App, area: Rect) {
    let block = panel("HELP ─ GUIDE".into(), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let w = inner.width.saturating_sub(2) as usize;
    let repo = report::target()
        .map(|t| t.repo)
        .unwrap_or_else(|_| "(not configured)".into());
    let key = |k: &str, d: &str| {
        Line::from(vec![
            Span::styled(
                format!("  {k:<11}"),
                theme::brand_cyan().add_modifier(Modifier::BOLD),
            ),
            Span::styled(d.to_string(), theme::text2()),
        ])
    };
    let mut lines = vec![
        Line::from(Span::styled(
            format!("  Atlas {} — github.com/{repo}", crate::cli::ATLAS_VERSION),
            theme::text(),
        )),
        Line::from(""),
        key("?", "every key, on one screen"),
        key(
            "q",
            "stops the SERVER (drain + exit) — it is not \"close window\"",
        ),
        key("/detach", "Terminal tab: leave the dashboard, keep serving"),
        key("7 7", "a second press flips Guide ↔ Report Issue"),
        Line::from(""),
        Line::from(Span::styled(
            match crate::tui::init::tee_file_path() {
                Some(p) => format!("  Logs tee to {p}"),
                None => "  Log tee unavailable in this session".to_string(),
            },
            theme::text2(),
        )),
        Line::from(""),
    ];
    lines.extend(
        wrap(
            &format!(
                "Report Issue posts a bug report to the public tracker at github.com/{repo}. \
                 The GitHub authorization it asks for is kept in memory only and is forgotten \
                 when the server exits."
            ),
            w,
            theme::dim(),
        )
        .into_iter()
        .map(|mut l| {
            l.spans.insert(0, Span::raw("  "));
            l
        }),
    );
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_report(f: &mut Frame, app: &App, area: Rect) {
    // A blanked fork or an empty override renders the refusal, not a
    // half-working form whose submit would fail with a GitHub error page.
    let repo = match report::target() {
        Ok(t) => t.repo,
        Err(m) => {
            let block = panel("REPORT ISSUE".into(), false);
            let inner = block.inner(area);
            f.render_widget(block, area);
            f.render_widget(
                Paragraph::new(wrap(
                    m,
                    inner.width.saturating_sub(2) as usize,
                    theme::warn(),
                )),
                inner,
            );
            return;
        }
    };
    match &app.help.phase {
        ReportPhase::Compose => draw_compose(f, app, &repo, area),
        ReportPhase::Preview => draw_preview(f, app, &repo, area),
        other => draw_status(f, app, &repo, other, area),
    }
}

fn draw_compose(f: &mut Frame, app: &App, repo: &str, area: Rect) {
    let h = &app.help;
    let block = panel(
        format!("REPORT ISSUE ─ posts publicly to github.com/{repo}"),
        false,
    );
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height < 8 || inner.width < 20 {
        // Too small for a three-field form; say so instead of clipping the
        // body box into an unlabelled sliver.
        f.render_widget(
            Paragraph::new(Span::styled(
                "terminal too small for the composer",
                theme::warn(),
            )),
            inner,
        );
        return;
    }
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(3),
            Constraint::Length(2),
            Constraint::Length(1),
        ])
        .split(inner);

    // Title field.
    let tblock = panel("TITLE".into(), h.field == ComposerField::Title);
    let tinner = tblock.inner(rows[0]);
    f.render_widget(tblock, rows[0]);
    let mut tspans = vec![Span::styled(h.title.clone(), theme::text())];
    if h.title_editing {
        tspans.push(Span::styled("▏", theme::brand_cyan()));
    } else if h.title.is_empty() {
        tspans.push(Span::styled("one line — what broke?", theme::dim()));
    }
    f.render_widget(Paragraph::new(Line::from(tspans)), tinner);

    // Body field — the one input in the product with real cursor movement,
    // because retyping the tail of a long bug report is not acceptable.
    let bblock = panel("BODY".into(), h.field == ComposerField::Body);
    let binner = bblock.inner(rows[1]);
    f.render_widget(bblock, rows[1]);
    f.render_widget(&h.body, binner);

    // Attach checkbox: glyphs, not hue, carry the state.
    let box_mark = if h.attach_logs { "[x]" } else { "[ ]" };
    let attach_style = if h.field == ComposerField::Attach {
        theme::selected()
    } else {
        ratatui::style::Style::default()
    };
    let attach = vec![
        Line::from(vec![
            Span::styled(
                format!(" {box_mark} "),
                theme::text().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "attach server logs (last 10,000 lines, redacted best-effort —",
                theme::text2(),
            ),
        ])
        .style(attach_style),
        Line::from(Span::styled(
            "     you will preview exactly what is sent before anything is sent)",
            theme::text2(),
        )),
    ];
    f.render_widget(Paragraph::new(attach), rows[2]);

    let hint = if h.attach_logs {
        " s review & submit · ⏎ edit · j/k field · a toggle logs"
    } else {
        " s submit (no logs attached) · ⏎ edit · j/k field · a toggle logs"
    };
    f.render_widget(Paragraph::new(Span::styled(hint, theme::dim())), rows[3]);
}

fn draw_preview(f: &mut Frame, app: &App, repo: &str, area: Rect) {
    let h = &app.help;
    let Some(c) = &h.preview else {
        // Preview phase with no composed body cannot happen through the
        // reducer; render the composer rather than a blank pane if it does.
        draw_compose(f, app, repo, area);
        return;
    };
    let block = panel(
        format!("PREVIEW ─ exactly what will be posted to github.com/{repo}"),
        false,
    );
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height < 4 {
        return;
    }
    let logs = if c.logs_total == 0 {
        "no logs attached".to_string()
    } else {
        format!(
            "logs trimmed to last {} of {} lines",
            c.logs_included, c.logs_total
        )
    };
    let header = vec![
        Line::from(Span::styled(
            format!(
                " body {} / {} chars · {logs}",
                c.chars,
                crate::tui::redact::GITHUB_BODY_LIMIT
            ),
            theme::text2(),
        )),
        Line::from(Span::styled(
            " redaction is best-effort — read before you send",
            theme::warn(),
        )),
    ];
    let view_h = (inner.height - 3) as usize;
    let w = inner.width.saturating_sub(2) as usize;
    // Pre-wrapped display rows, so the scroll offset and the ceiling both
    // count what is actually on screen — the chat pane's contract.
    let rows: Vec<Line> = c
        .body
        .lines()
        .flat_map(|l| {
            if l.is_empty() {
                vec![Line::from("")]
            } else {
                wrap(l, w, theme::text())
            }
        })
        .collect();
    let max = rows.len().saturating_sub(view_h);
    h.preview_scroll_max.set(max);
    let scroll = h.preview_scroll.min(max);
    let mut lines = header;
    lines.push(Line::from(Span::styled(
        format!(" ── line {} of {} ──", scroll + 1, rows.len().max(1)),
        theme::dim(),
    )));
    lines.extend(rows.into_iter().skip(scroll).take(view_h));
    f.render_widget(Paragraph::new(lines), inner);
    let hint = Rect {
        x: inner.x,
        y: area.bottom().saturating_sub(1),
        width: inner.width,
        height: 1,
    };
    f.render_widget(
        Paragraph::new(Span::styled(
            "─ j/k scroll · y send · a toggle logs · Esc back ─",
            theme::dim(),
        )),
        hint,
    );
}

fn draw_status(f: &mut Frame, app: &App, repo: &str, phase: &ReportPhase, area: Rect) {
    let block = panel(format!("REPORT ISSUE ─ github.com/{repo}"), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let spin = theme::SPINNER[(app.tick as usize / 2) % theme::SPINNER.len()];
    let w = inner.width.saturating_sub(2) as usize;
    let key = theme::brand_cyan().add_modifier(Modifier::BOLD);
    let lines: Vec<Line> = match phase {
        ReportPhase::RequestingCode => vec![
            Line::from(Span::styled(
                format!("  {spin} contacting github.com…"),
                theme::text(),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Esc", key),
                Span::styled("  cancel — the draft stays in the composer", theme::text2()),
            ]),
        ],
        ReportPhase::WaitingAuth {
            user_code,
            verification_uri,
            expires_at,
        } => {
            let left = expires_at
                .saturating_duration_since(std::time::Instant::now())
                .as_secs();
            vec![
                Line::from(Span::styled(
                    "  First, authorize Atlas on GitHub:",
                    theme::text(),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    format!("      {user_code}"),
                    theme::brand_cyan().add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    format!("  open {verification_uri} and enter the code"),
                    theme::text(),
                )),
                Line::from(Span::styled(
                    format!(
                        "  code valid {}m {:02}s · {spin} waiting for authorization",
                        left / 60,
                        left % 60
                    ),
                    theme::text2(),
                )),
                Line::from(Span::styled(
                    "  kept in memory only — forgotten when the server exits",
                    theme::dim(),
                )),
                Line::from(""),
                Line::from(vec![
                    Span::styled("  c", key),
                    Span::styled("  copy code", theme::text2()),
                    Span::styled("     Esc", key),
                    Span::styled("  cancel", theme::text2()),
                ]),
            ]
        }
        ReportPhase::Submitting => vec![Line::from(Span::styled(
            format!("  {spin} posting issue to github.com/{repo}…"),
            theme::text(),
        ))],
        ReportPhase::Done { number, url } => {
            let opened = if *number == 0 {
                "  ✓ issue opened (GitHub did not return its number)".to_string()
            } else {
                format!("  ✓ issue #{number} opened")
            };
            let mut v = vec![Line::from(Span::styled(
                opened,
                theme::brand_green().add_modifier(Modifier::BOLD),
            ))];
            if !url.is_empty() {
                v.push(Line::from(Span::styled(format!("  {url}"), theme::text())));
            }
            v.push(Line::from(""));
            v.push(Line::from(vec![
                Span::styled("  c", key),
                Span::styled("  copy link", theme::text2()),
                Span::styled("     Esc", key),
                Span::styled("  compose another", theme::text2()),
            ]));
            v
        }
        ReportPhase::Failed { message } => {
            let mut v = vec![Line::from(Span::styled(
                "  ✗ sending failed",
                theme::error(),
            ))];
            v.extend(wrap(message, w, theme::warn()).into_iter().map(|mut l| {
                l.spans.insert(0, Span::raw("  "));
                l
            }));
            v.push(Line::from(""));
            v.push(Line::from(vec![
                Span::styled("  s", key),
                Span::styled("  retry — the draft is intact", theme::text2()),
                Span::styled("     Esc", key),
                Span::styled("  back to the composer", theme::text2()),
            ]));
            v
        }
        ReportPhase::Compose | ReportPhase::Preview => Vec::new(),
    };
    f.render_widget(Paragraph::new(lines), inner);
}

#[cfg(test)]
#[path = "help_tab_tests.rs"]
mod tests;
