// SPDX-License-Identifier: AGPL-3.0-only

//! Main tab: startup checklist + weight-load hero (loading), READY strip
//! (serving), badge chips, live log pane; plus the Kernels subsection table.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Row, Sparkline, Table};

use super::{gradient_bar, panel};
use crate::tui::app::App;
use crate::tui::progress::PhaseState;
use crate::tui::{log_ring, logo, theme};

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let loading = !app.progress.ready;
    let top_h = if loading { 15 } else { 3 };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(top_h),
            Constraint::Length(2),
            Constraint::Min(5),
        ])
        .split(area);
    if loading {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(34), Constraint::Min(30)])
            .split(rows[0]);
        draw_phases(f, app, cols[0]);
        draw_weight_load(f, app, cols[1]);
    } else {
        draw_ready_strip(f, app, rows[0]);
    }
    draw_chips(f, app, rows[1]);
    draw_logs(f, app, rows[2]);
}

fn draw_phases(f: &mut Frame, app: &App, area: Rect) {
    let (done, total, secs) = app.progress.phase_counts();
    let block = panel(format!("STARTUP ─ {done}/{total} ── {secs:.1}s ─"), false);
    let mut lines = Vec::new();
    for p in &app.progress.phases {
        let (glyph, gstyle, label_style) = match p.state {
            PhaseState::Done => ("✓", theme::brand_green(), theme::text2()),
            PhaseState::Running => (
                theme::SPINNER[(app.tick as usize) % theme::SPINNER.len()],
                theme::brand_cyan(),
                theme::text(),
            ),
            PhaseState::Pending => ("○", theme::dim(), theme::dim()),
        };
        let secs = match p.state {
            PhaseState::Done if p.secs > 0.005 => format!("{:>7.1}s", p.secs),
            PhaseState::Running => {
                format!(
                    "{:>7.1}s",
                    p.started.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0)
                )
            }
            _ => "        ".into(),
        };
        lines.push(Line::from(vec![
            Span::styled(format!(" {glyph} "), gstyle),
            Span::styled(format!("{:<18}", p.name), label_style),
            Span::styled(secs, theme::text2()),
        ]));
    }
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_weight_load(f: &mut Frame, app: &App, area: Rect) {
    let p = &app.progress;
    let title = if p.shard_name.is_empty() {
        "WEIGHT LOAD ─".to_string()
    } else {
        format!("WEIGHT LOAD ─ {} ─", middle_truncate(&p.shard_name, 34))
    };
    let block = panel(title, false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let bar_w = inner.width.saturating_sub(22);

    let mut lines: Vec<Line> = vec![Line::default()];
    // OVERALL, gradient (the hero).
    let overall = p.displayed_overall();
    let mut l = vec![Span::styled(" OVERALL    ", theme::text2())];
    l.extend(gradient_bar(overall, bar_w).spans);
    l.push(Span::styled(
        format!(" {:>3.0}%", overall * 100.0),
        theme::text().add_modifier(Modifier::BOLD),
    ));
    lines.push(Line::from(l));
    if p.disk_gb > 0.0 {
        lines.push(Line::from(Span::styled(
            format!(
                "            {:.1} / {:.1} GB preflight",
                p.disk_gb * overall,
                p.disk_gb
            ),
            theme::dim(),
        )));
    }
    lines.push(Line::default());
    // SHARD, plain cyan (hierarchy: only the hero gets the gradient).
    if p.shard_total > 0 {
        let frac = p.shard_target();
        let filled = (frac * bar_w as f64) as usize;
        let bar: String = "█".repeat(filled) + &"░".repeat(bar_w as usize - filled);
        lines.push(Line::from(vec![
            Span::styled(
                format!(" SHARD {:>2}/{:<3}", p.shard, p.shard_total),
                theme::text2(),
            ),
            Span::styled(bar, theme::brand_cyan()),
        ]));
    }
    if p.layer_total > 0 {
        lines.push(Line::from(Span::styled(
            format!(
                " LAYERS {:>2}/{:<3} GPU used {:.1} · free {:.1} GB",
                p.layer, p.layer_total, p.gpu_used_gb, p.gpu_free_gb
            ),
            theme::text2(),
        )));
    }
    lines.push(Line::default());
    if let Some((rate, eta)) = p.rate_eta() {
        // Once the load window closes the rate is a final measurement, so report
        // what it took instead of an ETA that is necessarily 0:00.
        let tail = match p.load_secs() {
            Some(s) => format!("loaded in {}:{:02}", (s as u64) / 60, (s as u64) % 60),
            None => format!("eta {}:{:02}", (eta as u64) / 60, (eta as u64) % 60),
        };
        lines.push(Line::from(Span::styled(
            format!(" {rate:.2} GB/s · {tail}"),
            theme::brand_cyan(),
        )));
    }
    f.render_widget(Paragraph::new(lines), inner);
    // MEM sparkline along the bottom of the panel interior.
    if !p.mem_history.is_empty() && inner.height >= 8 {
        let spark_area = Rect {
            y: inner.y + inner.height - 1,
            height: 1,
            x: inner.x + 1,
            width: inner.width - 2,
        };
        f.render_widget(
            Sparkline::default()
                .data(&p.mem_history)
                .style(theme::brand_cyan()),
            spark_area,
        );
    }
}

fn draw_ready_strip(f: &mut Frame, app: &App, area: Rect) {
    let block = panel("READY ─".into(), false);
    let model = app
        .args
        .model_name
        .clone()
        .or_else(|| app.args.model.clone())
        .unwrap_or_default();
    let line = Line::from(vec![
        Span::styled(" ✓ ", theme::brand_green().add_modifier(Modifier::BOLD)),
        Span::styled(model, theme::text().add_modifier(Modifier::BOLD)),
        Span::styled(
            format!(
                " · loaded in {:.1}s · listening ",
                app.progress.ready_in_secs
            ),
            theme::text2(),
        ),
        Span::styled(
            format!("{}:{}", app.args.bind, app.args.port),
            theme::brand_green(),
        ),
    ]);
    f.render_widget(Paragraph::new(line).block(block), area);
}

fn draw_chips(f: &mut Frame, app: &App, area: Rect) {
    let chips = logo::badges(&app.args);
    let mut spans: Vec<Span> = Vec::new();
    for b in &chips {
        let tint = match b.tint {
            logo::BadgeTint::Model => theme::brand_purple(),
            logo::BadgeTint::Quant => theme::brand_cyan(),
            logo::BadgeTint::Role => theme::brand_green(),
            logo::BadgeTint::Neutral => theme::text2(),
        };
        spans.push(Span::styled(
            "▐",
            Style::default().fg(theme::BG_RAISED.color()),
        ));
        // First word tinted, rest secondary — restraint per the spec.
        let mut words = b.text.splitn(2, ' ');
        let first = words.next().unwrap_or_default().to_string();
        let rest = words.next().map(|r| format!(" {r}")).unwrap_or_default();
        spans.push(Span::styled(first, tint.bg(theme::BG_RAISED.color())));
        spans.push(Span::styled(
            rest,
            theme::text2().bg(theme::BG_RAISED.color()),
        ));
        spans.push(Span::styled(
            "▌",
            Style::default().fg(theme::BG_RAISED.color()),
        ));
        spans.push(Span::raw(" "));
    }
    f.render_widget(Paragraph::new(Line::from(spans)).wrapped(), area);
}

fn draw_logs(f: &mut Frame, app: &App, area: Rect) {
    let follow = app.log_scroll.is_none();
    let right = if follow {
        "⏵ follow".to_string()
    } else {
        format!("⏸ {}↑", app.log_scroll.unwrap_or(0))
    };
    let filter_part = if app.log_filter_editing {
        format!(" filter: {}▏", app.log_filter)
    } else if !app.log_filter.is_empty() {
        format!(" filter: {}", app.log_filter)
    } else {
        String::new()
    };
    let block = panel(format!("LOGS ── {right}{filter_part} ─"), follow);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let want = inner.height as usize + app.log_scroll.unwrap_or(0);
    let mut lines: Vec<Line> = log_ring::tail(want.max(64))
        .into_iter()
        .filter(|l| {
            app.log_filter.is_empty()
                || l.message
                    .to_lowercase()
                    .contains(&app.log_filter.to_lowercase())
                || l.target.contains(&app.log_filter)
        })
        .map(|l| {
            let ts = chrono_lite(l.at);
            let target_short: String = l
                .target
                .split("::")
                .next()
                .unwrap_or("")
                .chars()
                .take(14)
                .collect();
            Line::from(vec![
                Span::styled(format!("{ts} "), theme::dim()),
                Span::styled(format!("{:<5} ", l.level), theme::level_style(l.level)),
                Span::styled(format!("{target_short:<14} "), theme::dim()),
                Span::styled(l.message, theme::text()),
            ])
        })
        .collect();
    // Apply scroll: drop from the end when scrolled up.
    if let Some(up) = app.log_scroll {
        let keep = lines.len().saturating_sub(up);
        lines.truncate(keep);
    }
    let visible = lines.len().saturating_sub(inner.height as usize);
    let shown: Vec<Line> = lines.into_iter().skip(visible).collect();
    f.render_widget(Paragraph::new(shown), inner);
}

pub fn draw_kernels(f: &mut Frame, app: &App, area: Rect) {
    let Some(model) = &app.kernels else {
        let block = panel("KERNELS ─ waiting for startup ─".into(), false);
        f.render_widget(
            Paragraph::new(Span::styled(
                "  kernel audit runs at model load…",
                theme::dim(),
            ))
            .block(block),
            area,
        );
        return;
    };
    let mut constraints = vec![Constraint::Min(6)];
    if !model.missing.is_empty() {
        constraints.insert(0, Constraint::Length(model.missing.len().min(6) as u16 + 2));
    }
    let rows_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);
    let mut idx = 0;
    if !model.missing.is_empty() {
        let block = panel(
            format!("⚠ {} UNRESOLVED KERNEL LOOKUP(S) ─", model.missing.len()),
            false,
        )
        .border_style(theme::warn());
        let lines: Vec<Line> = model
            .missing
            .iter()
            .take(6)
            .map(|m| {
                Line::from(Span::styled(
                    format!("  {}::{}", m.module, m.func),
                    theme::warn(),
                ))
            })
            .collect();
        f.render_widget(Paragraph::new(lines).block(block), rows_layout[idx]);
        idx += 1;
    }
    let filtered: Vec<_> = model
        .rows
        .iter()
        .filter(|r| app.kernel_filter.is_empty() || r.module.contains(&app.kernel_filter))
        .collect();
    let title = format!("KERNELS ─ {} modules ─", filtered.len());
    let header = Row::new(vec!["MODULE", "PTX-HASH", "RESOLUTION"])
        .style(theme::text2().add_modifier(Modifier::BOLD));
    let table_rows: Vec<Row> = filtered
        .iter()
        .skip(app.kernel_scroll)
        .map(|r| {
            let (res, style) = match r.resolution {
                Some(true) => ("used", theme::brand_cyan()),
                Some(false) => ("** lookup FAILED **", theme::error()),
                None => ("-", theme::dim()),
            };
            let row_style = if r.resolution.is_none() {
                theme::dim()
            } else {
                theme::text()
            };
            Row::new(vec![
                Span::styled(r.module.clone(), row_style),
                Span::styled(r.ptx_hash.clone(), theme::dim()),
                Span::styled(res, style),
            ])
        })
        .collect();
    let table = Table::new(
        table_rows,
        [
            Constraint::Length(34),
            Constraint::Length(14),
            Constraint::Min(10),
        ],
    )
    .header(header)
    .block(panel(title, true));
    f.render_widget(table, rows_layout[idx]);
}

fn middle_truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let half = (max.saturating_sub(1)) / 2;
    let start: String = s.chars().take(half).collect();
    let end: String = s
        .chars()
        .rev()
        .take(half)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{start}…{end}")
}

/// hh:mm:ss from a SystemTime without a chrono dependency.
fn chrono_lite(t: std::time::SystemTime) -> String {
    let secs = t
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    format!("{h:02}:{m:02}:{s:02}")
}

trait Wrapped {
    fn wrapped(self) -> Self;
}
impl Wrapped for Paragraph<'_> {
    fn wrapped(self) -> Self {
        self.wrap(ratatui::widgets::Wrap { trim: false })
    }
}
