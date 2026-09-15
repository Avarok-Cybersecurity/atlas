// SPDX-License-Identifier: AGPL-3.0-only

//! Stats-tab lower panels: SEQUENCES & MEMORY, SPECULATION & CACHE, and the
//! `line_gauge` bar they share.
//!
//! Split out of `stats_tab.rs` when the PREFILL ingest tile and the
//! dual-scale throughput chart pushed that file past the repo's 500-LoC cap.
//! Follows the existing `main_tab` / `main_tab_kernels` sibling idiom rather
//! than introducing a directory. The cut is at a real seam: everything here
//! renders the BOTTOM half of the tab and nothing above it calls in, so the
//! two halves share only `App`, the `panel` chrome, and `fmt_ms`.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{LineGauge, Paragraph, Sparkline};

use super::panel;
use crate::tui::app::App;
use crate::tui::theme;

pub(super) fn line_gauge(
    f: &mut Frame,
    area: Rect,
    label: &str,
    used: f64,
    total: f64,
    gradient: bool,
) {
    let frac = if total > 0.0 {
        (used / total).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let color = theme::pressure_color(frac).unwrap_or(if gradient {
        theme::gradient_at(frac)
    } else {
        theme::CYAN.color()
    });
    let g = LineGauge::default()
        .ratio(frac)
        .filled_style(Style::default().fg(color))
        .unfilled_style(Style::default().fg(theme::GAUGE_TRACK.color()))
        .label(Span::styled(format!("{label:<4}"), theme::dim()));
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(10), Constraint::Length(16)])
        .split(area);
    f.render_widget(g, cols[0]);
    f.render_widget(
        Paragraph::new(Span::styled(
            format!("{used:.0}/{total:.0}"),
            theme::text2(),
        )),
        cols[1],
    );
}

pub(super) fn draw_sequences(f: &mut Frame, app: &App, area: Rect) {
    let block = panel("SEQUENCES & MEMORY ─".into(), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let s = &app.stats;
    let (active, prefill, swapped, queue) = s
        .sched
        .map(|x| {
            (
                x.active_seqs,
                x.prefilling_seqs,
                x.swapped_seqs,
                x.pending_len,
            )
        })
        .unwrap_or_default();
    // Every row below is placed by hand rather than by a `Layout`, so every
    // row has to be checked against the pane it is meant to be inside: a
    // `Rect` one line past the bottom is not clipped by ratatui, it panics —
    // and this pane is six rows tall on a terminal that is only eight, so the
    // dashboard (and with it the server's foreground) went down on a resize.
    let row = |y: u16| -> Option<Rect> {
        (y < inner.bottom()).then_some(Rect {
            y,
            height: 1,
            ..inner
        })
    };
    let mut y = inner.y;
    if let Some(r) = row(y) {
        f.render_widget(
            Paragraph::new(Line::from(vec![Span::styled(
                format!(
                    " active {active} · prefill {prefill} · swapped {swapped} · queue {queue}{} ",
                    // Last prefill dispatch width, and whether it took a
                    // fused/batched large-M arm. Without this you can only
                    // infer engagement from throughput.
                    match s.sched {
                        Some(x) if x.prefill_chunk_width > 0 => format!(
                            // Width + arm ONLY. The exact token pair was here
                            // too and pushed "fused" off the pane's right
                            // edge; progress is already on the PREFILL tile
                            // as a percent, whereas whether the fused arm
                            // engaged is not visible anywhere else.
                            " · M={}{}",
                            x.prefill_chunk_width,
                            if x.prefill_fused { " fused" } else { "" }
                        ),
                        _ => String::new(),
                    }
                ),
                theme::text(),
            )])),
            r,
        );
    }
    y += 1;
    let qh = s.queue_history.as_u64();
    if !qh.is_empty()
        && let Some(r) = row(y)
    {
        f.render_widget(
            Sparkline::default().data(&qh).style(theme::brand_cyan()),
            Rect {
                x: inner.x + 1,
                width: inner.width.saturating_sub(2),
                ..r
            },
        );
    }
    y += 2;
    if let Some(x) = s.sched {
        let used = (x.kv_blocks_total - x.kv_blocks_free) as f64;
        if let Some(r) = row(y) {
            line_gauge(f, r, " KV", used, x.kv_blocks_total as f64, true);
        }
        y += 1;
        if let Some(r) = row(y) {
            line_gauge(
                f,
                r,
                " SSM",
                x.ssm_slots_used as f64,
                x.ssm_slots_total as f64,
                false,
            );
        }
        y += 1;
    }
    if let Some(r) = row(y) {
        if s.gpu_known {
            line_gauge(
                f,
                r,
                " GPU",
                s.atlas_used_gb,
                s.gpu_total_gb.max(0.001),
                true,
            );
        } else {
            // A 0 % bar reads as "empty", which is a claim. Say nothing instead.
            f.render_widget(
                ratatui::widgets::Paragraph::new(Span::styled(" GPU  —", theme::dim())),
                r,
            );
        }
    }
    if let Some(r) = row(y + 1) {
        line_gauge(
            f,
            r,
            " RAM",
            (s.host_total_gb - s.host_avail_gb).max(0.0),
            s.host_total_gb.max(0.001),
            false,
        );
    }
}

pub(super) fn draw_spec_cache(f: &mut Frame, app: &App, area: Rect) {
    let block = panel("SPECULATION & CACHE ─".into(), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let s = &app.stats;
    let mut lines: Vec<Line> = Vec::new();
    if let Some(x) = s.sched {
        lines.push(Line::from(vec![
            Span::styled(
                format!(
                    " MTP gate {}",
                    crate::tui::format::mtp_mode_label(x.mtp_mode)
                ),
                theme::text(),
            ),
            Span::styled(
                format!(" · delivered {:.0} tok/s", x.delivered_tps),
                theme::text2(),
            ),
        ]));
    }
    for (k, accepted, total) in &s.spec_accept {
        if *total == 0 {
            continue;
        }
        let rate = *accepted as f64 / *total as f64;
        let w = 16usize;
        let filled = (rate * w as f64) as usize;
        let bar: String = "█".repeat(filled) + &"░".repeat(w - filled);
        lines.push(Line::from(vec![
            Span::styled(format!(" accept k={k:<3}"), theme::text2()),
            Span::styled(bar, theme::brand_cyan()),
            Span::styled(format!(" {:>3.0}%", rate * 100.0), theme::text()),
        ]));
    }
    lines.push(Line::default());
    let hit = s
        .prefix_hit_rate
        .map(|r| format!("{:.0}%", r * 100.0))
        .unwrap_or_else(|| "—".into());
    lines.push(Line::from(Span::styled(
        format!(" prefix-cache hit {hit} · {} tok warm", s.prefix_hit_tokens),
        theme::text(),
    )));
    lines.push(Line::from(Span::styled(
        format!(
            " tool calls {} · entropy {:.2}",
            s.tool_calls_total, s.entropy
        ),
        theme::text2(),
    )));
    f.render_widget(Paragraph::new(lines), inner);
    let eh = s.entropy_history.as_u64();
    if !eh.is_empty() && inner.height >= 6 {
        f.render_widget(
            Sparkline::default().data(&eh).style(theme::brand_cyan()),
            Rect {
                y: inner.y + inner.height - 1,
                height: 1,
                x: inner.x + 1,
                width: inner.width.saturating_sub(2),
            },
        );
    }
}
