// SPDX-License-Identifier: AGPL-3.0-only

//! Server Stats: tile row (requests / throughput / TTFT / GPU), TTFT
//! histogram, throughput chart, sequences & memory gauges, speculation &
//! cache panel.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, BarChart, Chart, Dataset, LineGauge, Paragraph, Sparkline};

use super::panel;
use crate::tui::app::App;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Percentage(45),
            Constraint::Min(8),
        ])
        .split(area);
    draw_tiles(f, app, rows[0]);
    let mid = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(rows[1]);
    draw_ttft_hist(f, app, mid[0]);
    draw_throughput(f, app, mid[1]);
    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
        .split(rows[2]);
    draw_sequences(f, app, bottom[0]);
    draw_spec_cache(f, app, bottom[1]);
}

fn tile(f: &mut Frame, area: Rect, title: &str, value: Line, spark: Option<&[u64]>) {
    let block = panel(format!("{title} ─"), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    f.render_widget(Paragraph::new(value), Rect { height: 1, ..inner });
    if let Some(data) = spark
        && inner.height >= 2
        && !data.is_empty()
    {
        f.render_widget(
            Sparkline::default().data(data).style(theme::brand_cyan()),
            Rect {
                y: inner.y + 1,
                height: 1,
                ..inner
            },
        );
    }
}

fn draw_tiles(f: &mut Frame, app: &App, area: Rect) {
    let s = &app.stats;
    let tiles = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);
    let req = Line::from(vec![
        Span::styled(
            format!(" {}", s.requests_total),
            theme::text().add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  ● {}", s.requests_active),
            if s.requests_active > 0 {
                theme::brand_green()
            } else {
                theme::dim()
            },
        ),
        Span::styled(
            format!(
                "  ↓{}/s ↑{}/s",
                human_bytes(s.bytes_in_rate),
                human_bytes(s.bytes_out_rate)
            ),
            theme::dim(),
        ),
    ]);
    tile(f, tiles[0], "REQUESTS", req, Some(&s.req_history.as_u64()));
    let tp = Line::from(Span::styled(
        format!(" {:.1} tok/s", s.gen_tps),
        theme::text().add_modifier(Modifier::BOLD),
    ));
    tile(
        f,
        tiles[1],
        "THROUGHPUT",
        tp,
        Some(&s.gen_tps_history.as_u64()),
    );
    let ttft = Line::from(vec![
        Span::styled(
            format!(" p50 {}", fmt_ms(s.ttft_p50_ms)),
            theme::text().add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("  p90 {}", fmt_ms(s.ttft_p90_ms)), theme::text2()),
    ]);
    tile(f, tiles[2], "TTFT", ttft, None);
    let gpu = Line::from(vec![
        Span::styled(
            format!(" atlas {:.1} GB", s.atlas_used_gb),
            theme::text().add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("  free {:.1}", s.gpu_free_gb), theme::text2()),
    ]);
    tile(f, tiles[3], "GPU", gpu, None);
}

fn draw_ttft_hist(f: &mut Frame, app: &App, area: Rect) {
    let block = panel("TTFT DISTRIBUTION ─".into(), false);
    // De-cumulate buckets into per-bucket counts; collapse the ≥2.5s tail.
    let mut bars: Vec<(String, u64, bool)> = Vec::new();
    let mut prev = 0u64;
    for (ub, cum) in &app.stats.ttft_buckets {
        if ub.is_infinite() {
            continue;
        }
        let count = cum.saturating_sub(prev);
        prev = *cum;
        let label = if *ub < 1.0 {
            format!(".{:02}", (ub * 100.0) as u32)
        } else {
            format!("{ub:.0}")
        };
        bars.push((label, count, *ub >= 2.5));
    }
    let data: Vec<ratatui::widgets::Bar> = bars
        .iter()
        .map(|(label, v, slow)| {
            ratatui::widgets::Bar::default()
                .value(*v)
                .label(Line::from(Span::styled(label.clone(), theme::dim())))
                .style(if *slow {
                    theme::warn()
                } else {
                    theme::brand_cyan()
                })
        })
        .collect();
    let chart = BarChart::default()
        .data(ratatui::widgets::BarGroup::default().bars(&data))
        .bar_width(3)
        .bar_gap(1)
        .block(block);
    f.render_widget(chart, area);
}

fn draw_throughput(f: &mut Frame, app: &App, area: Rect) {
    let block = panel("THROUGHPUT ── gen tok/s ─".into(), false);
    let pts: Vec<(f64, f64)> = app
        .stats
        .gen_tps_history
        .points
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64, *v))
        .collect();
    let max_y = pts.iter().map(|(_, v)| *v).fold(10.0_f64, f64::max) * 1.15;
    let datasets = vec![
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(ratatui::widgets::GraphType::Line)
            .style(theme::brand_cyan())
            .data(&pts),
    ];
    let caption = format!(
        "gen {:.0} tok/s · prompt {:.0} tok/s",
        app.stats.gen_tps, app.stats.prompt_tps
    );
    let chart = Chart::new(datasets)
        .x_axis(Axis::default().bounds([0.0, 120.0]))
        .y_axis(Axis::default().bounds([0.0, max_y]).labels(vec![
            Span::styled("0", theme::dim()),
            Span::styled(format!("{max_y:.0}"), theme::dim()),
        ]))
        .block(block.title_bottom(Line::from(Span::styled(caption, theme::text2()))));
    f.render_widget(chart, area);
}

fn line_gauge(f: &mut Frame, area: Rect, label: &str, used: f64, total: f64, gradient: bool) {
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

fn draw_sequences(f: &mut Frame, app: &App, area: Rect) {
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
    let mut y = inner.y;
    f.render_widget(
        Paragraph::new(Line::from(vec![Span::styled(
            format!(" active {active} · prefill {prefill} · swapped {swapped} · queue {queue} "),
            theme::text(),
        )])),
        Rect {
            y,
            height: 1,
            ..inner
        },
    );
    y += 1;
    let qh = s.queue_history.as_u64();
    if !qh.is_empty() {
        f.render_widget(
            Sparkline::default().data(&qh).style(theme::brand_cyan()),
            Rect {
                y,
                height: 1,
                x: inner.x + 1,
                width: inner.width.saturating_sub(2),
            },
        );
    }
    y += 2;
    if let Some(x) = s.sched {
        let used = (x.kv_blocks_total - x.kv_blocks_free) as f64;
        line_gauge(
            f,
            Rect {
                y,
                height: 1,
                ..inner
            },
            " KV",
            used,
            x.kv_blocks_total as f64,
            true,
        );
        y += 1;
        line_gauge(
            f,
            Rect {
                y,
                height: 1,
                ..inner
            },
            " SSM",
            x.ssm_slots_used as f64,
            x.ssm_slots_total as f64,
            false,
        );
        y += 1;
    }
    line_gauge(
        f,
        Rect {
            y,
            height: 1,
            ..inner
        },
        " GPU",
        s.atlas_used_gb,
        s.gpu_total_gb.max(0.001),
        true,
    );
    y += 1;
    line_gauge(
        f,
        Rect {
            y,
            height: 1,
            ..inner
        },
        " RAM",
        (s.host_total_gb - s.host_avail_gb).max(0.0),
        s.host_total_gb.max(0.001),
        false,
    );
}

fn draw_spec_cache(f: &mut Frame, app: &App, area: Rect) {
    let block = panel("SPECULATION & CACHE ─".into(), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let s = &app.stats;
    let mut lines: Vec<Line> = Vec::new();
    if let Some(x) = s.sched {
        lines.push(Line::from(vec![
            Span::styled(format!(" MTP gate {:?}", x.mtp_mode), theme::text()),
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

fn fmt_ms(v: Option<f64>) -> String {
    match v {
        Some(ms) if ms >= 1000.0 => format!("{:.1}s", ms / 1000.0),
        Some(ms) => format!("{ms:.0}ms"),
        None => "—".into(),
    }
}

fn human_bytes(rate: f64) -> String {
    if rate >= 1_048_576.0 {
        format!("{:.1}M", rate / 1_048_576.0)
    } else if rate >= 1024.0 {
        format!("{:.0}K", rate / 1024.0)
    } else {
        format!("{rate:.0}B")
    }
}
