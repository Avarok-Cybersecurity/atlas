// SPDX-License-Identifier: AGPL-3.0-only

//! Server Stats: tile row (requests / throughput / TTFT / GPU), TTFT
//! histogram, throughput chart, sequences & memory gauges, speculation &
//! cache panel.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, BarChart, Chart, Dataset, Paragraph, Sparkline};

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
    super::stats_tab_panels::draw_sequences(f, app, bottom[0]);
    // Thermal sits under speculation & cache rather than taking a column of its
    // own: it is a small fixed-height panel, and splitting the bottom row three
    // ways would squeeze the two that carry per-sequence detail.
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(6)])
        .split(bottom[1]);
    super::stats_tab_panels::draw_spec_cache(f, app, right[0]);
    super::stats_thermal::draw(f, app, right[1]);
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
        // Unequal on purpose: five EQUAL tiles truncated the REQUESTS row
        // ("1007 ● 8 ↓2.0 KB/s ↑3.0 MB/s" is ~32 cols and the widest content
        // here), which the tile test catches. Widths follow the content —
        // REQUESTS and GPU carry two figures each, THROUGHPUT carries one.
        .constraints([
            Constraint::Percentage(24),
            Constraint::Percentage(16),
            Constraint::Percentage(18),
            Constraint::Percentage(20),
            Constraint::Percentage(22),
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
                "  ↓{} ↑{}",
                crate::tui::format::rate(s.bytes_in_rate),
                crate::tui::format::rate(s.bytes_out_rate)
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
    // Prefill ingest. Counted per chunk as it lands (see
    // scheduler/phase_continue_prefills), so this is the rate at the moment
    // ingest happens — not, as it used to be, the whole prompt credited to
    // whenever the response finished. `● n` is the number of sequences
    // currently prefilling, straight off the scheduler snapshot.
    let pf = Line::from(vec![
        Span::styled(
            format!(" {:.0} tok/s", s.prompt_tps),
            theme::text().add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            match s.sched {
                Some(x) if x.prefilling_seqs > 0 => format!("  ● {}", x.prefilling_seqs),
                _ => String::new(),
            },
            theme::brand_cyan(),
        ),
        // In-flight progress as a PERCENT: this tile is the narrowest on the
        // row (18%) and the absolute pair "4.2k/12.6k" truncates in it. The
        // exact numbers go to the sequences pane, which has the width.
        // Rendered only while something is prefilling — a permanent 0% would
        // be noise on an idle server.
        Span::styled(
            match s.sched {
                Some(x) if x.prefill_tokens_total > 0 => format!(
                    "  {:.0}%",
                    100.0 * x.prefill_tokens_done as f64 / x.prefill_tokens_total as f64
                ),
                _ => String::new(),
            },
            theme::text2(),
        ),
    ]);
    tile(
        f,
        tiles[2],
        "PREFILL",
        pf,
        Some(&s.prompt_tps_history.as_u64()),
    );
    let ttft = Line::from(vec![
        Span::styled(
            format!(" p50 {}", fmt_ms(s.ttft_p50_ms)),
            theme::text().add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("  p90 {}", fmt_ms(s.ttft_p90_ms)), theme::text2()),
    ]);
    tile(f, tiles[3], "TTFT", ttft, None);
    // `—`, not 0.0, when the device never answered.
    let gpu = if s.gpu_known {
        Line::from(vec![
            Span::styled(
                format!(" atlas {:.1} GB", s.atlas_used_gb),
                theme::text().add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  free {:.1}", s.gpu_free_gb), theme::text2()),
        ])
    } else {
        Line::from(Span::styled(" —", theme::dim()))
    };
    tile(f, tiles[4], "GPU", gpu, None);
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
    let block = panel("THROUGHPUT ── gen ─ prefill ─".into(), false);
    let pts: Vec<(f64, f64)> = app
        .stats
        .gen_tps_history
        .points
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64, *v))
        .collect();
    let max_y = pts.iter().map(|(_, v)| *v).fold(10.0_f64, f64::max) * 1.15;

    // Prefill on the SAME plot but its OWN scale, printed up the right edge.
    // The two series differ by ~20x (≈890 vs ≈40 tok/s), so one shared axis
    // buries the generation line on the baseline. Instead the prefill series
    // is normalised onto the generation axis and the right-hand labels say
    // what full-scale means for it — a dual-axis chart, which ratatui's
    // `Chart` has no native support for, so the right scale is drawn by hand
    // below.
    let pf_raw: Vec<f64> = app
        .stats
        .prompt_tps_history
        .points
        .iter()
        .copied()
        .collect();
    let pf_max = pf_raw.iter().copied().fold(0.0_f64, f64::max);
    let pf_scale = if pf_max > 0.0 {
        max_y / (pf_max * 1.15)
    } else {
        0.0
    };
    let pf_pts: Vec<(f64, f64)> = pf_raw
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64, v * pf_scale))
        .collect();

    let mut datasets = vec![
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(ratatui::widgets::GraphType::Line)
            .style(theme::brand_cyan())
            .data(&pts),
    ];
    if pf_max > 0.0 {
        datasets.push(
            Dataset::default()
                .marker(symbols::Marker::Braille)
                .graph_type(ratatui::widgets::GraphType::Line)
                .style(theme::brand_purple())
                .data(&pf_pts),
        );
    }
    let caption = format!(
        "gen {:.0} tok/s · prefill {:.0} tok/s",
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

    // Right-hand scale for the prefill series, in the prefill colour so it
    // reads as "the purple line tops out here". Only drawn once prefill has
    // been observed — an idle server gets the single-scale chart it had.
    // Placed inside the block border, right-aligned, and only when the pane
    // is tall/wide enough to hold it without colliding with the plot.
    if pf_max > 0.0 && area.height >= 4 && area.width >= 24 {
        let label = format!("{:.0} ", pf_max * 1.15);
        let w = label.len() as u16;
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(label, theme::brand_purple()))),
            Rect {
                x: area.right().saturating_sub(w + 1),
                y: area.y + 1,
                width: w,
                height: 1,
            },
        );
    }
}

/// `12618` -> `12.6k`. The PREFILL tile is the narrowest on the row, so the
/// progress pair has to stay short enough not to clip it.
fn compact_tokens(n: u32) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

fn fmt_ms(v: Option<f64>) -> String {
    match v {
        Some(ms) if ms >= 1000.0 => format!("{:.1}s", ms / 1000.0),
        Some(ms) => format!("{ms:.0}ms"),
        None => "—".into(),
    }
}

// `human_bytes` was here: a private `K`/`M`/`B` ladder that named a magnitude
// and no unit. It is `crate::tui::format::rate` now, with the download row —
// see that function for why one formatter and why 1024.

#[cfg(test)]
#[path = "stats_tab_tests.rs"]
mod tests;
