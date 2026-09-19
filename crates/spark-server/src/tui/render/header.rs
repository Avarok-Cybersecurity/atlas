// SPDX-License-Identifier: AGPL-3.0-only

//! The sticky header: logo, status pill, and the mini-strip beside them.
//!
//! Split out of `render/mod.rs` when that file reached the 500-LoC cap. It is a
//! coherent unit on its own: these three are the only things that answer "what
//! is this server doing right now" above the fold, and they must agree with
//! each other — all three read `app.awaiting_model` rather than each deciding
//! for itself whether a model is loaded.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::live_model_name;
use crate::tui::app::App;
use crate::tui::{logo, theme};

pub(crate) fn status_pill(app: &App) -> Span<'static> {
    // Three states, not two: "loading" for a load that is not running reads as
    // a hang, which is exactly how a no-argument boot looked.
    let (label, bg) = if app.awaiting_model {
        (" ○ NO MODEL ", theme::TEXT_DIM)
    } else if app.progress.ready {
        (" ● SERVING ", theme::GREEN)
    } else {
        (" ● LOADING ", theme::WARN)
    };
    Span::styled(
        label,
        Style::default()
            .bg(bg.color())
            .fg(theme::BG_BASE.color())
            .add_modifier(Modifier::BOLD),
    )
}

/// The download chip: proof a transfer is alive, from ANY section.
///
/// ★ Why this exists. Every download signal used to live in the Library list —
/// the row mark, the glowing dot, the progress line. Switch to Main or Stats
/// and a 20 GB pull became completely invisible: reported as "I don't see ANY
/// indication of downloading except an 'x to cancel'-like tag at the bottom",
/// where that tag was the Library footer's STATIC hint, present whether or not
/// anything was running. So the one place that is on screen in every section
/// now carries it.
///
/// Returns `None` when nothing is downloading, so the header is byte-identical
/// to before in the common case.
///
/// Tiers by width rather than one truncated string: the fields degrade
/// right-to-left (rate, then name, then the number), because the glyph alone
/// still answers "is something moving" and that is the question this element
/// exists for.
pub(crate) fn download_chip(app: &App, width: u16) -> Option<Vec<Span<'static>>> {
    let job = app.download.job.as_ref()?;
    // The pulse is the liveness signal, and it is deliberately the SAME
    // divisor and phase as the list row's dot: two blinkers at different
    // cadences read as noise, one heartbeat reads as one system. At the 10 Hz
    // tick, /4 is a 0.8 s period — /2 strobes, /8 looks static on a glance.
    let pulse = if (app.tick / 4).is_multiple_of(2) {
        Modifier::BOLD
    } else {
        Modifier::DIM
    };
    // Stopping is not progress: the pulse stops so the state is legible
    // without reading the word, and under NO_COLOR that is the ONLY signal.
    let (glyph_style, steady) = if job.cancelling {
        (theme::warn(), true)
    } else {
        (theme::brand_cyan(), false)
    };
    let glyph = Span::styled(
        "\u{2193} ",
        if steady {
            glyph_style
        } else {
            glyph_style.add_modifier(pulse)
        },
    );
    if width < 48 {
        // Glyph only. Still pulsing, still positioned left of the pill: that
        // is enough to say "a transfer is running" without a number.
        return Some(vec![glyph, Span::raw(" ")]);
    }
    // `fraction()` is None when the Hub reported no sizes. Bytes moved is the
    // honest substitute — a percentage we cannot compute must never be faked,
    // and a bar stuck at zero was the original complaint.
    let measure = match job.fraction() {
        Some(f) => crate::tui::format::percent(f),
        None if job.done > 0 => crate::tui::format::bytes(job.done),
        None => "resolving\u{2026}".to_string(),
    };
    let mut out = vec![glyph];
    if job.cancelling {
        out.push(Span::styled("stopping", theme::warn()));
        out.push(Span::raw(" "));
        return Some(out);
    }
    if width >= 60 {
        // The repo tail, not the full id: the org prefix is the same for every
        // model a user pulls, so it spends columns without distinguishing.
        let tail = job.repo.rsplit('/').next().unwrap_or(&job.repo);
        let cap = if width >= 96 { 24 } else { 16 };
        let name: String = if tail.chars().count() > cap {
            tail.chars().take(cap - 1).collect::<String>() + "\u{2026}"
        } else {
            tail.to_string()
        };
        out.push(Span::styled(format!("{name} \u{b7} "), theme::text2()));
    }
    out.push(Span::styled(measure, theme::text2()));
    if width >= 96 && job.rate_bps > 0.0 {
        out.push(Span::styled(
            format!(" \u{b7} {}", crate::tui::format::rate(job.rate_bps)),
            theme::text2(),
        ));
    }
    out.push(Span::raw(" "));
    Some(out)
}

/// Header indicator for thermal throttling, or `None` when there is nothing to
/// say.
///
/// Deliberately silent in two cases that are NOT the same as each other but do
/// share a response: `Ok` (we have data and it is fine) and `Unknown` (no
/// reading, or a box with no throttle counters). Neither warrants pixels in a
/// header that is mostly status-quo, and an indicator that renders something
/// permanently is one people stop reading. The Stats section distinguishes them
/// in full.
///
/// THRASHING outranks THROTTLING: unstable clocks are the worse diagnosis even
/// at a modest throttle fraction, because the damage lands in latency VARIANCE —
/// p90 drifting away from a median that still looks healthy.
fn thermal_alert_span(app: &App) -> Option<Span<'static>> {
    use crate::tui::data::thermal::ThermalAlert;
    match app.thermal.snapshot().alert() {
        ThermalAlert::Unknown | ThermalAlert::Ok => None,
        ThermalAlert::Throttling => Some(Span::styled(
            " \u{26a0} THROTTLING ",
            theme::warn().add_modifier(ratatui::style::Modifier::BOLD),
        )),
        ThermalAlert::Thrashing => Some(Span::styled(
            " \u{26a0} THERMAL THRASH ",
            theme::error().add_modifier(ratatui::style::Modifier::BOLD),
        )),
    }
}

pub(crate) fn draw_header(f: &mut Frame, app: &App, area: Rect, tall: bool) {
    // Chevron wave only during loading (motion restraint).
    //
    // Same distinction as the status pill: `progress.ready` is the LISTENER,
    // and a settled logo next to "NO MODEL" reads as a finished load that has
    // not happened.
    let wave = if app.progress.ready && !app.awaiting_model {
        None
    } else {
        Some((app.tick / 3) as usize % 3)
    };
    let up = app.started.elapsed().as_secs();
    let uptime = fmt_uptime(up);
    // The chip sits LEFT of the pill, never in it: the pill answers "is the
    // server ok" and is reversed-bold for that reason; the chip answers "is my
    // transfer moving" at `text2` weight so a glance ranks the two correctly.
    let chip = download_chip(app, area.width);
    let mut right_spans = Vec::new();
    if !tall && let Some(c) = chip.as_ref() {
        right_spans.extend(c.iter().cloned());
        right_spans.push(Span::raw(" "));
    }
    if let Some(alert) = thermal_alert_span(app) {
        right_spans.push(alert);
        right_spans.push(Span::raw(" "));
    }
    right_spans.push(status_pill(app));
    right_spans.push(Span::styled(format!("  {uptime} "), theme::text2()));
    let right = Line::from(right_spans);
    if tall {
        let lines = logo::three_line(wave);
        for (i, line) in lines.into_iter().enumerate() {
            let row = Rect {
                y: area.y + i as u16,
                height: 1,
                ..area
            };
            f.render_widget(Paragraph::new(line), row);
        }
        // Right cluster row 0; model·quant·port row 1.
        f.render_widget(
            Paragraph::new(right).alignment(ratatui::layout::Alignment::Right),
            Rect {
                y: area.y,
                height: 1,
                ..area
            },
        );
        // The header's own mini-strip, and the one the user actually sees
        // first. Two things were wrong with it: with no model loaded it read
        // " · kv fp8 · :8123" — a KV dtype for a process that has loaded no KV
        // cache, from a clap default — and it read the BOOT argv, so after a
        // swap it went on describing the configuration the process started
        // with. `header_line` decides both, next to the chip strip's rule.
        let sub = Line::from(Span::styled(header_line(app), theme::text2()));
        f.render_widget(
            Paragraph::new(sub).alignment(ratatui::layout::Alignment::Right),
            Rect {
                y: area.y + 1,
                height: 1,
                ..area
            },
        );
        // Row 2 is the only right-hand row the tall header leaves empty (0 is
        // the pill cluster, 1 the mini-strip), so the chip lands there —
        // directly under the pill without sharing its row.
        if let Some(c) = chip {
            f.render_widget(
                Paragraph::new(Line::from(c)).alignment(ratatui::layout::Alignment::Right),
                Rect {
                    y: area.y + 2,
                    height: 1,
                    ..area
                },
            );
        }
    } else {
        f.render_widget(Paragraph::new(logo::one_line(wave)), area);
        f.render_widget(
            Paragraph::new(right).alignment(ratatui::layout::Alignment::Right),
            area,
        );
    }
}

/// The header's mini-strip: what is running, or the state and the way out.
///
/// Pure so it can be asserted on directly. Shares `app.awaiting_model` with the
/// status pill and the chip strip, so all three cannot disagree about whether a
/// model is loaded.
pub(crate) fn header_line(app: &App) -> String {
    if app.awaiting_model {
        // The port is the one claim still true with nothing loaded: the
        // listener binds before any model. The pill beside this already says
        // NO MODEL, so this does not repeat it.
        return format!("press 4 for Library · :{} ", app.args.port);
    }
    // The LIVE argv — a swap replaces it wholesale, and the boot value is not
    // what is serving afterwards.
    let live = app.host.as_ref().and_then(|h| h.args());
    let a = live.as_ref().unwrap_or(&app.args);
    format!(
        "{} · kv {} · :{} ",
        live_model_name(app),
        // An omitted --kv-cache-dtype resolves against MODEL.toml at load
        // time; the args argv cannot know the outcome, so label it "auto".
        a.kv_cache_dtype.as_deref().unwrap_or("auto"),
        a.port
    )
}

/// `up H:MM:SS`, or `up Nd HH:MM` past a day.
///
/// ★ This used to be `up {:02}:{:02}` over `up / 60 % 100` — minutes taken MOD
/// 100, with no hours at all. A server up 100 minutes displayed `up 00:xx` and
/// started counting again. This is the header of a dashboard meant to sit up
/// for days, so it is the most-looked-at number in the product.
pub(super) fn fmt_uptime(secs: u64) -> String {
    let (d, h, m, s) = (secs / 86_400, secs / 3_600 % 24, secs / 60 % 60, secs % 60);
    if d > 0 {
        format!("up {d}d {h:02}:{m:02}")
    } else {
        format!("up {h}:{m:02}:{s:02}")
    }
}

#[cfg(test)]
#[path = "header_tests.rs"]
mod tests;
