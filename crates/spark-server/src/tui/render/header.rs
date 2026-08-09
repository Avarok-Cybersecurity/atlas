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
    let right = Line::from(vec![
        status_pill(app),
        Span::styled(format!("  {uptime} "), theme::text2()),
    ]);
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
        a.kv_cache_dtype,
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
