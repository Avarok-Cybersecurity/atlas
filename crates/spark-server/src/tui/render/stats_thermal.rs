// SPDX-License-Identifier: AGPL-3.0-only

//! Stats subsection: thermal state and how hard the throttles are working.
//!
//! Split out of `stats_tab.rs` for the repo's 500-line cap.
//!
//! # What it answers
//!
//! "Is this box being held back right now, and is it steady about it?" A single
//! cumulative counter cannot answer either — see
//! `atlas_plugin::hardware::throttle_monitor`. This panel shows the differenced
//! fraction, the flip count that separates a steady hold from unstable clocks,
//! and the clock-versus-maximum ratio that makes the consequence concrete.
//!
//! # Unknown is rendered, not hidden
//!
//! Every field distinguishes "not known" from "fine". A thermal panel that
//! shows healthy defaults for a box it cannot read is how a throttled machine
//! gets mistaken for a slow model.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{LineGauge, Paragraph};

use super::panel;
use crate::tui::app::App;
use crate::tui::data::thermal::{THRASH_TRANSITIONS, ThermalAlert, ThermalSnapshot};
use crate::tui::theme;

pub(super) fn draw(f: &mut Frame, app: &App, area: Rect) {
    let snap = app.thermal.snapshot();
    let title = match snap.alert() {
        ThermalAlert::Thrashing => "THERMAL \u{26a0} THRASHING ─",
        ThermalAlert::Throttling => "THERMAL \u{26a0} THROTTLING ─",
        _ => "THERMAL ─",
    };
    let block = panel(title.to_string(), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height < 4 {
        return;
    }
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // temp · clock
            Constraint::Length(1), // throttled gauge
            Constraint::Length(1), // stability
            Constraint::Min(1),    // power cap note
        ])
        .split(inner);

    f.render_widget(Paragraph::new(temp_line(&snap)), rows[0]);
    render_throttle_gauge(f, rows[1], &snap);
    f.render_widget(Paragraph::new(stability_line(&snap)), rows[2]);
    f.render_widget(Paragraph::new(power_cap_line(&snap)), rows[3]);
}

fn unknown() -> Span<'static> {
    Span::styled("\u{2014}", theme::dim())
}

fn temp_line(s: &ThermalSnapshot) -> Line<'static> {
    let mut out = vec![Span::styled("temp ", theme::text2())];
    match s.gpu_temp_c {
        Some(c) => out.push(Span::styled(format!("{c:.0}\u{b0}C"), theme::text())),
        None => out.push(unknown()),
    }
    out.push(Span::styled("   clock ", theme::text2()));
    match (s.sm_clock_mhz, s.clock_frac()) {
        (Some(mhz), Some(frac)) => {
            // Colour on the RATIO, not the absolute: 1.4 GHz means nothing
            // without the part's ceiling, and the ceiling differs per SKU.
            let style = if frac < 0.7 {
                theme::warn()
            } else {
                theme::text()
            };
            out.push(Span::styled(format!("{mhz:.0} MHz"), style));
            out.push(Span::styled(
                format!(" ({:.0}% of max)", frac * 100.0),
                theme::text2(),
            ));
        }
        (Some(mhz), None) => out.push(Span::styled(format!("{mhz:.0} MHz"), theme::text())),
        _ => out.push(unknown()),
    }
    Line::from(out)
}

fn render_throttle_gauge(f: &mut Frame, area: Rect, s: &ThermalSnapshot) {
    let frac = s.window.and_then(|w| w.thermal_frac);
    let Some(frac) = frac else {
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("throttled ", theme::text2()),
                unknown(),
                Span::styled("  (no counters on this box)", theme::dim()),
            ])),
            area,
        );
        return;
    };
    let style = if frac >= 0.5 {
        theme::error()
    } else if frac >= 0.2 {
        theme::warn()
    } else {
        theme::brand_green()
    };
    f.render_widget(
        LineGauge::default()
            .label(Span::styled(
                format!("throttled {:>5.1}%", frac * 100.0),
                style,
            ))
            .ratio(frac)
            .filled_style(style)
            .unfilled_style(theme::dim()),
        area,
    );
}

/// The flip count, which is the part a fraction alone cannot tell you.
fn stability_line(s: &ThermalSnapshot) -> Line<'static> {
    let mut out = vec![Span::styled("stability ", theme::text2())];
    if s.samples == 0 {
        out.push(unknown());
        out.push(Span::styled("  (warming up)", theme::dim()));
        return Line::from(out);
    }
    let style = if s.transitions >= THRASH_TRANSITIONS {
        theme::error()
    } else {
        theme::text()
    };
    out.push(Span::styled(format!("{} flips", s.transitions), style));
    // The denominator matters: "0 flips over 1 sample" says nothing, and a
    // reader cannot tell the two apart from the numerator alone.
    out.push(Span::styled(
        format!(" over {} samples", s.samples),
        theme::text2(),
    ));
    if s.transitions >= THRASH_TRANSITIONS {
        out.push(Span::styled(
            "  clocks unsettled \u{2014} expect p90 drift",
            theme::error(),
        ));
    }
    Line::from(out)
}

/// SW power cap, stated as normal rather than folded into the verdict.
fn power_cap_line(s: &ThermalSnapshot) -> Line<'static> {
    let mut out = vec![Span::styled("power cap ", theme::text2())];
    match s.window.and_then(|w| w.power_cap_frac) {
        Some(f) => {
            out.push(Span::styled(format!("{:.0}%", f * 100.0), theme::text()));
            // Measured on a HEALTHY GB10: asserted 16,130 s of an 11.2-day
            // uptime. Saying so inline stops a permanently-high number here
            // from being read as a fault.
            out.push(Span::styled(
                "  (normal on a power-limited part)",
                theme::dim(),
            ));
        }
        None => out.push(unknown()),
    }
    Line::from(out)
}
