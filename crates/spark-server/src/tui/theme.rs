// SPDX-License-Identifier: AGPL-3.0-only

//! The Atlas TUI design system.
//!
//! Three brand chevron colors with fixed semantic momentum roles:
//! purple = identity/selection, cyan = activity/focus, green = success/ready.
//! Truecolor by default with pinned 256-color fallbacks; the terminal's own
//! ANSI 0-15 palette is never used, and the app paints its own surfaces so
//! contrast is controlled everywhere.

use ratatui::style::{Color, Modifier, Style};

/// Whether the terminal advertises 24-bit color.
fn truecolor() -> bool {
    std::env::var("COLORTERM")
        .map(|v| v.contains("truecolor") || v.contains("24bit"))
        .unwrap_or(false)
}

/// A themed color: truecolor value + 256-palette fallback index.
#[derive(Clone, Copy)]
pub struct C(pub u8, pub u8, pub u8, pub u8);

impl C {
    pub fn color(self) -> Color {
        if truecolor() {
            Color::Rgb(self.0, self.1, self.2)
        } else {
            Color::Indexed(self.3)
        }
    }
}

// ── Brand ──
pub const PURPLE: C = C(0xBE, 0x9D, 0xF8, 141);
pub const CYAN: C = C(0x49, 0xC3, 0xDB, 80);
pub const GREEN: C = C(0x12, 0xB9, 0x81, 36);
// ── Surfaces ──
pub const BG_BASE: C = C(0x0F, 0x11, 0x17, 232);
pub const BG_PANEL: C = C(0x15, 0x18, 0x23, 233);
pub const BG_RAISED: C = C(0x1E, 0x22, 0x30, 235);
pub const BG_SELECTION: C = C(0x2B, 0x26, 0x40, 237);
// ── Lines & text ──
pub const BORDER_DIM: C = C(0x2A, 0x2F, 0x3F, 237);
pub const TEXT: C = C(0xE6, 0xE9, 0xF0, 254);
pub const TEXT_2: C = C(0x93, 0x97, 0xA0, 246);
pub const TEXT_DIM: C = C(0x56, 0x5B, 0x68, 240);
// ── Status ──
pub const WARN: C = C(0xE5, 0xC0, 0x7B, 179);
pub const ERROR: C = C(0xF7, 0x76, 0x8E, 204);
pub const GAUGE_TRACK: C = C(0x25, 0x2A, 0x38, 236);

pub fn text() -> Style {
    Style::default().fg(TEXT.color())
}
pub fn text2() -> Style {
    Style::default().fg(TEXT_2.color())
}
pub fn dim() -> Style {
    Style::default().fg(TEXT_DIM.color())
}
pub fn brand_purple() -> Style {
    Style::default().fg(PURPLE.color())
}
pub fn brand_cyan() -> Style {
    Style::default().fg(CYAN.color())
}
pub fn brand_green() -> Style {
    Style::default().fg(GREEN.color())
}
pub fn warn() -> Style {
    Style::default().fg(WARN.color())
}
pub fn error() -> Style {
    Style::default()
        .fg(ERROR.color())
        .add_modifier(Modifier::BOLD)
}

/// Panel border style; `focused` flips it to brand cyan (color is the focus
/// signal — same glyph weight everywhere).
pub fn border(focused: bool) -> Style {
    if focused {
        brand_cyan()
    } else {
        Style::default().fg(BORDER_DIM.color())
    }
}

/// Panel title style (CAPS text; bold cyan when the panel has focus).
pub fn title(focused: bool) -> Style {
    if focused {
        brand_cyan().add_modifier(Modifier::BOLD)
    } else {
        text2()
    }
}

/// Log level color, per the spec's level ramp.
pub fn level_style(level: tracing::Level) -> Style {
    match level {
        tracing::Level::ERROR => error(),
        tracing::Level::WARN => warn(),
        tracing::Level::INFO => brand_cyan(),
        _ => dim(),
    }
}

/// Interpolate the signature progress gradient at `t ∈ [0,1]`:
/// purple → cyan on [0,0.5), cyan → green on [0.5,1]. In 256-color mode,
/// three hard bands (the fallback indices).
pub fn gradient_at(t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    if !truecolor() {
        return if t < 0.34 {
            Color::Indexed(PURPLE.3)
        } else if t < 0.67 {
            Color::Indexed(CYAN.3)
        } else {
            Color::Indexed(GREEN.3)
        };
    }
    let lerp = |a: u8, b: u8, f: f64| (a as f64 + (b as f64 - a as f64) * f).round() as u8;
    let (from, to, f) = if t < 0.5 {
        (PURPLE, CYAN, t * 2.0)
    } else {
        (CYAN, GREEN, (t - 0.5) * 2.0)
    };
    Color::Rgb(
        lerp(from.0, to.0, f),
        lerp(from.1, to.1, f),
        lerp(from.2, to.2, f),
    )
}

/// Gauge fill color override when nearly full: ≥97% error, ≥90% warn.
pub fn pressure_color(frac: f64) -> Option<Color> {
    if frac >= 0.97 {
        Some(ERROR.color())
    } else if frac >= 0.90 {
        Some(WARN.color())
    } else {
        None
    }
}

/// Braille spinner frames (1 rev/s at the 10 Hz tick).
pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
