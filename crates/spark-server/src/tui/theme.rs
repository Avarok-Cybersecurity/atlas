// SPDX-License-Identifier: AGPL-3.0-only

//! The Atlas TUI design system.
//!
//! Three brand chevron colors with fixed semantic momentum roles:
//! purple = identity/selection, cyan = activity/focus, green = success/ready.
//! Truecolor by default with pinned 256-color fallbacks; the terminal's own
//! ANSI 0-15 palette is never used, and the app paints its own surfaces so
//! contrast is controlled everywhere.

use ratatui::style::{Color, Modifier, Style};

/// How much color this terminal is to be given.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Depth {
    /// The user asked for none. Every signal this palette carries in hue has
    /// to be carried by a modifier instead, or it is simply gone.
    None,
    /// The pinned 256-color fallback indices.
    Ansi256,
    /// 24-bit.
    True,
}

/// Resolve the depth from the two environment variables that decide it, as
/// values rather than by reading the environment — so the precedence is
/// testable without a process to set variables in.
///
/// ★ **`NO_COLOR` outranks `COLORTERM`, and it is not a boolean.** Per
/// <https://no-color.org> the variable counts when it is present and non-empty,
/// *whatever* it is set to — so `NO_COLOR=0` means no color, and only
/// `NO_COLOR=` (empty) is ignored. Parsing it as a flag is the standard way to
/// get this wrong. The precedence is the substantive half: `COLORTERM` says
/// what the terminal CAN do and `NO_COLOR` says what the user WANTS, and a
/// capability must never overrule a preference.
pub fn depth_of(no_color: Option<&str>, colorterm: Option<&str>) -> Depth {
    if no_color.is_some_and(|v| !v.is_empty()) {
        return Depth::None;
    }
    match colorterm {
        Some(v) if v.contains("truecolor") || v.contains("24bit") => Depth::True,
        _ => Depth::Ansi256,
    }
}

/// This process's color depth. Read live rather than cached, because the
/// palette has always been read live and a `OnceLock` here would let whichever
/// test ran first decide the answer for the rest of the binary.
pub fn depth() -> Depth {
    depth_of(
        std::env::var("NO_COLOR").ok().as_deref(),
        std::env::var("COLORTERM").ok().as_deref(),
    )
}

/// Whether the terminal advertises 24-bit color.
fn truecolor() -> bool {
    depth() == Depth::True
}

/// A themed color: truecolor value + 256-palette fallback index.
#[derive(Clone, Copy)]
pub struct C(pub u8, pub u8, pub u8, pub u8);

impl C {
    pub fn color(self) -> Color {
        match depth() {
            // `Reset` rather than a black/white guess: the terminal's own
            // default is the only thing guaranteed to be legible against the
            // background the user actually chose.
            Depth::None => Color::Reset,
            Depth::Ansi256 => Color::Indexed(self.3),
            Depth::True => Color::Rgb(self.0, self.1, self.2),
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
    let s = Style::default().fg(WARN.color());
    // `error()` is bold in every mode, so without this a warning and an info
    // line are the same glyphs in the same weight once the hue is gone — and
    // the level ramp is the only thing that says which is which.
    if depth() == Depth::None {
        s.add_modifier(Modifier::BOLD)
    } else {
        s
    }
}

/// The style that says "this row is the one you are on".
///
/// ★ The six render modules that draw a selectable list each wrote
/// `Style::default().bg(theme::BG_SELECTION.color())` inline. That is fine
/// until the background is `Color::Reset` — under `NO_COLOR` the selected row
/// becomes indistinguishable from every other row, and a list you cannot see
/// your position in is not usable at all. Reverse video carries the same
/// meaning with no color, and there is now one place that decides.
pub fn selected() -> Style {
    if depth() == Depth::None {
        Style::default().add_modifier(Modifier::REVERSED)
    } else {
        Style::default().bg(BG_SELECTION.color())
    }
}
pub fn error() -> Style {
    Style::default()
        .fg(ERROR.color())
        .add_modifier(Modifier::BOLD)
}

/// Panel border style; `focused` flips it to brand cyan (color is the focus
/// signal — same glyph weight everywhere).
pub fn border(focused: bool) -> Style {
    if !focused {
        return Style::default().fg(BORDER_DIM.color());
    }
    // Focus is a colour signal by design ("same glyph weight everywhere").
    // With no colour there is no signal left, so it borrows weight instead.
    if depth() == Depth::None {
        Style::default().add_modifier(Modifier::BOLD)
    } else {
        brand_cyan()
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
    // The bar still fills; it just fills in the terminal's own ink.
    if depth() == Depth::None {
        return Color::Reset;
    }
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

/// The benchmark run-glow: brand cyan pulsing between dim and full over ~1.6 s
/// at the 10 Hz tick.
///
/// Motion is the signal that work is in flight, so it is restrained the same
/// way the loading chevron wave is — one slow sine, no hue shift. In
/// 256-color mode there is no room to interpolate, so it alternates between the
/// cyan index and the dim border index at the same cadence.
pub fn glow(tick: u64) -> Color {
    // A pulse is a hue animation and nothing else, so with no colour there is
    // nothing to pulse — a steady default beats a ring that flickers between
    // two identical shades of nothing.
    if depth() == Depth::None {
        return Color::Reset;
    }
    let phase = (tick % 16) as f64 / 16.0;
    // 0 -> 1 -> 0 over the period, never fully dark: the ring stays legible.
    let t = 0.35 + 0.65 * (1.0 - (phase * std::f64::consts::TAU).cos()) / 2.0;
    if !truecolor() {
        return if t > 0.6 {
            Color::Indexed(CYAN.3)
        } else {
            Color::Indexed(BORDER_DIM.3)
        };
    }
    let mix = |lit: u8, dark: u8| (dark as f64 + (lit as f64 - dark as f64) * t).round() as u8;
    Color::Rgb(
        mix(CYAN.0, BORDER_DIM.0),
        mix(CYAN.1, BORDER_DIM.1),
        mix(CYAN.2, BORDER_DIM.2),
    )
}

/// Map a benchmark's semantic cell style onto the palette. The single place
/// where `atlas-plugin`'s style-free results acquire color.
pub fn cell_style(style: atlas_plugin::CellStyle) -> Style {
    use atlas_plugin::CellStyle as S;
    match style {
        S::Neutral => text(),
        S::Dim => dim(),
        S::Accent => brand_cyan(),
        S::Good => brand_green(),
        S::Warn => warn(),
        S::Bad => error(),
    }
}

/// Braille spinner frames (1 rev/s at the 10 Hz tick).
pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[cfg(test)]
#[path = "theme_tests.rs"]
mod tests;
