// SPDX-License-Identifier: AGPL-3.0-only

//! Header logo art + CLI flag badge derivation.
//!
//! The logo reproduces assets/logo.svg's three chevrons as string constants —
//! purple, cyan, green, left to right. Two variants: a 1-line `❯❯❯ Atlas` for
//! short terminals and a 3-row half-block chevron for tall ones. During
//! LOADING a brightness wave walks the chevrons; it stops permanently once
//! SERVING (motion restraint per the design spec).

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::theme;

/// Rows of one half-block chevron cell. Three cells side by side, one column
/// gap, colored purple/cyan/green, read unmistakably as `>>>`.
pub const CHEVRON_ROWS: [&str; 3] = ["▀█▄ ", "  ██", "▄█▀ "];

/// 45%-luminance version of a brand color for the loading wave trough.
fn dimmed(c: theme::C) -> Color {
    let s = |v: u8| ((v as f64) * 0.45) as u8;
    match c.color() {
        Color::Rgb(r, g, b) => Color::Rgb(s(r), s(g), s(b)),
        other => other, // 256-color fallback: no dim variant, keep steady
    }
}

/// Per-chevron colors for animation step `wave` (None = steady/SERVING).
fn chevron_colors(wave: Option<usize>) -> [Color; 3] {
    let brand = [theme::PURPLE, theme::CYAN, theme::GREEN];
    match wave {
        None => brand.map(|c| c.color()),
        Some(step) => {
            let bright = step % 3;
            let mut out = [Color::Reset; 3];
            for (i, c) in brand.iter().enumerate() {
                out[i] = if i == bright { c.color() } else { dimmed(*c) };
            }
            out
        }
    }
}

/// The 1-line logo: `❯❯❯ Atlas`.
pub fn one_line(wave: Option<usize>) -> Line<'static> {
    let colors = chevron_colors(wave);
    let mut spans: Vec<Span> = colors
        .iter()
        .map(|c| Span::styled("❯", Style::default().fg(*c).add_modifier(Modifier::BOLD)))
        .collect();
    spans.push(Span::styled(
        " Atlas",
        theme::text().add_modifier(Modifier::BOLD),
    ));
    Line::from(spans)
}

/// The 3-row logo block with the wordmark. Returns exactly three lines.
pub fn three_line(wave: Option<usize>) -> [Line<'static>; 3] {
    let colors = chevron_colors(wave);
    let row = |r: usize, with_wordmark: Option<(&'static str, Style)>| -> Line<'static> {
        let mut spans: Vec<Span> = Vec::with_capacity(7);
        spans.push(Span::raw(" "));
        for (i, color) in colors.iter().enumerate() {
            spans.push(Span::styled(CHEVRON_ROWS[r], Style::default().fg(*color)));
            if i < 2 {
                spans.push(Span::raw("  "));
            }
        }
        if let Some((text, style)) = with_wordmark {
            spans.push(Span::raw("   "));
            spans.push(Span::styled(text, style));
        }
        Line::from(spans)
    };
    [
        row(0, None),
        row(
            1,
            Some(("A T L A S", theme::text().add_modifier(Modifier::BOLD))),
        ),
        row(2, Some(("I N F E R E N C E   E N G I N E", theme::dim()))),
    ]
}

/// One flag badge chip: `label` (dim) + `value` (primary), first-word category
/// tint applied by the caller via `tint`.
#[derive(Clone, Debug)]
pub struct Badge {
    pub text: String,
    pub tint: BadgeTint,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BadgeTint {
    Model,   // purple
    Quant,   // cyan
    Role,    // green
    Neutral, // none
}

/// Derive the header badge chips from ServeArgs — the "significant CLI flags"
/// strip on the Main tab. Order is display order; chips wrap.
///
/// `awaiting_model` is the load-bearing argument. Every chip below except the
/// address describes a *loaded model*, but they are all read from `ServeArgs`,
/// which is fully populated by clap defaults whether or not anything is
/// serving. On a no-model boot that produced a strip reading `<model>`,
/// `kv fp8`, `batch 8`, `ctx 32k` — a confident description of a configuration
/// that is not running. The listener, by contrast, really is up: it binds
/// before any model loads. So when awaiting we emit the state and the way out,
/// plus the address, and assert nothing else.
pub fn badges(a: &crate::cli::ServeArgs, awaiting_model: bool) -> Vec<Badge> {
    let mut out = Vec::new();
    if awaiting_model {
        out.push(Badge {
            text: "no model · press 4 for Library".into(),
            tint: BadgeTint::Neutral,
        });
        out.push(Badge {
            text: format!(":{}", a.port),
            tint: BadgeTint::Neutral,
        });
        return out;
    }
    let model = a
        .model_name
        .clone()
        .or_else(|| a.model.clone())
        .unwrap_or_else(|| "<model>".into());
    out.push(Badge {
        text: model,
        tint: BadgeTint::Model,
    });
    out.push(Badge {
        text: format!(
            "kv {} · lm {} · mtp {}",
            // Pre-resolution args: an omitted --kv-cache-dtype is decided
            // later against MODEL.toml, so "auto" is the honest label here.
            a.kv_cache_dtype.as_deref().unwrap_or("auto"),
            a.lm_head_dtype,
            a.mtp_quantization
        ),
        tint: BadgeTint::Quant,
    });
    if a.dflash {
        out.push(Badge {
            text: match a.dflash_gamma {
                Some(g) => format!("DFlash γ={g}"),
                None => "DFlash γ=auto".to_string(),
            },
            tint: BadgeTint::Quant,
        });
    } else if a.speculative || a.self_speculative || a.ngram_speculative {
        out.push(Badge {
            // Pre-resolution args: an omitted --num-drafts is decided later
            // against MODEL.toml, so the verify width is not yet known.
            text: match a.num_drafts {
                Some(n) => format!("MTP k={}", n + 1),
                None => "MTP k=auto".to_string(),
            },
            tint: BadgeTint::Quant,
        });
    } else {
        out.push(Badge {
            text: "spec off".into(),
            tint: BadgeTint::Neutral,
        });
    }
    out.push(Badge {
        text: format!("batch {}", a.max_batch_size),
        tint: BadgeTint::Neutral,
    });
    out.push(Badge {
        text: format!("ctx {}", human_tokens(a.max_seq_len)),
        tint: BadgeTint::Neutral,
    });
    let role = if a.rank == 0 { "head" } else { "worker" };
    out.push(Badge {
        text: format!(
            "{role} {}/{} · tp{} · ep{}",
            a.rank, a.world_size, a.tp_size, a.ep_size
        ),
        tint: BadgeTint::Role,
    });
    out.push(Badge {
        text: format!("sched {}", a.scheduling_policy),
        tint: BadgeTint::Neutral,
    });
    if a.enable_prefix_caching {
        out.push(Badge {
            text: format!(
                "prefix-cache · ssm {}@{}",
                a.ssm_cache_slots, a.ssm_checkpoint_interval
            ),
            tint: BadgeTint::Neutral,
        });
    }
    out.push(Badge {
        text: format!(":{}", a.port),
        tint: BadgeTint::Neutral,
    });
    out
}

fn human_tokens(n: usize) -> String {
    if n >= 1024 && n.is_multiple_of(1024) {
        format!("{}k", n / 1024)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
#[path = "logo_tests.rs"]
mod more_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chevron_rows_are_uniform_width() {
        // The half-block art must be column-stable or the header shears.
        let w = CHEVRON_ROWS[0].chars().count();
        assert!(CHEVRON_ROWS.iter().all(|r| r.chars().count() == w));
    }

    #[test]
    fn wave_brightens_one_chevron_at_a_time() {
        // In truecolor mode each wave step has exactly one full-brightness
        // chevron. (In 256-color fallback dimming is a no-op by design.)
        unsafe { std::env::set_var("COLORTERM", "truecolor") };
        for step in 0..3 {
            let colors = chevron_colors(Some(step));
            let brand: Vec<Color> = [theme::PURPLE, theme::CYAN, theme::GREEN]
                .iter()
                .map(|c| c.color())
                .collect();
            let bright = colors.iter().zip(&brand).filter(|(a, b)| a == b).count();
            assert_eq!(bright, 1, "step {step}");
        }
    }
}
