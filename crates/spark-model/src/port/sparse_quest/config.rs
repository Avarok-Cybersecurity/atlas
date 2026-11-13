// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after Quest (page min/max upper-bound selection) and the page-sparse prototype in
// arczhi/llama-tierkv scripts-5060ti/proto/proto_page_sparse.cu (MIT, Copyright (c) 2023-2026 The
// ggml authors; file has no header of its own). Rewritten for Atlas; the selection budget (sink
// pages, top-k pages, recent window, minimum context) follows the port sprint's report 77 BR-4.

//! Responsibility: the Quest selector's configuration (port lane L30): the
//! four serve flags (`--attn-sparse-topk-pages`, `--attn-sparse-min-ctx`,
//! `--attn-sparse-sink-pages`, `--attn-sparse-recent-tokens`), their defaults
//! and ranges, and the two diagnostic environment knobs. Pure: resolution and
//! validation are testable without a GPU, and an invalid value is a startup
//! error from `port_apply::validate`, never a panic later.
//!
//! The selector itself is switched by the `attn_page_sparse` row
//! (`AVAROK_ATTN_PAGE_SPARSE`, off on every target); the flags only shape it.

use anyhow::{Result, bail};

/// Pages chosen by score per sequence, layer and step (32 pages = 2,048 tokens).
pub const DEFAULT_TOPK_PAGES: u32 = 32;
/// The selector stays the identity at or below this many tokens of context.
pub const DEFAULT_MIN_CTX: u32 = 32_768;
/// Leading pages always kept (attention sinks, the system prompt's head).
pub const DEFAULT_SINK_PAGES: u32 = 1;
/// Trailing tokens always kept (the recent window, rounded down to a page).
pub const DEFAULT_RECENT_TOKENS: u32 = 2_048;

/// Largest `--attn-sparse-topk-pages`: half the candidates one compaction CTA
/// ranks (8,192 pages = 512K tokens), so a budget is always a selection.
pub const MAX_TOPK_PAGES: u32 = 4_096;
/// Largest `--attn-sparse-sink-pages` (64K tokens).
pub const MAX_SINK_PAGES: u32 = 1_024;
/// Largest `--attn-sparse-recent-tokens`.
pub const MAX_RECENT_TOKENS: u32 = 1 << 20;

/// How the bounds of one page combine over its query heads (GQA) and KV
/// heads. The brief leaves the choice to the needle test, so both exist.
pub const GQA_ENV: &str = "AVAROK_ATTN_SPARSE_GQA";
/// Diagnostic: log every eager selection (`quest select seq=.. layer=.. blocks=[..]`,
/// the line `bench/niah_grid.py --selection-log` reads). Eager steps only.
pub const LOG_ENV: &str = "AVAROK_ATTN_SPARSE_LOG";

/// Combination of per-head bounds into one page score.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GqaMode {
    /// The page's score is its largest bound over every query head: a page
    /// any head would attend to strongly is kept (the default).
    Max,
    /// The page's score is the sum of its bounds over every query head.
    Sum,
}

impl GqaMode {
    /// Parse `AVAROK_ATTN_SPARSE_GQA` (unset or empty: `max`).
    pub fn parse(raw: Option<&str>) -> Result<Self> {
        match raw.map(|v| v.trim().to_ascii_lowercase()) {
            None => Ok(Self::Max),
            Some(v) if v.is_empty() || v == "max" => Ok(Self::Max),
            Some(v) if v == "sum" => Ok(Self::Sum),
            Some(v) => bail!("{GQA_ENV}={v:?} is not a GQA mode; use `max` or `sum`"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Max => "max",
            Self::Sum => "sum",
        }
    }

    /// The kernel argument (`gqa_sum`).
    pub fn kernel_flag(self) -> u32 {
        u32::from(self == Self::Sum)
    }
}

/// The four serve flags as parsed (`None`: not given).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuestFlags {
    pub topk_pages: Option<u32>,
    pub min_ctx: Option<u32>,
    pub sink_pages: Option<u32>,
    pub recent_tokens: Option<u32>,
}

impl QuestFlags {
    /// True when any flag was given.
    pub fn any(&self) -> bool {
        self.topk_pages.is_some()
            || self.min_ctx.is_some()
            || self.sink_pages.is_some()
            || self.recent_tokens.is_some()
    }
}

/// The resolved configuration the selector runs with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuestConfig {
    pub topk_pages: u32,
    pub min_ctx: u32,
    pub sink_pages: u32,
    pub recent_tokens: u32,
    pub gqa: GqaMode,
    pub log_selection: bool,
}

impl Default for QuestConfig {
    fn default() -> Self {
        Self {
            topk_pages: DEFAULT_TOPK_PAGES,
            min_ctx: DEFAULT_MIN_CTX,
            sink_pages: DEFAULT_SINK_PAGES,
            recent_tokens: DEFAULT_RECENT_TOKENS,
            gqa: GqaMode::Max,
            log_selection: false,
        }
    }
}

/// The env toggle grammar (`0`/`false`/`off`/`no` or unset = off).
fn toggle(raw: Option<&str>) -> bool {
    match raw.map(str::trim) {
        None | Some("") => false,
        Some(v) => !matches!(
            v.to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
    }
}

impl QuestConfig {
    /// Flags over defaults, plus the two environment knobs (raw values passed
    /// in so this stays pure). Refuses out-of-range values.
    pub fn resolve(
        flags: &QuestFlags,
        gqa_raw: Option<&str>,
        log_raw: Option<&str>,
    ) -> Result<Self> {
        let d = Self::default();
        let cfg = Self {
            topk_pages: flags.topk_pages.unwrap_or(d.topk_pages),
            min_ctx: flags.min_ctx.unwrap_or(d.min_ctx),
            sink_pages: flags.sink_pages.unwrap_or(d.sink_pages),
            recent_tokens: flags.recent_tokens.unwrap_or(d.recent_tokens),
            gqa: GqaMode::parse(gqa_raw)?,
            log_selection: toggle(log_raw),
        };
        cfg.check()?;
        Ok(cfg)
    }

    /// The process environment's knobs over these flags.
    pub fn from_env(flags: &QuestFlags) -> Result<Self> {
        Self::resolve(
            flags,
            std::env::var(GQA_ENV).ok().as_deref(),
            std::env::var(LOG_ENV).ok().as_deref(),
        )
    }

    fn check(&self) -> Result<()> {
        if self.topk_pages > MAX_TOPK_PAGES {
            bail!(
                "--attn-sparse-topk-pages {} is above {MAX_TOPK_PAGES} (one selection ranks at most \
                 8,192 candidate pages)",
                self.topk_pages
            );
        }
        if self.sink_pages > MAX_SINK_PAGES {
            bail!(
                "--attn-sparse-sink-pages {} is above {MAX_SINK_PAGES}",
                self.sink_pages
            );
        }
        if self.recent_tokens == 0 || self.recent_tokens > MAX_RECENT_TOKENS {
            bail!(
                "--attn-sparse-recent-tokens {} is outside 1..={MAX_RECENT_TOKENS} (the page being \
                 written is always attended, so the window holds at least one token)",
                self.recent_tokens
            );
        }
        Ok(())
    }

    /// Tokens a sparse step attends at most: sinks, the top-k pages and the
    /// recent window plus the page it starts in and the tokens of this step.
    pub fn attended_tokens_bound(&self, page_tokens: u32, step_rows: u32) -> u64 {
        u64::from(self.sink_pages + self.topk_pages + 1) * u64::from(page_tokens)
            + u64::from(self.recent_tokens)
            + u64::from(step_rows)
    }

    /// The boot log's description.
    pub fn describe(&self) -> String {
        format!(
            "top-k {} pages, sinks {} pages, recent {} tokens, min context {} tokens, GQA {}{}",
            self.topk_pages,
            self.sink_pages,
            self.recent_tokens,
            self.min_ctx,
            self.gqa.as_str(),
            if self.log_selection {
                ", selection log on"
            } else {
                ""
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_flags_give_the_brief_defaults() {
        let c = QuestConfig::resolve(&QuestFlags::default(), None, None).unwrap();
        assert_eq!(c, QuestConfig::default());
        assert_eq!(
            (c.topk_pages, c.min_ctx, c.sink_pages, c.recent_tokens),
            (32, 32_768, 1, 2_048)
        );
        assert!(!QuestFlags::default().any());
    }

    #[test]
    fn flags_override_each_default() {
        let f = QuestFlags {
            topk_pages: Some(8),
            min_ctx: Some(0),
            sink_pages: Some(0),
            recent_tokens: Some(64),
        };
        assert!(f.any());
        let c = QuestConfig::resolve(&f, Some("sum"), Some("1")).unwrap();
        assert_eq!(
            (c.topk_pages, c.min_ctx, c.sink_pages, c.recent_tokens),
            (8, 0, 0, 64)
        );
        assert_eq!(c.gqa, GqaMode::Sum);
        assert!(c.log_selection);
        assert!(c.describe().contains("GQA sum"));
    }

    #[test]
    fn out_of_range_values_are_startup_errors() {
        let bad = [
            QuestFlags {
                topk_pages: Some(MAX_TOPK_PAGES + 1),
                ..Default::default()
            },
            QuestFlags {
                sink_pages: Some(MAX_SINK_PAGES + 1),
                ..Default::default()
            },
            QuestFlags {
                recent_tokens: Some(0),
                ..Default::default()
            },
            QuestFlags {
                recent_tokens: Some(MAX_RECENT_TOKENS + 1),
                ..Default::default()
            },
        ];
        for f in bad {
            assert!(QuestConfig::resolve(&f, None, None).is_err(), "{f:?}");
        }
    }

    #[test]
    fn gqa_mode_grammar() {
        assert_eq!(GqaMode::parse(None).unwrap(), GqaMode::Max);
        assert_eq!(GqaMode::parse(Some("")).unwrap(), GqaMode::Max);
        assert_eq!(GqaMode::parse(Some(" MAX ")).unwrap(), GqaMode::Max);
        assert_eq!(GqaMode::parse(Some("sum")).unwrap(), GqaMode::Sum);
        assert!(GqaMode::parse(Some("mean")).is_err());
        assert_eq!(GqaMode::Max.kernel_flag(), 0);
        assert_eq!(GqaMode::Sum.kernel_flag(), 1);
    }

    #[test]
    fn log_toggle_grammar() {
        for off in [None, Some(""), Some("0"), Some("off"), Some("No")] {
            assert!(!toggle(off), "{off:?}");
        }
        for on in [Some("1"), Some("yes"), Some("on")] {
            assert!(toggle(on), "{on:?}");
        }
    }

    #[test]
    fn the_attended_bound_at_the_defaults_is_about_four_thousand_tokens() {
        let c = QuestConfig::default();
        // 1 sink + 32 top-k + 1 partial page = 34 pages of 64, plus 2,048 recent, plus 1 row.
        assert_eq!(c.attended_tokens_bound(64, 1), 34 * 64 + 2_048 + 1);
    }
}
