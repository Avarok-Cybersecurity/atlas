// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the bounded KV view's knobs (port lane L32): the four
//! `--kv-view-*` serve flags (published once by `port_apply`), the two env-only
//! history knobs, their validation, and the device reservation estimate the
//! selector reports to the sparse-attention seam.
//!
//! Resolution: the view is on only when the `kv_view_selection` row resolves
//! on (`AVAROK_KV_VIEW`, else the target's declaration, off everywhere) AND
//! `--kv-view-budget-tokens` is above zero. The other flags fall back to the
//! defaults below when absent (or zero), so `PortArgs::default()` and an
//! absent flag mean the same thing.

use std::sync::OnceLock;

use anyhow::{Result, bail, ensure};

/// Default sink: the first block of the context (attention sinks).
pub const DEFAULT_SINK_TOKENS: u32 = 128;
/// Default recent window kept in full before the selected blocks.
pub const DEFAULT_RECENT_TOKENS: u32 = 2048;
/// Default query span: the last 512 tokens of the final user message.
pub const DEFAULT_QUERY_TOKENS: u32 = 512;
/// The longest query span captured (bounds the capture buffers).
pub const MAX_QUERY_TOKENS: u32 = 2048;
/// Tokens per scoring block (8 pages of 16 tokens on Atlas's default KV block).
pub const BLOCK_TOKENS: u32 = 128;

/// One-hop score propagation weight (0 = off). Env only.
pub const ENV_GRAPH_ALPHA: &str = "AVAROK_KV_VIEW_GRAPH_ALPHA";
/// Decayed relevance history weight (0 = off). Env only.
pub const ENV_HISTORY_LAMBDA: &str = "AVAROK_KV_VIEW_HISTORY_LAMBDA";

/// Diagnostic, off unless set to `1`, `true`, `yes` or `on`: log the view each
/// selection chose, one line per attention layer, in the shape
/// `bench/niah_grid.py --selection-log` reads (`select.rs`). Read once.
pub const ENV_SELECTION_LOG: &str = "AVAROK_KV_VIEW_SELECTION_LOG";

/// Upper bound accepted for the propagation weight (kvmem uses 0.2).
pub const MAX_GRAPH_ALPHA: f32 = 0.5;

/// Query-capture buffers kept at once (two final user messages may finish
/// their prefill in one batched step).
pub const CAPTURE_SLOTS: usize = 2;
/// Reservation estimate per captured query row, all attention layers
/// together (the 4B needs 64 KiB, the 27B 192 KiB; the actual size is logged
/// at allocation).
pub const CAPTURE_ROW_BYTES_ESTIMATE: u64 = 128 << 10;

/// The serve flags as `port_apply` publishes them (`None` or 0 = default).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CliKnobs {
    pub budget_tokens: Option<u32>,
    pub sink_tokens: Option<u32>,
    pub recent_tokens: Option<u32>,
    pub query_tokens: Option<u32>,
}

impl CliKnobs {
    /// The budget, 0 when absent.
    pub fn budget(&self) -> u32 {
        self.budget_tokens.unwrap_or(0)
    }
}

/// The resolved knobs of a serve with the view on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvViewConfig {
    /// Tokens the view holds at selection (sink + recent + top blocks).
    pub budget_tokens: u32,
    pub sink_tokens: u32,
    pub recent_tokens: u32,
    pub query_tokens: u32,
    /// One-hop propagation weight, 0 = off.
    pub graph_alpha: f32,
    /// History decay, 0 = off (rank by this round's scores).
    pub history_lambda: f32,
}

fn or_default(v: Option<u32>, d: u32) -> u32 {
    match v {
        None | Some(0) => d,
        Some(x) => x,
    }
}

/// Parse an env float knob: unset or empty is 0.
fn parse_knob(name: &str, raw: Option<&str>) -> Result<f32> {
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(0.0);
    };
    match raw.parse::<f32>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => bail!("{name}={raw:?} is not a number"),
    }
}

impl KvViewConfig {
    /// Resolve the knobs from the flags and the two env values.
    pub fn from_parts(cli: CliKnobs, alpha: Option<&str>, lambda: Option<&str>) -> Result<Self> {
        validate_cli(&cli)?;
        let graph_alpha = parse_knob(ENV_GRAPH_ALPHA, alpha)?;
        ensure!(
            (0.0..=MAX_GRAPH_ALPHA).contains(&graph_alpha),
            "{ENV_GRAPH_ALPHA}={graph_alpha} is outside 0..={MAX_GRAPH_ALPHA} (0 = off; kvmem uses 0.2)"
        );
        let history_lambda = parse_knob(ENV_HISTORY_LAMBDA, lambda)?;
        ensure!(
            (0.0..1.0).contains(&history_lambda),
            "{ENV_HISTORY_LAMBDA}={history_lambda} is outside 0 <= lambda < 1 (0 = off; kvmem uses 0.5)"
        );
        Ok(Self {
            budget_tokens: cli.budget(),
            sink_tokens: or_default(cli.sink_tokens, DEFAULT_SINK_TOKENS),
            recent_tokens: or_default(cli.recent_tokens, DEFAULT_RECENT_TOKENS),
            query_tokens: or_default(cli.query_tokens, DEFAULT_QUERY_TOKENS),
            graph_alpha,
            history_lambda,
        })
    }

    /// [`Self::from_parts`] over the process environment.
    pub fn from_env(cli: CliKnobs) -> Result<Self> {
        let alpha = std::env::var(ENV_GRAPH_ALPHA).ok();
        let lambda = std::env::var(ENV_HISTORY_LAMBDA).ok();
        Self::from_parts(cli, alpha.as_deref(), lambda.as_deref())
    }

    /// Most pages a selection keeps before its tail (sink + top blocks), for
    /// a KV block of `block_size` tokens. Sizes the per-view page pool.
    pub fn max_selected_pages(&self, block_size: u32) -> u32 {
        let bs = block_size.max(1);
        self.budget_tokens.div_ceil(bs) + BLOCK_TOKENS.div_ceil(bs)
    }

    /// Device bytes the selector reserves from the KV-block budget: the view
    /// table arena, the descriptor table and page pool, the prefill table,
    /// the score scratch and the query-capture buffers. The per-page key
    /// sketch is sized by the KV pool and allocated after the build (see
    /// `KVMEM-VIEW-NOTES.md`).
    pub fn reserve_estimate_bytes(&self) -> u64 {
        let fixed = super::device::fixed_bytes(self.max_selected_pages(16));
        let capture =
            CAPTURE_SLOTS as u64 * u64::from(self.query_tokens) * CAPTURE_ROW_BYTES_ESTIMATE;
        fixed + capture
    }
}

/// Range checks of the flags (a startup error, never a panic).
pub fn validate_cli(k: &CliKnobs) -> Result<()> {
    let budget = k.budget();
    if budget == 0 {
        return Ok(());
    }
    let sink = or_default(k.sink_tokens, DEFAULT_SINK_TOKENS);
    let recent = or_default(k.recent_tokens, DEFAULT_RECENT_TOKENS);
    let query = or_default(k.query_tokens, DEFAULT_QUERY_TOKENS);
    ensure!(
        query <= MAX_QUERY_TOKENS,
        "--kv-view-query-tokens {query} is above the {MAX_QUERY_TOKENS}-token capture limit"
    );
    let floor = sink + recent + BLOCK_TOKENS;
    ensure!(
        budget >= floor,
        "--kv-view-budget-tokens {budget} leaves no room for selected blocks: it must be at least \
         --kv-view-sink-tokens ({sink}) + --kv-view-recent-tokens ({recent}) + one {BLOCK_TOKENS}-token \
         block = {floor}"
    );
    Ok(())
}

static CLI: OnceLock<CliKnobs> = OnceLock::new();

/// Publish the serve flags (once, from `port_apply`). A second publication
/// with different values is an error.
pub fn set_cli(k: CliKnobs) -> Result<()> {
    validate_cli(&k)?;
    match CLI.set(k) {
        Ok(()) => Ok(()),
        Err(_) if CLI.get() == Some(&k) => Ok(()),
        Err(_) => {
            bail!("kv view: the --kv-view-* flags were published twice with different values")
        }
    }
}

/// The published flags (all absent when `port_apply` never ran, as in tests).
pub fn cli() -> CliKnobs {
    CLI.get().copied().unwrap_or_default()
}

/// The toggle grammar of [`ENV_SELECTION_LOG`]: unset, empty and anything but
/// `1`, `true`, `yes` and `on` (any case) is off.
pub fn parse_toggle(raw: Option<&str>) -> bool {
    raw.map(|r| r.trim().to_ascii_lowercase())
        .is_some_and(|r| matches!(r.as_str(), "1" | "true" | "yes" | "on"))
}

/// Whether the selection debug log is on (`AVAROK_KV_VIEW_SELECTION_LOG`; off
/// unless set; read once).
pub fn selection_log() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_toggle(std::env::var(ENV_SELECTION_LOG).ok().as_deref()))
}

/// Whether the flags and the row together turn the view on.
pub fn would_enable(k: &CliKnobs, row: bool) -> bool {
    row && k.budget() > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn knobs(budget: u32) -> CliKnobs {
        CliKnobs {
            budget_tokens: Some(budget),
            ..CliKnobs::default()
        }
    }

    #[test]
    fn absent_flags_take_the_defaults() {
        let c = KvViewConfig::from_parts(knobs(16384), None, None).unwrap();
        assert_eq!(c.sink_tokens, DEFAULT_SINK_TOKENS);
        assert_eq!(c.recent_tokens, DEFAULT_RECENT_TOKENS);
        assert_eq!(c.query_tokens, DEFAULT_QUERY_TOKENS);
        assert_eq!(c.graph_alpha, 0.0);
        assert_eq!(c.history_lambda, 0.0);
        let zeros = CliKnobs {
            budget_tokens: Some(16384),
            sink_tokens: Some(0),
            recent_tokens: Some(0),
            query_tokens: Some(0),
        };
        assert_eq!(KvViewConfig::from_parts(zeros, None, None).unwrap(), c);
    }

    #[test]
    fn a_zero_budget_is_off_and_never_an_error() {
        assert!(validate_cli(&CliKnobs::default()).is_ok());
        assert!(!would_enable(&CliKnobs::default(), true));
        assert!(!would_enable(&knobs(16384), false));
        assert!(would_enable(&knobs(16384), true));
    }

    #[test]
    fn the_budget_must_hold_sink_recent_and_one_block() {
        let floor = DEFAULT_SINK_TOKENS + DEFAULT_RECENT_TOKENS + BLOCK_TOKENS;
        assert!(validate_cli(&knobs(floor)).is_ok());
        assert!(validate_cli(&knobs(floor - 1)).is_err());
        let long_query = CliKnobs {
            query_tokens: Some(MAX_QUERY_TOKENS + 1),
            ..knobs(65536)
        };
        assert!(validate_cli(&long_query).is_err());
    }

    #[test]
    fn env_knobs_are_range_checked() {
        let k = knobs(16384);
        let c = KvViewConfig::from_parts(k, Some("0.2"), Some("0.5")).unwrap();
        assert_eq!(c.graph_alpha, 0.2);
        assert_eq!(c.history_lambda, 0.5);
        assert!(KvViewConfig::from_parts(k, Some("0.9"), None).is_err());
        assert!(KvViewConfig::from_parts(k, Some("-0.1"), None).is_err());
        assert!(KvViewConfig::from_parts(k, None, Some("1.0")).is_err());
        assert!(KvViewConfig::from_parts(k, None, Some("nan")).is_err());
        assert!(KvViewConfig::from_parts(k, Some("x"), None).is_err());
        assert_eq!(
            KvViewConfig::from_parts(k, Some(" "), Some(""))
                .unwrap()
                .graph_alpha,
            0.0
        );
    }

    #[test]
    fn the_selection_log_toggle_is_off_unless_asked() {
        for off in [
            None,
            Some(""),
            Some(" "),
            Some("0"),
            Some("off"),
            Some("No"),
            Some("2"),
        ] {
            assert!(!parse_toggle(off), "{off:?}");
        }
        for on in [Some("1"), Some("true"), Some("YES"), Some(" on ")] {
            assert!(parse_toggle(on), "{on:?}");
        }
    }

    #[test]
    fn the_reservation_covers_the_capture_buffers() {
        let c = KvViewConfig::from_parts(knobs(16384), None, None).unwrap();
        let capture = CAPTURE_SLOTS as u64 * 512 * CAPTURE_ROW_BYTES_ESTIMATE;
        assert!(c.reserve_estimate_bytes() > capture);
        assert_eq!(c.max_selected_pages(16), 1024 + 8);
    }
}
