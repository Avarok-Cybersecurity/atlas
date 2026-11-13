// SPDX-License-Identifier: AGPL-3.0-only
// Design after the TierKV pager of arczhi/llama-tierkv (kv-paging-design.md; MIT, Copyright (c) 2023-2026 The ggml authors): a VRAM desk of head sinks, a recent window and recalled pages over a host-RAM shelf for a logical context far past what VRAM holds, taken from the port plan's description (report 77 section 2.5); no code copied or fetched.
// Design after LMCache (https://github.com/LMCache/LMCache, commit 3aefae1, Apache-2.0) for the host-side copy rules (see `fetch.rs`); no code copied.

//! Responsibility: the host-resident window desk (port plan lane L36, row `kv_window_desk`,
//! APPROXIMATE attention, never on by default on any target): its serve flags and environment
//! knobs, its registration with the sparse-attention seam, the decode block hook, and its
//! `/metrics` series.
//!
//! # What it does
//! A long sequence keeps on the GPU only its "desk": the first `--kv-window-head-tokens`
//! positions (2,048: system prompt, task statement, attention sinks), the last
//! `--kv-window-tokens` positions (32,768) and up to `AVAROK_KV_WINDOW_RECALL_TOKENS` (2,048) of
//! recalled pages. Every other full KV block moves to host RAM in batches of 1,024 positions;
//! its logical position stays (the block table holds the desk's hole block there), so RoPE,
//! KV writes, verify rows and the scheduler's lengths are untouched. The full-attention layers
//! read the desk through a compacted block table (`selector.rs`), and the MTP heads attend a
//! bounded window of their own KV (`mtp_window.rs`). Pieces: `policy.rs` (what moves and
//! when), `desk.rs` (per-sequence state), `fetch.rs` (copies), `selector.rs` (attention).
//!
//! # When it serves
//! Only with the row on (`AVAROK_KV_WINDOW_DESK=1`; false on every target). It refuses the
//! prefix cache (a radix node must never point at a hidden page) and, through the seam, every
//! KV dtype but BF16 and FP8, `--high-speed-swap` and vision input. On a build without
//! `kv_page_fetch_ada.cu` it keeps every page on the GPU (one log line). With the row off the
//! block hook is one atomic load and nothing else in this module runs.
//!
//! # Not verified here
//! The mock tests prove the bookkeeping (identity below the window, demote then fetch restores
//! every byte, positions and references preserved). Recall quality needs a page scorer (lane
//! L30's Quest bounds, through `selector::PageScorer`) and the GPU pass on the 27B (plan Q14).

pub mod desk;
pub mod fetch;
pub mod mtp_window;
pub mod policy;
pub mod selector;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_desk;

use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow, bail};
use parking_lot::Mutex;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::kv_host_tier::pages::PageStats;
use spark_runtime::prefix_cache::PrefixCache;

use crate::traits::SequenceState;
use desk::{Desk, DeskConfig};

/// `--kv-window-tokens` default: the recent window kept on the GPU.
pub const DEFAULT_WINDOW_TOKENS: u32 = 32_768;
/// `--kv-window-head-tokens` default: head protection.
pub const DEFAULT_HEAD_TOKENS: u32 = 2_048;
/// Recalled pages a sequence may hold, in positions.
pub const DEFAULT_RECALL_TOKENS: u32 = 2_048;
/// Host RAM the desk may use, in GiB.
pub const DEFAULT_HOST_GIB: u32 = 16;

pub const ENV_WINDOW_TOKENS: &str = "AVAROK_KV_WINDOW_TOKENS";
pub const ENV_HEAD_TOKENS: &str = "AVAROK_KV_WINDOW_HEAD_TOKENS";
pub const ENV_RECALL_TOKENS: &str = "AVAROK_KV_WINDOW_RECALL_TOKENS";
pub const ENV_HOST_GB: &str = "AVAROK_KV_WINDOW_HOST_GB";

/// Upper bound of every token knob (16M positions).
const MAX_TOKENS: u32 = 1 << 24;

/// The desk's resolved sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowConfig {
    pub window_tokens: u32,
    pub head_tokens: u32,
    pub recall_tokens: u32,
    pub host_gib: u32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            window_tokens: DEFAULT_WINDOW_TOKENS,
            head_tokens: DEFAULT_HEAD_TOKENS,
            recall_tokens: DEFAULT_RECALL_TOKENS,
            host_gib: DEFAULT_HOST_GIB,
        }
    }
}

impl WindowConfig {
    /// Refuse sizes the desk cannot serve.
    pub fn check(&self) -> Result<()> {
        let step = policy::STEP_TOKENS as u32;
        if self.window_tokens < step || self.window_tokens > MAX_TOKENS {
            bail!(
                "--kv-window-tokens {} is outside {step}..={MAX_TOKENS} (the recent window must hold at \
                 least one demotion batch)",
                self.window_tokens
            );
        }
        if self.head_tokens > MAX_TOKENS {
            bail!(
                "--kv-window-head-tokens {} is above {MAX_TOKENS}",
                self.head_tokens
            );
        }
        if self.recall_tokens > self.window_tokens {
            bail!(
                "{ENV_RECALL_TOKENS}={} is larger than the recent window ({})",
                self.recall_tokens,
                self.window_tokens
            );
        }
        if self.host_gib == 0 || self.host_gib > 4096 {
            bail!("{ENV_HOST_GB}={} is outside 1..=4096", self.host_gib);
        }
        Ok(())
    }

    pub(crate) fn desk(&self) -> DeskConfig {
        DeskConfig {
            head_tokens: self.head_tokens as usize,
            window_tokens: self.window_tokens as usize,
            recall_tokens: self.recall_tokens as usize,
            host_budget_bytes: u64::from(self.host_gib) << 30,
        }
    }
}

/// Resolve the sizes: the CLI value first, then the environment, then the default (pure, so a
/// test passes its own `var`).
pub fn resolve_config(
    cli_window: Option<u32>,
    cli_head: Option<u32>,
    var: impl Fn(&str) -> Option<String>,
) -> Result<WindowConfig> {
    let num = |name: &str, cli: Option<u32>, default: u32| -> Result<u32> {
        if let Some(v) = cli {
            return Ok(v);
        }
        match var(name) {
            None => Ok(default),
            Some(s) if s.trim().is_empty() => Ok(default),
            Some(s) => s
                .trim()
                .parse::<u32>()
                .map_err(|_| anyhow!("{name}={s:?} is not a whole number")),
        }
    };
    let cfg = WindowConfig {
        window_tokens: num(ENV_WINDOW_TOKENS, cli_window, DEFAULT_WINDOW_TOKENS)?,
        head_tokens: num(ENV_HEAD_TOKENS, cli_head, DEFAULT_HEAD_TOKENS)?,
        recall_tokens: num(ENV_RECALL_TOKENS, None, DEFAULT_RECALL_TOKENS)?,
        host_gib: num(ENV_HOST_GB, None, DEFAULT_HOST_GIB)?,
    };
    cfg.check()?;
    Ok(cfg)
}

static ACTIVE: AtomicBool = AtomicBool::new(false);
static CONFIG: OnceLock<WindowConfig> = OnceLock::new();
static DESK: Mutex<Option<Desk>> = Mutex::new(None);
static STATS: OnceLock<Arc<PageStats>> = OnceLock::new();
static RESIDENT: AtomicU64 = AtomicU64::new(0);
static DEMOTED: AtomicU64 = AtomicU64::new(0);
static ERRORS: AtomicU64 = AtomicU64::new(0);
static WARNED_PREFIX: AtomicBool = AtomicBool::new(false);

fn stats() -> &'static Arc<PageStats> {
    STATS.get_or_init(|| Arc::new(PageStats::default()))
}

/// The `kv_window_desk` row.
pub fn row() -> bool {
    crate::layers::ops::port_rows::kv_window_desk::row()
}

/// The desk serves (the row was on when the flags were applied).
pub fn enabled() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// The sizes in force (the defaults before `apply_flags`).
pub fn config() -> WindowConfig {
    CONFIG.get().copied().unwrap_or_default()
}

fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// `port_apply::validate`: refuse bad sizes (with the row on or off).
pub fn validate_flags(window_tokens: Option<u32>, head_tokens: Option<u32>) -> Result<()> {
    resolve_config(window_tokens, head_tokens, env_var).map(|_| ())
}

/// `port_apply::apply`: publish the sizes; with the row on, register the desk's selector with
/// the seam and bound the MTP heads' attention. Before any model is built.
pub fn apply_flags(window_tokens: Option<u32>, head_tokens: Option<u32>) -> Result<()> {
    let cfg = resolve_config(window_tokens, head_tokens, env_var)?;
    let _ = CONFIG.set(cfg);
    if !row() {
        if window_tokens.is_some() || head_tokens.is_some() {
            tracing::info!(
                "kv window desk: --kv-window-* flags given with the kv_window_desk row off; they have no effect"
            );
        }
        return Ok(());
    }
    crate::layers::qwen3_attention::sparse_seam::register(Arc::new(
        selector::WindowDeskSelector::new(),
    ))?;
    mtp_window::apply(mtp_window::window_for(true, cfg.window_tokens));
    ACTIVE.store(true, Ordering::Relaxed);
    tracing::warn!(
        "kv window desk ON (approximate attention): {} head + {} recent + {} recalled positions stay on \
         the GPU per sequence, the other full KV blocks move to host RAM (up to {} GiB)",
        cfg.head_tokens,
        cfg.window_tokens,
        cfg.recall_tokens,
        cfg.host_gib
    );
    Ok(())
}

/// `serve_flags`: the desk refuses the RESOLVED prefix cache (`--hermetic` closes it).
pub fn check_prefix_cache(prefix_caching: bool) -> Result<()> {
    if row() && prefix_caching {
        bail!(
            "the kv_window_desk row (AVAROK_KV_WINDOW_DESK) needs the prefix cache off: a cached radix \
             node must never point at a page the desk moved to host RAM; drop --enable-prefix-caching \
             or turn the row off"
        );
    }
    Ok(())
}

/// The decode block hook (`block_mgmt::ensure_blocks_through_decode`, before every decode or
/// verify forward of `seq`). Row off: one atomic load. A desk error is logged and the step goes
/// on (every desk operation leaves the table valid when it fails).
pub fn on_decode_blocks(
    seq: &mut SequenceState,
    kv: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    if !ACTIVE.load(Ordering::Relaxed) {
        return Ok(());
    }
    if prefix_cache.is_active() || kv.config().cache_blocks_per_seq.is_some() {
        if !WARNED_PREFIX.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "kv window desk: the prefix cache or --high-speed-swap is on; the desk moves nothing"
            );
        }
        return Ok(());
    }
    let mut guard = DESK.lock();
    let desk = guard.get_or_insert_with(|| Desk::new(config().desk(), stats().clone()));
    let scorer = selector::page_scorer();
    match desk.maintain(
        &mut seq.block_table,
        seq.seq_len,
        kv,
        gpu,
        stream,
        &selector::SeamDevice,
        scorer,
    ) {
        Ok(rep) => {
            DEMOTED.fetch_add((rep.demoted + rep.rehidden) as u64, Ordering::Relaxed);
            if rep.demoted + rep.rehidden + rep.fetch_started + rep.installed + rep.reverted > 0 {
                tracing::debug!(
                    "kv window desk: kv={} demoted={} rehidden={} fetching={} installed={} freed={} reverted={}",
                    seq.seq_len,
                    rep.demoted,
                    rep.rehidden,
                    rep.fetch_started,
                    rep.installed,
                    rep.freed,
                    rep.reverted
                );
            }
        }
        Err(e) => {
            let n = ERRORS.fetch_add(1, Ordering::Relaxed);
            if n < 5 || n.is_multiple_of(1000) {
                tracing::warn!("kv window desk: step skipped ({} so far): {e:#}", n + 1);
            }
        }
    }
    RESIDENT.store(desk.resident_tokens() as u64, Ordering::Relaxed);
    selector::set_hidden(desk.any_hidden());
    Ok(())
}

/// A sequence ends (optional early release; the desk's sweep finds ended sequences anyway).
pub fn on_free_sequence(table: &[u32], gpu: &dyn GpuBackend) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    if let Some(desk) = DESK.lock().as_mut() {
        desk.on_free(table, gpu);
    }
}

/// Whether `table` holds a page the desk moved to host RAM.
pub fn holds_hidden_pages(table: &[u32]) -> bool {
    ACTIVE.load(Ordering::Relaxed) && selector::hole().is_some_and(|h| table.contains(&h))
}

/// Refuse to swap out a sequence with hidden pages (the swap image would hold the hole's bytes
/// at those positions).
pub fn swap_out_guard(table: &[u32]) -> Result<()> {
    if holds_hidden_pages(table) {
        bail!(
            "kv window desk: this sequence has KV pages in host RAM; a swap image of its block table \
             would lose them"
        );
    }
    Ok(())
}

/// The `atlas_kv_window_*` series (zeros with the row off).
pub fn render_prometheus(out: &mut String) {
    let s = stats().snapshot();
    let series: [(&str, &str, &str, String); 4] = [
        (
            "atlas_kv_window_resident_tokens",
            "gauge",
            "KV positions the window desk keeps on the GPU, summed over the sequences it manages (row kv_window_desk, approximate attention)",
            RESIDENT.load(Ordering::Relaxed).to_string(),
        ),
        (
            "atlas_kv_window_fetch_bytes_total",
            "counter",
            "Bytes the window desk copied from host RAM back to the GPU (recalled pages)",
            s.fetch_bytes.to_string(),
        ),
        (
            "atlas_kv_window_fetch_seconds_total",
            "counter",
            "Seconds the window desk's host-to-GPU page copies took (device time where timing events work)",
            format!("{:.6}", s.fetch_seconds),
        ),
        (
            "atlas_kv_window_demoted_pages_total",
            "counter",
            "KV blocks the window desk moved off the GPU (copied to host RAM or put back on the shelf)",
            DEMOTED.load(Ordering::Relaxed).to_string(),
        ),
    ];
    for (name, kind, help, v) in series {
        let _ = writeln!(
            out,
            "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}"
        );
    }
}
