// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after Quest (page min/max upper-bound selection) and the page-sparse prototype in
// arczhi/llama-tierkv scripts-5060ti/proto/proto_page_sparse.cu (MIT, Copyright (c) 2023-2026 The
// ggml authors; file has no header of its own). Kernels rewritten for Atlas paged KV; changes: GQA
// grouping, warp top-k, compacted block table instead of host staging, by Atlas contributors. See
// QUEST-PAGE-SPARSE-ATTRIBUTION.md.

//! Responsibility: Quest page-sparse decode attention (port lane L30), the
//! `attn_page_sparse` row. APPROXIMATE: off by default on every target and
//! never default on (master plan 3.1, 3.5).
//!
//! # What it does
//! Long-context decode on the 4060 Ti is bound by reading the whole KV history
//! once per generated token (the 4B reads 32 KiB of BF16 KV per context token:
//! about 17 ms per step at 128K). Quest keeps, per attention layer, page of 64
//! tokens and KV head, the channel-wise minimum and maximum of the keys, and
//! bounds each page's best attention logit with
//! `score_p = sum_c max(q_c * kmin_c, q_c * kmax_c)` (an upper bound of
//! `q . k` for every key of the page). Each decode step then attends only to
//! the first `sink` pages, the `k` best-scoring pages and the recent window,
//! through a compacted block table the unchanged paged-decode kernel reads.
//! Keys carry RoPE from the write, so skipping pages is position-safe.
//!
//! # Wiring
//! - `port_apply` calls [`validate`] and [`apply`]: with the row on, the
//!   selector is built from the four flags and registered with the seam
//!   (`layers/qwen3_attention/sparse_seam.rs`, L29) before the model build.
//! - The model build charges the page summaries to the KV budget
//!   ([`budget::charge_summaries`], one line in `factory/build.rs`); the seam
//!   charges the selector's fixed View arena.
//! - The seam calls the selector after every K/V write and before every decode
//!   or verify attention; see `selector.rs`.
//! - With the `kv_window_desk` row (lane L36) also on, `apply` registers a [`QuestScorer`] with the
//!   desk (`kv_window::selector::set_page_scorer`) instead of registering the selector with the
//!   seam: the desk keeps attention, Quest ranks which pages it recalls from host RAM
//!   (`scorer.rs`, `scorer_rank.rs`). Only one selector may be registered per serve, so this is
//!   "Quest-ranked recall into the desk", not a startup refusal.
//! - `/metrics`: `atlas_attn_sparse_steps_total`,
//!   `atlas_attn_sparse_pages_selected_total` ([`render_prometheus`]); with the desk scorer
//!   registered also `atlas_attn_sparse_desk_rank_total{outcome}`,
//!   `atlas_attn_sparse_desk_pages_ranked_total` and
//!   `atlas_attn_sparse_desk_summaries_total{result}`.
//!
//! # Off is byte-identical
//! With the row off nothing registers, nothing is charged, no kernel is looked
//! up and the seam's `active()` answers `None` at every call site.

use std::sync::{Arc, OnceLock};

use anyhow::Result;

mod arena;
pub mod budget;
pub mod config;
pub mod launch;
mod metrics;
pub mod reference;
pub mod scorer;
pub mod scorer_rank;
mod scorer_store;
mod selector;

#[cfg(test)]
mod mock_tests;
#[cfg(test)]
mod scorer_tests;
#[cfg(test)]
mod tests;

pub use config::{GqaMode, QuestConfig, QuestFlags};
pub use scorer::QuestScorer;
pub use selector::{QuestSelector, Refusal};

/// `port_metrics::render`: the two Quest series, plus the window-desk scorer's when it is registered.
pub fn render_prometheus(out: &mut String) {
    metrics::render_prometheus(out);
    scorer::render_prometheus(out);
}

static INSTALLED: OnceLock<Arc<QuestSelector>> = OnceLock::new();

/// The installed selector, `None` with the row off.
pub fn installed() -> Option<&'static Arc<QuestSelector>> {
    INSTALLED.get()
}

/// The `attn_page_sparse` row, resolved (`AVAROK_ATTN_PAGE_SPARSE`, else the target).
pub fn row_on() -> bool {
    crate::layers::ops::port_rows::attn_page_sparse::row()
}

/// `port_apply::validate`: refuse out-of-range flags and an unknown
/// `AVAROK_ATTN_SPARSE_GQA`, whether or not the row is on (a typo should
/// not wait for the day the row is switched on).
pub fn validate(flags: &QuestFlags) -> Result<()> {
    QuestConfig::from_env(flags).map(|_| ())
}

/// `port_apply::apply`: with the row on, build and register the selector.
/// With it off, nothing happens (flags given anyway are reported once).
pub fn apply(flags: &QuestFlags) -> Result<()> {
    if !row_on() {
        if flags.any() {
            tracing::info!(
                "--attn-sparse-* flags given but the attn_page_sparse row is off \
                 (AVAROK_ATTN_PAGE_SPARSE=1 turns Quest page-sparse decode on); flags ignored"
            );
        }
        return Ok(());
    }
    if INSTALLED.get().is_some() {
        return Ok(());
    }
    let cfg = QuestConfig::from_env(flags)?;
    let sel = Arc::new(QuestSelector::new(cfg));
    if crate::port::kv_window::row() {
        // The window desk (lane L36) owns the one seam registration; the Quest bounds rank its
        // recall pages instead (`scorer.rs`). The installed selector still sizes and charges the
        // page summaries at the model build and folds the keys through the scorer's hooks.
        crate::port::kv_window::selector::set_page_scorer(Arc::new(QuestScorer::new(sel.clone())))?;
        scorer::mark_active();
        let _ = INSTALLED.set(sel);
        tracing::warn!(
            "attn_page_sparse is ON with the kv_window_desk row: Quest bounds ({}) rank the pages \
             the window desk recalls from host RAM; it is APPROXIMATE, and page summaries are \
             charged to the KV budget at model build",
            cfg.describe()
        );
        return Ok(());
    }
    crate::layers::qwen3_attention::sparse_seam::register(sel.clone())?;
    let _ = INSTALLED.set(sel);
    tracing::warn!(
        "attn_page_sparse is ON: Quest page-sparse decode attention ({}) is APPROXIMATE above \
         {} tokens of context; page summaries are charged to the KV budget at model build",
        cfg.describe(),
        cfg.min_ctx
    );
    Ok(())
}
