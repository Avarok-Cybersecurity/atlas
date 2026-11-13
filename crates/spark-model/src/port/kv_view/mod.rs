// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the bounded KV view (port lanes L31 and L32), an APPROXIMATE
//! attention mode that is off on every target (row `kv_view_selection`,
//! `AVAROK_KV_VIEW`, plus `--kv-view-budget-tokens`). This module root holds
//! the types the scheduler and the model's `ModelPortExt` methods exchange and
//! the process-wide entry points; lane L31 owns `query_capture.rs` (declared
//! here by contract), lane L32 everything else.
//!
//! # What it does
//! Once per user turn, at the end of the final user message's prefill, the
//! blocks of the context are scored against that message's last query rows
//! (L31's page-softmax scorer over a per-page mean-key sketch) and a view is
//! picked: sink blocks, the recent window, the question, then the highest
//! scored blocks, within the token budget. From the second output token on,
//! every full-attention layer's decode and verify attention reads only the
//! view (selected pages in chronological order plus every page from the
//! recent window on, so new tokens are appended to it). The first output
//! token comes from the prefill's own logits; the question is not replayed.
//! The GDN layers and the MTP head's own attention are untouched (they still
//! see all history). The view also serves the paged prefill chunks of the
//! conversation's next turn until that turn selects again.
//!
//! # Turns
//! The scheduler (`spark-server` `scheduler/kv_view.rs`) classifies each
//! prompt: a new user message reselects; a tool result or an assistant
//! continuation keeps the conversation's selection (appending); a prompt that
//! no longer starts with the conversation's selected prefix (an edit) drops
//! it. Conversations are keyed by the request's session hash and checked by a
//! fingerprint of the prefix.
//!
//! # Exactness when it does not apply
//! With the row off nothing here runs. With it on, a context no longer than
//! the budget is never given a view, and a row without a view keeps its real
//! table and length (the identity), so `budget >= context` is token-for-token
//! the dense model.

use std::ops::Range;
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;

pub mod config;
pub mod core_api;
pub mod device;
pub mod history;
mod live;
pub mod metrics;
mod plans;
// Lane L31's query capture (declared here by contract, master plan L31 brief).
pub mod query_capture;
mod select;
pub mod selector;
mod state;
pub mod view;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_flow;
#[cfg(test)]
mod tests_flow_b;

pub use core_api::Geom;
pub use metrics::SkipReason;
pub use selector::KvViewSelector;

/// What the scheduler found at the end of a prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnKind {
    /// A new user message: select a new view.
    NewUser,
    /// A tool result or an assistant continuation: keep the selection.
    Continuation,
    /// No chat markers: select from the prompt's tail, no conversation.
    Unmarked,
}

/// A prompt's turn, as the scheduler classified it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnInfo {
    pub kind: TurnKind,
    /// Absolute positions of the query rows to capture (at most
    /// `--kv-view-query-tokens`, the end of the final user message).
    pub query: Range<u32>,
    /// Prompt tokens a later turn of the conversation must repeat to reuse
    /// this turn's selection (everything before the generation header).
    pub match_len: u32,
}

/// One stream of a prefill call the scheduler is about to make.
pub struct CallStream<'a> {
    /// The sequence's pool slot (`SequenceState::slot_idx`).
    pub slot: u64,
    /// The request's session hash (the conversation key).
    pub session: u64,
    pub prompt: &'a [u32],
    /// Absolute positions this call computes (`chunk_start..chunk_end`).
    pub chunk_start: u32,
    pub chunk_end: u32,
    pub turn: TurnInfo,
}

/// The sequence whose prefill just completed.
pub struct ViewSpec<'a> {
    pub prompt: &'a [u32],
    /// The real block table (physical ids, in logical order).
    pub block_table: &'a [u32],
    /// Tokens in the cache (the prompt length after its prefill).
    pub seq_len: u32,
    /// The stream the prefill ran on (scoring waits for it).
    pub prefill_stream: u64,
}

/// What `set_kv_view` did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ViewOutcome {
    /// The view is not on for this serve or model.
    Off,
    /// A fresh selection.
    Selected {
        context_tokens: u32,
        view_tokens: u32,
        recall_mass: f64,
    },
    /// The conversation's selection was kept (a continuation).
    Kept {
        context_tokens: u32,
        view_tokens: u32,
    },
    /// No view: the sequence attends its full context.
    Skipped(SkipReason),
    /// The sequence's arm and view were dropped.
    Discarded,
}

/// The model facts the view sizes itself by, read once per prefill call.
pub struct ModelFacts<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub geom: Geom,
    /// Per-layer KV head counts or head sizes (not served).
    pub heterogeneous: bool,
}

impl<'a> ModelFacts<'a> {
    /// From the model's parts (called inside `model/trait_impl/port_ext.rs`).
    pub fn from_parts(
        gpu: &'a dyn GpuBackend,
        kv: &PagedKvCache,
        q_heads: usize,
        rotary_dim: usize,
        rope_theta: f64,
        max_blocks_per_seq: u32,
    ) -> Self {
        let c = kv.config();
        Self {
            gpu,
            geom: Geom {
                block_size: kv.block_size() as u32,
                num_blocks: kv.num_blocks() as u32,
                attn_layers: kv.num_layers() as u32,
                kv_heads: c.num_kv_heads as u32,
                head_dim: c.head_dim as u32,
                q_heads: q_heads as u32,
                rotary_dim: rotary_dim as u32,
                rope_theta: rope_theta as f32,
                max_blocks_per_seq,
            },
            heterogeneous: !c.layer_dims.is_empty(),
        }
    }
}

static SELECTOR: OnceLock<Arc<KvViewSelector>> = OnceLock::new();

/// Whether the bounded view is on for this serve (the row on, a budget, and
/// the selector registered with the seam). One atomic load.
pub fn enabled() -> bool {
    SELECTOR.get().is_some()
}

/// The resolved knobs (`None` while off).
pub fn config() -> Option<config::KvViewConfig> {
    SELECTOR.get().map(|s| *s.config())
}

/// Register the selector with the sparse-attention seam when the row and the
/// budget turn the view on (one `port_apply` line, after `config::set_cli`).
pub fn register_if_enabled() -> Result<()> {
    let row = crate::layers::ops::port_rows::kv_view_selection::row();
    let cli = config::cli();
    if !config::would_enable(&cli, row) {
        if row {
            tracing::warn!(
                "kv_view_selection is on but --kv-view-budget-tokens is not set: the bounded KV \
                 view stays off"
            );
        } else if cli.budget() > 0 {
            tracing::warn!(
                "--kv-view-budget-tokens is set but the kv_view_selection row is off \
                 (AVAROK_KV_VIEW=1 turns it on): the bounded KV view stays off"
            );
        }
        return Ok(());
    }
    let cfg = config::KvViewConfig::from_env(cli)?;
    let sel = Arc::new(KvViewSelector::new(cfg));
    if SELECTOR.set(sel.clone()).is_err() {
        anyhow::bail!("kv view: the selector was registered twice");
    }
    crate::layers::qwen3_attention::sparse_seam::register(sel)?;
    tracing::warn!(
        "APPROXIMATE attention: bounded KV view on (budget {} tokens, sink {}, recent {}, query {}, \
         history lambda {}, graph alpha {}); decode attends at most the view after each user turn",
        cfg.budget_tokens,
        cfg.sink_tokens,
        cfg.recent_tokens,
        cfg.query_tokens,
        cfg.history_lambda,
        cfg.graph_alpha,
    );
    Ok(())
}

/// Before a prefill call (`ModelPortExt::kv_view_begin_call`).
pub fn begin_call(facts: &ModelFacts<'_>, streams: &[CallStream<'_>]) -> Result<()> {
    match SELECTOR.get() {
        Some(s) => s.begin_call(facts, streams),
        None => Ok(()),
    }
}

/// After the prefill call (`ModelPortExt::kv_view_end_call`).
pub fn end_call() {
    if let Some(s) = SELECTOR.get() {
        s.end_call();
    }
}

/// Select (or keep, or skip) a view for the sequence in `slot` whose prefill
/// completed (`Some`), or drop its arm and view (`None`). `ModelPortExt::set_kv_view`.
pub fn set_kv_view(
    gpu: &dyn GpuBackend,
    slot: u64,
    spec: Option<ViewSpec<'_>>,
) -> Result<ViewOutcome> {
    match SELECTOR.get() {
        Some(s) => s.set_kv_view(gpu, slot, spec),
        None => Ok(ViewOutcome::Off),
    }
}

/// A sequence is being freed: drop its arm and any view bound to its blocks
/// (called from the model's `free_sequence` before the blocks are released).
pub fn release_sequence(gpu: &dyn GpuBackend, slot: u64, block_table: &[u32], seq_len: u32) {
    if let Some(s) = SELECTOR.get() {
        s.release_sequence(gpu, slot, block_table, seq_len);
    }
}
