// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the bounded KV view's scheduler side (port lane L32, row
//! `kv_view_selection`, APPROXIMATE, off on every target): classify each
//! prompt's turn, wrap every prefill call so the model can arm the final user
//! message's query capture and name the call's streams, trigger the per-turn
//! selection when a prefill completes, and publish the flags and the
//! `atlas_kv_view_*` series. The view itself lives in `spark-model`
//! (`port/kv_view/`).
//!
//! # Turns (ChatML prompts)
//! The prompt ends in a generation header (`<|im_start|>assistant\n`, maybe a
//! think opener). The block before it is the final message:
//! - its role is the header's role (an assistant continuation), or its
//!   content starts with `<tool_response>` (a tool result): a continuation,
//!   which keeps the conversation's selection and appends to its view;
//! - otherwise a new user message, which selects a new view from the last
//!   `--kv-view-query-tokens` tokens of that message.
//!
//! A prompt with no generation header (no chat markers) selects from its own
//! tail and starts no conversation. An edited conversation (a prompt that no
//! longer repeats the selected prefix) is detected in `spark-model` and starts
//! over. The selection runs right after the final prefill chunk, so the first
//! output token comes from the prefill's own logits and the view applies from
//! the second token on.
//!
//! # Cost while off
//! Every hook first checks `spark_model::port::kv_view::enabled()` (one atomic
//! load) and does nothing else when it is false.

use spark_model::layers::qwen3_attention::sparse_seam::{self, StepGuard};
use spark_model::port::kv_view::{
    self as view, CallStream, TurnInfo, TurnKind, ViewOutcome, ViewSpec,
};
use spark_model::traits::{Model, SequenceState};

use crate::cli::serve_args_port::PortArgs;
use crate::scheduler::limits::SchedLimits;

/// Tokens after the last `<|im_start|>` a generation header may span
/// (`assistant`, a newline, a think opener and its newline, with room).
const HEADER_SLACK: usize = 16;

/// The chat-format token ids the turn classification reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Markers {
    pub(crate) im_start: Option<u32>,
    pub(crate) tool_response: Option<u32>,
}

impl Markers {
    /// The ids the tokenizer resolved at startup.
    pub(crate) fn from_limits(l: &SchedLimits) -> Self {
        Self {
            im_start: l.im_start_hard_stop,
            tool_response: l.tool_response_hard_stop,
        }
    }
}

fn to_u32(v: usize) -> u32 {
    u32::try_from(v).unwrap_or(u32::MAX)
}

fn unmarked(n: usize) -> TurnInfo {
    TurnInfo {
        kind: TurnKind::Unmarked,
        query: 0..to_u32(n),
        match_len: to_u32(n),
    }
}

/// The turn a prompt ends in. The query span is the whole final message
/// (the capture keeps its last `--kv-view-query-tokens` tokens).
pub(crate) fn classify(tokens: &[u32], m: Markers) -> TurnInfo {
    let n = tokens.len();
    let Some(ims) = m.im_start else {
        return unmarked(n);
    };
    let Some(g) = tokens.iter().rposition(|&t| t == ims) else {
        return unmarked(n);
    };
    if n - g > HEADER_SLACK {
        return unmarked(n);
    }
    let Some(m0) = tokens[..g].iter().rposition(|&t| t == ims) else {
        return unmarked(n);
    };
    // `<|im_start|> role \n content <|im_end|> \n <|im_start|> assistant \n`
    let content_start = (m0 + 3).min(g);
    let content_end = g.saturating_sub(2).max(content_start);
    let same_role = tokens.get(m0 + 1).is_some() && tokens.get(m0 + 1) == tokens.get(g + 1);
    let tool_result = m.tool_response.is_some()
        && content_start < content_end
        && tokens.get(content_start).copied() == m.tool_response;
    let kind = if same_role || tool_result {
        TurnKind::Continuation
    } else {
        TurnKind::NewUser
    };
    TurnInfo {
        kind,
        query: to_u32(content_start)..to_u32(content_end),
        match_len: to_u32(g),
    }
}

/// One stream of a prefill call.
pub(crate) struct Stream<'a> {
    pub(crate) slot: usize,
    pub(crate) session: u64,
    pub(crate) prompt: &'a [u32],
    /// First prompt position the call computes, and how many.
    pub(crate) chunk_start: usize,
    pub(crate) chunk_len: usize,
}

/// The streams of a batched prefill call: every in-progress prefill with its
/// chunk length this call.
pub(super) fn streams_of<'a>(
    prefilling: &'a [super::types::PrefillInProgress],
    chunk_lens: &[usize],
) -> Vec<Stream<'a>> {
    prefilling
        .iter()
        .zip(chunk_lens)
        .map(|(p, &chunk_len)| Stream::of(p, chunk_len))
        .collect()
}

impl<'a> Stream<'a> {
    /// An in-progress prefill's next chunk of `chunk_len` tokens.
    pub(super) fn of(p: &'a super::types::PrefillInProgress, chunk_len: usize) -> Self {
        Self {
            slot: p.seq.slot_idx,
            session: p.session_hash,
            prompt: p.prompt_tokens.as_slice(),
            chunk_start: p.chunk_offset,
            chunk_len,
        }
    }
}

/// How a stream's part of the call ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChunkEnd {
    /// More chunks follow.
    More,
    /// The prompt's last chunk succeeded: select now.
    Done,
    /// The call failed: drop the arm.
    Failed,
}

impl ChunkEnd {
    /// A chunk that was (`is_last`) or was not the prompt's last, and whether
    /// the call succeeded for it.
    pub(crate) fn of(is_last: bool, ok: bool) -> Self {
        match (is_last, ok) {
            (false, _) => Self::More,
            (true, true) => Self::Done,
            (true, false) => Self::Failed,
        }
    }
}

/// A prefill call in flight. Inert (and free) while the view is off.
pub(crate) struct CallHook<'m> {
    model: Option<&'m dyn Model>,
    _step: Option<StepGuard>,
}

impl CallHook<'_> {
    fn inert() -> Self {
        Self {
            model: None,
            _step: None,
        }
    }
}

/// Before a prefill call: arm the streams and publish their ids.
pub(crate) fn begin<'m>(
    model: &'m dyn Model,
    limits: &SchedLimits,
    streams: &[Stream<'_>],
) -> CallHook<'m> {
    if !view::enabled() || streams.is_empty() {
        return CallHook::inert();
    }
    let Some(ext) = model.port_ext() else {
        return CallHook::inert();
    };
    let markers = Markers::from_limits(limits);
    let calls: Vec<CallStream<'_>> = streams
        .iter()
        .map(|s| CallStream {
            slot: s.slot as u64,
            session: s.session,
            prompt: s.prompt,
            chunk_start: to_u32(s.chunk_start),
            chunk_end: to_u32(s.chunk_start + s.chunk_len),
            turn: classify(s.prompt, markers),
        })
        .collect();
    if let Err(e) = ext.kv_view_begin_call(&calls) {
        tracing::warn!("bounded KV view: arming a prefill call failed (dense attention): {e:#}");
        return CallHook::inert();
    }
    let ids: Vec<u64> = streams.iter().map(|s| s.slot as u64).collect();
    CallHook {
        model: Some(model),
        _step: Some(sparse_seam::publish_step(&ids, false)),
    }
}

impl CallHook<'_> {
    /// After the call, for each stream it carried.
    pub(crate) fn finish(
        &self,
        prompt: &[u32],
        seq: &SequenceState,
        end: ChunkEnd,
        prefill_stream: u64,
    ) {
        let Some(ext) = self.model.and_then(|m| m.port_ext()) else {
            return;
        };
        let slot = seq.slot_idx as u64;
        match end {
            ChunkEnd::More => {}
            ChunkEnd::Failed => {
                let _ = ext.set_kv_view(slot, None);
            }
            ChunkEnd::Done => {
                let spec = ViewSpec {
                    prompt,
                    block_table: &seq.block_table,
                    seq_len: to_u32(seq.seq_len),
                    prefill_stream,
                };
                match ext.set_kv_view(slot, Some(spec)) {
                    Ok(o) => log_outcome(slot, &o),
                    Err(e) => {
                        tracing::warn!(
                            "bounded KV view: selecting for slot {slot} failed (dense attention): {e:#}"
                        );
                        let _ = ext.set_kv_view(slot, None);
                    }
                }
            }
        }
    }
}

impl Drop for CallHook<'_> {
    fn drop(&mut self) {
        if let Some(ext) = self.model.and_then(|m| m.port_ext()) {
            ext.kv_view_end_call();
        }
    }
}

fn log_outcome(slot: u64, o: &ViewOutcome) {
    match o {
        ViewOutcome::Selected {
            context_tokens,
            view_tokens,
            recall_mass,
        } => tracing::debug!(
            "bounded KV view: slot {slot} attends {view_tokens} of {context_tokens} tokens \
             (recall mass {recall_mass:.3})"
        ),
        ViewOutcome::Kept {
            context_tokens,
            view_tokens,
        } => tracing::debug!(
            "bounded KV view: slot {slot} kept its conversation's view ({view_tokens} of \
             {context_tokens} tokens)"
        ),
        ViewOutcome::Skipped(r) => {
            tracing::debug!(
                "bounded KV view: slot {slot} attends its full context ({})",
                r.label()
            );
        }
        ViewOutcome::Off | ViewOutcome::Discarded => {}
    }
}

fn knobs(a: &PortArgs) -> view::config::CliKnobs {
    view::config::CliKnobs {
        budget_tokens: a.kv_view_budget_tokens,
        sink_tokens: a.kv_view_sink_tokens,
        recent_tokens: a.kv_view_recent_tokens,
        query_tokens: a.kv_view_query_tokens,
    }
}

/// `port_apply::validate`: range checks, the env knobs, and the combinations
/// the view refuses.
pub(crate) fn validate_args(a: &PortArgs) -> anyhow::Result<()> {
    let k = knobs(a);
    view::config::validate_cli(&k)?;
    let row = spark_model::layers::ops::port_rows::kv_view_selection::row();
    if view::config::would_enable(&k, row) {
        view::config::KvViewConfig::from_env(k)?;
        anyhow::ensure!(
            !spark_model::layers::ops::port_rows::kv_host_tier::row(),
            "the bounded KV view (kv_view_selection) cannot run with the host KV tier \
             (kv_host_tier): restored pages bypass the key sketch the view scores; turn one off"
        );
    }
    Ok(())
}

/// `port_apply::apply`: publish the flags and register the selector when on.
pub(crate) fn apply_args(a: &PortArgs) -> anyhow::Result<()> {
    view::config::set_cli(knobs(a))?;
    view::register_if_enabled()
}

/// The `atlas_kv_view_*` series (all zero while the view is off).
pub(crate) fn render_prometheus(out: &mut String) {
    use crate::api::port_metrics::{counter, gauge, hist_sum_count, labeled_counter};
    let m = view::metrics::snapshot();
    counter(
        out,
        "atlas_kv_view_selections_total",
        "Bounded KV view (approximate, off by default): per-turn selections made.",
        m.selections,
    );
    counter(
        out,
        "atlas_kv_view_kept_total",
        "Bounded KV view: continuations that kept their conversation's selection.",
        m.kept,
    );
    counter(
        out,
        "atlas_kv_view_select_seconds_total",
        "Bounded KV view: seconds spent scoring and picking selections.",
        m.select_seconds,
    );
    gauge(
        out,
        "atlas_kv_view_tokens",
        "Bounded KV view: tokens the live views attend, summed at selection.",
        m.view_tokens,
    );
    gauge(
        out,
        "atlas_kv_view_active",
        "Bounded KV view: live views.",
        m.active,
    );
    gauge(
        out,
        "atlas_kv_view_device_bytes",
        "Bounded KV view: device bytes of its buffers, sketch and capture.",
        m.device_bytes,
    );
    gauge(
        out,
        "atlas_kv_view_mtp_dense",
        "Bounded KV view: 1 while on; the MTP head's own attention stays dense (its KV is not \
         the paged pool the view selects from).",
        u8::from(view::enabled()),
    );
    let labels: Vec<(String, u64)> = m
        .skipped
        .iter()
        .map(|(l, v)| (format!("reason=\"{l}\""), *v))
        .collect();
    let series: Vec<(&str, u64)> = labels.iter().map(|(l, v)| (l.as_str(), *v)).collect();
    labeled_counter(
        out,
        "atlas_kv_view_skipped_total",
        "Bounded KV view: turns that got no view, by reason.",
        &series,
    );
    hist_sum_count(
        out,
        "atlas_kv_view_recall_mass",
        "Bounded KV view: fraction of the turn's block score mass inside the view.",
        m.recall_sum,
        m.recall_count,
    );
}

#[cfg(test)]
#[path = "kv_view_tests.rs"]
mod tests;
