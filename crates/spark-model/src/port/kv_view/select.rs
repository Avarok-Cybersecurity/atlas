// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the bounded view's per-turn selection (port lane L32), run
//! by `set_kv_view` right after the final prefill chunk of a sequence:
//!
//! 1. a context no longer than the budget gets no view (the identity);
//! 2. a continuation (tool result, assistant continuation) keeps its
//!    conversation's selection, bound to the new sequence's blocks, while the
//!    grown view stays under twice the budget;
//! 3. otherwise the blocks are scored against the captured query rows (L31's
//!    scorer, on the prefill stream after the prefill's writes), spread one
//!    hop (`AVAROK_KV_VIEW_GRAPH_ALPHA`), ranked by the decayed history
//!    (`AVAROK_KV_VIEW_HISTORY_LAMBDA`), picked (sink, recent, the question,
//!    then the best blocks within the budget), and the view is bound to the
//!    sequence on the device;
//! 4. the conversation remembers the selection and this round's ranking.

use std::time::Instant;

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::config::KvViewConfig;
use super::core_api::PickParams;
use super::history::propagate;
use super::metrics::{Counters, SkipReason};
use super::state::{Arm, Conv, Inner};
use super::view::{Desc, mandatory_blocks, pages_of, prefix_hash, shape_from_pick};
use super::{TurnKind, ViewOutcome, ViewSpec};

/// Debug (`AVAROK_KV_VIEW_SELECTION_LOG`, off by default): one log line per
/// attention layer naming the logical KV pages (Atlas KV blocks) of the view a
/// selection chose, in the shape `bench/niah_grid.py --selection-log` reads:
/// `seq=<pool slot> layer=<n> ... blocks=[b0,b1,...]`. One selection serves
/// every layer, so the lines of a selection are identical but for `layer`.
/// Decode steps add only tail pages to the view and run as replayed graphs
/// with no host code, so the selection is the only point that is logged.
fn log_selection(slot: u64, layers: u32, detail: &str, pages: impl Iterator<Item = u32>) {
    let list = pages.map(|p| p.to_string()).collect::<Vec<_>>().join(",");
    for layer in 0..layers {
        tracing::info!("kv_view selection seq={slot} layer={layer} {detail} blocks=[{list}]");
    }
}

impl Inner {
    /// Select a view for `slot`'s sequence, whose prefill just completed.
    pub(crate) fn select(
        &mut self,
        cfg: &KvViewConfig,
        gpu: &dyn GpuBackend,
        slot: u64,
        spec: &ViewSpec<'_>,
        counters: &Counters,
    ) -> Result<ViewOutcome> {
        if !self.ready(gpu) {
            return Ok(self.off.map_or(ViewOutcome::Off, ViewOutcome::Skipped));
        }
        let Some(arm) = self.arms.remove(&slot) else {
            counters.skip(SkipReason::NoQuery);
            return Ok(ViewOutcome::Skipped(SkipReason::NoQuery));
        };
        self.deactivate_owner(gpu, slot, counters)?;
        let out = self.select_arm(cfg, gpu, slot, &arm, spec, counters);
        if let Some(core) = &self.core {
            core.disarm(slot);
        }
        if let Ok(ViewOutcome::Skipped(r)) = &out {
            counters.skip(*r);
        }
        out
    }

    fn select_arm(
        &mut self,
        cfg: &KvViewConfig,
        gpu: &dyn GpuBackend,
        slot: u64,
        arm: &Arm,
        spec: &ViewSpec<'_>,
        counters: &Counters,
    ) -> Result<ViewOutcome> {
        let skip = |r| Ok(ViewOutcome::Skipped(r));
        let Some(geom) = self.geom else {
            return skip(SkipReason::NotReady);
        };
        let bs = geom.block_size;
        let seq_len = spec.seq_len;
        if seq_len <= cfg.budget_tokens {
            return skip(SkipReason::ShortContext);
        }
        let n_pages = pages_of(seq_len, bs);
        if spec.block_table.len() < n_pages as usize {
            return skip(SkipReason::ScoreError);
        }
        let pages = &spec.block_table[..n_pages as usize];

        if arm.turn.kind == TurnKind::Continuation
            && let Some(prior) = &arm.prior
            && let Some(desc) = Desc::bind(prior.shape.clone(), pages, seq_len, bs)
        {
            let view_tokens = seq_len - prior.shape.drop_tokens(bs);
            let overflow = view_tokens > cfg.budget_tokens.saturating_mul(2);
            if !overflow || arm.kept_span.is_none() {
                self.activate(gpu, slot, desc, view_tokens, counters)?;
                counters.kept();
                return Ok(ViewOutcome::Kept {
                    context_tokens: seq_len,
                    view_tokens,
                });
            }
            tracing::debug!(
                "bounded KV view: slot {slot}'s kept view grew to {view_tokens} tokens (budget {}); \
                 selecting again",
                cfg.budget_tokens
            );
        }
        if arm.gap {
            return skip(SkipReason::SketchGap);
        }
        let Some(span) = arm.kept_span.clone() else {
            return skip(if arm.busy {
                SkipReason::CaptureBusy
            } else {
                SkipReason::NoQuery
            });
        };

        let bp = geom.block_pages();
        let bt = geom.block_tokens();
        let n_blocks = n_pages.div_ceil(bp) as usize;
        let (Some(dev), Some(core)) = (self.dev.as_ref(), self.core.as_ref()) else {
            return skip(SkipReason::NotReady);
        };
        let (scores_dev, score_cap) = dev.scores();
        if n_blocks > score_cap {
            return skip(SkipReason::ScoreError);
        }
        let t0 = Instant::now();
        // The prefill's K, sketch and capture writes are on its stream (a
        // mixed forward runs them on the decode stream).
        gpu.synchronize(gpu.default_stream())?;
        gpu.synchronize(spec.prefill_stream)?;
        if !core.score(gpu, pages, slot, scores_dev, spec.prefill_stream)? {
            return skip(SkipReason::NoQuery);
        }
        gpu.synchronize(spec.prefill_stream)?;
        let mut bytes = vec![0u8; n_blocks * 4];
        gpu.copy_d2h(scores_dev, &mut bytes)?;
        let raw: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if raw.iter().any(|s| !s.is_finite()) {
            return skip(SkipReason::ScoreError);
        }

        let spread = propagate(&raw, &[], cfg.graph_alpha);
        let history = arm
            .prior
            .as_ref()
            .map(|p| p.history.clone())
            .unwrap_or_default();
        let rank = history.preview(&spread, cfg.history_lambda);
        let params = PickParams {
            budget_blocks: cfg.budget_tokens / bt,
            sink_blocks: cfg.sink_tokens.div_ceil(bt),
            recent_blocks: cfg.recent_tokens.div_ceil(bt),
            mandatory: mandatory_blocks(span.start.min(seq_len)..span.end.min(seq_len), bt),
        };
        let picked = core.pick(&rank, &params);
        let Some(shape) = shape_from_pick(&picked, n_pages, bp) else {
            return skip(SkipReason::NothingDropped);
        };
        if shape.sel_pages.len() > dev.sel_cap() as usize {
            return skip(SkipReason::ScoreError);
        }
        let recall = core.recall_mass(&raw, &picked);
        let view_tokens = seq_len - shape.drop_tokens(bs);
        let Some(desc) = Desc::bind(shape, pages, seq_len, bs) else {
            return skip(SkipReason::ScoreError);
        };
        tracing::debug!(
            "kv_view selection slot={slot} view_blocks={}/{n_blocks} tokens={view_tokens}/{seq_len} \
             recall_mass={recall:.3}",
            picked.len(),
        );
        if super::config::selection_log() {
            let detail = format!(
                "view_blocks={}/{n_blocks} tokens={view_tokens}/{seq_len} recall_mass={recall:.3}",
                picked.len()
            );
            let pages = desc
                .shape
                .sel_pages
                .iter()
                .copied()
                .chain(desc.shape.tail_page..n_pages);
            log_selection(slot, geom.attn_layers, &detail, pages);
        }
        let shape = desc.shape.clone();
        self.activate(gpu, slot, desc, view_tokens, counters)?;
        counters.selected(t0.elapsed().as_nanos() as u64, recall);

        let match_len = arm
            .turn
            .match_len
            .min(seq_len)
            .min(spec.prompt.len() as u32);
        if arm.turn.kind != TurnKind::Unmarked && shape.tail_page * bs <= match_len {
            let conv = Conv {
                match_len,
                match_hash: prefix_hash(&spec.prompt[..match_len as usize]),
                prompt_len: spec.prompt.len() as u32,
                shape,
                history,
                pending: (cfg.history_lambda > 0.0).then_some(rank),
                stamp: 0,
            };
            self.remember(arm.session, conv);
        } else {
            self.convs.remove(&arm.session);
        }
        Ok(ViewOutcome::Selected {
            context_tokens: seq_len,
            view_tokens,
            recall_mass: recall,
        })
    }
}
