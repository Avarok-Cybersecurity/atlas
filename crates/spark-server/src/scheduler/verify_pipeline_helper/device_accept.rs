// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the device-side speculative acceptance hook of the verify pick (lane L12, row
//! `mtp_device_accept`): plan the span, hand it to the model's device path, walk the results.
//!
//! `verify_pick_all_with_pipeline` calls [`try_device_picks`] after its fast paths and before its slow
//! tail (the `[K, vocab]` D2H, the host dequant, the pipeline, the argmax or sample). With the row off
//! the call returns `None` at its first line and the slow tail runs exactly as before. With it on, a
//! row the planner accepts is picked on the device:
//!
//! * greedy rows: the same picks as the slow tail, bit for bit (same edits, same f32 operations, the
//!   same first-index-wins argmax), including a span that crosses `</think>`;
//! * sampled rows: the same distribution per position (decision D10), one-hot draft acceptance with the
//!   residual when a draft is passed, a direct sample otherwise; the RNG is keyed by (seed or a fresh
//!   key, absolute position), not the host sampler's stream.
//!
//! The hook passes no drafts: the callers own them (`verify_k*_step`, the batched step), and with a
//! one-hot draft distribution "accept d with p(d), else sample the residual" and "sample p" give every
//! position the same marginal. [`picks_via`] takes any device function, so the tests drive it with the
//! CPU reference over host logits.
//!
//! Nothing here mutates the sequence: on any fallback (counted by reason in
//! `atlas_mtp_device_accept_fallback_total`) the caller's slow tail runs from the untouched state.

use std::time::Instant;

use spark_model::port::spec_accept::metrics::{self, Fallback};
use spark_model::port::spec_accept::{AcceptRequest, JobResult, status};
use spark_model::traits::Model;
use spark_runtime::gpu::DevicePtr;

use crate::scheduler::ActiveSeq;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::mtp_timing::Phase;

pub(super) mod plan;

/// The verify picks of `argmax_ids.len()` positions from the device, or `None` to run the host tail.
///
/// `row_base` is the first logits row of this sequence's span in the model's shared buffer (the same
/// argument the caller's slow tail reads with).
pub(super) fn try_device_picks(
    model: &dyn Model,
    argmax_ids: &[u32],
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
    row_base: usize,
) -> Option<Vec<u32>> {
    if !spark_model::layers::ops::port_rows::mtp_device_accept::row() {
        return None;
    }
    let vocab = model.vocab_size();
    let base = model.logits_buffer_ptr();
    if model.logits_ptr_is_fp32(base) {
        return fallback(Fallback::Fp32Logits);
    }
    let Some(ext) = model.port_ext() else {
        return fallback(Fallback::Unavailable);
    };
    let logits = base.offset(row_base * vocab * 2);
    picks_via(argmax_ids.len(), vocab, logits, a, ctx, |req| {
        ext.spec_accept(req)
    })
}

/// [`try_device_picks`] after the model checks, over any device function (the model's
/// `spec_accept`, or the CPU reference in tests).
pub(super) fn picks_via(
    k: usize,
    vocab: usize,
    logits: DevicePtr,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
    device: impl FnOnce(&AcceptRequest<'_>) -> anyhow::Result<Option<Vec<JobResult>>>,
) -> Option<Vec<u32>> {
    let t = Instant::now();
    let plan = match plan::build(vocab, k, a, ctx) {
        Ok(p) => p,
        Err(reason) => return fallback(reason),
    };
    let req = AcceptRequest {
        logits,
        vocab,
        n_pos: k,
        drafts: &[],
        mode: plan.mode,
        rng_key: plan.rng_key,
        pos_base: plan.pos_base,
        eos_until: plan.eos_until,
        rep_penalty: plan.rep_penalty,
        regimes: &plan.regimes,
    };
    let results = match device(&req) {
        Ok(Some(r)) => r,
        Ok(None) => return fallback(Fallback::Unavailable),
        Err(e) => {
            if ctx.stats.once("log:mtp_device_accept_error") {
                tracing::warn!(
                    "mtp_device_accept: device pick failed, verify stays on the host pipeline \
                     for this call: {e:#}"
                );
            }
            return fallback(Fallback::DeviceError);
        }
    };
    let picks = match walk(&results, k, plan.regimes.len(), vocab, ctx.think_end_token) {
        Ok(p) => p,
        Err(reason) => return fallback(reason),
    };
    // One device call replaces the D2h, Dequant, PipelineProc and Argmax phases of the slow tail; it
    // is recorded under Argmax so the AVAROK_MTP_TIMING phase set is unchanged.
    ctx.timing.record(Phase::Argmax, t);
    metrics::record_rows(k);
    if ctx.stats.once("log:mtp_device_accept") {
        tracing::info!(
            "verify device acceptance ACTIVE: verify picks on the device, no [K, vocab] D2H \
             (row mtp_device_accept; AVAROK_MTP_DEVICE_ACCEPT=0 restores the host pipeline)"
        );
    }
    Some(picks)
}

/// The span's picks from the job results: regime 0 until the first `</think>` pick of a span that
/// starts inside thinking, regime 1 after it (the host loop's flag flip). Any job the walk uses that
/// reports a fallback, or a pick outside the vocabulary, sends the whole call to the host.
pub(super) fn walk(
    results: &[JobResult],
    k: usize,
    n_regimes: usize,
    vocab: usize,
    think_end: Option<u32>,
) -> Result<Vec<u32>, Fallback> {
    if n_regimes == 0 || results.len() != k * n_regimes {
        return Err(Fallback::DeviceError);
    }
    let mut regime = 0usize;
    let mut picks = Vec::with_capacity(k);
    for i in 0..k {
        let r = results[regime * k + i];
        if r.status & status::EMPTY != 0 {
            return Err(Fallback::Empty);
        }
        if r.status & status::SUPPORT != 0 {
            return Err(Fallback::Support);
        }
        if r.pick as usize >= vocab {
            return Err(Fallback::DeviceError);
        }
        picks.push(r.pick);
        if regime == 0 && n_regimes > 1 && think_end == Some(r.pick) {
            regime = 1;
        }
    }
    Ok(picks)
}

fn fallback(reason: Fallback) -> Option<Vec<u32>> {
    metrics::record_fallback(reason);
    None
}
