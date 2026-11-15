// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the device-side speculative acceptance hook of the verify pick (lane L12, row
//! `mtp_device_accept`): plan the span, hand it to the model's device path, walk the results.
//!
//! `verify_pick_all_with_pipeline` calls [`try_device_picks`] FIRST, before its greedy shortcuts and
//! its slow tail (the `[K, vocab]` D2H, the host dequant, the pipeline, the argmax or sample). With
//! the row off the call returns `None` at its first line and everything after it runs exactly as
//! before. With it on, a row the planner accepts is picked on the device:
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
//! `atlas_mtp_device_accept_fallback_total`) the caller's own paths run from the untouched state.
//!
//! # Why the hook is first, and what that changes with the row on
//!
//! It used to sit after the helper's greedy shortcuts. For a grammar-free greedy row with neutral
//! penalties the grammarless shortcut answers every call with the verify forward's graphed argmax,
//! so on greedy traffic the hook was never reached: an earlier GPU run on this card booted with
//! the row on, ran 675 verify passes and reported 0 device rows and 0 for every reason. Consulted
//! first, each call now records its rows or its reason, and a row the planner accepts is picked
//! by the device path.
//!
//! What a covered greedy row gets with the row on is therefore the host pipeline's pick, computed on
//! the device, where the row off gives it the shortcut's raw argmax. The two differ in exactly the
//! places the shortcut is documented to differ from the pipeline:
//!
//! * an exact top tie: the graphed argmax keeps the tied id with the lower `index mod 1024`
//!   (`argmax_bf16`'s reduction prefers the lower thread), the pipeline and this path the lower
//!   index. Ties are not rare in BF16 logits: 178 of 13,406 greedy positions in the quality dumps
//!   of an earlier GPU run on this card tie exactly at the top (1.3%);
//! * a mask the shortcut does not apply (the `min_tokens` EOS mask, the post-close think mask).
//!
//! So greedy output with the row on is bit-identical to the HOST PIPELINE, which is the row's
//! claim and what `AVAROK_NO_FAST_GREEDY_CHAT=1` serves with the row off; it is not bit-identical
//! to the default row-off serve, whose greedy rows take the shortcut.
//!
//! # The identity screen's off arm (found in an earlier GPU run, fixed 2026-11-13)
//!
//! The screen compared `AVAROK_NO_FAST_GREEDY_CHAT=1` against the same plus the row and found 2 of
//! 8 prompts different. The off arm was not the host pipeline: the masked-verify shortcut
//! (`fast_masked`, "verify chat fast path ACTIVE" in its log) ignored the kill switch and answered
//! with the verify graph's raw argmax, which breaks an exact top tie by reduction order where the
//! host pipeline (and this path) take the lowest id. All eight K4 verify rows that differed between
//! the arms vanished once the off arm really was the host pipeline, and the two that reached the
//! output were exact BF16 ties. `fast_masked` now honors the switch (`kill_switch_tests.rs`). With the old
//! binary the same off arm needs `AVAROK_DISABLE_FAST_MASKED=1` as well; run that way, bin-fix2's
//! device path equals the off arm on all 8 prompts at C=1, the K4 trace (every row of every step,
//! emitted or not) included: `logs/devacc-fix/diag/diag-host*`.
//!
//! # No verify pass without a trace
//!
//! A verify pass that never reaches the helper (DFlash raw-argmax verify) cannot record here, so
//! the MTP step wrapper (`mtp_stats::account_step`) checks the counters around every step:
//! [`account_verify_passes`] counts each pass of a step that moved nothing as `not_consulted`.

use std::time::Instant;

use spark_model::port::spec_accept::metrics::{self, Fallback};
use spark_model::port::spec_accept::{AcceptRequest, JobResult, status};
use spark_model::traits::Model;
use spark_runtime::gpu::DevicePtr;

use crate::scheduler::ActiveSeq;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::mtp_timing::Phase;

pub(super) mod plan;

/// Is the `mtp_device_accept` row on (`AVAROK_MTP_DEVICE_ACCEPT`)? A cached read.
pub(in crate::scheduler) fn row() -> bool {
    spark_model::layers::ops::port_rows::mtp_device_accept::row()
}

/// How many of a step's `passes` verify passes left no trace in the device-accept counters: all of
/// them when the row is on and the counters did not move across the step, none otherwise.
///
/// The counters are process-wide and only the scheduler thread's verify picks move them, so "did
/// not move" means no pass of the step reached the hook. A step in which one pass was recorded and
/// another was not is not caught here; every caller of the pick helper records, so that needs a
/// second, helper-free verify path in the same step.
pub(in crate::scheduler) fn unaccounted_passes(
    row_on: bool,
    passes: u64,
    events_before: u64,
    events_after: u64,
) -> u64 {
    if row_on && passes > 0 && events_after == events_before {
        passes
    } else {
        0
    }
}

/// The per-step check of `mtp_stats::account_step`: `passes` verify passes ran since the counters
/// read `events_before`. With the row on, a step that moved nothing counts every one of its passes
/// as `not_consulted`, so rows and reasons can never both stand still while verify passes run.
pub(in crate::scheduler) fn account_verify_passes(row_on: bool, passes: u64, events_before: u64) {
    let missing = unaccounted_passes(row_on, passes, events_before, metrics::events());
    if missing == 0 {
        return;
    }
    metrics::record_fallbacks(Fallback::NotConsulted, missing);
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        tracing::warn!(
            "mtp_device_accept is on, but a verify step made its picks without the verify pick \
             helper, so the device path was never asked (DFlash raw-argmax verify takes this \
             route). Counted in atlas_mtp_device_accept_fallback_total as not_consulted."
        );
    });
}

/// The verify picks of `argmax_ids.len()` positions from the device, or `None` to run the caller's
/// own paths (its greedy shortcuts, then the host tail). `None` with the row on always comes with
/// one counted reason.
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
    if !row() {
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
