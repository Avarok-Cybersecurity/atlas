// SPDX-License-Identifier: AGPL-3.0-only

//! Per-stream fallback for a DECLINED batched-prefill wave, and the
//! advance-and-sample bookkeeping both dispatch paths share.
//!
//! A batched forward can refuse a wave for reasons that have nothing to do
//! with the requests in it (an arena that cannot hold the stacked chunk, a
//! geometry the kernel path does not cover). When the refusal is raised BEFORE
//! any stream was mutated — `spark_model::traits::BatchedPrefillDeclined` — the
//! streams are exactly as the scheduler handed them over and the correct
//! response is to prefill them one at a time, not to drop them.
//!
//! The scheduler used to drop them: `run_batched_prefill_step` pushed
//! `(i, None)` for every member and `promote_completed_prefills` freed the
//! sequence without touching its sink. On a streaming request the SSE body was
//! already committed with a 200, so the client saw a stream that ended with no
//! content, no `finish_reason` and no usage frame — `err=0`, nothing to retry
//! on. Measured on H100 round 11 (#927) cell E: `ATLAS_PREFILL_VARLEN=1` at
//! C=16 returned sixteen empty 200s on 3/3 reps, and intermittently dropped 3
//! of 16 on an otherwise-clean run. Silent truncation is the worst failure
//! shape available: it is indistinguishable from a correct empty answer.
//!
//! Two rules come out of that, and both are enforced here and in
//! `phase_promote_prefills`:
//!   1. a decline re-runs the whole wave per-stream, and
//!   2. a stream that still fails gets an error frame — never a closed channel.

use spark_model::traits::{Model, PrefillSlice};
use spark_runtime::gpu::DevicePtr;

use super::super::types::PrefillInProgress;
use super::super::{FirstTokenPolicy, sample_first_token};

/// Re-run one declined wave's members through the single-stream dispatch.
///
/// Each member is handed to `prefill_batch_chunk` on its own: at `n == 1` the
/// model's dispatcher delegates straight to the single-stream path, which is
/// the same code the non-batched scheduler runs. A member that fails on its own
/// fails alone — the rest of the wave still completes.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_wave_per_stream(
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    prefilling: &mut [PrefillInProgress],
    completed_indices: &mut Vec<(usize, Option<u32>)>,
    wave: &[usize],
    chunk_lens: &[usize],
    is_last_flags: &[bool],
    n: usize,
    prefill_stream: u64,
    prefill_event: u64,
    think_end_token: Option<u32>,
    tool_call_start_token: Option<u32>,
) {
    for &i in wave {
        let logits = {
            let p = &mut prefilling[i];
            let mut one = [PrefillSlice {
                prompt_tokens: &p.prompt_tokens,
                seq: &mut p.seq,
                chunk_start: p.chunk_offset,
                chunk_len: chunk_lens[i],
                is_last_chunk: is_last_flags[i],
            }];
            match model.prefill_batch_chunk(&mut one, prefill_stream) {
                Ok(v) => v.first().copied().unwrap_or(DevicePtr::NULL),
                Err(e) => {
                    tracing::error!(
                        "Per-stream prefill fallback[{i}/{n}] failed after a declined wave: {e:#}",
                    );
                    completed_indices.push((i, None));
                    continue;
                }
            }
        };
        let _ = model.record_event(prefill_event, prefill_stream);
        let _ = model.stream_wait_event(model.default_stream(), prefill_event);
        advance_and_sample(
            model,
            sched,
            &mut prefilling[i],
            i,
            n,
            chunk_lens[i],
            is_last_flags[i],
            logits,
            completed_indices,
            think_end_token,
            tool_call_start_token,
        );
    }
}

/// Advance one stream's `chunk_offset` and, when the chunk it just ran was its
/// last, sample the first token into `completed_indices`.
///
/// SSOT for both dispatch paths (the batched wave and the per-stream fallback)
/// so a stream cannot be advanced on one path and not the other.
#[allow(clippy::too_many_arguments)]
pub(super) fn advance_and_sample(
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    p: &mut PrefillInProgress,
    i: usize,
    n: usize,
    chunk_len: usize,
    is_last: bool,
    logits: DevicePtr,
    completed_indices: &mut Vec<(usize, Option<u32>)>,
    think_end_token: Option<u32>,
    tool_call_start_token: Option<u32>,
) {
    p.chunk_offset += chunk_len;
    if !is_last {
        return;
    }
    if logits == DevicePtr::NULL {
        tracing::error!(
            "Batched prefill: stream {i} marked is_last but model returned NULL logits"
        );
        completed_indices.push((i, None));
        return;
    }
    // #131: grammar-constrain the FIRST token (and advance the matcher);
    // no-op without a grammar.
    // P1-4 (2026-07-09): thread the resolved `min_p` — previously a
    // hardcoded 0.0 inside the sampler. Kill-switch: ATLAS_NO_MTP_MINP=1.
    match sample_first_token(
        model,
        logits,
        p.temperature,
        p.top_k,
        p.top_p,
        p.min_p,
        &p.eos_tokens,
        p.grammar_state.as_mut(),
        FirstTokenPolicy::for_birth(p.enable_thinking, think_end_token, tool_call_start_token),
        &sched.levers.sampling(),
    ) {
        Ok(first) => {
            tracing::info!(
                "Batched prefill[{i}/{n}] first token: {first} (chunk_len={chunk_len}, \
                 total_tokens={})",
                p.prompt_tokens.len(),
            );
            completed_indices.push((i, Some(first)));
        }
        Err(e) => {
            tracing::error!("Batched prefill[{i}] sampling: {e:#}");
            completed_indices.push((i, None));
        }
    }
}
