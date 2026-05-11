// SPDX-License-Identifier: AGPL-3.0-only

//! Phase: continue in-progress chunked prefills. When `active` is empty,
//! all chunks run back-to-back (TTFT minimisation). When active is
//! nonempty, exactly one chunk runs per scheduler iteration to bound
//! TPOT — except when mixed_forward fuses a prefill chunk + decode in a
//! single pass.
//!
//! Returns `did_mixed_step` so the caller can skip the standalone decode
//! call (mixed forward already processed decode logits).
//!
//! Layout: this file is the dispatcher only; the three per-path bodies
//! live in the sibling sub-modules under `phase_continue_prefills/` to
//! keep each unit ≤250 LoC per `crates/.../CLAUDE.md` core directive #4
//! and ≤500 LoC per `.github/workflows/file-size-cap.yml`.
//!
//!  - `run_standard`        — single-stream chunked-prefill body
//!                            (mixed_forward or plain prefill_chunk).
//!  - `run_batched_prefill` — Q12 N-stream batched-prefill step.
//!  - `run_batched_mixed`   — Q12 Phase 5 batched mixed (decode+prefill) step.

#[path = "phase_continue_prefills/run_batched_mixed.rs"]
mod run_batched_mixed;
#[path = "phase_continue_prefills/run_batched_prefill.rs"]
mod run_batched_prefill;
#[path = "phase_continue_prefills/run_standard.rs"]
mod run_standard;

use std::time::Instant;

use spark_model::traits::Model;

use super::phase_promote_prefills::promote_completed_prefills;
use super::sample_token;
use super::types::{ActiveSeq, PrefillInProgress};
use crate::scheduling_policy::{ActiveSeqTiming, SchedulingPolicy};

use run_batched_mixed::run_batched_mixed_step;
use run_batched_prefill::run_batched_prefill_step;
use run_standard::run_standard_chunk_loop;

#[allow(clippy::too_many_arguments)]
pub(super) fn continue_in_progress_prefills(
    model: &dyn Model,
    policy: &dyn SchedulingPolicy,
    active: &mut Vec<ActiveSeq>,
    prefilling: &mut Vec<PrefillInProgress>,
    max_prefill_tokens: usize,
    prefill_stream: u64,
    prefill_event: u64,
    use_mtp: bool,
    use_self_speculative: bool,
    use_ngram_speculative: bool,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    adaptive_sampling: bool,
) -> bool {
    let mut did_mixed_step = false;

    if prefilling.is_empty() {
        return did_mixed_step;
    }

    // Check policy: skip chunks if active sequences are near TBT deadline.
    let timings: Vec<ActiveSeqTiming> = active
        .iter()
        .map(|a| ActiveSeqTiming {
            last_token_time: a.last_token_time,
        })
        .collect();
    let do_chunks = active.is_empty() || policy.should_prefill(&timings);

    if !do_chunks {
        return did_mixed_step;
    }

    let mut completed_indices = Vec::new();

    // ── Batched-prefill paths (Q12) ──
    //
    // Two branches fire when 2+ streams are prefilling concurrently. Both
    // replace the FIFO `prefilling.first_mut()` advance that caused the
    // asymmetric 24+131 s TTFT documented in qwen-refactor notes §6.
    //
    // Phase 4a (commit 2ff926d): active.is_empty() case → prefill_batch_chunk.
    // Phase 5 (commit a542463): active.is_nonempty() case → mixed_forward_batch
    // (N decode tokens + M prefill chunks fused). Lifts the implicit MTP /
    // self-spec / N-gram-spec gating since those only apply to active
    // sequences, not freshly-prefilling ones — the active-side decode is
    // still handled by `step_decode_only` / `step_mtp` etc. via
    // `process_decode_logits` on the returned decode logits.
    //
    // Phase 4a/5 use the default trait impls (per-stream loops). No
    // kernel-level batching yet — the win is fairness/TTFT distribution.
    // Phase 2/3 of the plan replace the default impls with concrete
    // batched dispatch (true L2-amortised weight load + batched GDN/attn).
    //
    // Gates:
    //   - `prefilling.len() >= 2` — single-stream stays on the existing
    //     two-phase / chunked / mixed_forward path (preserves correctness
    //     and the long-prompt two-phase optimisation).
    //   - `!model.is_ep()` — EP=2 needs a new BATCH_PREFILL_CHUNK opcode
    //     (Phase 6) to broadcast batched chunks to the worker rank.
    //   - For the mixed-batch branch only: skip when active.len() == 1 AND
    //     a speculative path is active. Speculative decode (`step_mtp`,
    //     `step_self_spec`, `step_ngram`) handles its own forward; mixing
    //     it with `mixed_forward_batch` would double-decode the active
    //     stream. With more than one active sequence, speculative is off
    //     by construction (those step_* paths require active.len()==1) so
    //     the mixed branch is safe.
    let single_active_with_spec = active.len() == 1
        && (use_mtp || use_self_speculative || use_ngram_speculative);
    // BISECT: ATLAS_BISECT_Q12_DISABLE=1 forces the per-stream FIFO path
    // (pre-Q12 behavior) so we can isolate whether the chunked-prefill +
    // concurrent-decode crash originates in the Q12 batched-prefill
    // dispatch or pre-existing chunked-prefill state mutation.
    let q12_dispatch_disabled = std::env::var("ATLAS_BISECT_Q12_DISABLE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    let can_batch_prefill_only = !q12_dispatch_disabled
        && prefilling.len() >= 2
        && active.is_empty()
        && !model.is_ep();
    let can_batch_mixed = !q12_dispatch_disabled
        && prefilling.len() >= 2
        && !active.is_empty()
        && !single_active_with_spec
        && !model.is_ep();

    if can_batch_prefill_only {
        run_batched_prefill_step(
            model,
            prefilling,
            &mut completed_indices,
            max_prefill_tokens,
            prefill_stream,
            prefill_event,
        );
        promote_completed_prefills(
            model,
            prefilling,
            completed_indices,
            active,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
        );
        return did_mixed_step;
    }

    if can_batch_mixed {
        let t0_mixed = Instant::now();
        run_batched_mixed_step(
            model,
            active,
            prefilling,
            &mut completed_indices,
            max_prefill_tokens,
            prefill_stream,
            prefill_event,
            t0_mixed,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
            reflection_suppress_ids,
            adaptive_sampling,
            &mut did_mixed_step,
        );
        promote_completed_prefills(
            model,
            prefilling,
            completed_indices,
            active,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
        );
        return did_mixed_step;
    }

    // Process the FIRST in-progress prefill. When no active decode
    // sequences, run all remaining chunks in a tight loop to minimize
    // TTFT. Otherwise, run 1 chunk and yield to decode.
    if let Some(p) = prefilling.first_mut() {
        let idx = 0usize;

        // Two-phase SSM prefill: when the full sequence hasn't started
        // chunking yet (chunk_offset == 0) and is longer than one chunk,
        // use the two-phase path for better SSM state quality.
        let use_twophase = p.chunk_offset == 0 && p.prompt_tokens.len() > max_prefill_tokens;
        if use_twophase {
            tracing::info!(
                "Two-phase prefill: {} tokens, chunk_size={}",
                p.prompt_tokens.len(),
                max_prefill_tokens,
            );
            match model.prefill_twophase(
                &p.prompt_tokens,
                &mut p.seq,
                max_prefill_tokens,
                prefill_stream,
            ) {
                Ok(logits) => {
                    p.chunk_offset = p.prompt_tokens.len();
                    let _ = model.record_event(prefill_event, prefill_stream);
                    let _ = model.stream_wait_event(model.default_stream(), prefill_event);
                    match sample_token(
                        model,
                        logits,
                        p.temperature,
                        p.top_k,
                        p.top_p,
                        &p.eos_tokens,
                    ) {
                        Ok(first) => {
                            tracing::info!("Two-phase prefill first token: {first}");
                            completed_indices.push((idx, Some(first)));
                        }
                        Err(e) => {
                            tracing::error!("Two-phase prefill sampling: {e:#}");
                            completed_indices.push((idx, None));
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Two-phase prefill failed, falling back to chunked: {e:#}");
                    // Fall through to the standard chunk loop below
                }
            }
        }

        // Standard chunked prefill (also used as fallback if two-phase fails)
        if p.chunk_offset < p.prompt_tokens.len() {
            run_standard_chunk_loop(
                model,
                p,
                idx,
                active,
                max_prefill_tokens,
                prefill_stream,
                prefill_event,
                use_mtp,
                use_self_speculative,
                use_ngram_speculative,
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
                reflection_suppress_ids,
                adaptive_sampling,
                &mut completed_indices,
                &mut did_mixed_step,
            );
        }
    }

    // Move completed prefills to active (or free on error).
    promote_completed_prefills(
        model,
        prefilling,
        completed_indices,
        active,
        think_end_token,
        think_start_token,
        tool_call_start_token,
        tool_call_end_token,
    );

    did_mixed_step
}
