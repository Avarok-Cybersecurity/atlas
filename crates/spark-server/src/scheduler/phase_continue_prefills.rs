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
use super::sample_first_token;
use super::types::{ActiveSeq, PrefillInProgress};
use crate::scheduling_policy::{ActiveSeqTiming, SchedulingPolicy};

use run_batched_mixed::run_batched_mixed_step;
use run_batched_prefill::run_batched_prefill_step;
use run_standard::run_standard_chunk_loop;

/// Shared per-chunk InnerQ poll used by every prefill path (standard /
/// batched-prefill / batched-mixed). `maybe_finalize` is idempotent post
/// activation, and a no-op when `TURBO_INNERQ` was not set at startup —
/// so calling on every chunk costs one OnceLock load in the disabled case.
/// On non-cuda backends the driver doesn't exist (it talks to the CUDA
/// Driver API directly via `atlas_core::registry`), so this collapses to
/// a no-op via the `#[cfg]` gate.
#[cfg(feature = "cuda")]
pub(super) fn poll_innerq() {
    if let Some(driver) = spark_model::layers::qwen3_attention::INNERQ.get()
        && let Err(e) = driver.maybe_finalize(128)
    {
        tracing::warn!("InnerQ maybe_finalize failed: {e:#}");
    }
}

#[cfg(not(feature = "cuda"))]
pub(super) fn poll_innerq() {}

#[allow(clippy::too_many_arguments)]
pub(super) fn continue_in_progress_prefills(
    model: &dyn Model,
    policy: &dyn SchedulingPolicy,
    active: &mut Vec<ActiveSeq>,
    prefilling: &mut Vec<PrefillInProgress>,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    always_mixed: bool,
    prefill_stream: u64,
    prefill_event: u64,
    use_mtp: bool,
    use_self_speculative: bool,
    use_ngram_speculative: bool,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
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

    // single_active_with_spec: active.len()==1 AND a speculative path is
    // active (those step_* paths require active.len()==1 and mixing would
    // double-decode). Computed early because the always-mixed gate below
    // needs it too. (Also reused by the Q12 mixed-batch gate further down.)
    let single_active_with_spec =
        active.len() == 1 && (use_mtp || use_self_speculative || use_ngram_speculative);

    // ── Step 2 (spec): always-on fused mixed step ──
    //
    // slice_budget governs how many prefill tokens a fused mixed step
    // injects. When ATLAS_HOLO_ALWAYS_MIXED is OFF the scheduler is
    // BYTE-IDENTICAL to today: binary should_prefill gate, full-chunk
    // budget (full_chunk == max_prefill_tokens, the current cap).
    //
    // When ON: a request that can fuse a prefill chunk into the active
    // decode (`fusable_mixed`) takes a fused step even when should_prefill
    // would have suppressed it — sized by prefill_slice_budget so the
    // step stays under the TBT target. We only genuinely suppress (early
    // return) for the cases that truly cannot fuse: EP, single-active-spec,
    // no active decode + policy says wait, or the hard-deadline slice==0.
    let mut slice_budget = max_prefill_tokens;
    if always_mixed {
        // Can this iteration fuse a prefill chunk into active decode? The
        // single-stream mixed path (run_standard) requires: active decode,
        // not EP, not a single-active speculative path.
        let fusable_mixed = !active.is_empty() && !model.is_ep() && !single_active_with_spec;
        let slas_ok = active.is_empty() || policy.should_prefill(&timings);
        // Genuine suppress: nothing to fuse and policy says wait.
        if !slas_ok && !fusable_mixed {
            return did_mixed_step;
        }
        // Slice budget only governs the FUSED single-stream path. When the
        // prefill can't fuse (EP / no decode / single-active-spec) leave the
        // budget at full_chunk so the non-fused prefill_chunk keeps today's
        // sizing.
        if fusable_mixed {
            // Compute the prefill slice (cost-driven; 0 == hard-deadline suppress).
            slice_budget = policy.prefill_slice_budget(&timings, max_prefill_tokens);
            // Hard-deadline suppress: decode already past its TBT deadline —
            // skip prefill this tick, decode runs standalone at mod.rs:307.
            if slice_budget == 0 {
                return did_mixed_step;
            }
            // B4: clamp so padded decode + prefill slice fit the hidden
            // buffer (else mixed_forward silently de-fuses to sequential
            // decode_batch + prefill_chunk — weights loaded twice).
            let padded_n = [2usize, 4, 8]
                .iter()
                .copied()
                .find(|&s| s >= active.len())
                .unwrap_or(active.len());
            let fuse_cap = max_batch_tokens.saturating_sub(padded_n).max(4);
            debug_assert!(
                fuse_cap >= 4,
                "fuse cap underflow: max_batch_tokens={max_batch_tokens} padded_n={padded_n}"
            );
            slice_budget = slice_budget.min(fuse_cap);
        }
    } else {
        // Resting production path — unchanged binary gate.
        let do_chunks = active.is_empty() || policy.should_prefill(&timings);
        if !do_chunks {
            return did_mixed_step;
        }
    }

    let mut completed_indices = Vec::new();

    // Q12 batched-prefill paths. Two branches fire when 2+ streams are
    // prefilling concurrently (replaces the FIFO `prefilling.first_mut()`
    // advance — see qwen-refactor notes §6 for the asymmetric-TTFT
    // bug it fixes). The active-empty case routes to `prefill_batch_chunk`;
    // active-nonempty routes to `mixed_forward_batch` (N decode + M
    // prefill fused). Both call the default trait impl today (per-stream
    // loops); Q12 Phase 2/3 replace with kernel-level batched dispatch.
    //
    // Gates: N≥2 prefilling, no EP (worker opcode pending, Phase 6),
    // and for mixed-batch only: skip if active.len()==1 AND a speculative
    // path is active (those step_* paths require active.len()==1 and
    // mixing would double-decode). Spec is off by construction when
    // active.len() ≥ 2, so the mixed branch is safe there.
    // (`single_active_with_spec` computed near the top — reused here.)
    // BISECT: ATLAS_BISECT_Q12_DISABLE=1 forces the per-stream FIFO path
    // (pre-Q12 behavior) so we can isolate whether the chunked-prefill +
    // concurrent-decode crash originates in the Q12 batched-prefill
    // dispatch or pre-existing chunked-prefill state mutation.
    let q12_dispatch_disabled = std::env::var("ATLAS_BISECT_Q12_DISABLE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    // Prompt-logprob collection (legacy echo scoring) is single-stream
    // only: the batched dispatch's hidden-buffer stream offsets are not
    // wired into the collection helper. Rare debug/eval traffic.
    let any_collecting = prefilling
        .iter()
        .any(|p| p.seq.collect_prompt_logprobs.is_some());
    let can_batch_prefill_only = !q12_dispatch_disabled
        && !any_collecting
        && prefilling.len() >= 2
        && active.is_empty()
        && !model.is_ep();
    // When ATLAS_HOLO_ALWAYS_MIXED is on, COLLAPSE the multi-prefill+decode
    // case onto the single-stream fused path below (FIFO head prefill fused
    // with all active decodes via mixed_forward, sized by the slice budget)
    // instead of the serializing N-stream run_batched_mixed_step — that batched
    // path does NOT keep decode flowing, so with 2+ concurrent prefills the
    // fused step never fired (burst-TBT A/B showed Mixed forward = 0 and no TBT
    // improvement). The non-head prefill streams advance on subsequent ticks.
    let can_batch_mixed = !always_mixed
        && !q12_dispatch_disabled
        && !any_collecting
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
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
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
        //
        // Step 1 (spec blocker B3): ONLY when no decode is active. The
        // two-phase path runs the ENTIRE prompt as one monolithic forward
        // with no decode fused and ignoring the slice budget — on tick 1
        // of a long prefill that starves every active decode for the whole
        // prompt. With decodes active, force the chunked mixed path below
        // so tick 1 also fuses decode and respects the slice budget.
        // B3 fix is gated behind always_mixed so the flag-OFF path stays
        // byte-identical to today (the `active.is_empty()` guard only applies
        // when always-mixed is enabled; otherwise the original two-phase
        // condition is preserved exactly).
        let use_twophase = (!always_mixed || active.is_empty())
            && p.chunk_offset == 0
            && p.prompt_tokens.len() > max_prefill_tokens;
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
                    // #131: grammar-constrain the FIRST token (and advance the
                    // matcher); no-op without a grammar.
                    // P1-4 (2026-07-09): thread the resolved `min_p` —
                    // previously a hardcoded 0.0 inside the sampler.
                    // Kill-switch: ATLAS_NO_MTP_MINP=1.
                    match sample_first_token(
                        model,
                        logits,
                        p.temperature,
                        p.top_k,
                        p.top_p,
                        p.min_p,
                        &p.eos_tokens,
                        p.grammar_state.as_mut(),
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
                slice_budget,
                prefill_stream,
                prefill_event,
                use_mtp,
                use_self_speculative,
                use_ngram_speculative,
                think_end_token,
                think_start_token,
                code_fence_token,
                tool_call_start_token,
                tool_call_end_token,
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
