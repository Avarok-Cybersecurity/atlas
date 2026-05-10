// SPDX-License-Identifier: AGPL-3.0-only

//! Phase: continue in-progress chunked prefills. When `active` is empty,
//! all chunks run back-to-back (TTFT minimisation). When active is
//! nonempty, exactly one chunk runs per scheduler iteration to bound
//! TPOT — except when mixed_forward fuses a prefill chunk + decode in a
//! single pass.
//!
//! Returns `did_mixed_step` so the caller can skip the standalone decode
//! call (mixed forward already processed decode logits).

use anyhow::Result;
use spark_model::traits::{Model, PrefillSlice, SequenceState};
use spark_runtime::gpu::DevicePtr;
use std::time::Instant;

use super::decode_logits_step::process_decode_logits;
use super::phase_promote_prefills::promote_completed_prefills;
use super::*;
use crate::scheduling_policy::{ActiveSeqTiming, SchedulingPolicy};

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
    // Phase 5 (this commit): active.is_nonempty() case → mixed_forward_batch
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
    let can_batch_prefill_only = prefilling.len() >= 2
        && active.is_empty()
        && !model.is_ep();
    let can_batch_mixed = prefilling.len() >= 2
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

/// Inner loop: try mixed_forward first when conditions allow; else fall
/// back to plain prefill_chunk + EP broadcast.
#[allow(clippy::too_many_arguments)]
fn run_standard_chunk_loop(
    model: &dyn Model,
    p: &mut PrefillInProgress,
    idx: usize,
    active: &mut Vec<ActiveSeq>,
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
    completed_indices: &mut Vec<(usize, Option<u32>)>,
    did_mixed_step: &mut bool,
) {
    loop {
        let remaining = p.prompt_tokens.len() - p.chunk_offset;
        // MLA correctness gate: Atlas has no `prefill_attention_paged_mla_*`
        // kernel; the existing MLA prefill at qwen3_attention/prefill.rs:1723
        // only attends over the current chunk's K/V, so multi-chunk prefill
        // silently corrupts attention output. Force single-chunk until a
        // paged-MLA prefill kernel lands. Hurts cold TTFT on long MLA
        // prompts but preserves correctness.
        let effective_max = if model.is_mla() {
            remaining
        } else {
            max_prefill_tokens
        };
        let mut chunk_len = remaining.min(effective_max);
        let is_last = p.chunk_offset + chunk_len >= p.prompt_tokens.len();
        // Align intermediate chunks to GDN WY4 boundary (4 tokens).
        if !is_last && chunk_len >= 4 {
            chunk_len = (chunk_len / 4) * 4;
        }

        // ── Mixed forward: fuse prefill chunk + decode in one pass ──
        let can_mix = !active.is_empty()
            && !model.is_ep()
            && !use_mtp
            && !use_self_speculative
            && !use_ngram_speculative;

        if can_mix {
            let decode_tokens: Vec<u32> = active.iter().map(|a| a.last_token).collect();
            let mut decode_refs: Vec<&mut SequenceState> =
                active.iter_mut().map(|a| &mut a.seq).collect();
            let t0_mixed = Instant::now();

            match model.mixed_forward(
                &decode_tokens,
                &mut decode_refs,
                &p.prompt_tokens,
                &mut p.seq,
                p.chunk_offset,
                chunk_len,
                is_last,
                prefill_stream,
            ) {
                Ok(result) => {
                    p.chunk_offset += chunk_len;
                    tracing::info!(
                        "Mixed forward: prefill {}/{} tokens + {} decode",
                        p.chunk_offset,
                        p.prompt_tokens.len(),
                        decode_tokens.len(),
                    );

                    // Process prefill logits (if last chunk).
                    if is_last {
                        if let Err(e) = model.normalize_ssm_states(&p.seq, prefill_stream) {
                            tracing::warn!("SSM state normalization failed: {e:#}");
                        }
                        let _ = model.record_event(prefill_event, prefill_stream);
                        let _ = model.stream_wait_event(model.default_stream(), prefill_event);
                        match sample_token(
                            model,
                            result.prefill_logits,
                            p.temperature,
                            p.top_k,
                            p.top_p,
                            &p.eos_tokens,
                        ) {
                            Ok(first) => {
                                tracing::info!("Mixed prefill first token: {first}");
                                completed_indices.push((idx, Some(first)));
                            }
                            Err(e) => {
                                tracing::error!("Mixed prefill sampling: {e:#}");
                                completed_indices.push((idx, None));
                            }
                        }
                    }

                    // Process decode logits for active sequences.
                    let _ = model.record_event(prefill_event, prefill_stream);
                    let _ = model.stream_wait_event(model.default_stream(), prefill_event);
                    process_decode_logits(
                        model,
                        active,
                        result.decode_logits,
                        t0_mixed,
                        think_end_token,
                        think_start_token,
                        tool_call_start_token,
                        tool_call_end_token,
                        reflection_suppress_ids,
                        adaptive_sampling,
                    );
                    *did_mixed_step = true;
                }
                Err(e) => {
                    tracing::error!("Mixed forward error: {e:#}");
                    completed_indices.push((idx, None));
                }
            }
            break;
        }

        // ── Standard path: prefill chunk only, decode separately ──
        // EP: broadcast chunk tokens to worker (bulk, single NCCL op).
        let ep_ok = (|| -> Result<()> {
            model.ep_broadcast_cmd(0xFFFFFFF0)?;
            model.ep_broadcast_cmd(chunk_len as u32)?;
            model.ep_broadcast_cmd(p.chunk_offset as u32)?;
            model.ep_broadcast_cmd(p.prompt_tokens.len() as u32)?;
            model.ep_broadcast_tokens(&p.prompt_tokens)?;
            Ok(())
        })();
        if let Err(e) = ep_ok {
            tracing::error!("EP broadcast chunk: {e:#}");
            completed_indices.push((idx, None));
            break;
        }

        match model.prefill_chunk(
            &p.prompt_tokens,
            &mut p.seq,
            p.chunk_offset,
            chunk_len,
            is_last,
            prefill_stream,
        ) {
            Ok(logits) => {
                p.chunk_offset += chunk_len;
                tracing::info!(
                    "Prefill chunk {}/{} tokens",
                    p.chunk_offset,
                    p.prompt_tokens.len(),
                );
                // Normalize SSM states after EVERY chunk to prevent state drift.
                if let Err(e) = model.normalize_ssm_states(&p.seq, prefill_stream) {
                    tracing::warn!("SSM state normalization failed: {e:#}");
                }
                if is_last {
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
                            tracing::info!("Prefill first token: {first}");
                            completed_indices.push((idx, Some(first)));
                        }
                        Err(e) => {
                            tracing::error!("Chunked prefill argmax: {e:#}");
                            completed_indices.push((idx, None));
                        }
                    }
                    break;
                }
                // Always yield after 1 chunk so the outer scheduler loop
                // can drain new pending requests (Q12).
                //
                // Previously this only broke when `active.is_empty() == false`,
                // leaving back-to-back chunked prefill to monopolise the
                // scheduler when a single stream was prefilling alone. That
                // produced the asymmetric-TTFT bug observed in q12-repro:
                // a second client request that arrived 10 ms after the first
                // had to wait 2:10 for the first stream's full prefill before
                // the scheduler iterated again and picked it up.
                //
                // Cost: tiny — the outer loop re-enters this function on the
                // very next iteration and processes the same stream's next
                // chunk via `prefilling.first_mut()`. The single-stream
                // TTFT regression for short prompts (≤2 chunks) is one
                // extra scheduler-loop pass (~µs).
                break;
            }
            Err(e) => {
                tracing::error!("Prefill chunk error: {e:#}");
                completed_indices.push((idx, None));
                break;
            }
        }
    }
}

/// Batched-prefill step (Q12): advance every prefilling stream by one chunk
/// in a single `model.prefill_batch_chunk` call. Records first-token sample
/// in `completed_indices` for any stream that just finished its last chunk.
///
/// Phase 4a (default-impl wiring): the model's default `prefill_batch_chunk`
/// loops over single-stream `prefill_chunk`. No kernel batching yet — the
/// behavioural win is fairness (every stream advances per iteration vs the
/// FIFO `prefilling.first_mut()` starvation). Phase 2/3 replace the default
/// impl with batched kernel dispatch for true L2-amortised throughput.
fn run_batched_prefill_step(
    model: &dyn Model,
    prefilling: &mut [PrefillInProgress],
    completed_indices: &mut Vec<(usize, Option<u32>)>,
    max_prefill_tokens: usize,
    prefill_stream: u64,
    prefill_event: u64,
) {
    // Build per-stream chunk_len (capped at max_prefill_tokens) and
    // is_last_chunk flag, then construct PrefillSlice borrowing each
    // stream's prompt_tokens and seq.
    //
    // Capture per-stream chunk_len up-front so we can advance
    // `chunk_offset` after the model call (the slices borrow `&mut p.seq`
    // but not `&mut p.chunk_offset`, so post-call mutation is permitted
    // once the slices vec is dropped).
    let n = prefilling.len();
    let mut chunk_lens: Vec<usize> = Vec::with_capacity(n);
    let mut is_last_flags: Vec<bool> = Vec::with_capacity(n);
    for p in prefilling.iter() {
        let remaining = p.prompt_tokens.len() - p.chunk_offset;
        // Same MLA correctness gate as `run_standard_chunk_loop` — MLA
        // models lack a paged-MLA prefill kernel so multi-chunk prefill
        // silently corrupts attention. Force single-chunk for MLA.
        let effective_max = if model.is_mla() { remaining } else { max_prefill_tokens };
        let mut chunk_len = remaining.min(effective_max);
        let is_last = p.chunk_offset + chunk_len >= p.prompt_tokens.len();
        // Align intermediate chunks to GDN WY4 boundary (4 tokens).
        if !is_last && chunk_len >= 4 {
            chunk_len = (chunk_len / 4) * 4;
        }
        chunk_lens.push(chunk_len);
        is_last_flags.push(is_last);
    }

    // Build PrefillSlice borrows. Each slice borrows `&p.prompt_tokens`
    // (immutable) and `&mut p.seq` from a distinct `&mut PrefillInProgress`,
    // which is sound because the fields are disjoint.
    let mut slices: Vec<PrefillSlice<'_>> = prefilling
        .iter_mut()
        .enumerate()
        .map(|(i, p)| PrefillSlice {
            prompt_tokens: &p.prompt_tokens,
            seq: &mut p.seq,
            chunk_start: p.chunk_offset,
            chunk_len: chunk_lens[i],
            is_last_chunk: is_last_flags[i],
        })
        .collect();

    let t0_batch = Instant::now();
    let logits_per_stream = match model.prefill_batch_chunk(&mut slices, prefill_stream) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("Batched prefill error (streams={n}): {e:#}");
            // Mark every stream as failed so they get freed in
            // `promote_completed_prefills`.
            for i in 0..n {
                completed_indices.push((i, None));
            }
            return;
        }
    };
    drop(slices); // release the &mut p.seq borrows so we can advance chunk_offset

    // Sync prefill stream → default stream so subsequent decode sees
    // the prefill writes. Mirrors the existing single-stream path.
    let _ = model.record_event(prefill_event, prefill_stream);
    let _ = model.stream_wait_event(model.default_stream(), prefill_event);

    debug_assert_eq!(logits_per_stream.len(), n, "prefill_batch_chunk returned wrong logit count");

    // Advance offsets and sample first token where the chunk just completed.
    for (i, p) in prefilling.iter_mut().enumerate() {
        p.chunk_offset += chunk_lens[i];
        if !is_last_flags[i] {
            continue;
        }
        let logits = logits_per_stream[i];
        if logits == DevicePtr::NULL {
            tracing::error!(
                "Batched prefill: stream {i} marked is_last but model returned NULL logits",
            );
            completed_indices.push((i, None));
            continue;
        }
        match sample_token(model, logits, p.temperature, p.top_k, p.top_p, &p.eos_tokens) {
            Ok(first) => {
                tracing::info!(
                    "Batched prefill[{i}/{n}] first token: {first} (chunk_len={}, total_tokens={})",
                    chunk_lens[i],
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

    let elapsed = t0_batch.elapsed().as_micros();
    if elapsed > 1000 {
        tracing::debug!("Batched prefill step: {n} streams, {elapsed}µs total");
    }
}

/// Batched mixed step (Q12, Phase 5): N decode tokens + M prefill chunks
/// fused via `model.mixed_forward_batch`. Like `run_batched_prefill_step`
/// but for the active-nonempty case — replaces the existing single-prefill
/// `mixed_forward` call with the M-stream variant.
///
/// On success sets `did_mixed_step = true` so the caller skips the standalone
/// decode dispatch (active sequences' next tokens are already sampled here).
///
/// Phase 5 uses the default trait impl which serializes (`decode_batch` then
/// per-stream `prefill_chunk` loop). Phase 2/3 will override with true
/// kernel-level batched mixed forward.
#[allow(clippy::too_many_arguments)]
fn run_batched_mixed_step(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    prefilling: &mut [PrefillInProgress],
    completed_indices: &mut Vec<(usize, Option<u32>)>,
    max_prefill_tokens: usize,
    prefill_stream: u64,
    prefill_event: u64,
    t0_step: Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    adaptive_sampling: bool,
    did_mixed_step: &mut bool,
) {
    let n_prefill = prefilling.len();
    let n_decode = active.len();

    // Capture per-stream chunk_len + is_last (same MLA gate + WY4 alignment
    // as `run_batched_prefill_step`).
    let mut chunk_lens: Vec<usize> = Vec::with_capacity(n_prefill);
    let mut is_last_flags: Vec<bool> = Vec::with_capacity(n_prefill);
    for p in prefilling.iter() {
        let remaining = p.prompt_tokens.len() - p.chunk_offset;
        let effective_max = if model.is_mla() { remaining } else { max_prefill_tokens };
        let mut chunk_len = remaining.min(effective_max);
        let is_last = p.chunk_offset + chunk_len >= p.prompt_tokens.len();
        if !is_last && chunk_len >= 4 {
            chunk_len = (chunk_len / 4) * 4;
        }
        chunk_lens.push(chunk_len);
        is_last_flags.push(is_last);
    }

    // Gather decode-side inputs.
    let decode_tokens: Vec<u32> = active.iter().map(|a| a.last_token).collect();

    // Build slices in a temporary scope so the &mut borrows on prefilling
    // and active drop before we re-borrow active mutably for
    // `process_decode_logits`.
    let result = {
        let mut decode_refs: Vec<&mut SequenceState> =
            active.iter_mut().map(|a| &mut a.seq).collect();
        let mut prefill_slices: Vec<PrefillSlice<'_>> = prefilling
            .iter_mut()
            .enumerate()
            .map(|(i, p)| PrefillSlice {
                prompt_tokens: &p.prompt_tokens,
                seq: &mut p.seq,
                chunk_start: p.chunk_offset,
                chunk_len: chunk_lens[i],
                is_last_chunk: is_last_flags[i],
            })
            .collect();
        model.mixed_forward_batch(
            &decode_tokens,
            &mut decode_refs,
            &mut prefill_slices,
            prefill_stream,
        )
    };

    let result = match result {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(
                "Mixed-batch forward error (n_decode={n_decode}, n_prefill={n_prefill}): {e:#}",
            );
            for i in 0..n_prefill {
                completed_indices.push((i, None));
            }
            return;
        }
    };

    let _ = model.record_event(prefill_event, prefill_stream);
    let _ = model.stream_wait_event(model.default_stream(), prefill_event);

    // Advance prefill offsets and sample first tokens for streams that just
    // finished their last chunk.
    debug_assert_eq!(result.prefill_logits.len(), n_prefill);
    for (i, p) in prefilling.iter_mut().enumerate() {
        p.chunk_offset += chunk_lens[i];
        if !is_last_flags[i] {
            continue;
        }
        let logits = result.prefill_logits[i];
        if logits == DevicePtr::NULL {
            tracing::error!("Mixed-batch: stream {i} marked is_last but model returned NULL logits");
            completed_indices.push((i, None));
            continue;
        }
        match sample_token(model, logits, p.temperature, p.top_k, p.top_p, &p.eos_tokens) {
            Ok(first) => {
                tracing::info!(
                    "Mixed-batch prefill[{i}/{n_prefill}] first token: {first} (chunk_len={}, total_tokens={})",
                    chunk_lens[i],
                    p.prompt_tokens.len(),
                );
                completed_indices.push((i, Some(first)));
            }
            Err(e) => {
                tracing::error!("Mixed-batch prefill[{i}] sampling: {e:#}");
                completed_indices.push((i, None));
            }
        }
    }

    // Process decode logits for the active lanes — mirrors what
    // `run_standard_chunk_loop`'s mixed_forward branch does.
    if n_decode > 0 && result.decode_logits != DevicePtr::NULL {
        process_decode_logits(
            model,
            active,
            result.decode_logits,
            t0_step,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
            reflection_suppress_ids,
            adaptive_sampling,
        );
    }
    *did_mixed_step = true;
}
