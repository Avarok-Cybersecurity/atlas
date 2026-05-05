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
use spark_model::traits::{Model, SequenceState};
use std::time::Instant;

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
                // If active sequences exist, yield after 1 chunk.
                if !active.is_empty() {
                    break;
                }
                // Otherwise, continue processing next chunk immediately.
            }
            Err(e) => {
                tracing::error!("Prefill chunk error: {e:#}");
                completed_indices.push((idx, None));
                break;
            }
        }
    }
}
