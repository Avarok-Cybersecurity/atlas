// SPDX-License-Identifier: AGPL-3.0-only

//! Single-stream chunked-prefill body.
//!
//! Tries `mixed_forward` first when an active decode lane is available
//! (fuses 1 prefill chunk + N decode in one pass). Otherwise falls back
//! to plain `prefill_chunk` + EP broadcast.

use anyhow::Result;
use spark_model::traits::{Model, SequenceState};
use std::time::Instant;

use super::super::decode_logits_step::process_decode_logits;
use super::super::sample_first_token;
use super::super::types::{ActiveSeq, PrefillInProgress};

#[allow(clippy::too_many_arguments)]
pub(super) fn run_standard_chunk_loop(
    model: &dyn Model,
    p: &mut PrefillInProgress,
    idx: usize,
    active: &mut Vec<ActiveSeq>,
    max_prefill_tokens: usize,
    slice_budget: usize,
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
    completed_indices: &mut Vec<(usize, Option<u32>)>,
    did_mixed_step: &mut bool,
) {
    // TQ+ InnerQ: poll once per chunk to see if calibration has banked
    // enough K² stats to finalize scales. The driver itself is idempotent
    // (no-op after activation), so calling on every chunk is cheap.
    // Group size = 128 = Qwen3 head_dim (the only currently-supported shape).
    super::poll_innerq();
    // Single chunk per call — the outer scheduler loop re-enters this
    // function on the very next iteration to advance the next stream
    // or the next chunk. This yield keeps fairness across pending
    // requests (Q12: avoids back-to-back chunked prefill monopolising
    // the scheduler).
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
    // Step 2 (spec): cap the chunk to the policy's prefill slice budget so a
    // fused mixed step stays under the TBT target. With ATLAS_HOLO_ALWAYS_MIXED
    // OFF the caller passes slice_budget == max_prefill_tokens, so this `.min`
    // is a no-op and the chunk cap is unchanged (byte-identical resting path).
    // MLA keeps its forced full-remaining chunk (correctness gate above) — the
    // slice budget never applies there.
    let cap = if model.is_mla() {
        effective_max
    } else {
        effective_max.min(slice_budget)
    };
    let mut chunk_len = remaining.min(cap);
    let is_last = p.chunk_offset + chunk_len >= p.prompt_tokens.len();
    // Align intermediate chunks to GDN WY4 boundary (4 tokens).
    if !is_last && chunk_len >= 4 {
        chunk_len = (chunk_len / 4) * 4;
    }

    // ── Mixed forward: fuse prefill chunk + decode in one pass ──
    // ATLAS_BISECT_NO_MIX=1 forces this branch to false so we can
    // diagnose whether the chunked-prefill+concurrent CUDA-700 lives
    // inside `mixed_forward` (active+prefill fused) vs the pure
    // decode-batch path.
    let no_mix_bisect = std::env::var("ATLAS_BISECT_NO_MIX")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    let can_mix = !no_mix_bisect
        && !active.is_empty()
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
                    // NOTE: the SSM-state normalize for the mixed path is done
                    // INSIDE mixed_forward on default_stream (the stream the GDN
                    // recurrence wrote h_state on) — for EVERY chunk including
                    // this last one. Normalizing here on prefill_stream raced
                    // those writes (the B1 failure, 0/12 token-for-token), so it
                    // is gone. Do NOT re-add it.
                    let _ = model.record_event(prefill_event, prefill_stream);
                    let _ = model.stream_wait_event(model.default_stream(), prefill_event);
                    // #131: grammar-constrain the FIRST token (and advance the
                    // matcher); no-op without a grammar.
                    // P1-4 (2026-07-09): thread the resolved `min_p` —
                    // previously a hardcoded 0.0 inside the sampler.
                    // Kill-switch: ATLAS_NO_MTP_MINP=1.
                    match sample_first_token(
                        model,
                        result.prefill_logits,
                        p.temperature,
                        p.top_k,
                        p.top_p,
                        p.min_p,
                        &p.eos_tokens,
                        p.grammar_state.as_mut(),
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
                    code_fence_token,
                    tool_call_start_token,
                    tool_call_end_token,
                    adaptive_sampling,
                );
                *did_mixed_step = true;
            }
            Err(e) => {
                tracing::error!("Mixed forward error: {e:#}");
                completed_indices.push((idx, None));
            }
        }
        return;
    }

    // ── Standard path: prefill chunk only, decode separately ──
    // EP: broadcast chunk tokens to worker (bulk, single NCCL op).
    let ep_ok = (|| -> Result<()> {
        model.ep_broadcast_cmd_for_seq(p.seq.slot_idx as u32, 0xFFFFFFF0)?;
        model.ep_broadcast_cmd(chunk_len as u32)?;
        model.ep_broadcast_cmd(p.chunk_offset as u32)?;
        model.ep_broadcast_cmd(p.prompt_tokens.len() as u32)?;
        model.ep_broadcast_tokens(&p.prompt_tokens)?;
        Ok(())
    })();
    if let Err(e) = ep_ok {
        tracing::error!("EP broadcast chunk: {e:#}");
        completed_indices.push((idx, None));
        return;
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
                        tracing::info!("Prefill first token: {first}");
                        completed_indices.push((idx, Some(first)));
                    }
                    Err(e) => {
                        tracing::error!("Chunked prefill argmax: {e:#}");
                        completed_indices.push((idx, None));
                    }
                }
            }
        }
        Err(e) => {
            tracing::error!("Prefill chunk error: {e:#}");
            completed_indices.push((idx, None));
        }
    }
}
