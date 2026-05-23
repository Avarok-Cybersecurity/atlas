// SPDX-License-Identifier: AGPL-3.0-only

//! Q12 Phase 5 batched mixed step: N decode tokens + M prefill chunks
//! fused via `model.mixed_forward_batch`. Like `run_batched_prefill_step`
//! but for the active-nonempty case — replaces the existing single-prefill
//! `mixed_forward` call with the M-stream variant.
//!
//! On success sets `did_mixed_step = true` so the caller skips the standalone
//! decode dispatch (active sequences' next tokens are already sampled here).
//!
//! Phase 5 uses the default trait impl which serializes (`decode_batch` then
//! per-stream `prefill_chunk` loop). Phase 2/3 will override with true
//! kernel-level batched mixed forward.

use spark_model::traits::{Model, PrefillSlice, SequenceState};
use spark_runtime::gpu::DevicePtr;
use std::time::Instant;

use super::super::decode_logits_step::process_decode_logits;
use super::super::sample_first_token;
use super::super::types::{ActiveSeq, PrefillInProgress};

#[allow(clippy::too_many_arguments)]
pub(super) fn run_batched_mixed_step(
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
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
    did_mixed_step: &mut bool,
) {
    // Per-chunk InnerQ finalize poll — see `phase_continue_prefills::poll_innerq`.
    super::poll_innerq();
    let n_prefill = prefilling.len();
    let n_decode = active.len();

    // Capture per-stream chunk_len + is_last (same MLA gate + WY4 alignment
    // as `run_batched_prefill_step`).
    let mut chunk_lens: Vec<usize> = Vec::with_capacity(n_prefill);
    let mut is_last_flags: Vec<bool> = Vec::with_capacity(n_prefill);
    for p in prefilling.iter() {
        let remaining = p.prompt_tokens.len() - p.chunk_offset;
        let effective_max = if model.is_mla() {
            remaining
        } else {
            max_prefill_tokens
        };
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
            tracing::error!(
                "Mixed-batch: stream {i} marked is_last but model returned NULL logits"
            );
            completed_indices.push((i, None));
            continue;
        }
        // #131: grammar-constrain the FIRST token (and advance the matcher);
        // no-op without a grammar.
        match sample_first_token(
            model,
            logits,
            p.temperature,
            p.top_k,
            p.top_p,
            &p.eos_tokens,
            p.grammar_state.as_mut(),
        ) {
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
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            adaptive_sampling,
        );
    }
    *did_mixed_step = true;
}
