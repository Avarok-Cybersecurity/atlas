// SPDX-License-Identifier: AGPL-3.0-only

//! Q12 batched-prefill step: advance every prefilling stream by one chunk
//! in a single `model.prefill_batch_chunk` call. Records first-token sample
//! in `completed_indices` for any stream that just finished its last chunk.
//!
//! Phase 4a (default-impl wiring): the model's default `prefill_batch_chunk`
//! loops over single-stream `prefill_chunk`. No kernel batching yet — the
//! behavioural win is fairness (every stream advances per iteration vs the
//! FIFO `prefilling.first_mut()` starvation). Phase 2/3 replace the default
//! impl with batched kernel dispatch for true L2-amortised throughput.

use spark_model::traits::{Model, PrefillSlice};
use spark_runtime::gpu::DevicePtr;
use std::time::Instant;

use super::super::sample_first_token;
use super::super::types::PrefillInProgress;

pub(super) fn run_batched_prefill_step(
    model: &dyn Model,
    prefilling: &mut [PrefillInProgress],
    completed_indices: &mut Vec<(usize, Option<u32>)>,
    max_prefill_tokens: usize,
    prefill_stream: u64,
    prefill_event: u64,
) {
    // Per-chunk InnerQ finalize poll — see `phase_continue_prefills::poll_innerq`.
    super::poll_innerq();
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

    debug_assert_eq!(
        logits_per_stream.len(),
        n,
        "prefill_batch_chunk returned wrong logit count"
    );

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
