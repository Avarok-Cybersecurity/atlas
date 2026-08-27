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
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    did_mixed_step: &mut bool,
) {
    // Per-chunk InnerQ finalize poll — see `phase_continue_prefills::poll_innerq`.
    super::poll_innerq(model);
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
        // Land a chunk boundary on `ssm_tail_boundary` so the SSM snapshot saved
        // there is exactly what the NEXT turn's block-floored `matched_tokens`
        // looks up — otherwise the warm restore falls back to the coarse
        // --ssm-checkpoint-interval grid and replays ~254 SSM tokens per turn.
        if spark_runtime::ssm_tail_ckpt_enabled()
            && !spark_runtime::ssm_tail_midchunk_enabled()
            && let Some(bs) = model.kv_block_size()
            && let Some(tb) = spark_runtime::ssm_tail_boundary(p.prompt_tokens.len(), bs)
            && p.chunk_offset < tb
            && p.chunk_offset + chunk_len > tb
        {
            chunk_len = tb - p.chunk_offset;
        }
        let is_last = p.chunk_offset + chunk_len >= p.prompt_tokens.len();
        if !is_last && chunk_len >= 4 {
            chunk_len = (chunk_len / 4) * 4;
        }
        chunk_lens.push(chunk_len);
        is_last_flags.push(is_last);
    }

    // Gather decode-side inputs.
    // Sort by SSM pool slot before building the batch, exactly as
    // `decode_step.rs` does for pure-decode steps. Without it the batched
    // GDN recurrence declines: its precondition is that the n sequences sit
    // in CONSECUTIVE pool slots IN SLICE ORDER, and retire+compaction packs
    // the slot SET while scrambling the ORDER (a finishing sequence's slot
    // is backfilled by the highest-slot survivor, which stays last in the
    // vec). Pure-decode steps healed this and mixed steps did not, so every
    // step for the duration of a prefill fell back to the per-seq loop —
    // measured 853 us/layer vs 618 batched, a 28% penalty on ~30 ms/step.
    // Reordering whole ActiveSeq elements keeps the post-decode
    // position->seq mapping consistent (same argument as decode_step.rs).
    if active.len() > 1 {
        active.sort_by_key(|a| a.seq.ssm_slot_idx().unwrap_or(a.seq.slot_idx));
    }
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
        sched.snapshot.note_prefill_chunk(chunk_lens[i], true);
        // Prompt tokens counted AT INGEST, as each chunk lands. Counting them
        // at request completion (the old site) credited the whole prompt to the
        // moment the RESPONSE finished — seconds or minutes after the prefill
        // actually ran — so the Stats page's prefill rate was attributed to the
        // wrong instant entirely. Semantics are unchanged: this is the same
        // "prompt_tokens" the API reports (cache-hit tokens included; the
        // computed-vs-cached split is the prefix-cache panel's job).
        crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(chunk_lens[i] as u64);
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
            &sched.levers.sampling(),
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
            sched,
        );
    }
    *did_mixed_step = true;
}
