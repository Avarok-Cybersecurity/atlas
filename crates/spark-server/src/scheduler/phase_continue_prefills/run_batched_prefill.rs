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

use spark_model::traits::{BatchedPrefillDeclined, Model, PrefillSlice};
use std::time::Instant;

use super::super::types::PrefillInProgress;
use super::prefill_fallback::{advance_and_sample, run_wave_per_stream};
use super::prefill_waves::{WaveGeom, plan_prefill_waves};

pub(super) fn run_batched_prefill_step(
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    prefilling: &mut [PrefillInProgress],
    completed_indices: &mut Vec<(usize, Option<u32>)>,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    prefill_stream: u64,
    prefill_event: u64,
    think_end_token: Option<u32>,
    tool_call_start_token: Option<u32>,
) {
    // Per-chunk InnerQ finalize poll — see `phase_continue_prefills::poll_innerq`.
    super::poll_innerq(model);
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
    // VARLEN batched prefill: resolved once here — it governs the wave
    // planner below AND subsumes the codispatch shared-geometry hack (varlen
    // admits ragged chunk-0 batches directly, so equal-length coercion is
    // redundant; per-stream geometry is what the cu_seqlens path wants).
    // Precedence: when both `--prefill-varlen-batch` and the codispatch env
    // are set, varlen wins.
    let varlen = spark_model::layers::ops::prefill_varlen_enabled();
    // Co-dispatch (ATLAS_PREFILL_CODISPATCH=1): when all streams are at chunk 0
    // and equal-length, give them ONE shared geometry so the kernel-batched path
    // is eligible (check_kernel_batched_eligible requires identical chunk_len /
    // chunk_start / is_last across streams). Ragged or non-chunk-0 batches keep
    // per-stream geometry, which the dispatcher handles via per-stream fallback.
    let shared_geom: Option<(usize, bool)> = if !varlen
        && std::env::var("ATLAS_PREFILL_CODISPATCH")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        && !model.is_mla()
        && n >= 2
        && prefilling.iter().all(|p| p.chunk_offset == 0)
        && prefilling
            .iter()
            .all(|p| p.prompt_tokens.len() == prefilling[0].prompt_tokens.len())
    {
        let total = prefilling[0].prompt_tokens.len();
        let mut cl = total.min(max_prefill_tokens);
        let is_last = cl >= total;
        if !is_last && cl >= 4 {
            cl = (cl / 4) * 4;
        }
        Some((cl, is_last))
    } else {
        None
    };
    for p in prefilling.iter() {
        let (chunk_len, is_last) = if let Some((cl, il)) = shared_geom {
            (cl, il)
        } else {
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
            let mut is_last = p.chunk_offset + chunk_len >= p.prompt_tokens.len();
            // TAIL PRE-SPLIT (VARLEN only). `prefill_chunk_dispatch` splits a
            // model's FINAL prefill chunk once, one KV block below the last
            // block boundary under the prompt, to land an SSM tail checkpoint
            // a later turn's block-floored prefix match can actually use. The
            // batched path does not split, so handing it the whole prompt as
            // one `is_last` chunk would give a co-admitted stream a DIFFERENT
            // forward shape than the per-stream path gives the same prompt —
            // and BF16 accumulation is not associative, so at temperature 0
            // that is a different answer for the same request (the reason the
            // model-side split is unconditional in the first place).
            //
            // Asking the model where it would cut and cutting there keeps the
            // geometry identical per sequence, AND it is what lets the tails
            // batch: every member of a co-arriving burst of equal-length
            // prompts gets the same `chunk_start` for its tail, so the wave
            // planner groups all N tails into ONE forward. #927 measured the
            // standalone alternative at 29.3 ms for 25 tokens — 1 170 µs/token,
            // 11.7% of prefill GPU time for 2.1% of the tokens.
            //
            // The condition mirrors `prefill_chunk_dispatch`'s own
            // (`cut > chunk_start && cut < total`, on a last chunk) exactly,
            // plus the requirement that the cut lie inside THIS chunk — a
            // prompt longer than the budget reaches its final chunk at a
            // nonzero offset and splits there just the same.
            if varlen
                && is_last
                && let Some(cut) = model.prefill_tail_cut(&p.prompt_tokens)
                && cut > p.chunk_offset
                && cut < p.prompt_tokens.len()
                && cut <= p.chunk_offset + chunk_len
            {
                chunk_len = cut - p.chunk_offset;
                is_last = false;
            }
            // Align intermediate chunks to GDN WY4 boundary (4 tokens). The
            // tail cut is a multiple of the KV block size, which is itself a
            // multiple of 4 on every shipped config, so this is a no-op there.
            if !is_last && chunk_len >= 4 {
                chunk_len = (chunk_len / 4) * 4;
            }
            (chunk_len, is_last)
        };
        chunk_lens.push(chunk_len);
        is_last_flags.push(is_last);
    }

    // Wave planning. VARLEN batched prefill (`--prefill-varlen-batch`) caps
    // the concatenated M of one forward at the prefill token budget (clamped
    // to the hidden-buffer arena) and groups streams by the model-side
    // admission geometry (shared chunk_start / is_last). Waves run
    // back-to-back within this tick, so every stream still advances one
    // chunk per tick. Flag OFF ⇒ exactly one wave holding every stream — the
    // pre-wave dispatch, byte-identical (pinned in prefill_waves tests).
    let wave_cap = max_prefill_tokens.min(max_batch_tokens).max(1);
    let geoms: Vec<WaveGeom> = prefilling
        .iter()
        .enumerate()
        .map(|(i, p)| WaveGeom {
            chunk_start: p.chunk_offset,
            chunk_len: chunk_lens[i],
            is_last: is_last_flags[i],
        })
        .collect();
    let waves = plan_prefill_waves(&geoms, varlen, wave_cap);
    let n_waves = waves.len();
    if varlen {
        // Engagement proof for serve-log diagnosis: one INFO line per tick
        // with the planned wave shapes. M per wave = Σ chunk_len of its
        // members — the row count every fused per-layer GEMM launches at
        // (assuming the model-side dispatch admits; it logs its own verdict
        // under target "atlas::q12").
        let wave_m: Vec<usize> = waves
            .iter()
            .map(|w| w.iter().map(|&i| chunk_lens[i]).sum())
            .collect();
        tracing::info!(
            "Varlen prefill waves: {n} streams -> {n_waves} wave(s), M per wave {wave_m:?} \
             (cap {wave_cap})"
        );
    }

    let t0_batch = Instant::now();
    for wave in waves {
        // Build PrefillSlice borrows for this wave's members. Each slice
        // borrows `&p.prompt_tokens` (immutable) and `&mut p.seq` from a
        // distinct `&mut PrefillInProgress`, which is sound because the
        // fields are disjoint; the filter keeps the borrows within the wave.
        let mut in_wave = vec![false; n];
        for &i in &wave {
            in_wave[i] = true;
        }
        let mut slices: Vec<PrefillSlice<'_>> = prefilling
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| in_wave[*i])
            .map(|(i, p)| PrefillSlice {
                prompt_tokens: &p.prompt_tokens,
                seq: &mut p.seq,
                chunk_start: p.chunk_offset,
                chunk_len: chunk_lens[i],
                is_last_chunk: is_last_flags[i],
            })
            .collect();

        let logits_per_stream = match model.prefill_batch_chunk(&mut slices, prefill_stream) {
            Ok(v) => v,
            Err(e) => {
                let declined = BatchedPrefillDeclined::is_decline(&e);
                tracing::error!(
                    "Batched prefill error (wave of {} streams, {n} prefilling, \
                     declined={declined}): {e:#}",
                    wave.len()
                );
                drop(slices);
                if declined {
                    // DECLINE: the model refused before touching a single
                    // stream, so every member of this wave still has to be
                    // prefilled — one at a time, which is what the model would
                    // have done anyway. Dropping them here is what produced
                    // #927 cell E's sixteen empty HTTP 200s.
                    run_wave_per_stream(
                        model,
                        sched,
                        prefilling,
                        completed_indices,
                        &wave,
                        &chunk_lens,
                        &is_last_flags,
                        n,
                        prefill_stream,
                        prefill_event,
                        think_end_token,
                        tool_call_start_token,
                    );
                    continue;
                }
                // HARD ERROR: an admitted batch can already own KV blocks and
                // prefix reservations, so re-running it per-stream would
                // double-allocate. Fail ONLY this wave's streams —
                // `promote_completed_prefills` now sends each of them an error
                // frame, so a failed wave is visible to its clients instead of
                // closing sixteen streams with no content and no
                // `finish_reason`. Later waves are left untouched: they have
                // not advanced this tick and retry next tick rather than
                // dispatching after a failed forward.
                for &i in &wave {
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
            wave.len(),
            "prefill_batch_chunk returned wrong logit count"
        );

        // Advance offsets and sample first token where the chunk just
        // completed — BEFORE the next wave dispatches, because every wave
        // reuses the same logits rows.
        for (k, &i) in wave.iter().enumerate() {
            advance_and_sample(
                model,
                sched,
                &mut prefilling[i],
                i,
                n,
                chunk_lens[i],
                is_last_flags[i],
                logits_per_stream[k],
                completed_indices,
                think_end_token,
                tool_call_start_token,
            );
        }
    }

    let elapsed = t0_batch.elapsed().as_micros();
    if elapsed > 1000 {
        tracing::debug!("Batched prefill step: {n} streams, {n_waves} waves, {elapsed}µs total");
    }
}
