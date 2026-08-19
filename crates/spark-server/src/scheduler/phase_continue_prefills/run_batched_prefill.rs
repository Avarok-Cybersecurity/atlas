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
            let is_last = p.chunk_offset + chunk_len >= p.prompt_tokens.len();
            // Align intermediate chunks to GDN WY4 boundary (4 tokens).
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
                tracing::error!(
                    "Batched prefill error (wave of {} streams, {n} prefilling): {e:#}",
                    wave.len()
                );
                // Fail ONLY this wave's streams (freed in
                // `promote_completed_prefills`). Later waves are left
                // untouched — they have not advanced this tick and retry
                // next tick rather than dispatching after a failed forward.
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
            let p = &mut prefilling[i];
            p.chunk_offset += chunk_lens[i];
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
            let logits = logits_per_stream[k];
            if logits == DevicePtr::NULL {
                tracing::error!(
                    "Batched prefill: stream {i} marked is_last but model returned NULL logits",
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
    }

    let elapsed = t0_batch.elapsed().as_micros();
    if elapsed > 1000 {
        tracing::debug!("Batched prefill step: {n} streams, {n_waves} waves, {elapsed}µs total");
    }
}
