// SPDX-License-Identifier: AGPL-3.0-only

//! Verify-time pre-sample LogitsProcessor pipeline (Phase C-2 wiring).
//!
//! The MTP / speculative-decode verify paths used to consume the raw
//! GPU `argmax_bf16` ID at every verify position, completely bypassing
//! the 8-stage [`crate::scheduler::logit_processors`] pipeline that the
//! non-MTP path runs on every sampled token. Result: tokens emitted
//! through verify (the dominant decode path when MTP is enabled —
//! every accepted/bonus token came from `decode_verify_graphed`) never
//! saw mid-word `</think>` defer, post-close think mask, tool-during-
//! think mask, forced think-end injection, pin-to-tool-call, forced-
//! token fast-path, or grammar bitmask. This is the root cause of
//! grammar desync, malformed tool calls, mid-word `</think>` cuts and
//! stray `<think>` re-entry observed on Qwen3.6-FP8 (opencode-session
//! transcripts, 2026-05-24).
//!
//! This module replays the same dequant + pipeline on a host-side copy
//! of the verify logits buffer (`[K, vocab]` BF16, written by
//! `decode_verify_graphed_*` into `model.logits_buffer_ptr()`), then
//! picks the resulting argmax. Cost: ~0.8 ms per verify position for a
//! ~256k vocab on host, mirroring the non-MTP `process_seq_logits` path
//! in `decode_logits_seq.rs`. The CUDA-graphed `argmax_bf16` saving of
//! ~0.5 ms/step is preserved for the **draft** path (drafts already go
//! through a separate grammar-bitmask path in MTP propose); only the
//! **verify-time** argmax is replaced.
//!
//! Per-position semantics: the pipeline is applied independently to
//! each verify position 0..K. For position 0 the `ActiveSeq` state is
//! exactly the post-`last_token` state, identical to the non-MTP
//! decode site. For positions ≥ 1, the driver SPECULATIVELY ADVANCES
//! the xgrammar matcher via `gs.accept_token(pick_{i-1})` between
//! positions, so each position's bitmask reflects the matcher state
//! that will actually exist at `emit_token` time on the accept path.
//! Speculative advances are rolled back via `gs.rollback(n)` once all
//! K positions have been picked; the real `emit_token` calls then
//! re-advance the matcher normally for the verified tokens that
//! actually get emitted.
//!
//! **DO NOT remove the speculative advance.** Prior versions emitted
//! position-1 argmax against position-0 bitmask, which desynced
//! xgrammar on the accept path and tripped the non-silent
//! `accept_token` kill switch (observed live on
//! opencode-realfix.jsonl 2026-05-24: every response ended with
//! `length` + `tok=198 output_len=30-60` because the bonus token was
//! masked at position 0's state — a `\n` legal at JSON-value-start
//! is not legal at JSON-comma-or-closebrace).
//!
//! Penalty history is NOT stale: positions ≥ 1 thread the block's
//! earlier picks into `apply_penalties_and_bias` (2026-08-18), so the
//! penalty-aware argmax is byte-identical to sequential decode. Other
//! state-dependent masks (mid-word lookback, last_token reads) still see
//! slightly stale `output_tokens` for positions ≥ 1 — best-effort,
//! mirrors greedy unroll.

mod argmax;
mod fast_masked;
mod scratch;

use crate::scheduler::ActiveSeq;
use crate::scheduler::helpers::bf16_to_f32;
use crate::scheduler::logit_processors::LogitsContext;
use spark_model::traits::Model;

// `ATLAS_DISABLE_FAST_GREEDY` is now `SchedLevers::fast_greedy_grammar`,
// read off `LogitsContext::sampling` at the one site that gated on it.

// The DFlash verify statics are now `SchedLevers::dflash_*`.
// The `ATLAS_NO_MTP_VERIFY_SAMPLE` kill switch is now
// `SchedLevers::mtp_verify_sample`, carried on `LogitsContext`.

/// Per-position verify logits, dequantised + processed through the full
/// pre-sample pipeline. Returns the chosen token: either the forced
/// token from a [`crate::scheduler::logit_processors::forced_token::ForcedTokenFastPath`]
/// short-circuit, or the post-pipeline argmax.
///
/// `logits_bytes`: byte slice for ONE verify position; length
/// `vocab_size * 2` (BF16) or `vocab_size * 4` (FP32).
/// `is_fp32`: true when the model emits FP32 logits (Gemma-4 dense).
/// `a`: the active sequence; the pipeline mutates seq state in place
/// (F2 confidence arm, sentence_defer_count, etc.).
/// `ctx`: tokenizer special-token IDs used by the pipeline.
/// `verify_pos`: this position's index within the verify span (0..K) —
/// P1-3 (2026-07-09): used only to derive the per-token seed offset for
/// the temp>0 sampling branch, matching the `output_tokens.len()`-based
/// seed the non-MTP path would have used for the same emitted position.
///
/// Mirrors the host-side path of `decode_logits_seq::process_seq_logits`
/// for byte-identical pipeline semantics.
pub fn verify_pick_with_pipeline(
    logits_bytes: &[u8],
    is_fp32: bool,
    vocab_size: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
    verify_pos: usize,
    // Picks from positions 0..verify_pos of the SAME verify block, in order.
    // Threaded into the penalty history so penalty-aware verify argmax is
    // byte-identical to sequential decode (see process_position_logits).
    pending_picks: &[u32],
) -> u32 {
    use crate::scheduler::mtp_timing::Phase;
    // 1. Dequant per the same scheme as `process_seq_logits`, into a REUSED
    //    thread-local buffer rather than a fresh ~1 MB `Vec<f32>` per K position
    //    — see `scratch.rs`. Semantically inert: every `vocab_size` entry is
    //    overwritten before any read.
    let t_dequant = std::time::Instant::now();
    let mut f32_logits = scratch::DEQUANT_SCRATCH.with(|s| std::mem::take(&mut *s.borrow_mut()));
    f32_logits.clear();
    f32_logits.reserve(vocab_size);
    if is_fp32 {
        f32_logits.extend((0..vocab_size).map(|j| {
            let off = j * 4;
            f32::from_le_bytes([
                logits_bytes[off],
                logits_bytes[off + 1],
                logits_bytes[off + 2],
                logits_bytes[off + 3],
            ])
        }));
    } else {
        f32_logits.extend((0..vocab_size).map(|j| {
            let lo = logits_bytes[j * 2];
            let hi = logits_bytes[j * 2 + 1];
            bf16_to_f32(lo, hi)
        }));
    }
    ctx.timing.record(Phase::Dequant, t_dequant);
    // Hand the allocation back on EVERY exit below (forced-token short circuit,
    // temp>0 sample, argmax), or the next call allocates from scratch again and
    // the reuse is silently lost.
    let mut f32_logits = scratch::ScratchGuard(f32_logits);

    // 2. Build this position's penalty/bias params (Verify kind: greedy,
    //    seed-free, no caller bias — the builder still appends the A4 floor
    //    and the rep/presence/freq/LZ/DRY gates from `a`). Cloned before the
    //    `&mut a` borrow in `process_position_logits`.
    //
    //    Without these penalties MTP-VERIFIED tokens were decided by a
    //    penalty-FREE argmax, so the MODEL.toml `repetition_penalty` /
    //    `dry_multiplier` never reached the dominant decode path and the
    //    model degenerated into repeated tool-call argument junk. The
    //    resulting emission is a penalty-aware ARGMAX (greedy) — an intended
    //    behavioral delta for speculative acceptance. Backward-compatible: a
    //    no-op when the penalties are neutral (rep==1.0, dry==0.0, etc.).
    let penalties = crate::scheduler::sample_step::penalty_params_for(
        a,
        crate::scheduler::sample_step::PositionKind::Verify,
        0.0,
        None,
        Vec::new(),
    );

    // 3. Unified per-position post-processing (SSOT shared with the non-MTP
    //    path): force-temp-zero bypass → pipeline (forced-token short
    //    circuit) → penalties+bias. A `Some(tok)` return is the forced /
    //    bypass token — emit directly, no argmax scan. R1: this does NOT
    //    advance the grammar matcher; the K-loop in
    //    `verify_pick_all_with_pipeline` owns `accept_token` / `rollback`.
    let t_proc = std::time::Instant::now();
    if let Some(tok) = crate::scheduler::logit_processors::process_position_logits(
        &mut f32_logits,
        a,
        ctx,
        &penalties,
        crate::scheduler::sample_step::PositionKind::Verify,
        pending_picks,
    ) {
        ctx.timing.record(Phase::PipelineProc, t_proc);
        return tok;
    }
    ctx.timing.record(Phase::PipelineProc, t_proc);

    // 4a. P1-3 (2026-07-09): when the request asked for temperature > 0,
    //     SAMPLE from the processed logits instead of taking the argmax.
    //     The processors (grammar bitmask, think/tool sched, penalties+bias)
    //     already ran in place above, so masked tokens sit at -inf and the
    //     sampler's candidate filter excludes them — the sampled pick is
    //     grammar-mask-allowed by construction. This mirrors the non-MTP
    //     tail of `decode_logits_seq::process_seq_logits` exactly: neutral
    //     penalty params (penalties were applied in step 3, so the sampler's
    //     internal `apply_penalties_and_bias` is a no-op) + the sequence's
    //     temperature / top_k / top_p / top_n_sigma / min_p. min_p is the
    //     resolved request+MODEL.toml-floor value, subject to the P1-4
    //     ATLAS_NO_MTP_MINP kill-switch. The seed advances per emitted
    //     position (`output_tokens.len() + verify_pos`) — the same offset
    //     FinalDecode would use if this position is accepted and emitted.
    //     Unreachable under ATLAS_FORCE_TEMP_ZERO (the bypass in
    //     `process_position_logits` returns Some(argmax) before this point);
    //     the guard is kept as documentation. Kill-switch:
    //     ATLAS_NO_MTP_VERIFY_SAMPLE=1 reverts to the pinned argmax below.
    if ctx.sampling.mtp_verify_sample && a.temperature > 0.0 && !ctx.sampling.force_temp_zero {
        let t_sample = std::time::Instant::now();
        let step_seed = a
            .seed
            .map(|s| s.wrapping_add((a.output_tokens.len() + verify_pos) as u64));
        let sampler_shape = spark_runtime::sampler::SamplingParams {
            temperature: a.temperature,
            top_k: a.top_k,
            top_p: a.top_p,
            top_n_sigma: a.top_n_sigma,
            min_p: crate::scheduler::sample_step::effective_min_p(a.min_p, &ctx.sampling),
            logit_bias: Vec::new(),
            repetition_penalty: 1.0,
            repetition_penalty_window: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            lz_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: penalties.dry_base,
            dry_allowed_length: penalties.dry_allowed_length,
            dry_sequence_breakers: Vec::new(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: step_seed,
        };
        // SAFETY: `f32_logits` is a live Vec<f32> of `vocab_size` elements;
        // reinterpreting as bytes is the same cast the non-MTP sampler tail
        // uses (`decode_logits_seq.rs`).
        let f32_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
        let sampled =
            spark_runtime::sampler::sample_with_params_history(f32_bytes, &sampler_shape, &[]);
        // Recorded under the Argmax phase: it replaces the argmax pick and
        // keeps the mtp_timing phase set unchanged.
        ctx.timing.record(Phase::Argmax, t_sample);
        return sampled;
    }

    // 4. Argmax over the (now-masked-and-penalised) vector. Matches the
    //    sampler's argmax branch behaviour.
    let t_argmax = std::time::Instant::now();
    let best_id = argmax::argmax_first_wins(&f32_logits);
    ctx.timing.record(Phase::Argmax, t_argmax);
    best_id
}

/// Convenience: copy the full `[K, vocab]` verify logits buffer to
/// host and apply [`verify_pick_with_pipeline`] to every position,
/// returning the K processed token IDs. Falls back to the raw argmax
/// IDs if the D2H copy fails (matches `verify_resample` and
/// `extract_verify_logprobs` failure semantics).
///
/// `argmax_ids` is the GPU-graphed argmax already returned by
/// `decode_verify_graphed*`; used as the fallback for the failure
/// path and as the array length source.
///
/// `row_base` (batched-MTP E12): first logits row of THIS sequence's
/// verify span within the shared `[R, vocab]` logits buffer. Single-
/// sequence verify paths pass 0 (rows 0..K — unchanged behaviour);
/// the batched K=4 verify passes `i*4` for sequence i so every read
/// (fast-path single-logit probes and the slow-path D2H) targets the
/// sequence's own rows.
pub fn verify_pick_all_with_pipeline(
    model: &dyn Model,
    argmax_ids: &[u32],
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
    row_base: usize,
    // The step's draft tokens, when the caller has them. Position i's pick is
    // compared against drafts[i]: at the FIRST mismatch the pipeline STOPS —
    // sequential decode never scores positions past the correction token, and
    // running the stateful stages there (F2 confidence, defer counters — the
    // pipeline mutates `a` in place, and only the grammar matcher is rolled
    // back) drifts seq state ahead of the real output. Measured 2026-08-18 on
    // qwen3.8 video-fidelity: FTZ bypass = 13/13 both legs (forward exact),
    // pipeline-on + spec-on = FAIL — the divergence lives in these extra
    // stateful runs. Positions past the stop are padded with the RAW argmax:
    // the accept chain short-circuits before reading them, so they only feed
    // shadow logging. `None` preserves the old process-all behavior.
    drafts: Option<&[u32]>,
) -> Vec<u32> {
    use crate::scheduler::mtp_timing::Phase;
    let k = argmax_ids.len();
    if k == 0 {
        return Vec::new();
    }

    // ── CHAT FAST PATH (2026-07-08): masked-greedy == raw-argmax guard ──
    // See `fast_masked` module docs: for a grammarless request with no
    // forced/stateful stage armed and argmax-preserving penalties, the
    // pipeline provably cannot change any pick, so the raw argmax IS the
    // masked pick and the [K, vocab] D2H is skipped entirely. Any
    // ineligible position falls through to the slow path for the call.
    if let Some(picks) = fast_masked::try_chat_fast_path(model, argmax_ids, a, ctx, row_base) {
        return picks;
    }

    // ── FAST PATH (#3, 2026-06-02): on-GPU greedy pick under grammar ──
    //
    // Culprit #3 (regression hunt): the slow path below D2H-copies the full
    // [K, vocab] logits, CPU-dequants 248k BF16→F32 per position, and runs the
    // 8-stage pipeline + argmax — ~1-3 ms/token of host/PCIe serialization on
    // the dominant MTP verify path, the structural reason vLLM (GPU sampling)
    // out-decodes Atlas on tool/grammar workloads.
    //
    // But when decoding is GREEDY (temp=0 or ATLAS_FORCE_TEMP_ZERO), penalties
    // are neutral, and we're not inside <think>, the masked-greedy pick at each
    // verify position is EXACTLY the GPU argmax (`argmax_ids[i]`, already
    // computed by decode_verify_graphed*) WHENEVER that argmax is grammar-
    // allowed — because the global max that is also in the allowed set is, by
    // definition, the max over the allowed set. So we can emit it directly with
    // NO D2H/dequant/pipeline. This fires for the bulk of content tokens (the
    // permissive value ladder allows almost everything). We fall back to the
    // slow pipeline per-call only when some position's argmax is grammar-
    // DISALLOWED (structural/forced positions — rare) or the regime isn't
    // pure-greedy. The speculative matcher advance + history-delta rollback
    // (BUG#3) are preserved identically to the slow path, so on fallback the
    // matcher is restored to its exact pre-call state.
    //
    // Skipped in this fast path: the WS/AM/think/forced quality nudges. Those
    // are either no-ops in the content/greedy/neutral regime or acceptable
    // speed-for-quality trades (we hold a measured accuracy margin over vLLM).
    // Kill-switch: ATLAS_DISABLE_FAST_GREEDY=1.
    //
    // #237 (fix 4a): the all-penalties-neutral requirement is relaxed to the
    // SSOT `fast_greedy` gate — reduce-only penalties (rep>=1.0, presence/
    // frequency>=0, LZ/DRY off, no bias) provably cannot flip an argmax whose
    // token is NOT in the scoped penalty history and whose raw logit is > 0
    // (see `fast_greedy` module docs for the proof). The membership test uses
    // the SAME scoped history the slow path hands to
    // `apply_penalties_and_bias` (`penalty_history_scope`), which is also
    // deliberately STALE across positions ≥ 1 exactly like the slow path
    // (output_tokens does not grow until `emit_token`, after this helper).
    // P1-3 (2026-07-09): this temp==0 gate is load-bearing for verify-time
    // sampling — at temperature > 0 the fast GPU-argmax shortcut must NOT
    // fire, so every position routes through the slow pipeline below where
    // the temp>0 sampling branch (step 4a in `verify_pick_with_pipeline`)
    // draws from the processed logits. Confirmed and kept as-is.
    let fast_penalty_gate = if ctx.sampling.fast_greedy_grammar
        && a.grammar_state.is_some()
        && !a.inside_thinking
        && (a.temperature == 0.0 || ctx.sampling.force_temp_zero)
    {
        crate::scheduler::fast_greedy::classify_penalties(
            &crate::scheduler::sample_step::penalty_params_for(
                a,
                crate::scheduler::sample_step::PositionKind::Verify,
                0.0,
                None,
                Vec::new(),
            ),
        )
    } else {
        crate::scheduler::fast_greedy::PenaltyGate::Blocked
    };
    if fast_penalty_gate != crate::scheduler::fast_greedy::PenaltyGate::Blocked {
        let t_fast = std::time::Instant::now();
        let vocab = model.vocab_size();
        let logits_base = model.logits_buffer_ptr();
        // Scoped history for the ReduceOnly immunity test — cloned before the
        // `&mut a.grammar_state` borrow below.
        let scoped_history: Vec<u32> =
            if fast_penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly {
                crate::scheduler::sample_step::penalty_history_scope(
                    &a.output_tokens,
                    ctx.tool_call_end_token,
                )
                .to_vec()
            } else {
                Vec::new()
            };
        let before = a.grammar_state.as_ref().map(|gs| gs.num_history_steps());
        let mut fast: Vec<u32> = Vec::with_capacity(k);
        let mut all_allowed = true;
        // Scoped block so `gs`'s mutable borrow ends before the post-loop
        // rollback re-borrows `a.grammar_state`. let-else (not `.expect()`)
        // keeps clippy happy — `is_some()` is gated in the `if` condition above.
        {
            let Some(gs) = a.grammar_state.as_mut() else {
                unreachable!("grammar_state present (gated by is_some above)")
            };
            for (i, &tok) in argmax_ids.iter().enumerate() {
                // ReduceOnly regime: the argmax must be penalty-immune (not in
                // the scoped history + raw logit > 0) or we take the slow path.
                // In-flight repeat: a token already picked earlier in THIS
                // block is in sequential decode's penalty history for this
                // position even though output_tokens has not grown yet. The
                // immunity proof does not cover it — slow path.
                if fast_penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly
                    && fast.contains(&tok)
                {
                    all_allowed = false;
                    break;
                }
                if fast_penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly
                    && !crate::scheduler::fast_greedy::argmax_immune(tok, &scoped_history, || {
                        crate::scheduler::fast_greedy::logit_is_positive(
                            model,
                            logits_base,
                            row_base + i,
                            vocab,
                            tok,
                        )
                    })
                {
                    all_allowed = false;
                    break;
                }
                let allowed = if gs.is_terminated() {
                    true // no further constraint past grammar completion
                } else {
                    gs.fill_bitmask();
                    gs.is_token_allowed(tok)
                };
                if !allowed {
                    all_allowed = false;
                    break;
                }
                fast.push(tok);
                // Speculatively advance so position i+1's bitmask reflects the
                // post-emit state (mirrors the slow path). Skip after the last.
                if i + 1 < k && !gs.is_terminated() {
                    let _ = gs.accept_token(tok);
                }
            }
        }
        // Roll back the speculative advances to the exact pre-call state
        // (history delta — stop/terminated tokens don't advance; BUG#3).
        if let (Some(b), Some(gs)) = (before, a.grammar_state.as_mut()) {
            let adv = gs.num_history_steps().saturating_sub(b);
            if adv > 0 {
                gs.rollback(adv);
            }
        }
        ctx.timing.record(Phase::FastGreedy, t_fast);
        if all_allowed && fast.len() == k {
            return fast; // no D2H, no CPU pipeline — all positions GPU-greedy + grammar-legal
        }
        // else: fall through to the slow path (matcher restored above).
    }

    // ── GRAMMARLESS fast-greedy (2026-07-30): the same #237 gate, minus the
    // bitmask ──
    //
    // The fast path above required `grammar_state.is_some()`, so plain chat —
    // the EASIEST regime (no mask to consult at all) — unconditionally paid
    // the slow tail below: per SEQUENCE per STEP, a blocking D2H of its
    // [K+1, vocab] logits rows (2 x 248,077 x BF16 = 992,308 B on the 27B at
    // the 16:1 ladder). The C=16 profile (PROGRESS_LOG 6.12) measured 2,541
    // such copies = 2.5 GB per 16x300-token burst, ~16 stream-drain waits per
    // step — the single largest slice of the host-bound decode wall.
    //
    // Eligibility mirrors the grammar arm exactly: greedy (temp==0 or forced),
    // not inside thinking, penalties classified by the SSOT `fast_greedy`
    // gate (Neutral, or ReduceOnly with the per-token immunity proof — same
    // scoped history, same `logit_is_positive` 2-byte probe). When every
    // position qualifies, the GPU argmax IS the masked-greedy pick and the
    // [K,vocab] D2H is skipped entirely.
    //
    // Same behavioral trade #237 shipped for grammar sequences: GPU-argmax
    // tie-breaking near equal logits can differ from the host FP32 scan, so
    // emitted tokens are NOT byte-invariant vs the slow path at near-ties.
    // Kill switch: ATLAS_NO_FAST_GREEDY_CHAT=1 restores the slow path.
    let chat_fast_gate = if ctx.sampling.fast_greedy_chat
        && a.grammar_state.is_none()
        && !a.inside_thinking
        && (a.temperature == 0.0 || ctx.sampling.force_temp_zero)
    {
        crate::scheduler::fast_greedy::classify_penalties(
            &crate::scheduler::sample_step::penalty_params_for(
                a,
                crate::scheduler::sample_step::PositionKind::Verify,
                0.0,
                None,
                Vec::new(),
            ),
        )
    } else {
        crate::scheduler::fast_greedy::PenaltyGate::Blocked
    };
    if chat_fast_gate != crate::scheduler::fast_greedy::PenaltyGate::Blocked {
        let t_fast = std::time::Instant::now();
        let vocab = model.vocab_size();
        let logits_base = model.logits_buffer_ptr();
        let scoped_history: Vec<u32> =
            if chat_fast_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly {
                crate::scheduler::sample_step::penalty_history_scope(
                    &a.output_tokens,
                    ctx.tool_call_end_token,
                )
                .to_vec()
            } else {
                Vec::new()
            };
        let all_immune = argmax_ids.iter().enumerate().all(|(i, &tok)| {
            chat_fast_gate == crate::scheduler::fast_greedy::PenaltyGate::Neutral
                || (!argmax_ids[..i].contains(&tok)
                    && crate::scheduler::fast_greedy::argmax_immune(tok, &scoped_history, || {
                        crate::scheduler::fast_greedy::logit_is_positive(
                            model,
                            logits_base,
                            row_base + i,
                            vocab,
                            tok,
                        )
                    }))
        });
        ctx.timing.record(Phase::FastGreedy, t_fast);
        if all_immune {
            return argmax_ids.to_vec(); // no D2H, no CPU pipeline
        }
        // else: some position needs the penalty-aware pipeline — slow path.
    }

    let vocab = model.vocab_size();
    // BF16 always for verify path: `decode_verify_graphed_*` writes BF16
    // to `logits_buffer()`. The FP32-lm_head path (Gemma-4 dense) does
    // not go through verify (no MTP for dense Gemma).
    let elem_bytes = 2usize;
    let total = k * vocab * elem_bytes;
    let t_d2h = std::time::Instant::now();
    let mut buf = vec![0u8; total];
    if model
        .copy_logits_to_host(
            model
                .logits_buffer_ptr()
                .offset(row_base * vocab * elem_bytes),
            &mut buf,
        )
        .is_err()
    {
        return argmax_ids.to_vec();
    }
    ctx.timing.record(Phase::D2h, t_d2h);

    let mut picks: Vec<u32> = Vec::with_capacity(k);
    // Snapshot the matcher's history depth BEFORE speculative advances so we
    // roll back exactly the ACTUAL advances afterward. BUG#3 (2026-06-02):
    // stop/EOS and terminated tokens return true from `accept_token` WITHOUT
    // advancing the matcher, so a count of `accept_token`→true calls would
    // over-rewind. `emit_token` (run after this helper) re-advances from the
    // restored, clean state.
    let grammar_steps_before = a.grammar_state.as_ref().map(|gs| gs.num_history_steps());

    for i in 0..k {
        let slice = &buf[i * vocab * elem_bytes..(i + 1) * vocab * elem_bytes];
        // P1-3 (2026-07-09): `i` threads the verify-position index down for
        // the per-position seed offset of the temp>0 sampling branch.
        let pick = verify_pick_with_pipeline(slice, false, vocab, a, ctx, i, &picks);
        picks.push(pick);

        // Early stop at the first draft mismatch (see fn doc): `pick` is the
        // correction token; sequential decode re-scores everything after it
        // from the corrected prefix next step. Pad the tail with raw argmax
        // (read only by shadow logging) and run NO further stateful stages.
        if let Some(d) = drafts
            && i < d.len()
            && pick != d[i]
        {
            picks.extend_from_slice(&argmax_ids[i + 1..]);
            break;
        }

        // Speculatively advance the matcher with `pick[i]` so the next
        // position's bitmask reflects post-emit state. Skip on the last
        // position (no next position to mask) and when the seq has no
        // grammar (nothing to advance).
        if i + 1 < k
            && let Some(ref mut gs) = a.grammar_state
            && !a.inside_thinking
        {
            // Matcher advance can fail if `pick` is not in the current
            // bitmask. If our pipeline correctly applied the bitmask,
            // pick is the argmax over masked logits → MUST be in the
            // bitmask → advance MUST succeed. The defensive check
            // exists for forced-token fast-path returns where the
            // grammar may have terminated; those legitimately can't
            // advance further.
            if !gs.accept_token(pick) {
                tracing::debug!(
                    pick,
                    i,
                    "verify_pick: grammar speculative advance refused — pipeline picked a token outside the current bitmask. \
                     This indicates a stale bitmask in the pipeline or a forced-token fastpath that terminated grammar. \
                     Stopping speculation here; the real `accept_token` in emit_token will fail and end the response."
                );
                break;
            }
            // accept_token advanced the matcher as a side effect; the rollback
            // below counts the ACTUAL advances from matcher history (BUG#3).
        }
    }

    // Roll back exactly the ACTUAL speculative advances (history delta) so the
    // matcher returns to its pre-call state; `emit_token` then re-advances it
    // normally. BUG#3: counting from accept_token→true calls over-rewinds when
    // a stop/EOS/terminated token (which returns true WITHOUT advancing) lands
    // in the verified span.
    if let (Some(before), Some(gs)) = (grammar_steps_before, a.grammar_state.as_mut()) {
        let advanced = gs.num_history_steps().saturating_sub(before);
        if advanced > 0 {
            gs.rollback(advanced);
        }
    }

    picks
}
