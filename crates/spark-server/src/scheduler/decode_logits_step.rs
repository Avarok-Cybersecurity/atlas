// SPDX-License-Identifier: AGPL-3.0-only

//! process_decode_logits: post-decode logits processing.

use super::*;

// The reusable host staging buffer for the D2H logits copy is now
// `SchedCtx::scratch.host_bytes` — same zero-contention single-thread
// access, with a lifetime that ends when the run does.

thread_local! {
    /// Dequant scratch for ONE rayon worker on the parallel host-sampling arm.
    ///
    /// The run-owned `SchedCtx::scratch` is a `RefCell`, so it is neither
    /// `Sync` nor shareable across the pool; a fan-out needs one buffer per
    /// worker or it double-borrows. The serial arm — and every other caller —
    /// still uses the run's scratch, so this exists only for the `n > 1`
    /// fan-out and holds nothing model-derived: `copy_logits_to_host`
    /// overwrites every byte it reads, and `seq_f32` is resized per call.
    static PAR_SAMPLE_SCRATCH: crate::scheduler::sched_ctx::DecodeScratch =
        crate::scheduler::sched_ctx::DecodeScratch::default();
}

/// Build the pre-sample pipeline's context around a chosen scratch buffer.
///
/// SSOT: the serial arm and each parallel worker differ ONLY in which
/// `DecodeScratch` they borrow, so the other nine fields are assembled once
/// here rather than spelled twice.
fn logits_ctx<'a>(
    sched: &'a crate::scheduler::sched_ctx::SchedCtx,
    scratch: &'a crate::scheduler::sched_ctx::DecodeScratch,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
) -> crate::scheduler::logit_processors::LogitsContext<'a> {
    crate::scheduler::logit_processors::LogitsContext {
        think_end_token,
        think_start_token,
        tool_call_start_token,
        tool_call_end_token,
        watchdog: sched.watchdog,
        scratch,
        dumps: &sched.dumps,
        stats: sched.stats.clone(),
        boundary_mask: sched.masks.boundary.clone(),
        mid_word_mask: sched.masks.mid_word.clone(),
        sampling: sched.levers.sampling(),
        timing: sched.timing.clone(),
    }
}

/// Admit `think_ended` rows (which need only a 2-token mask) to the GPU argmax
/// fast path. Kill switch: `ATLAS_NO_THINKENDED_GPU_ARGMAX=1`.
fn think_ended_gpu_argmax_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_NO_THINKENDED_GPU_ARGMAX")
            .ok()
            .as_deref()
            != Some("1")
    })
}

/// Steps that fell back to the host path because a GPU argmax landed on a
/// masked think token. Expected to be a small fraction; a large count means the
/// fast path is not paying for itself.
static THINK_MASK_FALLBACKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Parallel host sampling toggle (ATLAS_PARALLEL_SAMPLE, default ON). Set to
/// "0" to force the serial per-seq sampling path — an escape hatch for the
/// telemetry-ordering caveat above, or for A/B measurement of the rayon win.
fn parallel_sample_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_PARALLEL_SAMPLE").as_deref() != Ok("0"))
}

/// DIAG (ATLAS_DECODE_TIMING=1): localize the host-path decode cost. Splits the
/// per-token wall into `copy` (D2H of the full 248k-vocab logits + the GPU
/// forward-wait absorbed by that sync) vs `sample` (the host scalar loops over
/// 248k: BF16→FP32 expand + penalties + masks + argmax). Emits a 100-token
/// running summary. Zero-cost when the env var is unset (OnceLock-gated).
fn decode_timing_record(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    copy_us: u64,
    sample_us: u64,
) {
    use std::sync::atomic::Ordering;
    if !sched.levers.decode_timing {
        return;
    }
    let stats = &sched.stats;
    stats.decode_copy_us.fetch_add(copy_us, Ordering::Relaxed);
    stats
        .decode_sample_us
        .fetch_add(sample_us, Ordering::Relaxed);
    let n = stats.decode_count.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_multiple_of(100) {
        let c = stats.decode_copy_us.swap(0, Ordering::Relaxed);
        let s = stats.decode_sample_us.swap(0, Ordering::Relaxed);
        stats.decode_count.store(0, Ordering::Relaxed);
        tracing::info!(
            "DECODE_TIMING (last 100 host-path tokens): copy+fwd-wait={:.2}ms/tok sample(248k host)={:.2}ms/tok",
            c as f64 / 100_000.0,
            s as f64 / 100_000.0,
        );
    }
}

/// Sample and process decode logits for all active sequences.
///
/// Factored out of `step_decode_only` so that `mixed_forward` can reuse
/// the same sampling + token-processing logic without duplication (SSOT).
/// `logits` must point to `[n, vocab_size]` BF16 on device where n = active.len().
pub fn process_decode_logits(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    logits: DevicePtr,
    t0: std::time::Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    let n = active.len();

    // Grammar bitmask is CPU-side, so any sequence with active grammar forces
    // the host-side sampling path for its logits slice.
    let any_grammar = active.iter().any(|a| a.grammar_state.is_some());
    let any_logprobs = active.iter().any(|a| a.top_logprobs.is_some());
    // FP32 lm_head models (Gemma-4 dense) MUST use the host-side path —
    // `argmax_batch` assumes BF16 layout and would interpret 4-byte FP32
    // values as 2-byte BF16 pairs, returning garbage tokens.
    let model_logits_fp32 = model.decode_logits_fp32();
    // `think_ended` alone does NOT need the host path. PostCloseThinkMask masks
    // exactly TWO ids (think_end_token, think_start_token) and nothing else; A4's
    // bias floor is gated on `inside_thinking` (sample_step.rs) so it is inert
    // here. With request penalties exactly neutral the pipeline is then a no-op
    // and the emission is the raw argmax — modulo those two ids, which we check
    // after the fact and fall back for.
    //
    // This matters far beyond a tuning win: `--disable-thinking` sets
    // `think_ended = true` for EVERY sequence at birth (prefill_a_step.rs:229 and
    // three siblings), so before this change ONE such row — i.e. all of them —
    // forced the entire batch through a 7.95 MB D2H plus n full-vocab host
    // passes. The GPU argmax fast path was dead code in that config, which is
    // also the MLPerf-edge config.
    //
    // Kill switch: ATLAS_NO_THINKENDED_GPU_ARGMAX=1.
    let think_ended_gpu_ok = |a: &ActiveSeq| {
        a.think_ended
            && !a.inside_thinking
            && a.grammar_state.is_none()
            && a.repetition_penalty == 1.0
            && a.presence_penalty == 0.0
            && a.frequency_penalty == 0.0
            && a.lz_penalty == 0.0
            && a.dry_multiplier == 0.0
    };
    let admit_think_ended = think_ended_gpu_argmax_enabled();
    let needs_host_logits = active.iter().any(|a| {
        let excused = admit_think_ended && think_ended_gpu_ok(a);
        (a.inside_thinking || a.think_ended || a.grammar_state.is_some()) && !excused
    }) || any_logprobs
        || model_logits_fp32;

    // Try the GPU argmax first. `None` here means "not eligible, or the result
    // needs the host pipeline after all" and falls through to the host branch —
    // it must never mean "emit nothing".
    let fast_tokens: Option<Vec<(u32, Option<crate::api::TokenLogprobs>)>> =
        if active.iter().all(|a| a.temperature == 0.0) && !any_grammar && !needs_host_logits {
            match model.argmax_batch(logits, n, 0) {
                Ok(t) => {
                    // The two masked ids are the ONLY thing the host pipeline
                    // would have done differently for a think_ended row. If an
                    // argmax actually landed on one (rare — the model seldom
                    // re-opens <think> mid-response), fall through and redo the
                    // step on the host so the emitted token is exactly what the
                    // pipeline would produce.
                    let hit_mask = t.iter().zip(active.iter()).any(|(&tok, a)| {
                        a.think_ended
                            && (Some(tok) == think_end_token || Some(tok) == a.think_start_token)
                    });
                    if hit_mask {
                        THINK_MASK_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        None
                    } else {
                        Some(t.into_iter().map(|tok| (tok, None)).collect())
                    }
                }
                Err(e) => {
                    tracing::error!("argmax_batch error: {e:#}");
                    for mut a in active.drain(..) {
                        send_error(model, &mut a, &format!("{e:#}"));
                    }
                    return;
                }
            }
        } else {
            None
        };

    let new_tokens: Vec<(u32, Option<crate::api::TokenLogprobs>)> = if let Some(t) = fast_tokens {
        t
    } else {
        // Host-side path: copy all batch logits to host, sample per-sequence.
        // Required when any sequence has temperature > 0 or grammar constraints.
        let vocab_size = model.vocab_size();
        // FP32 lm_head dispatch (Gemma-4 dense). When `use_fp32_logits` is
        // on, the per-token decode lm_head writes 4 bytes/element. The
        // passed `logits` pointer is whatever the most-recent forward
        // returned — that's already the correct buffer (prefill or decode).
        // We just need to read it with the matching width.
        let logits_fp32 = model.decode_logits_fp32();
        let elem_bytes = if logits_fp32 { 4 } else { 2 };
        let t_copy = std::time::Instant::now();
        // Reuse the run's staging buffer (restored at the end of this block).
        // `resize` only grows it; `copy_logits_to_host` overwrites every byte
        // so the residual/zero-fill is irrelevant.
        let mut buf = sched.scratch.host_bytes.borrow_mut().split_off(0);
        buf.resize(n * vocab_size * elem_bytes, 0);
        if let Err(e) = model.copy_logits_to_host(logits, &mut buf) {
            tracing::error!("copy_logits_to_host error: {e:#}");
            for mut a in active.drain(..) {
                send_error(model, &mut a, &format!("{e:#}"));
            }
            return;
        }
        let copy_us = t_copy.elapsed().as_micros() as u64;
        let t_sample = std::time::Instant::now();
        // Per-sequence host sampling is independent: each call reads a
        // disjoint `buf` slice, mutates its own `ActiveSeq`, uses its own
        // dequant scratch, and advances a per-seq seed. The collect is
        // order-preserving, so the emitted tokens are identical to the serial
        // path. (Process-global sampler telemetry — `sampler::LAST_ENTROPY`,
        // the AdaDec/B1 diagnostics — is synchronized but last-write-wins
        // across workers, so those best-effort gauges may report an arbitrary
        // in-step sequence's value under n>1; token output is unaffected.)
        // Each call scans the full ~250k vocab (BF16->FP32 expand + penalties
        // + argmax), the dominant host-path cost at n>=2. Fan out across the
        // rayon pool ONLY for n>1: at n=1 (the common single-stream /
        // opencode host path) the serial path avoids rayon's dispatch
        // overhead. `process_seq_logits` touches no GPU state (its `_model`
        // arg is unused), so no CUDA calls cross threads. The parallel path
        // is also gated off when the opt-in `ATLAS_LOGIT_DUMP` diagnostic is
        // active, since its shared per-step record would interleave across
        // workers.
        let parallel_sample = n > 1 && sched.dumps.logits.is_none() && parallel_sample_enabled();
        let sampled: Vec<(u32, Option<crate::api::TokenLogprobs>)> = if parallel_sample {
            use rayon::prelude::*;
            // `SchedCtx` itself is NOT `Sync` (its `DecodeScratch` is a
            // `RefCell`), so the fan-out closure must not capture it. Hoist
            // the pieces the pipeline reads — all of them `Copy`, `Arc` or
            // `&`-to-`Sync` — and leave the scratch to each worker.
            let dumps = &sched.dumps;
            let watchdog = sched.watchdog;
            let stats = sched.stats.clone();
            let boundary_mask = sched.masks.boundary.clone();
            let mid_word_mask = sched.masks.mid_word.clone();
            let sampling = sched.levers.sampling();
            let timing = sched.timing.clone();
            active
                .par_iter_mut()
                .enumerate()
                .map(|(i, a)| {
                    // The run's `DecodeScratch` is a `RefCell` — neither
                    // `Sync` nor shareable across the pool — so each worker
                    // borrows its own and builds the context around it. Same
                    // reuse, one buffer per worker instead of one per run.
                    PAR_SAMPLE_SCRATCH.with(|scratch| {
                        let ctx = crate::scheduler::logit_processors::LogitsContext {
                            think_end_token,
                            think_start_token,
                            tool_call_start_token,
                            tool_call_end_token,
                            watchdog,
                            scratch,
                            dumps,
                            stats: stats.clone(),
                            boundary_mask: boundary_mask.clone(),
                            mid_word_mask: mid_word_mask.clone(),
                            sampling,
                            timing: timing.clone(),
                        };
                        process_seq_logits(
                            model,
                            a,
                            &buf,
                            i,
                            vocab_size,
                            elem_bytes,
                            logits_fp32,
                            &ctx,
                            adaptive_sampling,
                        )
                    })
                })
                .collect()
        } else {
            let ctx = logits_ctx(
                sched,
                &sched.scratch,
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
            );
            active
                .iter_mut()
                .enumerate()
                .map(|(i, a)| {
                    process_seq_logits(
                        model,
                        a,
                        &buf,
                        i,
                        vocab_size,
                        elem_bytes,
                        logits_fp32,
                        &ctx,
                        adaptive_sampling,
                    )
                })
                .collect()
        };
        decode_timing_record(sched, copy_us, t_sample.elapsed().as_micros() as u64);
        // Return the staging buffer for reuse next token (its capacity is
        // preserved). The error path above intentionally drops it — that is
        // rare and only forfeits the cached capacity.
        *sched.scratch.host_bytes.borrow_mut() = buf;
        sampled
    };
    let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if tracing::enabled!(tracing::Level::DEBUG) {
        let token_ids: Vec<u32> = new_tokens.iter().map(|(t, _)| *t).collect();
        tracing::debug!(
            "DECODE: n={n} step={step_ms:.1}ms ({:.1} tok/s) tokens={:?}",
            1000.0 * n as f64 / step_ms,
            token_ids,
        );
    }

    let now = Instant::now();
    for (i, (tok, logprobs)) in new_tokens.into_iter().enumerate() {
        let a = &mut active[i];
        a.last_token = tok;
        a.last_token_time = now;

        // Fix B (2026-06-05, kill-switch): <tool_response> hard stop. This decode
        // path has no `<|im_start|>` hard-stop block (that lives only in
        // emit_step.rs), so add the guard at the earliest safe point in the
        // per-token handler — before grammar advance / EOS handling. The model
        // must never generate this control token; if it does (post-tool-call
        // runaway), end the turn. Uses `continue` (loop body), not `return`.
        if tool_response_stop_enabled()
            && let Some(trs) = sched.limits.tool_response_hard_stop
            && tok == trs
        {
            a.output_tokens.push(tok);
            a.finished = true;
            // Name the cut. Without this the ladder skips its guard rung,
            // budget is untouched, and the turn wires as "stop" -- telling
            // the client the model finished when a guard ended it. Unlike
            // `<|im_start|>`, this token is NOT eos-registered, so the
            // `is_eos` rung does not catch it either.
            a.guard_stop = Some(GUARD_STOP_TOOL_RESPONSE);
            tracing::debug!("<tool_response> hard-stop fired (id={trs}); ending turn");
            continue;
        }

        // Spontaneous <think>: model generates <think> even when thinking
        // was not requested. Enter thinking mode so EOS is suppressed and
        // thinking content is stripped. Matches vLLM's behavior of always
        // parsing <think>...</think> regardless of enable_thinking setting.
        //
        // F9+F10 (2026-04-26): the sample-time logit mask at line ~1716
        // hard-blocks `<think>` when `think_ended=true`, so this branch
        // should not fire after a watchdog has force-closed thinking.
        // Defence-in-depth: if the model somehow still emits <think>
        // (e.g. the start token differs from the masked one in edge
        // cases), decay the budget by `>> watchdog_fires.min(4)` so
        // each successive re-entry has a tighter window. After 4+
        // fires, the budget is 1/16 of normal — the watchdog kills
        // re-entry within a handful of tokens.
        if !a.inside_thinking && think_start_token == Some(tok) {
            let decay_shift = a.think_watchdog_fires.min(4);
            let decayed = a.spontaneous_think_budget >> decay_shift;
            a.inside_thinking = true;
            a.think_ended = false; // reset so </think> detection path works
            a.think_skip_count = 0;
            a.thinking_budget = Some(decayed.max(8)); // floor to keep watchdog functional
            if a.think_watchdog_fires > 0 {
                tracing::debug!(
                    fires = a.think_watchdog_fires,
                    decayed_budget = decayed,
                    "Spontaneous <think> re-entry after watchdog; decayed budget"
                );
            } else {
                tracing::debug!("Spontaneous <think> detected, entering thinking mode");
            }
            continue; // don't emit <think> as content
        }

        // Silently skip </think> tokens outside thinking mode.
        // At long context (37k+), models degenerate into repeating </think>.
        // Skip up to 50 occurrences, then force-stop. This gives cached
        // prompts a chance to produce content while limiting degenerate loops.
        if !a.inside_thinking && think_end_token == Some(tok) {
            a.think_skip_count += 1;
            if a.think_skip_count >= 50 {
                a.finished = true;
                // Name the cut — same class as the `<tool_response>` stop
                // above. `</think>` is NOT eos-registered (see
                // `GUARD_STOP_THINK_SKIP`), and this site SKIPS the token,
                // so `last_tok` stays a plain content token: unnamed, the
                // ladder wires "stop" and the agentic harness (which
                // recovers only on "length") ends the run silently.
                a.guard_stop = Some(GUARD_STOP_THINK_SKIP);
                tracing::debug!(
                    "</think> think-skip watchdog hard-stop fired (50 consecutive strays); \
                     ending turn"
                );
            }
            continue;
        }
        // Reset skip counter when a real content token is generated.
        if a.think_ended {
            a.think_skip_count = 0;
        }

        // Advance grammar state with the sampled token — but only
        // once thinking is finished, because thinking tokens are
        // stripped from the API output and should not consume grammar
        // slots (matches the bitmask-skip in the sampler above).
        if !a.inside_thinking
            && let Some(ref mut gs) = a.grammar_state
        {
            gs.accept_token(tok);
        }

        // §C-1 (DS4F hard-limit lane, 2026-07-21): thinking tokens draw down
        // the SAME completion budget (`remaining`) as content tokens, so a long
        // `<think>` block can no longer run the request past its declared
        // `max_tokens` (R1X overrun). `thinking_budget`/`max_thinking_budget`
        // stays the separate per-BLOCK cap (armed below); this is the overall
        // allowance. No-op for direct-mode (thinking-OFF) turns — they never
        // enter this branch — so the 35/40 baseline is unchanged. When this
        // decrement exhausts the budget the `remaining == 0` force-stop on the
        // commit path (below) finishes the sequence even while inside thinking
        // (§C-2 budget half). Every generated token (content OR thinking,
        // including `</think>`) now decrements exactly once.
        if a.inside_thinking {
            a.consume_generation_budget();
            if think_end_token == Some(tok) {
                a.inside_thinking = false;
                a.force_end_thinking = false;
                a.sentence_defer_count = 0;
                a.consecutive_confident = 0;
                a.in_code_fence = false;
                a.think_ended = true;
                // One-shot: pin the next sampled token to the
                // tool-call-start token if the request requires a
                // tool call (Change 3b). Cleared in the `else`
                // branch below on the next emit.
                a.think_just_ended = true;
            } else {
                a.thinking_tokens += 1;
                // Track ``` code-fence parity within the thinking block:
                // each fence token flips in/out of a fenced code span.
                // The F2 confidence early-stop (process_seq_logits) is
                // suppressed while `in_code_fence` — code is near-
                // deterministic (high top-1 prob) but that is NOT a
                // "done reasoning" signal; braking here truncates the
                // model mid-statement. THINK_LOOP (below) deliberately
                // stays active even inside fences: it catches
                // *repeating* fence-narration, not one coherent block.
                a.in_code_fence = toggle_code_fence(a.in_code_fence, tok, code_fence_token);
                // Set force_end_thinking when budget exhausted (picked up next iteration)
                if let Some(budget) = a.thinking_budget
                    && a.thinking_tokens >= budget
                    && !a.force_end_thinking
                {
                    a.force_end_thinking = true;
                    a.sentence_defer_count = 0;
                    // Name the budget's SOURCE: a 256-class cut with a large
                    // --max-thinking-budget in force means the CLIENT sent the
                    // budget (explicit tokens or a reasoning_effort rung) —
                    // the knob to turn is in the request, not the server.
                    tracing::info!(
                        source = if a.enable_thinking {
                            "request (client budget/effort; scaled by --max-thinking-budget)"
                        } else {
                            "spontaneous <think> (--max-thinking-budget / MODEL.toml)"
                        },
                        "Thinking budget exhausted ({budget} tokens), arming </think>; \
                         deferring up to {MAX_SENTENCE_DEFER_TOKENS} tokens for sentence boundary"
                    );
                }
                // Token-level fence-loop detection. Catches the Qwen3.5-35B
                // phrase attractor (`Running:\`\`\`bash cmd\`\`\`Executing:…`
                // cycling) within ~24-60 tokens of the loop starting,
                // instead of waiting for the 256-token thinking budget.
                if !sched.levers.disable_watchdogs
                    && sched.watchdog.enable_think_loop_watchdog
                    && !a.force_end_thinking
                    && a.thinking_tokens >= THINK_LOOP_MIN_TOKENS
                    && a.thinking_tokens.is_multiple_of(THINK_LOOP_CHECK_STRIDE)
                    && detect_thinking_token_loop_with(
                        &a.output_tokens,
                        a.repetition_detection,
                        sched.watchdog,
                    )
                {
                    a.force_end_thinking = true;
                    a.sentence_defer_count = 0;
                    a.think_watchdog_fires = a.think_watchdog_fires.saturating_add(1);
                    tracing::warn!(
                        thinking_tokens = a.thinking_tokens,
                        watchdog_fires = a.think_watchdog_fires,
                        "Thinking-loop watchdog fired (period-{}…{} repeat in tail); forcing </think> early",
                        THINK_LOOP_PERIOD_MIN,
                        THINK_LOOP_PERIOD_MAX,
                    );
                }
            }
        } else {
            // Content-phase token: budget bookkeeping + the content-loop
            // and inter-tool-prose watchdogs. Extracted to
            // `decode_logits_content.rs` to keep this file ≤500 LoC.
            // `model` is threaded through so a watchdog rollback can
            // restore SSM recurrent state on hybrid models (Phase-C).
            handle_content_token(a, model, sched);
        }

        // Track <tool_call> token: once seen, legacy tool call requirement is satisfied.
        // Guard with !inside_thinking — a <tool_call> inside thinking is spurious
        // and must not clear require_tool_call (which would allow premature EOS).
        if a.require_tool_call && tool_call_start_token == Some(tok) && !a.inside_thinking {
            a.require_tool_call = false;
            a.tool_call_opened = true;
        }
        // F2 (2026-04-26): reset the inter-tool prose budget on
        // every `<tool_call>` open. This keeps the budget scoped to
        // "free-text since the last tool call started" rather than
        // accumulating across the whole response.
        if tool_call_start_token == Some(tok) && !a.inside_thinking {
            a.prose_tokens_since_last_tool = 0;
            // Tool-call-repetition runaway guard. On a `tool_choice="auto"`
            // grammar turn the grammar never terminates after a tool call
            // (stop_after_first=false), so EOS stays grammar-suppressed and the
            // only stop path is the ATLAS_TOOL_EOS_ESCAPE hatch — which a
            // re-opened tool body defeats (its `!inside_tool_body` guard flips
            // false the moment the model emits another `<tool_call>`). A
            // degenerating FP8/long-context model loops emitting whole
            // `<tool_call>…</tool_call>` blocks as content; each closes cleanly
            // so the envelope-streak guard never fires, and the turn burns to
            // max_tokens. Count opens that happen AFTER a real call already
            // completed; once past threshold the turn is provably degenerating
            // and we force-finish it (below).
            if a.tool_call_completed {
                a.post_completion_tool_opens = a.post_completion_tool_opens.saturating_add(1);
                // Threshold = how many EXTRA `<tool_call>` openers (after the
                // first completed) mark the turn as a degenerate content-leak
                // loop with no legitimate continuation. Measured: real
                // degenerate runaways emit 50-58 blocks in a single decode
                // burning to the 8192 cap; a model making genuine back-to-back
                // calls in one decode tops out far lower and then STOPS. 8 sits
                // safely above any plausible legit single-decode multi-call
                // (catches the runaway at ~8 blocks ≈ ~1.2k tokens, an order of
                // magnitude below the 8k-token cap it used to hit) while leaving
                // generous headroom so a legitimate multi-call turn is never
                // truncated. This is the only path that reliably halts the
                // runaway — lifting the post-sample EOS suppression alone is not
                // enough if the grammar bitmask never surfaces an EOS token
                // during the auto-mode alternation. Mirrors the existing
                // MAX_TOOL_BODY_TOKENS envelope guard (emit_step.rs), which
                // force-finishes the never-closing variant; this handles the
                // closing-but-repeating variant.
                const MAX_POST_COMPLETION_TOOL_OPENS: u32 = 8;
                if a.post_completion_tool_opens >= MAX_POST_COMPLETION_TOOL_OPENS {
                    tracing::warn!(
                        opens = a.post_completion_tool_opens,
                        "tool-call repetition runaway: model re-opened {MAX_POST_COMPLETION_TOOL_OPENS}+ tool-call blocks after a completed call on a tool_choice=auto turn; ending response (was burning to max_tokens). Sanitizer keeps the first valid call(s)."
                    );
                    a.output_tokens.push(tok);
                    a.tool_call_opened = true;
                    if let Some(ref mut gs) = a.grammar_state {
                        gs.accept_token(tok);
                    }
                    a.finished = true;
                    continue;
                }
            }
        }
        // Safety: if require_tool_call is still set after 512 tokens, the model
        // isn't generating a tool call (grammar may have failed to compile).
        // Clear the flag so EOS is no longer suppressed — prevents infinite gen.
        if a.require_tool_call && a.output_tokens.len() > 512 {
            tracing::warn!(
                "require_tool_call safety: no <tool_call> after 512 tokens, clearing EOS suppression"
            );
            a.require_tool_call = false;
        }

        // Accumulate logprobs data for blocking responses.
        if let Some(lp) = logprobs {
            a.logprobs_data.push(lp);
        }

        // </tool_call> handling. Tool-armed requests (grammar active OR
        // `tools_present`) continue generating past a closed call so the model
        // can emit multiple/parallel calls (#192); only a NON-tool request
        // that spuriously emits `</tool_call>` hard-stops here.
        if tool_call_end_token == Some(tok) && !a.inside_thinking {
            a.output_tokens.push(tok);
            // Fix A (2026-06-05): mark the tool call complete so the EOS-escape
            // gate (below) can lift suppression. Inert unless
            // `tool_eos_escape_enabled()` (default OFF).
            a.tool_call_completed = true;
            if let ResponseSink::Streaming(ref tx) = a.sink {
                let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                    StreamEvent::TokenWithLogprobs(tok, lp)
                } else {
                    StreamEvent::Token(tok)
                };
                if !super::mod_helpers::bounded_stream_send(tx, event, "tool_call_end") {
                    tracing::warn!(
                        "Streaming receiver dropped during tool_call_end, finishing sequence"
                    );
                    a.finished = true;
                    continue;
                }
            }
            if a.grammar_state.is_none() && !a.tools_present {
                // Plain-chat hard stop: a request with NO tools declared has no
                // business emitting `<tool_call>` blocks — end the turn (the
                // historical "legacy mode" behavior, now scoped to non-tool
                // requests only).
                //
                // #192: when tools ARE declared (`tools_present`), a closed
                // tool call no longer finishes the sequence even without an
                // active grammar (grammar disengaged mid-response on a
                // model/matcher disagreement, opted out, or disabled). vLLM
                // parity: keep decoding so the model can emit PARALLEL calls;
                // the turn ends at natural EOS (require_tool_call was cleared
                // at `<tool_call>`, so EOS is no longer suppressed) or via the
                // tool watchdogs (post-completion open cap above, prose
                // budget, loop detectors) if it runs on.
                a.finished = true;
            }
            // Mirror finish_sequence (lines ~3445-3448): keep
            // `inside_tool_body` and the grammar FSM in sync with the
            // emitted token stream. The `continue;` below skips the
            // `emit_token()` path that would normally do this, so
            // without these two lines the flag stays `true` for all
            // subsequent prose tokens — sampler penalties stay
            // disabled for the rest of the response, and the grammar
            // bitmask drifts out of sync with the actual emission.
            // Root-caused 2026-04-26 (8-agent sweep, F1).
            a.inside_tool_body = false;
            if let Some(ref mut gs) = a.grammar_state {
                gs.accept_token(tok);
            }
            // F9 companion (2026-04-26): clear `think_ended` at every
            // </tool_call> boundary so legitimate post-tool
            // re-thinking is allowed. F9 masks <think> when
            // `think_ended=true`, but between tool calls the model
            // SHOULD be allowed to re-think (MiniMax-M2 / Qwen3.6
            // pattern per project_minimax_m27_final.md). F10's
            // watchdog-fire counter still applies — repeated
            // re-thinking that loops will decay its budget.
            a.think_ended = false;
            continue;
        }

        // EOS handling: grammar-based, legacy, or min_tokens.
        // Grammar-based: grammar controls when EOS is allowed (is_terminated()).
        // Legacy: require_tool_call suppresses EOS until <tool_call> is seen.
        // min_tokens: suppress EOS until output_tokens.len() >= min_tokens.
        // Fix A (2026-06-05, kill-switch): in tool_choice="auto" the grammar's
        // is_terminated() never becomes true after a tool call, so EOS is
        // suppressed forever — trapping the model into a hallucinated-transcript
        // runaway. When enabled and a tool call has completed (and we're not
        // inside a tool body / thinking), lift the grammar suppression so the
        // model's natural EOS ends the turn. Inert unless ATLAS_TOOL_EOS_ESCAPE=1.
        let eos_escape = tool_eos_escape_enabled()
            && a.tool_call_completed
            && !a.inside_tool_body
            && !a.inside_thinking;
        // #192: grammar EOS suppression is STOP-LEGALITY based (may the
        // response legally end at the current matcher position?), not
        // `!is_terminated()`. A tool_choice="auto" trigger grammar never
        // terminates, so the old gate suppressed EOS for the whole turn when
        // no call completed — armed-but-unused tools ran to
        // finish_reason="length" (live probe #6, 2026-07-02). Evaluated only
        // when the sampled token IS an EOS token: `grammar_blocks_stop`
        // fills a bitmask (`stop_legal`), too costly as a per-token
        // predicate and meaningless otherwise.
        let grammar_suppresses_eos = a.eos_tokens.contains(&tok)
            && !eos_escape
            && crate::grammar::grammar_blocks_stop(a.grammar_state.as_mut(), &a.eos_tokens);
        let legacy_suppresses_eos = a.require_tool_call;
        let min_tokens_suppresses = a.output_tokens.len() < a.min_tokens;
        // Suppress EOS during thinking: <|im_end|> inside <think> is spurious.
        // Only </think> (think_end_token) should end the thinking phase — EXCEPT
        // at a hard ceiling. §C-2 (DS4F hard-limit lane): when the completion
        // budget is exhausted or the served `max_seq_len` is reached, a
        // model-sampled EOS MUST be honored regardless of `inside_thinking`, so
        // generation cannot overrun. `remaining` already reflects this token's
        // decrement (thinking or content branch above); `a.seq.seq_len` is the
        // current KV position. No-op until a ceiling is actually hit.
        let hard_ceiling = hard_ceiling_hit(a.remaining, a.seq.seq_len, sched.limits.max_seq_len);
        let thinking_suppresses_eos = eos_suppressed_by_thinking(a.inside_thinking, hard_ceiling);
        // Post-thinking EOS guard. Empirically (dump fix22b 2026-04-25
        // ses_23b4781f7ffebc7UgkKWedTmjd seq=43): when the thinking-loop
        // watchdog force-closes `</think>` mid-narration, the model can
        // emerge into content mode briefly (often emitting a bare
        // `<write>\n\n` opener) and immediately sample EOS — the
        // session ends with a partial tool-call shell but no real
        // call. POST_THINK_MIN_CONTENT requires N non-thinking tokens
        // after `think_ended` before EOS is allowed, giving the model
        // room to actually open a `<tool_call>` block.
        //
        // 2026-05-24 narrowing (verified live against Qwen3.6-35B-A3B-FP8
        // T1-T6 battery): the guard was firing UNCONDITIONALLY for every
        // post-`</think>` response under 16 content tokens, including
        // genuine short-answer turns ("2+2"→"4", "first 5 primes"
        // →"2,3,5,7,11", "haiku featuring blue"→single line). The model
        // had emitted a perfectly valid short answer + `<|im_end|>` —
        // the guard then forced the model to keep generating, and it
        // collapsed into chat-template artefacts (`\nuser\nassistant\n`)
        // because there's no natural continuation. Scope the guard to
        // tool-call-eligible turns: when tools are armed (require_tool_call
        // OR `tools_active` per-seq) we keep the suppression; otherwise
        // a short post-thinking answer is the expected output and EOS
        // should fire normally. `min_tokens_suppresses` still enforces
        // any explicit caller-set floor.
        const POST_THINK_MIN_CONTENT: u32 = 16;
        let post_think_content_tokens =
            (a.output_tokens.len() as u32).saturating_sub(a.thinking_tokens);
        // Tools-armed scoping (the narrowing the 2026-05-24 comment above
        // describes but was never coded into this branch): the post-think guard
        // exists to give the model room to OPEN a `<tool_call>` after the
        // watchdog force-closes `</think>` mid-narration. That is only relevant
        // when tools are armed for this turn. On a plain (no-tool) thinking turn
        // a short post-`</think>` answer ("2+2"→"4", "say hello"→"Hello") plus
        // its `<|im_end|>`/`<|endoftext|>` IS the expected output, so the guard
        // must NOT fire — otherwise the legitimate EOS is discarded and the
        // model runs on into chat-template scaffold (`\nuser\nassistant`). The
        // MTP-verify emit path (`emit_step.rs`) has no such guard, which is why
        // MTP-on stopped here while MTP-off leaked; this restores parity.
        let tools_armed = a.require_tool_call || a.tool_request;
        let post_think_suppresses_eos =
            tools_armed && a.think_ended && post_think_content_tokens < POST_THINK_MIN_CONTENT;
        let suppress_eos = grammar_suppresses_eos
            || legacy_suppresses_eos
            || min_tokens_suppresses
            || thinking_suppresses_eos
            || post_think_suppresses_eos;

        if a.eos_tokens.contains(&tok) && !suppress_eos {
            // Stop/EOS token: do NOT stream to client (OpenAI spec: returned text
            // must not contain the stop sequence). The token is still added to
            // output_tokens for correct token count; the API layer strips the
            // decoded text for blocking responses.
            a.output_tokens.push(tok);
            crate::scheduler::emit_step::update_tool_param_state(a, tok);
            a.finished = true;
        } else if a.eos_tokens.contains(&tok) && suppress_eos {
            // EOS suppressed: grammar not terminated or legacy tool call not yet seen.
            // Don't stop, don't stream the EOS — the model must keep generating.
            // Don't add to output_tokens (EOS is discarded).
            //
            // This branch was silent, which made a real failure mode invisible:
            // a model that finishes reasoning, samples EOS, has it discarded,
            // and — having nothing to continue with — degenerates into repeating
            // a single token until something else stops it. The sibling
            // post-think guard above documents exactly that collapse
            // ("forced the model to keep generating, and it collapsed into
            // chat-template artefacts because there's no natural continuation").
            // Count it and say WHICH guard held, so the next occurrence is one
            // grep away instead of a token-id archaeology session.
            // THE MODEL IS TRYING TO STOP. If the ONLY reason we are holding it
            // back is that it is still inside <think>, discarding the EOS does
            // not buy a better answer -- it strands the model with no natural
            // continuation, and it samples EOS again on every subsequent step
            // until something else stops it. Observed on Laguna-S-2.1: EOS
            // sampled at thinking_tokens=18 and every step after, with content
            // degenerating into 'I\nI\nI\n...' / '####...' / backtick runs
            // until finish=length.
            //
            // So treat it as an IMPLICIT close of the thinking block, mirroring
            // exactly what the real `</think>` commit does above. We still drop
            // THIS EOS, which gives the model one clean shot at writing content;
            // if it immediately samples EOS again the block is no longer open,
            // `thinking_suppresses_eos` is false, and the turn ends normally.
            //
            // Scoped to the case where thinking is the SOLE suppressor: a
            // grammar that has not terminated, an unsatisfied tool call, an
            // explicit min_tokens floor or the post-think tool guard all have
            // their own reasons to keep generating and are left untouched.
            let thinking_is_sole_suppressor = thinking_suppresses_eos
                && !grammar_suppresses_eos
                && !legacy_suppresses_eos
                && !post_think_suppresses_eos
                && !min_tokens_suppresses;
            // Gated per model: honoring a mid-think EOS closes the block, and
            // from there only POST_THINK_MIN_CONTENT holds the turn open, so a
            // model that would have recovered instead emits ~16 tokens of
            // narration and stops WITHOUT its tool call. Laguna-S-2.1 needs
            // the close; Qwen3.6-35B-A3B measurably regresses on it (8/10 vs
            // 10/10 agentic, three consecutive tiers). Default false.
            let honor_eos_in_think = sched.watchdog.honor_eos_inside_thinking;
            if thinking_is_sole_suppressor && honor_eos_in_think {
                a.inside_thinking = false;
                a.think_force_closed = true;
                a.force_end_thinking = false;
                a.sentence_defer_count = 0;
                a.consecutive_confident = 0;
                a.in_code_fence = false;
                a.think_ended = true;
                a.think_just_ended = true;
            }
            tracing::debug!(
                target: "atlas::eos",
                tok,
                implicit_think_close = thinking_is_sole_suppressor && honor_eos_in_think,
                thinking_sole_suppressor = thinking_is_sole_suppressor,
                honor_eos_inside_thinking = honor_eos_in_think,
                inside_thinking = a.inside_thinking,
                think_ended = a.think_ended,
                thinking_tokens = a.thinking_tokens,
                by_thinking = thinking_suppresses_eos,
                by_grammar = grammar_suppresses_eos,
                by_legacy_tool = legacy_suppresses_eos,
                by_post_think = post_think_suppresses_eos,
                by_min_tokens = min_tokens_suppresses,
                "EOS suppressed; model forced to continue"
            );
        } else {
            // Per-token ledger (debug) — parity mirror of the `TOK` line in
            // `emit_step::emit_token`, which only the spec/MTP paths reach.
            // Without a twin here a serial run cannot be diffed against a
            // speculative run in TOKEN space, only in text.
            tracing::debug!(
                "TOK slot={} out_idx={} tok={}",
                a.seq.slot_idx,
                a.output_tokens.len(),
                tok,
            );
            a.output_tokens.push(tok);
            // SM1 (2026-05-26): drive the tool-body / parameter-body
            // state machine from the non-spec decode path. Previously
            // only spec/verify paths called this (via emit_token),
            // leaving every dependent gate (close-tag mask, AM1, B1,
            // A1) silently dead under `mtp=false`.
            crate::scheduler::emit_step::update_tool_param_state(a, tok);
            // Phase-C: if this committed token is a content-phase
            // boundary token (sentence end / newline) and the model is
            // hybrid (attention + SSM), snapshot the recurrent SSM
            // state now so a later watchdog rollback to this boundary
            // can also rewind h_state/conv_state — not just the KV
            // cache. Gated to content tokens because the watchdogs that
            // roll back all fire post-`</think>`, and `apply_rollback`
            // requires every dropped token to be a content token. No-op
            // for pure-attention models / disabled rings (see
            // `rollback::snapshot_boundary_if_ssm`).
            if !a.inside_thinking {
                rollback::snapshot_boundary_if_ssm(a, model, sched);
                // #155 iter3: block-aligned Marconi checkpoint on the
                // non-MTP decode path (live SSM state is canonical here).
                model.decode_marconi_checkpoint(&mut a.seq);
            }
            // OPENCODE FIX: when the model spontaneously emits `<think>` even
            // though the request didn't ask for thinking (`enable_thinking=false`),
            // the `<think>` open token itself is suppressed (line ~1356), but
            // the thinking-content tokens that follow MUST also be kept off the
            // wire — otherwise opencode persists them as `assistant.content` and
            // on the next turn the model sees its own past garbage (fake
            // `<function=…>`, fake `<tool_response>`) as a "format example" and
            // continues the pattern. Tokens stay in `output_tokens` for the
            // blocking response path's reasoning_content extraction.
            let suppress_stream = a.inside_thinking && !a.enable_thinking;
            if let ResponseSink::Streaming(ref tx) = a.sink
                && !suppress_stream
            {
                let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                    StreamEvent::TokenWithLogprobs(tok, lp)
                } else {
                    StreamEvent::Token(tok)
                };
                if !super::mod_helpers::bounded_stream_send(tx, event, "decode_logits token") {
                    tracing::debug!("Streaming receiver dropped (decode_logits), finishing seq");
                    a.finished = true;
                }
            }
            if a.remaining == 0 {
                // #144: non-MTP twin of the budget-aware close in
                // `emit_step::emit_token`. The grammar already accepted `tok`
                // above (line ~230), so it is at the current position; if it
                // is active and cannot legally stop here (open JSON string),
                // emit the shortest grammar-legal close so the length-stopped
                // output is still parseable.
                crate::scheduler::emit_step::emit_grammar_close(a);
                tracing::info!(
                    "process_decode_logits: remaining=0, output_tokens={}, thinking_tokens={}",
                    a.output_tokens.len(),
                    a.thinking_tokens
                );
                a.finished = true;
            }
            // §C-3 (DS4F hard-limit lane, 2026-07-21): per-step context-ceiling
            // stop. Independent of thinking state and of `max_tokens` — enforces
            // the served `max_seq_len` DURING decode instead of relying on the
            // on-completion true-up (`middleware.rs`), which let a long `<think>`
            // block run KV past the ceiling (R1X overrun past max_seq_len=8192).
            // Finishes with no EOS pushed → lifecycle reports finish=length.
            // No-op when `max_seq_len` is unset (0) or not yet reached, so
            // direct-mode short answers are unaffected.
            if !a.finished && seqlen_force_stop(a.seq.seq_len, sched.limits.max_seq_len) {
                tracing::info!(
                    seq_len = a.seq.seq_len,
                    max_seq_len = sched.limits.max_seq_len,
                    output_tokens = a.output_tokens.len(),
                    "process_decode_logits: max_seq_len ceiling reached; force-stop (finish=length)"
                );
                a.finished = true;
            }
            // Grammar termination = end of sequence. With `stop_after_first=true`
            // (tool_choice="required"), the structural-tag matcher transitions
            // to its terminal state right after the single tool call closes.
            // The model's free distribution past that point can be degenerate
            // (Nemotron-Super-120B emits a `</parameter>` loop and never
            // samples EOS naturally). Finish here instead of letting it run.
            if a.grammar_state
                .as_ref()
                .is_some_and(|gs| gs.is_terminated())
            {
                a.finished = true;
            }

            // Intra-response fuzzy repetition detection: if the last 2*W tokens
            // approximately match the same W-token pattern, the model is looping.
            // Uses Hamming distance with ~12% tolerance to catch loops where the
            // model narrates the same plan with slight wording variations.
            // Skip during tool calls: XML parameter tags have natural repetition
            // (<parameter=..>...</parameter>) that triggers false positives.
            // Use last occurrence positions — completed tool calls shouldn't
            // disable the detector for subsequent text generation.
            let last_tc_start = a
                .tool_call_start_token
                .and_then(|t| a.output_tokens.iter().rposition(|&tok| tok == t));
            let last_tc_end = a
                .tool_call_end_token
                .and_then(|t| a.output_tokens.iter().rposition(|&tok| tok == t));
            let inside_tool_call = match (last_tc_start, last_tc_end) {
                (Some(start), Some(end)) => start > end,
                (Some(_), None) => true,
                _ => false,
            };
            if sched.levers.loop_watchdog()
                && !a.finished
                && !a.inside_thinking
                && !inside_tool_call
                && let Some((pattern_len, mis_a, mis_b)) = detect_fuzzy_repetition(
                    &a.output_tokens,
                    sched.watchdog.fuzzy_repeat_tolerance_div,
                )
            {
                // Phase-C: roll back past the repeated window and
                // re-steer. `min_keep` = pattern_len * 3 guarantees all
                // three near-copies of the detected pattern are dropped
                // so generation cannot resume straight back into the
                // loop. Falls back to the hard stop when declined.
                let min_keep = pattern_len * 3;
                match rollback_to_boundary(a, min_keep, model, sched) {
                    RollbackOutcome::RolledBack { dropped } => {
                        tracing::warn!(
                            pattern_len,
                            mismatches = mis_a + mis_b,
                            dropped,
                            rollback = a.rollback_count,
                            "Fuzzy repetition detected; rolled back to boundary, re-steering"
                        );
                    }
                    RollbackOutcome::Fallback(reason) => {
                        tracing::warn!(
                            "Fuzzy repetition: {pattern_len}-tok pattern x3 ({mis_a}+{mis_b} \
                             mismatches), stopping at {} tokens (rollback declined: {reason:?})",
                            a.output_tokens.len()
                        );
                        a.guard_stop = Some("fuzzy_repetition");
                        a.finished = true;
                    }
                }
            }

            // The request-deadline check used to live here. It has moved to
            // `mod_helpers::enforce_request_deadlines`, which runs once per
            // scheduler iteration regardless of decode path — this function
            // is not called at all on the MTP/speculative path, so the
            // deadline was unenforced in the config of record.
        }
    }
}
