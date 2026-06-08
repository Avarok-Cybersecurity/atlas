// SPDX-License-Identifier: AGPL-3.0-only

//! Content-phase token handling for `process_decode_logits` — the
//! non-thinking branch of the per-token decode loop. Extracted from
//! `decode_logits_step.rs` to keep that file ≤500 LoC.
//!
//! Runs once per sampled token while the sequence is *outside*
//! `<think>…</think>`: budget bookkeeping plus the two content-phase
//! degeneration watchdogs (content-loop, inter-tool prose). Both
//! watchdogs were converted in Phase-C to roll back to the last
//! well-formed boundary and re-steer (`rollback_to_boundary`) instead
//! of hard-stopping the response.

use super::*;

/// Slow-path diagnostic: when `detect_content_token_loop` returns
/// `true`, re-scan to report which `(period, repeats)` matched. Used
/// only on the watchdog-fired branch — runs once per fire, never on
/// the steady-state stride check. Returns `None` if no period matched
/// (caller should not have invoked this).
/// 2026-05-24 v3: vLLM-style anchored detector. Returns the smallest
/// matched period for diagnostic logging when the watchdog fires.
/// "Count" is reported as the configured min_repeats since vLLM's
/// algorithm doesn't search past the minimum — once we've verified
/// `min_repeats` consecutive end-anchored windows, we stop.
fn describe_content_token_loop(tokens: &[u32]) -> Option<(usize, usize)> {
    let n = tokens.len();
    if n < CONTENT_LOOP_MIN_TOKENS as usize {
        return None;
    }
    if CONTENT_LOOP_MIN_REPEATS < 2 {
        return None;
    }
    for pattern_len in CONTENT_LOOP_PERIOD_MIN..=CONTENT_LOOP_PERIOD_MAX {
        if pattern_len * CONTENT_LOOP_MIN_REPEATS > n {
            return None;
        }
        // Inline anchored check (mirrors helpers::has_repeating_pattern_anchored
        // which is module-private to scheduler::helpers).
        let mut all_match = true;
        'outer: for offset_in_window in 1..=pattern_len {
            let target = tokens[n - offset_in_window];
            for m in 1..CONTENT_LOOP_MIN_REPEATS {
                let idx = n - (pattern_len * m + offset_in_window);
                if tokens[idx] != target {
                    all_match = false;
                    break 'outer;
                }
            }
        }
        if all_match {
            return Some((pattern_len, CONTENT_LOOP_MIN_REPEATS));
        }
    }
    None
}

/// Handle one sampled token that lands in the content phase (model is
/// not inside `<think>`). Mutates `a` in place: decrements the
/// generation budget, advances content counters, and runs the
/// content-loop + inter-tool-prose watchdogs.
///
/// `model` is needed by the Phase-C boundary rollback so it can restore
/// SSM recurrent state on hybrid models (see
/// [`super::rollback::rollback_to_boundary`]).
pub fn handle_content_token(a: &mut ActiveSeq, model: &dyn Model) {
    a.remaining -= 1;
    a.content_started = true;
    a.content_tokens = a.content_tokens.saturating_add(1);
    // F1 (2026-06-02): unconditional per-generation post-think content
    // cap — the non-MTP twin of the guard in `emit_step::emit_token`.
    // Fires regardless of `inside_tool_body` so it bounds the runaway no
    // matter which heuristic state machine desynced; the
    // `grammar_state.is_some()` gate ensures only tool-active requests are
    // ever capped (plain chat attaches no grammar). Default 100_000
    // (`MAX_POST_THINK_CONTENT_TOKENS`) = no-op; Qwen3.6-35B-A3B-FP8 sets
    // 1536 in MODEL.toml.
    if !crate::scheduler::helpers::disable_watchdogs()
        && a.grammar_state.is_some()
        && a.content_tokens > watchdog_params().max_post_think_content_tokens
    {
        tracing::warn!(
            content_tokens = a.content_tokens,
            max = watchdog_params().max_post_think_content_tokens,
            "post-think content cap exceeded in non-MTP decode path; ending response (tool-active request would otherwise burn to max_tokens)"
        );
        a.finished = true;
    }
    // think_just_ended is a one-shot: it was set when the prior
    // token was `</think>`; clear it now that we've emitted the
    // first content token (which Change 3b's mask pinned to
    // tool_call_start_token when require_tool_call was set).
    a.think_just_ended = false;

    // Content-phase loop watchdog (2026-04-26 Claude Code
    // degeneration fix). Catches the agentic-failure mode
    // where the model emits the same sentence over and over
    // ("I see I've been creating Cargo.toml files but the
    // user hasn't given me a task. Let me wait for their
    // instructions." × 12). LZ penalty at strength 0.2 nudges
    // but cannot break the attractor once established.
    //
    // 2026-05-23 sweep: REMOVED the `a.grammar_state.is_none()`
    // gate. opencode's `tool_choice="auto"` activates the grammar
    // FSM for the OUTER envelope (free prose between tool calls),
    // not just the tool body.
    //
    // 2026-05-24 sweep #3: Re-introduced the `!a.inside_tool_body`
    // gate. The previous removal was based on a real-loop observation
    // (`parameter>\nparameter>\n...` period-2 attractor) but turned
    // out to be triggered by a separate MTP-pipeline gap (see
    // bench/hotfix3-debug/SYNTHESIS.md): the MTP verify path skipped
    // the entire pre-sample LogitsProcessor pipeline, so the
    // tool-body false-positives the inside-body fires produced
    // appeared as massive regressions. With the pipeline correctly
    // applied to MTP verify, JSON structural repetition is bounded by
    // the grammar's terminal state — xgrammar already guarantees
    // structural termination inside the tool body, so the content-
    // loop watchdog should not fire there. The `parameter>\n`
    // real-loop case is still caught one tick AFTER the model exits
    // the tool body: its emission outside the body forms a tight
    // period-N tail that the outside-body watchdog will detect.
    if !crate::scheduler::helpers::disable_watchdogs()
        && enable_loop_watchdog()
        && !a.inside_tool_body
        && a.content_tokens >= CONTENT_LOOP_MIN_TOKENS
        && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
        && (detect_content_token_loop_with(&a.output_tokens, a.repetition_detection)
            || numeric_token_mask().as_deref().is_some_and(|m| {
                detect_content_token_loop_normalized_with(
                    &a.output_tokens,
                    m,
                    a.repetition_detection,
                )
            }))
    {
        // 2026-05-23 sweep: re-scan to report the matched
        // `(period, repeats)`. Slow path — only runs on fire, not
        // on the every-16-token stride check. Cost: O(period_max ×
        // scan_window) once per watchdog fire. Logging this makes
        // future occurrences self-debuggable: a period-3 repeat is
        // an interjection collapse, a period-30+ is a sentence loop.
        let pattern = describe_content_token_loop(&a.output_tokens);
        let (period, repeats) = pattern.unwrap_or((0, 0));
        // Phase-C: roll back to the last well-formed boundary
        // and re-steer instead of killing the response. `min_keep`
        // = CONTENT_LOOP_PERIOD_MAX so the rollback always escapes
        // the detected period. Falls back to the legacy hard stop
        // when disabled / capped / no boundary found.
        match rollback_to_boundary(a, CONTENT_LOOP_PERIOD_MAX, model) {
            RollbackOutcome::RolledBack { dropped } => {
                tracing::warn!(
                    content_tokens = a.content_tokens,
                    dropped,
                    rollback = a.rollback_count,
                    matched_period = period,
                    matched_repeats = repeats,
                    "Content-loop watchdog fired (period-{}…{} repeat); rolled back to boundary, re-steering",
                    CONTENT_LOOP_PERIOD_MIN,
                    CONTENT_LOOP_PERIOD_MAX,
                );
            }
            RollbackOutcome::Fallback(reason) => {
                tracing::warn!(
                    content_tokens = a.content_tokens,
                    output_len = a.output_tokens.len(),
                    matched_period = period,
                    matched_repeats = repeats,
                    ?reason,
                    "Content-loop watchdog fired (period-{}…{} repeat); ending response early (rollback declined).",
                    CONTENT_LOOP_PERIOD_MIN,
                    CONTENT_LOOP_PERIOD_MAX,
                );
                a.finished = true;
            }
        }
    }

    // F2 (2026-04-26): bounded inter-tool prose budget.
    // Counts only free-text tokens (not inside tool body,
    // not inside grammar-constrained emission). When the
    // budget trips we recover the turn so the next attempt can
    // re-plan, instead of letting the model emit
    // prose↔tool↔prose↔tool forever (the `tool_choice="auto"`
    // grammar never self-terminates — see grammar.rs:461-462).
    // F4 (2026-06-02): gate on the sticky `tool_request` flag instead of
    // `grammar_state.is_some()`. The grammar can gracefully DISENGAGE
    // mid-response (`emit_step` drops `grammar_state` to salvage a turn);
    // the prior gate then went inert and the prose budget never fired,
    // letting a disengaged tool turn wander to `max_tokens`.
    // `tool_request` is set at prefill and survives disengage.
    if !crate::scheduler::helpers::disable_watchdogs()
        && !a.inside_tool_body
        && a.tool_request
    {
        a.prose_tokens_since_last_tool = a.prose_tokens_since_last_tool.saturating_add(1);
        let max_prose = watchdog_params().max_inter_tool_prose;
        if a.prose_tokens_since_last_tool > max_prose {
            // Phase-C: roll back to the last boundary and
            // re-steer so the model can re-attempt the tool
            // call cleanly, instead of killing the turn
            // mid-plan. `rollback_to_boundary` rewinds the
            // grammar FSM in lock-step (step 5), so the
            // constrained tool-call decoder stays valid.
            // `min_keep` = CONTENT_LOOP_PERIOD_MAX drops a full
            // run-on sentence of stalled prose.
            match rollback_to_boundary(a, CONTENT_LOOP_PERIOD_MAX, model) {
                RollbackOutcome::RolledBack { dropped } => {
                    tracing::warn!(
                        max = max_prose,
                        dropped,
                        rollback = a.rollback_count,
                        "Inter-tool prose budget exhausted; rolled back to boundary, re-steering"
                    );
                }
                RollbackOutcome::Fallback(reason) => {
                    tracing::warn!(
                        prose_tokens = a.prose_tokens_since_last_tool,
                        max = max_prose,
                        ?reason,
                        "Inter-tool prose budget exhausted, ending response (rollback declined)"
                    );
                    a.finished = true;
                }
            }
        }
    }
}
