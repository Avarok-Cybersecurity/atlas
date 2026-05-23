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
    // Disabled inside grammar/tool-body because structured JSON
    // repeats are legitimate.
    if enable_loop_watchdog()
        && a.grammar_state.is_none()
        && !a.inside_tool_body
        && a.content_tokens >= CONTENT_LOOP_MIN_TOKENS
        && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
        && (detect_content_token_loop(&a.output_tokens)
            || numeric_token_mask()
                .as_deref()
                .is_some_and(|m| detect_content_token_loop_normalized(&a.output_tokens, m)))
    {
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
                    "Content-loop watchdog fired (period-{}…{} repeat); rolled back to boundary, re-steering",
                    CONTENT_LOOP_PERIOD_MIN,
                    CONTENT_LOOP_PERIOD_MAX,
                );
            }
            RollbackOutcome::Fallback(reason) => {
                tracing::warn!(
                    content_tokens = a.content_tokens,
                    output_len = a.output_tokens.len(),
                    ?reason,
                    "Content-loop watchdog fired (period-{}…{} repeat); ending response early (rollback declined)",
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
    if !a.inside_tool_body && a.grammar_state.is_some() {
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
