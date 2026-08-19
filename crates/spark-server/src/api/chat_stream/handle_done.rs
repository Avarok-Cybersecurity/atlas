// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Done { ... }` arm of the streaming `flat_map`
// closure (originally ~396 LoC).

use crate::ir::StreamDelta;
use crate::tool_parser;

use super::super::sanitizer::sanitize_content_chunk;
use super::super::stream_guards::flush_content_sanitizer;
use super::ctx::StreamCtx;
use super::state::StreamState;
use super::tool_handlers::{
    handle_complete_tool_call, handle_tool_call_args_fragment, handle_tool_call_delta,
    handle_tool_call_end, handle_tool_call_start,
};

type DeltaVec = Vec<StreamDelta>;

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_done(
    state: &mut StreamState,
    ctx: &StreamCtx,
    finish_reason: String,
    completion_tokens: usize,
    time_to_first_token_ms: f64,
    decode_time_ms: f64,
    reasoning_tokens: u32,
    cached_prompt_tokens: u32,
    accepted_prediction_tokens: usize,
) -> DeltaVec {
    let mut deltas: DeltaVec = Vec::new();

    // ── Stop-string hold-back flush ─────────────────────────────────
    // vLLM's `IncrementalDetokenizer` releases any bytes still in the
    // hold-back window when the stream finalises (see
    // `vllm/v1/engine/detokenizer.py`). Mirror that here: if a match
    // never triggered (`stop_string_triggered == false`) the tail
    // bytes are legitimate output and must be forwarded. Route them
    // through the active detector / sanitizer so the same envelope
    // and leak-marker rules apply — without this, a sub-stop-string
    // suffix that happens to contain a tool-call fragment would
    // bypass the live pipeline.
    if !ctx.stop_strings.is_empty()
        && !state.stop_string_triggered
        && state.stop_string_emitted_len < state.accumulated_content.len()
    {
        let tail = state.accumulated_content[state.stop_string_emitted_len..].to_string();
        state.stop_string_emitted_len = state.accumulated_content.len();
        if !tail.is_empty() {
            if let Some(det) = state.detector.as_mut() {
                let outputs = det.process(&tail);
                for output in outputs {
                    match output {
                        tool_parser::DetectorOutput::Content(text) => {
                            let sanitized = sanitize_content_chunk(
                                &text,
                                &mut state.tag_scan_buf,
                                &mut state.suppressing_param_leak,
                                &mut state.inside_envelope,
                                &ctx.leak_markers,
                            );
                            if !sanitized.is_empty() {
                                deltas.push(StreamDelta::Content {
                                    text: sanitized,
                                    token_ids: Vec::new(),
                                });
                            }
                        }
                        tool_parser::DetectorOutput::ToolCall(mut tc, tc_idx) => {
                            handle_complete_tool_call(state, ctx, &mut tc, tc_idx, &mut deltas);
                        }
                        tool_parser::DetectorOutput::ToolCallStart {
                            id: tc_id,
                            name,
                            idx,
                        } => {
                            handle_tool_call_start(state, ctx, tc_id, name, idx, &mut deltas);
                        }
                        tool_parser::DetectorOutput::ToolCallDelta { args, idx } => {
                            handle_tool_call_delta(state, ctx, args, idx, &mut deltas);
                        }
                        tool_parser::DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                            handle_tool_call_args_fragment(state, ctx, fragment, idx, &mut deltas);
                        }
                        tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                            handle_tool_call_end(state, ctx, idx);
                        }
                    }
                }
            } else {
                let sanitized = sanitize_content_chunk(
                    &tail,
                    &mut state.tag_scan_buf,
                    &mut state.suppressing_param_leak,
                    &mut state.inside_envelope,
                    &ctx.leak_markers,
                );
                if !sanitized.is_empty() {
                    if state.refusal_scan_buf.len() < 16_384 {
                        state.refusal_scan_buf.push_str(&sanitized);
                    }
                    deltas.push(StreamDelta::Content {
                        text: sanitized,
                        token_ids: Vec::new(),
                    });
                }
            }
        }
    }

    // ── Detector flush ──────────────────────────────────────────────
    if state.detector.is_some() {
        let outputs = {
            let det = state.detector.as_mut().expect("detector is Some");
            det.flush()
        };
        for output in outputs {
            match output {
                tool_parser::DetectorOutput::Content(text) => {
                    let sanitized = sanitize_content_chunk(
                        &text,
                        &mut state.tag_scan_buf,
                        &mut state.suppressing_param_leak,
                        &mut state.inside_envelope,
                        &ctx.leak_markers,
                    );
                    if !sanitized.is_empty() {
                        deltas.push(StreamDelta::Content {
                            text: sanitized,
                            token_ids: state.take_ids_if(ctx.req_return_token_ids),
                        });
                    }
                }
                tool_parser::DetectorOutput::ToolCall(mut tc, tc_idx) => {
                    handle_complete_tool_call(state, ctx, &mut tc, tc_idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallStart {
                    id: tc_id,
                    name,
                    idx,
                } => {
                    handle_tool_call_start(state, ctx, tc_id, name, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallDelta { args, idx } => {
                    handle_tool_call_delta(state, ctx, args, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                    handle_tool_call_args_fragment(state, ctx, fragment, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                    handle_tool_call_end(state, ctx, idx);
                }
            }
        }
    }

    // ── Sanitizer tail flush ────────────────────────────────────────
    let tail = flush_content_sanitizer(
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &ctx.leak_markers,
    );
    if !tail.is_empty() {
        if state.refusal_scan_buf.len() < 16_384 {
            state.refusal_scan_buf.push_str(&tail);
        }
        deltas.push(StreamDelta::Content {
            text: tail,
            token_ids: state.take_ids_if(ctx.req_return_token_ids),
        });
    }

    // ── Usage block (neutral IR; the wire encoder derives
    //    total_tokens and the details sub-objects from it) ───────────
    let tps = if decode_time_ms > 0.0 {
        completion_tokens.saturating_sub(1) as f64 / (decode_time_ms / 1000.0)
    } else {
        0.0
    };
    let usage = crate::ir::Usage {
        prompt_tokens: ctx.prompt_len,
        completion_tokens,
        cached_prompt_tokens: cached_prompt_tokens as usize,
        reasoning_tokens: reasoning_tokens as usize,
        accepted_prediction_tokens,
        time_to_first_token_ms,
        response_tokens_per_second: tps,
    };

    let fr = resolve_wire_finish_reason(
        &finish_reason,
        state.tool_loop_capped,
        state.detector.as_ref().is_some_and(|d| d.has_tool_calls()) || state.salvaged_tool_call,
        state.stop_string_matched,
        state.guard_stop,
    );

    // Refusal classification.
    let refusal_signal = if state.detector.as_ref().is_none_or(|d| !d.has_tool_calls()) {
        crate::refusal::detect(&state.refusal_scan_buf)
    } else {
        None
    };
    if let Some(ref r) = refusal_signal {
        deltas.push(StreamDelta::Refusal { text: r.clone() });
    }

    // Terminal delta: finish reason + usage. The `include_usage`
    // two-chunk framing (usage-only chunk before a usage-less finish
    // chunk) is the OpenAI encoder's decision
    // (`openai::delta_to_chunk_events`), not the core's. Residual
    // token ids — tokens whose decoded text was buffered/suppressed
    // and never rode a content delta — ride the Finish delta so
    // Σ token_ids == completion_tokens exactly.
    deltas.push(StreamDelta::Finish {
        reason: crate::ir::FinishReason::from(fr),
        usage,
        token_ids: state.take_ids_if(ctx.req_return_token_ids),
    });

    // Metrics. (REQUESTS_ACTIVE is released by the ActiveRequestGuard in
    // StreamCtx when the stream is dropped — not here, so a stream that ends
    // without a terminal event still decrements.)
    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(ctx.prompt_len as u64);
    // GENERATION_TOKENS_TOTAL is incremented PER TOKEN as the scheduler
    // emits it (scheduler/emit_step.rs, decode_logits_step.rs). Adding the
    // completion total here too would double-count — and counting only
    // here is what made the TUI throughput graph flat-then-spiky: the
    // counter did not move until a request finished.
    crate::metrics::TTFT_SECONDS
        .with_label_values(&[ctx.model.as_str()])
        .observe(time_to_first_token_ms / 1000.0);

    // Rate-limit true-up.
    if let Some(ref rctx) = ctx.req_ctx {
        let actual = (ctx.prompt_len + completion_tokens) as u64;
        let refund = rctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            ctx.state.rate_limiter.refund_tokens(&rctx.identity, refund);
        }
    }

    // --dump synthesized response entry. Diagnostics, not the stream:
    // the dump keeps the OpenAI wire-usage shape (same numbers the
    // encoder derives for the terminal chunk).
    if let (Some(seq), Some(dump)) = (ctx.dump_seq, ctx.state.dump_writer.as_ref()) {
        let has_tool_calls = state.detector.as_ref().is_some_and(|d| d.has_tool_calls());
        let usage_for_dump = crate::openai::Usage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.prompt_tokens + usage.completion_tokens,
            prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
                cached_tokens: usage.cached_prompt_tokens,
                audio_tokens: 0,
            }),
            completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
                reasoning_tokens: usage.reasoning_tokens,
                audio_tokens: 0,
                accepted_prediction_tokens: usage.accepted_prediction_tokens,
                rejected_prediction_tokens: 0,
            }),
            time_to_first_token_ms: usage.time_to_first_token_ms,
            response_tokens_per_second: usage.response_tokens_per_second,
        };
        let body = serde_json::json!({
            "id": ctx.id,
            "model": ctx.model,
            "object": "chat.completion.synthesized",
            "finish_reason": fr,
            "content": state.refusal_scan_buf,
            "has_tool_calls": has_tool_calls,
            "usage": usage_for_dump,
            "stop_string_triggered": state.stop_string_triggered,
            "loop_watchdog_triggered": state.loop_watchdog_triggered,
            "tool_loop_capped": state.tool_loop_capped,
            "guard_stop": state.guard_stop,
            "_note": "Synthesized from post-sanitizer accumulators; \
                      per-chunk capture is a follow-up.",
        });
        dump.dump_response("/v1/chat/completions", seq, &body, true);
    }

    deltas
}

/// Stream-layer overrides on top of the scheduler's finish reason.
/// Pure so the precedence is unit-testable. In order:
///
/// 1. `"timeout"` — the server-side request deadline cut this response
///    mid-flight. It outranks every override below: a truncated turn
///    that emitted a partial tool call would otherwise be reported
///    "tool_calls" and the client would run a half-parsed call as if it
///    were complete.
/// 2. `tool_loop_capped` → `"length"` — a tool-call loop guard (Bug-2
///    name-run cap, F11 within-dedup, F5 cross-flush dedup, or F44
///    perm-fail) forcibly ended the response. Signal "length" — OpenAI's
///    slot for a truncated response — so agent clients can break their
///    outer retry loop. Without this override the response otherwise
///    looks like a normal "tool_calls" completion (tool calls *were*
///    emitted) and agents (opencode, etc.) cheerfully run the tools and
///    ask the model to continue, perpetuating the loop one round at a
///    time. NOTE: this is a deliberate, pre-existing exception to the
///    scheduler-side invariant that "length" means "budget exhausted";
///    kept because it is a shipped contract for exactly this doom-loop
///    class.
/// 3. parsed/salvaged tool calls → `"tool_calls"`.
/// 4. `stop_string_matched` → `"stop"` — a client `stop` sequence ended
///    this response. OpenAI (and vLLM/SGLang/TGI/llama.cpp) report a
///    stop-sequence stop as "stop"; the scheduler can still say
///    "length" here when the cooperative cancel landed a step late
///    (tokens kept decoding with output suppressed and the budget ran
///    out first).
/// 5. a STREAM-side degeneration guard (`state.guard_stop`: the token
///    loop watchdog or the simhash semantic-loop trip) → `"length"`,
///    for the same reason as rule 2 and by the same invariant the
///    scheduler applies to its own guards.
///
///    ★ This rung is why the scheduler-side fix alone was not enough.
///    Those watchdogs live on the STREAM: they set `state.guard_stop`
///    and flip `cancel_flag`, but never touch `ActiveSeq::guard_stop`.
///    The scheduler therefore finalizes with no guard, falls to its
///    "early finalize, budget left" rule, and says `"stop"` — which
///    arrives here and passed through verbatim. Measured: after fixing
///    only the scheduler path, an episode still died at 4 turns having
///    written one file, its last turn a mid-word repetition splice
///    labelled `stop`. The stream knew it was a guard cut and simply
///    never said so.
/// 6. otherwise the scheduler's reason passes through verbatim.
fn resolve_wire_finish_reason<'a>(
    scheduler_reason: &'a str,
    tool_loop_capped: bool,
    has_tool_calls: bool,
    stop_string_matched: bool,
    stream_guard_stop: Option<&'static str>,
) -> &'a str {
    if scheduler_reason == crate::ir::FINISH_REASON_TIMEOUT {
        scheduler_reason
    } else if tool_loop_capped {
        "length"
    } else if has_tool_calls {
        "tool_calls"
    } else if stop_string_matched {
        "stop"
    } else if stream_guard_stop.is_some() {
        // A degeneration cut is a truncation: the model was mid-output.
        // Ordered AFTER tool_calls/stop_string so a guard that trips on
        // the same step a real tool call or client stop sequence landed
        // still reports what actually happened.
        "length"
    } else {
        scheduler_reason
    }
}

#[cfg(test)]
mod wire_finish_reason_tests {
    use super::resolve_wire_finish_reason;
    use crate::ir::FINISH_REASON_TIMEOUT;

    #[test]
    fn stop_string_match_is_stop_not_length() {
        // The second instance of the "length is a lie" bug: a matched
        // client stop sequence previously fell through to the
        // scheduler's reason, which reads "length" whenever the seq
        // burned to the budget with its output suppressed.
        assert_eq!(
            resolve_wire_finish_reason("length", false, false, true, None),
            "stop"
        );
        // With no match, the scheduler's reason passes through.
        assert_eq!(
            resolve_wire_finish_reason("length", false, false, false, None),
            "length"
        );
        assert_eq!(
            resolve_wire_finish_reason("stop", false, false, false, None),
            "stop"
        );
    }

    #[test]
    fn stream_side_guard_cut_reports_length() {
        // POSITIVE. The token-loop / simhash watchdogs live on the STREAM:
        // they set `state.guard_stop` and flip `cancel_flag`, but never
        // touch the scheduler's `ActiveSeq::guard_stop`. The scheduler
        // therefore finalizes with no guard and says "stop". Without this
        // rung that "stop" reached the client, and the agentic harness
        // read a mid-repetition truncation as a finished turn.
        //
        // ★ Fixing only the scheduler side left this hole: an episode
        // still died at 4 turns having written one file, its last turn a
        // mid-word splice labelled `stop`.
        for guard in ["simhash_semantic_loop", "token_loop_watchdog"] {
            assert_eq!(
                resolve_wire_finish_reason("stop", false, false, false, Some(guard)),
                "length",
                "guard={guard}"
            );
        }
    }

    #[test]
    fn stream_guard_does_not_outrank_what_actually_happened() {
        // NEGATIVE. The guard rung sits BELOW tool_calls and the
        // stop-string match, so a watchdog tripping on the same step a
        // real tool call or a client stop sequence landed still reports
        // the true event rather than claiming truncation.
        assert_eq!(
            resolve_wire_finish_reason("stop", false, true, false, Some("token_loop_watchdog")),
            "tool_calls"
        );
        assert_eq!(
            resolve_wire_finish_reason("stop", false, false, true, Some("token_loop_watchdog")),
            "stop"
        );
        // And with no stream guard the scheduler's reason is untouched.
        assert_eq!(
            resolve_wire_finish_reason("stop", false, false, false, None),
            "stop"
        );
    }

    #[test]
    fn timeout_and_tool_overrides_keep_their_rank() {
        // Shipped contracts, unchanged by this wave.
        assert_eq!(
            resolve_wire_finish_reason(FINISH_REASON_TIMEOUT, true, true, true, None),
            FINISH_REASON_TIMEOUT
        );
        assert_eq!(
            resolve_wire_finish_reason("stop", true, true, true, None),
            "length",
            "tool-loop cap outranks tool_calls and the stop-string match"
        );
        assert_eq!(
            resolve_wire_finish_reason("stop", false, true, true, None),
            "tool_calls",
            "parsed tool calls outrank the stop-string override"
        );
    }
}
