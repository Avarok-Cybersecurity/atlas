// SPDX-License-Identifier: AGPL-3.0-only

//! emit_token + compile_grammar_state + StartPrefillResult enum.

use super::*;

// `compile_grammar_state` + `StartPrefillResult` and the tool-body /
// parameter-body state machine (`update_tool_param_state`) were moved to
// the `grammar` / `tool_param_state` submodules to keep this file ≤500
// LoC. Re-exported so existing `emit_step::*` call sites are unaffected.
mod grammar;
mod tool_param_state;

pub use grammar::{StartPrefillResult, compile_grammar_state};
pub use tool_param_state::update_tool_param_state;

/// Emit a token for an active sequence (stream + bookkeeping).
///
/// Per OpenAI spec, stop/EOS tokens are NOT streamed to the client —
/// the returned text must not contain the stop sequence. The token is
/// still recorded in output_tokens for accurate token counting.
///
/// When `logprobs` is Some, the logprobs data is accumulated for blocking
/// responses and sent via `StreamEvent::TokenWithLogprobs` for streaming.
pub fn emit_token(a: &mut ActiveSeq, tok: u32, logprobs: Option<crate::api::TokenLogprobs>) {
    // Cooperative cancellation from the streaming pipeline. The
    // stream-side loop guards (Bug-2 name-run cap, F11 within-dedup,
    // F44 perm-fail, loop-watchdog) flip this flag when they decide
    // the response should end. Treat it like an EOS: finalise now so
    // `handle_done` runs with the proper `tool_loop_capped` /
    // `finish_reason="length"` machinery, instead of letting the
    // model keep emitting tokens that just get suppressed.
    if let Some(ref f) = a.cancel_flag
        && f.load(std::sync::atomic::Ordering::Acquire)
    {
        a.finished = true;
        return;
    }

    // ChatML role-boundary HARD stop (`<|im_start|>`).
    //
    // Handled BEFORE grammar advance / EOS suppression: if the model
    // hallucinated a `<|im_start|>` mid-turn, we must end the turn regardless
    // of grammar / require_tool_call / min_tokens. The regular EOS path at
    // line ~3020 honors `suppress_eos`, which is true while a tool-call
    // grammar is active — so if we fell through to it, the tokenizer would
    // strip `<|im_start|>` (special-token) but the following role literal
    // (`user` / `assistant` — regular tokens) would stream to the client,
    // poisoning its context and causing the observed multi-turn drift /
    // "file was corrupted" hallucinations in opencode.
    if let Some(ims) = im_start_hard_stop()
        && tok == ims
    {
        // Push the hard-stop token to output_tokens so lifecycle.rs reports
        // `finish_reason="stop"` (because `<|im_start|>` is registered in
        // `eos_tokens` at startup — see tokenizer_runtime.rs::im_start_id).
        // Without this push, `last_tok = output_tokens.last()` is the prior
        // content token, lifecycle's `is_eos` check fails, and the response
        // is mis-reported as `finish_reason="length"` (Bug 3 from OpenClaw
        // 2026-05-08 session: "Done: 13 tokens (length) despite max_tokens=
        // 8192" — clients then misinterpret the truncation as a real
        // length-limit hit and either retry or surface a wrong error).
        // The streamed-text path strips stop tokens server-side, so the
        // client never sees the literal `<|im_start|>` bytes.
        a.output_tokens.push(tok);
        a.finished = true;
        tracing::debug!(
            "<|im_start|> hard-stop fired (id={ims}); ending turn before grammar/suppress_eos"
        );
        return;
    }

    // Spontaneous <think>: model generates <think> even when thinking was not
    // requested. Enter thinking mode so EOS is suppressed and thinking content
    // is stripped. This handles MTP bootstrap/verify paths.
    if !a.inside_thinking && a.think_start_token == Some(tok) {
        a.inside_thinking = true;
        a.think_ended = false;
        a.think_skip_count = 0;
        a.thinking_budget = Some(a.spontaneous_think_budget);
        tracing::debug!("Spontaneous <think> detected in emit_token, entering thinking mode");
        return; // don't emit <think> as content
    }

    // Silently skip </think> tokens outside thinking mode (same as process_decode_logits).
    if !a.inside_thinking && a.think_end_token == Some(tok) {
        a.think_skip_count += 1;
        if a.think_skip_count >= 50 {
            a.finished = true;
        }
        return;
    }

    // Track <tool_call> token: once seen, legacy tool call requirement is satisfied.
    // Guard with !inside_thinking — tool calls inside thinking are spurious.
    if a.require_tool_call && a.tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.require_tool_call = false;
        a.tool_call_opened = true;
    }

    // Tool-body / parameter-body state machine.
    //
    // SM1 (2026-05-26): extracted from inline-in-emit_token to the free
    // function `update_tool_param_state` so the regular non-MTP decode
    // path (`decode_logits_step.rs`) can call it too. Previously the
    // state was ONLY updated when `emit_token` ran — which happens from
    // spec/verify paths but NOT from `process_decode_logits`. With
    // mtp=false (Qwen3.6 baseline), the state machine never ran and
    // every dependent gate (A1 rep-penalty toggle, B1 margin detector)
    // was silently dead code. (The pos-0 close-tag/AM1 logit-bias that
    // also depended on this state was removed 2026-06-03; the state is
    // still required for A1/B1 and the adadec_diag dump.)
    update_tool_param_state(a, tok);

    // F2 mirror (Iter 46, 2026-06-02): reset the inter-tool prose budget when
    // a tool call opens on the MTP/emit path — parity with the non-MTP reset
    // in `decode_logits_step.rs` (the `tool_call_start_token` branch). Without
    // this the budget would accrue across the whole response and the MTP-path
    // budget watchdog (added below) would false-fire after the first
    // `max_inter_tool_prose` content tokens even across legitimate multi-tool
    // turns. Keyed identically: tool-call open, not inside `<think>`.
    if a.tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.prose_tokens_since_last_tool = 0;
    }

    // Advance grammar state with the emitted token — skip while the
    // sequence is inside `<think>`…`</think>` so the matcher only
    // sees the final-output token stream.
    let mut disengage_grammar = false;
    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
    {
        let advanced = gs.accept_token(tok);
        if !advanced {
            // Grammar/model disagreement (BUG#2 class: e.g. a merged BPE token
            // like `><` or a `</X` content run the qwen3_coder value rule
            // forbids, often surfaced via an under-masked MTP draft). The token
            // is already a legitimate model emission; the matcher is now
            // desynced. Previously we set `a.finished = true` here — a
            // CATASTROPHIC cliff that lost the ENTIRE agentic turn on a single
            // refused token (root cause of the opencode webserver_ok gap).
            // Instead, DISENGAGE the grammar for the remainder of this response
            // and continue decoding UNCONSTRAINED — exactly what vLLM (the 10/10
            // reference) does by parsing tools post-hoc. Atlas's server-side
            // tool parser still extracts tool calls from the emitted text, so
            // the structural guarantee is gracefully traded for turn survival.
            tracing::warn!(
                tok,
                output_len = a.output_tokens.len(),
                "gs.accept_token returned false — grammar/model disagreement; disengaging grammar for the remainder of this response (free decode + post-hoc tool parse) instead of aborting the turn."
            );
            disengage_grammar = true;
        }
    }
    if disengage_grammar {
        // Drop the matcher: subsequent decode steps see `grammar_state == None`
        // and decode unconstrained. Set after the `ref mut gs` borrow ends.
        a.grammar_state = None;
    }

    // Accumulate logprobs data for blocking responses.
    if let Some(lp) = logprobs {
        a.logprobs_data.push(lp);
    }

    a.output_tokens.push(tok);

    // Thinking tokens are "free" (don't decrement remaining).
    // Detect </think> transition. Track thinking token count for budget enforcement.
    if a.inside_thinking {
        if a.think_end_token == Some(tok) {
            a.inside_thinking = false;
            a.force_end_thinking = false;
            a.sentence_defer_count = 0;
            a.think_ended = true;
            // One-shot for the next decode step: pin to
            // tool_call_start_token if require_tool_call (Change 3b).
            a.think_just_ended = true;
            tracing::info!(
                "Thinking ended after {} tokens (budget={:?})",
                a.thinking_tokens,
                a.thinking_budget,
            );
        } else {
            a.thinking_tokens += 1;
            if let Some(budget) = a.thinking_budget
                && a.thinking_tokens >= budget
                && !a.force_end_thinking
            {
                a.force_end_thinking = true;
                a.sentence_defer_count = 0;
                tracing::info!(
                    "Thinking budget exhausted ({budget} tokens), arming </think>; \
                     deferring to next sentence boundary"
                );
            }
        }
    } else {
        a.remaining -= 1;
        // Clear think_just_ended one-shot now that we've consumed the
        // token after </think>.
        a.think_just_ended = false;
        // Content-phase loop watchdog. Mirrored from
        // `handle_content_token` (decode_logits_content.rs) because
        // that handler is only invoked on the non-MTP decode path
        // (`process_decode_logits`). MTP speculative decode
        // (`verify_k2_step`) reaches every token through this
        // `emit_token` instead — without this mirror, the
        // content-loop watchdog never fires while MTP is enabled, and
        // the model can burn the full `max_tokens` budget on a
        // period-N attractor. Observed live 2026-05-24 on
        // opencode-hotfix2b.jsonl seq=13: 8193 content tokens of
        // `[29, 198, 510, 15704, …]` period-4 loop (the
        // `parameter>\n` attractor) with no watchdog fire,
        // finish=length.
        //
        // 2026-05-24 sweep #3: Re-introduced the `!a.inside_tool_body`
        // gate (mirrors the handle_content_token policy). The previous
        // inside-body false-positives turned out to be triggered by a
        // separate MTP-pipeline gap (see bench/hotfix3-debug/
        // SYNTHESIS.md). With the pipeline correctly applied to MTP
        // verify, JSON structural repetition is bounded by the
        // grammar's terminal state. The `parameter>\n` real-loop case
        // is still caught one tick AFTER the model exits the tool
        // body — its emission outside the body forms a tight period-N
        // tail.
        //
        // Skip rollback here — `emit_token` doesn't take `&dyn Model`
        // (the SSM rewind requires it) and plumbing it through every
        // call site would balloon the diff. Instead set `a.finished`
        // and let the lifecycle close the response. The non-MTP path
        // retains rollback via `handle_content_token`.
        use crate::scheduler::helpers::{
            CONTENT_LOOP_CHECK_STRIDE, CONTENT_LOOP_MIN_TOKENS, CONTENT_LOOP_PERIOD_MAX,
            CONTENT_LOOP_PERIOD_MIN, detect_content_token_loop_normalized_with,
            detect_content_token_loop_with, disable_watchdogs, enable_loop_watchdog,
            numeric_token_mask,
        };
        a.content_tokens = a.content_tokens.saturating_add(1);
        // F1 (2026-06-02): unconditional per-generation post-think content
        // cap. Fires regardless of `inside_tool_body` so it bounds the
        // runaway no matter which heuristic state machine desynced (RC1/
        // RC2/RC3). Gated on `grammar_state.is_some()` ⇒ only tool-active
        // requests are ever capped (plain chat attaches no grammar and is
        // never truncated). Default 100_000 (`MAX_POST_THINK_CONTENT_TOKENS`)
        // = no-op; Qwen3.6-35B-A3B-FP8 sets 1536 in MODEL.toml.
        if !disable_watchdogs()
            && a.grammar_state.is_some()
            && a.content_tokens > watchdog_params().max_post_think_content_tokens
        {
            tracing::warn!(
                content_tokens = a.content_tokens,
                max = watchdog_params().max_post_think_content_tokens,
                "post-think content cap exceeded in MTP/emit path; ending response (tool-active request would otherwise burn to max_tokens)"
            );
            a.finished = true;
        }
        if !disable_watchdogs()
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
            tracing::warn!(
                content_tokens = a.content_tokens,
                output_len = a.output_tokens.len(),
                "Content-loop watchdog fired in MTP/emit path (period-{}…{} repeat); ending response",
                CONTENT_LOOP_PERIOD_MIN,
                CONTENT_LOOP_PERIOD_MAX,
            );
            a.finished = true;
        }

        // F2 mirror (Iter 46, 2026-06-02): inter-tool PROSE-BUDGET watchdog on
        // the MTP/emit path. The 2026-05-24 mirror above copied only the
        // content-LOOP guard; the prose-budget guard (decode_logits_content.rs)
        // stayed non-MTP-only — so with `--num-drafts ≥ 1` (MTP/verify path),
        // a turn that wanders WITHOUT producing a parseable tool call had NO
        // bound and burned the whole `max_tokens` budget (~270s at 30 tok/s),
        // starving the agent of turns. This was the dominant opencode
        // `webserver_ok` 360s-timeout cause: at deep context the model flips
        // its tool opener to Anthropic-XML `<invoke name=…>`, which never
        // matches the qwen3_coder trigger, so the trigger-gated grammar stays
        // dormant and the wander is not a tight period-≤64 loop the content
        // watchdog catches. Same gates as the non-MTP block: free-text only
        // (`!inside_tool_body`) and grammar attached (`grammar_state.is_some()`
        // ⇒ a tool request, never plain chat — so a long chat answer is not
        // truncated). No rollback here: `emit_token` has no `&dyn Model` (the
        // SSM rewind needs it), so we hard-stop exactly like the content-loop
        // mirror; the sanitizer + post-hoc tool parser salvage what was emitted.
        // F4 (2026-06-02): gate on the sticky `tool_request` flag (set at
        // prefill, survives a graceful grammar disengage) instead of
        // `grammar_state.is_some()` — otherwise a disengaged tool turn on
        // the MTP path wanders to `max_tokens` with the budget inert.
        if !disable_watchdogs() && !a.inside_tool_body && a.tool_request {
            a.prose_tokens_since_last_tool = a.prose_tokens_since_last_tool.saturating_add(1);
            let max_prose = watchdog_params().max_inter_tool_prose;
            if a.prose_tokens_since_last_tool > max_prose {
                tracing::warn!(
                    prose_tokens = a.prose_tokens_since_last_tool,
                    max = max_prose,
                    output_len = a.output_tokens.len(),
                    "Inter-tool prose budget exhausted in MTP/emit path; ending response \
                     (no tool call after budget — would otherwise burn to max_tokens)."
                );
                a.finished = true;
            }
        }
    }

    // EOS handling: grammar-based, legacy, or min_tokens suppression.
    let grammar_suppresses_eos = a
        .grammar_state
        .as_ref()
        .is_some_and(|gs| !gs.is_terminated());
    let legacy_suppresses_eos = a.require_tool_call;
    let min_tokens_suppresses = a.output_tokens.len() < a.min_tokens;
    let suppress_eos = grammar_suppresses_eos || legacy_suppresses_eos || min_tokens_suppresses;

    if a.eos_tokens.contains(&tok) && !suppress_eos {
        a.finished = true;
        return;
    }
    if a.eos_tokens.contains(&tok) && suppress_eos {
        // EOS suppressed: grammar not terminated, legacy tool call not yet seen,
        // or min_tokens not reached. Don't stop — let the model continue generating.
        return;
    }
    // OPENCODE FIX: see process_decode_logits — same gate. Suppress streaming
    // of spontaneous-thinking content so it doesn't pollute opencode's history.
    let suppress_stream = a.inside_thinking && !a.enable_thinking;
    if let ResponseSink::Streaming(ref tx) = a.sink
        && !suppress_stream
    {
        let event = if let Some(lp) = a.logprobs_data.last().cloned() {
            StreamEvent::TokenWithLogprobs(tok, lp)
        } else {
            StreamEvent::Token(tok)
        };
        // Discriminate transient backpressure (channel full) from a real
        // consumer-drop (channel closed). The previous `try_send().is_err()`
        // collapsed the two and silently terminated the seq with
        // `finish_reason="length"` whenever the SSE consumer momentarily
        // stalled — surfaced as "request stops half-way" in Open WebUI.
        match tx.try_send(event) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                tracing::debug!("Streaming receiver dropped, finishing seq");
                a.finished = true;
                return;
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                if let Err(e) = tx.blocking_send(event) {
                    tracing::error!("Streaming send failed during backpressure: {e}");
                    a.finished = true;
                    return;
                }
            }
        }
    }
    if a.remaining == 0 {
        tracing::info!(
            "emit_token: remaining=0, output_tokens={}, thinking_tokens={}",
            a.output_tokens.len(),
            a.thinking_tokens
        );
        a.finished = true;
    }
}

// F72 (byte-level partial-trigger anchor) was removed in F73 / fix42.
// The sampler-side anchor hung the server in production; the broken
// envelope is now recovered at the streaming-sanitizer + parser
// layer. xgrammar's non-anchored TagDispatch limitation is pinned
// for documentation by
// `grammar.rs::tests::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`.
