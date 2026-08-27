// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Token` / `StreamEvent::TokenWithLogprobs` arm of the
// streaming `flat_map` closure (originally ~672 LoC at the top of the
// `chat_stream::chat_completions_stream` body).
//
// Returns the neutral stream deltas produced for this single token.
// The caller encodes them into wire SSE events at the surface seam
// (`openai::delta_to_chunk_events`).

use crate::ir::StreamDelta;
use crate::tool_parser;

use super::super::sanitizer::sanitize_content_chunk;
use super::super::stream_guards::{bump_f12_tool_call_count, check_loop_watchdog};
use super::ctx::StreamCtx;
use super::state::StreamState;

/// `ATLAS_SIMHASH_LOOP=0` disables the F4 SimHash semantic-loop guard.
/// Default ON — see the comment at the check site for why an operator would
/// turn it off (one-strike near-duplicate detection kills streams over
/// legitimately repetitive structured output).
fn simhash_loop_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_SIMHASH_LOOP").as_deref() != Ok("0"))
}
use super::strip::{
    maybe_log_decode_trace, strip_all_preserving_boundary, strip_preserving_boundary,
};
use super::tool_handlers::{
    handle_complete_tool_call, handle_tool_call_args_fragment, handle_tool_call_delta,
    handle_tool_call_end, handle_tool_call_start,
};

type DeltaVec = Vec<StreamDelta>;

/// Maximum consecutive tokens the stream may spend with
/// `state.suppressing_param_leak == true` (sanitizer holding content
/// because of an orphan `<parameter=` / `<tool_call>` opener without
/// a matching close). When the model degenerates into a doom-loop of
/// partial-envelope leakage — observed 2026-05-24 on
/// opencode-hotfix.jsonl seq=10: 8192 tokens emitted after Atlas
/// rejected a `write({})` call, all suppressed by the sanitizer, no
/// content-loop watchdog fire (the period exceeded 64) — this
/// threshold ends the stream cleanly instead of burning to
/// `max_tokens=8192`. 256 tokens is enough headroom for legitimately
/// long tool-call bodies that take many tokens to close (long
/// `content` field on a `write` call) while bounding worst-case
/// wasted decode at ~10s @ 30 tok/s.
const MAX_SUPPRESS_STREAK_TOKENS: u32 = 256;

/// Drop a delta that is nothing but a bare role literal (`user` /
/// `assistant` / `tool`) — a Qwen3.5/3.6 hallucination leak, companion to
/// the scheduler-side `<|im_start|>` hard-stop.
///
/// `inside_tool_call` MUST be true whenever the streaming tool-call
/// detector is mid-body (between `<tool_call>` and its close). Issue #222:
/// for a `tool_*`-prefixed tool name (`tool_search`, `tool_call`,
/// `tool_describe`) the byte-level BPE tokenizer emits a standalone `tool`
/// token as the leading fragment of the NAME. Without the guard this strip
/// clears that fragment, and the detector reassembles the name from the
/// remainder (`_search`), truncating the streamed tool-call name by exactly
/// `len("tool") == 4` chars. Non-streaming was unaffected because it parses
/// the whole buffer at once. The guard confines the strip to genuine
/// content leaks (no tool call in flight).
pub(super) fn strip_bare_role_literal(delta: &mut String, inside_tool_call: bool) {
    if inside_tool_call {
        return;
    }
    let trimmed = delta.trim();
    if delta.len() < 20 && matches!(trimmed, "user" | "assistant" | "tool") {
        tracing::debug!("role-literal strip: dropped bare '{trimmed}' delta");
        delta.clear();
    }
}

/// Process one token. Returns the stream deltas to forward to the
/// client (empty `Vec` is valid).
///
/// Thin wrapper around [`handle_token_inner`] that runs the
/// orphan-suppression streak watchdog after every token regardless
/// of which early-return branch fired in the body. The watchdog
/// can't live inside `handle_token_inner` because that function has
/// many early returns (one per emission path) — putting the check
/// at the end of the body would only fire when the natural fall-
/// through is taken, leaving the doom-loop case (long suppressed
/// stream of orphan `<tool_call>` openers) uncaught.
pub(super) fn handle_token(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> DeltaVec {
    let result = handle_token_inner(state, ctx, tok);
    // TTFT forensics companion to the stable-delta line inside the body:
    // brackets everything after it (sanitizer, detector, watchdogs) so a
    // first-delta latency bisects to inside-handle_token vs downstream.
    if !state.first_result_logged && !result.is_empty() {
        state.first_result_logged = true;
        tracing::debug!(
            "stream: first delta batch leaves handle_token ({} deltas, {} tokens seen)",
            result.len(),
            state.all_toks.len(),
        );
    }

    // Orphan-suppression streak watchdog. The sanitizer flips
    // `suppressing_param_leak=true` when it sees an orphan
    // `<tool_call>` / `<parameter=` opener without a matching close.
    // Suppressing forever (until max_tokens) burns the user's
    // patience and decode budget — observed live as an 8192-token
    // doom loop. If the streak exceeds the bound, end the stream.
    if state.suppressing_param_leak && !state.stop_string_triggered {
        state.suppress_streak_tokens = state.suppress_streak_tokens.saturating_add(1);
        if state.suppress_streak_tokens > MAX_SUPPRESS_STREAK_TOKENS {
            tracing::warn!(
                streak = state.suppress_streak_tokens,
                "orphan tool-call suppression streak exceeded {MAX_SUPPRESS_STREAK_TOKENS} tokens; ending stream",
            );
            state.loop_watchdog_triggered = true;
            state.stop_string_triggered = true;
            // ★ Name the guard so the wire reason comes out "length".
            // Without this the scheduler sees a bare `cancel_flag` with
            // budget left, falls to its early-finalize rule and reports
            // "stop" — telling the client the model finished, when in
            // fact we truncated it mid-doom-loop.
            //
            // This was the THIRD such path. The scheduler guards and the
            // simhash/token-loop watchdogs were fixed first; this one
            // survived both and cost the agentic gate runs 0 and 7 on
            // three consecutive shas, because the harness's
            // `was_cut_off()` only nudges on "length".
            //
            // ★ Note the harness's OTHER recovery route cannot cover it:
            // `emitted_unparsed_call` scans the text for `<tool_call>` /
            // `<function=`, and the sanitizer has already suppressed
            // exactly those bytes — which is why this streak fired at all.
            state.guard_stop = Some("suppress_streak");
            state
                .cancel_flag
                .store(true, std::sync::atomic::Ordering::Release);
        }
    } else if !state.suppressing_param_leak {
        state.suppress_streak_tokens = 0;
    }

    result
}

fn handle_token_inner(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> DeltaVec {
    let mut deltas: DeltaVec = Vec::new();
    state.all_toks.push(tok);
    // One push per call == one sampled token == one increment of
    // `usage.completion_tokens`. Drained onto the next client-visible
    // chunk (return_token_ids); a no-op clone-free take when opted out.
    if ctx.req_return_token_ids {
        state.pending_token_ids.push(tok);
    }

    // ── Thinking-phase: token-ID based </think> detection ────────────
    if !state.thinking_done {
        if let Some(end_id) = ctx.state.think_end_token_id
            && tok == end_id
        {
            state.thinking_done = true;
            // Emit only the residual reasoning delta not yet sent
            // by incremental streaming (e.g. trailing bytes held
            // back due to incomplete UTF-8 at prior token boundary).
            // The full reasoning has already been streamed
            // incrementally via reasoning_chunk deltas above —
            // re-emitting the full text here would double it.
            if ctx.enable_thinking && state.all_toks.len() > 1 {
                let full = ctx
                    .state
                    .tokenizer
                    .decode(&state.all_toks[..state.all_toks.len() - 1])
                    .unwrap_or_default();
                let stable = full.trim_end_matches('\u{FFFD}');
                if stable.len() > state.emitted {
                    let residual = &stable[state.emitted..];
                    // Same fix as the in-loop emit: whitespace-only residuals
                    // are legitimate `\n   ` indents that the model emitted;
                    // dropping them would lose chars permanently.
                    if !residual.is_empty() {
                        deltas.push(StreamDelta::Reasoning {
                            text: residual.to_string(),
                            token_ids: state.take_ids_if(ctx.req_return_token_ids),
                        });
                    }
                }
            }
            // Flush the reasoning sanitizer's tail buffer. Without this, up to
            // ~18 trailing bytes of the final thinking block (or anything held
            // back for partial-tag fusion) are silently dropped. Skip when
            // suppression is active (no close arrived during thinking) — those
            // bytes are intentionally not surfaced.
            if !state.reasoning_suppressing_leak && !state.reasoning_tag_scan_buf.is_empty() {
                let tail = std::mem::take(&mut state.reasoning_tag_scan_buf);
                // Whitespace-only tail can be a real trailing `\n   ` indent
                // — emit anything non-empty so byte boundaries align.
                if !tail.is_empty() {
                    deltas.push(StreamDelta::Reasoning {
                        text: tail,
                        token_ids: Vec::new(),
                    });
                }
            }
            // Reset tool detector to clear any thinking-era tag fragments.
            if let Some(ref mut det) = state.detector {
                det.reset();
            }
            state.emitted = 0; // Reset — next decode will be content-only
            state.all_toks.clear(); // Clear thinking tokens from accumulator
            state.content_decoded.clear();
            state.detok_prefix_offset = 0;
            state.detok_read_offset = 0;
            return deltas;
        }
        // Still in thinking — accumulate but don't emit as content
        if ctx.enable_thinking {
            // Layer-A one-shot guard: after the in-think tool-call leak
            // scanner has fired, suppress all subsequent reasoning
            // deltas for this stream. The scheduler's `cancel_flag`
            // (set when the scanner fired) finalises the sequence
            // within one token via `emit_step::emit_token`; this
            // guard catches the in-flight token race so the next
            // opener never reaches the client.
            if state.reasoning_xml_leak_detected {
                return deltas;
            }
            // Open thinking: emit as reasoning_content. Incrementally extend
            // the stable decoded text instead of re-decoding all_toks (O(n²)).
            let delta_stable = ctx.state.tokenizer.incremental_decode(
                &state.all_toks,
                &mut state.detok_prefix_offset,
                &mut state.detok_read_offset,
            );
            state.content_decoded.push_str(&delta_stable);
            let stable_end = state.content_decoded.len();
            if stable_end > state.emitted {
                let raw = state.content_decoded[state.emitted..stable_end].to_string();
                let mut cleaned = raw.clone();
                state.emitted = stable_end;
                // Strip format tokens that shouldn't appear in thinking.
                // `<think>` only fires at the literal opener (always
                // whitespace-adjacent in the prompt), so a plain replace
                // is safe here.
                cleaned = cleaned.replace("<think>", "");
                if let Some(rest) = cleaned.strip_prefix("assistant\n") {
                    cleaned = rest.to_string();
                } else if let Some(rest) = cleaned.strip_prefix("assistant") {
                    cleaned = rest.to_string();
                }
                // Boundary-preserving strip: see `strip_preserving_boundary`
                // doc — prevents `the<tool_call>...</tool_call>project`
                // from collapsing to `theproject`.
                while let Some(start) = cleaned.find("<tool_call>") {
                    if let Some(end_rel) = cleaned[start..].find("</tool_call>") {
                        let end = start + end_rel + "</tool_call>".len();
                        cleaned = strip_preserving_boundary(&cleaned, start, end);
                    } else {
                        cleaned = cleaned[..start].to_string();
                        break;
                    }
                }
                if let Some(start) = cleaned.find("<function=") {
                    cleaned = cleaned[..start].to_string();
                }
                // Strip leaked tool-call closing tags from reasoning
                // (observed pattern: `</parameter></function>` right
                // before a role-word repetition loop). Route through
                // `strip_all_preserving_boundary` (2026-05-23 sweep)
                // to avoid gluing words when a closing tag straddles
                // two reasoning sentences.
                for tag in &["</parameter>", "</function>", "</tool_call>"] {
                    cleaned = strip_all_preserving_boundary(&cleaned, tag);
                }
                // Collapse role-word repetition loops (Qwen3.5/3.6
                // post-tool-call hallucination): `userX...userX` →
                // "" until no adjacent pairs remain, then strip
                // line-bounded standalones (`\nuser\n` → `\n`).
                for word in &["user", "assistant", "tool"] {
                    let pair = format!("{word}{word}");
                    cleaned = strip_all_preserving_boundary(&cleaned, &pair);
                    let nl_form = format!("\n{word}\n");
                    while cleaned.contains(&nl_form) {
                        cleaned = cleaned.replace(&nl_form, "\n");
                    }
                }
                maybe_log_decode_trace(&raw, &cleaned, stable_end, stable_end - raw.len());
                // Layer-A in-think tool-call leak scanner. The per-
                // delta strippers above can miss boundary splits
                // (e.g. `<too` in delta N + `l_call>` in delta N+1)
                // and even when they strip, the model keeps emitting
                // the next repetition because its own KV already
                // contains the literal opener. This sliding-window
                // detector across deltas catches the opener on
                // arrival, drops the delta, sets the loop-cap flag
                // (→ finish_reason="length" via the PR #87 override)
                // and flips the scheduler cancel_flag so generation
                // terminates within one token via PR #89.
                let tools_active_request =
                    !ctx.tool_defs_for_backfill.is_empty() || state.detector.is_some();
                if tools_active_request {
                    state.reasoning_xml_scan_buf.push_str(&cleaned);
                    if state.reasoning_xml_scan_buf.len() > 256 {
                        let drop_to = state.reasoning_xml_scan_buf.len() - 256;
                        let cut = state
                            .reasoning_xml_scan_buf
                            .char_indices()
                            .find(|&(i, _)| i >= drop_to)
                            .map(|(i, _)| i)
                            .unwrap_or(state.reasoning_xml_scan_buf.len());
                        state.reasoning_xml_scan_buf.drain(..cut);
                    }
                    let opener = ["<tool_call>", "<function=", "<parameter=", "<invoke "]
                        .iter()
                        .copied()
                        .filter_map(|m| state.reasoning_xml_scan_buf.find(m).map(|at| (at, m)))
                        .min_by_key(|&(at, _)| at);
                    if let Some((at, op)) = opener {
                        // Snapshot the log tail BEFORE consuming the match,
                        // so the WARN still shows the opener in context.
                        let tail_start = state
                            .reasoning_xml_scan_buf
                            .char_indices()
                            .rev()
                            .nth(63)
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let tail = state.reasoning_xml_scan_buf[tail_start..].to_string();
                        // Consume the buffer through this opener so ONE
                        // occurrence is counted once, not on every
                        // subsequent delta the rolling tail still contains
                        // it in.
                        state.reasoning_xml_scan_buf.drain(..at + op.len());
                        state.reasoning_xml_opener_hits =
                            state.reasoning_xml_opener_hits.saturating_add(1);
                        let threshold = ctx.state.chat.in_think_leak_openers;
                        if threshold != 0 && state.reasoning_xml_opener_hits >= threshold {
                            state.reasoning_xml_leak_detected = true;
                            // Name the cut for the --dump body / Done event;
                            // `tool_loop_capped` wires finish_reason
                            // "length" (the shipped doom-loop contract).
                            state.guard_stop = Some("in_think_tool_leak");
                            state.tool_loop_capped = true;
                            state.stop_string_triggered = true;
                            state
                                .cancel_flag
                                .store(true, std::sync::atomic::Ordering::Release);
                            tracing::warn!(
                                model = %ctx.model,
                                request_id = %ctx.id,
                                opener = op,
                                hits = state.reasoning_xml_opener_hits,
                                threshold,
                                tail = %tail,
                                "in-think tool-call leak: opener threshold reached; cancelling \
                                 sequence (finish_reason \"length\", guard in_think_tool_leak). \
                                 Raise/disable via ATLAS_INTHINK_TOOL_LEAK_OPENERS (0 = strip-only)"
                            );
                            return deltas;
                        }
                        // Below threshold: keep streaming. A model whose
                        // native tool syntax surfaces in legitimate
                        // reasoning (qwen3_coder family) lands here instead
                        // of losing the whole turn. Same-delta openers were
                        // stripped by the cleanups above; an opener SPLIT
                        // across deltas may have partially reached the
                        // client's reasoning channel — cosmetic, and true
                        // of the pre-knob code's one free delta too.
                        tracing::warn!(
                            model = %ctx.model,
                            request_id = %ctx.id,
                            opener = op,
                            hits = state.reasoning_xml_opener_hits,
                            threshold,
                            "in-think tool-call opener observed in reasoning; below \
                             ATLAS_INTHINK_TOOL_LEAK_OPENERS threshold, not cancelling"
                        );
                    }
                }
                // F19: final structured sanitisation pass catches
                // any leak markers the hand-rolled cleanups missed.
                let cleaned = sanitize_content_chunk(
                    &cleaned,
                    &mut state.reasoning_tag_scan_buf,
                    &mut state.reasoning_suppressing_leak,
                    &mut state.reasoning_inside_envelope,
                    &ctx.leak_markers,
                );
                // Emit whitespace-only chunks too. The `sanitize_content_chunk`
                // holdback can roll out runs of `\n   ` (newline + indent) as
                // a single committed chunk when the suffix exceeds tag_max
                // chars; dropping those via `trim().is_empty()` permanently
                // loses byte boundaries because `state.emitted` already
                // advanced past them. Symptom: streamed reasoning has
                // `**\n -Calculate` where the model actually emitted
                // `**\n   - Calculate` — verified byte-for-byte against the
                // non-streaming response on temp=0 seed=42 (live A/B
                // 2026-05-25). Drop only TRULY empty chunks.
                if !cleaned.is_empty() {
                    deltas.push(StreamDelta::Reasoning {
                        text: cleaned,
                        token_ids: state.take_ids_if(ctx.req_return_token_ids),
                    });
                }
            }
        }
        return deltas;
    }

    // ── Content phase: full-decode + slice (matches reasoning path) ──
    //
    // Previously this path used the HF `tokenizers` crate's
    // `DecodeStream` (`decoder.step(tok)`). That incremental decoder
    // drops the leading metaspace byte at certain BPE-token boundaries
    // for byte-level tokenizers like Qwen's GPT-2-style BPE — verified
    // live 2026-05-25 against the FP8 Qwen3.6 model, opencode session
    // `ses_1a0e59bc7ffeFKSvtvWqoswsll`: tool-call `<parameter=content>`
    // for a Cargo.toml emitted `name = test-rust-axum-v32version =
    // 0.1.0edition = 2021` (no newlines between fields, no quotes
    // around values). Non-streaming `tokenizer.decode(&all_toks)`
    // for the same tokens produces the correct multi-line TOML.
    //
    // The fix: mirror the reasoning path — keep `state.all_toks`
    // populated with content tokens (already done at line 86), decode
    // the cumulative list, and emit the byte slice that's stable past
    // `state.emitted`. `trim_end_matches('\u{FFFD}')` defers any
    // incomplete UTF-8 multi-byte sequence at the tail until the next
    // token completes it. `state.all_toks` and `state.emitted` are
    // reset at `</think>` (line 147), so this slice references the
    // post-thinking content only.
    // Incrementally extend the stable decoded text instead of re-decoding the
    // whole `all_toks` list every token (O(n²)). `content_decoded` stays
    // byte-identical to the previous `decode(&all_toks)` trimmed of any
    // trailing incomplete-multibyte token.
    let delta_stable = ctx.state.tokenizer.incremental_decode(
        &state.all_toks,
        &mut state.detok_prefix_offset,
        &mut state.detok_read_offset,
    );
    state.content_decoded.push_str(&delta_stable);
    let stable_end = state.content_decoded.len();
    let _ = tok; // tok already in state.all_toks via line 86
    let mut delta = if stable_end > state.emitted {
        // TTFT forensics: the moment the FIRST stable content bytes exist
        // stream-side. Compare against the scheduler's "Prefill first
        // token" and the client's first-delta wall time to attribute any
        // emission-path latency (task: first-delta gap).
        if state.emitted == 0 {
            tracing::debug!(
                "stream: first stable content delta ({stable_end} bytes: {:?})",
                &state.content_decoded[..stable_end.min(24)],
            );
        }
        let raw = state.content_decoded[state.emitted..stable_end].to_string();
        state.emitted = stable_end;
        raw
    } else {
        return deltas;
    };
    // Retire the lazy `content_decoder` field — kept in StreamState
    // only to avoid a wider state-struct migration. The HF decoder is
    // no longer the source of truth.
    let _ = &state.content_decoder;

    // Strip residual think tags from content after thinking is done.
    if state.thinking_done {
        // SSOT with the blocking path (api/chat_blocking.rs), which had no
        // equivalent scrub and therefore leaked '</think>' into content.
        delta = crate::api::strip::scrub_think_markers(&delta);
        // If model re-opens <think>, suppress content from <think> onward.
        if let Some(pos) = delta.find("<think>") {
            delta = delta[..pos].to_string();
            state.thinking_done = false;
            state.all_toks.clear();
            state.emitted = 0;
            state.content_decoded.clear();
            state.detok_prefix_offset = 0;
            state.detok_read_offset = 0;
        }
        // Orphan tool-call markup in content when NO tools are defined: the
        // model emits a spurious `<tool_call>…` block after a normal answer and
        // the tool detector (which would strip it) is not running. Cut content
        // at the opener — SSOT with the blocking path's strip_orphan_tool_markup.
        // When tools ARE active the detector owns tool markup, so leave it.
        let tools_active_request =
            !ctx.tool_defs_for_backfill.is_empty() || state.detector.is_some();
        if !tools_active_request {
            delta = crate::api::strip::strip_orphan_tool_markup(&delta);
        }
    }

    // Bare role-literal leak (Qwen3.5/3.6) — companion to the
    // scheduler-side <|im_start|> hard-stop. Suppressed mid tool-call
    // body: there a standalone `tool` token is the leading BPE fragment
    // of a `tool_*` NAME (issue #222) being reassembled, not a role leak.
    {
        let inside_tool_call = state
            .detector
            .as_ref()
            .is_some_and(|d| d.inside_tool_call());
        strip_bare_role_literal(&mut delta, inside_tool_call);
    }

    if delta.is_empty() {
        return deltas;
    }

    // Multi-token stop sequences via string matching, with a vLLM-style
    // hold-back buffer (see `vllm/v1/engine/detokenizer.py`
    // `IncrementalDetokenizer.update`). All the state mutation lives in
    // `apply_stop_string_holdback` so the algorithm can be unit-tested
    // without spinning up a full `StreamCtx`.
    if !ctx.stop_strings.is_empty() && !state.stop_string_triggered {
        delta = apply_stop_string_holdback(
            &delta,
            &ctx.stop_strings,
            ctx.stop_string_buffer_len,
            &mut state.accumulated_content,
            &mut state.stop_string_emitted_len,
            &mut state.stop_string_triggered,
        );
        // Entry was gated on `!triggered`, so a flip here is a GENUINE
        // client stop-sequence match (the watchdog paths set the flag
        // elsewhere): record it for the wire finish_reason ("stop", per
        // OpenAI) and cancel generation.
        if state.stop_string_triggered {
            state.note_stop_string_match();
        }
        if delta.is_empty() {
            // Either everything is sitting in the hold-back window
            // (waiting for the next chunk / stream close) or a match
            // already truncated the emittable bytes to nothing.
            return deltas;
        }
    }

    if state.stop_string_triggered {
        // The tool guards (F11 within-response dedup, hard-validation
        // reject, loop cap) reuse `stop_string_triggered` as a generic
        // "end this response" flag. But the scheduler keeps generating for
        // a few tokens after the flag is set (cancel latency), and on a
        // tool-call *runaway* those tokens are raw markup —
        // `<tool_call><function=…><parameter=…>…</tool_call></_call>` —
        // re-emitted here. A raw passthrough (the old behaviour) leaked
        // that markup into `content` because this branch returns BEFORE the
        // detector/sanitizer fork below. Route the delta through the
        // buffered `sanitize_content_chunk` so multi-token markers that
        // straddle deltas (`</_call>` = `</` `_` `call` `>`) are reassembled
        // and scrubbed. For a genuine stop string, legitimate trailing
        // content is untouched — the sanitizer only removes tool markup.
        if !delta.is_empty() {
            let cleaned = sanitize_content_chunk(
                &delta,
                &mut state.tag_scan_buf,
                &mut state.suppressing_param_leak,
                &mut state.inside_envelope,
                &ctx.leak_markers,
            );
            if !cleaned.is_empty() {
                deltas.push(StreamDelta::Content {
                    text: cleaned,
                    token_ids: state.take_ids_if(ctx.req_return_token_ids),
                });
            }
        }
        return deltas;
    }

    // Fork: detector-active vs pure-content path.
    if state.detector.is_some() {
        // Drain the detector outputs into a local Vec so we can drop
        // the &mut borrow on `state.detector` before the helpers below
        // (which take other &mut state fields) run.
        let outputs = {
            let det = state.detector.as_mut().expect("detector is Some");
            det.process(&delta)
        };
        for output in outputs {
            match output {
                tool_parser::DetectorOutput::Content(text) => {
                    if let Some(events_out) = detector_content_arm(state, ctx, &text) {
                        deltas.extend(events_out);
                        return deltas;
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
            &delta,
            &mut state.tag_scan_buf,
            &mut state.suppressing_param_leak,
            &mut state.inside_envelope,
            &ctx.leak_markers,
        );
        if let Some(events_out) = process_detector_content(state, ctx, &sanitized) {
            deltas.extend(events_out);
            return deltas;
        }
        // process_detector_content does NOT pre-sanitize when called
        // from the no-detector branch — but the sanitizer was already
        // run above, so the helper's branch handling matches.
    }

    deltas
}

/// Common processing for a sanitized content chunk: SimHash semantic
/// guard, token-level loop watchdog, salvage on trip, otherwise
/// emit a `Content` delta. Returns `Some(deltas)` when the watchdog
/// fired (caller must short-circuit), else `None` (caller continues).
///
/// Note: when called from the detector-active branch, `sanitized`
/// has already been routed through `sanitize_content_chunk`. When
/// called from the no-detector branch, the caller must pre-sanitize
/// (the no-detector path uses the same sanitizer state).
fn process_detector_content(
    state: &mut StreamState,
    ctx: &StreamCtx,
    sanitized_or_raw: &str,
) -> Option<DeltaVec> {
    // From the detector-active branch the input is the Content(text)
    // payload that still needs sanitization. From the no-detector
    // branch the input is already sanitized. Distinguish via a thin
    // wrapper: detector branch ALSO sanitizes; non-detector branch
    // skips by passing the already-sanitized text. To keep the call
    // site simple, we sanitize here only when the input contains the
    // hallmark of an unfiltered Content payload — which we can't
    // reliably detect. Solution: split into two paths.
    //
    // Inlining: this helper is only called once per branch with the
    // correct input type; it never re-sanitizes. The parameter is the
    // post-sanitizer text in both call sites.
    let sanitized = sanitized_or_raw;

    // F4 SimHash guard. `ATLAS_SIMHASH_LOOP=0` disables it (house watchdog
    // convention, same shape as ATLAS_TOOL_ENVELOPE_WATCHDOG): the guard is
    // ONE-STRIKE at Jaccard 0.55 over a 16-sentence ring, which legitimate
    // structured output crosses easily — per-method docstrings, enumerations,
    // boilerplate-heavy code all produce >=0.55 bigram overlap between
    // honest sentences, and a fire KILLS the stream mid-reply. Before the
    // #699 verdict fix most of its fires were masking genuinely degenerate
    // output; with generation clean, the remaining fires skew
    // false-positive (observed 2026-08-21: fired on a healthy TUI session).
    let semantic_trip = if simhash_loop_enabled() && !state.loop_watchdog_triggered {
        state.simhash_pending.push_str(sanitized);
        let mut dup = false;
        if crate::loop_simhash::ends_at_sentence_boundary(&state.simhash_pending).is_some()
            || state.simhash_pending.len() >= 1024
        {
            dup = state.simhash_guard.check(&state.simhash_pending);
            state.simhash_pending.clear();
        }
        if state.simhash_pending.len() > 4096 {
            let drop_to = state.simhash_pending.len() / 2;
            state.simhash_pending.drain(..drop_to);
        }
        dup
    } else {
        false
    };

    let token_trip = check_loop_watchdog(
        sanitized,
        &mut state.loop_scan_buf,
        state.loop_watchdog_triggered,
    );

    if semantic_trip || token_trip {
        if semantic_trip {
            tracing::warn!(
                ring_len = state.simhash_guard.len(),
                "SimHash semantic-loop watchdog fired (paraphrased sentence repeat)"
            );
        }
        state.loop_watchdog_triggered = true;
        state.stop_string_triggered = true;
        state.guard_stop = Some(if semantic_trip {
            "simhash_semantic_loop"
        } else {
            "token_loop_watchdog"
        });
        state
            .cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);

        // Watchdog fired: short-circuit the stream with no further
        // content. The model emitted a degenerate loop; we end the
        // response here rather than salvaging a synthetic tool call.
        return Some(DeltaVec::new());
    }

    if !sanitized.is_empty() {
        if state.refusal_scan_buf.len() < 16_384 {
            state.refusal_scan_buf.push_str(sanitized);
        }
        let out: DeltaVec = vec![StreamDelta::Content {
            text: sanitized.to_string(),
            token_ids: state.take_ids_if(ctx.req_return_token_ids),
        }];
        return Some(out);
    }
    None
}

/// Detector-active branch's `Content(text)` arm: sanitize first,
/// then run the shared semantic/token watchdog + emit pipeline.
fn detector_content_arm(state: &mut StreamState, ctx: &StreamCtx, text: &str) -> Option<DeltaVec> {
    let sanitized = sanitize_content_chunk(
        text,
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &mut state.inside_envelope,
        &ctx.leak_markers,
    );
    process_detector_content(state, ctx, &sanitized)
}

/// Pure stop-string accumulator + hold-back algorithm. Returns the
/// bytes that should be forwarded to the client this delta; the
/// remainder (≤ `buffer_len` bytes) stays withheld inside
/// `accumulated_content` until the next call or until `handle_done`
/// flushes the tail at stream close.
///
/// Mirrors vLLM's `IncrementalDetokenizer.update`
/// (`vllm/v1/engine/detokenizer.py`):
/// 1. Append `new_chars` to the accumulator.
/// 2. Search the accumulator for any stop string.
/// 3a. On hit, truncate the accumulator AND the emittable delta at
///     the match position (Atlas never echoes the stop literal).
/// 3b. On miss, hold back the last `buffer_len` bytes; emit
///     everything between the previously emitted offset and the
///     hold-back boundary, snapped to a valid UTF-8 char boundary.
///
/// Pre/postconditions:
/// - `*triggered` must be `false` on entry (callers gate on this).
/// - On match, `*triggered` is flipped to `true` and the accumulator
///   is truncated to the prefix that precedes the stop string.
/// - On miss, `*triggered` stays `false`.
pub(super) fn apply_stop_string_holdback(
    new_chars: &str,
    stop_strings: &[String],
    buffer_len: usize,
    accumulated_content: &mut String,
    emitted_len: &mut usize,
    triggered: &mut bool,
) -> String {
    debug_assert!(!*triggered, "caller must gate on !triggered");
    accumulated_content.push_str(new_chars);

    // Bounded search window: only the suffix that could contain a stop
    // string straddling the newly appended chars can hold a *new* match —
    // every prior call already full-scanned (and found nothing in) the
    // content before this window. A match can begin at earliest
    // `max_stop_len - 1` bytes before the new chars; we also back up over
    // the held-back `buffer_len` bytes for margin. This keeps per-token
    // cost O(new + buffer + max_stop) instead of O(total), turning the
    // whole-response scan from O(n²) into O(n).
    let max_stop_len = stop_strings.iter().map(String::len).max().unwrap_or(0);
    let search_start = {
        let raw = accumulated_content
            .len()
            .saturating_sub(new_chars.len() + buffer_len + max_stop_len);
        accumulated_content.floor_char_boundary(raw)
    };
    let matched_pos = stop_strings
        .iter()
        .filter_map(|s| accumulated_content[search_start..].find(s.as_str()))
        .min()
        .map(|rel| rel + search_start);

    if let Some(pos) = matched_pos {
        accumulated_content.truncate(pos);
        let emit_start = (*emitted_len).min(pos);
        let out = accumulated_content[emit_start..pos].to_string();
        *emitted_len = pos;
        *triggered = true;
        return out;
    }

    // No match: hold back the last `buffer_len` bytes. Snap to a UTF-8
    // boundary so the emitted prefix is always valid Rust `str` and
    // the held-back tail never contains a partial codepoint.
    let acc_len = accumulated_content.len();
    let raw_emit_end = acc_len.saturating_sub(buffer_len);
    let emit_end = accumulated_content.floor_char_boundary(raw_emit_end);
    let emit_start = (*emitted_len).min(emit_end);
    let out = accumulated_content[emit_start..emit_end].to_string();
    *emitted_len = emit_end;
    out
}

#[cfg(test)]
mod stop_string_holdback_tests {
    use super::apply_stop_string_holdback;

    /// Stop string spanning a chunk boundary must not leak the
    /// partial prefix in the first delta. When the suffix arrives in
    /// the next chunk the full output up to (but excluding) the stop
    /// string is emitted; the stop literal itself is consumed.
    #[test]
    fn stop_string_spanning_chunk_boundary_does_not_leak() {
        let stops = vec!["<|im_start|>".to_string()];
        let buffer_len = "<|im_start|>".len() - 1; // 11
        let mut acc = String::new();
        let mut emitted = 0usize;
        let mut triggered = false;

        // Delta 1: "hello " — entirely inside the hold-back window
        // (6 bytes < buffer_len=11). Nothing emitted.
        let out = apply_stop_string_holdback(
            "hello ",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "");
        assert_eq!(acc, "hello ");
        assert!(!triggered);

        // Delta 2: "<|im_st" — partial stop string. acc="hello <|im_st"
        // (len=13). raw_emit_end=13-11=2, so we emit "he".
        // Crucially, "<|im_st" is HELD BACK — never sent to client.
        let out = apply_stop_string_holdback(
            "<|im_st",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "he");
        assert!(!out.contains("<|im_st"), "partial stop leaked to client");
        assert!(!triggered);

        // Delta 3: "art|>" completes the stop string. We match at
        // pos=6, truncate acc to "hello ", and emit "llo " (bytes
        // 2..6 of acc).
        let out = apply_stop_string_holdback(
            "art|>",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "llo ");
        assert_eq!(acc, "hello ");
        assert!(triggered);

        // Concatenating all emitted deltas yields the pre-stop output
        // ("hello ") with the stop literal consumed. No partial leak.
        let total = String::new() + "" + "he" + "llo ";
        assert_eq!(total, "hello ");
        assert!(!total.contains("<|im_st"));
        assert!(!total.contains("<|im_start|>"));
    }

    /// When no stop strings are configured, `buffer_len=0` and the
    /// hold-back collapses to a pass-through: every byte of every
    /// delta is emitted immediately.
    #[test]
    fn no_stop_strings_is_zero_behavior_change() {
        let stops: Vec<String> = Vec::new();
        let buffer_len = 0usize;
        let mut acc = String::new();
        let mut emitted = 0usize;
        let mut triggered = false;

        let out = apply_stop_string_holdback(
            "hello ",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "hello ");
        assert!(!triggered);

        let out = apply_stop_string_holdback(
            "world",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "world");
        assert!(!triggered);

        // Even a string that LOOKS like a stop marker is forwarded
        // verbatim because no stop strings are configured.
        let out = apply_stop_string_holdback(
            "<|im_start|>",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "<|im_start|>");
        assert!(!triggered);
    }

    /// Multi-byte UTF-8 inside the hold-back window must never be
    /// sliced mid-codepoint. `floor_char_boundary` snaps the cut to a
    /// valid boundary so the emitted prefix is always valid `str`.
    #[test]
    fn utf8_boundary_safety_in_holdback() {
        // "é" is 2 bytes (0xC3 0xA9). Build an accumulator whose
        // raw cut would land inside the codepoint and verify
        // floor_char_boundary saves us.
        let stops = vec!["STOP".to_string()];
        let buffer_len = 3usize; // > 0 to exercise the hold-back
        let mut acc = String::new();
        let mut emitted = 0usize;
        let mut triggered = false;

        // acc becomes "aébc" (5 bytes). raw_emit_end = 5-3 = 2 lands
        // mid-codepoint of 'é' (1..3). floor_char_boundary snaps to
        // 1, so we emit "a" only.
        let out = apply_stop_string_holdback(
            "aébc",
            &stops,
            buffer_len,
            &mut acc,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(out, "a");
        assert!(out.is_char_boundary(out.len()));
        assert!(!triggered);
    }
}

#[cfg(test)]
mod role_literal_strip_tests {
    use super::strip_bare_role_literal;
    use crate::tool_parser::{DetectorOutput, StreamingToolDetector};

    /// Faithfully mirror the `handle_token` content-phase pipeline for the
    /// two steps under test: the bare role-literal strip (guarded by the
    /// detector's `inside_tool_call()`) feeding the surviving delta into the
    /// streaming tool-call detector. Returns every `ToolCallStart` name.
    fn stream_names(chunks: &[&str]) -> Vec<String> {
        let mut det = StreamingToolDetector::new();
        let mut names = Vec::new();
        for &c in chunks {
            let mut delta = c.to_string();
            // Same call the real content phase makes, in the same order:
            // read the detector's in-body flag BEFORE feeding this delta.
            let inside_tool_call = det.inside_tool_call();
            strip_bare_role_literal(&mut delta, inside_tool_call);
            if delta.is_empty() {
                continue;
            }
            for o in det.process(&delta) {
                if let DetectorOutput::ToolCallStart { name, .. } = o {
                    names.push(name);
                }
            }
        }
        for o in det.flush() {
            if let DetectorOutput::ToolCallStart { name, .. } = o {
                names.push(name);
            }
        }
        names
    }

    /// Issue #222: a `tool_search` NAME whose leading `tool` arrives as a
    /// standalone BPE fragment must stream intact, not truncate to `_search`.
    #[test]
    fn tool_search_name_split_after_tool_streams_intact() {
        let names = stream_names(&[
            "<tool_call>\n{\"name\": \"",
            "tool", // standalone BPE fragment of the NAME
            "_search\", \"arguments\": {\"query\": \"CRM\"}}",
            "\n</tool_call>",
        ]);
        assert_eq!(names, vec!["tool_search".to_string()]);
    }

    /// `tool_call` — the name that collides most directly with the markup.
    #[test]
    fn tool_call_name_split_after_tool_streams_intact() {
        let names = stream_names(&[
            "<tool_call>\n{\"name\": \"",
            "tool",
            "_call\", \"arguments\": {}}",
            "\n</tool_call>",
        ]);
        assert_eq!(names, vec!["tool_call".to_string()]);
    }

    /// `tool_describe` — same class, different suffix.
    #[test]
    fn tool_describe_name_split_after_tool_streams_intact() {
        let names = stream_names(&[
            "<tool_call>\n{\"name\": \"",
            "tool",
            "_describe\", \"arguments\": {\"id\": 7}}",
            "\n</tool_call>",
        ]);
        assert_eq!(names, vec!["tool_describe".to_string()]);
    }

    /// Non-`tool_*` name fed in fragments must remain intact (control).
    #[test]
    fn ordinary_name_streams_intact() {
        let names = stream_names(&[
            "<tool_call>\n{\"name\": \"get",
            "_weather\", \"arguments\": {\"city\": \"NYC\"}}",
            "\n</tool_call>",
        ]);
        assert_eq!(names, vec!["get_weather".to_string()]);
    }

    /// The #204/#205/#206 leak suppression the guard must NOT regress: a bare
    /// role literal in genuine content (no tool call in flight) is still
    /// cleared. Verified for all three literals and for a `tool` fragment
    /// that is only a real leak when `inside_tool_call == false`.
    #[test]
    fn bare_role_literal_still_stripped_outside_tool_call() {
        for lit in ["user", "assistant", "tool", "  tool  "] {
            let mut d = lit.to_string();
            strip_bare_role_literal(&mut d, false);
            assert!(
                d.is_empty(),
                "bare role literal {lit:?} must be stripped in content"
            );
        }
    }

    /// Inside a tool-call body the same fragments must survive untouched.
    #[test]
    fn bare_role_literal_preserved_inside_tool_call() {
        for lit in ["user", "assistant", "tool"] {
            let mut d = lit.to_string();
            strip_bare_role_literal(&mut d, true);
            assert_eq!(d, lit, "fragment {lit:?} must survive inside a tool call");
        }
    }

    #[test]
    fn dsml_split_opener_preserves_tool_fragment_and_streams_call() {
        let mut det = StreamingToolDetector::new();
        let chunks = [
            "\n\n",
            "<｜DSM",
            "L",
            "｜",
            "tool",
            "_calls>\n<｜DSML｜invoke name=\"get_weather\">\n",
            "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        ];
        let mut outputs = Vec::new();
        for chunk in chunks {
            let mut delta = chunk.to_string();
            strip_bare_role_literal(&mut delta, det.inside_tool_call());
            if !delta.is_empty() {
                outputs.extend(det.process(&delta));
            }
        }
        outputs.extend(det.flush());

        let mut content = String::new();
        let mut calls = Vec::new();
        for output in outputs {
            match output {
                DetectorOutput::Content(text) => content.push_str(&text),
                DetectorOutput::ToolCall(call, index) => calls.push((index, call)),
                _ => {}
            }
        }
        assert!(content.is_empty(), "DSML envelope leaked: {content:?}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, 0);
        assert_eq!(calls[0].1.function.name, "get_weather");
        assert_eq!(calls[0].1.function.arguments, r#"{"city":"Paris"}"#);
    }

    /// Ordinary content (not a bare role literal) is never touched, in or out
    /// of a tool call.
    #[test]
    fn ordinary_content_untouched() {
        for inside in [false, true] {
            let mut d = "the tool ran".to_string();
            strip_bare_role_literal(&mut d, inside);
            assert_eq!(d, "the tool ran");
        }
    }
}
