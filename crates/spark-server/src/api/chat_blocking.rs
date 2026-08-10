// SPDX-License-Identifier: AGPL-3.0-only

//! Blocking (non-streaming) `/v1/chat/completions` path. Extracted from
//! `chat_completions_inner` (refactor wave-4e) to keep `chat.rs` under
//! the 500 LoC cap. Supports `n >= 1` (multiple choices per request) by
//! looping the scheduler send + decode + tool-parse pipeline once per
//! choice index.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::AppState;
use crate::ir;
use crate::tool_parser;

use super::compact::openai_error_response;
use super::inference_impl::{extract_thinking, strip_stop_sequences};
use super::inference_types::{GrammarSpec, InferenceRequest};

pub(super) struct BlockingPathArgs {
    pub state: Arc<AppState>,
    pub req: crate::ir::ChatRequest,
    pub req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    pub prompt_tokens: Vec<u32>,
    pub session_hash: u64,
    /// M2 per-request LoRA routing: resolved adapter slot (`-1` = defer to active).
    pub adapter_slot: i32,
    /// Resolved source-language token id (0 = deployment default).
    pub src_lang_id: u32,
    /// Resolved target-language token id (0 = deployment default).
    pub tgt_lang_id: u32,
    /// NLLB beam search: beams per request (1 = greedy).
    pub num_beams: u32,
    /// NLLB beam search: length penalty.
    pub length_penalty: f32,
    /// NLLB beam search: early stopping.
    pub early_stopping: bool,
    pub image_pixels: Vec<(Vec<f32>, usize, usize)>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    pub lz_penalty: f32,
    pub logit_bias: Vec<(u32, f32)>,
    pub stop_tokens: Vec<u32>,
    pub enable_thinking: bool,
    pub thinking_budget: Option<u32>,
    pub tools_active: bool,
    pub tool_choice_required: bool,
    pub suppress_tool_call: bool,
    pub grammar_spec: Option<GrammarSpec>,
    pub top_logprobs: Option<u8>,
    pub timeout_at: Option<std::time::Instant>,
    pub cwd_hint: Option<String>,
    pub prompt_len: usize,
}

pub(super) async fn run_blocking_path(args: BlockingPathArgs) -> super::chat::ChatOutcome {
    let BlockingPathArgs {
        state,
        req,
        req_ctx,
        prompt_tokens,
        session_hash,
        adapter_slot,
        src_lang_id,
        tgt_lang_id,
        num_beams,
        length_penalty,
        early_stopping,
        image_pixels,
        max_tokens,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        stop_tokens,
        enable_thinking,
        thinking_budget,
        tools_active,
        tool_choice_required,
        suppress_tool_call,
        grammar_spec,
        top_logprobs,
        timeout_at,
        cwd_hint,
        prompt_len,
    } = args;

    let n = req.n.max(1);
    let mut all_choices: Vec<ir::Choice> = Vec::with_capacity(n);
    let mut total_completion_tokens = 0usize;
    let mut first_ttft = 0.0f64;
    let mut last_decode_time_ms = 0.0f64;
    let mut total_reasoning_tokens = 0u32;
    let mut total_cached_prompt_tokens = 0u32;

    // Arc-wrap the prompt tokens ONCE. Per-choice scheduler requests
    // and the Tier 5c retry path all share the same Arc — no Vec<u32>
    // deep clones (~40 KB on a typical long-context opencode prompt).
    let prompt_tokens = std::sync::Arc::new(prompt_tokens);

    for choice_idx in 0..n {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let request = InferenceRequest::Blocking {
            prompt_tokens: prompt_tokens.clone(),
            session_hash,
            adapter_slot,
            src_lang_id,
            tgt_lang_id,
            num_beams,
            length_penalty,
            early_stopping,
            image_pixels: if choice_idx == 0 {
                image_pixels.clone()
            } else {
                Vec::new()
            },
            max_tokens,
            min_tokens: req.min_tokens,
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            lz_penalty,
            logit_bias: logit_bias.clone(),
            stop_tokens: stop_tokens.clone(),
            enable_thinking,
            thinking_budget,
            repetition_detection: req.repetition_detection,
            require_tool_call: tool_choice_required,
            tools_present: tools_active,
            suppress_tool_call,
            disable_mtp: false,
            grammar_spec: grammar_spec.clone(),
            seed: req.seed.map(|s| s.wrapping_add(choice_idx as u64)),
            top_logprobs,
            prompt_logprobs: None,
            echo: false,
            timeout_at,
            response_tx: tx,
        };

        if state.request_tx.send(request).await.is_err() {
            return super::chat::ChatOutcome::Http(openai_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "Scheduler queue full".to_string(),
            ));
        }

        let response = match rx.await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                return super::chat::ChatOutcome::Http(openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Inference error: {e}"),
                ));
            }
            Err(_) => {
                return super::chat::ChatOutcome::Http(openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Inference cancelled".to_string(),
                ));
            }
        };

        if choice_idx == 0 {
            first_ttft = response.time_to_first_token_ms;
        }
        last_decode_time_ms = response.decode_time_ms;

        let num_completion = response.output_tokens.len();
        total_completion_tokens += num_completion;
        total_reasoning_tokens += response.reasoning_tokens;
        // cached_prompt_tokens is a per-request prefix-cache hit count; for
        // n>1 we only charge once (same prompt reused).
        total_cached_prompt_tokens = total_cached_prompt_tokens.max(response.cached_prompt_tokens);

        let (reasoning_content_i, output_text_i) =
            decode_response_text(&state, &response, enable_thinking);
        let (output_text_i, matched_stop) =
            super::inference_impl::strip_stop_sequences_matched(output_text_i, &req.stop);

        let mut choice = build_choice_message(
            &state,
            &req,
            &response,
            reasoning_content_i,
            output_text_i,
            tools_active,
            cwd_hint.as_deref(),
            choice_idx,
        );
        choice.index = choice_idx;
        choice.matched_stop = matched_stop;
        choice.logprobs = build_logprobs(&state, &response);
        all_choices.push(choice);
    }

    finalize_response(
        state,
        req_ctx,
        all_choices,
        total_completion_tokens,
        first_ttft,
        last_decode_time_ms,
        total_reasoning_tokens,
        total_cached_prompt_tokens,
        prompt_len,
    )
}

/// Decode `(reasoning_content, output_text)` from the scheduler's
/// response. When `enable_thinking=true`, split at the first `</think>`
/// token. When `enable_thinking=false`, decode all output_tokens as
/// content — mirrors streaming's `thinking_done = !enable_thinking`
/// init in chat_stream/state.rs and recovers the answer Qwen3.x emits
/// inside `<think>...</think>` when it ignores a closed-thinking
/// prefill (issue #40).
fn decode_response_text(
    state: &AppState,
    response: &super::inference_types::InferenceResponse,
    enable_thinking: bool,
) -> (Option<String>, String) {
    let output_tokens =
        output_tokens_without_stop(&response.output_tokens, response.finish_reason.as_str());
    if let Some(think_tok) = state.think_end_token_id {
        if let Some((thinking_tokens, content_tokens)) =
            split_at_first_think_end(output_tokens, think_tok, enable_thinking)
        {
            let reasoning = if !thinking_tokens.is_empty() {
                state
                    .tokenizer
                    .decode(thinking_tokens)
                    .ok()
                    .filter(|s| !s.trim().is_empty())
            } else {
                None
            };
            // The split consumes only the FIRST close. Laguna emits the close
            // more than once (observed: [19, answer, 19, answer] for a question
            // it declines to reason about), so scrub any survivor — the
            // streaming path has always done this, the blocking path did not,
            // which is why stream:false leaked '</think>' into content while
            // stream:true did not.
            let content = super::strip::scrub_think_markers(
                state
                    .tokenizer
                    .decode(content_tokens)
                    .unwrap_or_default()
                    .trim_start(),
            );
            return (reasoning, content);
        }
        // Fallback: the exact close TOKEN was not found. That does not mean
        // the model never closed — it routinely spells the close as ordinary
        // BPE tokens instead of id 19, because the scheduler masks id 19:
        // PostCloseThinkMask pins logits[19] = -inf once think_ended is set,
        // and MidWordThinkEndMask masks it at ~93% of thinking positions. Both
        // push probability mass onto the text spelling ["</", "think", ">"],
        // which decodes byte-identically and is invisible to every `== 19`
        // comparison in the stack.
        //
        // So fall back to the STRING-level reasoning parser rather than
        // dumping everything into content. It is configured for this model and
        // was previously unreachable here (the `else` below only runs when
        // think_end_token_id is None), and it handles the unterminated case by
        // routing the chain of thought to reasoning instead of content.
        let text = state.tokenizer.decode(output_tokens).unwrap_or_default();
        let (reasoning, content) =
            extract_thinking(&text, enable_thinking, state.reasoning_parser.as_deref());
        (reasoning, super::strip::scrub_think_markers(&content))
    } else {
        let text = state.tokenizer.decode(output_tokens).unwrap_or_default();
        extract_thinking(&text, enable_thinking, state.reasoning_parser.as_deref())
    }
}

/// Split native reasoning from assistant content at the first end marker.
/// Any later marker belongs to post-reasoning output and must not move an
/// already-complete answer or tool call back into the reasoning channel.
fn split_at_first_think_end(
    tokens: &[u32],
    think_end_token: u32,
    enable_thinking: bool,
) -> Option<(&[u32], &[u32])> {
    if !enable_thinking {
        return None;
    }

    let pos = tokens.iter().position(|&token| token == think_end_token)?;
    Some((&tokens[..pos], &tokens[pos + 1..]))
}

/// The scheduler retains a terminal stop token in blocking output for usage
/// accounting. Do not expose its decoded text when a tokenizer does not mark
/// that EOS token as special (Laguna's `</assistant>` is one such token).
fn output_tokens_without_stop<'a>(tokens: &'a [u32], finish_reason: &str) -> &'a [u32] {
    if finish_reason == "stop" {
        tokens.split_last().map_or(tokens, |(_, visible)| visible)
    } else {
        tokens
    }
}

/// Build the assistant message + finish_reason for one choice. Tool
/// parsing, validation, content-strip + refusal-classifier all live
/// here.
///
/// Deliberately NOT `async`: it awaits nothing, and marking pure CPU work as
/// async only hides where that work runs. If it ever grows expensive enough to
/// matter, that becomes a visible decision to move it to the blocking pool
/// rather than something already buried inside a future.
#[allow(clippy::too_many_arguments)]
fn build_choice_message(
    state: &AppState,
    req: &crate::ir::ChatRequest,
    response: &super::inference_types::InferenceResponse,
    reasoning_content_i: Option<String>,
    output_text_i: String,
    tools_active: bool,
    cwd_hint: Option<&str>,
    choice_idx: usize,
) -> ir::Choice {
    let _ = response; // currently only used for finish_reason.clone() below
    // Neutral locals — the wire annotations (URL citations) are derived
    // at encode time by the surfaces that emit them.
    let mut reasoning_content = reasoning_content_i;
    let mut msg_content: Option<String> = Some(output_text_i.clone());
    let mut msg_tool_calls: Option<Vec<tool_parser::ToolCall>> = None;
    let mut msg_refusal: Option<String> = None;
    let mut finish_reason_i = response.finish_reason.clone();

    if tools_active {
        if std::env::var("ATLAS_LOG_TOOL_RAW").as_deref() == Ok("1") {
            tracing::info!(
                target: "atlas::tool_debug",
                "raw pre-parse output (tools_active, choice {choice_idx}): {output_text_i:?}"
            );
        }
        // F7 (2026-05-26): also scan `reasoning_content_i` for tool calls.
        // When the model emits a `<tool_call>...</tool_call>` block INSIDE
        // its `<think>...</think>` reasoning, `decode_response_text` splits
        // at `</think>` and routes the tool call into reasoning_content,
        // hiding it from the post-`</think>` parser below — the tool call
        // is silently dropped (matches vLLM #39055 pattern). When found in
        // reasoning, hoist the calls back into the assistant message and
        // scrub the residual XML from the reasoning trace so it isn't
        // double-emitted to the client.
        let parser_name = state.tool_call_parser.as_ref().map(|parser| parser.name());
        let (hoisted_reasoning, hoisted_tool_calls) =
            extract_hoisted_tool_calls(reasoning_content.as_deref(), parser_name);
        if !hoisted_tool_calls.is_empty() {
            tracing::info!(
                "F7: hoisted {} tool-call(s) from inside <think> block (would have been silently dropped)",
                hoisted_tool_calls.len()
            );
            reasoning_content = hoisted_reasoning;
        }
        let (content, parsed_tool_calls) = tool_parser::parse_tool_calls(&output_text_i);
        let mut tool_calls_i = merge_hoisted_tool_calls(hoisted_tool_calls, parsed_tool_calls);
        if !tool_calls_i.is_empty() {
            let tools_ref = req.tools.clone();
            tool_parser::backfill_required_params(&mut tool_calls_i, &tools_ref);
            if state
                .tool_call_parser
                .as_ref()
                .is_some_and(|p| p.wants_typed_arguments())
            {
                tool_parser::coerce_all(&mut tool_calls_i, &tools_ref);
            }
            if let Some(cwd) = cwd_hint {
                tool_parser::normalize_paths(&mut tool_calls_i, cwd);
            }
            let validated = tool_parser::validate_tool_calls(tool_calls_i, &tools_ref);
            if !validated.errors.is_empty() {
                for err in &validated.errors {
                    tracing::warn!("Tool call validation error: {err}");
                }
            }
            // Strip orphan tool call XML tags + ```lang fences from content
            // (Qwen3-Coder pattern: emits markdown narration AND structured
            // tool_call for the same payload).
            let content = content.map(|mut c| {
                for tag in &["</parameter>", "</function>", "</tool_call>", "<tool_call>"] {
                    c = c.replace(tag, "");
                }
                while let Some(start) = c.find("<function=") {
                    let end = c[start..]
                        .find('>')
                        .map(|p| start + p + 1)
                        .unwrap_or(c.len());
                    c = format!("{}{}", &c[..start], &c[end..]);
                }
                while let Some(start) = c.find("```") {
                    let after_open = start + 3;
                    let Some(rel_close) = c[after_open..].find("```") else {
                        break;
                    };
                    let close_end = after_open + rel_close + 3;
                    c = format!("{}{}", &c[..start], &c[close_end..]);
                }
                c.trim().to_string()
            });
            msg_content = content;
            if !validated.valid.is_empty() {
                for tc in &validated.valid {
                    let p: String = tc.function.arguments.chars().take(120).collect();
                    let s = ["", "…"][usize::from(tc.function.arguments.len() > p.len())];
                    tracing::info!("Tool call: {}({p}{s})", tc.function.name);
                    crate::metrics::TOOL_CALLS_TOTAL.inc();
                }
                msg_tool_calls = Some(validated.valid);
                // A deadline cut outranks "tool_calls": the turn was
                // truncated, so a call parsed out of it may be partial and
                // the client must not treat it as a completed tool turn.
                if finish_reason_i != ir::FINISH_REASON_TIMEOUT {
                    finish_reason_i = "tool_calls".to_string();
                }
            }
        }
    }

    // Orphan tool-call markup: the model can emit a `<tool_call>…` block after
    // a normal answer even when no tool call is used (tools inactive, or tools
    // active but nothing valid parsed). The tools_active parser above only
    // scrubbed content when it produced a call, so any surviving markup here is
    // spurious — cut it from content so it does not reach the client.
    if msg_tool_calls.is_none() {
        msg_content = msg_content.map(|c| super::strip::strip_orphan_tool_markup(&c));
    }

    // Refusal classifier: when the model's assistant text opens with
    // a known refusal pattern AND no tool call fired, populate
    // `refusal` and null out `content` per the OpenAI spec.
    if msg_tool_calls.is_none()
        && let Some(content_text) = msg_content.as_deref()
        && let Some(refusal_sentence) = crate::refusal::detect(content_text)
    {
        msg_refusal = Some(refusal_sentence);
        msg_content = None;
    }

    // Validated wire tool calls → IR (arguments are serde-normalized
    // strings from the parser, so the parse here is lossless).
    let tool_calls: Vec<ir::message::ToolCall> = msg_tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(|tc| ir::message::ToolCall {
            id: tc.id,
            name: tc.function.name,
            arguments: serde_json::from_str(&tc.function.arguments)
                .unwrap_or_else(|_| serde_json::Value::Object(Default::default())),
        })
        .collect();

    ir::Choice {
        index: choice_idx,
        content: msg_content,
        reasoning: reasoning_content,
        tool_calls,
        refusal: msg_refusal,
        finish_reason: ir::FinishReason::from(finish_reason_i.as_str()),
        matched_stop: None, // caller fills
        logprobs: None,     // caller fills
    }
}

/// Recover tool calls from reasoning for legacy parsers that depend on F7.
/// Poolside v1 uses native interleaved reasoning: only post-reasoning content
/// is executable, so an envelope in the reasoning channel remains trace text.
fn extract_hoisted_tool_calls(
    reasoning_content: Option<&str>,
    parser_name: Option<&str>,
) -> (Option<String>, Vec<tool_parser::ToolCall>) {
    let Some(reasoning) = reasoning_content else {
        return (None, Vec::new());
    };
    if parser_name == Some("poolside_v1") {
        return (Some(reasoning.to_string()), Vec::new());
    }

    tool_parser::parse_tool_calls(reasoning)
}

/// Merge calls recovered from reasoning with calls in assistant content.
///
/// Some reasoning models emit the same native tool envelope immediately
/// before and after their final `</think>`. Prefer the content copy in that
/// case, while preserving intentional repeated calls within either channel.
fn merge_hoisted_tool_calls(
    mut hoisted: Vec<tool_parser::ToolCall>,
    parsed: Vec<tool_parser::ToolCall>,
) -> Vec<tool_parser::ToolCall> {
    hoisted.retain(|candidate| {
        !parsed.iter().any(|call| {
            call.function.name == candidate.function.name
                && call.function.arguments == candidate.function.arguments
        })
    });
    hoisted.extend(parsed);
    hoisted
}

/// Convert internal logprobs to OpenAI `ChoiceLogprobs` format.
fn build_logprobs(
    state: &AppState,
    response: &super::inference_types::InferenceResponse,
) -> Option<ir::ChoiceLogprobs> {
    if response.logprobs.is_empty() {
        return None;
    }
    Some(ir::ChoiceLogprobs {
        content: response
            .logprobs
            .iter()
            .map(|lp| {
                let token_str = state.tokenizer.decode(&[lp.token_id]).unwrap_or_default();
                ir::TokenLogprob {
                    token: token_str,
                    logprob: lp.logprob,
                    top: lp
                        .top
                        .iter()
                        .map(|&(tid, lp_val)| {
                            (state.tokenizer.decode(&[tid]).unwrap_or_default(), lp_val)
                        })
                        .collect(),
                }
            })
            .collect(),
    })
}

/// Core finalization: usage assembly, metrics, and the rate-limit
/// true-up. Returns the canonical response IR — wire encoding (plus
/// `store:`/`--dump` handling) happens in the per-surface encoders.
#[allow(clippy::too_many_arguments)]
fn finalize_response(
    state: Arc<AppState>,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    all_choices: Vec<ir::Choice>,
    total_completion_tokens: usize,
    first_ttft: f64,
    last_decode_time_ms: f64,
    total_reasoning_tokens: u32,
    total_cached_prompt_tokens: u32,
    prompt_len: usize,
) -> super::chat::ChatOutcome {
    let tokens_per_second = if last_decode_time_ms > 0.0 && total_completion_tokens > 0 {
        (total_completion_tokens.saturating_sub(1)) as f64 / (last_decode_time_ms / 1000.0)
    } else {
        0.0
    };
    let usage = ir::Usage {
        prompt_tokens: prompt_len,
        completion_tokens: total_completion_tokens,
        cached_prompt_tokens: total_cached_prompt_tokens as usize,
        reasoning_tokens: total_reasoning_tokens as usize,
        time_to_first_token_ms: first_ttft,
        response_tokens_per_second: tokens_per_second,
    };

    // REQUESTS_ACTIVE released by the caller's ActiveRequestGuard on return.
    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(prompt_len as u64);
    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(total_completion_tokens as u64);
    crate::metrics::TTFT_SECONDS
        .with_label_values(&[state.model_name.as_str()])
        .observe(first_ttft / 1000.0);

    // Rate-limit true-up. Middleware admitted with a conservative
    // reservation of `max_seq_len` tokens; refund the difference.
    if let Some(axum::extract::Extension(ref ctx)) = req_ctx {
        let actual = (prompt_len + total_completion_tokens) as u64;
        let refund = ctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            state.rate_limiter.refund_tokens(&ctx.identity, refund);
        }
    }

    super::chat::ChatOutcome::Blocking(Box::new(ir::ChatResponse {
        id: crate::ids::uuid_v4(),
        model: state.model_name.clone(),
        created: crate::ids::unix_timestamp(),
        choices: all_choices,
        usage,
    }))
}

#[cfg(test)]
mod tests {
    use super::{
        extract_hoisted_tool_calls, merge_hoisted_tool_calls, output_tokens_without_stop,
        split_at_first_think_end,
    };
    use crate::tool_parser::{FunctionCall, ToolCall};

    fn tool_call(id: &str, city: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "get_weather".into(),
                arguments: format!(r#"{{"city":"{city}"}}"#),
            },
        }
    }

    #[test]
    fn duplicate_call_across_reasoning_and_content_is_emitted_once() {
        let merged = merge_hoisted_tool_calls(
            vec![tool_call("reasoning", "Boston")],
            vec![tool_call("content", "Boston")],
        );

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].id, "content");
    }

    #[test]
    fn distinct_calls_across_reasoning_and_content_are_preserved() {
        let merged = merge_hoisted_tool_calls(
            vec![tool_call("reasoning", "Boston")],
            vec![tool_call("content", "Seattle")],
        );

        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn blocking_decode_excludes_terminal_stop_token() {
        let tokens = [10, 11, 24];

        assert_eq!(output_tokens_without_stop(&tokens, "stop"), &[10, 11]);
        assert_eq!(output_tokens_without_stop(&tokens, "length"), &tokens);
    }

    #[test]
    fn blocking_thinking_split_uses_first_end_token() {
        let tokens = [10, 19, 20, 19, 30];

        let split = split_at_first_think_end(&tokens, 19, true);

        assert_eq!(split, Some((&[10][..], &[20, 19, 30][..])));
        assert_eq!(split_at_first_think_end(&tokens, 19, false), None);
    }

    #[test]
    fn poolside_tool_call_in_reasoning_is_not_hoisted() {
        let reasoning = "plan <tool_call>write_file<arg_key>path</arg_key>\
            <arg_value>/tmp/x</arg_value></tool_call> more";

        let (preserved, calls) = extract_hoisted_tool_calls(Some(reasoning), Some("poolside_v1"));

        assert_eq!(preserved.as_deref(), Some(reasoning));
        assert!(calls.is_empty());
    }

    #[test]
    fn non_poolside_tool_call_in_reasoning_is_still_hoisted() {
        let reasoning =
            r#"plan <tool_call>{"name":"get_weather","arguments":{"city":"Boston"}}</tool_call>"#;

        let (_scrubbed, calls) = extract_hoisted_tool_calls(Some(reasoning), Some("hermes"));

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }
}
