// SPDX-License-Identifier: AGPL-3.0-only

//! Per-choice response decoding + assistant-message construction, split out
//! of `chat_blocking.rs` to keep that file ≤500 LoC.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use crate::AppState;
use crate::openai::ChatCompletionRequest;
use crate::tool_parser;

use super::super::inference_impl::{extract_thinking, strip_stop_sequences};

/// Decode `(reasoning_content, output_text)` from the scheduler's
/// response. When `enable_thinking=true`, split at the last `</think>`
/// token. When `enable_thinking=false`, decode all output_tokens as
/// content — mirrors streaming's `thinking_done = !enable_thinking`
/// init in chat_stream/state.rs and recovers the answer Qwen3.x emits
/// inside `<think>...</think>` when it ignores a closed-thinking
/// prefill (issue #40).
pub(super) fn decode_response_text(
    state: &AppState,
    response: &super::super::inference_types::InferenceResponse,
    enable_thinking: bool,
) -> (Option<String>, String) {
    if let Some(think_tok) = state.think_end_token_id {
        let last_think_pos = if enable_thinking {
            response.output_tokens.iter().rposition(|&t| t == think_tok)
        } else {
            None
        };
        if let Some(pos) = last_think_pos {
            let thinking_tokens = &response.output_tokens[..pos];
            let content_tokens = &response.output_tokens[pos + 1..];
            let reasoning = if !thinking_tokens.is_empty() {
                state
                    .tokenizer
                    .decode(thinking_tokens)
                    .ok()
                    .filter(|s| !s.trim().is_empty())
            } else {
                None
            };
            let content = state
                .tokenizer
                .decode(content_tokens)
                .unwrap_or_default()
                .trim_start()
                .to_string();
            return (reasoning, content);
        }
        let text = state
            .tokenizer
            .decode(&response.output_tokens)
            .unwrap_or_default();
        (None, text)
    } else {
        let text = state
            .tokenizer
            .decode(&response.output_tokens)
            .unwrap_or_default();
        extract_thinking(&text, enable_thinking, state.reasoning_parser.as_deref())
    }
}

/// Build the assistant message + finish_reason for one choice. Tool
/// parsing, validation, content-strip + refusal-classifier all live
/// here.
#[allow(clippy::too_many_arguments)]
pub(super) async fn build_choice_message(
    state: &AppState,
    req: &ChatCompletionRequest,
    response: &super::super::inference_types::InferenceResponse,
    reasoning_content_i: Option<String>,
    output_text_i: String,
    tools_active: bool,
    cwd_hint: Option<&str>,
    choice_idx: usize,
) -> (crate::openai::ChatMessage, String) {
    let _ = response; // currently only used for finish_reason.clone() below
    let mut message = crate::openai::ChatMessage {
        role: "assistant".to_string(),
        reasoning_content: reasoning_content_i.clone(),
        reasoning: reasoning_content_i,
        annotations: crate::citation::merged_annotations(&output_text_i),
        refusal: None,
        content: Some(output_text_i.clone()),
        tool_calls: None,
    };
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
        let (hoisted_reasoning, hoisted_tool_calls): (Option<String>, Vec<_>) =
            if let Some(ref rc) = message.reasoning_content {
                let (scrubbed, tcs) = tool_parser::parse_tool_calls(rc);
                (scrubbed, tcs)
            } else {
                (None, Vec::new())
            };
        if !hoisted_tool_calls.is_empty() {
            tracing::info!(
                "F7: hoisted {} tool-call(s) from inside <think> block (would have been silently dropped)",
                hoisted_tool_calls.len()
            );
            message.reasoning_content = hoisted_reasoning.clone();
            message.reasoning = hoisted_reasoning;
        }
        let (content, parsed_tool_calls) = tool_parser::parse_tool_calls(&output_text_i);
        let mut tool_calls_i = hoisted_tool_calls;
        tool_calls_i.extend(parsed_tool_calls);
        if !tool_calls_i.is_empty() {
            let tools_ref = req.tools.as_ref().cloned().unwrap_or_default();
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
            message.content = content;
            if !validated.valid.is_empty() {
                for tc in &validated.valid {
                    let p: String = tc.function.arguments.chars().take(120).collect();
                    let s = ["", "…"][usize::from(tc.function.arguments.len() > p.len())];
                    tracing::info!("Tool call: {}({p}{s})", tc.function.name);
                    crate::metrics::TOOL_CALLS_TOTAL.inc();
                }
                message.tool_calls = Some(validated.valid);
                finish_reason_i = "tool_calls".to_string();
            }
        }
    }

    // Refusal classifier: when the model's assistant text opens with
    // a known refusal pattern AND no tool call fired, populate
    // `message.refusal` and null out `content` per the OpenAI spec.
    if message.tool_calls.is_none()
        && let Some(content_text) = message.content.as_deref()
        && let Some(refusal_sentence) = crate::refusal::detect(content_text)
    {
        message.refusal = Some(refusal_sentence);
        message.content = None;
        message.annotations = None;
    }

    (message, finish_reason_i)
}
