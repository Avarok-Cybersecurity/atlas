// SPDX-License-Identifier: AGPL-3.0-only

//! Per-choice assembly for the blocking `/v1/chat/completions` path:
//! tool-call parse/validate/coerce, refusal classification, and the
//! logprobs conversion. Exact piecewise copy out of `chat_blocking.rs`
//! (2026-08-09) to keep that file under the 500 LoC cap; behaviour
//! unchanged.

#![allow(clippy::too_many_arguments)]

use crate::AppState;
use crate::ir;
use crate::tool_parser;

use super::chat_blocking::{extract_hoisted_tool_calls, merge_hoisted_tool_calls};

/// Build the assistant message + finish_reason for one choice. Tool
/// parsing, validation, content-strip + refusal-classifier all live
/// here.
///
/// Deliberately NOT `async`: it awaits nothing, and marking pure CPU work as
/// async only hides where that work runs. If it ever grows expensive enough to
/// matter, that becomes a visible decision to move it to the blocking pool
/// rather than something already buried inside a future.
pub(super) fn build_choice_message(
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

    // Permanent diagnostic (RUST_LOG=...,spark::api::chat_blocking_choice=debug):
    // the choice's final decoded text, before tool parsing. The A/B that
    // pinned the qwen3.8 spec-decode divergence needed byte-exact response
    // texts from both legs, and nothing on the serve side logged them.
    tracing::debug!("chat response (choice {choice_idx}): {output_text_i:?}");
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
        let promote_bare_names = state
            .tool_call_parser
            .as_ref()
            .is_some_and(|p| p.promotes_bare_call_names());
        let (content, parsed_tool_calls) = if promote_bare_names {
            tool_parser::parse_tool_calls_promoting_bare_names(&output_text_i)
        } else {
            tool_parser::parse_tool_calls(&output_text_i)
        };
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

/// Convert internal logprobs to OpenAI `ChoiceLogprobs` format.
pub(super) fn build_logprobs(
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
