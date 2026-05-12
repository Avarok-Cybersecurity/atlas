// SPDX-License-Identifier: AGPL-3.0-only

use super::helpers::*;
use super::types::*;

/// Audit the Anthropic→OpenAI translation for structural drift.
///
/// We don't do a full inverse round-trip (that would be a separate
/// translator); instead we check three count-level invariants that
/// compound across long agent sessions when violated:
///
///   1. Every Anthropic user/assistant message produces at least one
///      OpenAI message (a user-with-tool_result splits into 1+
///      `role="tool"` messages plus an optional `role="user"`).
///   2. Every Anthropic `tool_use` content block becomes a
///      `tool_calls[i]` entry on its assistant message.
///   3. The system field (text or blocks) becomes the first OpenAI
///      message with `role="system"`, whose text is non-empty
///      whenever the Anthropic system field had any non-billing text.
///
/// On any mismatch: increment the drift metric, optionally log the
/// diff (gated by `ATLAS_DEBUG_TRANSLATION_DRIFT`).
pub(super) fn audit_translation_drift(req: &MessagesRequest, chat_json: &serde_json::Value) {
    let mut anomalies: Vec<String> = Vec::new();

    // Count Anthropic-side input.
    let anth_msgs = req.messages.len();
    let mut anth_tool_uses: usize = 0;
    let mut anth_tool_results: usize = 0;
    for m in &req.messages {
        if let AnthropicContent::Blocks(blocks) = &m.content {
            for b in blocks {
                match b {
                    ContentBlock::ToolUse { .. } => anth_tool_uses += 1,
                    ContentBlock::ToolResult { .. } => anth_tool_results += 1,
                    _ => {}
                }
            }
        }
    }
    let anth_system_nonempty = req
        .system
        .as_ref()
        .map(|s| match s {
            SystemContent::Text(t) => !t.trim().is_empty(),
            SystemContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| b.text.as_ref())
                .any(|t| !t.starts_with("x-anthropic-") && !t.trim().is_empty()),
        })
        .unwrap_or(false);

    // Count OpenAI-side output.
    let openai_msgs = chat_json
        .get("messages")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    let openai_tool_calls = chat_json
        .get("messages")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("tool_calls").and_then(|t| t.as_array()))
                .map(|tcs| tcs.len())
                .sum::<usize>()
        })
        .unwrap_or(0);
    let openai_role_tool = chat_json
        .get("messages")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
                .count()
        })
        .unwrap_or(0);
    let openai_first_role = chat_json
        .get("messages")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str());
    let openai_first_content = chat_json
        .get("messages")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str());

    // Invariant 1: Anthropic message count ≤ OpenAI message count
    // (split-on-tool_result can only INCREASE).
    if openai_msgs < anth_msgs {
        anomalies.push(format!(
            "message-count regressed: anthropic={anth_msgs} → openai={openai_msgs}"
        ));
    }

    // Invariant 2: tool_use block count == tool_calls count.
    if anth_tool_uses != openai_tool_calls {
        anomalies.push(format!(
            "tool_use mismatch: anthropic blocks={anth_tool_uses}, openai tool_calls={openai_tool_calls}"
        ));
    }

    // Invariant 3: tool_result blocks ≤ role=tool messages.
    if anth_tool_results > openai_role_tool {
        anomalies.push(format!(
            "tool_result lost: anthropic blocks={anth_tool_results}, openai role=tool msgs={openai_role_tool}"
        ));
    }

    // Invariant 4: non-empty system field → first openai message is system.
    if anth_system_nonempty
        && (openai_first_role != Some("system")
            || openai_first_content.is_none_or(|s| s.trim().is_empty()))
    {
        anomalies.push(
            "non-empty Anthropic system did NOT produce a non-empty system message".to_string(),
        );
    }

    if anomalies.is_empty() {
        return;
    }

    crate::metrics::ANTHROPIC_TRANSLATION_DRIFTS.inc_by(anomalies.len() as u64);
    if std::env::var("ATLAS_DEBUG_TRANSLATION_DRIFT").is_ok() {
        for a in &anomalies {
            tracing::warn!(
                anth_msgs,
                anth_tool_uses,
                anth_tool_results,
                openai_msgs,
                openai_tool_calls,
                openai_role_tool,
                "anthropic translation drift: {a}"
            );
        }
    }
}

pub(super) fn anthropic_to_chat_request_json(req: &MessagesRequest) -> serde_json::Value {
    let mut messages: Vec<serde_json::Value> = Vec::with_capacity(req.messages.len() + 1);

    // System message
    if let Some(sys) = &req.system {
        let sys_text = match sys {
            SystemContent::Text(s) => s.clone(),
            SystemContent::Blocks(blocks) => blocks
                .iter()
                .filter(|b| {
                    b.block_type == "text"
                        && !b.text.as_deref().unwrap_or("").starts_with("x-anthropic-")
                })
                .filter_map(|b| b.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        };
        if !sys_text.is_empty() {
            messages.push(serde_json::json!({
                "role": "system",
                "content": sys_text,
            }));
        }
    }

    // Conversation history
    for m in &req.messages {
        let role = match m.role.as_str() {
            "assistant" => "assistant",
            _ => "user",
        };
        match &m.content {
            AnthropicContent::Text(s) => {
                messages.push(serde_json::json!({"role": role, "content": s}));
            }
            AnthropicContent::Blocks(blocks) => {
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_calls: Vec<serde_json::Value> = Vec::new();
                let mut tool_results: Vec<(String, String)> = Vec::new();
                for b in blocks {
                    match b {
                        ContentBlock::Text { text } => text_parts.push(text.clone()),
                        ContentBlock::ToolUse { id, name, input } => {
                            tool_calls.push(serde_json::json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": input.to_string(),
                                },
                            }));
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            let text = content.as_ref().map(|c| c.to_text()).unwrap_or_default();
                            // F6 (2026-04-26): when Anthropic's
                            // is_error flag is set, prepend an
                            // explicit `[tool error]\n` marker so the
                            // model has a structural signal that the
                            // tool call failed. Without this, the
                            // model has hallucinated success after
                            // `Exit code 127\ncargo: command not
                            // found` (observed in dump fix26 seq 27).
                            let prefixed = if is_error.unwrap_or(false) {
                                format!("[tool error]\n{text}")
                            } else {
                                text
                            };
                            tool_results.push((tool_use_id.clone(), prefixed));
                        }
                        ContentBlock::Thinking { .. } | ContentBlock::Unknown => {}
                    }
                }
                let text_content = text_parts.join("");
                if role == "assistant" {
                    let mut msg = serde_json::json!({
                        "role": "assistant",
                        "content": text_content,
                    });
                    if !tool_calls.is_empty() {
                        msg["tool_calls"] = serde_json::Value::Array(tool_calls);
                    }
                    messages.push(msg);
                } else {
                    if !text_content.is_empty() {
                        messages.push(serde_json::json!({
                            "role": "user",
                            "content": text_content,
                        }));
                    }
                    for (tool_use_id, text) in tool_results {
                        messages.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": text,
                        }));
                    }
                }
            }
        }
    }

    // Tools array
    let tools_json = req.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                })
            })
            .collect::<Vec<_>>()
    });

    // tool_choice mapping
    let tool_choice_json: Option<serde_json::Value> =
        req.tool_choice
            .as_ref()
            .map(|tc| match tc.choice_type.as_str() {
                "any" => serde_json::json!("required"),
                "auto" => serde_json::json!("auto"),
                "none" => serde_json::json!("none"),
                "tool" => {
                    if let Some(name) = &tc.name {
                        serde_json::json!({
                            "type": "function",
                            "function": { "name": name },
                        })
                    } else {
                        serde_json::json!("auto")
                    }
                }
                _ => serde_json::json!("auto"),
            });

    let mut chat = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "max_tokens": req.max_tokens,
        "stream": req.stream,
    });
    if let Some(t) = req.temperature {
        chat["temperature"] = serde_json::json!(t);
    }
    if let Some(k) = req.top_k {
        chat["top_k"] = serde_json::json!(k);
    }
    if let Some(p) = req.top_p {
        chat["top_p"] = serde_json::json!(p);
    }
    if !req.stop_sequences.is_empty() {
        chat["stop"] = serde_json::json!(req.stop_sequences);
    }
    if let Some(tools) = tools_json {
        chat["tools"] = serde_json::Value::Array(tools);
    }
    if let Some(tc) = tool_choice_json {
        chat["tool_choice"] = tc;
    }

    // Anthropic thinking config — preserve via the `thinking` field that
    // ChatCompletionRequest already accepts (vLLM-compatible shape).
    if let Some(thinking) = &req.thinking {
        let mut t = serde_json::json!({"type": thinking.thinking_type});
        if let Some(b) = thinking.budget_tokens {
            t["budget_tokens"] = serde_json::json!(b);
        }
        chat["thinking"] = t;
    }

    chat
}

/// Translate a non-streaming chat-completion JSON body into Anthropic's
/// `MessagesResponse` shape. Reads the body via untyped `serde_json::Value`
/// so we don't need ChatCompletionResponse to derive Deserialize.
pub(super) fn chat_to_anthropic_response(
    chat_value: &serde_json::Value,
    model: String,
) -> MessagesResponse {
    let id = chat_value
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| format!("msg_{}", s.trim_start_matches("chatcmpl-")))
        .unwrap_or_else(|| "msg_unknown".to_string());

    let usage = chat_value.get("usage").cloned().unwrap_or_default();
    let input_tokens = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let output_tokens = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let choice = chat_value
        .get("choices")
        .and_then(|c| c.get(0))
        .cloned()
        .unwrap_or_default();
    let msg = choice.get("message").cloned().unwrap_or_default();
    let finish_reason = choice
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("stop");

    let mut content: Vec<ResponseBlock> = Vec::new();

    // Reasoning → thinking block
    let reasoning = msg
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .or_else(|| msg.get("reasoning").and_then(|v| v.as_str()))
        .filter(|s| !s.is_empty());
    if let Some(r) = reasoning {
        content.push(ResponseBlock::Thinking {
            thinking: r.to_string(),
        });
    }

    // Text content
    if let Some(text) = msg.get("content").and_then(|v| v.as_str())
        && !text.is_empty()
    {
        content.push(ResponseBlock::Text {
            text: text.to_string(),
        });
    }

    // Tool calls
    if let Some(tcs) = msg.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tcs {
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let function = tc.get("function").cloned().unwrap_or_default();
            let name = function
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args_str = function
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let input: serde_json::Value = serde_json::from_str(args_str)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            content.push(ResponseBlock::ToolUse { id, name, input });
        }
    }

    MessagesResponse {
        id,
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model,
        stop_reason: Some(convert_stop_reason(finish_reason).to_string()),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens,
            output_tokens,
        },
    }
}
