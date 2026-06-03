// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[derive(Debug)]
pub enum LowerResponsesError {
    /// Request body was malformed (bad `input` shape, etc.).
    BadRequest(String),
    /// `previous_response_id` was set but the store had no entry. Maps to
    /// `400 invalid_request_error, code=response_not_found`.
    PriorNotFound(String),
}

impl LowerResponsesError {
    pub fn message(&self) -> &str {
        match self {
            Self::BadRequest(m) | Self::PriorNotFound(m) => m.as_str(),
        }
    }
}

/// Lower an OpenAI Responses-API request into a ChatCompletionRequest
/// so the existing chat-completions pipeline can satisfy it.
///
/// Prior-turn resume: when `previous_response_id` is set, look it up via
/// `resolve_prior` (the handler passes a closure that consults
/// [`crate::response_store::ResponseStore`]). The stored transcript is
/// prepended to the current input so the pipeline sees the full history.
/// Unknown / expired ids yield [`LowerResponsesError::PriorNotFound`].
pub fn lower_responses_to_chat(
    r: ResponsesRequest,
    resolve_prior: impl FnOnce(&str) -> Option<Vec<IncomingMessage>>,
) -> Result<ChatCompletionRequest, LowerResponsesError> {
    let mut messages: Vec<IncomingMessage> = Vec::new();

    // Prior-turn transcript (when resuming a conversation).
    if let Some(prior_id) = r.previous_response_id.as_deref() {
        match resolve_prior(prior_id) {
            Some(prior) => messages.extend(prior),
            None => {
                return Err(LowerResponsesError::PriorNotFound(format!(
                    "previous_response_id '{prior_id}' not found or expired"
                )));
            }
        }
    }

    if let Some(instr) = r.instructions.clone() {
        // Per the Responses spec, only the CURRENT turn's `instructions`
        // should apply. If we're resuming via previous_response_id, the
        // prior transcript may already contain a synthetic-system
        // message — drop it so the new instructions don't stack.
        // Also insert at position 0 so the chat template sees system
        // first (the conventional layout).
        messages.retain(|m| m.role != "system");
        messages.insert(0, IncomingMessage::synthetic_system(instr));
    }
    match &r.input {
        serde_json::Value::String(s) => {
            messages.push(IncomingMessage::synthetic_user_text(s.clone()));
        }
        serde_json::Value::Array(items) => {
            for it in items {
                if let Some(m) = IncomingMessage::from_responses_input_item(it) {
                    messages.push(m);
                }
            }
        }
        _ => {
            return Err(LowerResponsesError::BadRequest(
                "`input` must be a string or array of input items".into(),
            ));
        }
    }

    // Tools pass: separate function tools (supported) from built-in
    // hosted tools (not supported on a self-hosted inference server).
    // Built-ins each get their own error message so the client knows
    // exactly what's missing; function tools are parsed into
    // `ToolDefinition` via serde round-trip.
    let tools: Option<Vec<crate::tool_parser::ToolDefinition>> = match r.tools {
        None => None,
        Some(list) => {
            let mut parsed = Vec::with_capacity(list.len());
            for raw in list {
                let ty = raw.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match ty {
                    "function" | "" => {
                        // Responses API uses a flat `{type, name, description,
                        // parameters, strict}` shape for function tools, while
                        // chat-completions wraps the function fields inside a
                        // nested `function` object. Re-shape the flat form so
                        // the existing chat-format ToolDefinition deserializer
                        // accepts both.
                        let normalized = if raw.get("function").is_some() {
                            raw
                        } else if let Some(obj) = raw.as_object() {
                            let mut function = serde_json::Map::new();
                            for key in ["name", "description", "parameters", "strict"] {
                                if let Some(v) = obj.get(key) {
                                    function.insert(key.to_string(), v.clone());
                                }
                            }
                            serde_json::json!({
                                "type": "function",
                                "function": serde_json::Value::Object(function),
                            })
                        } else {
                            raw
                        };
                        match serde_json::from_value::<crate::tool_parser::ToolDefinition>(
                            normalized,
                        ) {
                            Ok(td) => parsed.push(td),
                            Err(e) => {
                                return Err(LowerResponsesError::BadRequest(format!(
                                    "invalid tool definition: {e}"
                                )));
                            }
                        }
                    }
                    builtin @ ("web_search"
                    | "web_search_preview"
                    | "file_search"
                    | "computer_use_preview"
                    | "code_interpreter"
                    | "image_generation"
                    | "mcp"
                    | "local_shell"
                    | "custom_tool") => {
                        return Err(LowerResponsesError::BadRequest(format!(
                            "built-in tool '{builtin}' is not supported by this server. Atlas serves inference only and does not ship hosted tools (web search, file search, code interpreter, computer use, image generation, MCP). Provide your own `function`-type tools instead."
                        )));
                    }
                    other => {
                        return Err(LowerResponsesError::BadRequest(format!(
                            "unknown tool type '{other}'. Supported types: 'function'."
                        )));
                    }
                }
            }
            if parsed.is_empty() {
                None
            } else {
                Some(parsed)
            }
        }
    };
    Ok(ChatCompletionRequest {
        model: r.model,
        messages,
        max_tokens: r.max_output_tokens.unwrap_or_else(default_max_tokens),
        temperature: r.temperature,
        top_k: None,
        top_p: r.top_p,
        top_n_sigma: None,
        min_p: None,
        repetition_penalty: None,
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        stream: r.stream,
        // Responses API has no token-IDs knob; lowered requests keep
        // the default off (PCND — no implicit behavior change).
        return_token_ids: false,
        enable_thinking: false,
        thinking: None,
        thinking_token_budget: None,
        reasoning: r.reasoning,
        chat_template_kwargs: None,
        tools,
        tool_choice: match r.tool_choice {
            None => None,
            Some(raw) => {
                let normalized = match raw.as_str() {
                    Some(_) => raw,
                    None => match raw.as_object() {
                        Some(obj) if obj.get("function").is_some() => raw,
                        Some(obj)
                            if obj.get("type").and_then(|v| v.as_str()) == Some("function")
                                && obj.get("name").is_some() =>
                        {
                            let mut function = serde_json::Map::new();
                            function.insert(
                                "name".to_string(),
                                obj.get("name").cloned().unwrap_or(serde_json::Value::Null),
                            );
                            serde_json::json!({
                                "type": "function",
                                "function": serde_json::Value::Object(function),
                            })
                        }
                        _ => raw,
                    },
                };
                match serde_json::from_value::<crate::tool_parser::ToolChoice>(normalized) {
                    Ok(tc) => Some(tc),
                    Err(e) => {
                        return Err(LowerResponsesError::BadRequest(format!(
                            "invalid tool_choice: {e}"
                        )));
                    }
                }
            }
        },
        stop: Vec::new(),
        response_format: None,
        min_tokens: 0,
        seed: None,
        logprobs: None,
        top_logprobs: None,
        timeout: None,
        n: 1,
        stream_options: None,
        parallel_tool_calls: None,
        verbosity: None,
        service_tier: r.service_tier,
        store: r.store,
        metadata: r.metadata,
        safety_identifier: None,
        prompt_cache_key: None,
        user: None,
        modalities: None,
        audio: None,
        prediction: None,
        web_search_options: None,
        reasoning_effort: None,
    })
}

/// UUID v4 generation using OS randomness (no external crate needed).
pub(crate) fn uuid_v4() -> String {
    let mut bytes = [0u8; 16];
    // Use getrandom via std (available since Rust 1.36)
    if let Ok(()) = getrandom(&mut bytes) {
        // Set version 4 (bits 48-51) and variant 1 (bits 64-65)
        bytes[6] = (bytes[6] & 0x0F) | 0x40; // version 4
        bytes[8] = (bytes[8] & 0x3F) | 0x80; // variant 1
    } else {
        // Fallback: nanosecond timestamp (unique but not random)
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        bytes = t.to_le_bytes();
    }
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15],
    )
}
