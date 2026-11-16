// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /apply-template`: render the chat template, return the prompt text.
//!
//! llama.cpp's route runs the same message parsing as `/v1/chat/completions`
//! but stops before inference, so a client can edit the prompt (prefill the
//! answer, say) and send it to `/completion`. Atlas renders through the same
//! tokenizer-side renderer `/v1/chat/completions` uses, without encoding.

use serde_json::{Map, Value, json};

use super::error::CompatError;

/// A parsed `/apply-template` request.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ApplyTemplateRequest {
    pub messages: Vec<Value>,
    pub tools: Option<Vec<Value>>,
    /// From `chat_template_kwargs.enable_thinking`; `None` leaves the server's
    /// own default for the model.
    pub enable_thinking: Option<bool>,
    /// From `chat_template_kwargs.reasoning_effort`.
    pub reasoning_effort: Option<String>,
}

/// Flatten a message `content` to the string the template renders: a string
/// stays, an array of `{"type":"text","text":...}` parts is joined, and any
/// non-text part is refused.
fn flatten_content(content: &Value, index: usize) -> Result<Value, CompatError> {
    match content {
        Value::Null | Value::String(_) => Ok(content.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                match (part.get("type").and_then(Value::as_str), part.get("text")) {
                    (Some("text"), Some(Value::String(t))) => out.push_str(t),
                    (Some("text"), _) => {
                        return Err(CompatError::invalid(format!(
                            "messages[{index}].content text parts need a string 'text'"
                        )));
                    }
                    (Some(other), _) => {
                        return Err(CompatError::unsupported(
                            &format!("messages[{index}].content"),
                            &format!("'{other}' content parts are not rendered by /apply-template"),
                        ));
                    }
                    _ => {
                        return Err(CompatError::invalid(format!(
                            "messages[{index}].content parts must be {{\"type\":\"text\",\"text\":...}}"
                        )));
                    }
                }
            }
            Ok(Value::String(out))
        }
        _ => Err(CompatError::invalid(format!(
            "messages[{index}].content must be a string or an array of text parts"
        ))),
    }
}

/// Parse the body: `messages` (required), `tools`, `chat_template_kwargs`,
/// `add_generation_prompt`.
pub(crate) fn parse_apply_template(
    body: &Map<String, Value>,
) -> Result<ApplyTemplateRequest, CompatError> {
    let messages = match body.get("messages") {
        Some(Value::Array(a)) if !a.is_empty() => a,
        Some(Value::Array(_)) => {
            return Err(CompatError::invalid("'messages' must not be empty"));
        }
        _ => {
            return Err(CompatError::invalid(
                "'messages' is required and must be an array",
            ));
        }
    };
    let mut out = Vec::with_capacity(messages.len());
    for (i, m) in messages.iter().enumerate() {
        let Some(obj) = m.as_object() else {
            return Err(CompatError::invalid(format!(
                "messages[{i}] must be an object"
            )));
        };
        if !obj.get("role").is_some_and(Value::is_string) {
            return Err(CompatError::invalid(format!(
                "messages[{i}].role is required and must be a string"
            )));
        }
        let mut obj = obj.clone();
        if let Some(c) = obj.get("content") {
            let flat = flatten_content(c, i)?;
            obj.insert("content".into(), flat);
        }
        out.push(Value::Object(obj));
    }

    // Atlas always renders the generation prompt that opens the assistant turn.
    if matches!(body.get("add_generation_prompt"), Some(Value::Bool(false))) {
        return Err(CompatError::unsupported(
            "add_generation_prompt",
            "the generation prompt is always appended; only true can be honored",
        ));
    }

    let tools = match body.get("tools") {
        None | Some(Value::Null) => None,
        Some(Value::Array(a)) if a.is_empty() => None,
        Some(Value::Array(a)) => Some(a.clone()),
        Some(_) => return Err(CompatError::invalid("'tools' must be an array")),
    };

    let mut enable_thinking = None;
    let mut reasoning_effort = None;
    match body.get("chat_template_kwargs") {
        None | Some(Value::Null) => {}
        Some(Value::Object(kw)) => {
            for (k, v) in kw {
                match (k.as_str(), v) {
                    ("enable_thinking", Value::Bool(b)) => enable_thinking = Some(*b),
                    ("reasoning_effort", Value::String(s)) => reasoning_effort = Some(s.clone()),
                    ("enable_thinking" | "reasoning_effort", _) => {
                        return Err(CompatError::invalid(format!(
                            "chat_template_kwargs.{k} has the wrong type"
                        )));
                    }
                    _ => {
                        return Err(CompatError::unsupported(
                            &format!("chat_template_kwargs.{k}"),
                            "only enable_thinking and reasoning_effort are passed to the template",
                        ));
                    }
                }
            }
        }
        Some(_) => {
            return Err(CompatError::invalid(
                "'chat_template_kwargs' must be an object",
            ));
        }
    }

    Ok(ApplyTemplateRequest {
        messages: out,
        tools,
        enable_thinking,
        reasoning_effort,
    })
}

/// The response body: `{"prompt": "<rendered text>"}`.
pub(crate) fn apply_template_response(prompt: String) -> Value {
    json!({ "prompt": prompt })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: Value) -> Result<ApplyTemplateRequest, CompatError> {
        parse_apply_template(v.as_object().expect("object"))
    }

    #[test]
    fn messages_in_prompt_out() {
        let r = parse(json!({"messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"}
        ]}))
        .unwrap();
        assert_eq!(r.messages.len(), 2);
        assert_eq!(r.messages[1]["content"], "hi");
        assert!(r.tools.is_none() && r.enable_thinking.is_none());
        assert_eq!(apply_template_response("P".into()), json!({"prompt": "P"}));
    }

    #[test]
    fn text_parts_are_joined_and_other_parts_refused() {
        let r = parse(json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}
        ]}]}))
        .unwrap();
        assert_eq!(r.messages[0]["content"], "ab");

        let err = parse(json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "x"}}
        ]}]}))
        .expect_err("images are not rendered here");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("image_url"));
    }

    #[test]
    fn tool_call_fields_pass_through_untouched() {
        let r = parse(json!({"messages": [
            {"role": "assistant", "content": null, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"}
        ]}))
        .unwrap();
        assert_eq!(r.messages[0]["tool_calls"][0]["id"], "1");
        assert_eq!(r.messages[1]["tool_call_id"], "1");
    }

    #[test]
    fn bad_messages_are_400s() {
        for bad in [
            json!({}),
            json!({"messages": []}),
            json!({"messages": "hi"}),
            json!({"messages": ["hi"]}),
            json!({"messages": [{"content": "no role"}]}),
            json!({"messages": [{"role": 5}]}),
            json!({"messages": [{"role": "user", "content": 5}]}),
        ] {
            let err = parse(bad.clone()).expect_err(&bad.to_string());
            assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST, "{bad}");
        }
    }

    #[test]
    fn template_kwargs_are_honored_or_refused() {
        let r = parse(json!({
            "messages": [{"role": "user", "content": "x"}],
            "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "low"}
        }))
        .unwrap();
        assert_eq!(r.enable_thinking, Some(true));
        assert_eq!(r.reasoning_effort.as_deref(), Some("low"));

        let err = parse(json!({
            "messages": [{"role": "user", "content": "x"}],
            "chat_template_kwargs": {"custom_flag": 1}
        }))
        .expect_err("unknown kwargs must not be dropped");
        assert!(err.message.contains("custom_flag"));
    }

    #[test]
    fn add_generation_prompt_false_is_refused() {
        let msgs = json!([{"role": "user", "content": "x"}]);
        let err = parse(json!({"messages": msgs, "add_generation_prompt": false}))
            .expect_err("cannot omit the generation prompt");
        assert!(err.message.contains("add_generation_prompt"));
        assert!(parse(json!({"messages": msgs, "add_generation_prompt": true})).is_ok());
    }

    #[test]
    fn tools_are_forwarded_and_empty_means_none() {
        let msgs = json!([{"role": "user", "content": "x"}]);
        let r = parse(json!({"messages": msgs, "tools": [{"type": "function"}]})).unwrap();
        assert_eq!(r.tools.map(|t| t.len()), Some(1));
        let r = parse(json!({"messages": msgs, "tools": []})).unwrap();
        assert!(r.tools.is_none());
        assert!(parse(json!({"messages": msgs, "tools": "x"})).is_err());
    }
}
