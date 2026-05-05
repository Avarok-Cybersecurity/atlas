// SPDX-License-Identifier: AGPL-3.0-only

//! Tokenizer wrapper using HuggingFace tokenizers + minijinja chat template.
//!
//! Loads the model's official Jinja template from `tokenizer_config.json` and
//! renders it with minijinja for byte-exact alignment with the model's training
//! format. No fallback — if there's no Jinja template, the model is misconfigured.

use anyhow::Result;
use tokenizers::Tokenizer;

/// F76 (2026-04-29): pre-parse `tool_calls[*].function.arguments` from
/// OpenAI's wire format (JSON-encoded string) into the JSON value the
/// model's chat template expects. MiniMax M2.7's template iterates
/// `tool_call.function.arguments.items()` which crashes on a string.
/// We rebuild the message list with parsed arguments where present,
/// leaving every other field untouched. Returns a fresh Vec rather
/// than mutating the caller's slice.
fn normalize_tool_call_arguments(messages: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut total_parsed = 0usize;
    let mut total_seen = 0usize;
    let out: Vec<_> = messages
        .iter()
        .map(|msg| {
            let mut msg = msg.clone();
            let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) else {
                return msg;
            };
            for tc in tool_calls.iter_mut() {
                let Some(function) = tc.get_mut("function") else {
                    continue;
                };
                let Some(args) = function.get_mut("arguments") else {
                    continue;
                };
                total_seen += 1;
                let parsed_owned = if let Some(s) = args.as_str() {
                    serde_json::from_str::<serde_json::Value>(s).ok()
                } else {
                    None
                };
                if let Some(parsed) = parsed_owned {
                    *args = parsed;
                    total_parsed += 1;
                }
                // If parse fails or args wasn't a string, leave as-is —
                // template may handle via tojson, or surface the
                // original error for the operator.
            }
            msg
        })
        .collect();
    if total_seen > 0 {
        tracing::debug!(
            "F76 normalize: {}/{} tool_call arguments parsed string→dict",
            total_parsed,
            total_seen,
        );
    }
    out
}

/// Wraps a HuggingFace tokenizer with Jinja chat template support.
mod chat_impl;
mod jinja_helpers;

pub struct ChatTokenizer {
    tokenizer: Tokenizer,
    eos_token_id: u32,
    supports_thinking: bool,
    /// Compiled Jinja chat template (from tokenizer_config.json).
    #[allow(dead_code)]
    chat_template: String,
    /// Precompiled minijinja environment (avoids re-creating + re-compiling each call).
    jinja_env: minijinja::Environment<'static>,
    /// OpenAI-variant template: gates historical `<think>` wrappers on enable_thinking.
    /// Falls back to jinja_env if no openai/ variant exists.
    openai_jinja_env: Option<minijinja::Environment<'static>>,
}

/// Wrapper around tokenizers::DecodeStream that hides the generic parameters.
/// O(1) per step vs O(n) for full re-decode.
pub struct StreamingDecoder<'a> {
    inner: tokenizers::DecodeStream<
        'a,
        tokenizers::models::ModelWrapper,
        tokenizers::normalizers::NormalizerWrapper,
        tokenizers::pre_tokenizers::PreTokenizerWrapper,
        tokenizers::processors::PostProcessorWrapper,
        tokenizers::decoders::DecoderWrapper,
    >,
}

impl StreamingDecoder<'_> {
    /// Feed one token. Returns Some(text) when valid UTF-8 is ready.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        self.inner
            .step(id)
            .map_err(|e| anyhow::anyhow!("Streaming decode error: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalize_tool_call_arguments_parses_string_to_dict() {
        // The shape opencode sends back on the second turn: assistant
        // message with tool_calls whose function.arguments is a JSON
        // string. F76: must round-trip into a dict for MiniMax's
        // template `_args.items()` to work.
        let messages = vec![json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": "{\"command\":\"mkdir -p /tmp/x\",\"description\":\"make dir\"}"
                }
            }]
        })];
        let normalized = normalize_tool_call_arguments(&messages);
        let args = &normalized[0]["tool_calls"][0]["function"]["arguments"];
        assert!(args.is_object(), "expected dict, got {args:?}");
        assert_eq!(args["command"], "mkdir -p /tmp/x");
        assert_eq!(args["description"], "make dir");
    }

    #[test]
    fn normalize_tool_call_arguments_leaves_non_tool_messages_alone() {
        let messages = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hello"}),
        ];
        let normalized = normalize_tool_call_arguments(&messages);
        assert_eq!(normalized, messages);
    }

    #[test]
    fn normalize_tool_call_arguments_passes_through_already_dict() {
        // Some clients send args pre-parsed as a dict — must not double-encode.
        let messages = vec![json!({
            "role": "assistant",
            "tool_calls": [{
                "function": {"name": "bash", "arguments": {"command": "ls"}}
            }]
        })];
        let normalized = normalize_tool_call_arguments(&messages);
        assert_eq!(
            normalized[0]["tool_calls"][0]["function"]["arguments"]["command"],
            "ls"
        );
    }

    /// F76 integration: render the actual MiniMax M2.7 chat template
    /// with a second-turn shape (assistant has tool_calls with string
    /// args). Without F76 this errors with `unknown method: map has
    /// no method named items` on line 112.
    #[test]
    fn render_minimax_template_with_string_tool_call_args() {
        let template_path = "/workspace/.cache/huggingface/hub/models--lukealonso--MiniMax-M2.7-NVFP4/snapshots/ba6a625013cdacdc560f6203d177c0f27d41775e/chat_template.jinja";
        let Ok(template) = std::fs::read_to_string(template_path) else {
            eprintln!("MiniMax template not on disk; skipping");
            return;
        };
        let env = super::jinja_helpers::build_jinja_env(&template).expect("template compiles");
        let tmpl = env.get_template("chat").unwrap();
        // The exact wire shape opencode sends back on turn 2.
        let messages = vec![
            json!({"role": "user", "content": "List /tmp"}),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": "{\"command\":\"ls -la /tmp\"}"
                    }
                }]
            }),
            json!({"role": "tool", "tool_call_id": "call_0", "content": "total 0"}),
            json!({"role": "user", "content": "Now uname -r"}),
        ];
        let normalized = normalize_tool_call_arguments(&messages);
        let messages_val = minijinja::Value::from_serialize(&normalized);
        let ctx = minijinja::context! {
            messages => messages_val,
            tools => minijinja::Value::UNDEFINED,
            add_generation_prompt => true,
            enable_thinking => true,
            reasoning_effort => "high",
            disable_tool_steering => false,
            add_vision_id => false,
        };
        let rendered = tmpl
            .render(ctx)
            .expect("F76 must keep MiniMax template from raising on second-turn");
        // Sanity check: rendered output should contain the bash invoke
        // with command parameter — the items() iteration produced output.
        assert!(
            rendered.contains("<invoke name=\"bash\">"),
            "expected `<invoke name=\"bash\">` in render: {rendered}"
        );
        assert!(
            rendered.contains("<parameter name=\"command\">"),
            "expected `<parameter name=\"command\">` from .items() iteration: {rendered}"
        );
        assert!(
            rendered.contains("ls -la /tmp"),
            "expected the parsed command value in render: {rendered}"
        );
    }

    #[test]
    fn normalize_tool_call_arguments_invalid_json_string_left_alone() {
        // If args is a string but not valid JSON, leave as-is so the
        // template either coerces via tojson or the operator sees the
        // original error.
        let messages = vec![json!({
            "role": "assistant",
            "tool_calls": [{
                "function": {"name": "bash", "arguments": "not valid json {"}
            }]
        })];
        let normalized = normalize_tool_call_arguments(&messages);
        assert_eq!(
            normalized[0]["tool_calls"][0]["function"]["arguments"],
            "not valid json {"
        );
    }
}
