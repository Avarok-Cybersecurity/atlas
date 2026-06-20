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

    fn render_minimax_openai_template(
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
    ) -> String {
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../jinja-templates/openai/minimax_m2.jinja"
        );
        let raw = std::fs::read_to_string(template_path)
            .expect("bundled MiniMax OpenAI template must be present in the repo");
        let converted = super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
        let env = super::jinja_helpers::build_jinja_env(&converted).expect("template compiles");
        let tmpl = env.get_template("chat").unwrap();
        let messages_for_render = normalize_tool_call_arguments(messages);
        let messages_val = minijinja::Value::from_serialize(&messages_for_render);
        let tools_val = tools.map(minijinja::Value::from_serialize);
        let reasoning_effort: minijinja::Value = if enable_thinking {
            "high".into()
        } else {
            "none".into()
        };
        let ctx = minijinja::context! {
            messages => messages_val,
            tools => tools_val.unwrap_or(minijinja::Value::UNDEFINED),
            add_generation_prompt => true,
            enable_thinking => enable_thinking,
            reasoning_effort => reasoning_effort,
            disable_tool_steering => false,
            add_vision_id => false,
        };
        tmpl.render(ctx).expect("template renders")
    }

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
    fn render_minimax_openai_template_closes_think_prompt_when_disabled() {
        let messages = vec![json!({"role": "user", "content": "Reply with exactly: OK"})];
        let rendered = render_minimax_openai_template(&messages, None, false);
        assert!(
            rendered.ends_with("]~b]ai\n<think>\n\n</think>\n\n"),
            "expected closed-thinking assistant generation prompt: {rendered}"
        );
        let generation_tail = rendered
            .rsplit_once("]~b]ai\n")
            .map(|(_, tail)| tail)
            .expect("assistant generation prompt is present");
        assert_eq!(
            generation_tail, "<think>\n\n</think>\n\n",
            "disabled thinking must not leave the model inside <think>: {rendered}"
        );
    }

    #[test]
    fn render_minimax_openai_template_opens_think_prompt_when_enabled() {
        let messages = vec![json!({"role": "user", "content": "Think before answering"})];
        let rendered = render_minimax_openai_template(&messages, None, true);
        assert!(
            rendered.ends_with("]~b]ai\n<think>\n"),
            "expected thinking assistant generation prompt: {rendered}"
        );
    }

    #[test]
    fn render_minimax_openai_template_omits_think_prompt_with_tools_when_disabled() {
        let messages = vec![json!({"role": "user", "content": "List the current directory"})];
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        })];
        let rendered = render_minimax_openai_template(&messages, Some(&tools), false);
        assert!(
            rendered.contains("<tools>"),
            "expected tool schema block in render: {rendered}"
        );
        assert!(
            rendered.contains("<minimax:tool_call>"),
            "expected MiniMax tool-call instructions in render: {rendered}"
        );
        assert!(
            rendered.ends_with("]~b]ai\n<think>\n\n</think>\n\n"),
            "tool-active disabled-thinking requests must use a closed-thinking assistant prompt: {rendered}"
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

    /// Regression: Gemma-4's bundled template calls `text.split('<channel|>')`
    /// inside its `strip_thinking` macro. minijinja has no `.split()` *method*
    /// on strings, so before the unknown-method bridge every assistant
    /// (model-role) turn raised `string has no method named split` and the
    /// whole chat request 400'd. A null-content tool message is part of the
    /// same conversation shape (coherence test "null content / tool role").
    #[test]
    fn render_gemma4_template_with_assistant_and_null_tool_content() {
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../jinja-templates/gemma4.jinja"
        );
        let raw = std::fs::read_to_string(template_path)
            .expect("bundled gemma4.jinja must be present in the repo");
        let converted = super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
        let env = super::jinja_helpers::build_jinja_env(&converted).expect("template compiles");
        let tmpl = env.get_template("chat").unwrap();
        // The exact shape of the "null content / tool role" coherence case:
        // an assistant turn (exercises strip_thinking → .split) plus a
        // tool-role message whose content is null.
        let messages = vec![
            json!({"role": "user", "content": "What time is it?"}),
            json!({"role": "assistant", "content": "I'll check."}),
            json!({"role": "tool", "content": null}),
            json!({"role": "user", "content": "Thanks."}),
        ];
        let messages_val = minijinja::Value::from_serialize(&messages);
        let ctx = minijinja::context! {
            messages => messages_val,
            tools => minijinja::Value::UNDEFINED,
            add_generation_prompt => true,
            enable_thinking => false,
            bos_token => "<bos>",
        };
        let rendered = tmpl
            .render(ctx)
            .expect("Gemma-4 template must render assistant + null-content tool message");
        // The assistant content survived strip_thinking's .split() round-trip.
        assert!(
            rendered.contains("I'll check."),
            "expected assistant content in render: {rendered}"
        );
    }

    /// Byte-match guard: the `{{ tool | tojson }}` filter used by the
    /// `<tools>` block in jinja-templates/openai/qwen3_5_moe.jinja must
    /// produce EXACTLY what transformers' jinja2 `tojson` does, which is
    /// `json.dumps(x, ensure_ascii=False, sort_keys=False)` — spaces
    /// after `:`/`,` and keys in insertion/declaration order. Without
    /// this Atlas fed the model a compact, key-sorted `<tools>` block
    /// (~26% fewer tokens), diverging from vLLM at the first `:`.
    ///
    /// The fixture and expected string mirror the Python reference:
    ///   json.dumps({"type":"function","function":{"name":"bash",
    ///     "description":"Execute a bash command","parameters":{...}}},
    ///     ensure_ascii=False, sort_keys=False)
    #[test]
    fn tojson_filter_byte_matches_python_json_dumps() {
        // Production path step 1 (api/chat/template.rs:85):
        // `serde_json::to_value(ToolDefinition)`. ToolDefinition's serde
        // field order is {type, function} and FunctionDefinition's is
        // {name, description, parameters} (tool_parser.rs:27-41), so the
        // `to_value` output is byte-equivalent to the literal below.
        // We build the Value directly here so the test stays in the
        // `--lib` target (which does not re-export `tool_parser`); the
        // filter under test is identical either way. With serde_json's
        // `preserve_order`, this literal key order is preserved.
        let tool_value = serde_json::json!({
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to run"
                        }
                    },
                    "required": ["command"]
                }
            }
        });

        // Production path step 2 (tokenizer/chat_impl.rs:197):
        // minijinja::Value::from_serialize over the tool list. With
        // minijinja's `preserve_order`, the map is an IndexMap, so key
        // order survives into the filter.
        let mini_val = minijinja::Value::from_serialize(&tool_value);

        // Production path step 3: the `tojson` filter registered in
        // build_jinja_env. Render via the same env the server uses.
        let env = super::jinja_helpers::build_jinja_env("{{ tool | tojson }}")
            .expect("inline template compiles");
        let tmpl = env.get_template("chat").unwrap();
        let rendered = tmpl
            .render(minijinja::context! { tool => mini_val })
            .expect("tojson render");

        // Ground truth: Python `json.dumps(tool, ensure_ascii=False,
        // sort_keys=False)` of the identical structure (verified with
        // python3 against this exact fixture — len 234).
        let expected = "{\"type\": \"function\", \"function\": {\"name\": \"bash\", \"description\": \"Execute a bash command\", \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\", \"description\": \"The command to run\"}}, \"required\": [\"command\"]}}}";

        assert_eq!(
            rendered, expected,
            "\nAtlas tojson:\n{rendered}\nPython json.dumps:\n{expected}\n"
        );
    }
}
