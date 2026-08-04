// SPDX-License-Identifier: AGPL-3.0-only

//! Extracted piecewise from `tokenizer.rs` (500-LoC cap).

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

fn render_holo_template(messages: &[serde_json::Value], enable_thinking: bool) -> String {
    let template_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../jinja-templates/holo3_1_moe.jinja"
    );
    let raw = std::fs::read_to_string(template_path)
        .expect("bundled Holo3.1 template must be present in the repo");
    let converted = super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = super::jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();
    let messages_for_render = normalize_tool_call_arguments(messages);
    let messages_val = minijinja::Value::from_serialize(&messages_for_render);
    let ctx = minijinja::context! {
        messages => messages_val,
        tools => minijinja::Value::UNDEFINED,
        add_generation_prompt => true,
        enable_thinking => enable_thinking,
        reasoning_effort => "none",
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
fn render_holo_template_accepts_vllm_thinking_controls() {
    let messages = vec![
        json!({"role": "developer", "content": "<|think_off|>Follow the instruction."}),
        json!({"role": "user", "content": "Reply with OK."}),
    ];
    let rendered = render_holo_template(&messages, true);
    assert!(rendered.contains("Follow the instruction."));
    assert!(!rendered.contains("<|think_off|>"));
    assert!(
        rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "expected closed thinking prompt from think_off: {rendered}"
    );
}

#[test]
fn render_holo_template_autocloses_think_before_tool_call() {
    let messages = vec![
        json!({"role": "user", "content": "Use bash."}),
        json!({
            "role": "assistant",
            "content": "<think>\nNeed a directory listing.\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
        }),
    ];
    let rendered = render_holo_template(&messages, true);
    assert!(
        rendered.contains("Need a directory listing.\n</think>\n\n<tool_call>"),
        "expected unclosed think to be closed before tool call: {rendered}"
    );
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

/// DS4F reasoning-mode primer (2026-07-21): the generation prompt appends the
/// real `<think>` primer (official DS4F thinking suffix
/// `<｜Assistant｜><think>`) ONLY on explicit `enable_thinking == true`.
/// Default (undefined) and `enable_thinking == false` emit NO primer —
/// byte-identical to the verified 35/40 direct-mode baseline (which appended
/// the blanked think tokens = empty string). Before this fix the template
/// blanked BOTH think tokens, so `enable_thinking` was a silent no-op and the
/// model never reasoned (§B thinking-state verdict).
#[test]
fn deepseek_v4_reasoning_primer_is_opt_in() {
    let template_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../jinja-templates/deepseek_v4.jinja"
    );
    let raw = std::fs::read_to_string(template_path)
        .expect("bundled deepseek_v4.jinja must be present in the repo");
    let converted = super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = super::jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();

    let messages = vec![json!({"role": "user", "content": "2+2?"})];
    let mv = minijinja::Value::from_serialize(&messages);
    let render = |et: minijinja::Value| {
        tmpl.render(minijinja::context! {
            messages => mv.clone(),
            tools => minijinja::Value::UNDEFINED,
            add_generation_prompt => true,
            enable_thinking => et,
        })
        .expect("deepseek_v4 template must render")
    };

    // Reasoning mode (enable_thinking==true): suffix ends <｜Assistant｜><think>
    // (official DS4F thinking contract, encoding_dsv4.py:388).
    let thinking = render(minijinja::Value::from(true));
    assert!(
        thinking.ends_with("<｜Assistant｜><think>"),
        "reasoning mode must prime <think>: {thinking:?}"
    );

    // Direct mode (explicit false): suffix ends <｜Assistant｜></think> — the
    // official direct contract (thinking pre-closed; model answers directly).
    let direct = render(minijinja::Value::from(false));
    assert!(
        direct.ends_with("<｜Assistant｜></think>"),
        "direct suffix must be the official <｜Assistant｜></think>: {direct:?}"
    );
    assert!(
        !direct.contains("<think>"),
        "direct mode must NOT open a <think> block: {direct:?}"
    );

    // Default (enable_thinking undefined) resolves to direct at the API edge
    // (thinking_default=false), so the template's else-branch must also produce
    // the official direct suffix — identical to explicit-false.
    let default = render(minijinja::Value::UNDEFINED);
    assert_eq!(
        default, direct,
        "default (unspecified) must render the official direct suffix"
    );
}

/// The DS4F reasoning parser must be wired via `tool_defaults.toml` because
/// `supports_thinking = has_ssm || has_mamba2` is FALSE for this pure-attention
/// MLA model — without this entry `think_end_token` never resolves and
/// `inside_thinking` can never engage. Guards the `[reasoning]` entry + that it
/// maps to the `<think>`/`</think>` (DeepSeek-R1) contract.
#[test]
fn deepseek_v4_reasoning_parser_is_registered() {
    use crate::reasoning_parser::ReasoningFormat;
    let defaults_toml = include_str!("../../tool_defaults.toml");
    let defaults: toml::Value = toml::from_str(defaults_toml).expect("tool_defaults parses");
    let fmt_str = defaults
        .get("reasoning")
        .and_then(|t| t.get("deepseek_v4"))
        .and_then(|s| s.as_str())
        .expect("tool_defaults [reasoning] must register deepseek_v4");
    let fmt: ReasoningFormat = fmt_str
        .parse()
        .expect("deepseek_v4 reasoning format parses");
    let p = fmt.into_parser();
    assert_eq!(p.start_tag(), "<think>", "DS4F reasoning start tag");
    assert_eq!(p.end_tag(), "</think>", "DS4F reasoning end tag");
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
fn tojson_filter_default_compact_hf_ref_opt_in() {
    // ToolDefinition serde order is {type, function:{name, description,
    // parameters}} (tool_parser.rs); built directly so the test stays in the
    // `--lib` target. `preserve_order` keeps this key order through the filter.
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

    // DEFAULT (ST-995 fix): minijinja's builtin COMPACT `tojson` — no spaces.
    // This is what the GDN Qwen3.6-27B needs for correct BFCL irrelevance
    // (the spaced HF-reference form regressed hallucination 93.70 -> 30).
    let env_default = super::jinja_helpers::build_jinja_env("{{ tool | tojson }}")
        .expect("inline template compiles");
    let compact = "{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"description\":\"Execute a bash command\",\"parameters\":{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\",\"description\":\"The command to run\"}},\"required\":[\"command\"]}}}";
    let got_default = env_default
        .get_template("chat")
        .unwrap()
        .render(minijinja::context! { tool => minijinja::Value::from_serialize(&tool_value) })
        .expect("tojson render");
    assert_eq!(
        got_default, compact,
        "default tojson must be COMPACT (ST-995 GDN irrelevance fix)\ngot:\n{got_default}\n"
    );

    // OPT-IN (ATLAS_USE_HF_REF_JSON_DUMPS=1): spaced, byte-parity with Python
    // `json.dumps(..., ensure_ascii=False, sort_keys=False)` — the #90 / HF
    // reference serialization (len 234). SAFETY: this is the only test that
    // reads this env var; it is set and cleared within this scope.
    unsafe {
        std::env::set_var("ATLAS_USE_HF_REF_JSON_DUMPS", "1");
    }
    let env_hf = super::jinja_helpers::build_jinja_env("{{ tool | tojson }}")
        .expect("inline template compiles");
    unsafe {
        std::env::remove_var("ATLAS_USE_HF_REF_JSON_DUMPS");
    }
    let spaced = "{\"type\": \"function\", \"function\": {\"name\": \"bash\", \"description\": \"Execute a bash command\", \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\", \"description\": \"The command to run\"}}, \"required\": [\"command\"]}}}";
    let got_hf = env_hf
        .get_template("chat")
        .unwrap()
        .render(minijinja::context! { tool => minijinja::Value::from_serialize(&tool_value) })
        .expect("tojson render");
    assert_eq!(
        got_hf, spaced,
        "ATLAS_USE_HF_REF_JSON_DUMPS=1 must restore Python json.dumps byte parity\ngot:\n{got_hf}\n"
    );
}
