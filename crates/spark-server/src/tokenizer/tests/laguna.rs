// SPDX-License-Identifier: AGPL-3.0-only

//! Laguna Jinja-template render tests. Split from `tests.rs` (500-LoC cap).

use serde_json::json;

use crate::tokenizer::normalize_tool_call_arguments;

fn render_laguna_template(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    enable_thinking: bool,
) -> String {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../jinja-templates/laguna.jinja"
    ))
    .expect("bundled Laguna template must be present");
    let env = crate::tokenizer::jinja_helpers::build_jinja_env(&raw).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();
    let messages = normalize_tool_call_arguments(messages);
    tmpl.render(minijinja::context! {
        messages => minijinja::Value::from_serialize(&messages),
        tools => tools.map(minijinja::Value::from_serialize).unwrap_or(minijinja::Value::UNDEFINED),
        add_generation_prompt => true,
        enable_thinking => enable_thinking,
        disable_tool_steering => false,
    })
    .expect("template renders")
}

#[test]
fn laguna_template_renders_native_tool_round_trip() {
    let messages = vec![json!({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "Bash",
                "arguments": "{\"command\":\"pwd\"}"
            }
        }]
    })];
    let rendered = render_laguna_template(&messages, None, true);
    assert!(rendered.contains(
        "<assistant><think></think><tool_call>Bash<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call></assistant>"
    ));
    assert!(rendered.ends_with("<assistant><think>"));
}

#[test]
fn laguna_template_uses_checkpoint_tool_json_and_reasoning_controls() {
    let messages = vec![json!({"role": "user", "content": "weather"})];
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    })];

    let no_think = render_laguna_template(&messages, Some(&tools), false);
    assert!(no_think.contains(
        r#"{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}"#
    ));
    assert!(no_think.ends_with("<assistant></think>"));

    let think = render_laguna_template(&messages, Some(&tools), true);
    assert!(think.ends_with("<assistant><think>"));
}
