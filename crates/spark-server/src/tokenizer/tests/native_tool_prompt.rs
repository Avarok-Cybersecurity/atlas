// SPDX-License-Identifier: AGPL-3.0-only

//! H100 rental reproduction: the same get_weather request reached Atlas as
//! 1135 tokens and vLLM as 320. Test the production parser contribution and
//! render core against the pinned checkpoint template, without loading a GPU.

use super::super::chat_render::{RenderFlags, render_chat};
use super::super::jinja_helpers::{ToolJsonStyle, build_jinja_env_with};
use crate::api::chat::prepare::{inject_tool_system_prompt, parser_tool_prompt};
use crate::ir::{ContentPart, Message, Role};
use crate::tool_parser::{
    HermesParser, PromptLevers, Qwen3CoderParser, Qwen3XmlParser, ToolCallParser, ToolChoice,
    ToolChoiceFunction, ToolDefinition,
};
use serde_json::json;

fn user() -> Message {
    Message {
        role: Role::User,
        content: vec![ContentPart::Text(
            "What is the weather in Reykjavik over the next three days? Use the tool.".into(),
        )],
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

fn tools() -> Vec<ToolDefinition> {
    serde_json::from_value(json!([{"type":"function","function":{
        "name":"get_weather","description":"Look up the current weather for a city.",
        "parameters":{"type":"object","properties":{
            "city":{"type":"string","description":"City name."},
            "days":{"type":"integer","description":"Forecast horizon in days."}},
            "required":["city","days"]}}}]))
    .unwrap()
}

fn render(messages: &[Message], style: ToolJsonStyle) -> String {
    // Qwen/Qwen3.6-35B-A3B-FP8 @ 95a723d08a9490559dae23d0cff1d9466213d989.
    let raw = include_str!("../../../test_data/chat_templates/qwen3.6-35b-a3b-fp8.jinja");
    let converted = super::super::jinja_helpers::convert_python_jinja_to_minijinja(raw);
    let env = build_jinja_env_with(&converted, style).unwrap();
    let messages: Vec<_> = messages
        .iter()
        .map(|m| {
            json!({
                "role": if m.role == Role::System {"system"} else {"user"},
                "content": m.text(),
            })
        })
        .collect();
    let tools: Vec<_> = tools()
        .iter()
        .map(|t| serde_json::to_value(t).unwrap())
        .collect();
    render_chat(&env, &messages, Some(&tools), RenderFlags::default()).unwrap()
}

#[test]
fn native_qwen_tool_prompt_has_one_instruction_source() {
    let original = vec![user()];
    let mut messages = original.clone();
    let contribution = parser_tool_prompt(
        &Qwen3CoderParser,
        &tools(),
        &ToolChoice::Mode("auto".into()),
        &PromptLevers::OFF,
        true,
    );
    inject_tool_system_prompt(&mut messages, contribution);
    let actual = render(&messages, ToolJsonStyle::Compact);
    let native = render(&original, ToolJsonStyle::Compact);
    let hf = render(&original, ToolJsonStyle::HfSpaced);
    println!(
        "RENDER_RECEIPT={}",
        json!({"actual":actual,"native_compact":native,"native_hf_spaced":hf})
    );
    assert_eq!(
        actual.matches("# Tools").count(),
        1,
        "duplicate tool instructions"
    );
    assert_eq!(
        actual, native,
        "parser must not duplicate checkpoint tool instructions"
    );
    assert_eq!(
        messages, original,
        "client messages must not gain an unrelated system block"
    );
}

#[test]
fn native_qwen_tool_prompt_keeps_required_and_named_choices() {
    for parser in [&Qwen3CoderParser as &dyn ToolCallParser, &Qwen3XmlParser] {
        for choice in [
            ToolChoice::Mode("required".into()),
            ToolChoice::Specific {
                function: ToolChoiceFunction {
                    name: "get_weather".into(),
                },
            },
        ] {
            let prompt = parser_tool_prompt(parser, &tools(), &choice, &PromptLevers::OFF, true);
            assert!(
                prompt.contains("MUST call"),
                "tool choice enforcement retained"
            );
            assert!(
                !prompt.contains("# Tools"),
                "schema belongs to native template"
            );
            if matches!(choice, ToolChoice::Specific { .. }) {
                assert!(prompt.contains("'get_weather'"));
            }
        }
    }
}

#[test]
fn native_qwen_tool_prompt_preserves_fallback_tscg_and_other_parsers() {
    let choice = ToolChoice::Mode("auto".into());
    for parser in [
        &Qwen3CoderParser as &dyn ToolCallParser,
        &Qwen3XmlParser,
        &HermesParser,
    ] {
        for (native, tscg) in [(false, false), (false, true), (true, true)] {
            let levers = PromptLevers::new(tscg);
            assert_eq!(
                parser_tool_prompt(parser, &tools(), &choice, &levers, native),
                parser.system_prompt(&tools(), &choice, &levers)
            );
        }
    }
    assert_eq!(
        parser_tool_prompt(&HermesParser, &tools(), &choice, &PromptLevers::OFF, true),
        HermesParser.system_prompt(&tools(), &choice, &PromptLevers::OFF)
    );
}

#[test]
fn native_qwen_tool_prompt_preserves_user_system_message() {
    let mut messages = vec![
        Message::synthetic_system("User supplied system policy".into()),
        user(),
    ];
    let before = messages.clone();
    let contribution = parser_tool_prompt(
        &Qwen3CoderParser,
        &tools(),
        &ToolChoice::Mode("auto".into()),
        &PromptLevers::OFF,
        true,
    );
    inject_tool_system_prompt(&mut messages, contribution);
    assert_eq!(messages, before);
}

#[test]
fn native_qwen_tool_prompt_ownership_tracks_actual_template_selection() {
    use super::super::ChatTokenizer;
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    let repo = root.path().join("repo");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::create_dir_all(repo.join("jinja-templates/openai")).unwrap();
    tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default())
        .save(model.join("tokenizer.json"), false)
        .unwrap();
    let load =
        |kind| ChatTokenizer::from_model_dir(&model, 0, true, kind, Some(&repo), false).unwrap();
    // No template is the ChatML fallback, even for a known Qwen model type.
    assert!(!load("qwen3_6_moe").uses_native_qwen_tool_template());
    std::fs::write(
        model.join("tokenizer_config.json"),
        json!({"chat_template":"{{ messages }}"}).to_string(),
    )
    .unwrap();
    assert!(
        !load("qwen3_6_moe").uses_native_qwen_tool_template(),
        "a custom checkpoint template without tools needs parser instructions"
    );
    let raw = include_str!("../../../test_data/chat_templates/qwen3.6-35b-a3b-fp8.jinja");
    std::fs::write(
        model.join("tokenizer_config.json"),
        json!({"chat_template":raw}).to_string(),
    )
    .unwrap();
    for kind in ["qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_6_moe"] {
        assert!(load(kind).uses_native_qwen_tool_template(), "{kind}");
    }
    assert!(!load("nemotron_h").uses_native_qwen_tool_template());
    let custom = repo.join("jinja-templates/qwen3_6_moe.jinja");
    std::fs::write(&custom, "{{ messages }}").unwrap();
    assert!(!load("qwen3_6_moe").uses_native_qwen_tool_template());
    std::fs::remove_file(custom).unwrap();
    std::fs::write(
        repo.join("jinja-templates/openai/qwen3_6_moe.jinja"),
        "{{ messages }}",
    )
    .unwrap();
    assert!(!load("qwen3_6_moe").uses_native_qwen_tool_template());
}
