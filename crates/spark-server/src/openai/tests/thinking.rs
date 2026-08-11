// SPDX-License-Identifier: AGPL-3.0-only

//! Thinking directive tests: client channels → `ir::ThinkingDirective`.

use crate::ir::ThinkingDirective;
use crate::openai::*;

fn chat_req(body: serde_json::Value) -> ChatCompletionRequest {
    serde_json::from_value(body).expect("valid chat request")
}

fn base_body() -> serde_json::Value {
    serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
    })
}

#[test]
fn silent_request_is_unspecified() {
    let req = chat_req(base_body());
    assert_eq!(
        req.client_thinking_directive(),
        ThinkingDirective::Unspecified
    );
    assert!(!req.client_thinking_directive().is_explicit());
}

#[test]
fn anthropic_thinking_channel() {
    // type=disabled wins even with a budget present.
    let mut b = base_body();
    b["thinking"] = serde_json::json!({"type": "disabled", "budget_tokens": 100});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    let mut b = base_body();
    b["thinking"] = serde_json::json!({"type": "enabled", "budget_tokens": 512});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: Some(512) }
    );

    // Adaptive / budget-less thinking object → think as long as needed
    // (budget defers to the per-model max_thinking_budget).
    let mut b = base_body();
    b["thinking"] = serde_json::json!({"type": "adaptive"});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: None }
    );
}

#[test]
fn thinking_token_budget_channel() {
    let mut b = base_body();
    b["thinking_token_budget"] = serde_json::json!(512);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: Some(512) }
    );

    let mut b = base_body();
    b["thinking_token_budget"] = serde_json::json!(0);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );
}

#[test]
fn reasoning_effort_channel() {
    for (effort, expect) in [
        ("none", ThinkingDirective::Off),
        ("minimal", ThinkingDirective::On { budget: Some(64) }),
        ("low", ThinkingDirective::On { budget: Some(128) }),
        ("medium", ThinkingDirective::On { budget: Some(256) }),
        ("high", ThinkingDirective::On { budget: Some(512) }),
        ("xhigh", ThinkingDirective::On { budget: Some(1024) }),
        ("max", ThinkingDirective::On { budget: Some(1024) }),
        // Unknown efforts fall back to the conservative default budget.
        ("bogus", ThinkingDirective::On { budget: Some(256) }),
    ] {
        let mut b = base_body();
        b["reasoning"] = serde_json::json!({"effort": effort});
        assert_eq!(
            chat_req(b).client_thinking_directive(),
            expect,
            "effort={effort}"
        );
    }
}

#[test]
fn top_level_reasoning_effort_channel() {
    // The Chat Completions wire spelling 2026 SDKs send (e.g. OpenAI
    // .NET `ReasoningEffortLevel`). Must ride the same budget ladder as
    // the nested object — `"none"` included, which forces thinking OFF.
    for (effort, expect) in [
        ("none", ThinkingDirective::Off),
        ("minimal", ThinkingDirective::On { budget: Some(64) }),
        ("low", ThinkingDirective::On { budget: Some(128) }),
        ("medium", ThinkingDirective::On { budget: Some(256) }),
        ("high", ThinkingDirective::On { budget: Some(512) }),
    ] {
        let mut b = base_body();
        b["reasoning_effort"] = serde_json::json!(effort);
        assert_eq!(
            chat_req(b).client_thinking_directive(),
            expect,
            "effort={effort}"
        );
    }

    // Nested object wins when a client sends both spellings.
    let mut b = base_body();
    b["reasoning"] = serde_json::json!({"effort": "high"});
    b["reasoning_effort"] = serde_json::json!("low");
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: Some(512) }
    );
}

#[test]
fn thinking_budget_aliases_thinking_token_budget() {
    // DashScope/Qwen spelling injected top-level by OpenAI-compatible
    // gateways; must map onto the explicit-budget rung.
    let mut b = base_body();
    b["thinking_budget"] = serde_json::json!(2048);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: Some(2048) }
    );

    let mut b = base_body();
    b["thinking_budget"] = serde_json::json!(0);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    // Explicit budget outranks the effort ladder.
    let mut b = base_body();
    b["thinking_budget"] = serde_json::json!(2048);
    b["reasoning_effort"] = serde_json::json!("low");
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: Some(2048) }
    );
}

#[test]
fn top_level_reasoning_effort_channel_and_nested_priority() {
    let mut top_level = base_body();
    top_level["reasoning_effort"] = serde_json::json!("max");
    let req = chat_req(top_level);
    assert_eq!(
        req.client_thinking_directive(),
        ThinkingDirective::On { budget: Some(1024) }
    );
    assert_eq!(
        req.client_reasoning_effort(),
        Some(crate::ir::ReasoningEffort::Max)
    );

    let mut both = base_body();
    both["reasoning_effort"] = serde_json::json!("max");
    both["reasoning"] = serde_json::json!({"effort": "high"});
    let req = chat_req(both);
    assert_eq!(
        req.client_thinking_directive(),
        ThinkingDirective::On { budget: Some(512) }
    );
    assert_eq!(
        req.client_reasoning_effort(),
        Some(crate::ir::ReasoningEffort::High)
    );
}

#[test]
fn chat_template_kwargs_channel() {
    // Struct still parses as a request-body wire field.
    let kw: ChatTemplateKwargs =
        serde_json::from_str(r#"{"enable_thinking":true,"thinking_budget":1024}"#)
            .expect("should parse");
    assert_eq!(kw.enable_thinking, Some(true));
    assert_eq!(kw.thinking_budget, Some(1024));

    // Budget rung wins over the enable flag.
    let mut b = base_body();
    b["chat_template_kwargs"] =
        serde_json::json!({"enable_thinking": false, "thinking_budget": 1024});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: Some(1024) }
    );

    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({"thinking_budget": 0});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    // enable_thinking with no explicit budget defers to the per-model
    // max_thinking_budget (budget: None), not the conservative 256-token
    // default — a hard cut force-injects </think> mid-reasoning and
    // wrecks agentic tool selection.
    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({"enable_thinking": true});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: None }
    );

    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({"enable_thinking": false});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    // Empty kwargs object carries no intent.
    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Unspecified
    );
}

#[test]
fn legacy_enable_thinking_channel() {
    let mut b = base_body();
    b["enable_thinking"] = serde_json::json!(true);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: None }
    );

    // Explicit false now DISABLES thinking (it is Option<bool>, so it is
    // distinguishable from absent). Previously it was silently ignored.
    let mut b = base_body();
    b["enable_thinking"] = serde_json::json!(false);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    // Field ABSENT → Unspecified, so a client that doesn't send the flag
    // inherits the model's design intent (thinking_default).
    let b = base_body();
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Unspecified
    );
}
