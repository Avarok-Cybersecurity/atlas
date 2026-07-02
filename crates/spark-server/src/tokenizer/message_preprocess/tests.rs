// SPDX-License-Identifier: AGPL-3.0-only

//! Unit + e2e tests for [`super`] (message preprocessing). Split from
//! `message_preprocess.rs` to stay under the 500-LoC-per-file cap.

use super::*;
use serde_json::json;

#[test]
fn autoclose_inserts_before_tool_call() {
    let content = "<think>reasoning here<tool_call>\n<function=foo>";
    let fixed = autoclose_think_before_tool_call(content);
    assert_eq!(
        fixed,
        "<think>reasoning here</think><tool_call>\n<function=foo>"
    );
}

#[test]
fn autoclose_noop_when_already_closed() {
    let content = "<think>reasoning</think>\nanswer<tool_call>\n<function=foo>";
    let fixed = autoclose_think_before_tool_call(content);
    assert_eq!(fixed, content, "already-closed think must be untouched");
}

#[test]
fn autoclose_noop_without_tool_call() {
    let content = "<think>still thinking";
    assert_eq!(autoclose_think_before_tool_call(content), content);
}

#[test]
fn autoclose_noop_without_think() {
    let content = "plain answer<tool_call>\n<function=foo>";
    assert_eq!(autoclose_think_before_tool_call(content), content);
}

#[test]
fn autoclose_assistant_only() {
    let mut messages = vec![
        json!({"role": "user", "content": "<think>x<tool_call>"}),
        json!({"role": "assistant", "content": "<think>x<tool_call>\n<function=f>"}),
    ];
    autoclose_assistant_think(&mut messages);
    // user content untouched
    assert_eq!(messages[0]["content"], "<think>x<tool_call>");
    // assistant content closed
    assert_eq!(
        messages[1]["content"],
        "<think>x</think><tool_call>\n<function=f>"
    );
}

#[test]
fn think_control_off_disables() {
    let messages = vec![json!({"role": "system", "content": "Be terse <|think_off|>"})];
    let (out, effective) = resolve_think_control(&messages);
    assert_eq!(effective, Some(false));
    assert_eq!(out[0]["content"], "Be terse ");
}

#[test]
fn think_control_on_enables() {
    let messages = vec![json!({"role": "user", "content": "<|think_on|>solve this"})];
    let (out, effective) = resolve_think_control(&messages);
    assert_eq!(effective, Some(true));
    assert_eq!(out[0]["content"], "solve this");
}

#[test]
fn think_control_last_wins_across_messages() {
    let messages = vec![
        json!({"role": "system", "content": "<|think_on|>"}),
        json!({"role": "user", "content": "now <|think_off|> please"}),
    ];
    let (_out, effective) = resolve_think_control(&messages);
    assert_eq!(
        effective,
        Some(false),
        "last control token across messages wins"
    );
}

#[test]
fn think_control_absent_returns_none() {
    let messages = vec![json!({"role": "user", "content": "hello"})];
    let (out, effective) = resolve_think_control(&messages);
    assert_eq!(effective, None);
    assert_eq!(out, messages);
}

#[test]
fn think_control_strips_in_array_content() {
    let messages = vec![json!({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "describe <|think_off|>"}
        ]
    })];
    let (out, effective) = resolve_think_control(&messages);
    assert_eq!(effective, Some(false));
    assert_eq!(out[0]["content"][1]["text"], "describe ");
}

#[test]
fn remap_developer_to_system() {
    let mut messages = vec![
        json!({"role": "developer", "content": "You are terse."}),
        json!({"role": "user", "content": "hi"}),
        json!({"role": "system", "content": "unchanged"}),
    ];
    remap_developer_role(&mut messages);
    assert_eq!(messages[0]["role"], "system", "developer → system");
    assert_eq!(
        messages[0]["content"], "You are terse.",
        "content untouched"
    );
    assert_eq!(messages[1]["role"], "user", "other roles untouched");
    assert_eq!(messages[2]["role"], "system");
}

// --- End-to-end: Holo renders off the MODEL's own template + Rust behaviors ---
//
// The bundled fixture is the Holo-3.1-35B model's OWN
// `chat_template.jinja` (copied verbatim — grep-verified to contain
// NONE of `<|think_on|>`, `rfind`, or `reasoning_effort`). The
// hand-maintained `jinja-templates/holo3_1_moe.jinja` override that
// used to add those behaviors has been removed. These tests prove the
// behaviors now come from Rust preprocessing applied ahead of the
// model's own template, exactly as the production
// `preprocess_for_render` does.

/// Path to the bundled Holo model template fixture.
const HOLO_MODEL_TEMPLATE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/holo3_1_moe.model_template.jinja"
);

/// Calls the real production `chat_impl::preprocess_for_render`
/// (F76 arg-parse → developer remap → autoclose-think → think-control)
/// so the e2e assertions exercise the exact pipeline, with no mirror to
/// drift out of sync.
fn preprocess(messages: &[Value], enable_thinking: bool) -> (Vec<Value>, bool) {
    super::super::chat_impl::preprocess_for_render(messages, enable_thinking)
}

fn render_holo_model_template(messages: &[Value], enable_thinking: bool) -> String {
    let raw = std::fs::read_to_string(HOLO_MODEL_TEMPLATE)
        .expect("bundled Holo model template fixture must be present");
    // The Holo template is the model's OWN, raw transformers jinja —
    // so it MUST go through the same conversion the production loader
    // applies (proving we render off the model template, not a
    // pre-massaged Atlas override).
    let converted = super::super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = super::super::jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();
    let (prepared, effective) = preprocess(messages, enable_thinking);
    let reasoning_effort: minijinja::Value = if effective { "high" } else { "none" }.into();
    let ctx = minijinja::context! {
        messages => minijinja::Value::from_serialize(&prepared),
        tools => minijinja::Value::UNDEFINED,
        add_generation_prompt => true,
        enable_thinking => effective,
        reasoning_effort => reasoning_effort,
        disable_tool_steering => false,
        add_vision_id => false,
    };
    tmpl.render(ctx).expect("Holo model template renders")
}

/// The Holo model's OWN template renders (no override needed), and a
/// simple turn produces the expected `<|im_start|>` framing — proof
/// the model template is the renderer.
#[test]
fn holo_renders_off_model_template() {
    let messages = vec![json!({"role": "user", "content": "Hi"})];
    let rendered = render_holo_model_template(&messages, true);
    assert!(
        rendered.contains("<|im_start|>user\nHi<|im_end|>"),
        "expected Holo model-template framing: {rendered}"
    );
    // The generation prompt opens a think block when thinking is on.
    assert!(
        rendered.ends_with("<|im_start|>assistant\n<think>\n"),
        "expected open-think generation prompt: {rendered}"
    );
}

/// Developer-role regression (ported from the deleted
/// `render_holo_template_accepts_vllm_thinking_controls`, which passed a
/// `role: developer` message): the Holo model's OWN template raises
/// `Unexpected message role.` on `developer`, so without the
/// developer→system remap in `preprocess_for_render` the render hard-fails.
/// This proves the remap lets such a request render as a system message.
#[test]
fn holo_renders_developer_message_as_system() {
    let messages = vec![
        json!({"role": "developer", "content": "You are a terse assistant."}),
        json!({"role": "user", "content": "Hi"}),
    ];
    // Must NOT panic ("Holo model template renders" would fail on the
    // template's `Unexpected message role.` raise otherwise).
    let rendered = render_holo_model_template(&messages, true);
    // The developer instruction is emitted under system framing.
    assert!(
        rendered.contains("<|im_start|>system\nYou are a terse assistant.<|im_end|>"),
        "developer content must render as a system turn: {rendered}"
    );
    assert!(
        rendered.contains("<|im_start|>user\nHi<|im_end|>"),
        "user turn still renders: {rendered}"
    );
}

/// Behavior 1 (ported to Rust): an assistant-history turn that opened
/// `<think>` then emitted `<tool_call>` without closing it is repaired
/// BEFORE the model template renders it.
///
/// The repair is load-bearing because the Holo model template gates
/// reasoning extraction on `'</think>' in content`. WITHOUT autoclose,
/// the model template never sees a `</think>`, so it treats the entire
/// `<think>reasoning…<tool_call>…` blob as visible assistant content and
/// leaks the reasoning text (and a literal unclosed `<think>`) into the
/// prompt. WITH autoclose, the template cleanly splits reasoning from
/// the tool call. We assert the post-repair invariants against the
/// model's OWN template render.
#[test]
fn holo_autocloses_think_before_tool_call() {
    let messages = vec![
        json!({"role": "user", "content": "List /tmp"}),
        json!({
            "role": "assistant",
            "content": "<think>I should list it<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>"
        }),
        json!({"role": "user", "content": "thanks"}),
    ];
    let with_fix = render_holo_model_template(&messages, true);

    // The leaked reasoning text must NOT survive into the rendered
    // prompt (the template extracted it as reasoning_content and, this
    // being a historical turn, dropped it).
    assert!(
        !with_fix.contains("I should list it"),
        "autoclose must let the template separate reasoning from content: {with_fix}"
    );
    // No literal unclosed `<think>` abutting the tool call.
    assert!(
        !with_fix.contains("<think>I should list it"),
        "dangling <think> must not leak into the prompt: {with_fix}"
    );
    // The tool call itself is still rendered.
    assert!(
        with_fix.contains("<tool_call>") && with_fix.contains("<function=bash>"),
        "tool call must survive the repair: {with_fix}"
    );

    // Contrast: rendering the SAME messages through the model template
    // WITHOUT the autoclose repair leaks the reasoning text — proving
    // the Rust behavior is load-bearing, not cosmetic.
    let raw = std::fs::read_to_string(HOLO_MODEL_TEMPLATE).unwrap();
    let converted = super::super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = super::super::jinja_helpers::build_jinja_env(&converted).unwrap();
    let tmpl = env.get_template("chat").unwrap();
    let unrepaired = tmpl
        .render(minijinja::context! {
            messages => minijinja::Value::from_serialize(&messages),
            tools => minijinja::Value::UNDEFINED,
            add_generation_prompt => true,
            enable_thinking => true,
            reasoning_effort => "high",
            disable_tool_steering => false,
            add_vision_id => false,
        })
        .unwrap();
    assert!(
        unrepaired.contains("I should list it"),
        "sanity: without autoclose the reasoning DOES leak: {unrepaired}"
    );
}

/// Behavior 2 (ported to Rust): `<|think_off|>` in a message strips
/// from the rendered prompt AND forces the closed-thinking generation
/// prompt, even though the caller passed `enable_thinking=true`.
#[test]
fn holo_think_off_control_closes_generation_prompt() {
    let messages = vec![json!({"role": "user", "content": "Answer fast <|think_off|>"})];
    let rendered = render_holo_model_template(&messages, true);
    assert!(
        !rendered.contains("<|think_off|>"),
        "control token must be stripped from prompt: {rendered}"
    );
    assert!(
        rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "think_off must yield a CLOSED-think generation prompt: {rendered}"
    );
}

/// Behavior 2 (inverse): `<|think_on|>` forces the open-think prompt
/// even when the caller passed `enable_thinking=false`.
#[test]
fn holo_think_on_control_opens_generation_prompt() {
    let messages = vec![json!({"role": "user", "content": "<|think_on|>reason it out"})];
    let rendered = render_holo_model_template(&messages, false);
    assert!(!rendered.contains("<|think_on|>"));
    assert!(
        rendered.ends_with("<|im_start|>assistant\n<think>\n"),
        "think_on must yield an OPEN-think generation prompt: {rendered}"
    );
}

// Behavior 3 (`reasoning_effort` → `enable_thinking`) is intentionally
// NOT implemented in this module: it is resolved UPSTREAM in
// `ChatCompletionRequest::resolve_thinking` (openai/chat_request.rs),
// whose output flows into the `enable_thinking` argument this module's
// preprocessing receives. The pin for that contract lives next to the
// implementation — see `reasoning_effort_maps_to_enable_thinking` in
// `openai/chat_request.rs`'s test module. (The `openai` module is
// declared in the binary crate, not the lib, so it is not reachable
// from this lib-test target.)
