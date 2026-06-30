// SPDX-License-Identifier: AGPL-3.0-only
//
//! Cross-cutting chat-message preprocessing applied *before* the Jinja
//! template renders, independent of which template is in use.
//!
//! Historically these behaviors lived inside hand-maintained
//! `jinja-templates/{model_type}.jinja` overrides (notably the
//! now-removed `holo3_1_moe.jinja`). The model's *own* shipped
//! `chat_template.jinja` does not implement them, so encoding them in
//! Jinja meant every model that wanted them needed a bespoke override
//! that was otherwise a byte-for-byte copy of the model template.
//!
//! Porting them to Rust (next to `normalize_tool_call_arguments`) makes
//! them apply universally — every model gets them while still rendering
//! off its OWN template — and lets the redundant overrides be deleted.
//!
//! Behaviors implemented here:
//!   1. [`autoclose_think_before_tool_call`] — insert a missing
//!      `</think>` before a `<tool_call>` in assistant *history* content.
//!   2. [`resolve_think_control`] — strip inline `<|think_on|>` /
//!      `<|think_off|>` control tokens and resolve the effective
//!      `enable_thinking` they request.
//!
//! (`reasoning_effort` → `enable_thinking` is intentionally NOT
//! re-implemented here: `ChatCompletionRequest::resolve_thinking`
//! already maps OpenAI `reasoning.effort` to `(enable_thinking,
//! budget)` upstream — see `openai/chat_request.rs`. Duplicating it in
//! template preprocessing would double-resolve it. The pin lives next to
//! that resolver: `reasoning_effort_maps_to_enable_thinking`.)

use serde_json::Value;

/// Inline control tokens a client may embed in message content to toggle
/// thinking, mirroring the vLLM/Holo convention.
const THINK_ON: &str = "<|think_on|>";
const THINK_OFF: &str = "<|think_off|>";

/// Auto-close an unclosed `<think>` block that precedes a `<tool_call>`
/// in a single assistant-history content string.
///
/// SAFETY behavior ported from the Holo override: a multi-turn history
/// can contain an assistant turn that opened `<think>` but emitted a
/// `<tool_call>` without ever closing it. Left as-is, the malformed
/// dangling-think corrupts the rendered prompt (the model template wraps
/// it verbatim). We insert `</think>` immediately before the first
/// `<tool_call>` when the most recent `<think>` is still open at that
/// point. This is a no-op when there is no `<think>`, no `<tool_call>`,
/// or the think is already closed before the tool call.
///
/// Mirrors the jinja logic:
/// ```jinja
/// {%- if '<think>' in content and '<tool_call>' in content %}
///   last_think = content.rfind('<think>')
///   last_close = content.rfind('</think>')
///   tool_pos   = content.find('<tool_call>')
///   {%- if last_close < last_think or last_close == -1 %}
///     {%- if tool_pos > last_think %}
///       content = content[:tool_pos] + '</think>' + content[tool_pos:]
///     {%- else %}
///       content = content + '</think>'
/// ```
pub(super) fn autoclose_think_before_tool_call(content: &str) -> std::borrow::Cow<'_, str> {
    let Some(tool_pos) = content.find("<tool_call>") else {
        return std::borrow::Cow::Borrowed(content);
    };
    let Some(last_think) = content.rfind("<think>") else {
        return std::borrow::Cow::Borrowed(content);
    };
    let last_close = content.rfind("</think>");
    // The most recent `<think>` is unclosed when there is no `</think>`
    // at all, or the last `</think>` precedes the last `<think>`.
    let think_unclosed = match last_close {
        None => true,
        Some(close) => close < last_think,
    };
    if !think_unclosed {
        return std::borrow::Cow::Borrowed(content);
    }
    if tool_pos > last_think {
        // The tool call sits after the open think — close it right before.
        let mut out = String::with_capacity(content.len() + "</think>".len());
        out.push_str(&content[..tool_pos]);
        out.push_str("</think>");
        out.push_str(&content[tool_pos..]);
        std::borrow::Cow::Owned(out)
    } else {
        // Open think comes after the tool call (unusual) — append a close.
        std::borrow::Cow::Owned(format!("{content}</think>"))
    }
}

/// Strip inline `<|think_on|>` / `<|think_off|>` control tokens from
/// every message's content and resolve the effective thinking request.
///
/// Returns the rewritten message list plus `Some(true|false)` when any
/// control token was seen (last occurrence across all messages wins,
/// matching the jinja override's sequential `set ns_flags.enable_thinking`
/// loop), or `None` when no control token was present (caller keeps its
/// upstream-resolved `enable_thinking`).
///
/// Only the textual content is rewritten; structured (array) content
/// items have their `text` fields scrubbed in place. Non-string,
/// non-`text` items pass through untouched.
pub(crate) fn resolve_think_control(messages: &[Value]) -> (Vec<Value>, Option<bool>) {
    let mut effective: Option<bool> = None;
    let out = messages
        .iter()
        .map(|msg| {
            let mut msg = msg.clone();
            if let Some(content) = msg.get_mut("content") {
                strip_controls_in_content(content, &mut effective);
            }
            msg
        })
        .collect();
    (out, effective)
}

/// Scrub control tokens out of a single message's `content` (string or
/// array-of-parts), updating `effective` to the last toggle seen.
fn strip_controls_in_content(content: &mut Value, effective: &mut Option<bool>) {
    match content {
        Value::String(s) => {
            if let Some(stripped) = strip_controls_in_str(s, effective) {
                *s = stripped;
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                if let Some(text) = item.get_mut("text").and_then(|t| match t {
                    Value::String(s) => Some(s),
                    _ => None,
                }) && let Some(stripped) = strip_controls_in_str(text, effective)
                {
                    *text = stripped;
                }
            }
        }
        _ => {}
    }
}

/// Returns `Some(stripped)` when a control token was found and removed
/// (so the caller can replace the original), or `None` when the string
/// was untouched. Updates `effective` per token, last-write-wins.
fn strip_controls_in_str(s: &str, effective: &mut Option<bool>) -> Option<String> {
    if !s.contains(THINK_ON) && !s.contains(THINK_OFF) {
        return None;
    }
    // Walk the string left-to-right so the LAST control token wins, then
    // remove every occurrence. The jinja override evaluated `think_off`
    // before `think_on` within one message; to preserve that ordering we
    // resolve by scanning positions rather than by token kind.
    let mut positions: Vec<(usize, bool)> = Vec::new();
    for (idx, _) in s.match_indices(THINK_OFF) {
        positions.push((idx, false));
    }
    for (idx, _) in s.match_indices(THINK_ON) {
        positions.push((idx, true));
    }
    positions.sort_by_key(|(idx, _)| *idx);
    if let Some((_, last)) = positions.last() {
        *effective = Some(*last);
    }
    let stripped = s.replace(THINK_OFF, "").replace(THINK_ON, "");
    Some(stripped)
}

/// Apply [`autoclose_think_before_tool_call`] to assistant messages'
/// string content. Non-assistant roles and array content are left
/// untouched (the autoclose only ever applied to assistant-history
/// text in the override).
pub(crate) fn autoclose_assistant_think(messages: &mut [Value]) {
    for msg in messages.iter_mut() {
        let is_assistant = msg.get("role").and_then(|r| r.as_str()) == Some("assistant");
        if !is_assistant {
            continue;
        }
        let Some(Value::String(content)) = msg.get_mut("content") else {
            continue;
        };
        if let std::borrow::Cow::Owned(fixed) = autoclose_think_before_tool_call(content) {
            *content = fixed;
        }
    }
}

#[cfg(test)]
mod tests {
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

    /// Mirror of `chat_impl::preprocess_for_render`: F76 arg-parse →
    /// autoclose-think → think-control, returning prepared messages and
    /// the effective thinking flag. Kept in sync with production so the
    /// e2e assertions exercise the real pipeline.
    fn preprocess(messages: &[Value], enable_thinking: bool) -> (Vec<Value>, bool) {
        let mut prepared = super::super::normalize_tool_call_arguments(messages);
        autoclose_assistant_think(&mut prepared);
        let (prepared, control) = resolve_think_control(&prepared);
        (prepared, control.unwrap_or(enable_thinking))
    }

    fn render_holo_model_template(messages: &[Value], enable_thinking: bool) -> String {
        let raw = std::fs::read_to_string(HOLO_MODEL_TEMPLATE)
            .expect("bundled Holo model template fixture must be present");
        // The Holo template is the model's OWN, raw transformers jinja —
        // so it MUST go through the same conversion the production loader
        // applies (proving we render off the model template, not a
        // pre-massaged Atlas override).
        let converted = super::super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
        let env =
            super::super::jinja_helpers::build_jinja_env(&converted).expect("template compiles");
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
}
