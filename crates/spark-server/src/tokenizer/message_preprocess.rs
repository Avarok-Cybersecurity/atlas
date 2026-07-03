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
//!   0. [`remap_developer_role`] — rewrite `role: "developer"` to
//!      `"system"` (folding developer+system into one leading system;
//!      a model's own template raises `Unexpected message
//!      role.` on the OpenAI developer role).
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

/// Remap any `role: "developer"` message to `role: "system"` before the
/// model template renders.
///
/// The OpenAI **developer** role (the o1-style system-instruction role) is
/// accepted across the Atlas API surface, and the now-removed Holo override
/// handled it in three places. But a model's OWN shipped
/// `chat_template.jinja` does not know the role and raises
/// `Unexpected message role.` on it — so a request carrying a developer
/// message would hard-fail rendering. `developer` and `system` share
/// system-instruction semantics, so remapping developer→system here (ahead
/// of the model template, alongside the other cross-cutting behaviors) lets
/// such requests render off the model's own template instead of erroring.
///
/// Only the `role` field is rewritten; content is untouched.
pub(crate) fn remap_developer_role(messages: Vec<Value>) -> Vec<Value> {
    let role_of = |m: &Value| -> Option<String> {
        m.get("role").and_then(|r| r.as_str()).map(str::to_string)
    };
    let is_sys_level =
        |m: &Value| matches!(role_of(m).as_deref(), Some("developer") | Some("system"));

    let has_dev = messages
        .iter()
        .any(|m| role_of(m).as_deref() == Some("developer"));
    if !has_dev {
        return messages; // nothing to remap
    }

    // Simple case: the developer message(s) are the only system-level
    // messages — remap developer→system in place, no coalescing needed.
    if messages.iter().filter(|m| is_sys_level(m)).count()
        == messages
            .iter()
            .filter(|m| role_of(m).as_deref() == Some("developer"))
            .count()
    {
        let mut messages = messages;
        for m in messages.iter_mut() {
            if role_of(m).as_deref() == Some("developer")
                && let Some(role) = m.get_mut("role")
            {
                *role = Value::String("system".to_string());
            }
        }
        return messages;
    }

    // A `system` message already exists alongside `developer`. Remapping
    // both to `system` would leave two system messages, which a model's own
    // template rejects (only a leading system is allowed). Fold all
    // system-level (`developer` + `system`) STRING content into ONE system
    // message at the first such position, preserving order. A (rare)
    // non-string system message is kept as its own message untouched.
    let mut merged = String::new();
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    let mut slot: Option<usize> = None;
    for m in messages {
        if is_sys_level(&m) {
            if let Some(s) = m.get("content").and_then(|c| c.as_str()) {
                if !merged.is_empty() && !s.is_empty() {
                    merged.push_str("\n\n");
                }
                merged.push_str(s);
                if slot.is_none() {
                    slot = Some(out.len());
                    out.push(Value::Null); // reserve; filled below
                }
            } else {
                // Non-string system-level content (rare): keep as its own
                // system message rather than drop the payload.
                let mut m = m;
                if let Some(role) = m.get_mut("role") {
                    *role = Value::String("system".to_string());
                }
                out.push(m);
            }
        } else {
            out.push(m);
        }
    }
    if let Some(i) = slot {
        let mut sys = serde_json::Map::new();
        sys.insert("role".to_string(), Value::String("system".to_string()));
        sys.insert("content".to_string(), Value::String(merged));
        out[i] = Value::Object(sys);
    }
    out
}

#[cfg(test)]
mod tests;
