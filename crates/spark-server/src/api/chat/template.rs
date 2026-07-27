// SPDX-License-Identifier: AGPL-3.0-only
//
// Chat-template rendering: build the JSON-message array, run the
// optional auto-compact, apply the Jinja template, expand image
// pad-tokens, and detect template-forced thinking.
//
// Lifted out of `chat::chat_completions_inner` (wave 4g).

use axum::http::StatusCode;
use axum::response::Response;
use std::sync::Arc;

use crate::AppState;

use super::super::compact::{compact_messages, openai_error_response};
use super::msg_entry::MsgEntry;

/// Outputs of [`render_template`]. Threaded into the streaming /
/// blocking dispatch.
pub(super) struct TemplateOut {
    pub(super) prompt_tokens: Vec<u32>,
    /// Possibly overridden by template-forced-thinking detection.
    pub(super) enable_thinking: bool,
    pub(super) thinking_budget: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)]
pub(super) fn render_template(
    state: &Arc<AppState>,
    tools: &[crate::tool_parser::ToolDefinition],
    messages: &[MsgEntry],
    image_pad_counts: &[usize],
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    tools_active: bool,
) -> Result<TemplateOut, Response> {
    // Use closed thinking when client doesn't explicitly enable it.
    let template_thinking = enable_thinking;

    // Build JSON messages with structured tool_calls for Jinja.
    let json_messages = build_json_messages(messages);
    // When TSCG is enabled the parser's `system_prompt()` has already
    // placed the compact tool signatures into messages[0]; passing
    // `tools` to Jinja as well would re-render the full JSON schema and
    // defeat the compaction. Pass `None` so the template's `{% if tools
    // %}` branch falls through — the tool-call format instructions
    // still come from `system_prompt()`.
    let jinja_tools: Option<Vec<serde_json::Value>> =
        if tools_active && !crate::tscg::tscg_enabled() {
            Some(
                tools
                    .iter()
                    .map(|t| serde_json::to_value(t).unwrap_or_default())
                    .collect(),
            )
        } else {
            None
        };

    // Progressive auto-compact (DISABLED BY DEFAULT 2026-04-25 —
    // see project_no_auto_compaction memory feedback).
    let auto_compact_active = state
        .auto_compact_threshold
        .map(|t| t > 0.0)
        .unwrap_or(false);
    let json_messages = if auto_compact_active && json_messages.len() > 4 {
        let trial_tokens = state
            .tokenizer
            .apply_chat_template_openai(
                &json_messages,
                jinja_tools.as_deref(),
                template_thinking,
                state.behavior.disable_tool_steering,
            )
            .map(|t| t.len())
            .unwrap_or(0);
        if trial_tokens > (state.max_seq_len as f32 * 0.70) as usize {
            compact_messages(&json_messages, trial_tokens, state.max_seq_len)
        } else {
            json_messages
        }
    } else {
        json_messages
    };

    let prompt_tokens = match state.tokenizer.apply_chat_template_openai(
        &json_messages,
        jinja_tools.as_deref(),
        template_thinking,
        state.behavior.disable_tool_steering,
    ) {
        Ok(t) => t,
        Err(e) => {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Tokenization error: {e}"),
            ));
        }
    };

    // Expand image pads when needed.
    let prompt_tokens = if image_pad_counts.iter().any(|&c| c > 1) {
        state
            .tokenizer
            .expand_image_pads(prompt_tokens, image_pad_counts)
    } else {
        prompt_tokens
    };

    // Template-forced thinking detection.
    let (enable_thinking, thinking_budget) = if let Some(think_start) = state.think_start_token_id {
        let tail = &prompt_tokens[prompt_tokens.len().saturating_sub(8)..];
        let last_start = tail.iter().rposition(|t| *t == think_start);
        let has_unclosed_think = match (last_start, state.think_end_token_id) {
            (Some(si), Some(end_tok)) => !tail[si + 1..].contains(&end_tok),
            (Some(_), None) => true,
            (None, _) => false,
        };
        if has_unclosed_think && !enable_thinking {
            tracing::info!(
                "Template-forced thinking detected (unclosed \\<think\\> in prompt tail) — \
                 overriding enable_thinking=true with budget={}",
                state.behavior.max_thinking_budget,
            );
            (true, Some(state.behavior.max_thinking_budget))
        } else {
            (enable_thinking, thinking_budget)
        }
    } else {
        (enable_thinking, thinking_budget)
    };

    Ok(TemplateOut {
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    })
}

/// Build the Jinja-facing JSON message array from the processed
/// [`MsgEntry`] vec. Pure (no tokenizer/state) so it can be
/// characterization-tested directly:
///   * `image_count == 0` → `content` is a plain string,
///   * `image_count > 0`  → `content` is `[{type:image} * N, {type:text}]`
///     (text part omitted when empty),
///   * `tool_calls` / `reasoning_content` attached when present.
///
/// Roles arrive canonical (`developer` → `system` normalization happens
/// at MsgEntry build time).
pub(super) fn build_json_messages(messages: &[MsgEntry]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|m| {
            let content_val = if m.image_count > 0 {
                let mut items: Vec<serde_json::Value> = Vec::with_capacity(m.image_count + 1);
                for _ in 0..m.image_count {
                    items.push(serde_json::json!({"type": "image"}));
                }
                if !m.content.is_empty() {
                    items.push(serde_json::json!({"type": "text", "text": m.content}));
                }
                serde_json::Value::Array(items)
            } else {
                serde_json::Value::String(m.content.clone())
            };
            // `developer` → `system` normalization happens upstream at
            // MsgEntry build time (msg_entry.rs), so the role arrives
            // canonical here.
            let mut msg = serde_json::json!({"role": m.role, "content": content_val});
            if let Some(ref tcs) = m.tool_calls {
                msg["tool_calls"] = serde_json::Value::Array(tcs.clone());
            }
            // F1: forward historical reasoning trace to the Jinja
            // template. The template at qwen3_5_moe.jinja:90-104
            // reads `message.reasoning_content` and rehydrates the
            // `<think>` block for the historical assistant turns.
            // Without this, every historical assistant message
            // rendered an empty `<think>\n\n</think>\n\n` wrapper
            // (empty-think poisoning, → premature `<|im_end|>`).
            if let Some(ref rc) = m.reasoning_content {
                msg["reasoning_content"] = serde_json::Value::String(rc.clone());
            }
            msg
        })
        .collect()
}

#[cfg(test)]
mod json_message_tests {
    use super::MsgEntry;
    use super::build_json_messages;

    fn entry(role: &str, content: &str, image_count: usize) -> MsgEntry {
        MsgEntry {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: None,
            image_count,
            reasoning_content: None,
        }
    }

    /// PROMPT-STABILITY GATE. Characterization golden for the full
    /// `Vec<ir::Message>` → `build_msg_entries` → `build_json_messages`
    /// path — the exact JSON the Jinja chat template consumes. Any
    /// change here changes rendered prompts, which breaks kv-cache
    /// prefix reuse across requests of the same conversation. Update
    /// ONLY for an intentional, documented behavior change.
    #[test]
    fn prompt_json_stability_gate() {
        use crate::ir::message::{Reasoning, ToolCall};
        use crate::ir::{ContentPart, Message, Role};

        fn text_msg(role: Role, t: &str) -> Message {
            Message {
                role,
                content: vec![ContentPart::Text(t.into())],
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
                reasoning: None,
                tool_error: false,
            }
        }

        let mut assistant = text_msg(Role::Assistant, "Sure.");
        assistant.reasoning = Some(Reasoning {
            text: "plan the verification".into(),
        });
        assistant.tool_calls = vec![
            ToolCall {
                id: "c1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"command": "cargo test"}),
            },
            ToolCall {
                id: "c2".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "a.rs"}),
            },
        ];
        let mut tool_result = text_msg(Role::Tool, "total 0");
        tool_result.tool_call_id = Some("c1".into());
        tool_result.name = Some("bash".into());

        let msgs = vec![
            text_msg(
                Role::System,
                "You are helpful.\nworking directory: /tmp/proj",
            ),
            text_msg(Role::Other("developer".into()), "be terse"),
            text_msg(Role::User, "run the tests"),
            assistant,
            tool_result,
            text_msg(Role::User, "thanks"),
        ];

        let out = super::super::msg_entry::build_msg_entries(None, None, &msgs, true)
            .expect("fixture builds");
        assert_eq!(out.cwd_hint.as_deref(), Some("/tmp/proj"));
        let json = build_json_messages(&out.messages);

        let expected = serde_json::json!([
            {
                "role": "system",
                "content": "You are helpful.\nworking directory: /tmp/proj\n<environment>\nworking_directory: /tmp/proj\n</environment>"
            },
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "run the tests"},
            {
                "role": "assistant",
                "content": "Sure.",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": {"command": "cargo test"}}},
                    {"id": "c2", "type": "function", "function": {"name": "read", "arguments": {"path": "a.rs"}}}
                ],
                "reasoning_content": "plan the verification"
            },
            {"role": "tool", "content": "total 0"},
            {"role": "user", "content": "thanks"}
        ]);
        assert_eq!(
            serde_json::Value::Array(json.clone()),
            expected,
            "prompt JSON drifted — this breaks kv-cache prefix stability:\n{}",
            serde_json::to_string_pretty(&json).unwrap()
        );
    }

    #[test]
    fn plain_text_message_serializes_to_string_content() {
        let out = build_json_messages(&[entry("user", "hi", 0)]);
        assert_eq!(
            out,
            vec![serde_json::json!({"role": "user", "content": "hi"})]
        );
    }

    #[test]
    fn images_expand_to_structured_content_array_with_text_last() {
        let out = build_json_messages(&[entry("user", "look", 2)]);
        assert_eq!(
            out,
            vec![serde_json::json!({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "look"}
                ]
            })]
        );
    }

    #[test]
    fn empty_text_with_images_omits_text_part() {
        let out = build_json_messages(&[entry("user", "", 1)]);
        assert_eq!(
            out,
            vec![serde_json::json!({"role": "user", "content": [{"type": "image"}]})]
        );
    }

    #[test]
    fn tool_calls_and_reasoning_are_attached() {
        let mut e = entry("assistant", "", 0);
        e.tool_calls = Some(vec![serde_json::json!({"id": "c1"})]);
        e.reasoning_content = Some("because".to_string());
        let out = build_json_messages(&[e]);
        assert_eq!(
            out,
            vec![serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1"}],
                "reasoning_content": "because"
            })]
        );
    }
}
