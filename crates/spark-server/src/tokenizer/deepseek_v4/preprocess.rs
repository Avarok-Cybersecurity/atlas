// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::collections::HashMap;

fn role(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

fn last_user_index(messages: &[Value]) -> Option<usize> {
    messages
        .iter()
        .rposition(|m| matches!(role(m), Some("user" | "developer")))
}

pub(super) fn attach_top_level_tools(
    messages: &mut Vec<Value>,
    tools: Option<&[Value]>,
) -> Result<()> {
    let Some(tools) = tools.filter(|v| !v.is_empty()) else {
        return Ok(());
    };
    if messages.iter().any(|m| {
        m.get("tools")
            .and_then(Value::as_array)
            .is_some_and(|v| !v.is_empty())
    }) {
        return Ok(());
    }

    let tools = Value::Array(tools.to_vec());
    let recipient = messages
        .iter()
        .position(|m| role(m) == Some("system"))
        .or_else(|| messages.iter().position(|m| role(m) == Some("developer")));
    if let Some(index) = recipient {
        let message = &mut messages[index];
        message
            .as_object_mut()
            .context("DeepSeek-V4 message must be an object")?
            .insert("tools".into(), tools);
    } else {
        messages.insert(0, json!({"role": "system", "content": "", "tools": tools}));
    }
    Ok(())
}

pub(super) fn merge_tool_messages(messages: &[Value]) -> Result<Vec<Value>> {
    let mut merged: Vec<Value> = Vec::with_capacity(messages.len());
    for message in messages {
        let message_role = role(message).unwrap_or_default();
        match message_role {
            "tool" => {
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id").cloned().unwrap_or(Value::String(String::new())),
                    "content": message.get("content").cloned().unwrap_or(Value::String(String::new())),
                });
                if let Some(blocks) = merged.last_mut().and_then(user_content_blocks_mut) {
                    blocks.push(block);
                } else {
                    merged.push(json!({"role": "user", "content_blocks": [block]}));
                }
            }
            "user" => {
                let content = message
                    .get("content")
                    .cloned()
                    .unwrap_or(Value::String(String::new()));
                let block = json!({"type": "text", "text": content});
                let can_merge = merged.last().is_some_and(|m| {
                    role(m) == Some("user")
                        && m.get("content_blocks").and_then(Value::as_array).is_some()
                        && m.get("task").is_none()
                });
                if can_merge {
                    merged
                        .last_mut()
                        .and_then(user_content_blocks_mut)
                        .expect("checked content_blocks")
                        .push(block);
                } else {
                    let mut new_message = message.clone();
                    new_message
                        .as_object_mut()
                        .context("DeepSeek-V4 user message must be an object")?
                        .insert("content_blocks".into(), Value::Array(vec![block]));
                    merged.push(new_message);
                }
            }
            _ => merged.push(message.clone()),
        }
    }
    Ok(merged)
}

fn user_content_blocks_mut(message: &mut Value) -> Option<&mut Vec<Value>> {
    (role(message) == Some("user"))
        .then(|| message.get_mut("content_blocks")?.as_array_mut())
        .flatten()
}

pub(super) fn sort_tool_results_by_call_order(messages: &mut [Value]) {
    let mut call_order: HashMap<String, usize> = HashMap::new();
    for message in messages {
        match role(message) {
            Some("assistant") => {
                let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) else {
                    continue;
                };
                if tool_calls.is_empty() {
                    continue;
                }
                call_order.clear();
                for (index, call) in tool_calls.iter().enumerate() {
                    let id = call
                        .get("id")
                        .and_then(Value::as_str)
                        .or_else(|| {
                            call.get("function")
                                .and_then(|v| v.get("id"))
                                .and_then(Value::as_str)
                        })
                        .unwrap_or_default();
                    if !id.is_empty() {
                        call_order.insert(id.to_string(), index);
                    }
                }
            }
            Some("user") if !call_order.is_empty() => {
                let Some(blocks) = message
                    .get_mut("content_blocks")
                    .and_then(Value::as_array_mut)
                else {
                    continue;
                };
                let mut results: Vec<Value> = blocks
                    .iter()
                    .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
                    .cloned()
                    .collect();
                if results.len() < 2 {
                    continue;
                }
                results.sort_by_key(|b| {
                    b.get("tool_use_id")
                        .and_then(Value::as_str)
                        .and_then(|id| call_order.get(id).copied())
                        .unwrap_or(0)
                });
                let mut results = results.into_iter();
                for block in blocks {
                    if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                        *block = results.next().expect("same number of result blocks");
                    }
                }
            }
            _ => {}
        }
    }
}

pub(super) fn drop_thinking_messages(messages: &[Value]) -> Vec<Value> {
    let Some(last_user) = last_user_index(messages) else {
        return messages.to_vec();
    };
    let mut out = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        if index >= last_user
            || matches!(
                role(message),
                // `developer` is load-bearing here: msg_entry lowers it
                // deliberately for this encoder (preserve_developer_role), so
                // omitting it from the keep-list silently DELETED developer
                // instructions from the prompt, with no error.
                Some(
                    "user"
                        | "system"
                        | "developer"
                        | "tool"
                        | "latest_reminder"
                        | "direct_search_results"
                )
            )
        {
            out.push(message.clone());
            continue;
        }
        if let Some("assistant") = role(message) {
            let mut message = message.clone();
            if let Some(object) = message.as_object_mut() {
                object.remove("reasoning_content");
            }
            out.push(message);
        }
    }
    out
}
