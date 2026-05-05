// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Back-compat wrapper around the pipeline for existing callers.
/// Models sometimes output tool calls without the `<tool_call>` wrapper,
/// especially at lower quantization levels. This catches those cases.
pub(super) fn parse_bare_function_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    ToolCallPipeline::bare_function_default().run(text)
}

// ── JSON fallback tool call parser ──

/// Extract markdown code blocks from text (returns the full ```...``` strings).
pub(super) fn extract_json_code_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("```") {
        let after_fence = &rest[start + 3..];
        // Skip optional language tag (```json, ```\n, etc.)
        let content_start = after_fence.find('\n').map_or(0, |p| p + 1);
        if let Some(end) = after_fence[content_start..].find("```") {
            let full_block = &rest[start..start + 3 + content_start + end + 3];
            blocks.push(full_block.to_string());
            rest = &rest[start + 3 + content_start + end + 3..];
        } else {
            break;
        }
    }
    blocks
}

/// Parse tool calls from JSON embedded in code blocks or bare JSON.
/// Catches cases where the model writes tool invocations in wrong format.
///
/// Recognized formats:
/// 1. `{"name": "Write", "arguments": {...}}` (Hermes-style)
/// 2. `["Write", {...}]` (Array-style)
/// 3. Code-block wrapped variants of the above
pub(super) fn parse_json_fallback_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();

    // Collect candidate JSON strings from code blocks and bare text
    let mut candidates: Vec<String> = Vec::new();
    for block in extract_json_code_blocks(text) {
        // Strip ``` fences and language tag
        let inner = block.trim_start_matches("```");
        let inner = if let Some(pos) = inner.find('\n') {
            &inner[pos + 1..]
        } else {
            inner
        };
        let inner = inner.trim_end_matches("```").trim();
        if !inner.is_empty() {
            candidates.push(inner.to_string());
        }
    }

    // Also scan for bare JSON objects/arrays in the text (outside code blocks)
    for line in text.lines() {
        let trimmed = line.trim();
        if (trimmed.starts_with('{') && trimmed.ends_with('}'))
            || (trimmed.starts_with('[') && trimmed.ends_with(']'))
        {
            candidates.push(trimmed.to_string());
        }
    }

    // Also scan for JSON objects embedded in prose. Nemotron-Super outputs
    // narrated tool calls like: `Here is the call: {"name": "get_weather",
    // "arguments": {"city": "Paris"}}.` — the JSON isn't on its own line so
    // the line-based check above misses it. Walk the text character by
    // character looking for `{"name"` and extract balanced JSON starting
    // there. Only used as a last-resort fallback; bails out if the JSON
    // doesn't balance cleanly.
    {
        let bytes = text.as_bytes();
        let needle = b"{\"name\"";
        let mut search_start = 0;
        while let Some(rel) = text[search_start..].find("{\"name\"") {
            let start = search_start + rel;
            // Find balanced closing brace.
            let mut depth = 0i32;
            let mut in_str = false;
            let mut escape = false;
            let mut end = start;
            for i in start..bytes.len() {
                let c = bytes[i];
                if escape {
                    escape = false;
                    end = i;
                    continue;
                }
                if in_str {
                    match c {
                        b'\\' => escape = true,
                        b'"' => in_str = false,
                        _ => {}
                    }
                    end = i;
                    continue;
                }
                match c {
                    b'"' => in_str = true,
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
                end = i + 1;
            }
            if depth == 0 && end > start {
                let slice = &text[start..end];
                // Only add as candidate if it contains "arguments" and parses.
                if slice.contains("\"arguments\"")
                    && serde_json::from_str::<serde_json::Value>(slice).is_ok()
                {
                    candidates.push(slice.to_string());
                }
                search_start = end;
            } else {
                // Couldn't find balance — skip past this `{"name"` occurrence.
                search_start = start + needle.len();
            }
        }
    }

    // Dedupe — line-based and walk-based scans can both pick up the same
    // tool-call JSON when the model emits a leading brace + newline before
    // the proper call (Nemotron-Super: `{\n{"name":"get_weather",...}`).
    candidates.sort();
    candidates.dedup();

    for candidate in &candidates {
        // Try Hermes-style: {"name": "tool", "arguments": {...}}
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(candidate) {
            if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
                let args = obj
                    .get("arguments")
                    .map(|a| serde_json::to_string(a).unwrap_or_default())
                    .unwrap_or_else(|| "{}".to_string());
                calls.push(ToolCall {
                    id: next_tool_call_id(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.to_string(),
                        arguments: args,
                    },
                });
                continue;
            }

            // Try array-style: ["tool_name", {"param": "value"}]
            if let Some(arr) = obj.as_array()
                && arr.len() == 2
                && let (Some(name), Some(_args_obj)) = (arr[0].as_str(), arr[1].as_object())
            {
                calls.push(ToolCall {
                    id: next_tool_call_id(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.to_string(),
                        arguments: serde_json::to_string(&arr[1]).unwrap_or_default(),
                    },
                });
                continue;
            }
        }

        // Try multiline Hermes-style (JSON spanning multiple lines)
        // Look for {"name": "..." patterns in multi-line candidates
        if candidate.contains("\"name\"")
            && candidate.contains("\"arguments\"")
            && let Ok(obj) = serde_json::from_str::<serde_json::Value>(candidate)
            && let Some(name) = obj.get("name").and_then(|n| n.as_str())
        {
            let args = obj
                .get("arguments")
                .map(|a| serde_json::to_string(a).unwrap_or_default())
                .unwrap_or_else(|| "{}".to_string());
            calls.push(ToolCall {
                id: next_tool_call_id(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: args,
                },
            });
        }
    }

    calls
}

// ── Streaming tool call detector ──
