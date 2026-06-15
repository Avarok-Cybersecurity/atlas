// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Auto-detect and parse inner content of a `<tool_call>` block.
/// Tries Gemma-4 native, JSON (hermes), qwen3_coder XML, then tag-style XML fallback.
pub(super) fn parse_one_call(text: &str, idx: u32) -> Option<ToolCall> {
    // Try Gemma-4 native: call:fn_name{...} or _call:fn_name{...}
    if text.starts_with("call:") || text.starts_with("_call:") {
        return parse_gemma4_native_call(text);
    }
    // Try JSON (hermes format) — complete JSON first
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
        let name = normalize_tool_name(v.get("name")?.as_str()?);
        let args = v
            .get("arguments")
            .map(|a| {
                // If arguments is already a string (pre-serialized JSON), use directly.
                // Otherwise serialize the object/array value to a JSON string.
                if let Some(s) = a.as_str() {
                    s.to_string()
                } else {
                    serde_json::to_string(a).unwrap_or_else(|_| "{}".into())
                }
            })
            .unwrap_or_else(|| "{}".into());
        return Some(ToolCall {
            id: next_tool_call_id(),
            call_type: "function".into(),
            function: FunctionCall {
                name,
                arguments: args,
            },
        });
    }
    // Try truncated JSON — max_tokens may have cut the response mid-tool-call.
    // Extract function name and whatever arguments are available.
    if text.contains("\"name\"") && text.contains("\"arguments\"") {
        // Extract name via simple string search (avoid regex dependency)
        let name = extract_json_string(text, "name");
        if let Some(name) = name {
            let name = normalize_tool_name(&name);
            // Try to extract arguments — may be truncated JSON
            let args = if let Some(args_start) = text.find("\"arguments\"") {
                let after = &text[args_start + "\"arguments\"".len()..];
                let colon = after.find(':').map(|p| p + 1).unwrap_or(0);
                let args_text = after[colon..].trim();
                // Try full JSON parse first
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(args_text) {
                    serde_json::to_string(&v).unwrap_or_else(|_| "{}".into())
                } else {
                    // Truncated — try truncating at the last `}` or `]` boundary
                    let mut best = "{}".to_string();
                    for (i, ch) in args_text.char_indices().rev() {
                        if (ch == '}' || ch == ']')
                            && serde_json::from_str::<serde_json::Value>(&args_text[..=i]).is_ok()
                        {
                            best = args_text[..=i].to_string();
                            break;
                        }
                    }
                    best
                }
            } else {
                "{}".to_string()
            };
            return Some(ToolCall {
                id: next_tool_call_id(),
                call_type: "function".into(),
                function: FunctionCall {
                    name,
                    arguments: args,
                },
            });
        }
    }
    // Try MiniMax XML invoke/parameter shape: <invoke name="NAME"><parameter name="K">V</parameter></invoke>
    if let Some(tc) = parse_minimax_xml_call(text, idx) {
        return Some(tc);
    }
    // Try XML attribute-style (qwen3_coder format): <function=NAME>
    if let Some(tc) = parse_qwen3_coder_call(text, idx) {
        return Some(tc);
    }
    // Try XML tag-style fallback: <function>NAME</function>
    parse_tag_style_call(text, idx)
}

/// Parse MiniMax XML inner content:
/// `<invoke name="NAME"><parameter name="K">V</parameter>...</invoke>`.
///
/// The outer `<minimax:tool_call>` wrapper is already stripped by
/// `parse_tool_calls` (normalized to plain `<tool_call>` on entry and
/// then dropped by the outer loop). This function only sees the
/// inner body.
fn parse_minimax_xml_call(text: &str, idx: u32) -> Option<ToolCall> {
    let _ = idx;
    let invoke_start = text
        .find("<invoke name=\"")
        .or_else(|| text.find("<invoke name='"))?;
    let after = &text[invoke_start + "<invoke name=".len()..];
    let quote = after.as_bytes().first().copied()?;
    if quote != b'"' && quote != b'\'' {
        return None;
    }
    let name_start = 1; // skip opening quote
    let name_end = after[name_start..]
        .find(quote as char)
        .map(|p| p + name_start)?;
    let func_name = normalize_tool_name(&after[name_start..name_end]);
    if func_name.is_empty() {
        return None;
    }

    let mut rest = &after[name_end + 1..];
    // Skip to end of `<invoke ...>` open tag.
    if let Some(gt) = rest.find('>') {
        rest = &rest[gt + 1..];
    } else {
        return None;
    }

    let mut args = serde_json::Map::new();
    while let Some(p) = rest.find("<parameter name=") {
        rest = &rest[p + "<parameter name=".len()..];
        let q = rest.as_bytes().first().copied()?;
        if q != b'"' && q != b'\'' {
            break;
        }
        let key_start = 1;
        let key_end = rest[key_start..].find(q as char).map(|p| p + key_start)?;
        let param_name = rest[key_start..key_end].trim().to_string();
        rest = &rest[key_end + 1..];
        // Skip to end of `<parameter ...>` open tag.
        let gt = rest.find('>')?;
        rest = &rest[gt + 1..];

        // Stop at the LEAST of: `</parameter>` (proper close),
        // `<parameter=` (next param — recovery for a missing close),
        // or `</function>` (function ended without closing this param).
        // Without the recovery cases, a missing `</parameter>` swallows
        // every subsequent param into one giant value (vllm #38158
        // regression "streaming_missing_closing_tag").
        let proper = rest.find("</parameter>");
        let next_param = rest.find("<parameter=");
        let func_close = rest.find("</function>");
        let mut val_end = rest.len();
        let mut consumed_close = false;
        if let Some(p) = proper {
            val_end = p;
            consumed_close = true;
        }
        for cand in [next_param, func_close].into_iter().flatten() {
            if cand < val_end {
                val_end = cand;
                consumed_close = false;
            }
        }
        let raw_value = rest[..val_end].trim();
        rest = if consumed_close {
            &rest[val_end + "</parameter>".len()..]
        } else if val_end < rest.len() {
            // Recovery path: leave the next `<parameter=` / `</function>`
            // in place so the surrounding loop sees it.
            &rest[val_end..]
        } else {
            ""
        };

        // Keep as string — matches qwen3_coder behavior. Clients
        // validate against the tool's JSON schema and coerce.
        args.insert(param_name, serde_json::Value::String(raw_value.to_string()));
    }

    // F80b (2026-04-30): empty path-like parameters are almost always a
    // model self-truncation bug — the model emits long content for the
    // first parameter then "completes" the JSON with `<parameter
    // name="filePath"></parameter>` and the post-write tool fails with
    // EISDIR. Live opencode session ses_22136d7a6ffekfmY4valiZYmsV
    // looped 8 turns on this. Drop the call here so the streaming
    // detector treats it as a parse failure (no tool emitted, response
    // ends with finish_reason=stop). F78 still validates if this slips
    // through — both layers fail closed.
    //
    // Conservative scope: only path-like keys on file-mutation tools
    // get this treatment; bash/curl/etc. are unaffected.
    const PATH_KEYS: &[&str] = &["file_path", "filePath", "path"];
    let is_write_tool = matches!(
        func_name.as_str(),
        "Write" | "write" | "Edit" | "edit" | "MultiEdit" | "multiEdit" | "multi_edit",
    );
    if is_write_tool {
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(v)) = args.get(*key)
                && v.trim().is_empty()
            {
                tracing::warn!(
                    tool = %func_name,
                    key = key,
                    "F80b: dropping minimax_xml call with empty required path; \
                     model self-truncation"
                );
                return None;
            }
        }
    }

    Some(ToolCall {
        id: next_tool_call_id(),
        call_type: "function".into(),
        function: FunctionCall {
            name: func_name,
            arguments: serde_json::to_string(&serde_json::Value::Object(args))
                .unwrap_or_else(|_| "{}".into()),
        },
    })
}

/// Extract every `<invoke name="...">…</invoke>` block from a MiniMax
/// envelope's inner content. The streaming detector hands us the body
/// between `<minimax:tool_call>` (or its broken `<minimax:_call>`
/// variant) and the matching close tag. MiniMax's documented format
/// allows multiple `<invoke>` blocks in one envelope — F75
/// (2026-04-29): live-observed in opencode session
/// `ses_224cc79f4ffeUtq7NFV9YMTVMH` where the model emitted two
/// `<invoke>bash` blocks (mkdir src + mkdir tests) and the detector
/// dropped the second, leaving `has_tool_calls=false`.
pub(crate) fn parse_minimax_xml_calls_all(text: &str) -> Vec<ToolCall> {
    let mut out = Vec::new();
    let mut rest = text;
    let mut idx: u32 = 0;
    while let Some(start) = rest.find("<invoke name=") {
        let chunk = &rest[start..];
        // Bound the search for this invoke to its own `</invoke>` close
        // so a malformed second block can't run into the first's body.
        let end = match chunk.find("</invoke>") {
            Some(e) => e + "</invoke>".len(),
            None => chunk.len(),
        };
        if let Some(tc) = parse_minimax_xml_call(&chunk[..end], idx) {
            out.push(tc);
            idx += 1;
        }
        rest = &chunk[end..];
    }
    out
}

/// Parse qwen3_coder XML: `<function=NAME><parameter=KEY>VALUE</parameter></function>`
/// Extract a JSON string value by key from potentially truncated JSON.
/// Simple substring search — no full JSON parsing needed.
fn extract_json_string(text: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let key_pos = text.find(&pattern)?;
    let after_key = &text[key_pos + pattern.len()..];
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let val_start = 1; // skip opening quote
    let mut i = val_start;
    let bytes = after_colon.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 2; // skip escaped char
        } else if bytes[i] == b'"' {
            return Some(after_colon[val_start..i].to_string());
        } else {
            i += 1;
        }
    }
    None
}
