// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Parse a single Mistral-native segment of the form `name[ARGS]{json}`.
///
/// Expects the `[TOOL_CALLS]` prefix to already be stripped. Handles both
/// complete and max_tokens-truncated JSON (uses `parse_truncated_json`
/// heuristics from the JSON fallback path if complete parsing fails).
///
/// Tolerates the `[ARGS]` delimiter being omitted — Mistral-Small-4 NVFP4
/// quantized sometimes emits `name{json}` directly (pass-8 regression vs
/// pass-7). When `[ARGS]` is missing, split on the first `{`.
pub(super) fn parse_mistral_native_call(segment: &str) -> Option<ToolCall> {
    let segment = segment.trim_start();
    let (name, json_slice) = if let Some(args_pos) = segment.find(MISTRAL_ARGS_TAG) {
        let name = segment[..args_pos].trim().to_string();
        let after_args = &segment[args_pos + MISTRAL_ARGS_TAG.len()..];
        let raw_args = after_args.trim();
        let json_start = raw_args.find('{').unwrap_or(0);
        (name, &raw_args[json_start..])
    } else if let Some(brace_pos) = segment.find('{') {
        // Tolerant path: `name{json}` with no [ARGS] delimiter.
        let name = segment[..brace_pos].trim().to_string();
        (name, &segment[brace_pos..])
    } else {
        return None;
    };
    if name.is_empty() {
        return None;
    }
    // Reject obvious non-identifier "names" (e.g. "Hello, world" before the
    // first `{`) — accept only alphanumeric + underscore, no whitespace.
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    {
        return None;
    }
    // Try to parse the complete JSON. If it doesn't parse (max_tokens cut the
    // response), fall back to the largest balanced-brace prefix we can find.
    let args = if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_slice) {
        serde_json::to_string(&v).unwrap_or_else(|_| "{}".into())
    } else if let Some(valid) = largest_balanced_json_prefix(json_slice) {
        valid.to_string()
    } else {
        "{}".to_string()
    };
    Some(ToolCall {
        id: next_tool_call_id(),
        call_type: "function".into(),
        function: FunctionCall {
            name,
            arguments: args,
        },
    })
}

/// Scan `s` starting at an opening `{` and return the byte offset
/// immediately after the matching closing `}`. Returns `None` if the
/// JSON object is incomplete or `s` does not start with `{`.
pub(super) fn find_balanced_json_end(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Find the largest prefix of `s` that is a valid balanced JSON object.
/// Used to salvage truncated Mistral tool calls when max_tokens cuts the
/// response mid-JSON. Tracks brace depth and string escaping.
fn largest_balanced_json_prefix(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;
    let mut last_valid: Option<usize> = None;
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    last_valid = Some(i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = last_valid?;
    let prefix = &s[..end];
    serde_json::from_str::<serde_json::Value>(prefix)
        .ok()
        .and_then(|v| serde_json::to_string(&v).ok())
}

/// Scan text for `identifier{json}` patterns and promote them to tool calls.
///
/// Matches bare function calls where the model dropped the format envelope:
///   `I'll check the weather for you.get_weather{"city": "Paris"}`
///
/// Only matches identifiers that look like function names (alphanumeric +
/// underscore + dash + dot) directly adjacent to a balanced `{...}` object.
/// Skips punctuation and non-identifier tokens so narrative text like
/// "Hello, world" {"foo": 1} won't match (the space before `{` breaks the
/// adjacency requirement).
///
/// Conservative: requires at least 2 identifier chars, requires the JSON to
/// start with `{` (not `[` or a primitive), and the identifier must be
/// followed immediately by `{` — no whitespace in between.
pub(super) fn parse_bare_identifier_json_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    let mut calls: Vec<ToolCall> = Vec::new();
    let mut content_parts: Vec<String> = Vec::new();
    let mut last_end = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            // Walk backwards to find identifier.
            let mut start = i;
            while start > 0 {
                let b = bytes[start - 1];
                if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.' {
                    start -= 1;
                } else {
                    break;
                }
            }
            let name_bytes = &bytes[start..i];
            // Require ≥2 chars and must start with a letter/underscore (not digit).
            if name_bytes.len() >= 2
                && (name_bytes[0].is_ascii_alphabetic() || name_bytes[0] == b'_')
                // Exclude common non-tool-call identifiers that can precede `{`
                // in narrative text: "function", "object", "params", and a few
                // JavaScript keywords. Keep this short; the JSON parse is the
                // actual validator.
                && !matches!(name_bytes,
                    b"function" | b"object" | b"params" | b"return" | b"const" |
                    b"let" | b"var" | b"if" | b"else" | b"for" | b"while" | b"class")
            {
                // Find balanced JSON end.
                let suffix = &text[i..];
                if let Some(end_rel) = find_balanced_json_end(suffix) {
                    let json_slice = &suffix[..end_rel];
                    // Try strict JSON first, then a relaxed repair pass for
                    // models (e.g. Gemma-4-26B NVFP4 Search) that emit
                    // `name{key:bareword string}` with unquoted keys/values.
                    let parsed = serde_json::from_str::<serde_json::Value>(json_slice)
                        .ok()
                        .or_else(|| {
                            let repaired = repair_bare_object_json(json_slice);
                            serde_json::from_str::<serde_json::Value>(&repaired).ok()
                        });
                    if let Some(v) = parsed {
                        // Must be an object (not just any balanced {...}).
                        if v.is_object() {
                            let name = std::str::from_utf8(name_bytes).unwrap_or("").to_string();
                            let args = serde_json::to_string(&v).unwrap_or_else(|_| "{}".into());
                            // Capture any content between last_end and `start`.
                            if start > last_end {
                                let chunk = text[last_end..start].trim();
                                if !chunk.is_empty() {
                                    content_parts.push(chunk.to_string());
                                }
                            }
                            calls.push(ToolCall {
                                id: next_tool_call_id(),
                                call_type: "function".into(),
                                function: FunctionCall {
                                    name,
                                    arguments: args,
                                },
                            });
                            last_end = i + end_rel;
                            i = last_end;
                            continue;
                        }
                    }
                }
            }
        }
        i += 1;
    }
    if !calls.is_empty() && last_end < text.len() {
        let tail = text[last_end..].trim();
        if !tail.is_empty() {
            content_parts.push(tail.to_string());
        }
    }
    let content = if content_parts.is_empty() {
        None
    } else {
        Some(content_parts.join("\n"))
    };
    (content, calls)
}

/// Repair common bare-JSON issues from models that emit unquoted keys
/// or unquoted bareword string values (e.g. Gemma-4-26B Search WARN
/// where output is `web_search{query:current Bitcoin price}`).
///
/// Strategy: walk top-level object members, quote any unquoted key
/// (alphanumeric+`_`+`-` followed by `:`), and quote any value that
/// is a sequence of identifier-like words (not a number, bool, null,
/// `[`, `{`, or already-quoted string). Conservative: a single bad
/// member breaks the whole repair (returns input unchanged) so we
/// don't accidentally produce worse JSON than we started with.
fn repair_bare_object_json(s: &str) -> String {
    let trimmed = s.trim();
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return s.to_string();
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut out = String::with_capacity(s.len() + 32);
    out.push('{');
    let mut first = true;
    let mut depth: i32 = 0;
    let mut start = 0usize;
    let bytes = inner.as_bytes();
    let mut in_str = false;
    // Split on top-level commas (depth=0, not inside string).
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if b == b'"' && (i == 0 || bytes[i - 1] != b'\\') {
                in_str = false;
            }
            continue;
        }
        if b == b'"' {
            in_str = true;
            continue;
        }
        if b == b'{' || b == b'[' {
            depth += 1;
            continue;
        }
        if b == b'}' || b == b']' {
            depth -= 1;
            continue;
        }
        if b == b',' && depth == 0 {
            if !append_repaired_member(&mut out, &inner[start..i], &mut first) {
                return s.to_string();
            }
            start = i + 1;
        }
    }
    if start < inner.len() && !append_repaired_member(&mut out, &inner[start..], &mut first) {
        return s.to_string();
    }
    out.push('}');
    out
}

fn append_repaired_member(out: &mut String, member: &str, first: &mut bool) -> bool {
    let m = member.trim();
    if m.is_empty() {
        return true;
    }
    let colon = match m.find(':') {
        Some(c) => c,
        None => return false,
    };
    let key = m[..colon].trim();
    let val = m[colon + 1..].trim();
    if key.is_empty() || val.is_empty() {
        return false;
    }
    // Quote key if not already quoted. Only allow simple identifier chars.
    let key_quoted = if key.starts_with('"') && key.ends_with('"') {
        key.to_string()
    } else if key
        .bytes()
        .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
    {
        format!("\"{key}\"")
    } else {
        return false;
    };
    // Quote value if it's a bareword phrase (not number/bool/null/array/object/string).
    let val_quoted = if val.starts_with('"')
        || val.starts_with('{')
        || val.starts_with('[')
        || val == "true"
        || val == "false"
        || val == "null"
        || val.parse::<f64>().is_ok()
    {
        val.to_string()
    } else {
        // Bareword phrase — quote it. Escape any embedded `"` as `\"`.
        let escaped = val.replace('"', "\\\"");
        format!("\"{escaped}\"")
    };
    if !*first {
        out.push(',');
    }
    *first = false;
    out.push_str(&key_quoted);
    out.push(':');
    out.push_str(&val_quoted);
    true
}
