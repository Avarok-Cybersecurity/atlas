// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Parse all Mistral native tool calls from a completed response text.
/// Returns `(content_before_first_call, tool_calls)`.
pub(super) fn parse_mistral_native_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut content: Option<String> = None;
    let first_tag = match text.find(MISTRAL_TOOL_CALLS_TAG) {
        Some(p) => p,
        None => return (None, calls),
    };
    let before = text[..first_tag].trim();
    if !before.is_empty() {
        content = Some(before.to_string());
    }
    // Split on the tag; first element is empty (everything before the first tag
    // was already captured as content). Remaining elements are each call's
    // `name[ARGS]{json}` segment.
    let segments = text[first_tag..].split(MISTRAL_TOOL_CALLS_TAG).skip(1);
    for segment in segments {
        if segment.trim().is_empty() {
            continue;
        }
        if let Some(tc) = parse_mistral_native_call(segment) {
            calls.push(tc);
        }
    }
    (content, calls)
}

/// Format a JSON value in Gemma-4 native notation.
pub(super) fn format_gemma4_value(out: &mut String, val: &serde_json::Value) {
    match val {
        serde_json::Value::String(s) => {
            out.push_str("<|\"|>");
            out.push_str(s);
            out.push_str("<|\"|>");
        }
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => out.push_str(&n.to_string()),
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                format_gemma4_value(out, item);
            }
            out.push(']');
        }
        serde_json::Value::Object(obj) => {
            out.push('{');
            let mut first = true;
            for (key, v) in obj {
                if !first {
                    out.push(',');
                }
                first = false;
                out.push_str(key);
                out.push(':');
                format_gemma4_value(out, v);
            }
            out.push('}');
        }
    }
}

/// Parse a Gemma-4 native tool call: `call:fn_name{key:val,...}`
/// Converts native `<|"|>` delimited values to JSON.
pub(super) fn parse_gemma4_native_call(text: &str) -> Option<ToolCall> {
    let text = text.trim();
    // Accept both `call:` and `_call:` (sentencepiece `▁call:` → `_call:` in some tokenizer decodes)
    let rest = text
        .strip_prefix("call:")
        .or_else(|| text.strip_prefix("_call:"))?;
    let brace_pos = rest.find('{')?;
    let name = normalize_tool_name(&rest[..brace_pos]);
    if name.is_empty() {
        return None;
    }

    // Extract the arguments between { and the last }
    let args_str = &rest[brace_pos..];
    // Convert native format to JSON:
    // 1. Replace <|"|> with "
    // 2. Quote unquoted keys
    let json_str = gemma4_to_json(args_str);
    let arguments = if let Ok(_v) = serde_json::from_str::<serde_json::Value>(&json_str) {
        json_str
    } else {
        // Fallback: try to salvage partial content
        "{}".to_string()
    };

    Some(ToolCall {
        id: next_tool_call_id(),
        call_type: "function".into(),
        function: FunctionCall { name, arguments },
    })
}

/// Convert Gemma-4 native `{key:<|"|>val<|"|>,...}` to JSON `{"key":"val",...}`.
pub(super) fn gemma4_to_json(native: &str) -> String {
    // Replace <|"|> pairs with "
    let s = native.replace("<|\"|>", "\"");
    // Now we need to quote unquoted keys: `key:` → `"key":`
    // Keys are word characters before a `:` that aren't already quoted
    let mut result = String::with_capacity(s.len() + 32);
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '"' {
            // Skip quoted strings entirely
            result.push('"');
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    result.push(chars[i]);
                    i += 1;
                }
                result.push(chars[i]);
                i += 1;
            }
            if i < chars.len() {
                result.push('"');
                i += 1;
            }
        } else if chars[i].is_alphanumeric() || chars[i] == '_' {
            // Potential unquoted key — accumulate until we see `:` or non-key char
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word = &s[start..i];
            if i < chars.len() && chars[i] == ':' {
                // This is a key — quote it
                result.push('"');
                result.push_str(word);
                result.push('"');
            } else {
                // Not a key — check if it's a keyword (true/false/null)
                match word {
                    "true" | "false" | "null" => result.push_str(word),
                    _ => {
                        // Unknown bare word — quote it as a string
                        result.push('"');
                        result.push_str(word);
                        result.push('"');
                    }
                }
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    result
}

/// Shared tool_choice instruction appended to all system prompts.
pub(super) fn append_tool_choice_instruction(prompt: &mut String, tool_choice: &ToolChoice) {
    match tool_choice {
        ToolChoice::Mode(s) if s == "required" => {
            prompt.push_str(
                "\n\n<IMPORTANT>\nYou MUST call at least one function. \
                 Do NOT respond with text — respond ONLY with a <tool_call> block.\n</IMPORTANT>",
            );
        }
        ToolChoice::Specific { function } => {
            prompt.push_str(&format!(
                "\n\n<IMPORTANT>\nYou MUST call the '{}' function. \
                 Do NOT respond with text — respond ONLY with a <tool_call> block \
                 calling '{}'.\n</IMPORTANT>",
                function.name, function.name,
            ));
        }
        _ => {}
    }
}

/// Render the `<tools>` body for a parser's `system_prompt()`.
///
/// When TSCG is enabled (MODEL.toml `[behavior].tscg`), returns the
/// compact function-signature block. Otherwise calls `render_json` —
/// the parser's existing JSON serialization — so the TSCG-off path is
/// byte-identical to before this helper existed.
pub(super) fn tool_list_body(
    tools: &[ToolDefinition],
    render_json: impl FnOnce() -> String,
) -> String {
    if crate::tscg::tscg_enabled() {
        crate::tscg::compile_tools(tools)
    } else {
        render_json()
    }
}

// ── Output parsing (format-agnostic) ──
