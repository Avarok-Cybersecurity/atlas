// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Parse tool calls from completed model output.
///
/// Scans for `<tool_call></tool_call>` tags and auto-detects inner format
/// (JSON for hermes, XML for qwen3_coder).
///
/// Returns `(content, tool_calls)` where content is text outside tags.
pub fn parse_tool_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    // Strip <think>...</think> before parsing tool calls (matches vLLM behavior).
    // Tool calls inside thinking blocks are model deliberation, not real invocations.
    let text = if let Some(think_end) = text.find("</think>") {
        &text[think_end + 8..]
    } else {
        text
    };
    // MiniMax uses `<minimax:tool_call>` as the outer wrapper (different
    // tag from Qwen's `<tool_call>`). Normalize both wrappers to the
    // same outer form so the scanning loop below doesn't need
    // per-parser branches. Allocation only when the namespaced tag
    // actually appears.
    let owned_normalized: String;
    let text: &str = if text.contains("<minimax:tool_call>")
        || text.contains("</minimax:tool_call>")
        // F71 (2026-04-29): also accept the live-observed broken
        // BPE-merge variants (`<minimax:_call>` / `</minimax:_call>`)
        // and a few other near-miss forms. xgrammar's TagDispatch is
        // non-anchored across BPE merge boundaries, so the model can
        // emit `<minimax:_call>` instead of `<minimax:tool_call>`
        // without the matcher catching it. If the inner
        // `<invoke name="...">…</invoke>` block is still intact, we
        // can salvage a valid tool_call by normalising the broken
        // outer envelope back to the canonical form.
        || text.contains("<minimax:_call>")
        || text.contains("</minimax:_call>")
    {
        owned_normalized = text
            .replace("<minimax:tool_call>", "<tool_call>")
            .replace("</minimax:tool_call>", "</tool_call>")
            .replace("<minimax:_call>", "<tool_call>")
            .replace("</minimax:_call>", "</tool_call>");
        owned_normalized.as_str()
    } else {
        text
    };
    let mut calls = Vec::new();
    let mut content_parts = Vec::new();
    let mut rest = text;
    let mut idx = 0u32;

    // Returns the byte offset of the next `</tool_call>` close that is NOT
    // inside a `<parameter=...>...</parameter>` block. Qwen3-Coder tool
    // arguments may contain a literal `</tool_call>` substring inside a
    // string-typed parameter value; the bare `find` would terminate the
    // call body early and corrupt the parsed args. We scan with a
    // parameter-depth counter and only accept a close at depth 0.
    fn find_unescaped_tool_call_close(buf: &str) -> Option<usize> {
        let bytes = buf.as_bytes();
        let mut i = 0;
        let mut depth: i32 = 0;
        while i < bytes.len() {
            // Look at a candidate position: try to match `</tool_call>`,
            // `<parameter=`, or `</parameter>` starting at byte i.
            if buf[i..].starts_with("</tool_call>") && depth == 0 {
                return Some(i);
            }
            if buf[i..].starts_with("<parameter=") {
                depth += 1;
                i += "<parameter=".len();
                continue;
            }
            if buf[i..].starts_with("</parameter>") {
                if depth > 0 {
                    depth -= 1;
                }
                i += "</parameter>".len();
                continue;
            }
            // Advance one UTF-8 character. Fall back to one byte if the
            // boundary isn't found (defensive — bytes here are ASCII for
            // the markers, but parameter values can be UTF-8).
            let step = buf[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
            i += step;
        }
        None
    }

    loop {
        match rest.find("<tool_call>") {
            Some(start) => {
                let before = rest[..start].trim();
                if !before.is_empty() {
                    content_parts.push(before.to_string());
                }
                rest = &rest[start + 11..];
                match find_unescaped_tool_call_close(rest) {
                    Some(end) => {
                        if let Some(tc) = parse_one_call(rest[..end].trim(), idx) {
                            calls.push(tc);
                            idx += 1;
                        }
                        rest = &rest[end + 12..];
                    }
                    None => {
                        // No closing </tool_call> — likely truncated by max_tokens.
                        // Try to parse the truncated content as a tool call anyway.
                        // The JSON may be incomplete, but parse_one_call handles this.
                        if let Some(tc) = parse_one_call(rest.trim(), idx) {
                            calls.push(tc);
                        } else {
                            content_parts.push(format!("<tool_call>{rest}"));
                        }
                        break;
                    }
                }
            }
            None => {
                let after = rest.trim();
                if !after.is_empty() {
                    content_parts.push(after.to_string());
                }
                break;
            }
        }
    }
    // Fallback 00: Gemma-4 bare `fn_name{key:val,...}` at text start (no
    // wrapper). The gemma4.jinja tool-steering prefix injects
    // `<|tool_call>call:` into the prompt, so model output begins directly
    // with `fn_name{...}` — parse_bare_identifier_json_calls rejects because
    // keys are unquoted and strings use `<|"|>`. Apply gemma4_to_json
    // conversion first, then attempt JSON parse.
    //
    // Narrowly gated: text must START with an identifier directly followed
    // by `{`, so prose won't match. If successful, trailing content after
    // the balanced `}` is discarded (model often spews garbage tokens after
    // the call).
    if calls.is_empty() {
        let mut trimmed = text.trim_start();
        // Strip leading `namespace:` prefixes like `google:google_search{...}`
        // that Gemma-4-26B-A4B NVFP4A16 produces when its weakened
        // instruction-following lets pretraining priors ("google search"
        // canonical phrase) override the declared tool name. The trailing
        // identifier after the colon is the part that should match a tool
        // name (exactly or fuzzily via downstream logic).
        if let Some(colon) = trimmed.find(':') {
            let head = &trimmed[..colon];
            let is_ident = !head.is_empty()
                && head
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.');
            if is_ident && !head.starts_with("call") && !head.starts_with("_call") {
                // Only strip when there's another identifier followed by `{`
                // after the colon — keeps things like `call:fn{...}` for
                // the existing Gemma-4 native-format path untouched.
                let rest = &trimmed[colon + 1..];
                let rest_id_end = rest
                    .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '.')
                    .unwrap_or(rest.len());
                if rest_id_end >= 2 && rest.as_bytes().get(rest_id_end) == Some(&b'{') {
                    trimmed = rest;
                }
            }
        }
        let id_end = trimmed
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '.')
            .unwrap_or(trimmed.len());
        if id_end >= 2
            && trimmed.as_bytes().get(id_end) == Some(&b'{')
            && trimmed.as_bytes()[0].is_ascii_alphabetic()
        {
            let name = trimmed[..id_end].to_string();
            let args_part = &trimmed[id_end..];
            if let Some(end_rel) = find_balanced_json_end(args_part) {
                let json_slice = &args_part[..end_rel];
                let converted = gemma4_to_json(json_slice);
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&converted)
                    && v.is_object()
                {
                    let args = serde_json::to_string(&v).unwrap_or_else(|_| "{}".into());
                    calls.push(ToolCall {
                        id: next_tool_call_id(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name,
                            arguments: args,
                        },
                    });
                    return (None, calls);
                }
            }
        }
    }

    // Fallback 0a: Mistral native format `[TOOL_CALLS]name[ARGS]{json}`.
    // Must run before the Gemma-4 and JSON fallbacks so Mistral's bracketed
    // tokens don't get misparsed as bare JSON.
    if calls.is_empty() && text.contains(MISTRAL_TOOL_CALLS_TAG) {
        let (m_content, m_calls) = parse_mistral_native_calls(text);
        if !m_calls.is_empty() {
            return (m_content, m_calls);
        }
    }

    // Fallback 0b: bare Mistral-style `fn_name{json}` without [TOOL_CALLS]/[ARGS].
    // Pass-8 regression: Mistral-Small-4 NVFP4 emits e.g.
    //   `I'll check the current weather in Paris for you.get_weather{"city": "Paris"}`
    // The model dropped the [TOOL_CALLS] prefix in its quantized form but still
    // produces identifier-{json}. Scan for identifier patterns followed by a
    // balanced JSON object and promote them to tool calls.
    if calls.is_empty() {
        let (bc_content, bc_calls) = parse_bare_identifier_json_calls(text);
        if !bc_calls.is_empty() {
            return (bc_content, bc_calls);
        }
    }

    // Fallback 0: Gemma-4 native format `<|tool_call>call:fn{...}<tool_call|>`
    if calls.is_empty() {
        let mut g4_rest = text;
        let mut g4_content = Vec::new();
        loop {
            match g4_rest.find("<|tool_call>") {
                Some(start) => {
                    let before = g4_rest[..start].trim();
                    if !before.is_empty() {
                        g4_content.push(before.to_string());
                    }
                    g4_rest = &g4_rest[start + 12..]; // len("<|tool_call>") = 12
                    let end = g4_rest.find("<tool_call|>").unwrap_or(g4_rest.len());
                    let inner = g4_rest[..end].trim();
                    if let Some(tc) = parse_gemma4_native_call(inner) {
                        calls.push(tc);
                    }
                    g4_rest = if end < g4_rest.len() {
                        &g4_rest[end + 12..]
                    } else {
                        ""
                    };
                }
                None => {
                    let after = g4_rest.trim();
                    if !after.is_empty() {
                        g4_content.push(after.to_string());
                    }
                    break;
                }
            }
        }
        if !calls.is_empty() {
            let content = if g4_content.is_empty() {
                None
            } else {
                Some(g4_content.join("\n"))
            };
            return (content, calls);
        }
    }

    // Fallback 1: look for <tools>JSON</tools> wrapper (some models use this
    // instead of <tool_call>).
    if calls.is_empty() {
        let (tools_content, tools_calls) = parse_tools_tag_calls(text);
        if !tools_calls.is_empty() {
            return (tools_content, tools_calls);
        }
    }

    // Fallback 2a: bare `call:fn{...}` without <|tool_call> wrapper (Gemma-4 NVFP4).
    if calls.is_empty() {
        let trimmed = text.trim();
        if trimmed.starts_with("call:")
            || trimmed.starts_with("_call:")
            || trimmed.contains("\ncall:")
            || trimmed.contains("\n_call:")
        {
            let mut bare_content = Vec::new();
            for line in trimmed.split('\n') {
                let line = line.trim();
                if line.starts_with("call:") || line.starts_with("_call:") {
                    if let Some(tc) = parse_gemma4_native_call(line) {
                        calls.push(tc);
                    }
                } else if !line.is_empty() {
                    bare_content.push(line.to_string());
                }
            }
            if !calls.is_empty() {
                let content = if bare_content.is_empty() {
                    None
                } else {
                    Some(bare_content.join("\n"))
                };
                return (content, calls);
            }
        }
    }

    // Fallback 2b: look for bare <function> or <function= tags without <tool_call> wrapper.
    // Models at lower quantization sometimes omit the wrapper tags.
    if calls.is_empty() {
        let (bare_content, bare_calls) = parse_bare_function_calls(text);
        if !bare_calls.is_empty() {
            return (bare_content, bare_calls);
        }
    }

    // Fallback 3: look for JSON tool calls in code blocks or bare JSON.
    // When the model ignores the XML format entirely and writes tool invocations
    // as JSON in markdown code blocks or bare Hermes-style JSON, catch them here.
    // This is the "guaranteed catch-all" — if the model wrote a valid tool name
    // with arguments in any recognizable JSON format, we'll find it.
    if calls.is_empty() {
        let json_calls = parse_json_fallback_calls(text);
        if !json_calls.is_empty() {
            // Strip the JSON source from content
            let mut clean_content = text.to_string();
            for pattern in extract_json_code_blocks(text) {
                clean_content = clean_content.replace(&pattern, "");
            }
            let clean = clean_content.trim().to_string();
            let content = if clean.is_empty() { None } else { Some(clean) };
            return (content, json_calls);
        }
    }

    let content = if content_parts.is_empty() {
        None
    } else {
        Some(content_parts.join("\n"))
    };
    (content, calls)
}
