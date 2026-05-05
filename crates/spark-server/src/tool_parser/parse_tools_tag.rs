// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Fallback: scan for bare `<function>` or `<function=` tags not wrapped in `<tool_call>`.
///
/// Parse tool calls wrapped in `<tools>JSON</tools>` tags.
/// Some models (e.g., Qwen3.5-35B-A3B at NVFP4) use `<tools>` instead of `<tool_call>`.
pub(super) fn parse_tools_tag_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut content_parts = Vec::new();
    let mut rest = text;
    let mut idx = 0u32;
    loop {
        match rest.find("<tools>") {
            Some(start) => {
                let before = rest[..start].trim();
                if !before.is_empty() {
                    content_parts.push(before.to_string());
                }
                rest = &rest[start + 7..]; // len("<tools>") = 7
                match rest.find("</tools>") {
                    Some(end) => {
                        if let Some(tc) = parse_one_call(rest[..end].trim(), idx) {
                            calls.push(tc);
                            idx += 1;
                        }
                        rest = &rest[end + 8..]; // len("</tools>") = 8
                    }
                    None => {
                        if let Some(tc) = parse_one_call(rest.trim(), idx) {
                            calls.push(tc);
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
    let content = if content_parts.is_empty() {
        None
    } else {
        Some(content_parts.join("\n"))
    };
    (content, calls)
}

// ── Bare-function parsing pipeline ──────────────────────────────────────
//
// Models sometimes output tool calls without the `<tool_call>` wrapper (common
// at lower quantization levels), or outright mangle the format (e.g. swapping
// `<function=X>` for `<parameter=X>`). We recognize these via a pipeline of
// passes: PRIMARY passes match canonical formats the model was prompted to
// emit; SALVAGE passes heuristically recover malformed output that expressed
// clear tool-calling intent but got the syntax wrong. Primary passes run
// first — the first one that extracts any call wins. If none do, salvage
// passes run in order; the first salvage that extracts a call wins. Each
// pass is a `ToolCallPass` impl and mutates a shared `PassState`.
