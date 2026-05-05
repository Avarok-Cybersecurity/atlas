// SPDX-License-Identifier: AGPL-3.0-only

//! Schema-driven last-resort salvage for tool-call intent expressed
//! in the wrong syntax.
//!
//! Replaces three special-case salvage functions that used to live
//! in `api.rs`:
//!  - `salvage_bash_from_loop` (looks for ```bash fences)
//!  - `salvage_xml_tool_call` (looks for `<NAME>...</NAME>` blocks)
//!  - `salvage_write_from_prose` (looks for filename headers + body)
//!
//! Each was hardcoded to one tool's name and one parameter shape.
//! New tool definitions got no salvage support; new dictation
//! styles needed a new function.
//!
//! ## Method
//!
//! Given the response content and the list of declared tools, walk
//! every tool and try a small fixed set of structured-data shapes
//! against it. Each shape extracts `(parameter_name, value)` pairs
//! using the tool's own JSON schema as the parser guide:
//!
//!  - **XML form**: `<TOOL>...<PARAM>VALUE</PARAM>...</TOOL>` —
//!    inner tag names are matched to schema property names
//!    case-insensitively.
//!  - **Fenced-code form**: ` ```LANG\nBODY\n``` ` — when LANG
//!    matches the tool name (or a known alias) AND the tool has
//!    exactly ONE required string parameter (e.g. bash→command),
//!    the body becomes that parameter's value.
//!  - **Header+body form**: a standalone "filename-like" line
//!    followed by a body — when the tool has both a path-like
//!    parameter (`path`/`file_path`/`filePath`) and a content-like
//!    parameter (`content`/`text`/`body`).
//!  - **Heredoc form**: `cat > /path << 'EOF' ... EOF` — same
//!    target shape as Header+body.
//!
//! No tool's name or parameter list is hardcoded. A new declared
//! tool with the same SHAPE (e.g. a `WriteFile` with `path` +
//! `content` parameters) automatically gets salvage support
//! through the Header+body matcher; a new tool with a different
//! shape gets salvage support if it fits any matcher's template.

mod extract;
mod extract_more;
mod shape;

#[cfg(test)]
mod tests;

use crate::tool_parser::{ToolCall, ToolDefinition};
use shape::ToolShape;

/// Public entry point — returns synthesised tool calls in
/// document order. The caller is responsible for gating: typically
/// only invoke this when the model produced no native tool_calls
/// AND the content is non-empty AND the tool registry is non-empty.
pub fn salvage(content: &str, tools: &[ToolDefinition]) -> Vec<ToolCall> {
    if content.trim().is_empty() || tools.is_empty() {
        return Vec::new();
    }
    let matchers: Vec<ToolShape<'_>> = tools.iter().map(ToolShape::new).collect();

    let mut out: Vec<ToolCall> = Vec::new();
    let emit_unique = |tc: ToolCall, out: &mut Vec<ToolCall>| {
        // Dedupe by exact (name, args) match OR by (name, file_path)
        // when both calls carry one. The file_path-based dedup
        // catches the case where two extractors recover the same
        // file-write intent with different `content` bodies (e.g.
        // header_body extracts everything after the path line as
        // body, while file_content_pair takes only the content
        // between <content>...</content> — both emit Write calls
        // for the same path).
        let new_path = parsed_file_path(&tc.function.arguments);
        let already = out.iter().any(|prev| {
            if prev.function.name != tc.function.name {
                return false;
            }
            if prev.function.arguments == tc.function.arguments {
                return true;
            }
            match (parsed_file_path(&prev.function.arguments), &new_path) {
                (Some(p), Some(np)) => p == *np,
                _ => false,
            }
        });
        if !already {
            out.push(tc);
        }
    };

    // Try shapes in priority order. Higher-precision shapes first
    // (XML wraps everything explicitly; fenced+heredoc are tied;
    // header+body and bare-invocation are the loosest).
    for tc in extract::extract_xml(content, &matchers) {
        emit_unique(tc, &mut out);
    }
    for tc in extract::extract_fenced(content, &matchers) {
        emit_unique(tc, &mut out);
    }
    for tc in extract::extract_heredoc(content, &matchers) {
        emit_unique(tc, &mut out);
    }
    // `<file>PATH</file><content>BODY</content>` pair — opencode's
    // subagent-task syntax that the model mimics in prose (2026-
    // 04-25 dump seq=104..111). The model wraps file-write intent
    // in a `<task><description>...</description><file>...</file>
    // <content>...</content></task>` envelope; the parent `<task>`
    // tag isn't a declared tool name so `extract_xml` misses it.
    // Runs BEFORE `extract_header_body` so the inner-pair body is
    // preferred over the looser path-header heuristic when both
    // shapes match the same file_path.
    for tc in extract_more::extract_file_content_pair(content, &matchers) {
        emit_unique(tc, &mut out);
    }
    for tc in extract::extract_header_body(content, &matchers) {
        emit_unique(tc, &mut out);
    }
    // Outer-boundary prose-pseudotool catch (#3, 2026-04-25). Looks
    // for "<ToolName> <arg>" or "<ToolName>(<arg>)" lines that ARE
    // the bare prose form of a tool invocation. Catches the failure
    // mode where the model emits "Write /tmp/x.toml\n\n[package]…"
    // or "Bash ls -la" outside any structured wrapper. Lower
    // priority than the other shapes — only fires if nothing else
    // already extracted that intent.
    for tc in extract_more::extract_bare_tool_invocation(content, &matchers) {
        emit_unique(tc, &mut out);
    }
    out
}

/// Extract the `file_path` value from a tool-call's `arguments`
/// JSON string. Used by the dedupe layer to recognise that two
/// extractors recovered the same file-write intent with different
/// `content` bodies. Returns `None` if the JSON is malformed or has
/// no `file_path` / `filePath` key.
fn parsed_file_path(args: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args).ok()?;
    let obj = v.as_object()?;
    obj.get("file_path")
        .or_else(|| obj.get("filePath"))
        .or_else(|| obj.get("path"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}
