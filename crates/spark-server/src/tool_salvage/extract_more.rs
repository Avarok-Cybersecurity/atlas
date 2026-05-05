// SPDX-License-Identifier: AGPL-3.0-only

use super::shape::ToolShape;
use crate::tool_parser::{FunctionCall, ToolCall};

/// `<ToolName> <args>` — bare-prose tool invocation (#3, 2026-04-25).
///
/// Catches failure mode: the model emits a tool name followed by
/// arguments as plain prose, escaping XGrammar's `triggered_tags`
/// boundary because the trigger (`<function=` etc.) was never
/// emitted. Examples:
///
/// ```text
/// "Write /tmp/x.toml\n\n[package]\nname=…"
/// "Bash ls -la /tmp"
/// "Read /path/to/file"
/// ```
///
/// Strategy:
///  - For tools with a single required string param (bash, read):
///    treat everything after the tool-name token on the same line as
///    that param's value.
///  - For tools with path + content shape (write, edit): use first
///    token after tool-name as the path; the next blank-line-
///    delimited block as content.
///
/// Strict gates (very conservative — this is the loosest shape):
///  - Tool name MUST appear at the start of a line (after whitespace
///    only) AND be followed by exactly one space + non-empty arg.
///  - Tool name length >= 3 (avoids matching prose words like "ls"
///    inside sentences).
///  - Inline mentions ("I will Write the file") are rejected: the
///    line MUST start with the tool name, not contain it.
///  - For path+content shape, body must still be >= 40 chars.
pub(super) fn extract_bare_tool_invocation(content: &str, matchers: &[ToolShape]) -> Vec<ToolCall> {
    let mut out = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();
        // Skip blank lines and lines starting with markdown markers.
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with('-')
            || trimmed.starts_with('*')
            || trimmed.starts_with("//")
        {
            i += 1;
            continue;
        }
        // Try each declared tool — match if the line starts with
        // the tool's name (case-insensitive) followed by a space.
        let mut matched: Option<(&ToolShape, String, String)> = None;
        for m in matchers {
            let name = m.name();
            if name.len() < 3 {
                continue;
            }
            // Match either: `<NAME> <arg>` (whitespace-separated) OR
            // `<NAME>(<arg>)` (paren form Claude Code's UI uses).
            // Reject inline mentions where NAME is not at line start.
            let lower_trimmed = trimmed.to_ascii_lowercase();
            let lower_name = name.to_ascii_lowercase();
            if !lower_trimmed.starts_with(&lower_name) {
                continue;
            }
            let after = &trimmed[name.len()..];
            // The character right after the name must be a separator
            // (space, tab, or `(`) — anything else means it's a
            // longer identifier (`Bashing`, `Writer`, …) not our tool.
            let next_ch = after.chars().next();
            let raw_arg: String = match next_ch {
                Some(c) if c.is_ascii_whitespace() => after.trim_start().to_string(),
                Some('(') => {
                    // Find the matching close paren on the same line.
                    let body = &after[1..];
                    let close = body.find(')').unwrap_or(body.len());
                    body[..close].to_string()
                }
                _ => continue,
            };
            // Strip trailing punctuation/colon that prose attaches.
            let raw_arg = raw_arg.trim_end_matches([':', ',', ';']).trim().to_string();
            if raw_arg.is_empty() {
                continue;
            }
            matched = Some((m, name.to_string(), raw_arg));
            break;
        }
        let Some((m, _name_lit, arg_text)) = matched else {
            i += 1;
            continue;
        };

        // Path + content shape: arg_text is the path; next blank-
        // line-delimited block is the content.
        if let Some((path_prop, content_prop)) = m.path_and_content() {
            // Use just the first whitespace-separated token of arg_text as path
            // (the rest of the line, if any, is treated as a comment).
            let path = arg_text.split_whitespace().next().unwrap_or("").to_string();
            if path.is_empty() {
                i += 1;
                continue;
            }
            // Skip blank lines, then collect body until the next blank line OR
            // a line that itself starts a new bare-tool invocation.
            let mut j = i + 1;
            while j < lines.len() && lines[j].trim().is_empty() {
                j += 1;
            }
            let body_start = j;
            while j < lines.len() {
                let lt = lines[j].trim_start();
                // Stop at the next bare-tool invocation candidate.
                let stops_here = matchers.iter().any(|m2| {
                    if m2.name().len() < 3 {
                        return false;
                    }
                    let first = lt.split_whitespace().next().unwrap_or("");
                    first.eq_ignore_ascii_case(m2.name())
                });
                if stops_here {
                    break;
                }
                j += 1;
            }
            let body = lines[body_start..j].join("\n").trim_end().to_string();
            if body.len() >= 40 {
                let mut args = serde_json::Map::new();
                args.insert(path_prop, serde_json::Value::String(path));
                args.insert(content_prop, serde_json::Value::String(body));
                if let Some(tc) = synthesise(m.name(), &args, "bare_invocation") {
                    out.push(tc);
                }
            }
            i = j;
            continue;
        }

        // Single-required-string shape (e.g. bash): arg_text is the
        // whole arg value.
        if let Some(prop) = m.single_required_string() {
            // Require arg to look like an actual command/path — at
            // least 2 chars, not pure punctuation.
            if arg_text.len() >= 2 && arg_text.chars().any(|c| c.is_alphanumeric() || c == '/') {
                let mut args = serde_json::Map::new();
                args.insert(prop, serde_json::Value::String(arg_text));
                if let Some(tc) = synthesise(m.name(), &args, "bare_invocation") {
                    out.push(tc);
                }
            }
            i += 1;
            continue;
        }

        i += 1;
    }
    out
}

/// `<file>PATH</file><content>BODY</content>` pair extractor.
///
/// Reference failure: dump 2026-04-25 seq=104..111 (opencode session
/// with calc-test60 path drift). The model wrapped file-write intent
/// in opencode's subagent-task envelope:
///
///   <task>
///     <description>Fix GUI code</description>
///     <prompt>Edit the file</prompt>
///     <file>/tmp/calc-test60/src/gui/src/lib.rs</file>
///     <content>...rust code...</content>
///   </task>
///
/// Outer `<task>` is not a declared tool name → `extract_xml` skips
/// the whole envelope. The inner `<file>...</file><content>...
/// </content>` pair carries the actual write intent and we recover
/// it here. Multiple pairs in one response yield one synthetic
/// Write call each.
///
/// Strict gates against false positives:
///   - The declared tools MUST include a write-shape tool (path +
///     content properties). Otherwise: skip.
///   - `<file>` and `<content>` must appear in that order, with
///     `<content>` starting WITHIN 256 bytes of `</file>` close
///     (loose proximity gate — empirically the model puts them
///     adjacent).
///   - PATH must be a non-empty token-like string after trim.
///   - BODY must be non-empty after trim.
pub(super) fn extract_file_content_pair(content: &str, matchers: &[ToolShape]) -> Vec<ToolCall> {
    // Locate the write-shape tool. If none declared, no salvage.
    let write_shape = matchers.iter().find(|m| m.path_and_content().is_some());
    let Some(write_shape) = write_shape else {
        return Vec::new();
    };
    let (path_prop, content_prop) = match write_shape.path_and_content() {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut out = Vec::new();
    let mut search = 0usize;
    while let Some(rel) = content[search..].find("<file>") {
        let file_open = search + rel;
        let body_start = file_open + 6; // len("<file>")
        let Some(close_rel) = content[body_start..].find("</file>") else {
            break;
        };
        let path_text = content[body_start..body_start + close_rel].trim();
        let after_file = body_start + close_rel + 7; // len("</file>")
        if path_text.is_empty() || path_text.len() > 200 {
            search = after_file;
            continue;
        }
        // Look for <content> within 256 bytes of </file>.
        let probe_end = (after_file + 256).min(content.len());
        let probe = &content[after_file..probe_end];
        let Some(content_open_rel) = probe.find("<content>") else {
            search = after_file;
            continue;
        };
        let content_body_start = after_file + content_open_rel + 9; // len("<content>")
        let Some(content_close_rel) = content[content_body_start..].find("</content>") else {
            break;
        };
        let body_text = &content[content_body_start..content_body_start + content_close_rel];
        let body_trimmed = body_text.trim();
        if body_trimmed.is_empty() {
            search = content_body_start + content_close_rel + 10;
            continue;
        }
        let mut args = serde_json::Map::new();
        args.insert(
            path_prop.clone(),
            serde_json::Value::String(path_text.to_string()),
        );
        args.insert(
            content_prop.clone(),
            serde_json::Value::String(body_text.trim_end().to_string()),
        );
        if let Some(tc) = synthesise(write_shape.name(), &args, "file_content_pair") {
            out.push(tc);
        }
        search = content_body_start + content_close_rel + 10;
    }
    out
}

const KNOWN_EXTS: &[&str] = &[
    "toml", "rs", "md", "json", "yml", "yaml", "py", "js", "ts", "tsx", "jsx", "html", "css",
    "scss", "sh", "bash", "zsh", "sql", "cfg", "ini", "env", "txt", "go", "java", "kt", "swift",
    "c", "cpp", "cc", "h", "hpp", "rb", "php", "pl", "lua", "vim", "tf", "proto", "graphql", "xml",
    "csv", "tsv",
];

const KNOWN_EXACT: &[&str] = &[
    "dockerfile",
    "makefile",
    ".gitignore",
    ".env",
    "license",
    "readme",
    "cargo.lock",
    ".dockerignore",
    ".editorconfig",
];

pub(super) fn looks_like_path(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.len() > 200 {
        return None;
    }
    if trimmed
        .chars()
        .any(|c| c.is_ascii_whitespace() || c == '"' || c == '\'')
    {
        return None;
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return None;
    }
    let lower = trimmed.to_ascii_lowercase();
    if KNOWN_EXACT.iter().any(|&k| lower == k) {
        return Some(trimmed.to_string());
    }
    let stem = trimmed.trim_end_matches([':', ',', '.', ';']);
    if stem.is_empty() {
        return None;
    }
    let last_dot = stem.rfind('.')?;
    let ext = stem[last_dot + 1..].to_ascii_lowercase();
    if !KNOWN_EXTS.iter().any(|&e| e == ext) {
        return None;
    }
    let body = &stem[..last_dot];
    if body.is_empty()
        || !body
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '/' || c == '.')
    {
        return None;
    }
    Some(stem.to_string())
}

pub(super) fn synthesise(
    name: &str,
    args: &serde_json::Map<String, serde_json::Value>,
    source: &'static str,
) -> Option<ToolCall> {
    let arguments = serde_json::to_string(&serde_json::Value::Object(args.clone())).ok()?;
    let id = format!(
        "call_salvage_{source}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros()
    );
    Some(ToolCall {
        id,
        call_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments,
        },
    })
}
