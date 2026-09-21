// SPDX-License-Identifier: AGPL-3.0-only

//! What a tool call *is*, for display and for logs.
//!
//! Split out of `sanitizer.rs` when it crossed the 500-line cap. The boundary
//! is not arbitrary: everything here answers "what did this tool do", given a
//! finished tool call, while its parent answers "what is safe to emit right
//! now", given a half-arrived stream. They shared a file and nothing else, and
//! the tests that lived in that file only ever covered this half.

use super::*;

/// F21 (2026-04-26): extract the FINAL meaningful command from a
/// Bash chain. Splits on `&&`, `||`, `;`, newlines. Drops `cd …`
/// boilerplate. Truncates the result to F7_BASH_COMMAND_PREFIX_LEN
/// characters (UTF-8-boundary safe). The returned string is the
/// F7 bucket key for Bash tool calls.
///
/// Examples:
///   "mkdir -p /tmp/x && cd /tmp/x && cargo init --name a"
///     → "cargo init --name a"
///   "mkdir -p /tmp/x/src && cd /tmp/x && cargo init --name a"
///     → "cargo init --name a"  (collapses with above)
///   "ls -la /tmp/x"
///     → "ls -la /tmp/x"
pub fn extract_bash_final_action(command: &str) -> String {
    // Split on shell-chain operators. We split on each character
    // class separately to keep it simple; any of the splitters
    // collapses adjacent empty pieces below.
    let parts: Vec<&str> = command
        .split(['&', '|', ';', '\n'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && !s.starts_with("cd ") && !s.starts_with("cd\t") && *s != "cd")
        .collect();
    let action = parts.last().copied().unwrap_or(command);
    let n = action.len().min(F7_BASH_COMMAND_PREFIX_LEN);
    let mut cut = n;
    while cut > 0 && !action.is_char_boundary(cut) {
        cut -= 1;
    }
    action[..cut].to_string()
}

/// Extract a "primary arg" string from a tool call's JSON arguments.
/// The choice of primary arg is per-tool: Write/Edit/Read use
/// `file_path`; Bash uses `command` (truncated so flag-only diffs
/// collapse); other tools fall back to the first non-empty
/// string-valued field in canonical key order.
/// F51 (2026-04-27): tool-name family classifier (SSOT). opencode
/// sends lowercase tool names (`write`, `bash`, `edit`); Claude
/// Code sends uppercase Anthropic-style. All match arms in F-fix
/// helpers must accept both. Centralised here so adding a new tool
/// family touches one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    Bash,
    Write,
    Edit,
    Read,
    MultiEdit,
    Other,
}

pub fn classify_tool(name: &str) -> ToolKind {
    if name.eq_ignore_ascii_case("Bash") {
        ToolKind::Bash
    } else if name.eq_ignore_ascii_case("Write") {
        ToolKind::Write
    } else if name.eq_ignore_ascii_case("Edit") {
        ToolKind::Edit
    } else if name.eq_ignore_ascii_case("Read") {
        ToolKind::Read
    } else if name.eq_ignore_ascii_case("MultiEdit") {
        ToolKind::MultiEdit
    } else {
        ToolKind::Other
    }
}

pub fn primary_arg_for_tool(name: &str, args_json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args_json).ok()?;
    let obj = v.as_object()?;
    let kind = classify_tool(name);
    let key_for_well_known = match kind {
        ToolKind::Write | ToolKind::Edit | ToolKind::Read | ToolKind::MultiEdit => {
            // opencode uses `filePath`; Claude Code uses `file_path`.
            // Try both — the helper at args_json -> obj is structured.
            if obj.get("file_path").and_then(|v| v.as_str()).is_some() {
                Some("file_path")
            } else if obj.get("filePath").and_then(|v| v.as_str()).is_some() {
                Some("filePath")
            } else {
                Some("file_path")
            }
        }
        ToolKind::Bash => Some("command"),
        ToolKind::Other => None,
    };
    if let Some(k) = key_for_well_known {
        let val = obj.get(k).and_then(|v| v.as_str())?;
        if matches!(kind, ToolKind::Bash) {
            // F21 (2026-04-26): bucket on the FINAL command in the
            // shell chain rather than the first 80 chars. Splits on
            // `&&`, `||`, `;`, `\n`. Drops leading `cd …` segments
            // (which the model uses as boilerplate). All five fix30
            // cargo-init variants (different mkdir prefixes) collapse
            // to the same bucket "cargo init --name axum_echo_server".
            return Some(extract_bash_final_action(val));
        }
        return Some(val.to_string());
    }
    // Fallback: first non-empty string-valued field, sorted by key.
    let mut keys: Vec<&String> = obj.keys().collect();
    keys.sort();
    for k in keys {
        if let Some(s) = obj.get(k).and_then(|v| v.as_str())
            && !s.is_empty()
        {
            let n = s.len().min(F7_OTHER_ARG_FALLBACK_LEN);
            let mut cut = n;
            while cut > 0 && !s.is_char_boundary(cut) {
                cut -= 1;
            }
            return Some(format!("{}={}", k, &s[..cut]));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{ToolKind, classify_tool, extract_bash_final_action, primary_arg_for_tool};

    #[test]
    fn bash_final_action_returns_last_segment() {
        let out =
            extract_bash_final_action("mkdir -p /tmp/x/src && cd /tmp/x && cargo init --name a");
        assert!(out.starts_with("cargo init"), "got: {out}");
    }

    #[test]
    fn bash_final_action_no_chain_returns_original() {
        let out = extract_bash_final_action("ls -la /tmp/x");
        assert!(out.starts_with("ls -la"));
    }

    #[test]
    fn bash_final_action_empty_returns_empty() {
        assert_eq!(extract_bash_final_action(""), "");
    }

    #[test]
    fn classify_tool_case_insensitive() {
        assert_eq!(classify_tool("Bash"), ToolKind::Bash);
        assert_eq!(classify_tool("bash"), ToolKind::Bash);
        assert_eq!(classify_tool("BASH"), ToolKind::Bash);
        assert_eq!(classify_tool("Write"), ToolKind::Write);
        assert_eq!(classify_tool("Edit"), ToolKind::Edit);
        assert_eq!(classify_tool("Read"), ToolKind::Read);
        assert_eq!(classify_tool("MultiEdit"), ToolKind::MultiEdit);
        assert_eq!(classify_tool("multiedit"), ToolKind::MultiEdit);
    }

    #[test]
    fn classify_tool_unknown_is_other() {
        assert_eq!(classify_tool("GetWeather"), ToolKind::Other);
        assert_eq!(classify_tool(""), ToolKind::Other);
        assert_eq!(classify_tool("Bashly"), ToolKind::Other);
    }

    #[test]
    fn primary_arg_write_snake_and_camel() {
        let out = primary_arg_for_tool("Write", r#"{"file_path":"/tmp/x.rs"}"#);
        assert_eq!(out.as_deref(), Some("/tmp/x.rs"));
        let out = primary_arg_for_tool("write", r#"{"filePath":"/tmp/y.rs"}"#);
        assert_eq!(out.as_deref(), Some("/tmp/y.rs"));
    }

    #[test]
    fn primary_arg_bash_collapses_chain() {
        let out = primary_arg_for_tool("Bash", r#"{"command":"cd /tmp && cargo build"}"#);
        assert!(out.as_ref().is_some_and(|s| s.starts_with("cargo build")));
    }

    #[test]
    fn primary_arg_unknown_tool_falls_back() {
        // ToolKind::Other has no well-known key but fallback path may
        // return the first non-empty string field.
        let out = primary_arg_for_tool("GetWeather", r#"{"location":"Paris"}"#);
        assert!(
            out.is_some(),
            "fallback path should return some(location=Paris)"
        );
    }

    #[test]
    fn primary_arg_malformed_json_returns_none() {
        assert_eq!(primary_arg_for_tool("Write", "not json"), None);
    }

    #[test]
    fn primary_arg_missing_key_returns_none() {
        let out = primary_arg_for_tool("Write", r#"{"content":"fn main(){}"}"#);
        assert_eq!(out, None);
    }
}
