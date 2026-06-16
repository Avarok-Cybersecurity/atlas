// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

//! File-path normalization for tool-call arguments, split out of
//! `validation.rs` to keep that file ≤500 LoC.

use super::super::*;

/// Normalize file paths in tool call arguments to be relative to the working directory.
///
/// OPENCODE BUG FIX (2026-04-22): the previous behaviour stripped the leading `/`
/// of any absolute path NOT under cwd, mangling user-intended paths like
/// `/tmp/calc-test16/calc.py` into `tmp/calc-test16/calc.py`. opencode then
/// resolved that relative path under `Instance.directory` (= `$HOME`), so the
/// file ended up at `$HOME/tmp/calc-test16/calc.py` instead of
/// `/tmp/calc-test16/`. The model spent 8+ turns trying to "fix" the directory
/// before the user noticed.
///
/// New behaviour:
/// - Paths under cwd → made relative (still helpful for Claude-Code-style clients)
/// - Paths starting with `/` but NOT under cwd → **PASS THROUGH UNCHANGED**.
///   The model knew what it wanted (e.g. user said "put it in /tmp/..."); we
///   should not second-guess. If it really is wrong, the filesystem op will
///   fail with a clear error and the model can self-correct.
/// - Already relative paths → unchanged
pub fn normalize_paths(calls: &mut [ToolCall], cwd: &str) {
    // Common parameter names that contain file paths
    const PATH_KEYS: &[&str] = &["file_path", "filePath", "path", "file"];
    let cwd_slash = if cwd.ends_with('/') {
        cwd.to_string()
    } else {
        format!("{cwd}/")
    };

    for call in calls.iter_mut() {
        let Ok(mut args) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
            &call.function.arguments,
        ) else {
            continue;
        };
        let mut changed = false;
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key) {
                // Long-context FP8 drift mode (2026-05-28): the model
                // sometimes emits the value with XML-attribute-style
                // framing — `="/tmp/x/main.rs"` instead of `/tmp/x/main.rs`.
                // The qwen3_coder grammar accepts the literal `=` and quotes
                // as part of the parameter body. Strip them here so the
                // downstream path-shape check and write dispatch see a
                // clean path. vLLM's tool_parser does similar leniency.
                let trimmed = path.trim();
                let mut sanitized: &str = trimmed;
                if let Some(rest) = sanitized.strip_prefix('=') {
                    sanitized = rest.trim_start();
                }
                // FP8 drift (2026-05-29, fencecontent run 1): the model
                // sometimes leaks a JSON-fragment-shaped value like
                // `"/tmp/x/Cargo.toml",` — the path wrapped in quotes with a
                // trailing comma. Drop trailing commas/whitespace first so the
                // surrounding-quote strip below sees a clean `"…"`; otherwise
                // the file is created with the quotes+comma literally in its
                // name and the project never builds.
                sanitized = sanitized.trim_end_matches([',', ' ', '\t']);
                if sanitized.len() >= 2 && sanitized.starts_with('"') && sanitized.ends_with('"') {
                    sanitized = &sanitized[1..sanitized.len() - 1];
                }
                if sanitized != path.as_str() {
                    args.insert(
                        key.to_string(),
                        serde_json::Value::String(sanitized.to_string()),
                    );
                    changed = true;
                }
                // Re-read after possible sanitization
                let Some(serde_json::Value::String(path)) = args.get(*key) else {
                    continue;
                };
                if !path.starts_with('/') {
                    continue; // Already relative — leave it
                }
                if !path.starts_with(&cwd_slash) {
                    // Absolute path NOT under cwd — pass through verbatim. The
                    // user explicitly asked for this location (e.g. "/tmp/..."),
                    // and trimming `/` here breaks downstream clients that
                    // resolve relative paths against THEIR own working dir.
                    continue;
                }
                let new_path = path[cwd_slash.len()..].to_string();
                if new_path != *path && !new_path.is_empty() {
                    args.insert(key.to_string(), serde_json::Value::String(new_path));
                    changed = true;
                }
            }
        }
        if changed && let Ok(new_args) = serde_json::to_string(&serde_json::Value::Object(args)) {
            call.function.arguments = new_args;
        }
    }
}

#[cfg(test)]
mod path_sanitizer_tests {
    #[test]
    fn malformed_quoted_comma_filepath_sanitized() {
        // FP8 drift: filePath value leaked as a JSON fragment `"…/Cargo.toml",`
        // (surrounding quotes + trailing comma). The unconditional path
        // sanitizer must clean it to a cwd-relative `Cargo.toml` so the file
        // lands with a usable name.
        use crate::tool_parser::{FunctionCall, ToolCall};
        let mut calls = vec![ToolCall {
            id: "x".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "write".into(),
                arguments: serde_json::json!({
                    "filePath": "\"/tmp/proj/Cargo.toml\",",
                    "content": "[package]\nname = \"x\"\n"
                })
                .to_string(),
            },
        }];
        super::normalize_paths(&mut calls, "/tmp/proj");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["filePath"], "Cargo.toml");
    }
}
