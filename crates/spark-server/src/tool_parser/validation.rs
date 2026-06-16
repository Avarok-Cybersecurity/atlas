// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::fuzzy_match::fuzzy_match_tool_name;
use super::*;

// Schema-aware coercion / backfill (`backfill_required_params`,
// `find_empty_required_params`) and path normalization (`normalize_paths`)
// were moved to the `coerce` / `paths` submodules to keep this file ≤500
// LoC. Re-exported so existing `tool_parser::*` call sites are unaffected.
mod coerce;
mod paths;

pub use coerce::{backfill_required_params, find_empty_required_params};
pub use paths::normalize_paths;

// ── Tool call validation ──

/// Result of validating a batch of tool calls against their schemas.
pub struct ValidatedToolCalls {
    /// Tool calls that passed all validations.
    pub valid: Vec<ToolCall>,
    /// Human-readable error messages for invalid calls.
    /// These should be injected into the response content so the model
    /// sees clear, actionable feedback instead of cryptic client errors.
    pub errors: Vec<String>,
}

/// Validate tool calls against their schemas. Returns valid calls and
/// error messages for invalid ones.
///
/// Checks:
/// 1. Tool name exists in definitions
/// 2. Arguments are valid JSON
/// 3. Required string params are non-empty
/// 4. file_path params don't look like directories (end with `/`)
pub fn validate_tool_calls(
    mut calls: Vec<ToolCall>,
    tools: &[ToolDefinition],
) -> ValidatedToolCalls {
    let mut valid = Vec::new();
    let mut errors = Vec::new();

    for call in &mut calls {
        // Fuzzy name repair: if model hallucinates a close-but-wrong name,
        // map to the closest available tool (NVFP4 models often drop prefixes
        // like "get_" or use abbreviations like "weather" for "get_weather").
        if tools.iter().all(|t| t.function.name != call.function.name)
            && let Some(best) = fuzzy_match_tool_name(&call.function.name, tools)
        {
            tracing::info!(
                "Fuzzy tool name repair: '{}' -> '{}'",
                call.function.name,
                best
            );
            call.function.name = best;
        }
        match validate_single_tool_call(call, tools) {
            Ok(()) => valid.push(call.clone()),
            Err(msg) => errors.push(msg),
        }
    }

    ValidatedToolCalls { valid, errors }
}

/// Validate a single tool call. Returns `Ok(())` if valid,
/// `Err(error_message)` with a clear, actionable error if invalid.
pub fn validate_single_tool_call(call: &ToolCall, tools: &[ToolDefinition]) -> Result<(), String> {
    let name = &call.function.name;

    // 1. Check tool name exists
    let tool_def = tools.iter().find(|t| t.function.name == *name);
    if tool_def.is_none() {
        let available: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        return Err(format!(
            "Error: Unknown tool '{}'. Available tools: {}",
            name,
            available.join(", ")
        ));
    }
    let tool_def = tool_def.unwrap();

    // 2. Check arguments are valid JSON
    let args: serde_json::Map<String, serde_json::Value> =
        match serde_json::from_str(&call.function.arguments) {
            Ok(a) => a,
            Err(_) => {
                return Err(format!(
                    "Error: {} arguments must be valid JSON. Got: {}",
                    name,
                    &call.function.arguments[..call.function.arguments.len().min(100)]
                ));
            }
        };

    // 3. Check required params are present. Do NOT enforce non-empty strings —
    // that is the client's schema concern. Empty-string rejection here broke
    // Theia IDE's getWorkspaceFileList, which legitimately passes path="".
    if let Some(ref params_schema) = tool_def.function.parameters {
        let required: Vec<&str> = params_schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        for key in &required {
            if args.get(*key).is_none() {
                return Err(format!(
                    "Error: {} requires parameter '{}' but it was not provided.",
                    name, key
                ));
            }
        }
    }

    // 4. Path-specific validation for file tools
    const FILE_TOOLS: &[&str] = &["Write", "write", "Edit", "edit", "Read", "read"];
    const PATH_KEYS: &[&str] = &["file_path", "filePath", "path"];
    // F78 (2026-04-30): file MUTATION tools must have a non-empty
    // path. Live opencode session
    // `ses_2215a95d6ffe6gAzHMBrcXqGXX` looped 11 turns because the
    // model emitted `{"content":"...","filePath":""}` (the model
    // self-truncated the content string and grammar-completed
    // filePath with the empty default). opencode's Write tool
    // returned EISDIR; the model retried with the same empty path.
    // Rejecting here turns the malformed tool_call into a no-op so
    // the response falls through to text — the model gets a single
    // chance to recover instead of opencode echoing EISDIR forever.
    // Read/Glob/LS keep the lenient behavior (Theia's
    // getWorkspaceFileList legitimately passes path="").
    const WRITE_FAMILY: &[&str] = &[
        "Write",
        "write",
        "Edit",
        "edit",
        "MultiEdit",
        "multiEdit",
        "multi_edit",
    ];
    if WRITE_FAMILY.contains(&name.as_str()) {
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key) {
                let trimmed = path.trim();
                if trimmed.is_empty() {
                    // #211 option-B diagnostic (env-gated): pinpoint the
                    // empty_path drift — generation vs parse. Logs the full
                    // post-parse arg shape (keys + per-value lengths). An
                    // empty filePath alongside a large `content` is the
                    // self-truncation generation pattern (F78); filePath
                    // absent ⇒ omission; a path under an unexpected key ⇒
                    // parser. Inert unless ATLAS_TOOLCALL_DEBUG=1.
                    if std::env::var("ATLAS_TOOLCALL_DEBUG").as_deref() == Ok("1") {
                        let shape: Vec<String> = args
                            .iter()
                            .map(|(k, v)| match v {
                                serde_json::Value::String(s) => {
                                    format!("{k}=str(len={})", s.len())
                                }
                                other => format!("{k}={}", other),
                            })
                            .collect();
                        tracing::warn!(
                            tool = %name, empty_key = %key,
                            "ATLAS_TOOLCALL_DEBUG empty-path arg shape: [{}]",
                            shape.join(", ")
                        );
                    }
                    return Err(format!(
                        "Error: {name} requires a non-empty '{key}'. \
                             Got empty string — provide an absolute path \
                             like '/tmp/calc-test75/Cargo.toml'."
                    ));
                }
                // Long-context FP8 drift mode: model occasionally emits
                // the value with XML-attribute-style framing — e.g.
                // `<parameter=filePath>="/tmp/x/main.rs"</parameter>`
                // — leaking the `="..."` shape into the value. Strip a
                // leading `=` and a single pair of surrounding ASCII
                // double-quotes before the path-shape check so these
                // drifted-but-recoverable calls still resolve. vLLM's
                // tool_parser does similar leniency.
                // opencode resolves write paths against the agent cwd
                // (`--dir`), so bare RELATIVE filenames like `Cargo.toml`
                // or `src/main.rs` are legitimate — vLLM accepts them and
                // the model emits them constantly. The previous rule
                // required a `/`, `./`, or `../` prefix and rejected
                // `Cargo.toml`, which made opencode loop on rejections and
                // abandon the task. Accept any non-empty path EXCEPT ones
                // carrying shell metacharacters / whitespace, which signal
                // a leaked command (e.g. `created && ls -R`) rather than a
                // real path — those we still reject (also closes CWE-78
                // command-leak-as-path).
                const SHELL_META: &[char] = &[
                    ' ', '\t', '\n', '\r', '&', '|', ';', '`', '$', '<', '>', '(', ')', '*', '?',
                ];
                let looks_like_command = trimmed.contains(SHELL_META);
                if looks_like_command || trimmed.len() < 3 {
                    return Err(format!(
                        "Error: {name} '{key}' must be a filesystem path (absolute or relative \
                         to the working directory), at least 3 chars, with no shell \
                         metacharacters or whitespace. Got {path:?}."
                    ));
                }
            }
        }
    }
    // Shell-execution tools must have a non-empty command. Mirrors F78
    // for the Write family. Without this, the `any_text` qwen3_coder
    // body grammar (2026-05-25) accepts an immediately-closed parameter
    // `<parameter=command></parameter>`; opencode's bash handler then
    // returns "The argument 'file' cannot be empty. Received ''" and
    // the model burns to max_tokens retrying the same empty call.
    // Previously the `json_schema` body grammar combined with
    // `enforce_min_length_on_required_strings` (`grammar/schema.rs`)
    // enforced min_length 1 at the FSM level; lifting that check to
    // the validator post-parse keeps the same invariant while letting
    // the grammar body be `any_text` (native XML wire format).
    const SHELL_FAMILY: &[&str] = &[
        "bash", "Bash", "shell", "Shell", "exec", "Exec", "run", "Run", "execute", "Execute",
        "terminal", "Terminal",
    ];
    const CMD_KEYS: &[&str] = &["command", "cmd", "script", "code"];
    if SHELL_FAMILY.contains(&name.as_str()) {
        for key in CMD_KEYS {
            if let Some(serde_json::Value::String(cmd)) = args.get(*key)
                && (cmd.trim().is_empty() || cmd.trim().len() < 2)
            {
                return Err(format!(
                    "Error: {name} requires a non-empty '{key}'. \
                         Got empty string — provide the shell command \
                         to execute, e.g. 'ls /tmp'."
                ));
            }
        }
    }
    if FILE_TOOLS.contains(&name.as_str()) {
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key) {
                if path.ends_with('/') {
                    return Err(format!(
                        "Error: {} file_path must be a FILE, not a directory. Got '{}'. Use e.g. '{}/index.ts'",
                        name,
                        path,
                        path.trim_end_matches('/')
                    ));
                }
                // Check if it looks like just a directory name (no extension, no dots, no uppercase)
                // Allow extensionless files like LICENSE, Makefile, Dockerfile, Cargo.lock etc.
                if !path.is_empty()
                    && !path.contains('.')
                    && !path.contains('/')
                    && path
                        .chars()
                        .all(|c| c.is_lowercase() || c == '-' || c == '_')
                {
                    return Err(format!(
                        "Error: {} file_path '{}' looks like a directory. Add a filename, e.g. '{}/index.ts'",
                        name, path, path
                    ));
                }
            }
        }
    }

    Ok(())
}
