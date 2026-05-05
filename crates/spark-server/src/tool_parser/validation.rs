// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Fix tool call arguments: schema-aware type coercion + backfill missing params.
///
/// The qwen3_coder XML format emits all parameter values as raw text. This function:
/// 1. **Type coercion**: Converts string values to the schema-expected type
///    (number, boolean, integer, object, array). Prevents "expected number,
///    received string" errors from clients like OpenCode.
/// 2. **Backfill**: Adds empty strings for missing required string parameters.
///    Prevents cascading error loops from missing params.
///
/// Matches vLLM's qwen3coder_tool_parser behavior (schema-aware type conversion).
///
/// Resolves the effective type from a JSON schema property, handling `anyOf`/`oneOf`
/// wrappers (e.g., Pydantic v2's `Optional[int]` → `{"anyOf": [{"type":"integer"},{"type":"null"}]}`).
fn resolve_schema_type(schema: &serde_json::Value) -> Option<&str> {
    // Direct "type" field
    if let Some(t) = schema.get("type").and_then(|t| t.as_str()) {
        return Some(t);
    }
    // anyOf / oneOf: pick first non-null type
    for key in ["anyOf", "oneOf"] {
        if let Some(variants) = schema.get(key).and_then(|v| v.as_array()) {
            for variant in variants {
                if let Some(t) = variant.get("type").and_then(|t| t.as_str())
                    && t != "null"
                {
                    return Some(t);
                }
            }
        }
    }
    None
}

pub fn backfill_required_params(calls: &mut [ToolCall], tools: &[ToolDefinition]) {
    for call in calls.iter_mut() {
        let Some(tool_def) = tools.iter().find(|t| t.function.name == call.function.name) else {
            continue;
        };
        let Some(ref params_schema) = tool_def.function.parameters else {
            continue;
        };
        let required = params_schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();
        let properties = params_schema.get("properties").and_then(|p| p.as_object());
        let Ok(mut args) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
            &call.function.arguments,
        ) else {
            continue;
        };
        let mut changed = false;

        // 1. Coerce existing parameters to schema-expected types.
        if let Some(props) = properties {
            for (key, value) in args.iter_mut() {
                let expected_type = props.get(key).and_then(|p| resolve_schema_type(p));
                if let (Some(expected), serde_json::Value::String(s)) = (expected_type, &value) {
                    let coerced = match expected {
                        "number" => s.parse::<f64>().ok().map(|n| {
                            serde_json::Value::Number(
                                serde_json::Number::from_f64(n)
                                    .unwrap_or(serde_json::Number::from(0)),
                            )
                        }),
                        "integer" => s
                            .parse::<i64>()
                            .ok()
                            .map(|n| serde_json::Value::Number(n.into())),
                        "boolean" => match s.to_lowercase().as_str() {
                            "true" | "1" | "yes" => Some(serde_json::Value::Bool(true)),
                            "false" | "0" | "no" => Some(serde_json::Value::Bool(false)),
                            _ => None,
                        },
                        "object" | "array" => serde_json::from_str(s).ok(),
                        _ => None, // "string" or unknown — keep as-is
                    };
                    if let Some(new_val) = coerced {
                        *value = new_val;
                        changed = true;
                    }
                }
            }
        }

        // 2. Normalize parameter names to match the schema.
        // The model sometimes emits camelCase (filePath) when the schema
        // defines snake_case (file_path), or vice versa. This is a known
        // Qwen3-Coder issue (vLLM #35347, llama.cpp #19382).
        if let Some(props) = properties {
            // Build case-insensitive lookup: "filepath" → "file_path" (schema name)
            let schema_normalized: std::collections::HashMap<String, &str> = props
                .keys()
                .map(|k| (k.to_lowercase().replace('_', ""), k.as_str()))
                .collect();

            let keys_to_fix: Vec<(String, String)> = args
                .keys()
                .filter(|k| !props.contains_key(*k))
                .filter_map(|k| {
                    let norm = k.to_lowercase().replace('_', "");
                    schema_normalized
                        .get(&norm)
                        .map(|schema_key| (k.clone(), schema_key.to_string()))
                })
                .collect();

            for (wrong_key, right_key) in keys_to_fix {
                if let Some(val) = args.remove(&wrong_key) {
                    args.entry(right_key).or_insert(val);
                    changed = true;
                }
            }
        }

        // 3. Backfill missing required string parameters.
        for key in &required {
            if !args.contains_key(*key) {
                let is_string = properties
                    .and_then(|p| p.get(*key))
                    .and_then(|v| v.get("type"))
                    .and_then(|t| t.as_str())
                    .is_none_or(|t| t == "string"); // default to string if no schema
                if is_string {
                    args.insert(key.to_string(), serde_json::Value::String(String::new()));
                    changed = true;
                }
            }
        }

        // 4. Auto-fill empty parameters with sensible defaults.
        // The model often generates empty required fields. Instead of rejecting
        // (which causes error loops), fill them with context-derived defaults.
        let func_name = call.function.name.clone();
        for key in &required {
            if let Some(serde_json::Value::String(val)) = args.get(*key) {
                if !val.trim().is_empty() {
                    continue;
                }
                let auto_val = match *key {
                    "description" => {
                        if let Some(serde_json::Value::String(cmd)) = args.get("command") {
                            if cmd.len() > 50 {
                                format!("Run: {}...", &cmd[..47])
                            } else {
                                format!("Run: {cmd}")
                            }
                        } else {
                            format!("{func_name} operation")
                        }
                    }
                    "filePath" | "file_path" => {
                        // Can't guess the path — leave empty so validation catches it
                        continue;
                    }
                    "oldString" | "old_string" => {
                        // Can't guess what to replace — leave empty
                        continue;
                    }
                    _ => continue,
                };
                args.insert(key.to_string(), serde_json::Value::String(auto_val));
                changed = true;
            }
        }

        if changed && let Ok(new_args) = serde_json::to_string(&serde_json::Value::Object(args)) {
            call.function.arguments = new_args;
        }
    }
}

/// Check if a tool call has empty required parameters that can't be auto-filled.
/// Returns the names of empty required params, or empty vec if all are filled.
pub fn find_empty_required_params(call: &ToolCall, tools: &[ToolDefinition]) -> Vec<String> {
    let Some(tool_def) = tools.iter().find(|t| t.function.name == call.function.name) else {
        return Vec::new();
    };
    let Some(ref params_schema) = tool_def.function.parameters else {
        return Vec::new();
    };
    let required = params_schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let args: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&call.function.arguments).unwrap_or_default();
    let mut empty = Vec::new();
    for key in &required {
        match args.get(key.as_str()) {
            None => empty.push(key.clone()),
            Some(serde_json::Value::String(s)) if s.trim().is_empty() => empty.push(key.clone()),
            Some(serde_json::Value::Null) => empty.push(key.clone()),
            _ => {}
        }
    }
    empty
}

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

/// Fuzzy match a hallucinated tool name to the closest available tool.
///
/// Strategies (in priority order):
/// 1. Substring containment: "weather" matches "get_weather"
/// 2. Available tool name contains the model's name: "get_weather" contains "weather"
/// 3. Model's name contains available tool: "weather_report" contains "weather"
///
/// Only returns a match if exactly one tool matches (ambiguous = reject).
fn fuzzy_match_tool_name(model_name: &str, tools: &[ToolDefinition]) -> Option<String> {
    if tools.is_empty() || model_name.is_empty() {
        return None;
    }

    let lower = model_name.to_lowercase();

    // Strategy 1: exact substring — model name is substring of a tool name
    let matches: Vec<&str> = tools
        .iter()
        .filter(|t| t.function.name.to_lowercase().contains(&lower))
        .map(|t| t.function.name.as_str())
        .collect();
    if matches.len() == 1 {
        return Some(matches[0].to_string());
    }

    // Strategy 2: tool name is substring of model's name
    let matches: Vec<&str> = tools
        .iter()
        .filter(|t| lower.contains(&t.function.name.to_lowercase()))
        .map(|t| t.function.name.as_str())
        .collect();
    if matches.len() == 1 {
        return Some(matches[0].to_string());
    }

    // Strategy 3: if only one tool available, use it (model clearly intended to call a tool)
    if tools.len() == 1 {
        return Some(tools[0].function.name.clone());
    }

    None
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
            if let Some(serde_json::Value::String(path)) = args.get(*key)
                && path.trim().is_empty()
            {
                return Err(format!(
                    "Error: {name} requires a non-empty '{key}'. \
                         Got empty string — provide an absolute path \
                         like '/tmp/calc-test75/Cargo.toml'."
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
