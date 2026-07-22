// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::fuzzy_match::fuzzy_match_tool_name;
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

/// Map a model-emitted parameter `key` to its canonical schema property name
/// for the tool `call_name`. SSOT for the camelCase↔snake_case repair used by
/// both `backfill_required_params` (post-parse) and live argument streaming
/// (`streaming_impl::coerce_kv`). Returns `key` unchanged when the tool is
/// unknown, has no `properties`, `key` is already a schema property, or no
/// case-insensitive/underscore-insensitive match exists.
pub(crate) fn normalize_param_name(tools: &[ToolDefinition], call_name: &str, key: &str) -> String {
    let Some(tool_def) = tools.iter().find(|t| t.function.name == call_name) else {
        return key.to_string();
    };
    let Some(props) = tool_def
        .function
        .parameters
        .as_ref()
        .and_then(|p| p.get("properties"))
        .and_then(|p| p.as_object())
    else {
        return key.to_string();
    };
    if props.contains_key(key) {
        return key.to_string();
    }
    // Build case-insensitive lookup: "filepath" → "file_path" (schema name).
    let schema_normalized: std::collections::HashMap<String, &str> = props
        .keys()
        .map(|k| (k.to_lowercase().replace('_', ""), k.as_str()))
        .collect();
    let norm = key.to_lowercase().replace('_', "");
    schema_normalized
        .get(&norm)
        .map(|schema_key| schema_key.to_string())
        .unwrap_or_else(|| key.to_string())
}

/// Echo-attractor salvage (2026-07-09): at long context the model can emit
/// the XML wire format shifted by one structural token —
/// `<parameter=parameter>filePath>\n/tmp/x</parameter>` or
/// `<parameter=write>content>…</parameter>` — so the REAL key lands as the
/// leading `IDENT>` of the value while the key slot holds an echoed
/// structural word (`parameter`, the function name, …). The grammar's
/// key-slot masking makes this shape sticky once entered (`=` is masked, so
/// the echo closes early), and the real path/content are fully present in
/// the value. Recover them: when `key` is NOT a schema property and the
/// value begins with `PROP>` for a schema property `PROP`, re-split into
/// `(PROP, rest)`.
///
/// SSOT — used by both the buffered pipeline (`backfill_required_params`
/// step 2.5) and live argument streaming (`streaming_emit::coerce_kv`), the
/// same dual-wiring as [`normalize_param_name`]. Deliberately conservative:
/// never fires when the emitted key is itself a schema property (a legit
/// value could start with `IDENT>`), and requires the exact `PROP>` prefix.
pub(crate) fn salvage_echoed_param(
    tools: &[ToolDefinition],
    call_name: &str,
    key: &str,
    value: &str,
) -> Option<(String, String)> {
    let props = tools
        .iter()
        .find(|t| t.function.name == call_name)?
        .function
        .parameters
        .as_ref()?
        .get("properties")?
        .as_object()?;
    if props.is_empty() || props.contains_key(key) {
        return None;
    }
    // Longest property first so e.g. `filePath>` wins over a hypothetical
    // `file>` prefix.
    let mut names: Vec<&String> = props.keys().collect();
    names.sort_by_key(|n| std::cmp::Reverse(n.len()));
    for prop in names {
        if let Some(rest) = value
            .strip_prefix(prop.as_str())
            .and_then(|r| r.strip_prefix('>'))
        {
            // Both parse paths `.trim()` parameter values; stay consistent.
            return Some((prop.clone(), rest.trim().to_string()));
        }
    }
    None
}

/// Extract agent-type names from a delegation tool's prose description.
///
/// Matches lines shaped like `- <name>: …` — the convention both opencode
/// and Claude Code use under an "Available agent types" heading. `<name>`
/// is the token before the first `:` and must be a single bare identifier
/// (alphanumeric + `-`/`_`), which excludes prose bullets such as
/// `- If you want to read a file, use Read instead`.
fn parse_agent_types(description: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in description.lines() {
        let Some(rest) = line.trim_start().strip_prefix("- ") else {
            continue;
        };
        let Some((name, _)) = rest.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if !name.is_empty()
            && name.len() <= 64
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            out.push(name.to_string());
        }
    }
    out
}

/// Choose a valid `subagent_type` for a delegation tool (opencode / Claude
/// Code `task`).
///
/// `subagent_type` is a REQUIRED free-form string whose legal values are
/// listed only as prose in the tool description — there is no JSON-Schema
/// `enum`, so the model frequently omits it. The missing-required backfill
/// below would otherwise insert `""`, which the client rejects with an
/// opaque `Unknown agent type:  is not a valid agent type` that the model
/// cannot self-correct. Filling a VALID agent name instead lets delegation
/// actually succeed.
///
/// Prefers a general-purpose agent (`general`, `general-purpose`), else the
/// first agent listed, with a final fallback of `"general"` (opencode's
/// built-in general agent).
fn infer_default_subagent_type(description: Option<&str>) -> String {
    let candidates = description.map(parse_agent_types).unwrap_or_default();
    candidates
        .iter()
        .find(|c| c.to_ascii_lowercase().contains("general"))
        .or_else(|| candidates.first())
        .cloned()
        .unwrap_or_else(|| "general".to_string())
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
        // Qwen3-Coder issue (vLLM #35347, llama.cpp #19382). Delegated to the
        // shared `normalize_param_name` SSOT helper so live argument streaming
        // (`streaming_impl::coerce_kv`) applies the identical mapping.
        if properties.is_some() {
            let keys_to_fix: Vec<(String, String)> = args
                .keys()
                .map(|k| {
                    (
                        k.clone(),
                        normalize_param_name(tools, &call.function.name, k),
                    )
                })
                .filter(|(orig, mapped)| orig != mapped)
                .collect();

            for (wrong_key, right_key) in keys_to_fix {
                if let Some(val) = args.remove(&wrong_key) {
                    args.entry(right_key).or_insert(val);
                    changed = true;
                }
            }
        }

        // 2.5. Echo-attractor salvage: re-split `{"parameter": "filePath>…"}`
        // shapes where the real key leaked into the value (see
        // `salvage_echoed_param`). Runs BEFORE the missing-required backfill
        // so the recovered key satisfies `required` instead of being
        // synthesized as "" (which clients reject with errors the degraded
        // model then retries verbatim — the 42.5k opencode doom loop).
        if properties.is_some() {
            let salvageable: Vec<(String, String, String)> = args
                .iter()
                .filter_map(|(k, v)| {
                    let s = v.as_str()?;
                    salvage_echoed_param(tools, &call.function.name, k, s)
                        .map(|(real_key, real_val)| (k.clone(), real_key, real_val))
                })
                .collect();
            for (echoed_key, real_key, real_val) in salvageable {
                let target_empty = matches!(
                    args.get(&real_key),
                    None | Some(serde_json::Value::String(_))
                ) && args
                    .get(&real_key)
                    .and_then(|v| v.as_str())
                    .is_none_or(|s| s.trim().is_empty());
                if target_empty {
                    args.remove(&echoed_key);
                    args.insert(real_key, serde_json::Value::String(real_val));
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
                                // Truncate on a char boundary — `cmd` is raw
                                // model output and a byte slice at [..47] would
                                // panic if a multibyte char straddles byte 47.
                                let head: String = cmd.chars().take(47).collect();
                                format!("Run: {head}...")
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
                    // Delegation tools (opencode / Claude Code `task`) require a
                    // free-form `subagent_type` whose legal values live only in
                    // the description prose (no JSON-Schema enum). An empty value
                    // is rejected downstream with an opaque "Unknown agent type:
                    // …" the model can't self-correct, so fill a VALID agent name
                    // parsed from the description instead of "".
                    "subagent_type" | "subagentType" => {
                        infer_default_subagent_type(tool_def.function.description.as_deref())
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
        match assess_tool_call(call, tools) {
            Ok(()) => valid.push(call.clone()),
            // Recoverable (missing required param): attach the call as the
            // model produced it — the client's schema check gives the model
            // actionable feedback. Dropping it here returned EMPTY responses
            // (2026-07-03 ST-995 collapse, 160/995 empties).
            Err(ToolCallIssue::MissingParam(msg)) => {
                valid.push(call.clone());
                errors.push(msg);
            }
            // EmptyRequired keeps the F78 blocking disposition (call becomes
            // a no-op so the response falls through to text); Hard is never
            // attached (phantom name, unparseable args, command-as-path).
            Err(issue) => errors.push(issue.into_message()),
        }
    }

    ValidatedToolCalls { valid, errors }
}

/// Typed validation outcome. Severity decides delivery, not logging:
/// the 2026-07-03 ST-995 collapse (55.78 vs 89.04) traced to blocking-mode
/// dropping EVERY call with a validation error — a call missing one trailing
/// required param became an EMPTY response (160/995 samples). Recoverable
/// issues must ship the call as the model produced it; the client's own
/// schema check gives the model actionable feedback.
#[derive(Debug)]
pub enum ToolCallIssue {
    /// Required parameter absent. Attach the call anyway (both paths).
    MissingParam(String),
    /// Required string present but empty (F78 path tools, shell tools).
    /// Streaming passes these through (2026-05-25 disposition); blocking
    /// keeps the F78 no-op-to-text behavior.
    EmptyRequired(String),
    /// Unrecoverable: phantom tool name, unparseable args JSON, path-shaped
    /// command injection. Never attached.
    Hard(String),
}

impl ToolCallIssue {
    pub fn message(&self) -> &str {
        match self {
            Self::MissingParam(m) | Self::EmptyRequired(m) | Self::Hard(m) => m,
        }
    }
    pub fn into_message(self) -> String {
        match self {
            Self::MissingParam(m) | Self::EmptyRequired(m) | Self::Hard(m) => m,
        }
    }
}

/// Compatibility wrapper over [`assess_tool_call`] for callers/tests that
/// only care about pass/fail.
pub fn validate_single_tool_call(call: &ToolCall, tools: &[ToolDefinition]) -> Result<(), String> {
    assess_tool_call(call, tools).map_err(ToolCallIssue::into_message)
}

/// Validate a single tool call. Returns `Ok(())` if valid, or a
/// [`ToolCallIssue`] whose severity the caller maps to delivery behavior.
pub fn assess_tool_call(call: &ToolCall, tools: &[ToolDefinition]) -> Result<(), ToolCallIssue> {
    let name = &call.function.name;

    // 1. Check tool name exists
    let tool_def = tools.iter().find(|t| t.function.name == *name);
    if tool_def.is_none() {
        let available: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        return Err(ToolCallIssue::Hard(format!(
            "Error: Unknown tool '{}'. Available tools: {}",
            name,
            available.join(", ")
        )));
    }
    let tool_def = tool_def.unwrap();

    // 2. Check arguments are valid JSON
    let args: serde_json::Map<String, serde_json::Value> =
        match serde_json::from_str(&call.function.arguments) {
            Ok(a) => a,
            Err(_) => {
                // Truncate on a char boundary — `arguments` is raw model
                // output (the hermes path stores it as a verbatim string), so
                // a byte slice at [..100] would panic if a multibyte char
                // straddles byte 100.
                let preview: String = call.function.arguments.chars().take(100).collect();
                return Err(ToolCallIssue::Hard(format!(
                    "Error: {name} arguments must be valid JSON. Got: {preview}"
                )));
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
                return Err(ToolCallIssue::MissingParam(format!(
                    "Error: {} requires parameter '{}' but it was not provided.",
                    name, key
                )));
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
                    return Err(ToolCallIssue::EmptyRequired(format!(
                        "Error: {name} requires a non-empty '{key}'. \
                             Got empty string — provide an absolute path \
                             like '/tmp/calc-test75/Cargo.toml'."
                    )));
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
                    return Err(ToolCallIssue::Hard(format!(
                        "Error: {name} '{key}' must be a filesystem path (absolute or relative \
                         to the working directory), at least 3 chars, with no shell \
                         metacharacters or whitespace. Got {path:?}."
                    )));
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
                return Err(ToolCallIssue::EmptyRequired(format!(
                    "Error: {name} requires a non-empty '{key}'. \
                         Got empty string — provide the shell command \
                         to execute, e.g. 'ls /tmp'."
                )));
            }
        }
    }
    if FILE_TOOLS.contains(&name.as_str()) {
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key) {
                if path.ends_with('/') {
                    return Err(ToolCallIssue::Hard(format!(
                        "Error: {} file_path must be a FILE, not a directory. Got '{}'. Use e.g. '{}/index.ts'",
                        name,
                        path,
                        path.trim_end_matches('/')
                    )));
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
                    return Err(ToolCallIssue::Hard(format!(
                        "Error: {} file_path '{}' looks like a directory. Add a filename, e.g. '{}/index.ts'",
                        name, path, path
                    )));
                }
            }
        }
    }

    Ok(())
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
