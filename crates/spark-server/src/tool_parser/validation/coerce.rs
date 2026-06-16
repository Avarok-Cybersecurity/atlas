// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

//! Schema-aware argument coercion + required-param backfill, split out of
//! `validation.rs` to keep that file ≤500 LoC.

use super::super::*;

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
