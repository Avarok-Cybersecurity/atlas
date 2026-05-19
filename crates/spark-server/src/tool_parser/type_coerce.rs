// SPDX-License-Identifier: AGPL-3.0-only

use super::{ToolCall, ToolDefinition};

/// Apply schema-driven type coercion to all calls in `calls`.
///
/// Matches each call to its [`ToolDefinition`] by name, then rewrites
/// JSON-string argument values to the type declared in the schema's
/// `properties` object. Never panics and never drops fields — unrecognised
/// or unparseable values are left as-is.
pub fn coerce_all(calls: &mut [ToolCall], tools: &[ToolDefinition]) {
    for call in calls.iter_mut() {
        let def = tools.iter().find(|t| t.function.name == call.function.name);
        coerce_call_args(call, def);
    }
}

fn coerce_call_args(call: &mut ToolCall, tool_def: Option<&ToolDefinition>) {
    let Some(schema) = tool_def.and_then(|t| t.function.parameters.as_ref()) else {
        return;
    };
    let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
        return;
    };

    let Ok(mut args) = serde_json::from_str::<serde_json::Value>(&call.function.arguments) else {
        return;
    };
    let Some(obj) = args.as_object_mut() else {
        return;
    };

    let mut changed = false;
    for (key, prop) in props {
        let Some(ty) = prop.get("type").and_then(|t| t.as_str()) else {
            continue;
        };
        let Some(val) = obj.get_mut(key) else {
            continue;
        };
        match ty {
            "integer" | "number" => {
                if let serde_json::Value::String(s) = val {
                    if let Ok(n) = s.parse::<f64>() {
                        if let Some(num) = serde_json::Number::from_f64(n) {
                            *val = serde_json::Value::Number(num);
                            changed = true;
                        }
                    }
                }
            }
            "boolean" => {
                if let serde_json::Value::String(s) = val {
                    match s.as_str() {
                        "true" | "True" => {
                            *val = serde_json::Value::Bool(true);
                            changed = true;
                        }
                        "false" | "False" => {
                            *val = serde_json::Value::Bool(false);
                            changed = true;
                        }
                        _ => {}
                    }
                }
            }
            "array" | "object" => {
                if let serde_json::Value::String(s) = val {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                        *val = parsed;
                        changed = true;
                    }
                }
            }
            "null" => {
                if matches!(val, serde_json::Value::String(s) if s == "null") {
                    *val = serde_json::Value::Null;
                    changed = true;
                }
            }
            _ => {}
        }
    }

    if changed {
        if let Ok(s) = serde_json::to_string(&args) {
            call.function.arguments = s;
        }
    }
}
