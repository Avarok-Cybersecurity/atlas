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

    // Recursive empty-key repair (CC plan-mode `ExitPlanMode` loop fix,
    // 2026-06-07). Under long-context degeneration the model emits an object
    // with an empty-string key for a required property — e.g. an
    // `allowedPrompts` item `{"": "Bash", "prompt": "..."}` whose schema
    // requires `{tool, prompt}`. Claude Code's validator rejects it → retry →
    // self-reinforcing loop. When exactly ONE required property is missing and
    // the orphaned `""` value matches that property's type/enum, the key is
    // unambiguous → rename it. Walks nested array `items` + object properties.
    let mut changed = repair_empty_keys(&mut args, schema);

    let Some(obj) = args.as_object_mut() else {
        if changed && let Ok(s) = serde_json::to_string(&args) {
            call.function.arguments = s;
        }
        return;
    };

    for (key, prop) in props {
        let Some(ty) = prop.get("type").and_then(|t| t.as_str()) else {
            continue;
        };
        let Some(val) = obj.get_mut(key) else {
            continue;
        };
        match ty {
            "integer" => {
                // Coerce to a true integer (not f64) so strict schema
                // validators accept it — `"10"` → `10`, never `10.0`.
                if let serde_json::Value::String(s) = val {
                    if let Ok(n) = s.parse::<i64>() {
                        *val = serde_json::Value::Number(n.into());
                        changed = true;
                    } else if let Ok(f) = s.parse::<f64>()
                        && let Some(num) = serde_json::Number::from_f64(f)
                    {
                        *val = serde_json::Value::Number(num);
                        changed = true;
                    }
                }
            }
            "number" => {
                if let serde_json::Value::String(s) = val
                    && let Ok(n) = s.parse::<f64>()
                    && let Some(num) = serde_json::Number::from_f64(n)
                {
                    *val = serde_json::Value::Number(num);
                    changed = true;
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
                if let serde_json::Value::String(s) = val
                    && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s)
                {
                    *val = parsed;
                    changed = true;
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

    if changed && let Ok(s) = serde_json::to_string(&args) {
        call.function.arguments = s;
    }
}

/// Recursively repair empty-string object keys (`""`) to the unique missing
/// `required` schema property, descending into array `items` and nested
/// object `properties`. Returns `true` if any key was renamed.
///
/// Conservative by construction (PCND): a key is only renamed when there is
/// EXACTLY ONE required property absent from the object AND the orphaned value
/// satisfies that property's declared `type`/`enum`. Otherwise the object is
/// left untouched (the downstream validator still reports the malformed call,
/// exactly as before). This never invents data — it only re-labels a value the
/// model already produced under the one schema slot it can belong to.
fn repair_empty_keys(val: &mut serde_json::Value, schema: &serde_json::Value) -> bool {
    let mut changed = false;
    match val {
        serde_json::Value::Object(map) => {
            // 1. Repair an empty key at THIS object level.
            if map.contains_key("") {
                let props = schema.get("properties").and_then(|p| p.as_object());
                let missing: Vec<String> = schema
                    .get("required")
                    .and_then(|r| r.as_array())
                    .map(|req| {
                        req.iter()
                            .filter_map(|r| r.as_str())
                            .filter(|r| !map.contains_key(*r))
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                if missing.len() == 1 {
                    let cand = &missing[0];
                    let prop_schema = props.and_then(|p| p.get(cand));
                    // Unwrap is safe: contains_key("") was just checked.
                    let orphan = map.get("").cloned().unwrap_or(serde_json::Value::Null);
                    if value_matches_schema(&orphan, prop_schema) {
                        map.remove("");
                        map.insert(cand.clone(), orphan);
                        changed = true;
                    }
                }
            }
            // 2. Recurse into properties that have a child schema.
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                for (k, v) in map.iter_mut() {
                    if let Some(child) = props.get(k) {
                        changed |= repair_empty_keys(v, child);
                    }
                }
            }
        }
        serde_json::Value::Array(arr) => {
            if let Some(items) = schema.get("items") {
                for elem in arr.iter_mut() {
                    changed |= repair_empty_keys(elem, items);
                }
            }
        }
        _ => {}
    }
    changed
}

/// True if `val` is compatible with `prop_schema`'s `enum` (membership) and
/// `type` (JSON kind). Absent schema ⇒ permissive. Numeric/boolean strings are
/// accepted because the type-coercion pass that follows will normalise them.
fn value_matches_schema(val: &serde_json::Value, prop_schema: Option<&serde_json::Value>) -> bool {
    let Some(ps) = prop_schema else {
        return true;
    };
    if let Some(en) = ps.get("enum").and_then(|e| e.as_array())
        && !en.iter().any(|e| e == val)
    {
        return false;
    }
    if let Some(ty) = ps.get("type").and_then(|t| t.as_str()) {
        let ok = match ty {
            "string" => val.is_string(),
            "integer" | "number" => val.is_number() || val.is_string(),
            "boolean" => val.is_boolean() || val.is_string(),
            "array" => val.is_array(),
            "object" => val.is_object(),
            "null" => val.is_null(),
            _ => true,
        };
        if !ok {
            return false;
        }
    }
    true
}
