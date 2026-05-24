// SPDX-License-Identifier: AGPL-3.0-only
//
// TSCG signature renderer — operators TAS, DRO, CFO, CAS, SAD-F.
//
//   TAS  : every delimiter used here — `(` `)` `:` ` ` `?` `|` newline —
//          is a single BPE token in the Qwen / MiniMax / Mistral
//          ByteLevel vocabularies, so the layout is tokenizer-aligned by
//          construction.
//   DRO  : JSON type names are compacted (`string`→`str`, …) and enums
//          are rendered as `a|b|c` instead of a `"enum":[…]` block.
//   CFO  : required parameters are emitted before optional ones.
//   CAS  : the highest-fragility atoms — the tool name and the required
//          parameters — are front-loaded into the signature head.
//   SAD-F: when a tool has many required parameters, a compact `!needs:`
//          recap re-anchors them at the stanza tail (recency).

use crate::tool_parser::ToolDefinition;

use super::sdm::densify;

/// SAD-F fires only when a tool has at least this many required params —
/// below it the recap is pure overhead.
const SAD_F_MIN_REQUIRED: usize = 4;
/// An enum with more than this many variants renders as the bare base
/// type; spelling them all out stops being a density win.
const ENUM_INLINE_MAX: usize = 8;

/// Compile one tool to its TSCG stanza (no trailing newline).
pub fn compile_one(tool: &ToolDefinition) -> String {
    let f = &tool.function;
    let schema = f.parameters.as_ref();

    let (props, required) = match schema {
        Some(s) => (
            s.get("properties").and_then(|p| p.as_object()),
            s.get("required")
                .and_then(|r| r.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
        ),
        None => (None, Vec::new()),
    };

    // CFO: required params first, then optional. Within each group,
    // serde_json's object iteration order (stable for a given input).
    let mut req_params: Vec<(&String, &serde_json::Value)> = Vec::new();
    let mut opt_params: Vec<(&String, &serde_json::Value)> = Vec::new();
    if let Some(props) = props {
        for (name, spec) in props {
            if required.iter().any(|r| r == name) {
                req_params.push((name, spec));
            } else {
                opt_params.push((name, spec));
            }
        }
    }

    // ── Signature line (CAS: name + required params lead) ──
    let mut sig = String::new();
    sig.push_str(&f.name);
    sig.push('(');
    let mut first = true;
    for (name, spec) in req_params.iter() {
        if !first {
            sig.push(' ');
        }
        first = false;
        sig.push_str(name);
        sig.push(':');
        sig.push_str(&compact_type(spec));
    }
    for (name, spec) in opt_params.iter() {
        if !first {
            sig.push(' ');
        }
        first = false;
        sig.push_str(name);
        sig.push_str("?:");
        sig.push_str(&compact_type(spec));
    }
    sig.push(')');

    let mut out = sig;

    // ── Description line (SDM) ──
    if let Some(desc) = f.description.as_deref() {
        let d = densify(desc);
        if !d.is_empty() {
            out.push_str("\n|");
            out.push_str(&d);
        }
    }

    // ── Per-parameter doc lines (SDM), informative only ──
    for (name, spec) in req_params.iter().chain(opt_params.iter()) {
        if let Some(doc) = param_doc(name, spec) {
            out.push_str("\n  ");
            out.push_str(name);
            out.push_str(": ");
            out.push_str(&doc);
        }
    }

    // ── SAD-F: recap required params for many-arg tools ──
    if req_params.len() >= SAD_F_MIN_REQUIRED {
        out.push_str("\n!needs:");
        for (name, _) in req_params.iter() {
            out.push(' ');
            out.push_str(name);
        }
    }

    out
}

/// DRO: compact a JSON-schema property spec to a short type token.
fn compact_type(spec: &serde_json::Value) -> String {
    // Enum: render variants inline when the set is small.
    if let Some(vals) = spec.get("enum").and_then(|e| e.as_array()) {
        let strs: Vec<String> = vals
            .iter()
            .map(|v| match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect();
        if !strs.is_empty() && strs.len() <= ENUM_INLINE_MAX {
            return strs.join("|");
        }
    }
    match spec.get("type").and_then(|t| t.as_str()) {
        Some("string") => "str".to_string(),
        Some("integer") => "int".to_string(),
        Some("number") => "num".to_string(),
        Some("boolean") => "bool".to_string(),
        Some("object") => "obj".to_string(),
        Some("array") => {
            // `[itemtype]` when the element type is known, else `list`.
            match spec.get("items") {
                Some(items) => format!("[{}]", compact_type(items)),
                None => "list".to_string(),
            }
        }
        Some(other) => other.to_string(),
        None => "any".to_string(),
    }
}

/// SDM-compressed per-parameter doc. Returns `None` when the description
/// is absent or, after densification, adds nothing beyond the param
/// name itself (a trivial echo like param `query` → "the query").
fn param_doc(name: &str, spec: &serde_json::Value) -> Option<String> {
    let raw = spec.get("description").and_then(|d| d.as_str())?;
    let d = densify(raw);
    if d.is_empty() {
        return None;
    }
    // Trivial-echo guard: densified doc is just the param name (any case).
    if d.eq_ignore_ascii_case(name) {
        return None;
    }
    Some(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_parser::{FunctionDefinition, ToolDefinition};

    fn tool(name: &str, desc: &str, params: serde_json::Value) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: Some(desc.to_string()),
                parameters: Some(params),
            },
        }
    }

    #[test]
    fn dro_compacts_types() {
        let t = tool(
            "f",
            "d",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "flag": {"type": "boolean"},
                    "items": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["n", "flag", "items"]
            }),
        );
        let out = compile_one(&t);
        assert!(out.contains("n:int"), "got: {out}");
        assert!(out.contains("flag:bool"), "got: {out}");
        assert!(out.contains("items:[str]"), "got: {out}");
    }

    #[test]
    fn enum_renders_inline() {
        let t = tool(
            "f",
            "d",
            serde_json::json!({
                "type": "object",
                "properties": {"mode": {"type": "string", "enum": ["read", "write"]}},
                "required": ["mode"]
            }),
        );
        assert!(compile_one(&t).contains("mode:read|write"));
    }

    #[test]
    fn cfo_orders_required_first() {
        let t = tool(
            "f",
            "d",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "opt": {"type": "string"},
                    "req": {"type": "string"}
                },
                "required": ["req"]
            }),
        );
        let sig = compile_one(&t);
        let line0 = sig.lines().next().unwrap();
        assert!(
            line0.find("req:str").unwrap() < line0.find("opt?:str").unwrap(),
            "got: {line0}"
        );
    }

    #[test]
    fn sad_f_recaps_many_required() {
        let t = tool(
            "f",
            "d",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "a": {"type": "string"}, "b": {"type": "string"},
                    "c": {"type": "string"}, "d": {"type": "string"}
                },
                "required": ["a", "b", "c", "d"]
            }),
        );
        assert!(compile_one(&t).contains("!needs:"));
    }

    #[test]
    fn no_params_renders_bare_signature() {
        let t = ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "now".to_string(),
                description: Some("Current time".to_string()),
                parameters: None,
            },
        };
        assert_eq!(compile_one(&t), "now()\n|Current time");
    }
}
