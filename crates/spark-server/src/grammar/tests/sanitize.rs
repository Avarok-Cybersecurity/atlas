// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn test_enforce_min_length_on_required_strings() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "description": {"type": "string"},
            "count": {"type": "integer"},
            "optional_str": {"type": "string"},
            "has_min": {"type": "string", "minLength": 5}
        },
        "required": ["command", "description", "count", "has_min"]
    });

    let result = enforce_min_length_on_required_strings(&schema);
    let props = result["properties"].as_object().unwrap();

    // Required string without minLength => minLength: 1 added
    assert_eq!(props["command"]["minLength"], 1);
    assert_eq!(props["description"]["minLength"], 1);

    // Required integer => no minLength added
    assert!(props["count"].get("minLength").is_none());

    // Optional string => not touched
    assert!(props["optional_str"].get("minLength").is_none());

    // Already has minLength => not overwritten
    assert_eq!(props["has_min"]["minLength"], 5);
}

#[test]
fn test_enforce_min_length_no_required() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        }
    });
    let result = enforce_min_length_on_required_strings(&schema);
    // No required array => nothing changed
    assert!(result["properties"]["name"].get("minLength").is_none());
}

#[test]
fn test_enforce_min_length_empty_schema() {
    let schema = serde_json::json!({"type": "object", "properties": {}});
    let result = enforce_min_length_on_required_strings(&schema);
    assert_eq!(result, schema);
}

// ── Schema sanitization tests ──

#[test]
fn test_sanitize_empty_enum() {
    let schema = serde_json::json!({"type": "string", "enum": []});
    let result = sanitize_schema_for_grammar(&schema).unwrap();
    assert!(result.get("enum").is_none());
    assert_eq!(result["type"], "string");
}

#[test]
fn test_sanitize_nonempty_enum_unchanged() {
    let schema = serde_json::json!({"type": "string", "enum": ["a", "b"]});
    let result = sanitize_schema_for_grammar(&schema).unwrap();
    assert_eq!(result["enum"].as_array().unwrap().len(), 2);
}

#[test]
fn test_sanitize_empty_anyof() {
    let schema = serde_json::json!({"anyOf": []});
    let result = sanitize_schema_for_grammar(&schema).unwrap();
    assert!(result.get("anyOf").is_none());
}

#[test]
fn test_sanitize_empty_oneof() {
    let schema = serde_json::json!({"oneOf": []});
    let result = sanitize_schema_for_grammar(&schema).unwrap();
    assert!(result.get("oneOf").is_none());
}

#[test]
fn test_sanitize_single_element_anyof() {
    let schema = serde_json::json!({"anyOf": [{"type": "string"}]});
    let result = sanitize_schema_for_grammar(&schema).unwrap();
    assert!(result.get("anyOf").is_none());
    assert_eq!(result["type"], "string");
}

#[test]
fn test_sanitize_multi_element_anyof_preserved() {
    let schema = serde_json::json!({"anyOf": [{"type": "string"}, {"type": "integer"}]});
    let result = sanitize_schema_for_grammar(&schema).unwrap();
    assert_eq!(result["anyOf"].as_array().unwrap().len(), 2);
}

#[test]
fn test_sanitize_empty_properties_no_additional() {
    let schema = serde_json::json!({"type": "object", "properties": {}});
    let result = sanitize_schema_for_grammar(&schema).unwrap();
    assert_eq!(result["additionalProperties"], true);
}

#[test]
fn test_sanitize_object_no_properties_key() {
    let schema = serde_json::json!({"type": "object"});
    let result = sanitize_schema_for_grammar(&schema).unwrap();
    assert_eq!(result["additionalProperties"], true);
}

#[test]
fn test_sanitize_object_with_properties_untouched() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"name": {"type": "string"}}
    });
    let result = sanitize_schema_for_grammar(&schema).unwrap();
    assert!(result.get("additionalProperties").is_none());
}

#[test]
fn test_sanitize_object_with_additional_properties_untouched() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {},
        "additionalProperties": false
    });
    let result = sanitize_schema_for_grammar(&schema).unwrap();
    assert_eq!(result["additionalProperties"], false);
}

#[test]
fn test_sanitize_allof_single() {
    let schema = serde_json::json!({
        "allOf": [{"type": "object", "properties": {"x": {"type": "string"}}}]
    });
    let result = sanitize_schema_for_grammar(&schema).unwrap();
    assert!(result.get("allOf").is_none());
    assert!(result["properties"]["x"].is_object());
}

#[test]
fn test_sanitize_allof_multiple() {
    let schema = serde_json::json!({
        "allOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"properties": {"b": {"type": "integer"}}, "required": ["b"]}
        ]
    });
    let result = sanitize_schema_for_grammar(&schema).unwrap();
    assert!(result.get("allOf").is_none());
    assert!(result["properties"]["a"].is_object());
    assert!(result["properties"]["b"].is_object());
}

#[test]
fn test_sanitize_nested_empty_enum() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": []},
            "name": {"type": "string"}
        },
        "required": ["name"]
    });
    let result = sanitize_schema_for_grammar(&schema).unwrap();
    assert!(result["properties"]["status"].get("enum").is_none());
}

#[test]
fn test_sanitize_ref_resolution() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "item": {"$ref": "#/$defs/Item"}
        },
        "$defs": {
            "Item": {"type": "string"}
        }
    });
    let result = sanitize_schema_for_grammar(&schema).unwrap();
    assert_eq!(result["properties"]["item"]["type"], "string");
}

#[test]
fn test_sanitize_false_schema_returns_none() {
    assert!(sanitize_schema_for_grammar(&serde_json::Value::Bool(false)).is_none());
}

#[test]
fn test_sanitize_true_schema() {
    let result = sanitize_schema_for_grammar(&serde_json::Value::Bool(true)).unwrap();
    assert!(result.is_object());
}

#[test]
fn test_sanitize_deeply_nested() {
    let mut schema = serde_json::json!({"type": "string"});
    for _ in 0..35 {
        schema = serde_json::json!({"type": "object", "properties": {"n": schema}});
    }
    assert!(sanitize_schema_for_grammar(&schema).is_some());
}

// ── Integration: grammar compilation with problematic schemas ──

#[test]
fn test_hermes_grammar_empty_enum_tool() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "set_status".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": []}
                }
            })),
        },
    }];
    // Must not crash — either compiles or gracefully returns an error.
    let _result = engine.compile_hermes_tool_grammar(&tools, false);
}

#[test]
fn test_qwen3_grammar_empty_properties() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "do_nothing".to_string(),
            description: None,
            parameters: Some(serde_json::json!({"type": "object", "properties": {}})),
        },
    }];
    // Must not crash.
    let _result = engine.compile_qwen3_coder_tool_grammar(&tools, false);
}

#[test]
fn test_hermes_grammar_empty_anyof_tool() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "test_tool".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "value": {"anyOf": []}
                }
            })),
        },
    }];
    let _result = engine.compile_hermes_tool_grammar(&tools, false);
}
