// SPDX-License-Identifier: AGPL-3.0-only
//! Second half of the schema-object tests, split from `tests_objects.rs`.
//!
//! Purely mechanical: #897 grew that file from 495 to 579 lines and the repo
//! caps .rs at 500. The allow-list demands a rationale and a tracking issue for
//! an exception, so the file is split instead. Every test below is a byte-exact
//! copy of what was there.

// SPDX-License-Identifier: AGPL-3.0-only
//
// JSON-schema converter tests — object-schema cases. Child module
// of `tests` (see tests.rs); kept separate for the 500-LoC cap.

use super::*;
use crate::grammar::parse_ebnf_default;

#[test]
fn ref_to_definitions() {
    let schema = r##"{"definitions":{"Name":{"type":"string"}},"type":"object","properties":{"n":{"$ref":"#/definitions/Name"}},"required":["n"]}"##;
    let got = json_schema_to_ebnf(schema, &no_space()).unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
    // Same oracle as `ref_to_defs`, but for the legacy `definitions`
    // (draft-4/6/7) pointer root instead of `$defs`.
    let opts = no_space();
    assert!(schema_accepts(schema, &opts, r#"{"n": "hello"}"#));
    assert!(
        !schema_accepts(schema, &opts, r#"{"n": 5}"#),
        "$ref target's type:string must be enforced, not degraded to any"
    );
}

#[test]
fn ref_recursive_self() {
    // A node referring to itself via `#` must not infinite-loop.
    let got = json_schema_to_ebnf(
        r##"{"type":"object","properties":{"child":{"$ref":"#"}},"required":["child"]}"##,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn ref_unresolvable_errors() {
    let err = json_schema_to_ebnf(
        r##"{"type":"object","properties":{"x":{"$ref":"#/$defs/Missing"}},"required":["x"]}"##,
        &no_space(),
    )
    .unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::RefResolution);
}

#[test]
fn ref_malformed_uri_falls_back_to_any() {
    // C++ warns and yields "any" for a non-`#/...` URI.
    let got = json_schema_to_ebnf(
        r#"{"type":"object","properties":{"x":{"$ref":"http://example.com"}},"required":["x"]}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn ref_must_be_string() {
    let err = json_schema_to_ebnf(r#"{"$ref":123}"#, &no_space()).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

// ===================== Nested schemas =====================

#[test]
fn deeply_nested_object() {
    let got = json_schema_to_ebnf(
        r#"{"type":"object","properties":{"a":{"type":"object","properties":{"b":{"type":"object","properties":{"c":{"type":"integer"}},"required":["c"]}},"required":["b"]}},"required":["a"]}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn nested_array_of_objects() {
    let got = json_schema_to_ebnf(
        r#"{"type":"array","items":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn nested_anyof_inside_array() {
    let got = json_schema_to_ebnf(
        r#"{"type":"array","items":{"anyOf":[{"type":"integer"},{"type":"string"}]}}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

// ===================== Annotations are ignored =====================

#[test]
fn annotations_do_not_affect_output() {
    let plain = json_schema_to_ebnf(r#"{"type":"integer"}"#, &no_space()).unwrap();
    let annotated = json_schema_to_ebnf(
        r#"{"type":"integer","title":"X","description":"d","default":0}"#,
        &no_space(),
    )
    .unwrap();
    assert_eq!(plain, annotated);
}

// ===================== Whitespace / indent options =====================

#[test]
fn indent_option_produces_newlines() {
    let opts = SchemaConverterOptions {
        any_whitespace: false,
        indent: Some(2),
        ..SchemaConverterOptions::default()
    };
    let got = json_schema_to_ebnf(
        r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#,
        &opts,
    )
    .unwrap();
    assert!(got.contains("\\n"));
}

#[test]
fn max_whitespace_cnt_caps_whitespace() {
    let opts = SchemaConverterOptions {
        max_whitespace_cnt: Some(4),
        ..SchemaConverterOptions::default()
    };
    let got = json_schema_to_ebnf(r#"{"type":"array","items":{"type":"integer"}}"#, &opts).unwrap();
    assert!(got.contains("[ \\n\\t]{0,4}"));
}

#[test]
fn custom_separators() {
    let opts = SchemaConverterOptions {
        any_whitespace: false,
        separators: Some((",".to_string(), ":".to_string())),
        ..SchemaConverterOptions::default()
    };
    let got = json_schema_to_ebnf(
        r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#,
        &opts,
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

// ===================== Generated grammar parses =====================

#[test]
fn generated_grammar_is_parseable() {
    for schema in [
        "{}",
        r#"{"type":"integer"}"#,
        r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}"#,
        r#"{"type":"array","items":{"type":"number"}}"#,
        r#"{"enum":["x","y"]}"#,
        r#"{"anyOf":[{"type":"integer"},{"type":"null"}]}"#,
    ] {
        let g = json_schema_to_grammar(schema, &no_space()).expect("conversion ok");
        // GrammarData should have at least the root rule.
        let _ = g;
    }
}

// ===================== XML tool-calling formats =====================

#[test]
fn qwen_xml_tool_calling() {
    let got = qwen_xml_tool_calling_to_ebnf(
        r#"{"type":"object","properties":{"loc":{"type":"string"}},"required":["loc"]}"#,
    )
    .unwrap();
    assert!(got.contains("xml_string ::= TagDispatch"));
    assert!(got.contains("<parameter=loc>"));
    assert!(got.contains("</parameter>"));
}

#[test]
fn minimax_xml_tool_calling() {
    let got = minimax_xml_tool_calling_to_ebnf(
        r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#,
    )
    .unwrap();
    assert!(got.contains("<parameter name=\\\"city\\\">"));
}

#[test]
fn deepseek_xml_tool_calling() {
    let got = deepseek_xml_tool_calling_to_ebnf(
        r#"{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}"#,
    )
    .unwrap();
    assert!(got.contains("DSML"));
}

#[test]
fn xml_tool_calling_requires_object_type() {
    let err = qwen_xml_tool_calling_to_ebnf(r#"{"type":"integer"}"#).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

#[test]
fn xml_tool_calling_rejects_boolean_schema() {
    let err = qwen_xml_tool_calling_to_ebnf("true").unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

#[test]
fn xml_inner_values_use_json_format() {
    // A nested object inside a parameter uses JSON braces.
    let got = qwen_xml_tool_calling_to_ebnf(
        r#"{"type":"object","properties":{"obj":{"type":"object","properties":{"k":{"type":"integer"}},"required":["k"]}},"required":["obj"]}"#,
    )
    .unwrap();
    assert!(got.contains("\"{\""));
}

// ===================== Malformed input =====================

#[test]
fn malformed_json_errors() {
    let err = json_schema_to_ebnf("{not json", &no_space()).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

#[test]
fn non_object_non_bool_schema_errors() {
    let err = json_schema_to_ebnf("42", &no_space()).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

#[test]
fn trailing_garbage_errors() {
    let err = json_schema_to_ebnf("{} trailing", &no_space()).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

// ===================== Builtin JSON grammar =====================

#[test]
fn builtin_json_grammar_is_basic_any() {
    let ebnf = builtin_json_grammar_ebnf();
    assert!(ebnf.contains("basic_any"));
    assert!(ebnf.contains("root ::= basic_any"));
}

#[test]
fn builtin_json_grammar_parses() {
    let g = builtin_json_grammar().expect("builtin grammar should build");
    let _ = g;
}

// patternProperties + properties tests live in tests_pattern_props.rs
// (split out to keep this file under the 500-LoC cap).
#[path = "tests_pattern_props.rs"]
mod pattern_props;

// ===================== Caching / dedup =====================

#[test]
fn identical_subschemas_share_a_rule() {
    // Two properties with the same integer schema reuse basic_integer.
    let got = json_schema_to_ebnf(
        r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}"#,
        &no_space(),
    )
    .unwrap();
    // Both properties should reference basic_integer, not a fresh rule.
    let count = got.matches("basic_integer").count();
    assert!(count >= 2, "expected shared basic_integer references");
}
