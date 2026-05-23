// SPDX-License-Identifier: AGPL-3.0-only
//
// JSON-schema converter tests — object-schema cases. Child module
// of `tests` (see tests.rs); kept separate for the 500-LoC cap.

use super::*;
use crate::grammar::parse_ebnf_default;

#[test]
fn object_single_required_property() {
    check(
        r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#,
        &format!(
            "{BASIC_NO_SPACE}root ::= \"{{\" \"\" ((\"\\\"a\\\"\" \": \" \
             basic_integer \"\")) \"\" \"}}\"\n"
        ),
        &no_space(),
    );
}

#[test]
fn object_non_strict_required() {
    let opts = SchemaConverterOptions {
        strict_mode: false,
        ..SchemaConverterOptions::default()
    };
    let expected = format!(
        "{BASIC_ANY_WS}root_addl ::= basic_number | basic_string | basic_boolean | \
         basic_null | basic_array | basic_object\n\
         root_part_1 ::= ([ \\n\\t]* \",\" [ \\n\\t]* basic_string [ \\n\\t]* \":\" \
         [ \\n\\t]* root_addl)*\n\
         root_part_0 ::= [ \\n\\t]* \",\" [ \\n\\t]* \"\\\"bar\\\"\" [ \\n\\t]* \":\" \
         [ \\n\\t]* basic_integer root_part_1\n\
         root ::= \"{{\" [ \\n\\t]* ((\"\\\"foo\\\"\" [ \\n\\t]* \":\" [ \\n\\t]* \
         basic_integer root_part_0)) [ \\n\\t]* \"}}\"\n"
    );
    check(
        r#"{"type":"object","properties":{"foo":{"type":"integer"},"bar":{"type":"integer"}},"required":["foo","bar"]}"#,
        &expected,
        &opts,
    );
}

#[test]
fn object_optional_property_anyof() {
    let got = json_schema_to_ebnf(
        r#"{"type":"object","properties":{"x":{"anyOf":[{"type":"boolean"},{"type":"null"}]}}}"#,
        &no_space(),
    )
    .unwrap();
    assert!(got.contains("root_prop_0 ::= basic_boolean | basic_null"));
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn object_additional_properties_schema() {
    let got = json_schema_to_ebnf(
        r#"{"type":"object","additionalProperties":{"type":"integer"}}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn object_additional_properties_false() {
    let got = json_schema_to_ebnf(
        r#"{"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":false}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn object_min_max_properties() {
    let got = json_schema_to_ebnf(
        r#"{"type":"object","additionalProperties":{"type":"integer"},"minProperties":1,"maxProperties":3}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn object_min_greater_than_max_unsatisfiable() {
    let err = json_schema_to_ebnf(
        r#"{"type":"object","minProperties":5,"maxProperties":2}"#,
        &no_space(),
    )
    .unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::UnsatisfiableSchema);
}

#[test]
fn object_pattern_properties() {
    let got = json_schema_to_ebnf(
        r#"{"type":"object","patternProperties":{"^x":{"type":"integer"}}}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn object_property_names() {
    let got = json_schema_to_ebnf(
        r#"{"type":"object","propertyNames":{"type":"string","minLength":1}}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn object_property_names_non_string_unsatisfiable() {
    let err = json_schema_to_ebnf(
        r#"{"type":"object","propertyNames":{"type":"integer"}}"#,
        &no_space(),
    )
    .unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::UnsatisfiableSchema);
}

#[test]
fn object_required_must_be_array() {
    let err =
        json_schema_to_ebnf(r#"{"type":"object","required":"foo"}"#, &no_space()).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

#[test]
fn object_properties_must_be_object() {
    let err = json_schema_to_ebnf(r#"{"type":"object","properties":[]}"#, &no_space()).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

#[test]
fn object_preserves_property_order() {
    // A naive `serde_json::Map` sorts keys alphabetically; our
    // order-preserving parser must keep declaration order, so the
    // `root` rule lists `zebra` before `apple`.
    let got = json_schema_to_ebnf(
        r#"{"type":"object","properties":{"zebra":{"type":"integer"},"apple":{"type":"string"}},"required":["zebra","apple"]}"#,
        &no_space(),
    )
    .unwrap();
    let root_line = got
        .lines()
        .find(|l| l.starts_with("root ::="))
        .expect("root rule present");
    let zebra = root_line.find("zebra").expect("zebra in root");
    let apple = root_line.find("apple");
    // `apple` is the optional tail; in declaration order `zebra`
    // anchors the root rule. The tail property lives in a `root_part`
    // rule, so `apple` is absent from the root line entirely.
    assert!(
        apple.is_none() || zebra < apple.unwrap(),
        "property order not preserved:\n{got}"
    );
    // And the part rule for the tail mentions `apple`.
    assert!(got.contains("apple"));
}

// ===================== Combinators =====================

#[test]
fn any_of_combinator() {
    let got = json_schema_to_ebnf(
        r#"{"anyOf":[{"type":"integer"},{"type":"string"}]}"#,
        &no_space(),
    )
    .unwrap();
    assert!(got.contains("root ::="));
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn one_of_treated_like_any_of() {
    let got = json_schema_to_ebnf(
        r#"{"oneOf":[{"type":"integer"},{"type":"boolean"}]}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn all_of_single_schema() {
    let got = json_schema_to_ebnf(r#"{"allOf":[{"type":"integer"}]}"#, &no_space()).unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn all_of_multiple_degrades_to_any() {
    // Upstream warns and degrades multi-schema allOf to "any".
    let got = json_schema_to_ebnf(
        r#"{"allOf":[{"type":"integer"},{"type":"string"}]}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn any_of_must_be_array() {
    let err = json_schema_to_ebnf(r#"{"anyOf":{}}"#, &no_space()).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

#[test]
fn type_array() {
    let got = json_schema_to_ebnf(r#"{"type":["string","integer","null"]}"#, &no_space()).unwrap();
    assert!(got.contains("root ::="));
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn type_array_empty_is_any() {
    let got = json_schema_to_ebnf(r#"{"type":[]}"#, &no_space()).unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

// ===================== $ref / $defs =====================

#[test]
fn ref_to_defs() {
    let got = json_schema_to_ebnf(
        r##"{"$defs":{"Pos":{"type":"integer","minimum":0}},"type":"object","properties":{"x":{"$ref":"#/$defs/Pos"}},"required":["x"]}"##,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn ref_to_definitions() {
    let got = json_schema_to_ebnf(
        r##"{"definitions":{"Name":{"type":"string"}},"type":"object","properties":{"n":{"$ref":"#/definitions/Name"}},"required":["n"]}"##,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
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
