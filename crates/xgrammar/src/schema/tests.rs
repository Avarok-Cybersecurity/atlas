// SPDX-License-Identifier: AGPL-3.0-only
//
// Test suite for the JSON-schema -> EBNF converter.
//
// EBNF fixtures are ported from the upstream xgrammar Python tests
// (`tests/python/test_json_schema_converter.py` and
// `test_function_calling_converter.py`). Where a fixture asserts an
// exact grammar string we reproduce it byte-for-byte; elsewhere we
// assert structural properties (rule presence, error kinds, grammar
// parseability).

use super::*;
use crate::grammar::parse_ebnf_default;

/// The basic-rule prelude every converted grammar starts with, in
/// `any_whitespace = false` mode — matches the Python
/// `basic_json_rules_ebnf_no_space` fixture.
const BASIC_NO_SPACE: &str = concat!(
    "basic_escape ::= [\"\\\\/bfnrt] | \"u\" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]\n",
    "basic_string_sub ::= (\"\\\"\" | [^\\0-\\x1f\\\"\\\\\\r\\n] basic_string_sub | \"\\\\\" basic_escape basic_string_sub) (= [ \\n\\t]* [,}\\]:])\n",
    "basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object\n",
    "basic_integer ::= (\"0\" | \"-\"? [1-9] [0-9]*)\n",
    "basic_number ::= \"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?\n",
    "basic_string ::= [\"] basic_string_sub\n",
    "basic_boolean ::= \"true\" | \"false\"\n",
    "basic_null ::= \"null\"\n",
    "basic_array ::= ((\"[\" \"\" basic_any (\", \" basic_any)* \"\" \"]\") | (\"[\" \"\" \"]\"))\n",
    "basic_object ::= (\"{\" \"\" basic_string \": \" basic_any (\", \" basic_string \": \" basic_any)* \"\" \"}\") | \"{\" \"}\"\n",
);

/// The basic-rule prelude in `any_whitespace = true` mode — matches
/// the Python `basic_json_rules_ebnf` fixture.
const BASIC_ANY_WS: &str = concat!(
    "basic_escape ::= [\"\\\\/bfnrt] | \"u\" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]\n",
    "basic_string_sub ::= (\"\\\"\" | [^\\0-\\x1f\\\"\\\\\\r\\n] basic_string_sub | \"\\\\\" basic_escape basic_string_sub) (= [ \\n\\t]* [,}\\]:])\n",
    "basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object\n",
    "basic_integer ::= (\"0\" | \"-\"? [1-9] [0-9]*)\n",
    "basic_number ::= \"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?\n",
    "basic_string ::= [\"] basic_string_sub\n",
    "basic_boolean ::= \"true\" | \"false\"\n",
    "basic_null ::= \"null\"\n",
    "basic_array ::= ((\"[\" [ \\n\\t]* basic_any ([ \\n\\t]* \",\" [ \\n\\t]* basic_any)* [ \\n\\t]* \"]\") | (\"[\" [ \\n\\t]* \"]\"))\n",
    "basic_object ::= (\"{\" [ \\n\\t]* basic_string [ \\n\\t]* \":\" [ \\n\\t]* basic_any ([ \\n\\t]* \",\" [ \\n\\t]* basic_string [ \\n\\t]* \":\" [ \\n\\t]* basic_any)* [ \\n\\t]* \"}\") | \"{\" [ \\n\\t]* \"}\"\n",
);

/// Options helper: `any_whitespace = false`, strict.
fn no_space() -> SchemaConverterOptions {
    SchemaConverterOptions {
        any_whitespace: false,
        ..SchemaConverterOptions::default()
    }
}

/// Convert and assert exact equality with `expected`.
fn check(schema: &str, expected: &str, opts: &SchemaConverterOptions) {
    let got = json_schema_to_ebnf(schema, opts).expect("conversion should succeed");
    assert_eq!(got, expected, "\n--- got ---\n{got}\n--- want ---\n{expected}");
}

// ===================== Basic prelude =====================

#[test]
fn empty_schema_is_basic_any() {
    check("{}", &format!("{BASIC_NO_SPACE}root ::= basic_any\n"), &no_space());
}

#[test]
fn any_whitespace_prelude_matches_fixture() {
    let got = json_schema_to_ebnf("{}", &SchemaConverterOptions::default()).unwrap();
    assert!(got.starts_with(BASIC_ANY_WS), "prelude mismatch:\n{got}");
}

#[test]
fn boolean_true_schema_is_any() {
    // A `true` schema accepts any value. Its cache key is the literal
    // `true`, distinct from `{}`, so the converter inlines the "any"
    // body rather than referencing the `basic_any` rule.
    check(
        "true",
        &format!(
            "{BASIC_NO_SPACE}root ::= basic_number | basic_string | basic_boolean | \
             basic_null | basic_array | basic_object\n"
        ),
        &no_space(),
    );
}

#[test]
fn boolean_false_schema_is_unsatisfiable() {
    let err = json_schema_to_ebnf("false", &no_space()).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::UnsatisfiableSchema);
}

// ===================== Scalar types =====================

#[test]
fn integer_type() {
    check(
        r#"{"type":"integer"}"#,
        &format!("{BASIC_NO_SPACE}root ::= basic_integer\n"),
        &no_space(),
    );
}

#[test]
fn number_type() {
    check(
        r#"{"type":"number"}"#,
        &format!("{BASIC_NO_SPACE}root ::= basic_number\n"),
        &no_space(),
    );
}

#[test]
fn string_type() {
    check(
        r#"{"type":"string"}"#,
        &format!("{BASIC_NO_SPACE}root ::= basic_string\n"),
        &no_space(),
    );
}

#[test]
fn boolean_type() {
    check(
        r#"{"type":"boolean"}"#,
        &format!("{BASIC_NO_SPACE}root ::= basic_boolean\n"),
        &no_space(),
    );
}

#[test]
fn null_type() {
    check(
        r#"{"type":"null"}"#,
        &format!("{BASIC_NO_SPACE}root ::= basic_null\n"),
        &no_space(),
    );
}

#[test]
fn unsupported_type_errors() {
    let err = json_schema_to_ebnf(r#"{"type":"widget"}"#, &no_space()).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

#[test]
fn non_string_type_errors() {
    let err = json_schema_to_ebnf(r#"{"type":123}"#, &no_space()).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

// ===================== Integer / number bounds =====================

#[test]
fn integer_with_bounds_uses_range_regex() {
    let got =
        json_schema_to_ebnf(r#"{"type":"integer","minimum":1,"maximum":10}"#, &no_space())
            .unwrap();
    // Range [1,10] uses a generated regex, not the plain basic_integer.
    assert!(got.contains("root ::="));
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn integer_inverted_bounds_unsatisfiable() {
    let err =
        json_schema_to_ebnf(r#"{"type":"integer","minimum":10,"maximum":1}"#, &no_space())
            .unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::UnsatisfiableSchema);
}

#[test]
fn integer_exclusive_bounds() {
    let got = json_schema_to_ebnf(
        r#"{"type":"integer","exclusiveMinimum":0,"exclusiveMaximum":11}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn number_with_bounds() {
    let got =
        json_schema_to_ebnf(r#"{"type":"number","minimum":1.5,"maximum":9.5}"#, &no_space())
            .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn number_non_integer_bound_for_integer_type_errors() {
    let err = json_schema_to_ebnf(r#"{"type":"integer","minimum":1.5}"#, &no_space())
        .unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

// ===================== Range / float regex =====================

#[test]
fn range_regex_examples() {
    assert_eq!(generate_range_regex(Some(12), Some(16)), r"^((1[2-6]))$");
    assert_eq!(generate_range_regex(Some(1), Some(10)), r"^(([1-9]|10))$");
    assert_eq!(generate_range_regex(None, None), r"^-?\d+$");
    assert_eq!(generate_range_regex(Some(5), Some(5)), r"^((5))$");
    assert_eq!(
        generate_range_regex(Some(-5), Some(10)),
        r"^(-([1-5])|0|([1-9]|10))$"
    );
}

#[test]
fn range_regex_inverted_is_empty() {
    assert_eq!(generate_range_regex(Some(10), Some(1)), "^()$");
}

#[test]
fn float_regex_unbounded() {
    assert_eq!(
        generate_float_range_regex(None, None, 6),
        r"^-?\d+(\.\d{1,6})?$"
    );
}

#[test]
fn float_regex_inverted_is_empty() {
    assert_eq!(generate_float_range_regex(Some(9.0), Some(1.0), 6), "^()$");
}

// ===================== Strings: pattern / format / length =====================

#[test]
fn string_pattern() {
    let got = json_schema_to_ebnf(r#"{"type":"string","pattern":"[0-9]+"}"#, &no_space())
        .unwrap();
    assert!(got.contains("root ::="));
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn string_format_date() {
    let got = json_schema_to_ebnf(r#"{"type":"string","format":"date"}"#, &no_space())
        .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn string_format_email_uuid_uri() {
    for fmt in ["email", "uuid", "uri", "ipv4", "date-time", "hostname"] {
        let schema = format!(r#"{{"type":"string","format":"{fmt}"}}"#);
        let got = json_schema_to_ebnf(&schema, &no_space()).unwrap();
        assert!(parse_ebnf_default(&got).is_ok(), "format {fmt} failed");
    }
}

#[test]
fn string_unknown_format_falls_back() {
    // Unknown format degrades to the default string body. The cache
    // key differs from plain `{"type":"string"}`, so the body is
    // inlined (`["] basic_string_sub`) rather than referencing
    // `basic_string`.
    check(
        r#"{"type":"string","format":"nonsense"}"#,
        &format!("{BASIC_NO_SPACE}root ::= [\"] basic_string_sub\n"),
        &no_space(),
    );
}

#[test]
fn string_length_constraints() {
    let got = json_schema_to_ebnf(
        r#"{"type":"string","minLength":2,"maxLength":5}"#,
        &no_space(),
    )
    .unwrap();
    assert!(got.contains("{2,5}"));
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn string_min_length_only() {
    let got =
        json_schema_to_ebnf(r#"{"type":"string","minLength":3}"#, &no_space()).unwrap();
    assert!(got.contains("{3,}"));
}

#[test]
fn string_length_inverted_unsatisfiable() {
    let err = json_schema_to_ebnf(
        r#"{"type":"string","minLength":5,"maxLength":2}"#,
        &no_space(),
    )
    .unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::UnsatisfiableSchema);
}

// ===================== const / enum =====================

#[test]
fn const_string() {
    check(
        r#"{"const":"hello"}"#,
        &format!("{BASIC_NO_SPACE}root ::= \"\\\"hello\\\"\"\n"),
        &no_space(),
    );
}

#[test]
fn const_number() {
    check(
        r#"{"const":42}"#,
        &format!("{BASIC_NO_SPACE}root ::= \"42\"\n"),
        &no_space(),
    );
}

#[test]
fn enum_strings() {
    check(
        r#"{"enum":["a","b","c"]}"#,
        &format!("{BASIC_NO_SPACE}root ::= (\"\\\"a\\\"\") | (\"\\\"b\\\"\") | (\"\\\"c\\\"\")\n"),
        &no_space(),
    );
}

#[test]
fn enum_mixed_values() {
    check(
        r#"{"enum":[1,"a",true]}"#,
        &format!("{BASIC_NO_SPACE}root ::= (\"1\") | (\"\\\"a\\\"\") | (\"true\")\n"),
        &no_space(),
    );
}

#[test]
fn enum_must_be_array() {
    let err = json_schema_to_ebnf(r#"{"enum":"notarray"}"#, &no_space()).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

// ===================== Arrays =====================

#[test]
fn array_of_strings_any_ws() {
    check(
        r#"{"type":"array","items":{"type":"string"}}"#,
        &format!(
            "{BASIC_ANY_WS}root ::= ((\"[\" [ \\n\\t]* basic_string ([ \\n\\t]* \",\" \
             [ \\n\\t]* basic_string)* [ \\n\\t]* \"]\") | (\"[\" [ \\n\\t]* \"]\"))\n"
        ),
        &SchemaConverterOptions::default(),
    );
}

#[test]
fn array_prefix_items_tuple() {
    let got = json_schema_to_ebnf(
        r#"{"type":"array","prefixItems":[{"type":"string"},{"type":"integer"}]}"#,
        &no_space(),
    )
    .unwrap();
    assert!(got.contains("root ::="));
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn array_min_max_items() {
    let got = json_schema_to_ebnf(
        r#"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":3}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn array_min_greater_than_max_unsatisfiable() {
    let err = json_schema_to_ebnf(
        r#"{"type":"array","minItems":5,"maxItems":2}"#,
        &no_space(),
    )
    .unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::UnsatisfiableSchema);
}

#[test]
fn array_items_false_disallows_additional() {
    let got = json_schema_to_ebnf(
        r#"{"type":"array","prefixItems":[{"type":"integer"}],"items":false}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn array_max_items_negative_errors() {
    let err = json_schema_to_ebnf(
        r#"{"type":"array","maxItems":-1}"#,
        &no_space(),
    )
    .unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

#[test]
fn array_non_strict_adds_any_items() {
    let opts = SchemaConverterOptions {
        any_whitespace: false,
        strict_mode: false,
        ..SchemaConverterOptions::default()
    };
    let got = json_schema_to_ebnf(
        r#"{"type":"array","prefixItems":[{"type":"integer"}]}"#,
        &opts,
    )
    .unwrap();
    // Non-strict => trailing additional items allowed.
    assert!(got.contains("root_additional"));
}

// ===================== Objects =====================

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
    let err = json_schema_to_ebnf(
        r#"{"type":"object","required":"foo"}"#,
        &no_space(),
    )
    .unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

#[test]
fn object_properties_must_be_object() {
    let err =
        json_schema_to_ebnf(r#"{"type":"object","properties":[]}"#, &no_space())
            .unwrap_err();
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
    let got = json_schema_to_ebnf(
        r#"{"allOf":[{"type":"integer"}]}"#,
        &no_space(),
    )
    .unwrap();
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
    let got =
        json_schema_to_ebnf(r#"{"type":["string","integer","null"]}"#, &no_space())
            .unwrap();
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
    let got = json_schema_to_ebnf(r#"{"type":"array","items":{"type":"integer"}}"#, &opts)
        .unwrap();
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
