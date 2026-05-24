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
    assert_eq!(
        got, expected,
        "\n--- got ---\n{got}\n--- want ---\n{expected}"
    );
}

// ===================== Basic prelude =====================

#[test]
fn empty_schema_is_basic_any() {
    check(
        "{}",
        &format!("{BASIC_NO_SPACE}root ::= basic_any\n"),
        &no_space(),
    );
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
    let got = json_schema_to_ebnf(
        r#"{"type":"integer","minimum":1,"maximum":10}"#,
        &no_space(),
    )
    .unwrap();
    // Range [1,10] uses a generated regex, not the plain basic_integer.
    assert!(got.contains("root ::="));
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn integer_inverted_bounds_unsatisfiable() {
    let err = json_schema_to_ebnf(
        r#"{"type":"integer","minimum":10,"maximum":1}"#,
        &no_space(),
    )
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
    let got = json_schema_to_ebnf(
        r#"{"type":"number","minimum":1.5,"maximum":9.5}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn number_non_integer_bound_for_integer_type_errors() {
    let err = json_schema_to_ebnf(r#"{"type":"integer","minimum":1.5}"#, &no_space()).unwrap_err();
    assert_eq!(err.kind, SchemaErrorKind::InvalidSchema);
}

#[test]
fn number_float_bound_dot_is_literal_not_wildcard() {
    // Regression for upstream c4cf39f (#642): a float boundary's decimal
    // point must compile to a literal-dot EBNF token `"."`, never the
    // any-character wildcard `[\0-\U0010ffff]` that an unescaped regex
    // `.` produces. Otherwise the grammar would accept `0,5` etc.
    let got = json_schema_to_ebnf(r#"{"type":"number","minimum":0.5}"#, &no_space()).unwrap();
    let root_line = got
        .lines()
        .find(|l| l.starts_with("root ::="))
        .expect("root rule present");
    assert!(
        root_line.contains(r#""0" "." "5""#),
        "float boundary `0.5` must compile to a literal dot:\n{root_line}"
    );
    assert!(
        !root_line.contains(r"[\0-\U0010ffff]"),
        "float boundary dot must not be the wildcard:\n{root_line}"
    );
    assert!(parse_ebnf_default(&got).is_ok());
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
    let got = json_schema_to_ebnf(r#"{"type":"string","pattern":"[0-9]+"}"#, &no_space()).unwrap();
    assert!(got.contains("root ::="));
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn string_format_date() {
    let got = json_schema_to_ebnf(r#"{"type":"string","format":"date"}"#, &no_space()).unwrap();
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
    let got = json_schema_to_ebnf(r#"{"type":"string","minLength":3}"#, &no_space()).unwrap();
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
    let err = json_schema_to_ebnf(r#"{"type":"array","minItems":5,"maxItems":2}"#, &no_space())
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
    let err = json_schema_to_ebnf(r#"{"type":"array","maxItems":-1}"#, &no_space()).unwrap_err();
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

// Object-schema test cases — split into a child module to keep
// this file under the 500-LoC cap.
#[path = "tests_objects.rs"]
mod objects;
