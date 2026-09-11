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
    let schema =
        r#"{"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":false}"#;
    let got = json_schema_to_ebnf(schema, &no_space()).unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
    // The oracle: `additionalProperties:false` means the compiled
    // grammar must accept only the declared property and reject any
    // extra key (JSON Schema `additionalProperties` semantics) — not
    // merely "the generated EBNF text parses".
    let opts = no_space();
    assert!(
        schema_accepts(schema, &opts, r#"{"a": 1}"#),
        "declared property alone must be accepted"
    );
    assert!(
        !schema_accepts(schema, &opts, r#"{"a": 1, "b": 2}"#),
        "additionalProperties:false must reject an undeclared key"
    );
}

#[test]
fn object_min_max_properties() {
    let schema = r#"{"type":"object","additionalProperties":{"type":"integer"},"minProperties":1,"maxProperties":3}"#;
    let got = json_schema_to_ebnf(schema, &no_space()).unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
    // Oracle: `minProperties`/`maxProperties` must actually bound the
    // number of properties the compiled grammar accepts, not merely
    // appear as inert numbers that never reach the generated repeat
    // count.
    let opts = no_space();
    assert!(
        !schema_accepts(schema, &opts, r#"{}"#),
        "minProperties:1 must reject the empty object"
    );
    assert!(schema_accepts(schema, &opts, r#"{"a": 1}"#));
    assert!(schema_accepts(schema, &opts, r#"{"a": 1, "b": 2, "c": 3}"#));
    assert!(
        !schema_accepts(schema, &opts, r#"{"a": 1, "b": 2, "c": 3, "d": 4}"#),
        "maxProperties:3 must reject a fourth property"
    );
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
    let schema = r#"{"type":"object","patternProperties":{"^x":{"type":"integer"}}}"#;
    let got = json_schema_to_ebnf(schema, &no_space()).unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
    // Oracle: a `patternProperties` key regex must actually gate which
    // keys are legal, not just decorate a grammar that accepts any key.
    // (xgrammar's regex converter compiles a `pattern` to a *whole-key*
    // match — `^`/`$` are stray no-ops it strips per `regex/mod.rs` —
    // so `"^x"` accepts only the literal key `x`, not any key merely
    // starting with `x`; see that module's documented anchor handling.)
    let opts = no_space();
    assert!(
        schema_accepts(schema, &opts, r#"{"x": 1}"#),
        "the key matching the (whole-key) pattern must be accepted"
    );
    assert!(
        !schema_accepts(schema, &opts, r#"{"y": 1}"#),
        "a key NOT matching the pattern must be rejected (strict_mode has no additionalProperties)"
    );
}

#[test]
fn object_property_names() {
    let schema = r#"{"type":"object","propertyNames":{"type":"string","minLength":1}}"#;
    let got = json_schema_to_ebnf(schema, &no_space()).unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
    // Oracle: `propertyNames` sub-schema constraints (here `minLength:1`)
    // must actually gate keys, not just get parsed and then dropped for
    // an unconstrained key rule.
    let opts = no_space();
    assert!(schema_accepts(schema, &opts, r#"{"a": 1}"#));
    assert!(
        !schema_accepts(schema, &opts, r#"{"": 1}"#),
        "propertyNames minLength:1 must reject the empty-string key"
    );
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
    let schema = r#"{"anyOf":[{"type":"integer"},{"type":"string"}]}"#;
    let got = json_schema_to_ebnf(schema, &no_space()).unwrap();
    assert!(got.contains("root ::="));
    assert!(parse_ebnf_default(&got).is_ok());
    // Oracle: `anyOf` must restrict to the union of its branches, not
    // merely list them alongside an implicit catch-all that admits any
    // JSON value (e.g. a bare boolean, which is in neither branch).
    let opts = no_space();
    assert!(schema_accepts(schema, &opts, "5"));
    assert!(schema_accepts(schema, &opts, "\"x\""));
    assert!(
        !schema_accepts(schema, &opts, "true"),
        "boolean is in neither anyOf branch and must be rejected"
    );
}

#[test]
fn one_of_treated_like_any_of() {
    // NONCLAIM: xgrammar's grammar-based constraint cannot enforce
    // `oneOf`'s "exactly one branch matches" exclusivity at generation
    // time (matching upstream's documented simplification) — this test
    // only claims the union-of-branches restriction that `anyOf` gives,
    // same underlying `generate_any_of` as `any_of_combinator` above
    // (redundant with it for a mutant that widens that shared union to
    // "any"; kept separately because the two exercise disjoint branch
    // types: integer|string there vs integer|boolean here).
    let schema = r#"{"oneOf":[{"type":"integer"},{"type":"boolean"}]}"#;
    let got = json_schema_to_ebnf(schema, &no_space()).unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
    let opts = no_space();
    assert!(schema_accepts(schema, &opts, "5"));
    assert!(schema_accepts(schema, &opts, "true"));
    assert!(
        !schema_accepts(schema, &opts, "\"x\""),
        "string is in neither oneOf branch and must be rejected"
    );
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
    let schema = r#"{"type":["string","integer","null"]}"#;
    let got = json_schema_to_ebnf(schema, &no_space()).unwrap();
    assert!(got.contains("root ::="));
    assert!(parse_ebnf_default(&got).is_ok());
    // Oracle: a type array must *restrict* to the listed types, not
    // merely list them alongside an implicit escape hatch that accepts
    // everything else too.
    let opts = no_space();
    assert!(schema_accepts(schema, &opts, "\"hello\""));
    assert!(schema_accepts(schema, &opts, "5"));
    assert!(schema_accepts(schema, &opts, "null"));
    assert!(
        !schema_accepts(schema, &opts, "true"),
        "boolean is not in the type array and must be rejected"
    );
}

#[test]
fn type_array_empty_is_any() {
    let got = json_schema_to_ebnf(r#"{"type":[]}"#, &no_space()).unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}

// ===================== $ref / $defs =====================

#[test]
fn ref_to_defs() {
    let schema = r##"{"$defs":{"Pos":{"type":"integer","minimum":0}},"type":"object","properties":{"x":{"$ref":"#/$defs/Pos"}},"required":["x"]}"##;
    let got = json_schema_to_ebnf(schema, &no_space()).unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
    // Oracle: the `$ref`-resolved target schema's constraints
    // (`type: integer`) must actually reach the compiled grammar — a
    // resolver that silently degraded every ref to "any" (as it does
    // for a malformed URI) would still pass `parse_ebnf_default`.
    let opts = no_space();
    assert!(schema_accepts(schema, &opts, r#"{"x": 5}"#));
    assert!(
        !schema_accepts(schema, &opts, r#"{"x": "not-an-integer"}"#),
        "$ref target's type:integer must be enforced, not degraded to any"
    );
}
