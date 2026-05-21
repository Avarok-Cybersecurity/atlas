// SPDX-License-Identifier: AGPL-3.0-only
//
// JSON-schema converter tests — `patternProperties` / `propertyNames`
// coexisting with named `properties`. Regression coverage for the port
// of upstream commit a6aeabb (#594). Child module of `tests::objects`;
// kept separate for the 500-LoC cap.

use super::super::*;
use crate::grammar::parse_ebnf_default;

#[test]
fn pattern_properties_alongside_properties_keeps_named_props() {
    // Regression for upstream a6aeabb (#594): when both `properties` and
    // `patternProperties` are present, the named properties must NOT be
    // dropped. The named property `name` must still appear as a literal.
    let got = json_schema_to_ebnf(
        r#"{"type":"object",
            "properties":{"name":{"type":"string"}},
            "required":["name"],
            "patternProperties":{"^x-":{"type":"integer"}}}"#,
        &no_space(),
    )
    .unwrap();
    assert!(
        got.contains("\\\"name\\\""),
        "named property must survive when patternProperties is present:\n{got}"
    );
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn multiple_pattern_properties_with_properties() {
    let got = json_schema_to_ebnf(
        r#"{"type":"object",
            "properties":{"id":{"type":"integer"}},
            "patternProperties":{"^a":{"type":"string"},"^b":{"type":"boolean"}}}"#,
        &no_space(),
    )
    .unwrap();
    assert!(
        got.contains("\\\"id\\\""),
        "named property must survive:\n{got}"
    );
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn property_names_alongside_properties() {
    let got = json_schema_to_ebnf(
        r#"{"type":"object",
            "properties":{"title":{"type":"string"}},
            "propertyNames":{"pattern":"^[a-z]+$"},
            "additionalProperties":{"type":"integer"}}"#,
        &no_space(),
    )
    .unwrap();
    assert!(
        got.contains("\\\"title\\\""),
        "named property must survive when propertyNames is present:\n{got}"
    );
    assert!(parse_ebnf_default(&got).is_ok());
}

#[test]
fn pattern_properties_only_still_works() {
    // Case 1b (no named properties) must keep behaving as before.
    let got = json_schema_to_ebnf(
        r#"{"type":"object","patternProperties":{"^x-":{"type":"integer"}}}"#,
        &no_space(),
    )
    .unwrap();
    assert!(parse_ebnf_default(&got).is_ok());
}
