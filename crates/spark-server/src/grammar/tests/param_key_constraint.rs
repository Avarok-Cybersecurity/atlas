// SPDX-License-Identifier: AGPL-3.0-only

//! Echo-attractor fix (2026-07-09): the qwen3_coder `<parameter=NAME>` key
//! slot is constrained to the tool schema's property names, so a model
//! drifting into the wire-format echo attractor (`<parameter=parameter>…`,
//! `<parameter=write>…` — live opencode collapse at 42.5k tokens) can no
//! longer emit a mis-keyed parameter: xgrammar's mask forces the key slot
//! onto a real schema property.

use super::super::compile_tools::{schema_param_names, xml_param_value_body_ebnf};

#[test]
fn schema_param_names_extracts_properties() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}},
        "required": ["content", "filePath"]
    });
    let mut names = schema_param_names(&schema).expect("closed schema must constrain");
    names.sort();
    assert_eq!(names, vec!["content".to_string(), "filePath".to_string()]);
}

#[test]
fn schema_param_names_leaves_open_schemas_unconstrained() {
    // Explicitly permissive additionalProperties → keep the identifier rule.
    let open = serde_json::json!({
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": true
    });
    assert_eq!(schema_param_names(&open), None);
    // Empty / missing properties → nothing to constrain to.
    assert_eq!(
        schema_param_names(&serde_json::json!({"type": "object"})),
        None
    );
    assert_eq!(
        schema_param_names(&serde_json::json!({"type": "object", "properties": {}})),
        None
    );
    // additionalProperties: false is as closed as absent.
    let closed = serde_json::json!({
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": false
    });
    assert_eq!(schema_param_names(&closed), Some(vec!["a".to_string()]));
}

#[test]
fn body_ebnf_constrains_paramname_to_schema_alternation() {
    let names = vec!["filePath".to_string(), "content".to_string()];
    let ebnf = xml_param_value_body_ebnf("</parameter>", Some(&names));
    assert!(
        ebnf.contains(r#"paramname ::= "filePath" | "content""#),
        "constrained rule must be a literal alternation of schema keys:\n{ebnf}"
    );
    assert!(
        !ebnf.contains("[a-zA-Z_] [a-zA-Z_0-9]*"),
        "the permissive identifier rule must be gone when names are known:\n{ebnf}"
    );
}

#[test]
fn body_ebnf_keeps_identifier_rule_without_names() {
    for names in [None, Some(&[][..])] {
        let ebnf = xml_param_value_body_ebnf("</parameter>", names);
        assert!(
            ebnf.contains("paramname ::= [a-zA-Z_] [a-zA-Z_0-9]*"),
            "schema-less path must keep the historical identifier rule:\n{ebnf}"
        );
    }
}

#[test]
fn body_ebnf_first_content_allows_lt_via_close_ladder() {
    let ebnf = xml_param_value_body_ebnf("</parameter>", None);
    // The first-content rule must carry the `<`-arms of the close ladder …
    assert!(
        ebnf.contains(r#"first_content ::= [^ \t\r\n<=>] | "<" [^/]"#),
        "first_content must allow `<` unless it starts the close tag:\n{ebnf}"
    );
    // … including the deepest prefix arm (everything but the final `>`).
    assert!(
        ebnf.contains(r#""</parameter" [^>]"#),
        "the full close-prefix arm must be present:\n{ebnf}"
    );
}

#[test]
fn p1_opts_empty_value_and_force_close_shapes() {
    use super::super::compile_tools::xml_param_value_body_ebnf_opts;
    // Defaults (both off) == the shipped shape.
    let d = xml_param_value_body_ebnf_opts("</parameter>", None, false, false);
    assert!(d.contains("value ::= leading_ws first_content rest"));
    assert!(d.contains(r#""</parameter" [^>]"#));
    // P1-1: empty value representable.
    let ev = xml_param_value_body_ebnf_opts("</parameter>", None, true, false);
    assert!(
        ev.contains("value ::= leading_ws (first_content rest)?"),
        "empty-value opt-in must make content optional:\n{ev}"
    );
    // P1-2: deepest ladder arm gone → after `</parameter` only `>` is legal.
    let fc = xml_param_value_body_ebnf_opts("</parameter>", None, false, true);
    assert!(
        !fc.contains(r#""</parameter" [^>]"#),
        "force-close must drop the deepest arm:\n{fc}"
    );
    assert!(
        fc.contains(r#""</paramete" [^r]"#),
        "shallower arms must survive:\n{fc}"
    );
    // first_content's `<`-arms follow the same forced ladder.
    assert!(!fc.contains(r#"first_content ::= [^ \t\r\n<=>] | "<" [^/] | "</" [^p] | "</p" [^a] | "</pa" [^r] | "</par" [^a] | "</para" [^m] | "</param" [^e] | "</parame" [^t] | "</paramet" [^e] | "</paramete" [^r] | "</parameter" [^>]"#));
}
