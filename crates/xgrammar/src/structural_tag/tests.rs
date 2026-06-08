// SPDX-License-Identifier: AGPL-3.0-only
//
// Structural-tag tests — ported from
// `tests/python/test_structural_tag_converter.py`. Each case builds a
// grammar from a structural tag and checks string acceptance through
// the optimized Earley parser, plus malformed-input error coverage.

use std::sync::Arc;

use super::*;
use crate::earley::EarleyParser;
use crate::grammar::GrammarData;
use crate::grammar::functor::GrammarOptimizer;
use crate::grammar::printer::print_grammar;

/// Optimize a structural-tag grammar so the Earley parser can run.
fn optimized(grammar: GrammarData) -> Arc<GrammarData> {
    Arc::new(GrammarOptimizer::apply(grammar))
}

/// True if `s` is fully accepted by `grammar`.
fn accepts(grammar: Arc<GrammarData>, s: &str) -> bool {
    let mut p = EarleyParser::from_grammar(grammar);
    for b in s.bytes() {
        if !p.advance(b) {
            return false;
        }
    }
    p.is_completed()
}

/// Build a grammar from a structural-tag format object (the `format`
/// body), wrapping it in the full `structural_tag` document.
fn grammar_from_format(format_json: &str) -> StructuralTagResult<GrammarData> {
    let doc = format!(r#"{{"type":"structural_tag","format":{format_json}}}"#);
    structural_tag_to_grammar(&doc)
}

/* ------------------------- const_string ------------------------- */

#[test]
fn const_string_accepts_exact() {
    let g = grammar_from_format(r#"{"type":"const_string","value":"Hello!"}"#).unwrap();
    let g = optimized(g);
    assert!(accepts(Arc::clone(&g), "Hello!"));
    assert!(!accepts(Arc::clone(&g), "Hello"));
    assert!(!accepts(Arc::clone(&g), "Hello!!"));
    assert!(!accepts(g, "HELLO!"));
}

#[test]
fn const_string_empty() {
    let g = grammar_from_format(r#"{"type":"const_string","value":""}"#).unwrap();
    let g = optimized(g);
    assert!(accepts(Arc::clone(&g), ""));
    assert!(!accepts(g, "x"));
}

#[test]
fn const_string_grammar_shape() {
    let g = grammar_from_format(r#"{"type":"const_string","value":"Hello!"}"#).unwrap();
    let printed = print_grammar(&g);
    assert!(printed.contains("const_string ::="));
    assert!(printed.contains("\"Hello!\""));
    assert!(printed.contains("root ::="));
}

/* ------------------------- json_schema -------------------------- */

#[test]
fn json_schema_accepts_conforming_object() {
    let g = grammar_from_format(
        r#"{"type":"json_schema","json_schema":{"type":"object","properties":{"a":{"type":"string"}}}}"#,
    )
    .unwrap();
    let g = optimized(g);
    assert!(accepts(Arc::clone(&g), r#"{"a": "hello"}"#));
    assert!(!accepts(Arc::clone(&g), r#"{"a": 123}"#));
    assert!(!accepts(g, "invalid json"));
}

#[test]
fn json_schema_preserves_property_order() {
    // Property order must survive the embed/re-serialize round-trip:
    // the schema converter emits `required` properties in declared
    // order, so an object with the keys reversed must be rejected.
    let g = grammar_from_format(
        r#"{"type":"json_schema","json_schema":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}}"#,
    )
    .unwrap();
    let g = optimized(g);
    assert!(accepts(Arc::clone(&g), r#"{"name": "Bob", "age": 5}"#));
    assert!(
        !accepts(g, r#"{"age": 5, "name": "Bob"}"#),
        "reversed key order must be rejected — property order preserved"
    );
}

#[test]
fn json_schema_qwen_xml_style() {
    let g = grammar_from_format(
        r#"{"type":"json_schema","style":"qwen_xml","json_schema":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}"#,
    )
    .unwrap();
    let g = optimized(g);
    assert!(accepts(g, "<parameter=name>Bob</parameter>"));
}

#[test]
fn qwen_xml_parameter_type() {
    let g = grammar_from_format(
        r#"{"type":"qwen_xml_parameter","json_schema":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}"#,
    )
    .unwrap();
    let g = optimized(g);
    assert!(accepts(g, "<parameter=name>Bob</parameter>"));
}

/* ------------------------- single tag --------------------------- */

#[test]
fn single_tag_envelope() {
    // The tool-call envelope: begin + json body + end.
    let g = grammar_from_format(
        r#"{"type":"tag","begin":"<tool_call>","content":{"type":"const_string","value":"x"},"end":"</tool_call>"}"#,
    )
    .unwrap();
    let g = optimized(g);
    assert!(accepts(Arc::clone(&g), "<tool_call>x</tool_call>"));
    assert!(!accepts(Arc::clone(&g), "<tool_call>x"));
    assert!(!accepts(g, "x</tool_call>"));
}

#[test]
fn tag_with_multiple_end_markers() {
    let g = grammar_from_format(
        r#"{"type":"tag","begin":"<r>","content":{"type":"const_string","value":"a"},"end":["</r>","</answer>"]}"#,
    )
    .unwrap();
    let g = optimized(g);
    assert!(accepts(Arc::clone(&g), "<r>a</r>"));
    assert!(accepts(Arc::clone(&g), "<r>a</answer>"));
    assert!(!accepts(g, "<r>a</x>"));
}

#[test]
fn tag_with_json_schema_body() {
    let g = grammar_from_format(
        r#"{"type":"tag","begin":"<tool_call>","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"a":{"type":"string"}}}},"end":"</tool_call>"}"#,
    )
    .unwrap();
    let g = optimized(g);
    assert!(accepts(
        Arc::clone(&g),
        r#"<tool_call>{"a": "hi"}</tool_call>"#
    ));
    assert!(!accepts(g, r#"<tool_call>{"a": 1}</tool_call>"#));
}

/* ----------------------- triggered tags ------------------------- */

#[test]
fn triggered_tags_free_text_then_tag() {
    let g = grammar_from_format(
        r#"{"type":"triggered_tags","triggers":["<TOOL>"],"tags":[{"type":"tag","begin":"<TOOL>get","content":{"type":"const_string","value":"x"},"end":"</TOOL>"}]}"#,
    )
    .unwrap();
    let g = optimized(g);
    // Free text before the trigger, then the tagged body.
    assert!(accepts(Arc::clone(&g), "hello <TOOL>getx</TOOL>"));
    assert!(accepts(Arc::clone(&g), "<TOOL>getx</TOOL>"));
    assert!(accepts(g, "just free text"));
}

#[test]
fn triggered_tags_multiple_tags() {
    let g = grammar_from_format(
        r#"{"type":"triggered_tags","triggers":["A"],"tags":[{"type":"tag","begin":"A1","content":{"type":"const_string","value":""},"end":"A"},{"type":"tag","begin":"A2","content":{"type":"const_string","value":""},"end":"A"}]}"#,
    )
    .unwrap();
    let printed = print_grammar(&g);
    assert!(printed.contains("triggered_tags"));
    let g = optimized(g);
    assert!(accepts(Arc::clone(&g), "A1A"));
    assert!(accepts(g, "A2A"));
}

#[test]
fn triggered_tags_at_least_one_stop_after_first() {
    let g = grammar_from_format(
        r#"{"type":"triggered_tags","triggers":["A"],"at_least_one":true,"stop_after_first":true,"tags":[{"type":"tag","begin":"A1","content":{"type":"const_string","value":""},"end":"A"}]}"#,
    )
    .unwrap();
    let g = optimized(g);
    // Exactly one tag, no leading free text.
    assert!(accepts(Arc::clone(&g), "A1A"));
    assert!(!accepts(g, "free A1A"));
}

#[test]
fn triggered_tags_grammar_has_tag_dispatch() {
    let g = grammar_from_format(
        r#"{"type":"triggered_tags","triggers":["<TOOL>"],"tags":[{"type":"tag","begin":"<TOOL>","content":{"type":"json_schema","json_schema":{"type":"object"}},"end":"</TOOL>"}]}"#,
    )
    .unwrap();
    let printed = print_grammar(&g);
    assert!(printed.contains("TagDispatch("));
    assert!(printed.contains("loop_after_dispatch=true"));
}

/* -------------------- tags with separator ----------------------- */

#[test]
fn tags_with_separator_repeats() {
    let g = grammar_from_format(
        r#"{"type":"tags_with_separator","separator":",","tags":[{"type":"tag","begin":"<t>","content":{"type":"const_string","value":"x"},"end":"</t>"}]}"#,
    )
    .unwrap();
    let g = optimized(g);
    assert!(accepts(Arc::clone(&g), "<t>x</t>"));
    assert!(accepts(Arc::clone(&g), "<t>x</t>,<t>x</t>"));
    assert!(accepts(g, ""));
}

#[test]
fn tags_with_separator_at_least_one() {
    let g = grammar_from_format(
        r#"{"type":"tags_with_separator","separator":",","at_least_one":true,"tags":[{"type":"tag","begin":"<t>","content":{"type":"const_string","value":"x"},"end":"</t>"}]}"#,
    )
    .unwrap();
    let g = optimized(g);
    assert!(accepts(Arc::clone(&g), "<t>x</t>"));
    assert!(!accepts(g, ""));
}

/* --------------------- sequence / or ---------------------------- */

#[test]
fn sequence_format() {
    let g = grammar_from_format(
        r#"{"type":"sequence","elements":[{"type":"const_string","value":"a"},{"type":"const_string","value":"b"}]}"#,
    )
    .unwrap();
    let g = optimized(g);
    assert!(accepts(Arc::clone(&g), "ab"));
    assert!(!accepts(g, "ba"));
}

#[test]
fn or_format() {
    let g = grammar_from_format(
        r#"{"type":"or","elements":[{"type":"const_string","value":"yes"},{"type":"const_string","value":"no"}]}"#,
    )
    .unwrap();
    let g = optimized(g);
    assert!(accepts(Arc::clone(&g), "yes"));
    assert!(accepts(Arc::clone(&g), "no"));
    assert!(!accepts(g, "maybe"));
}

/* ----------------------- regex / grammar ------------------------ */

#[test]
fn regex_format() {
    let g = grammar_from_format(r#"{"type":"regex","pattern":"[0-9]+"}"#).unwrap();
    let g = optimized(g);
    assert!(accepts(Arc::clone(&g), "123"));
    assert!(!accepts(g, "abc"));
}

#[test]
fn grammar_format() {
    let g = grammar_from_format(r#"{"type":"grammar","grammar":"root ::= \"ok\"\n"}"#).unwrap();
    let g = optimized(g);
    assert!(accepts(Arc::clone(&g), "ok"));
    assert!(!accepts(g, "no"));
}

/* --------------------- legacy items API ------------------------- */

#[test]
fn legacy_items_tool_call_envelope() {
    let items = vec![StructuralTagItem::new(
        "<function=get_weather>",
        r#"{"type":"object","properties":{"city":{"type":"string"}}}"#,
        "</function>",
    )];
    let triggers = vec!["<function=".to_string()];
    let g = structural_tag_from_items(&items, &triggers).unwrap();
    let g = optimized(g);
    assert!(accepts(
        Arc::clone(&g),
        r#"<function=get_weather>{"city": "Paris"}</function>"#
    ));
    // Free text before the trigger is allowed.
    assert!(accepts(
        g,
        r#"Let me check. <function=get_weather>{"city": "Paris"}</function>"#
    ));
}

#[test]
fn legacy_items_multiple_tools() {
    let items = vec![
        StructuralTagItem::new("<fn=a>", r#"{"type":"object"}"#, "</fn>"),
        StructuralTagItem::new("<fn=b>", r#"{"type":"object"}"#, "</fn>"),
    ];
    let triggers = vec!["<fn=".to_string()];
    let g = structural_tag_from_items(&items, &triggers).unwrap();
    let g = optimized(g);
    assert!(accepts(Arc::clone(&g), "<fn=a>{}</fn>"));
    assert!(accepts(g, "<fn=b>{}</fn>"));
}

#[test]
fn structural_tag_item_constructs() {
    let item = StructuralTagItem::new("b", "{}", "e");
    assert_eq!(item.begin, "b");
    assert_eq!(item.schema, "{}");
    assert_eq!(item.end, "e");
}

/* ----------------------- malformed input ------------------------ */

#[test]
fn error_invalid_json() {
    let err = structural_tag_to_grammar("not json").unwrap_err();
    assert!(matches!(err, StructuralTagError::InvalidJson(_)));
}

#[test]
fn error_not_an_object() {
    let err = structural_tag_to_grammar("[1,2,3]").unwrap_err();
    assert!(matches!(err, StructuralTagError::InvalidStructuralTag(_)));
}

#[test]
fn error_missing_format() {
    let err = structural_tag_to_grammar(r#"{"type":"structural_tag"}"#).unwrap_err();
    assert!(matches!(err, StructuralTagError::InvalidStructuralTag(_)));
}

#[test]
fn error_wrong_type_field() {
    let err = structural_tag_to_grammar(
        r#"{"type":"wrong","format":{"type":"const_string","value":"x"}}"#,
    )
    .unwrap_err();
    assert!(matches!(err, StructuralTagError::InvalidStructuralTag(_)));
}

#[test]
fn error_unrecognized_format_type() {
    let err = grammar_from_format(r#"{"type":"bogus"}"#).unwrap_err();
    assert!(matches!(err, StructuralTagError::InvalidStructuralTag(_)));
}

#[test]
fn error_tag_missing_end() {
    let err = grammar_from_format(
        r#"{"type":"tag","begin":"<a>","content":{"type":"const_string","value":"x"}}"#,
    )
    .unwrap_err();
    assert!(matches!(err, StructuralTagError::InvalidStructuralTag(_)));
}

#[test]
fn error_tag_empty_end_array() {
    let err = grammar_from_format(
        r#"{"type":"tag","begin":"<a>","content":{"type":"const_string","value":"x"},"end":[]}"#,
    )
    .unwrap_err();
    assert!(matches!(err, StructuralTagError::InvalidStructuralTag(_)));
}

#[test]
fn error_const_string_missing_value() {
    let err = grammar_from_format(r#"{"type":"const_string"}"#).unwrap_err();
    assert!(matches!(err, StructuralTagError::InvalidStructuralTag(_)));
}

#[test]
fn error_sequence_empty_elements() {
    let err = grammar_from_format(r#"{"type":"sequence","elements":[]}"#).unwrap_err();
    assert!(matches!(err, StructuralTagError::InvalidStructuralTag(_)));
}

#[test]
fn error_triggered_tags_no_triggers() {
    let err = grammar_from_format(
        r#"{"type":"triggered_tags","triggers":[],"tags":[{"type":"tag","begin":"x","content":{"type":"const_string","value":""},"end":"y"}]}"#,
    )
    .unwrap_err();
    assert!(matches!(err, StructuralTagError::InvalidStructuralTag(_)));
}

#[test]
fn error_triggered_tags_tag_matches_no_trigger() {
    let err = grammar_from_format(
        r#"{"type":"triggered_tags","triggers":["Z"],"tags":[{"type":"tag","begin":"A1","content":{"type":"const_string","value":""},"end":"A"}]}"#,
    )
    .unwrap_err();
    assert!(matches!(err, StructuralTagError::InvalidStructuralTag(_)));
}

#[test]
fn error_legacy_item_bad_schema_json() {
    let items = vec![StructuralTagItem::new("<a>", "not json", "</a>")];
    let triggers = vec!["<a>".to_string()];
    let err = structural_tag_from_items(&items, &triggers).unwrap_err();
    assert!(matches!(err, StructuralTagError::InvalidJson(_)));
}
