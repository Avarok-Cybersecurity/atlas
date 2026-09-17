// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use xgrammar::{CompiledGrammar, GrammarMatcher};

fn grammar_accepts(compiled: &CompiledGrammar, input: &str) -> bool {
    let mut matcher =
        GrammarMatcher::new(compiled, None, true, -1).expect("GrammarMatcher::new failed");
    matcher.accept_string(input, false) && matcher.is_terminated()
}

#[test]
fn poolside_grammar_accepts_native_call() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .expect("compile must succeed");
    let call = "Working on it.\n<tool_call>get_weather<arg_key>location</arg_key>\
                <arg_value>Boston</arg_value></tool_call>";

    assert!(grammar_accepts(&compiled, call));
}

#[test]
fn poolside_grammar_rejects_malformed_argument_close() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .expect("compile must succeed");
    let malformed = "<tool_call>get_weather<arg_key>location</arg_key\
                     <arg_value>Boston</arg_value></tool_call>";

    assert!(!grammar_accepts(&compiled, malformed));
}

#[test]
fn poolside_grammar_rejects_unknown_parameter_key() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .expect("compile must succeed");
    let malformed = "<tool_call>get_weather<arg_key>city</arg_key>\
                     <arg_value>Boston</arg_value></tool_call>";

    assert!(!grammar_accepts(&compiled, malformed));
}

#[test]
fn poolside_grammar_rejects_unknown_tool() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .expect("compile must succeed");
    let unknown = "<tool_call>lookup<arg_key>location</arg_key>\
                   <arg_value>Boston</arg_value></tool_call>";

    assert!(!grammar_accepts(&compiled, unknown));
}

#[test]
fn poolside_grammar_accepts_markup_inside_argument_value() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .expect("compile must succeed");
    let call = "<tool_call>get_weather<arg_key>location</arg_key>\
                <arg_value><div>Boston</div></arg_value></tool_call>";

    assert!(grammar_accepts(&compiled, call));
}

#[test]
fn poolside_grammar_rejects_missing_required_argument() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .expect("compile must succeed");

    assert!(!grammar_accepts(
        &compiled,
        "<tool_call>get_weather</tool_call>"
    ));
}

#[test]
fn poolside_grammar_rejects_empty_tool_list() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let result = engine.compile_poolside_v1_tool_grammar(&[], true, "</arg_value>");

    assert!(matches!(result, Err(GrammarError::NoTools)));
}

#[test]
fn poolside_grammar_accepts_complete_zero_argument_call() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "get_status".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {}
            })),
        },
    }];
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&tools, true, "</arg_value>")
        .expect("compile must succeed");

    assert!(grammar_accepts(
        &compiled,
        "<tool_call>get_status</tool_call>"
    ));
}

#[test]
fn poolside_parser_reports_grammar_support() {
    assert!(crate::tool_parser::ToolCallFormat::PoolsideV1.has_grammar());
}

/// Tool defs mixing a required string, an optional string, and a required
/// non-string, so the min-length guard can be shown to touch only the first.
fn mixed_tool_defs() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "book".to_string(),
            description: Some("Book a slot".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "note": {"type": "string"},
                    "seats": {"type": "number"},
                },
                "required": ["title", "seats"]
            })),
        },
    }]
}

/// tool-eval-bench TC-43: `web_search` whose only parameter is a required
/// `query` string. The model emitted `{"query": ""}` and the scenario hard-failed.
fn web_search_tool_defs() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "web_search".to_string(),
            description: Some("Search the web".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            })),
        },
    }]
}

fn compile(tools: &[ToolDefinition]) -> CompiledGrammar {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    engine
        .compile_poolside_v1_tool_grammar(tools, true, "</arg_value>")
        .expect("compile must succeed")
}

#[test]
fn poolside_grammar_rejects_empty_required_string() {
    let compiled = compile(&test_tool_defs());
    let empty = "<tool_call>get_weather<arg_key>location</arg_key>\
                 <arg_value></arg_value></tool_call>";

    assert!(
        !grammar_accepts(&compiled, empty),
        "an empty required string must be un-generatable"
    );
}

#[test]
fn poolside_grammar_accepts_non_empty_required_string() {
    let compiled = compile(&test_tool_defs());
    let ok = "<tool_call>get_weather<arg_key>location</arg_key>\
              <arg_value>Boston</arg_value></tool_call>";

    assert!(grammar_accepts(&compiled, ok));
}

#[test]
fn poolside_grammar_still_allows_empty_optional_string() {
    let compiled = compile(&mixed_tool_defs());
    let optional_empty = "<tool_call>book<arg_key>note</arg_key>\
                          <arg_value></arg_value></tool_call>";

    assert!(
        grammar_accepts(&compiled, optional_empty),
        "the guard must not touch optional strings"
    );
}

#[test]
fn poolside_grammar_leaves_required_non_string_alone() {
    let compiled = compile(&mixed_tool_defs());
    let ok = "<tool_call>book<arg_key>seats</arg_key>\
              <arg_value>4</arg_value></tool_call>";
    assert!(grammar_accepts(&compiled, ok));

    let required_string_empty = "<tool_call>book<arg_key>title</arg_key>\
                                 <arg_value></arg_value></tool_call>";
    assert!(
        !grammar_accepts(&compiled, required_string_empty),
        "the required STRING in the same schema must still be guarded"
    );
}

#[test]
fn poolside_grammar_cannot_emit_the_tc43_empty_query() {
    let compiled = compile(&web_search_tool_defs());
    let tc43 = "<tool_call>web_search<arg_key>query</arg_key>\
                <arg_value></arg_value></tool_call>";

    assert!(
        !grammar_accepts(&compiled, tc43),
        "tool-eval-bench TC-43 must be structurally impossible"
    );
}

#[test]
fn poolside_grammar_accepts_a_real_web_search_query() {
    let compiled = compile(&web_search_tool_defs());
    let ok = "<tool_call>web_search<arg_key>query</arg_key>\
              <arg_value>today's top news headlines</arg_value></tool_call>";

    assert!(grammar_accepts(&compiled, ok));
}

#[test]
fn poolside_grammar_matches_qwen3_coder_on_empty_required_strings() {
    // Behavioural parity, not implementation parity. qwen3_coder rejects an
    // empty required value in its own value EBNF
    // (`qwen3_coder_grammar_rejects_empty_parameter_body`); poolside must reach
    // the same verdict on the same schema, through its own envelope.
    let tools = web_search_tool_defs();

    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let qwen = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("qwen3_coder compile must succeed");
    let qwen_empty = "<tool_call>\n<function=web_search>\n\
                      <parameter=query></parameter>\n</function>\n</tool_call>";
    assert!(!grammar_accepts(&qwen, qwen_empty));

    let poolside = compile(&tools);
    let poolside_empty = "<tool_call>web_search<arg_key>query</arg_key>\
                          <arg_value></arg_value></tool_call>";
    assert!(
        !grammar_accepts(&poolside, poolside_empty),
        "poolside must not be the one parser that lets an empty required string through"
    );
}

// ── A117 (2026-09-17): required strings must be non-BLANK, not just ────────
// non-empty. tool-eval-bench TC-43 on GLM-5.3-Flash evaded the A85 guard
// with `web_search {"query":" "}` (a single space) — the empty-string guard
// alone does not stop whitespace-only values. See `poolside_v1`'s
// `req_value` rule.

/// A single space is exactly the TC-43 evasion: the value is non-empty, so
/// the pre-A117 guard let it through.
#[test]
fn poolside_grammar_rejects_single_space_required_string() {
    let compiled = compile(&web_search_tool_defs());
    let single_space = "<tool_call>web_search<arg_key>query</arg_key>\
                        <arg_value> </arg_value></tool_call>";

    assert!(
        !grammar_accepts(&compiled, single_space),
        "a single-space required string must be un-generatable"
    );
}

/// Whitespace-only isn't just ASCII space: tab + newline with no other
/// content must also be rejected.
#[test]
fn poolside_grammar_rejects_tabs_and_newline_required_string() {
    let compiled = compile(&web_search_tool_defs());
    let blank = "<tool_call>web_search<arg_key>query</arg_key>\
                <arg_value>\t\n</arg_value></tool_call>";

    assert!(
        !grammar_accepts(&compiled, blank),
        "a tab/newline-only required string must be un-generatable"
    );
}

/// The empty string must still be rejected post-A117 (behaviour carried
/// over unchanged from A85/PR #1103, not just re-derived from the new rule).
#[test]
fn poolside_grammar_still_rejects_empty_required_string() {
    let compiled = compile(&web_search_tool_defs());
    let empty = "<tool_call>web_search<arg_key>query</arg_key>\
                <arg_value></arg_value></tool_call>";

    assert!(
        !grammar_accepts(&compiled, empty),
        "an empty required string must remain un-generatable"
    );
}

/// A single non-whitespace character is enough — the guard requires ONE
/// non-blank byte, not any minimum length beyond that.
#[test]
fn poolside_grammar_accepts_single_char_required_string() {
    let compiled = compile(&web_search_tool_defs());
    let one_char = "<tool_call>web_search<arg_key>query</arg_key>\
                    <arg_value>a</arg_value></tool_call>";

    assert!(grammar_accepts(&compiled, one_char));
}

/// The FIRST character after `<arg_value>` may still be whitespace — the
/// guard requires a non-blank byte SOMEWHERE in the value, not that the
/// value starts with one.
#[test]
fn poolside_grammar_accepts_leading_space_before_required_content() {
    let compiled = compile(&web_search_tool_defs());
    let leading_space = "<tool_call>web_search<arg_key>query</arg_key>\
                         <arg_value> a</arg_value></tool_call>";

    assert!(
        grammar_accepts(&compiled, leading_space),
        "leading whitespace before real content must remain ACCEPTED"
    );
}

/// Non-string required parameters are untouched by the A117 guard: `seats`
/// (required, type `number`) is unconstrained content, exactly as before.
#[test]
fn poolside_grammar_required_non_string_still_unconstrained() {
    let compiled = compile(&mixed_tool_defs());
    let ok = "<tool_call>book<arg_key>seats</arg_key>\
             <arg_value>4</arg_value></tool_call>";

    assert!(grammar_accepts(&compiled, ok));
}

/// Optional strings are explicitly OUT OF SCOPE for A117 (the PR #1103
/// commit message scoped the empty-string guard to required strings only,
/// and this ticket doesn't widen that). A whitespace-only optional value
/// is accepted today and must remain accepted — `optpair`'s `value ::=
/// value_part*` rule is untouched by this change.
#[test]
fn poolside_grammar_optional_string_whitespace_only_unchanged() {
    let compiled = compile(&mixed_tool_defs());
    let optional_space = "<tool_call>book<arg_key>note</arg_key>\
                          <arg_value> </arg_value></tool_call>";

    assert!(
        grammar_accepts(&compiled, optional_space),
        "A117 must not touch optional strings: whitespace-only optional \
         values are accepted both before and after this change"
    );
}
