// SPDX-License-Identifier: AGPL-3.0-only
//
// Regression tests for the speculative fast-accept LCP poisoning bug
// (corrupted token bitmask inside JSON-string values, 2026-07).
//
// `token_mask_with_first_char_check` fast-accepts tokens made only of
// "safe" self-loop bytes without running them through the parser. The
// port erroneously updated `prev_token` on that path, so the next
// parser-walked token sharing a prefix with a fast-accepted neighbor
// hit `lcp > prev_matched` in `scan_one_token` and was wrongly
// classified rejected (plus its whole trie subtree). Observable as
// fused tokens like ` "` (whitespace -> string-open) being masked
// illegal at whitespace states of JSON-schema grammars while
// `accept_token` accepts them.
//
// These tests assert full mask == parser agreement over an entire
// vocabulary that contains safe/unsafe fused-token adjacencies, at the
// states where the poisoning occurred (no lookahead assertions in the
// grammars, so exact equivalence must hold).

use crate::compiler::{CompiledGrammar, GrammarCompiler};
use crate::matcher::GrammarMatcher;
use crate::tokenizer::{TokenizerInfo, VocabType};

use super::super::bitmask::bitmask_size;

/// RAW vocab with fused tokens that straddle grammar-element
/// boundaries: whitespace+quote, letters+quote, brace+quote, escapes.
fn fused_tok() -> TokenizerInfo {
    let vocab: Vec<String> = [
        "</s>", "{", "}", " ", "\t", "\"", "a", "b", ",", ":", " \"", " a", "ab", "ab\"", "a\"",
        "\"a", "{\"", "\\", "\\\"", "  ", " {", "l", "o", "c", "loc",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    TokenizerInfo::new(&vocab, VocabType::Raw, None, Some(vec![0]), false)
}

fn compile(ebnf: &str) -> CompiledGrammar {
    GrammarCompiler::new(fused_tok(), 1, true, -1)
        .compile_grammar_from_ebnf(ebnf, "root")
        .expect("grammar compiles")
}

/// Assert that, at the state reached by `prefix`, the filled bitmask
/// agrees with `accept_token` for EVERY token id in the vocabulary.
fn assert_mask_matches_parser(compiled: &CompiledGrammar, prefix: &str) {
    let info = compiled.tokenizer_info();
    let vocab_size = info.vocab_size();
    let mut m = GrammarMatcher::from_compiled_grammar(compiled.clone());
    assert!(m.accept_string(prefix, false), "prefix {prefix:?} accepted");
    let mut mask = vec![0i32; bitmask_size(vocab_size)];
    m.fill_next_token_bitmask(&mut mask, 0, false)
        .expect("fill succeeds");

    for id in 0..vocab_size as i32 {
        let mask_legal = (mask[id as usize / 32] >> (id % 32)) & 1 == 1;
        let mut probe = GrammarMatcher::from_compiled_grammar(compiled.clone());
        assert!(probe.accept_string(prefix, false));
        let parser_accepts = probe.accept_token(id, false);
        assert_eq!(
            mask_legal,
            parser_accepts,
            "prefix {prefix:?} token {id} ({:?}): mask says {mask_legal}, parser says {parser_accepts}",
            String::from_utf8_lossy(&info.decoded_vocab()[id as usize]),
        );
    }
}

/// Whitespace-star node followed by a rule-ref (the `[ \n\t]*` before
/// `basic_string` in JSON-schema grammars). ` ` is fast-accepted; the
/// bug then rejected ` "` (and its subtree) via the poisoned LCP.
#[test]
fn ws_star_rule_ref_crossing_mask_matches_parser() {
    let cg = compile("root ::= \"{\" [ \\n\\t]* str\nstr ::= \"\\\"\" [a-z]* \"\\\"\"\n");
    for prefix in ["{", "{ ", "{ \t"] {
        assert_mask_matches_parser(&cg, prefix);
    }
}

/// Self-recursive string-content rule (Case B speculative: first edge
/// leads to a state that recursively calls the rule — the shape of
/// `basic_string_sub`). `ab` is fast-accepted; the bug then rejected
/// `ab"` (content + close quote) via the poisoned LCP.
#[test]
fn string_content_recursion_mask_matches_parser() {
    let cg = compile("root ::= \"\\\"\" sub\nsub ::= \"\\\"\" | [^\\\"\\\\\\r\\n] sub\n");
    for prefix in ["\"", "\"a", "\"ab"] {
        assert_mask_matches_parser(&cg, prefix);
    }
}

/// The exact fused-token symptom: after `{` (whitespace may follow),
/// the fused ` "` token must be mask-legal — it crosses the ws loop
/// into the string sub-rule, exactly like ` "` after `"location":` in
/// the tool-call schema.
#[test]
fn fused_ws_quote_token_is_legal_at_ws_state() {
    let cg = compile("root ::= \"{\" [ \\n\\t]* str\nstr ::= \"\\\"\" [a-z]* \"\\\"\"\n");
    let info = cg.tokenizer_info();
    let id = info
        .decoded_vocab()
        .iter()
        .position(|t| t.as_slice() == b" \"")
        .expect("` \"` in vocab") as i32;
    let mut m = GrammarMatcher::from_compiled_grammar(cg.clone());
    assert!(m.accept_string("{", false));
    let mut mask = vec![0i32; bitmask_size(info.vocab_size())];
    m.fill_next_token_bitmask(&mut mask, 0, false).unwrap();
    assert!(
        (mask[id as usize / 32] >> (id % 32)) & 1 == 1,
        "fused ` \"` token masked illegal at the whitespace state",
    );
    assert!(m.accept_token(id, false));
}

/// minLength must be enforced at the matcher level: with
/// `minLength: 1` the closing quote is illegal for an empty string.
#[test]
fn min_length_blocks_empty_string() {
    let schema = r#"{"type":"object","properties":{"loc":{"type":"string","minLength":1}},"required":["loc"]}"#;
    let cg = GrammarCompiler::new(fused_tok(), 1, true, -1)
        .compile_json_schema(schema, true, None, None, true, None)
        .expect("schema compiles");
    let info = cg.tokenizer_info();
    let quote = info
        .decoded_vocab()
        .iter()
        .position(|t| t.as_slice() == b"\"")
        .unwrap() as i32;

    let mut m = GrammarMatcher::from_compiled_grammar(cg.clone());
    assert!(m.accept_string("{\"loc\":\"", false));
    let mut mask = vec![0i32; bitmask_size(info.vocab_size())];
    m.fill_next_token_bitmask(&mut mask, 0, false).unwrap();
    assert!(
        (mask[quote as usize / 32] >> (quote % 32)) & 1 == 0,
        "closing quote mask-legal for empty string despite minLength 1",
    );
    assert!(
        !m.accept_token(quote, false),
        "closing quote accepted for empty string despite minLength 1",
    );
    // One content char satisfies minLength; the close becomes legal.
    assert!(m.accept_string("a", false));
    assert!(m.accept_token(quote, false));
}

/// minLength must also survive the structural-tag embedding (the
/// serving path compiles the tool schema inside a triggered-tags
/// structural tag, not via `compile_json_schema` directly).
#[test]
fn min_length_enforced_through_structural_tag() {
    let st = r#"{"type":"structural_tag","format":{"type":"triggered_tags","triggers":["<t>"],"tags":[{"type":"tag","begin":"<t>{\"arguments\": ","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"loc":{"type":"string","minLength":1}},"required":["loc"]}},"end":"}</t>"}],"at_least_one":false,"stop_after_first":false}}"#;
    let cg = GrammarCompiler::new(fused_tok(), 1, true, -1)
        .compile_structural_tag(st)
        .expect("structural tag compiles");
    let info = cg.tokenizer_info();
    let quote = info
        .decoded_vocab()
        .iter()
        .position(|t| t.as_slice() == b"\"")
        .unwrap() as i32;
    let mut m = GrammarMatcher::from_compiled_grammar(cg.clone());
    assert!(m.accept_string("<t>{\"arguments\": {\"loc\":\"", false));
    assert!(
        !m.accept_token(quote, false),
        "empty-string close accepted despite minLength 1 (structural-tag path)",
    );
    assert!(m.accept_string("a", false));
    assert!(m.accept_token(quote, false));
}
