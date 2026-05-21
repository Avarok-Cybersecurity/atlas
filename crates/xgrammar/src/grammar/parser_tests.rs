// SPDX-License-Identifier: AGPL-3.0-only
//
// Tests for the EBNF parser, ported from xgrammar
// `tests/python/test_grammar_parser.py`. xgrammar asserts on the
// printed grammar; the EBNF printer is a later port wave (W3), so
// these assert on the parsed `GrammarData` structure instead.

use super::{parse_ebnf, parse_ebnf_default};
use crate::grammar::data::GrammarData;
use crate::grammar::expr::GrammarExprType;

#[path = "parser_error_tests.rs"]
mod errors;
#[path = "parser_grammar_tests.rs"]
mod larger;

fn parse(src: &str) -> GrammarData {
    parse_ebnf_default(src).expect("expected successful parse")
}

/// Walk the rule body of `root` and collect every reachable expr kind.
fn kinds(g: &GrammarData) -> Vec<GrammarExprType> {
    let mut seen = Vec::new();
    for id in 0..g.num_exprs() {
        seen.push(g.expr(id).kind);
    }
    seen
}

#[test]
fn basic_string_literal() {
    let g = parse("root ::= \"hello\"\n");
    assert_eq!(g.num_rules(), 1);
    assert_eq!(g.root_rule().name, "root");
    assert!(kinds(&g).contains(&GrammarExprType::ByteString));
}

#[test]
fn empty_string() {
    let g = parse("root ::= \"\"\n");
    assert!(kinds(&g).contains(&GrammarExprType::EmptyStr));
}

#[test]
fn character_class() {
    let g = parse("root ::= [a-z]\n");
    let cc = (0..g.num_exprs())
        .map(|i| g.expr(i))
        .find(|e| e.kind == GrammarExprType::CharacterClass)
        .unwrap();
    // [is_negative=0, lower='a', upper='z']
    assert_eq!(cc.data, &[0, 'a' as i32, 'z' as i32]);
}

#[test]
fn negated_character_class() {
    let g = parse("root ::= [^a-z]\n");
    let cc = (0..g.num_exprs())
        .map(|i| g.expr(i))
        .find(|e| e.kind == GrammarExprType::CharacterClass)
        .unwrap();
    assert_eq!(cc.data[0], 1); // is_negative
}

#[test]
fn complex_character_class_ranges_and_singles() {
    // [a-zA-Z0-9_-]: ranges + lone underscore + trailing dash.
    let g = parse("root ::= [a-zA-Z0-9_-]\n");
    let cc = (0..g.num_exprs())
        .map(|i| g.expr(i))
        .find(|e| e.kind == GrammarExprType::CharacterClass)
        .unwrap();
    // is_negative + (a-z, A-Z, 0-9, _-_, ---).
    assert_eq!(cc.data.len(), 1 + 5 * 2);
    assert_eq!(cc.data[1], 'a' as i32);
    assert_eq!(cc.data[2], 'z' as i32);
    // lone '_' and trailing '-' each become a [c, c] range.
    assert_eq!(cc.data[7], '_' as i32);
    assert_eq!(cc.data[8], '_' as i32);
    assert_eq!(cc.data[9], '-' as i32);
    assert_eq!(cc.data[10], '-' as i32);
}

#[test]
fn sequence() {
    let g = parse("root ::= \"a\" \"b\" \"c\"\n");
    let seq = (0..g.num_exprs())
        .map(|i| g.expr(i))
        .find(|e| e.kind == GrammarExprType::Sequence)
        .unwrap();
    assert_eq!(seq.len(), 3);
}

#[test]
fn choice() {
    let g = parse("root ::= \"a\" | \"b\" | \"c\"\n");
    let ch = (0..g.num_exprs())
        .map(|i| g.expr(i))
        .find(|e| e.kind == GrammarExprType::Choices && e.len() == 3)
        .unwrap();
    assert_eq!(ch.len(), 3);
}

#[test]
fn grouping() {
    let g = parse("root ::= (\"a\" \"b\") | (\"c\" \"d\")\n");
    assert_eq!(g.num_rules(), 1);
    assert!(kinds(&g).contains(&GrammarExprType::Choices));
}

#[test]
fn star_quantifier_creates_rule() {
    let g = parse("root ::= \"a\"*\n");
    // star on non-char-class creates a recursive helper rule.
    assert_eq!(g.num_rules(), 2);
    assert!(kinds(&g).contains(&GrammarExprType::RuleRef));
}

#[test]
fn plus_quantifier_creates_rule() {
    let g = parse("root ::= \"a\"+\n");
    assert_eq!(g.num_rules(), 2);
}

#[test]
fn question_quantifier_creates_rule() {
    let g = parse("root ::= \"a\"?\n");
    assert_eq!(g.num_rules(), 2);
}

#[test]
fn character_class_star() {
    let g = parse("root ::= [a-z]*\n");
    // char-class star stays a single CharacterClassStar expr, no new rule.
    assert_eq!(g.num_rules(), 1);
    assert!(kinds(&g).contains(&GrammarExprType::CharacterClassStar));
}

#[test]
fn repetition_range_exact() {
    let g = parse("root ::= \"a\"{3}\n");
    let seq = (0..g.num_exprs())
        .map(|i| g.expr(i))
        .find(|e| e.kind == GrammarExprType::Sequence && e.len() == 3)
        .unwrap();
    assert_eq!(seq.len(), 3);
}

#[test]
fn repetition_range_zero() {
    // {0} produces an empty sequence.
    let g = parse("root ::= \"a\"{0}\n");
    assert!(
        (0..g.num_exprs())
            .map(|i| g.expr(i))
            .any(|e| e.kind == GrammarExprType::Sequence && e.is_empty())
    );
}

#[test]
fn repetition_range_min_max() {
    let g = parse("root ::= \"a\"{2,4}\n");
    // helper rules created for the (max-min) optional tail.
    assert!(g.num_rules() >= 3);
}

#[test]
fn repetition_range_min_only() {
    let g = parse("root ::= \"a\"{2,}\n");
    assert!(g.num_rules() >= 2);
}

#[test]
fn lookahead_assertion_simple() {
    let g = parse("root ::= \"a\" (=\"b\")\n");
    assert_ne!(g.root_rule().lookahead_assertion_id, -1);
}

#[test]
fn complex_lookahead() {
    let g = parse("root ::= \"a\" (=\"b\" \"c\" [0-9])\n");
    assert_ne!(g.root_rule().lookahead_assertion_id, -1);
}

#[test]
fn escape_sequences() {
    // `\n \t \r \" \\` — five single-byte escapes in one literal.
    let g = parse("root ::= \"\\n\\t\\r\\\"\\\\\"\n");
    let bs = (0..g.num_exprs())
        .map(|i| g.expr(i))
        .find(|e| e.kind == GrammarExprType::ByteString)
        .unwrap();
    assert_eq!(bs.len(), 5);
    assert_eq!(bs.data[0], b'\n' as i32);
    assert_eq!(bs.data[1], b'\t' as i32);
}

#[test]
fn unicode_escape() {
    let g = parse("root ::= \"\\u0041\\u0042\"\n");
    // AB == "AB", 2 ASCII bytes.
    assert_eq!(g.byte_string(0), "AB");
}

#[test]
fn multi_rule_grammar() {
    let g = parse("root ::= a b\na ::= \"a\"\nb ::= \"b\"\n");
    assert_eq!(g.num_rules(), 3);
    assert_eq!(g.root_rule().name, "root");
}

#[test]
fn bnf_comment() {
    let src = "# top comment\nroot ::= a b # inline comment\na ::= \"a\"\nb ::= \"b\"\n# bottom\n";
    let g = parse(src);
    assert_eq!(g.num_rules(), 3);
}

#[test]
fn whitespace_tolerant() {
    let src = "\n\nroot::=\"a\"  \"b\" (\"c\"\"d\"\n\"e\") |\n\n\"f\" | \"g\"\n";
    let g = parse(src);
    assert_eq!(g.num_rules(), 1);
}

#[test]
fn empty_parentheses() {
    let g = parse("root ::= \"a\" ( ) \"b\"\n");
    assert!(kinds(&g).contains(&GrammarExprType::EmptyStr));
}

#[test]
fn forward_rule_reference() {
    // `b` is referenced before it is defined.
    let g = parse("root ::= b\nb ::= \"b\"\n");
    assert_eq!(g.num_rules(), 2);
}

#[test]
fn custom_root_rule_name() {
    let g = parse_ebnf("start ::= \"a\"\n", "start").unwrap();
    assert_eq!(g.root_rule().name, "start");
}

#[test]
fn nested_quantifiers() {
    let g = parse("root ::= (\"a\"*)+\n");
    assert!(g.num_rules() >= 3);
}

// See the `larger` and `errors` submodules (declared above) for
// end-to-end grammar, tag-dispatch and error-case tests.
