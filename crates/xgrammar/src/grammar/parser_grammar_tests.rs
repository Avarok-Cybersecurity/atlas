// SPDX-License-Identifier: AGPL-3.0-only
//
// EBNF parser tests for larger, end-to-end grammars and the
// `TagDispatch` macro. Split out of `parser_tests.rs` to keep each
// file under the 250-line cap. The `complex_grammar` and JSON-like
// fixtures correspond to xgrammar `test_grammar_parser.py`'s
// `test_complex_grammar` / `test_e2e_json_grammar`.

use super::parse;
use crate::grammar::expr::GrammarExprType;

#[test]
fn complex_grammar() {
    let src = "root ::= expr\n\
        expr ::= term (\"+\" term | \"-\" term)*\n\
        term ::= factor (\"*\" factor | \"/\" factor)*\n\
        factor ::= number | \"(\" expr \")\"\n\
        number ::= [0-9]+ (\".\" [0-9]+)?\n";
    let g = parse(src);
    // 5 named rules + helper rules from the quantifiers.
    assert!(g.num_rules() >= 5);
    assert_eq!(g.root_rule().name, "root");
}

#[test]
fn json_like_grammar() {
    // A compact JSON-ish EBNF exercising classes, escapes, lookahead,
    // recursion and quantifiers together.
    let src = "root ::= \"{\" [ \\n\\t]* members \"}\"\n\
        members ::= \"\\\"\" chars \"\\\"\" \":\" value\n\
        chars ::= [^\"\\\\] chars | \"\"\n\
        value ::= \"true\" | \"false\" | \"null\" | number\n\
        number ::= [0-9]+\n";
    let g = parse(src);
    assert!(g.num_rules() >= 5);
}

#[test]
fn tag_dispatch_basic() {
    let src = "root ::= TagDispatch((\"<call>\", handler))\nhandler ::= \"x\"\n";
    let g = parse(src);
    let td = (0..g.num_exprs())
        .map(|i| g.expr(i))
        .find(|e| e.kind == GrammarExprType::TagDispatch);
    assert!(td.is_some());
}

#[test]
fn tag_dispatch_named_args() {
    let src =
        "root ::= TagDispatch((\"<a>\", h), stop_eos=false, stop_str=(\"END\"))\nh ::= \"x\"\n";
    let g = parse(src);
    let td_id = (0..g.num_exprs())
        .find(|&i| g.expr(i).kind == GrammarExprType::TagDispatch)
        .unwrap();
    let decoded = g.tag_dispatch(td_id);
    assert!(!decoded.stop_eos);
    assert_eq!(decoded.stop_str, vec!["END".to_string()]);
}
