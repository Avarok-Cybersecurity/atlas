// SPDX-License-Identifier: AGPL-3.0-only
//
// EBNF parser error-case tests. Ported from the error-case tables in
// xgrammar `tests/python/test_grammar_parser.py`
// (`test_lexer_parser_errors`, `test_end_to_end_errors`,
// `test_error_consecutive_quantifiers`). Split out of `parser_tests.rs`
// to keep each file under the 250-line cap.

use crate::grammar::parser::{ParseError, parse_ebnf_default};

fn err(src: &str) -> ParseError {
    parse_ebnf_default(src).expect_err("expected a parse error")
}

#[test]
fn err_unterminated_string() {
    let e = err("root ::= \"a\" \"");
    assert!(matches!(e, ParseError::Lex(_)));
}

#[test]
fn err_newline_in_char_class() {
    let e = err("root ::= [a\n]");
    assert!(matches!(e, ParseError::Lex(_)));
}

#[test]
fn err_invalid_escape() {
    let e = err("root ::= \"\\@\"");
    assert!(matches!(e, ParseError::Lex(_)));
}

#[test]
fn err_invalid_unicode_escape() {
    // `\uFF` — too few hex digits for a `\u` escape.
    let e = err("root ::= \"\\uFF\"");
    assert!(matches!(e, ParseError::Lex(_)));
}

#[test]
fn err_assign_first_token() {
    let e = err("::= \"a\"");
    match e {
        ParseError::Lex(le) => assert!(le.msg.contains("first token")),
        _ => panic!("expected lex error"),
    }
}

#[test]
fn err_undefined_rule() {
    let e = err("root ::= a b");
    match e {
        ParseError::Parse { msg, .. } => assert!(msg.contains("is not defined")),
        _ => panic!("expected parse error"),
    }
}

#[test]
fn err_dangling_pipe() {
    let e = err("root ::= \"a\" |");
    match e {
        ParseError::Parse { msg, .. } => assert!(msg.contains("Expect element")),
        _ => panic!("expected parse error"),
    }
}

#[test]
fn err_inverted_char_range() {
    let e = err("root ::= [Z-A]");
    match e {
        ParseError::Parse { msg, .. } => assert!(msg.contains("lower bound is larger")),
        _ => panic!("expected parse error"),
    }
}

#[test]
fn err_duplicate_rule() {
    let e = err("root ::= \"a\"\nroot ::= \"b\"\n");
    match e {
        ParseError::Parse { msg, .. } => assert!(msg.contains("defined multiple times")),
        _ => panic!("expected parse error"),
    }
}

#[test]
fn err_missing_root_rule() {
    let e = err("a ::= \"a\"\n");
    match e {
        ParseError::Parse { msg, .. } => assert!(msg.contains("root rule")),
        _ => panic!("expected parse error"),
    }
}

#[test]
fn err_consecutive_quantifiers() {
    let e = err("root ::= \"a\"{1,3}{1,3}\n");
    match e {
        ParseError::Parse { msg, .. } => assert!(msg.contains("Expect element")),
        _ => panic!("expected parse error"),
    }
    assert!(matches!(
        err("root ::= \"a\"++\n"),
        ParseError::Parse { .. }
    ));
    assert!(matches!(
        err("root ::= \"a\"??\n"),
        ParseError::Parse { .. }
    ));
}

#[test]
fn err_lookahead_then_lookahead() {
    // `(="a") (="b")` — second lookahead has nothing to attach to.
    let e = err("root ::= \"a\" (=\"a\") (=\"b\")");
    assert!(matches!(e, ParseError::Parse { .. }));
}

#[test]
fn err_negative_repetition_lower() {
    let e = err("root ::= \"a\"{-1}\n");
    assert!(matches!(e, ParseError::Parse { .. }));
}

#[test]
fn err_repetition_lower_gt_upper() {
    let e = err("root ::= \"a\"{4,2}\n");
    match e {
        ParseError::Parse { msg, .. } => assert!(msg.contains("larger than upper")),
        _ => panic!("expected parse error"),
    }
}

#[test]
fn err_char_class_special_escape_unsupported() {
    // `[\d]` — regex special escapes are lexed but rejected by the
    // EBNF parser ("not supported yet in EBNF").
    let e = err("root ::= [\\d]\n");
    match e {
        ParseError::Parse { msg, .. } => assert!(msg.contains("not supported")),
        _ => panic!("expected parse error"),
    }
}
