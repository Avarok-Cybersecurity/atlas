// SPDX-License-Identifier: AGPL-3.0-only
//
// Tests for the EBNF lexer. Split out of `lexer.rs` to keep each file
// under the 250-line cap.

use super::{EbnfLexer, TokenType, TokenValue};

fn types(src: &str) -> Vec<TokenType> {
    EbnfLexer::tokenize(src)
        .unwrap()
        .into_iter()
        .map(|t| t.ty)
        .collect()
}

#[test]
fn simple_rule_tokens() {
    assert_eq!(
        types("root ::= \"a\"\n"),
        vec![
            TokenType::RuleName,
            TokenType::Assign,
            TokenType::StringLiteral,
            TokenType::EndOfFile,
        ]
    );
}

#[test]
fn char_class_tokens() {
    assert_eq!(
        types("root ::= [a-z]\n"),
        vec![
            TokenType::RuleName,
            TokenType::Assign,
            TokenType::LBracket,
            TokenType::CharInCharClass,
            TokenType::Dash,
            TokenType::CharInCharClass,
            TokenType::RBracket,
            TokenType::EndOfFile,
        ]
    );
}

#[test]
fn negated_char_class_tokens() {
    let toks = types("root ::= [^a]\n");
    assert!(toks.contains(&TokenType::Caret));
}

#[test]
fn comments_skipped() {
    let toks = EbnfLexer::tokenize("# hello\nroot ::= \"a\" # trailing\n").unwrap();
    assert_eq!(toks[0].ty, TokenType::RuleName);
}

#[test]
fn unterminated_string_errors() {
    let e = EbnfLexer::tokenize("root ::= \"a").unwrap_err();
    assert!(e.msg.contains("Expect \""));
    assert_eq!(e.line, 1);
}

#[test]
fn newline_in_char_class_errors() {
    let e = EbnfLexer::tokenize("root ::= [a\n]").unwrap_err();
    assert!(e.msg.contains("newline"));
}

#[test]
fn assign_first_token_errors() {
    let e = EbnfLexer::tokenize("::= \"a\"").unwrap_err();
    assert!(e.msg.contains("first token"));
}

#[test]
fn unterminated_char_class_errors() {
    let e = EbnfLexer::tokenize("root ::= [ab").unwrap_err();
    assert!(e.msg.contains("Unterminated"));
}

#[test]
fn quantifier_and_brace_tokens() {
    assert_eq!(
        types("root ::= \"a\"*\n"),
        vec![
            TokenType::RuleName,
            TokenType::Assign,
            TokenType::StringLiteral,
            TokenType::Star,
            TokenType::EndOfFile,
        ]
    );
}

#[test]
fn integer_literal() {
    let toks = EbnfLexer::tokenize("root ::= \"a\"{2,4}\n").unwrap();
    let ints: Vec<i64> = toks
        .iter()
        .filter_map(|t| match t.value {
            TokenValue::Int(v) => Some(v),
            _ => None,
        })
        .collect();
    assert_eq!(ints, vec![2, 4]);
}

#[test]
fn leading_dash_lexes_as_identifier() {
    // `-` is a name char in xgrammar's lexer, so `-1` after `{`
    // tokenizes as an identifier, not a signed integer. The parser
    // then rejects it where an integer is expected.
    let toks = EbnfLexer::tokenize("root ::= \"a\"{-1}\n").unwrap();
    assert!(
        toks.iter()
            .any(|t| t.ty == TokenType::Identifier && t.lexeme == "-1")
    );
}

#[test]
fn signed_integer_after_comma() {
    // `+5` after a comma: `+` is not a name char, so this lexes as a
    // signed integer literal.
    let toks =
        EbnfLexer::tokenize("root ::= TagDispatch((\"t\", h), x=+5)\nh ::= \"h\"\n").unwrap();
    assert!(toks.iter().any(|t| matches!(t.value, TokenValue::Int(5))));
}

#[test]
fn lookahead_lparen() {
    let toks = EbnfLexer::tokenize("root ::= \"a\" (=\"b\")\n").unwrap();
    assert!(toks.iter().any(|t| t.ty == TokenType::LookaheadLParen));
}

#[test]
fn boolean_literal() {
    let toks = EbnfLexer::tokenize("root ::= TagDispatch(stop_eos=true)\n").unwrap();
    assert!(
        toks.iter()
            .any(|t| t.ty == TokenType::BooleanLiteral && t.value == TokenValue::Bool(true))
    );
}

#[test]
fn string_value_decodes_escapes() {
    let toks = EbnfLexer::tokenize("root ::= \"\\n\"\n").unwrap();
    let s = toks
        .iter()
        .find(|t| t.ty == TokenType::StringLiteral)
        .unwrap();
    assert_eq!(s.value, TokenValue::Str("\n".to_string()));
}

#[test]
fn unicode_escape_in_string() {
    let toks = EbnfLexer::tokenize("root ::= \"\\u0041\"\n").unwrap();
    let s = toks
        .iter()
        .find(|t| t.ty == TokenType::StringLiteral)
        .unwrap();
    assert_eq!(s.value, TokenValue::Str("A".to_string()));
}

#[test]
fn unexpected_character_errors() {
    let e = EbnfLexer::tokenize("root ::= \"a\" @").unwrap_err();
    assert!(e.msg.contains("Unexpected character"));
}

#[test]
fn line_and_column_tracking() {
    let toks = EbnfLexer::tokenize("root ::= \"a\"\nb ::= \"b\"\n").unwrap();
    // The second rule name sits on line 2.
    let rule_names: Vec<i32> = toks
        .iter()
        .filter(|t| t.ty == TokenType::RuleName)
        .map(|t| t.line)
        .collect();
    assert_eq!(rule_names, vec![1, 2]);
}
