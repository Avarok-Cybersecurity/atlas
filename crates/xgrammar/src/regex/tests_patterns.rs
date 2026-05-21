// SPDX-License-Identifier: AGPL-3.0-only
//
// Regex-converter tests, part 2 — real-world patterns, empty
// regexes, malformed escapes and the `regex_to_grammar` AST path.
// Split out of `tests.rs` to keep each file under the 250-line cap.

use super::*;

/// Convenience: convert and unwrap the `root ::= …\n` form.
fn ebnf(regex: &str) -> String {
    regex_to_ebnf(regex, true).expect("regex should convert")
}

/// Convenience: the rule-body-only form.
fn body(regex: &str) -> String {
    regex_to_ebnf(regex, false).expect("regex should convert")
}
// ---- groups ----------------------------------------------------------

#[test]
fn groups() {
    assert_eq!(
        ebnf(r"(a|b)(c|d)"),
        "root ::= ( \"a\" | \"b\" ) ( \"c\" | \"d\" )\n"
    );
}

#[test]
fn empty_parentheses() {
    assert_eq!(ebnf("()"), "root ::= ( )\n");
    assert_eq!(ebnf("a()b"), "root ::= \"a\" ( ) \"b\"\n");
}

#[test]
fn empty_alternative() {
    assert_eq!(ebnf("(a|)"), "root ::= ( \"a\" | \"\" )\n");
    assert_eq!(ebnf("ab(c|)"), "root ::= \"a\" \"b\" ( \"c\" | \"\" )\n");
}

#[test]
fn group_modifiers_supported() {
    // Non-capturing group.
    assert_eq!(ebnf("(?:abc)"), "root ::= ( \"a\" \"b\" \"c\" )\n");
    // Named capturing group — name is ignored.
    assert_eq!(ebnf("(?<name>abc)"), "root ::= ( \"a\" \"b\" \"c\" )\n");
}

#[test]
fn group_modifiers_unsupported() {
    for regex in ["(?=abc)", "(?!abc)", "(?<=abc)", "(?<!abc)", "(?i)abc"] {
        assert!(
            regex_to_ebnf(regex, true).is_err(),
            "regex {regex:?} should be rejected"
        );
    }
}

#[test]
fn invalid_named_group() {
    let e = regex_to_ebnf("(?<name", true).unwrap_err();
    assert!(e.message.contains("Invalid named capturing group"));
}

// ---- parenthesis matching -------------------------------------------

#[test]
fn unmatched_closing_paren() {
    let e = regex_to_ebnf("abc)", true).unwrap_err();
    assert!(e.message.contains("Unmatched ')'"));
}

#[test]
fn unclosed_paren() {
    let e = regex_to_ebnf("abc((a)", true).unwrap_err();
    assert!(e.message.contains("The parenthesis is not closed"));
}

// ---- real-world patterns --------------------------------------------

#[test]
fn ipv4_pattern() {
    let regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?).)((25[0-5]|2[0-4]\d|[01]?\d\d?).)((25[0-5]|2[0-4]\d|[01]?\d\d?).)(25[0-5]|2[0-4]\d|[01]?\d\d?)";
    let expected = concat!(
        "root ::= ( ( \"2\" \"5\" [0-5] | \"2\" [0-4] [0-9] | [01]? [0-9] [0-9]? ) ",
        "[\\u0000-\\U0010FFFF] ) ( ( \"2\" \"5\" [0-5] | \"2\" [0-4] [0-9] | [01]? [0-9] ",
        "[0-9]? ) [\\u0000-\\U0010FFFF] ) ( ( \"2\" \"5\" [0-5] | \"2\" [0-4] [0-9] | [01]? [0-9] ",
        "[0-9]? ) [\\u0000-\\U0010FFFF] ) ( \"2\" \"5\" [0-5] | \"2\" [0-4] [0-9] | [01]? [0-9] [0-9]? )\n",
    );
    assert_eq!(ebnf(regex), expected);
}

#[test]
fn date_pattern() {
    let regex = r"^\d\d\d\d-(0[1-9]|1[0-2])-([0-2]\d|3[01])$";
    let expected = concat!(
        "root ::= [0-9] [0-9] [0-9] [0-9] \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ",
        "( [0-2] [0-9] | \"3\" [01] )\n",
    );
    assert_eq!(ebnf(regex), expected);
}

#[test]
fn time_pattern() {
    let regex = r"^([01]\d|2[0123]):[0-5]\d:[0-5]\d(\.\d+)?(Z|[+-]([01]\d|2[0123]):[0-5]\d)$";
    let expected = concat!(
        "root ::= ( [01] [0-9] | \"2\" [0123] ) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ",
        "( \".\" [0-9]+ )? ( \"Z\" | [+-] ( [01] [0-9] | \"2\" [0123] ) \":\" [0-5] [0-9] )\n",
    );
    assert_eq!(ebnf(regex), expected);
}

#[test]
fn email_pattern_converts() {
    // Just check it converts without error — the structural test
    // lives in the grammar matcher suite.
    let regex = concat!(
        r"^([\w!#$%&'*+/=?^_`{|}~-]+(\.[\w!#$%&'*+/=?^_`{|}~-]+)*",
        r#"|"([\w!#$%&'*+/=?^_`{|}~\-(),:;<>@[\].]|\\")+")@(([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+"#,
        r"[a-z0-9]([a-z0-9-]*[a-z0-9])?)$",
    );
    assert!(regex_to_ebnf(regex, true).is_ok());
}

// ---- empty regexes ---------------------------------------------------

#[test]
fn empty_regexes_produce_empty_string() {
    for regex in ["", "^$", "^", "$"] {
        let out = body(regex);
        assert!(
            out.contains("\"\""),
            "regex {regex:?} should yield an empty string, got {out:?}"
        );
    }
}

#[test]
fn fully_empty_regex() {
    assert_eq!(ebnf(""), "root ::= \"\"\n");
}

// ---- malformed escapes ----------------------------------------------

#[test]
fn unfinished_escape() {
    let e = regex_to_ebnf("a\\", true).unwrap_err();
    assert!(e.message.contains("Escape sequence is not finished"));
}

#[test]
fn backreference_rejected() {
    for regex in [r"a\1", r"\k<x>"] {
        let e = regex_to_ebnf(regex, true).unwrap_err();
        assert!(e.message.contains("Backreference is not supported"));
    }
}

#[test]
fn unicode_property_rejected() {
    let e = regex_to_ebnf(r"\p{L}", true).unwrap_err();
    assert!(
        e.message
            .contains("Unicode character class escape sequence is not supported")
    );
}

#[test]
fn word_boundary_rejected() {
    let e = regex_to_ebnf(r"a\bc", true).unwrap_err();
    assert!(e.message.contains("Word boundary is not supported"));
}

#[test]
fn invalid_control_escape() {
    let e = regex_to_ebnf(r"\c1", true).unwrap_err();
    assert!(e.message.contains("Invalid control character escape"));
}

#[test]
fn invalid_brace_unicode_escape() {
    // `\u{zz}` — non-hex content inside the braces.
    let e = regex_to_ebnf(r"\u{zzzz}", true).unwrap_err();
    assert!(e.message.contains("Invalid Unicode escape sequence"));
}

#[test]
fn unrecognized_escape_warns_and_matches_literally() {
    // `\z` is not a known escape: it warns and matches `z`.
    let out = RegexConverter::new(r"\z").unwrap().convert().unwrap();
    assert_eq!(out.ebnf, "\"z\"");
    assert_eq!(out.warnings.len(), 1);
    assert!(out.warnings[0].contains("is not recognized"));
}

// ---- error position -------------------------------------------------

#[test]
fn error_carries_position() {
    let e = regex_to_ebnf("ab)", true).unwrap_err();
    // The ')' is the third codepoint -> 1-based position 3.
    assert_eq!(e.position, 3);
}

#[test]
fn error_display_format() {
    let e = RegexError {
        position: 5,
        message: "boom".to_string(),
    };
    assert_eq!(format!("{e}"), "Regex parsing error at position 5: boom");
}

// ---- regex_to_grammar (regex -> GrammarData AST) ---------------------

#[test]
fn regex_to_grammar_builds_ast() {
    // The regex→grammar-AST path: lower to EBNF then parse it.
    let grammar = regex_to_grammar("abc").expect("should build a grammar");
    // A well-formed grammar has at least the root rule.
    assert!(grammar.num_rules() >= 1);
}

#[test]
fn regex_to_grammar_with_groups_and_quantifiers() {
    let grammar = regex_to_grammar(r"(a|b)+[0-9]*").expect("should build a grammar");
    assert!(grammar.num_rules() >= 1);
}

#[test]
fn regex_to_grammar_propagates_regex_error() {
    let err = regex_to_grammar("abc)").unwrap_err();
    match err {
        RegexToGrammarError::Regex(e) => assert!(e.message.contains("Unmatched ')'")),
        other => panic!("expected a Regex error, got {other:?}"),
    }
}

#[test]
fn regex_to_grammar_empty() {
    let grammar = regex_to_grammar("").expect("empty regex should build a grammar");
    assert!(grammar.num_rules() >= 1);
}
