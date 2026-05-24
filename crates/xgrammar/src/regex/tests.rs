// SPDX-License-Identifier: AGPL-3.0-only
//
// Tests for the regex converter — ported from xgrammar's
// `tests/python/test_regex_converter.py` plus extra Rust-side
// coverage of error paths.

use super::*;

/// Convenience: convert and unwrap the `root ::= …\n` form.
fn ebnf(regex: &str) -> String {
    regex_to_ebnf(regex, true).expect("regex should convert")
}

/// Convenience: the rule-body-only form.
fn body(regex: &str) -> String {
    regex_to_ebnf(regex, false).expect("regex should convert")
}

// ---- basic literals --------------------------------------------------

#[test]
fn basic_literal() {
    assert_eq!(ebnf("123"), "root ::= \"1\" \"2\" \"3\"\n");
}

#[test]
fn body_without_rule_name() {
    assert_eq!(body("123"), "\"1\" \"2\" \"3\"");
}

#[test]
fn unicode_literals() {
    // ww我😁  ->  "w" "w" "我" "\U0001f601"
    assert_eq!(
        ebnf("ww我😁"),
        "root ::= \"w\" \"w\" \"\\u6211\" \"\\U0001f601\"\n"
    );
}

// ---- escapes ---------------------------------------------------------

#[test]
fn escape_special_chars() {
    let regex = r"\^\$\.\*\+\?\\\(\)\[\]\{\}\|\/";
    let expected = "root ::= \"^\" \"$\" \".\" \"*\" \"+\" \"\\?\" \"\\\\\" \"(\" \")\" \"[\" \"]\" \
         \"{\" \"}\" \"|\" \"/\"\n";
    assert_eq!(ebnf(regex), expected);
}

#[test]
fn escape_c_style() {
    // \"\'\a\f\n\r\t\v\0\e
    let regex = "\\\"\\'\\a\\f\\n\\r\\t\\v\\0\\e";
    let expected = "root ::= \"\\\"\" \"\\'\" \"\\a\" \"\\f\" \"\\n\" \"\\r\" \"\\t\" \"\\v\" \"\\0\" \"\\e\"\n";
    assert_eq!(ebnf(regex), expected);
}

#[test]
fn escape_unicode_hex_control() {
    // \u{20BB7}̀\x1F\cJ
    let regex = r"\u{20BB7}̀\x1F\cJ";
    let expected = "root ::= \"\\U00020bb7\" \"\\u0300\" \"\\x1f\" \"\\n\"\n";
    assert_eq!(ebnf(regex), expected);
}

#[test]
fn escape_char_class_with_escapes() {
    // [\r\n\$\u0010-\u006F\]\--]+
    let regex = r"[\r\n\$\u0010-\u006F\]\--]+";
    let expected = "root ::= [\\r\\n$\\x10-o\\]\\--]+\n";
    assert_eq!(ebnf(regex), expected);
}

// ---- escaped character classes ---------------------------------------

#[test]
fn escaped_char_classes() {
    // \w\w\W\d\D\s\S
    let regex = r"\w\w\W\d\D\s\S";
    let expected = "root ::= [a-zA-Z0-9_] [a-zA-Z0-9_] [^a-zA-Z0-9_] [0-9] [^0-9] \
         [\\f\\n\\r\\t\\v\\u0020\\u00a0] [^[\\f\\n\\r\\t\\v\\u0020\\u00a0]\n";
    assert_eq!(ebnf(regex), expected);
}

// ---- character class -------------------------------------------------

#[test]
fn char_class_with_ranges_and_dash() {
    let regex = r"[-a-zA-Z+--]+";
    assert_eq!(ebnf(regex), "root ::= [-a-zA-Z+--]+\n");
}

#[test]
fn empty_char_class_is_error() {
    let e = regex_to_ebnf("[]", true).unwrap_err();
    assert!(e.message.contains("Empty character class is not allowed"));
}

#[test]
fn unclosed_char_class_is_error() {
    let e = regex_to_ebnf("[abc", true).unwrap_err();
    assert!(e.message.contains("Unclosed '['"));
}

// ---- anchors ---------------------------------------------------------

#[test]
fn anchors_are_stripped() {
    assert_eq!(ebnf("^abc$"), "root ::= \"a\" \"b\" \"c\"\n");
}

#[test]
fn stray_anchor_warns_but_succeeds() {
    let out = RegexConverter::new("a^b").unwrap().convert().unwrap();
    assert_eq!(out.ebnf, "\"a\" \"b\"");
    assert_eq!(out.warnings.len(), 1);
    assert!(out.warnings[0].contains("'^' should be at the start"));
}

#[test]
fn stray_dollar_warns() {
    let out = RegexConverter::new("a$b").unwrap().convert().unwrap();
    assert_eq!(out.warnings.len(), 1);
    assert!(out.warnings[0].contains("'$' should be at the end"));
}

// ---- alternation -----------------------------------------------------

#[test]
fn disjunction_with_group() {
    let regex = r"abc|de(f|g)";
    assert_eq!(
        ebnf(regex),
        "root ::= \"a\" \"b\" \"c\" | \"d\" \"e\" ( \"f\" | \"g\" )\n"
    );
}

#[test]
fn spaces_are_literal() {
    let regex = r" abc | df | g ";
    let expected =
        "root ::= \" \" \"a\" \"b\" \"c\" \" \" | \" \" \"d\" \"f\" \" \" | \" \" \"g\" \" \"\n";
    assert_eq!(ebnf(regex), expected);
}

// ---- quantifiers -----------------------------------------------------

#[test]
fn quantifiers_basic() {
    let regex = r"(a|b)?[a-z]+(abc)*";
    let expected = "root ::= ( \"a\" | \"b\" )? [a-z]+ ( \"a\" \"b\" \"c\" )*\n";
    assert_eq!(ebnf(regex), expected);
}

#[test]
fn repetition_range_exact() {
    assert_eq!(ebnf("a{3}"), "root ::= \"a\"{3}\n");
}

#[test]
fn repetition_range_open() {
    assert_eq!(ebnf("a{2,}"), "root ::= \"a\"{2,}\n");
}

#[test]
fn repetition_range_closed() {
    assert_eq!(ebnf("a{1,3}"), "root ::= \"a\"{1,3}\n");
}

#[test]
fn non_greedy_modifiers_dropped() {
    // a{1,3}? -> "a"{1,3}
    assert_eq!(ebnf("a{1,3}?"), "root ::= \"a\"{1,3}\n");
    assert_eq!(ebnf("a+?"), "root ::= \"a\"+\n");
    assert_eq!(ebnf("a*?"), "root ::= \"a\"*\n");
    assert_eq!(ebnf("a??"), "root ::= \"a\"?\n");
}

#[test]
fn consecutive_quantifiers_rejected() {
    for regex in ["a{1,3}?{1,3}", "a???", "a++", "a+?{1,3}", "a*+"] {
        let e = regex_to_ebnf(regex, true).unwrap_err();
        assert!(
            e.message
                .contains("Two consecutive repetition modifiers are not allowed"),
            "regex {regex:?} should be rejected"
        );
    }
}

#[test]
fn invalid_repetition_count() {
    for regex in ["a{}", "a{,3}", "a{1,", "a{1,x}", "a{1x}"] {
        let e = regex_to_ebnf(regex, true).unwrap_err();
        assert!(
            e.message.contains("Invalid repetition count") || e.message.contains("repetition"),
            "regex {regex:?} should be rejected, got {e}"
        );
    }
}

// ---- the dot ---------------------------------------------------------

#[test]
fn dot_matches_anything() {
    let regex = r".+a.+";
    let expected = "root ::= [\\u0000-\\U0010FFFF]+ \"a\" [\\u0000-\\U0010FFFF]+\n";
    assert_eq!(ebnf(regex), expected);
}
