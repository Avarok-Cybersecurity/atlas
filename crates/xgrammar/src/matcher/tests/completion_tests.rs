// SPDX-License-Identifier: AGPL-3.0-only
//
// Tests for `GrammarMatcher::find_completion_to_accept` — the shortest
// grammar-legal close to a stop-legal state, used for Atlas budget-aware
// graceful close of structured outputs (#144).

use super::matcher;

// A single open construct: after "{a" the matcher sits inside `inner`,
// which may continue ([a-c]) or close ("}"). The shortest legal close is
// the single "}" — exactly the choice point where `forced_token` returns
// None but a completion still exists.
const OPEN_OBJ: &str = "root ::= \"{\" inner \"}\"\ninner ::= [a-c]*\n";

// Two nested constructs: closing "{{a" needs "}}".
const NESTED_OBJ: &str = "root ::= \"{\" \"{\" inner \"}\" \"}\"\ninner ::= [a-c]*\n";

#[test]
fn completion_closes_open_structure() {
    let mut m = matcher(OPEN_OBJ);
    assert!(m.accept_string("{a", false));
    assert!(!m.is_grammar_completed(), "'{{a' is not a complete object");

    let before = m.num_history_steps();
    let close = m
        .find_completion_to_accept(16)
        .expect("a legal close exists");
    assert_eq!(close, b"}".to_vec(), "shortest close is a single '}}'");
    // The search must not mutate the matcher.
    assert_eq!(m.num_history_steps(), before, "search left state advanced");

    // Applying the close reaches a stop-legal (completed) state.
    let s = String::from_utf8(close).unwrap();
    assert!(m.accept_string(&s, false));
    assert!(
        m.is_grammar_completed(),
        "after the close the grammar can stop"
    );
}

#[test]
fn completion_closes_nested_structure() {
    let mut m = matcher(NESTED_OBJ);
    assert!(m.accept_string("{{a", false));
    assert!(!m.is_grammar_completed());

    let close = m
        .find_completion_to_accept(16)
        .expect("a legal close exists");
    assert_eq!(close, b"}}".to_vec(), "shortest close is '}}}}'");

    let s = String::from_utf8(close).unwrap();
    assert!(m.accept_string(&s, false));
    assert!(m.is_grammar_completed());
}

#[test]
fn completion_none_when_budget_too_small() {
    let mut m = matcher(NESTED_OBJ);
    assert!(m.accept_string("{{a", false));
    // The shortest close is 2 bytes; a 1-byte budget cannot reach it.
    assert_eq!(m.find_completion_to_accept(1), None);
}

#[test]
fn completion_empty_when_already_complete() {
    let mut m = matcher("root ::= \"a\"\n");
    assert!(m.accept_string("a", false));
    assert!(m.is_grammar_completed());
    // Already stop-legal: the close is the empty sequence.
    assert_eq!(m.find_completion_to_accept(8), Some(Vec::new()));
}

// A close that must pass a structural separator then a (here empty) required
// field before the brace — the shape a pure breadth-first search loses to
// content-branch explosion. After "{a" the grammar may extend `first`
// ([a-c]*) or move on; the legal close is ",}" (separator, empty `second`,
// brace). The closure-preferring DFS must take the separator/closers rather
// than extending `first`.
const SEP_OBJ: &str =
    "root ::= \"{\" first \",\" second \"}\"\nfirst ::= [a-c]*\nsecond ::= [a-c]*\n";

#[test]
fn completion_closes_through_required_separator() {
    let mut m = matcher(SEP_OBJ);
    assert!(m.accept_string("{a", false));
    assert!(!m.is_grammar_completed());
    let close = m
        .find_completion_to_accept(16)
        .expect("a close through the separator exists");
    assert_eq!(
        close,
        b",}".to_vec(),
        "close = separator, then empty `second`, then brace"
    );
    let s = String::from_utf8(close).unwrap();
    assert!(m.accept_string(&s, false));
    assert!(m.is_grammar_completed());
}

#[test]
fn completion_token_ids_encode_close() {
    // The fixture vocab (tests.rs::tok) has "}" at id 10.
    let mut m = matcher(OPEN_OBJ);
    assert!(m.accept_string("{a", false));
    assert_eq!(m.find_completion_token_ids(16), Some(vec![10]));

    let mut n = matcher(NESTED_OBJ);
    assert!(n.accept_string("{{a", false));
    assert_eq!(n.find_completion_token_ids(16), Some(vec![10, 10]));
}

#[test]
fn completion_leaves_choice_point_intact() {
    // After the search, the matcher must still accept both a continuation
    // ('b', extending `inner`) and the close ('}') — proving the round-trip
    // restored the choice point exactly.
    let mut m = matcher(OPEN_OBJ);
    assert!(m.accept_string("{a", false));
    let _ = m.find_completion_to_accept(16).expect("close exists");

    let mut cont = matcher(OPEN_OBJ);
    assert!(cont.accept_string("{a", false));
    assert!(
        cont.accept_string("b", false),
        "inner still extendable after search"
    );

    let mut closed = matcher(OPEN_OBJ);
    assert!(closed.accept_string("{a", false));
    assert!(
        closed.accept_string("}", false),
        "close still legal after search"
    );
}
