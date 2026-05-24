// SPDX-License-Identifier: AGPL-3.0-only
//
// Rollback + jump-forward tests — adapted from
// `test_grammar_matcher_basic.py` (`test_rollback`,
// `test_find_jump_forward_string`, `test_graceful_rollback_failure`).
//
// The key rollback invariant: accept N tokens, roll back N, and the
// next-token bitmask is bit-identical to before the accepts.

use super::super::bitmask::bitmask_size;
use super::{STOP_ID, accepted_ids, id_of, matcher};

// ----- rollback -----------------------------------------------------

#[test]
fn rollback_one_token_restores_bitmask() {
    let mut m = matcher("root ::= \"abc\"\n");
    let before = accepted_ids(&mut m);
    assert!(m.accept_token(id_of("a"), false));
    m.rollback(1);
    let after = accepted_ids(&mut m);
    assert_eq!(before, after, "bitmask must be identical after rollback");
    assert_eq!(m.num_history_steps(), 0);
}

#[test]
fn rollback_multi_token_restores_bitmask() {
    let mut m = matcher("root ::= \"abc\"\n");
    let before = accepted_ids(&mut m);
    assert!(m.accept_token(id_of("a"), false));
    assert!(m.accept_token(id_of("b"), false));
    m.rollback(2);
    let after = accepted_ids(&mut m);
    assert_eq!(before, after);
}

#[test]
fn rollback_multibyte_token_restores_state() {
    // A multi-char token ("abc") is one rollback unit despite the
    // three grammar bytes it consumed. After "abc" the grammar wants
    // "d" only, a genuinely different accepted set.
    let mut m = matcher("root ::= \"abc\" \"d\"\n");
    let before = accepted_ids(&mut m);
    assert!(m.accept_token(id_of("abc"), false));
    assert_ne!(before, accepted_ids(&mut m));
    m.rollback(1);
    assert_eq!(before, accepted_ids(&mut m));
}

#[test]
fn rollback_then_reaccept_is_deterministic() {
    // Mirrors the Python `test_rollback`: accept 2, roll back 2,
    // re-accept the same 2 — masks at each step must match.
    let mut m = matcher("root ::= \"abc\" \"abc\"\n");
    let vocab = m.tokenizer_info().vocab_size();
    let words = bitmask_size(vocab);

    let mut mask_a = vec![0i32; words];
    m.fill_next_token_bitmask(&mut mask_a, 0, false).unwrap();
    assert!(m.accept_token(id_of("abc"), false));
    let mut mask_b = vec![0i32; words];
    m.fill_next_token_bitmask(&mut mask_b, 0, false).unwrap();
    assert!(m.accept_token(id_of("abc"), false));

    m.rollback(2);

    let mut mask_a2 = vec![0i32; words];
    m.fill_next_token_bitmask(&mut mask_a2, 0, false).unwrap();
    assert_eq!(mask_a, mask_a2);
    assert!(m.accept_token(id_of("abc"), false));
    let mut mask_b2 = vec![0i32; words];
    m.fill_next_token_bitmask(&mut mask_b2, 0, false).unwrap();
    assert_eq!(mask_b, mask_b2);
}

#[test]
fn rollback_stop_token_un_terminates() {
    let mut m = matcher("root ::= \"a\"\n");
    assert!(m.accept_token(id_of("a"), false));
    assert!(m.accept_token(STOP_ID, false));
    assert!(m.is_terminated());
    // Rolling back the stop-token step un-terminates the matcher.
    m.rollback(1);
    assert!(!m.is_terminated());
    // The matcher is usable again — the stop token is still legal.
    assert!(m.accept_token(STOP_ID, false));
    assert!(m.is_terminated());
}

#[test]
fn rollback_zero_is_noop() {
    let mut m = matcher("root ::= \"abc\"\n");
    assert!(m.accept_token(id_of("a"), false));
    m.rollback(0);
    assert_eq!(m.num_history_steps(), 1);
}

#[test]
#[should_panic(expected = "cannot roll back")]
fn rollback_past_history_panics() {
    let mut m = matcher("root ::= \"abc\"\n");
    assert!(m.accept_token(id_of("a"), false));
    m.rollback(5);
}

#[test]
fn rollback_accept_string_unit() {
    // An accepted string is one rollback unit.
    let mut m = matcher("root ::= \"abc\"\n");
    let before = accepted_ids(&mut m);
    assert!(m.accept_string("ab", false));
    assert_eq!(m.num_history_steps(), 1);
    m.rollback(1);
    assert_eq!(before, accepted_ids(&mut m));
}

// ----- jump-forward -------------------------------------------------

#[test]
fn jump_forward_fully_determined_grammar() {
    // root ::= "abc" — every byte is forced; the jump-forward string
    // is the whole literal.
    let mut m = matcher("root ::= \"abc\"\n");
    assert_eq!(m.find_jump_forward_string(), b"abc");
}

#[test]
fn jump_forward_does_not_change_state() {
    let mut m = matcher("root ::= \"abc\"\n");
    let before = accepted_ids(&mut m);
    let _ = m.find_jump_forward_string();
    let after = accepted_ids(&mut m);
    assert_eq!(before, after, "jump-forward must not mutate the matcher");
    assert_eq!(m.num_history_steps(), 0);
}

#[test]
fn jump_forward_stops_at_choice_point() {
    // root ::= "yes" | "no" — the first byte is ambiguous ('y' or 'n')
    // so there is no forced prefix.
    let mut m = matcher("root ::= \"yes\" | \"no\"\n");
    assert!(m.find_jump_forward_string().is_empty());
}

#[test]
fn jump_forward_partial_then_choice() {
    // root ::= "ab" ("c" | "d") — "ab" is forced, then a choice.
    let mut m = matcher("root ::= \"ab\" (\"c\" | \"d\")\n");
    assert_eq!(m.find_jump_forward_string(), b"ab");
}

#[test]
fn jump_forward_after_partial_accept() {
    // After consuming "a", the remaining forced prefix is "bc".
    let mut m = matcher("root ::= \"abc\"\n");
    assert!(m.accept_token(id_of("a"), false));
    assert_eq!(m.find_jump_forward_string(), b"bc");
}

#[test]
fn jump_forward_empty_when_completed() {
    // A completed root rule has no forced continuation.
    let mut m = matcher("root ::= \"a\"\n");
    assert!(m.accept_token(id_of("a"), false));
    assert!(m.find_jump_forward_string().is_empty());
}

#[test]
fn jump_forward_lossy_string_form() {
    let mut m = matcher("root ::= \"abc\"\n");
    assert_eq!(m.find_jump_forward_string_lossy(), "abc");
}
