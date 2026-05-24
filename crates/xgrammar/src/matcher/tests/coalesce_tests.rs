// SPDX-License-Identifier: AGPL-3.0-only
//
// Coalescence forced-token fast-path tests (Tier 3b).
//
// Verifies `GrammarMatcher::forced_token`, `forced_from_bitmask` and
// `next_forced_tokens`:
//   * a forced state reports exactly the unique legal token;
//   * a choice point reports "not forced";
//   * accepting a forced token lands the matcher in the SAME state the
//     normal sample-then-accept path would (the core correctness
//     guarantee — a forced token must be a true no-op vs sampling);
//   * the forced chain follows multi-token determined runs;
//   * `forced_token` / `next_forced_tokens` never mutate the matcher.
//
// The `tok()` fixture has multi-char tokens (`ab`, `abc`, `abcd`), so
// `"abc"` is NOT forced at the start (a/ab/abc all legal). `"x"` is the
// fixture's only token with no prefix-sharing sibling, so an `"x"`-run
// grammar is genuinely forced — exactly the case Coalescence targets.

use super::super::bitmask::bitmask_size;
use super::{STOP_ID, accepted_ids, id_of, matcher};

// ----- forced_token: detection --------------------------------------

#[test]
fn forced_token_single_legal_token() {
    // root ::= "x" — `x` is the fixture's only token, no prefix
    // sibling; the start state admits exactly one legal token.
    let mut m = matcher("root ::= \"x\"\n");
    assert_eq!(m.forced_token(), Some(id_of("x")));
}

#[test]
fn forced_token_none_at_choice_point() {
    // root ::= "yes" | "no" — first byte is 'y' or 'n': a real choice.
    let mut m = matcher("root ::= \"yes\" | \"no\"\n");
    assert_eq!(m.forced_token(), None);
}

#[test]
fn forced_token_none_with_prefix_siblings() {
    // root ::= "abc" — tokens `a`, `ab`, `abc` all legal: not forced
    // even though the byte path is fully determined. This is the
    // tokenizer-ambiguity case Coalescence must NOT mis-fire on.
    let mut m = matcher("root ::= \"abc\"\n");
    assert!(m.forced_token().is_none());
}

#[test]
fn forced_token_none_when_terminated() {
    let mut m = matcher("root ::= \"x\"\n");
    assert!(m.accept_token(id_of("x"), false));
    assert!(m.accept_token(STOP_ID, false));
    assert!(m.is_terminated());
    assert_eq!(m.forced_token(), None);
}

#[test]
fn forced_token_is_stop_when_only_completion_left() {
    // After consuming the whole literal the root rule is complete and
    // nothing else can be appended — the stop token is forced.
    let mut m = matcher("root ::= \"x\"\n");
    assert!(m.accept_token(id_of("x"), false));
    assert_eq!(m.forced_token(), Some(STOP_ID));
}

// ----- forced_token: does not mutate --------------------------------

#[test]
fn forced_token_does_not_change_state() {
    let mut m = matcher("root ::= \"x\" \"x\"\n");
    let before = accepted_ids(&mut m);
    let _ = m.forced_token();
    let after = accepted_ids(&mut m);
    assert_eq!(before, after, "forced_token must not mutate the matcher");
    assert_eq!(m.num_history_steps(), 0);
}

// ----- correctness: forced accept == normal path --------------------

#[test]
fn accepting_forced_token_equals_normal_path() {
    // The forced token, fed back through accept_token, must leave the
    // matcher in EXACTLY the state the normal sample path would —
    // verified by bit-identical next-token bitmasks.
    let mut forced = matcher("root ::= \"x\" \"x\" \"x\"\n");
    let mut normal = matcher("root ::= \"x\" \"x\" \"x\"\n");

    let tok = forced.forced_token().expect("first step is forced");
    // Normal path: the mask had exactly this one bit; sampling could
    // only have picked `tok`.
    assert!(forced.accept_token(tok, false));
    assert!(normal.accept_token(id_of("x"), false));

    let vocab = forced.tokenizer_info().vocab_size();
    let mut a = vec![0i32; bitmask_size(vocab)];
    let mut b = vec![0i32; bitmask_size(vocab)];
    forced.fill_next_token_bitmask(&mut a, 0, false).unwrap();
    normal.fill_next_token_bitmask(&mut b, 0, false).unwrap();
    assert_eq!(a, b, "forced-accept state must match normal-accept state");
    assert_eq!(forced.num_history_steps(), normal.num_history_steps());
}

// ----- forced_from_bitmask: zero-extra-work entry point -------------

#[test]
fn forced_from_bitmask_agrees_with_forced_token() {
    let mut m = matcher("root ::= \"x\"\n");
    let vocab = m.tokenizer_info().vocab_size();
    let mut buf = vec![0i32; bitmask_size(vocab)];
    m.fill_next_token_bitmask(&mut buf, 0, false).unwrap();
    assert_eq!(m.forced_from_bitmask(&mut buf, 0), Some(id_of("x")));
}

#[test]
fn forced_from_bitmask_none_at_choice_point() {
    let mut m = matcher("root ::= \"yes\" | \"no\"\n");
    let vocab = m.tokenizer_info().vocab_size();
    let mut buf = vec![0i32; bitmask_size(vocab)];
    m.fill_next_token_bitmask(&mut buf, 0, false).unwrap();
    assert_eq!(m.forced_from_bitmask(&mut buf, 0), None);
}

// ----- next_forced_tokens: forced chain -----------------------------

#[test]
fn forced_chain_follows_determined_run() {
    // root ::= "x" "x" "x" — three forced `x`s, then the root rule
    // completes and the stop token is the only legal continuation.
    let mut m = matcher("root ::= \"x\" \"x\" \"x\"\n");
    let x = id_of("x");
    let chain = m.next_forced_tokens(usize::MAX);
    assert_eq!(chain, vec![x, x, x, STOP_ID]);
}

#[test]
fn forced_chain_stops_at_choice_point() {
    // root ::= "x" ("yes" | "no") — `x` is forced, then a choice.
    let mut m = matcher("root ::= \"x\" (\"yes\" | \"no\")\n");
    let chain = m.next_forced_tokens(usize::MAX);
    assert_eq!(chain, vec![id_of("x")]);
}

#[test]
fn forced_chain_empty_at_choice_point() {
    let mut m = matcher("root ::= \"yes\" | \"no\"\n");
    assert!(m.next_forced_tokens(usize::MAX).is_empty());
}

#[test]
fn forced_chain_respects_max_cap() {
    let mut m = matcher("root ::= \"x\" \"x\" \"x\"\n");
    let x = id_of("x");
    assert!(m.next_forced_tokens(0).is_empty());
    assert_eq!(m.next_forced_tokens(2), vec![x, x]);
}

#[test]
fn forced_chain_does_not_change_state() {
    let mut m = matcher("root ::= \"x\" \"x\" \"x\"\n");
    let before = accepted_ids(&mut m);
    let steps_before = m.num_history_steps();
    let _ = m.next_forced_tokens(usize::MAX);
    let after = accepted_ids(&mut m);
    assert_eq!(before, after, "next_forced_tokens must not mutate matcher");
    assert_eq!(m.num_history_steps(), steps_before);
    assert!(
        !m.is_terminated(),
        "virtual stop-token accept must be rolled back"
    );
}

#[test]
fn forced_chain_accept_equals_normal_path() {
    // Accepting the whole forced chain must equal accepting the same
    // tokens one-by-one on a fresh matcher.
    let mut chained = matcher("root ::= \"x\" \"x\" \"x\"\n");
    let mut stepwise = matcher("root ::= \"x\" \"x\" \"x\"\n");

    let chain = chained.next_forced_tokens(usize::MAX);
    for &t in &chain {
        assert!(chained.accept_token(t, false));
    }
    let x = id_of("x");
    for &t in &[x, x, x, STOP_ID] {
        assert!(stepwise.accept_token(t, false));
    }
    assert!(chained.is_terminated());
    assert!(stepwise.is_terminated());
    assert_eq!(chained.num_history_steps(), stepwise.num_history_steps());
}

#[test]
fn forced_chain_after_partial_accept() {
    // After consuming the first `x`, the remaining forced run is the
    // last two `x`s plus the forced stop token.
    let mut m = matcher("root ::= \"x\" \"x\" \"x\"\n");
    assert!(m.accept_token(id_of("x"), false));
    let x = id_of("x");
    assert_eq!(m.next_forced_tokens(usize::MAX), vec![x, x, STOP_ID]);
}

// ----- accept_forced_chain: detect-and-keep -------------------------

#[test]
fn accept_forced_chain_advances_matcher() {
    // `accept_forced_chain` keeps the run accepted in place — the
    // matcher ends terminated, the chain having reached the stop token.
    let mut m = matcher("root ::= \"x\" \"x\" \"x\"\n");
    let x = id_of("x");
    let accepted = m.accept_forced_chain(usize::MAX);
    assert_eq!(accepted, vec![x, x, x, STOP_ID]);
    assert!(m.is_terminated());
    assert_eq!(m.num_history_steps(), 4);
}

#[test]
fn accept_forced_chain_equals_peek_then_accept() {
    // Detect-and-keep must land in the SAME state as peek-then-accept.
    let mut kept = matcher("root ::= \"x\" \"x\" \"x\"\n");
    let mut peeked = matcher("root ::= \"x\" \"x\" \"x\"\n");

    let kept_chain = kept.accept_forced_chain(usize::MAX);
    let peek_chain = peeked.next_forced_tokens(usize::MAX);
    for &t in &peek_chain {
        assert!(peeked.accept_token(t, false));
    }
    assert_eq!(kept_chain, peek_chain);
    assert_eq!(kept.is_terminated(), peeked.is_terminated());
    assert_eq!(kept.num_history_steps(), peeked.num_history_steps());
}

#[test]
fn accept_forced_chain_stops_at_choice_point() {
    // root ::= "x" ("yes" | "no") — accepts only the forced `x`, then
    // stops at the choice; the matcher is left ready to be sampled.
    let mut m = matcher("root ::= \"x\" (\"yes\" | \"no\")\n");
    assert_eq!(m.accept_forced_chain(usize::MAX), vec![id_of("x")]);
    assert!(!m.is_terminated());
    // The next position is a genuine choice — `yes` and `no` both legal.
    let ids = accepted_ids(&mut m);
    assert!(ids.contains(&id_of("yes")));
    assert!(ids.contains(&id_of("no")));
}

#[test]
fn accept_forced_chain_respects_max_cap() {
    let mut m = matcher("root ::= \"x\" \"x\" \"x\"\n");
    let x = id_of("x");
    assert!(m.accept_forced_chain(0).is_empty());
    assert_eq!(m.accept_forced_chain(2), vec![x, x]);
    assert_eq!(m.num_history_steps(), 2);
    assert!(!m.is_terminated());
}

#[test]
fn accept_forced_chain_empty_at_choice_point() {
    let mut m = matcher("root ::= \"yes\" | \"no\"\n");
    assert!(m.accept_forced_chain(usize::MAX).is_empty());
    assert_eq!(m.num_history_steps(), 0);
}

#[test]
fn accept_forced_chain_rollback_restores_state() {
    // The accepted forced run is ordinary history — rolling it back
    // returns the matcher to its pre-chain state.
    let mut m = matcher("root ::= \"x\" \"x\" \"x\"\n");
    let before = accepted_ids(&mut m);
    let n = m.accept_forced_chain(usize::MAX).len() as i32;
    m.rollback(n);
    assert_eq!(before, accepted_ids(&mut m));
    assert_eq!(m.num_history_steps(), 0);
}
