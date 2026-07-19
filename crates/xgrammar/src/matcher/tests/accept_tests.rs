// SPDX-License-Identifier: AGPL-3.0-only
//
// Token / string acceptance tests — adapted from
// `test_grammar_matcher_basic.py` (`test_accept_string`,
// `test_token_operations`) and `test_grammar_matcher_ebnf.py`.

use super::super::{BatchGrammarMatcher, GrammarMatcher};
use super::{STOP_ID, compile_ebnf, id_of, matcher, tok};

// ----- accept_string against EBNF grammars --------------------------

#[test]
fn accept_string_literal_grammar() {
    let mut m = matcher("root ::= \"abc\"\n");
    assert!(m.accept_string("abc", false));
}

#[test]
fn accept_string_rejects_wrong_literal() {
    let mut m = matcher("root ::= \"abc\"\n");
    assert!(!m.accept_string("abd", false));
}

#[test]
fn accept_string_negative_char_class() {
    // [^a]+  — one or more non-'a' bytes.
    let mut m = matcher("root ::= [^a]+\n");
    assert!(m.accept_string("bbb", false));
    let mut m2 = matcher("root ::= [^a]+\n");
    assert!(!m2.accept_string("bba", false));
}

#[test]
fn accept_string_partial_then_reject_keeps_state() {
    // Rejecting "abd" must leave the matcher able to accept "abc".
    let mut m = matcher("root ::= \"abc\"\n");
    assert!(!m.accept_string("abd", false));
    assert_eq!(m.num_history_steps(), 0);
    assert!(m.accept_string("abc", false));
}

#[test]
fn accept_string_alternation() {
    let mut m = matcher("root ::= \"yes\" | \"no\"\n");
    assert!(m.accept_string("yes", false));
    let mut m2 = matcher("root ::= \"yes\" | \"no\"\n");
    assert!(m2.accept_string("no", false));
    let mut m3 = matcher("root ::= \"yes\" | \"no\"\n");
    assert!(!m3.accept_string("maybe", false));
}

#[test]
fn accept_string_recursive_grammar() {
    // root ::= "a" root | "b"  — any number of 'a' then a 'b'.
    let mut m = matcher("root ::= \"a\" root | \"b\"\n");
    assert!(m.accept_string("aaab", false));
}

// ----- accept_token -------------------------------------------------

#[test]
fn accept_token_multibyte_token() {
    // The "abc" token (id 6) advances three grammar bytes at once.
    let mut m = matcher("root ::= \"abc\"\n");
    assert!(m.accept_token(id_of("abc"), false));
    assert_eq!(m.num_history_steps(), 1);
    assert!(m.is_grammar_completed());
}

#[test]
fn accept_token_sequence_of_tokens() {
    // Build "abc" out of three single-char tokens.
    let mut m = matcher("root ::= \"abc\"\n");
    assert!(m.accept_token(id_of("a"), false));
    assert!(m.accept_token(id_of("b"), false));
    assert!(m.accept_token(id_of("c"), false));
    assert!(m.is_grammar_completed());
}

#[test]
fn accept_token_rejects_ungrammatical() {
    let mut m = matcher("root ::= \"abc\"\n");
    // "x" is not a legal first byte.
    assert!(!m.accept_token(id_of("x"), false));
    assert_eq!(m.num_history_steps(), 0);
    // The matcher is unchanged: "a" still works.
    assert!(m.accept_token(id_of("a"), false));
}

#[test]
fn accept_token_partial_multibyte_rollback() {
    // Token "abcd" (id 17): grammar accepts "abc" then rejects 'd'.
    // The matcher must roll back the 3 consumed bytes.
    let mut m = matcher("root ::= \"abc\"\n");
    assert!(!m.accept_token(id_of("abcd"), false));
    assert_eq!(m.num_history_steps(), 0);
    assert!(m.accept_token(id_of("abc"), false));
}

#[test]
fn accept_token_out_of_range_rejected() {
    let mut m = matcher("root ::= \"a\"\n");
    let vocab = m.tokenizer_info().vocab_size() as i32;
    assert!(!m.accept_token(-1, false));
    assert!(!m.accept_token(vocab, false));
    assert!(!m.accept_token(vocab + 100, false));
}

#[test]
fn accept_token_special_token_rejected() {
    // Token id 18 decodes to "" -> a special token; never accepted.
    let special = tok().special_token_ids().to_vec();
    assert!(special.contains(&18));
    let mut m = matcher("root ::= \"a\"\n");
    assert!(!m.accept_token(18, false));
}

// ----- stop token / termination ------------------------------------

#[test]
fn stop_token_accepted_after_completion() {
    let mut m = matcher("root ::= \"a\"\n");
    assert!(m.accept_token(id_of("a"), false));
    assert!(m.is_grammar_completed());
    assert!(m.accept_token(STOP_ID, false));
    assert!(m.is_terminated());
}

#[test]
fn stop_token_rejected_before_completion() {
    let mut m = matcher("root ::= \"abc\"\n");
    assert!(m.accept_token(id_of("a"), false));
    // Root rule not yet complete: the stop token is illegal.
    assert!(!m.accept_token(STOP_ID, false));
    assert!(!m.is_terminated());
}

#[test]
fn accept_after_termination_rejected() {
    let mut m = matcher("root ::= \"a\"\n");
    assert!(m.accept_token(id_of("a"), false));
    assert!(m.accept_token(STOP_ID, false));
    // Terminated matcher rejects everything.
    assert!(!m.accept_token(id_of("a"), false));
    assert!(!m.accept_string("a", false));
}

#[test]
fn empty_grammar_accepts_only_stop_token() {
    // root ::= "" — the empty string; root completes immediately.
    let mut m = matcher("root ::= \"\"\n");
    assert!(m.is_grammar_completed());
    assert!(m.accept_token(STOP_ID, false));
    assert!(m.is_terminated());
}

// ----- batched accept ----------------------------------------------

#[test]
fn batch_accept_token_parallel_results() {
    let cg = compile_ebnf("root ::= \"abc\"\n");
    let mut ms = vec![
        GrammarMatcher::from_compiled_grammar(cg.clone()),
        GrammarMatcher::from_compiled_grammar(cg.clone()),
        GrammarMatcher::from_compiled_grammar(cg),
    ];
    // matcher 0 gets legal 'a', matcher 1 gets illegal 'x', 2 gets 'a'.
    let res =
        BatchGrammarMatcher::accept_token(&mut ms, &[id_of("a"), id_of("x"), id_of("a")], false);
    assert_eq!(res, vec![true, false, true]);
}

#[test]
fn batch_accept_string_results() {
    let cg = compile_ebnf("root ::= \"yes\" | \"no\"\n");
    let mut ms = vec![
        GrammarMatcher::from_compiled_grammar(cg.clone()),
        GrammarMatcher::from_compiled_grammar(cg),
    ];
    let res = BatchGrammarMatcher::accept_string(&mut ms, &["yes", "maybe"], false);
    assert_eq!(res, vec![true, false]);
}

#[test]
#[should_panic(expected = "equal length")]
fn batch_accept_token_length_mismatch_panics() {
    let cg = compile_ebnf("root ::= \"a\"\n");
    let mut ms = vec![GrammarMatcher::from_compiled_grammar(cg)];
    let _ = BatchGrammarMatcher::accept_token(&mut ms, &[1, 2], false);
}

// Regression for upstream xgrammar #630 ("segfault in RepetitionRangeExpander
// from wrong grammar object lookup"): two unbounded repeats whose lower bound
// exceeds the unzip threshold (128) in the SAME rule crashed the C++ expander,
// which looked a builder-side expr id up in `base_grammar_` (OOB read). The
// Rust port expands repetition ranges at parse time against a single builder
// (no separate RepetitionRangeExpander functor, no base/builder split), so the
// bug class cannot occur. This pins that invariant end-to-end: the grammar
// compiles and the matcher enforces both lower bounds.
#[test]
fn repetition_unbounded_above_threshold_same_rule() {
    let a129 = "a".repeat(129);
    let b129 = "b".repeat(129);
    let mut ok = matcher("root ::= \"a\"{129,} \"b\"{129,}\n");
    assert!(
        ok.accept_string(&(a129 + &b129), false),
        "129×'a' then 129×'b' must satisfy both unbounded-above-threshold repeats"
    );
    // Fewer than 129 leading 'a's must not let a 'b' through.
    let mut bad = matcher("root ::= \"a\"{129,} \"b\"{129,}\n");
    assert!(
        !bad.accept_string("ab", false),
        "'b' before the 129-'a' lower bound is satisfied must be rejected"
    );
}
