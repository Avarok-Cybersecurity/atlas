// SPDX-License-Identifier: AGPL-3.0-only
//
// Earley parser tests. The end-to-end cases mirror xgrammar's
// `tests/python/test_grammar_matcher_ebnf.py` at the parser level:
// build an optimized grammar, advance the parser over a string, and
// check `is_completed`.

use std::sync::Arc;

use super::*;
use crate::grammar::functor::{GrammarNormalizer, GrammarOptimizer};
use crate::grammar::parse_ebnf_default;

/// Build an optimized, FSM-accelerated grammar from an EBNF string.
fn optimized(ebnf: &str) -> Arc<crate::grammar::GrammarData> {
    let g = parse_ebnf_default(ebnf).expect("parse EBNF");
    let g = GrammarNormalizer::apply(g);
    Arc::new(GrammarOptimizer::apply(g))
}

/// True if `s` is fully accepted by `grammar` — advance every byte then
/// check completion. This is `_is_grammar_accept_string` at parser level.
fn accepts(grammar: Arc<crate::grammar::GrammarData>, s: &str) -> bool {
    let mut p = EarleyParser::from_grammar(grammar);
    for &b in s.as_bytes() {
        if !p.advance(b) {
            return false;
        }
    }
    p.is_completed()
}

/* -------------------- basic accept / reject -------------------- */

#[test]
fn accept_byte_string() {
    let g = optimized("root ::= \"hello\"\n");
    assert!(accepts(g.clone(), "hello"));
    assert!(!accepts(g.clone(), "hell"));
    assert!(!accepts(g.clone(), "helloo"));
    assert!(!accepts(g, "world"));
}

#[test]
fn reject_unknown_first_byte() {
    let g = optimized("root ::= \"abc\"\n");
    let mut p = EarleyParser::from_grammar(g);
    assert!(!p.advance(b'x'));
    // Rejected byte leaves the parser usable.
    assert!(p.advance(b'a'));
}

#[test]
fn accept_alternation() {
    let g = optimized("root ::= \"yes\" | \"no\"\n");
    assert!(accepts(g.clone(), "yes"));
    assert!(accepts(g.clone(), "no"));
    assert!(!accepts(g.clone(), "maybe"));
    assert!(!accepts(g, "y"));
}

#[test]
fn accept_character_class() {
    let g = optimized("root ::= [a-z]\n");
    assert!(accepts(g.clone(), "a"));
    assert!(accepts(g.clone(), "m"));
    assert!(accepts(g.clone(), "z"));
    assert!(!accepts(g.clone(), "A"));
    assert!(!accepts(g, "0"));
}

#[test]
fn accept_negated_character_class() {
    let g = optimized("root ::= [^a-z]\n");
    assert!(accepts(g.clone(), "A"));
    assert!(accepts(g.clone(), "0"));
    assert!(!accepts(g, "a"));
}

/* -------------------- nested rules / recursion -------------------- */

#[test]
fn nested_rules_simple() {
    // test_grammar_matcher_ebnf.py::test_simple
    let g = optimized(
        "root ::= rule1 rule2\n\
         rule1 ::= (rule2 | rule3) \"a\"\n\
         rule2 ::= \"b\"\n\
         rule3 ::= \"c\"\n",
    );
    assert!(accepts(g.clone(), "bab"));
    assert!(!accepts(g.clone(), "abb"));
    assert!(accepts(g, "cab"));
}

#[test]
fn left_recursion_balanced_parens() {
    let g = optimized("root ::= \"(\" root \")\" | \"\"\n");
    assert!(accepts(g.clone(), ""));
    assert!(accepts(g.clone(), "()"));
    assert!(accepts(g.clone(), "(())"));
    assert!(accepts(g.clone(), "((()))"));
    assert!(!accepts(g.clone(), "("));
    assert!(!accepts(g, "(()"));
}

#[test]
fn right_recursive_list() {
    let g = optimized("root ::= \"a\" root | \"a\"\n");
    assert!(accepts(g.clone(), "a"));
    assert!(accepts(g.clone(), "aaaa"));
    assert!(accepts(g.clone(), "aaaaaaaa"));
    assert!(!accepts(g, ""));
}

/* -------------------- empty rules -------------------- */

#[test]
fn empty_rule_accepts_empty_string() {
    let g = optimized("root ::= \"\"\n");
    let p = EarleyParser::from_grammar(g);
    assert!(p.is_completed());
}

#[test]
fn optional_rule() {
    let g = optimized("root ::= \"a\"? \"b\"\n");
    assert!(accepts(g.clone(), "b"));
    assert!(accepts(g.clone(), "ab"));
    assert!(!accepts(g, "aab"));
}

/* -------------------- repetition -------------------- */

#[test]
fn repetition_bounds() {
    // test_grammar_matcher_ebnf.py::test_repetition
    let g = optimized("root ::= rule {2, 3}\nrule ::= (\"a\" | [bc] {4,})\n");
    assert!(accepts(g.clone(), "aaa"));
    assert!(accepts(g.clone(), "abcbc"));
    assert!(accepts(g.clone(), "bcbcbcbcbc"));
    assert!(!accepts(g.clone(), "d"));
    assert!(!accepts(g, "aaaa"));
}

#[test]
fn repetition_with_empty() {
    // test_grammar_matcher_ebnf.py::test_repetition_with_empty
    let g = optimized("root ::= rule {2, 3} \"d\"?\nrule ::= (\"a\" | [bc] {4,}) | \"\"\n");
    assert!(accepts(g.clone(), "aaa"));
    assert!(accepts(g.clone(), ""));
    assert!(accepts(g.clone(), "a"));
    assert!(accepts(g.clone(), "d"));
    assert!(!accepts(g, "aaaa"));
}

#[test]
fn star_quantifier() {
    let g = optimized("root ::= [a-z]*\n");
    assert!(accepts(g.clone(), ""));
    assert!(accepts(g.clone(), "abc"));
    assert!(accepts(g.clone(), "xyzxyz"));
    assert!(!accepts(g, "abc1"));
}

/* -------------------- UTF-8 / FSM-accelerated path -------------------- */

#[test]
fn utf8_multibyte_class() {
    // test_grammar_matcher_ebnf.py::test_utf8
    let g = optimized("root ::= [，]+\n");
    assert!(accepts(g.clone(), "，"));
    assert!(accepts(g.clone(), "，，，"));
    assert!(!accepts(g, "a"));
}

#[test]
fn fsm_accelerated_concatenation() {
    let g = optimized("root ::= \"foo\" \"bar\" \"baz\"\n");
    assert!(accepts(g.clone(), "foobarbaz"));
    assert!(!accepts(g.clone(), "foobar"));
    assert!(!accepts(g, "foobazbar"));
}

/* -------------------- termination detection -------------------- */

#[test]
fn is_completed_tracks_each_position() {
    let g = optimized("root ::= \"ab\" | \"abc\"\n");
    let mut p = EarleyParser::from_grammar(g);
    assert!(!p.is_completed());
    assert!(p.advance(b'a'));
    assert!(!p.is_completed());
    assert!(p.advance(b'b'));
    assert!(p.is_completed()); // "ab" matched
    assert!(p.advance(b'c'));
    assert!(p.is_completed()); // "abc" matched
}

#[test]
fn no_further_advance_after_full_match() {
    let g = optimized("root ::= \"x\"\n");
    let mut p = EarleyParser::from_grammar(g);
    assert!(p.advance(b'x'));
    assert!(p.is_completed());
    assert!(!p.advance(b'x'));
}

/* -------------------- acceptable-next-byte -------------------- */

#[test]
fn acceptable_bytes_byte_string() {
    let g = optimized("root ::= \"abc\"\n");
    let p = EarleyParser::from_grammar(g);
    assert!(p.can_accept(b'a'));
    assert!(!p.can_accept(b'b'));
    assert!(!p.can_accept(b'z'));
}

#[test]
fn acceptable_bytes_character_class() {
    let g = optimized("root ::= [d-f]\n");
    let p = EarleyParser::from_grammar(g);
    let mask = p.acceptable_byte_mask();
    assert!(!mask[b'c' as usize]);
    assert!(mask[b'd' as usize]);
    assert!(mask[b'e' as usize]);
    assert!(mask[b'f' as usize]);
    assert!(!mask[b'g' as usize]);
}

#[test]
fn acceptable_bytes_alternation_union() {
    let g = optimized("root ::= \"cat\" | \"dog\"\n");
    let p = EarleyParser::from_grammar(g);
    assert!(p.can_accept(b'c'));
    assert!(p.can_accept(b'd'));
    assert!(!p.can_accept(b'x'));
}

#[test]
fn acceptable_bytes_updates_after_advance() {
    let g = optimized("root ::= \"hi\"\n");
    let mut p = EarleyParser::from_grammar(g);
    assert!(p.can_accept(b'h'));
    assert!(!p.can_accept(b'i'));
    p.advance(b'h');
    assert!(p.can_accept(b'i'));
    assert!(!p.can_accept(b'h'));
}

/* -------------------- rollback correctness -------------------- */

#[test]
fn rollback_one_restores_prior_state() {
    let g = optimized("root ::= \"abcd\"\n");
    let mut p = EarleyParser::from_grammar(g);
    p.advance(b'a');
    p.advance(b'b');
    let states_before = p.latest_scanable_states().to_vec();
    let completed_before = p.is_completed();
    let steps_before = p.num_steps();

    p.advance(b'c');
    p.pop_last_states(1);

    assert_eq!(p.num_steps(), steps_before);
    assert_eq!(p.latest_scanable_states(), states_before.as_slice());
    assert_eq!(p.is_completed(), completed_before);
    // Parser still works after rollback.
    assert!(p.advance(b'c'));
    assert!(p.advance(b'd'));
    assert!(p.is_completed());
}

#[test]
fn rollback_n_restores_prior_state() {
    let g = optimized("root ::= \"0123456789\"\n");
    let mut p = EarleyParser::from_grammar(g);
    p.advance(b'0');
    p.advance(b'1');
    let snapshot = p.latest_scanable_states().to_vec();
    let steps = p.num_steps();

    for b in b"23456" {
        p.advance(*b);
    }
    p.pop_last_states(5);

    assert_eq!(p.num_steps(), steps);
    assert_eq!(p.latest_scanable_states(), snapshot.as_slice());
    for b in b"23456789" {
        assert!(p.advance(*b));
    }
    assert!(p.is_completed());
}

#[test]
fn rollback_after_recursion() {
    let g = optimized("root ::= \"(\" root \")\" | \"\"\n");
    let mut p = EarleyParser::from_grammar(g);
    p.advance(b'(');
    p.advance(b'(');
    let snapshot = p.latest_scanable_states().to_vec();
    let steps = p.num_steps();
    p.advance(b'(');
    p.advance(b')');
    p.pop_last_states(2);
    assert_eq!(p.num_steps(), steps);
    assert_eq!(p.latest_scanable_states(), snapshot.as_slice());
    // Finish the balanced string from the rolled-back point.
    assert!(p.advance(b')'));
    assert!(p.advance(b')'));
    assert!(p.is_completed());
}

#[test]
#[should_panic(expected = "cannot pop")]
fn rollback_past_initial_state_panics() {
    let g = optimized("root ::= \"a\"\n");
    let mut p = EarleyParser::from_grammar(g);
    p.advance(b'a');
    // 2 steps recorded (initial + 1); popping 2 would empty history.
    p.pop_last_states(2);
}

/* -------------------- reset -------------------- */

#[test]
fn reset_returns_to_fresh_parse() {
    let g = optimized("root ::= \"ab\"\n");
    let mut p = EarleyParser::from_grammar(g);
    p.advance(b'a');
    p.advance(b'b');
    assert!(p.is_completed());
    p.reset();
    assert_eq!(p.num_steps(), 1);
    assert!(!p.is_completed());
    assert!(p.advance(b'a'));
    assert!(p.advance(b'b'));
    assert!(p.is_completed());
}

/* -------------------- push-one-state probe -------------------- */

#[test]
fn push_one_state_to_check_then_pop() {
    let g = optimized("root ::= \"ab\"\n");
    let mut p = EarleyParser::from_grammar(g);
    p.advance(b'a');
    let steps = p.num_steps();
    let probe = p.latest_scanable_states()[0];
    p.push_one_state_to_check(probe);
    assert_eq!(p.num_steps(), steps + 1);
    p.pop_last_states(1);
    assert_eq!(p.num_steps(), steps);
    assert!(p.advance(b'b'));
    assert!(p.is_completed());
}

/* -------------------- construction guards -------------------- */

#[test]
#[should_panic(expected = "requires an optimized grammar")]
fn unoptimized_grammar_panics() {
    let g = parse_ebnf_default("root ::= \"a\"\n").unwrap();
    // Skip the optimizer: `optimized` flag is false.
    EarleyParser::from_grammar(Arc::new(g));
}

#[test]
fn deeply_nested_grammar() {
    let g = optimized("root ::= a\na ::= b b\nb ::= c c\nc ::= d d\nd ::= \"x\"\n");
    // root -> 2 b -> 4 c -> 8 d -> 8 'x'
    assert!(accepts(g.clone(), "xxxxxxxx"));
    assert!(!accepts(g.clone(), "xxxxxxx"));
    assert!(!accepts(g, "xxxxxxxxx"));
}

#[test]
fn json_like_recursive_grammar() {
    // `members` is either empty or a comma-separated list of pairs.
    let g = optimized(
        "root ::= obj\n\
         obj ::= \"{\" members \"}\"\n\
         members ::= nonempty | \"\"\n\
         nonempty ::= pair | pair \",\" nonempty\n\
         pair ::= \"k\" \":\" value\n\
         value ::= \"v\" | obj\n",
    );
    assert!(accepts(g.clone(), "{}"));
    assert!(accepts(g.clone(), "{k:v}"));
    assert!(accepts(g.clone(), "{k:v,k:v}"));
    assert!(accepts(g.clone(), "{k:{k:v}}"));
    assert!(!accepts(g.clone(), "{k:v,}"));
    assert!(!accepts(g.clone(), "{k:v,,k:v}"));
    assert!(!accepts(g, "{k}"));
}

/* -------------------- ZapFormat dead-state pruning -------------------- */
//
// These tests cover Tier 3a: the co-accessibility analysis in `prune.rs`
// and its hook in `advance` / `push_state_and_expand`. They verify both
// halves of the correctness contract:
//   * pruning never removes a state that could lead to acceptance — so
//     the accepted language and `is_completed` are unchanged;
//   * the productivity analysis classifies nodes exactly.

#[path = "prune_tests.rs"]
mod prune_tests;
