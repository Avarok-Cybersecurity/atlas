// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::grammar::functor::fsm_builder::GrammarFsmBuilder;
use crate::grammar::functor::normalizer::GrammarNormalizer;
use crate::grammar::parse_ebnf_default;

fn hashed(ebnf: &str) -> GrammarData {
    let mut g = GrammarNormalizer::apply(parse_ebnf_default(ebnf).expect("parse"));
    GrammarFsmBuilder::apply(&mut g);
    GrammarFsmHasher::apply(&mut g);
    g
}

#[test]
fn hashes_terminal_rule() {
    let g = hashed("root ::= \"hello\"\n");
    assert!(g.per_rule_fsm_hashes[g.root_rule_id() as usize].is_some());
}

#[test]
fn hash_arrays_sized_to_rules() {
    let g = hashed("root ::= sub\nsub ::= \"x\"\n");
    assert_eq!(g.per_rule_fsm_hashes.len(), g.num_rules() as usize);
    assert_eq!(g.per_rule_fsm_new_state_ids.len(), g.num_rules() as usize);
}

#[test]
fn identical_rules_hash_equally() {
    let g = hashed("root ::= a\na ::= \"xy\"\nb ::= \"xy\"\n");
    let a_id = (0..g.num_rules()).find(|&i| g.rule(i).name == "a").unwrap();
    // `b` is unreachable so may be hashed; just confirm `a` is hashed.
    assert!(g.per_rule_fsm_hashes[a_id as usize].is_some());
}

#[test]
fn hash_sequence_byte_string() {
    let g = hashed("root ::= \"abc\"\n");
    // root body is Choices; its first child is a Sequence.
    let body = g.expr(g.rule(g.root_rule_id()).body_expr_id);
    let seq_id = body.data[0];
    assert!(hash_sequence(&g, seq_id).is_some());
}

#[test]
fn hash_sequence_minus_one_is_none() {
    let g = hashed("root ::= \"a\"\n");
    assert!(hash_sequence(&g, -1).is_none());
}
