// SPDX-License-Identifier: AGPL-3.0-only
//
// AdaptiveTokenMask partition-correctness and interval-helper tests.

use std::collections::HashSet;

use super::{compiler, idx_of, small_tokenizer};
use crate::compiler::AdaptiveTokenMask;
use crate::compiler::mask::StoreType;
use crate::compiler::mask_gen::possible_token_intervals;

// ----- adaptive token mask correctness -----------------------------

#[test]
fn mask_partition_is_a_total_cover() {
    // root ::= "a" — every state's accept/reject/uncertain partition
    // must be disjoint and cover the whole sorted vocabulary.
    let c = compiler(1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"a\"\n", "root")
        .unwrap();
    let info = cg.tokenizer_info();
    let n = info.sorted_decoded_vocab().len() as i32;

    for (_, mask) in cg.all_reachable_masks() {
        let (accepted, rejected) = mask.materialize(info.sorted_decoded_vocab());
        let uncertain: HashSet<i32> = mask.uncertain_indices.iter().copied().collect();
        let mut all: HashSet<i32> = accepted.iter().copied().collect();
        for r in &rejected {
            assert!(all.insert(*r), "index {r} in both accepted and rejected");
        }
        for u in &uncertain {
            assert!(all.insert(*u), "index {u} double-counted");
        }
        assert_eq!(all.len() as i32, n, "partition does not cover all tokens");
    }
}

#[test]
fn mask_root_rule_has_no_uncertain() {
    let c = compiler(1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"abc\"\n", "root")
        .unwrap();
    for (_, mask) in cg.all_reachable_masks() {
        assert!(
            mask.uncertain_indices.is_empty(),
            "root rule states must have no uncertain tokens"
        );
    }
}

#[test]
fn mask_accepts_expected_first_token() {
    // root ::= "abc" — the initial state must accept the single-char
    // token "a" and the multi-char tokens "ab" / "abc", reject "b".
    let c = compiler(1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"abc\"\n", "root")
        .unwrap();
    let info = cg.tokenizer_info();
    let root_id = cg.grammar().root_rule_id();
    let root_body = cg.grammar().rule(root_id).body_expr_id;
    let fsm = cg.grammar().per_rule_fsms[root_id as usize]
        .as_ref()
        .unwrap();
    let start = fsm.start() as i32;
    let state = crate::earley::ParserState::new(root_id, root_body, start, -1, 0);
    // Drive the JIT path: the start state is the root rule's canonical
    // key, so `is_root` is true.
    let mask = cg.get_or_compute_mask(state, true);
    let (accepted, _) = mask.materialize(info.sorted_decoded_vocab());
    let acc_set: HashSet<i32> = accepted.into_iter().collect();
    assert!(acc_set.contains(&idx_of(info, b"a")), "should accept 'a'");
    assert!(acc_set.contains(&idx_of(info, b"ab")), "should accept 'ab'");
    assert!(
        acc_set.contains(&idx_of(info, b"abc")),
        "should accept 'abc'"
    );
    assert!(!acc_set.contains(&idx_of(info, b"b")), "must reject 'b'");
}

#[test]
fn mask_store_type_thresholds() {
    let info = small_tokenizer();
    let sorted = info.sorted_decoded_vocab();
    // Few accepted, many rejected -> Accepted form.
    let m = AdaptiveTokenMask::from_accepted_rejected(
        info.vocab_size(),
        sorted,
        &[0],
        &[1, 2, 3, 4],
        &[],
    );
    assert_eq!(m.store_type, StoreType::Accepted);
    // Many accepted, few rejected -> Rejected form.
    let m = AdaptiveTokenMask::from_accepted_rejected(
        info.vocab_size(),
        sorted,
        &[1, 2, 3, 4],
        &[0],
        &[],
    );
    assert_eq!(m.store_type, StoreType::Rejected);
}

#[test]
fn mask_from_accepted_only() {
    let info = small_tokenizer();
    let m = AdaptiveTokenMask::from_accepted(
        info.vocab_size(),
        info.sorted_decoded_vocab(),
        &[0, 1],
        &[2],
    );
    assert_eq!(m.store_type, StoreType::Accepted);
    assert_eq!(m.accepted_indices, vec![0, 1]);
    assert_eq!(m.uncertain_indices, vec![2]);
}

#[test]
fn mask_materialize_round_trips_accepted_form() {
    let info = small_tokenizer();
    let n = info.sorted_decoded_vocab().len();
    let m = AdaptiveTokenMask::from_accepted_rejected(
        info.vocab_size(),
        info.sorted_decoded_vocab(),
        &[0, 2],
        &(1..n as i32).filter(|i| *i != 2).collect::<Vec<_>>(),
        &[],
    );
    let (accepted, rejected) = m.materialize(info.sorted_decoded_vocab());
    assert_eq!(accepted, vec![0, 2]);
    assert_eq!(accepted.len() + rejected.len(), n);
}

// ----- possible-token-interval helper ------------------------------

#[test]
fn possible_intervals_match_first_char_mask() {
    let info = small_tokenizer();
    let mut mask = [false; 256];
    mask[b'a' as usize] = true;
    let (intervals, count) = possible_token_intervals(info.sorted_decoded_vocab(), &mask);
    assert!(!intervals.is_empty());
    for (lo, hi) in &intervals {
        for i in *lo..*hi {
            assert_eq!(info.sorted_decoded_vocab()[i as usize].1[0], b'a');
        }
    }
    // 'a', 'ab', 'abc' all start with 'a'.
    assert_eq!(count, 3);
}

#[test]
fn possible_intervals_empty_mask() {
    let info = small_tokenizer();
    let mask = [false; 256];
    let (intervals, count) = possible_token_intervals(info.sorted_decoded_vocab(), &mask);
    assert!(intervals.is_empty());
    assert_eq!(count, 0);
}
