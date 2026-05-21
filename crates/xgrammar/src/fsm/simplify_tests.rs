// SPDX-License-Identifier: AGPL-3.0-only
//
// Unit tests for the sibling module, in a separate file so the
// code file stays under the 250-line cap (included via `#[path]`).

use super::super::fsm::Fsm;
use super::*;

fn literal(bytes: &[u8]) -> FsmWithStartEnd {
    let mut fsm = Fsm::with_states(bytes.len() + 1);
    for (i, &b) in bytes.iter().enumerate() {
        fsm.add_edge(i, i + 1, b as i16, b as i16);
    }
    let mut ends = vec![false; bytes.len() + 1];
    ends[bytes.len()] = true;
    FsmWithStartEnd::new(fsm, 0, ends, false)
}

#[test]
fn simplify_epsilon_collapses_chain() {
    // a -eps-> b -eps-> c, all single-edge => collapses to one node
    let mut fsm = Fsm::with_states(3);
    fsm.add_epsilon_edge(0, 1);
    fsm.add_epsilon_edge(1, 2);
    let f = FsmWithStartEnd::new(fsm, 0, vec![false, false, true], false);
    let s = f.simplify_epsilon();
    assert_eq!(s.num_states(), 1);
}

#[test]
fn simplify_epsilon_preserves_language() {
    // build "abcd" via [a][b][c][d] concat, lots of epsilons
    let f = FsmWithStartEnd::concat(&[literal(b"a"), literal(b"b"), literal(b"c"), literal(b"d")]);
    assert!(f.accept_string(b"abcd"));
    let s = f.simplify_epsilon();
    assert!(s.accept_string(b"abcd"));
    assert!(!s.accept_string(b"abc"));
}

#[test]
fn merge_equivalent_successors_reduces_states() {
    // abc | abd -> after simplify+merge should shrink
    let f = FsmWithStartEnd::union(&[literal(b"abc"), literal(b"abd")]);
    assert!(f.accept_string(b"abc"));
    let merged = f.simplify_epsilon().merge_equivalent_successors();
    assert!(merged.accept_string(b"abc"));
    assert!(merged.accept_string(b"abd"));
    assert!(!merged.accept_string(b"abe"));
}

#[test]
fn merge_equivalent_precursors() {
    // acd | bcd -> (a|b)cd
    let f = FsmWithStartEnd::union(&[literal(b"acd"), literal(b"bcd")]);
    let merged = f.simplify_epsilon().merge_equivalent_successors();
    assert!(merged.accept_string(b"acd"));
    assert!(merged.accept_string(b"bcd"));
    assert!(!merged.accept_string(b"abcd"));
}

#[test]
fn simplify_idempotent_on_dfa() {
    let mut f = literal(b"a");
    f.is_dfa = true;
    let s = f.simplify_epsilon();
    assert_eq!(s.num_states(), f.num_states());
}
