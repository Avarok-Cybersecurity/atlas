// SPDX-License-Identifier: AGPL-3.0-only
//
// Tests for the DFA algorithms (`algorithms.rs` + `dfa_ops.rs`).
// A separate file (included via `#[path]`) so the code files stay
// under the 250-line cap.

use super::DEFAULT_MAX_STATES;
use crate::fsm::fsm::Fsm;
use crate::fsm::with_start_end::FsmWithStartEnd;

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
fn nfa_is_not_dfa() {
    let mut f = literal(b"a").star(); // has epsilon edges
    assert!(!f.is_dfa());
}

#[test]
fn to_dfa_removes_epsilon_and_accepts() {
    let f = FsmWithStartEnd::union(&[literal(b"abc"), literal(b"abd")]);
    let dfa = f.to_dfa(DEFAULT_MAX_STATES).unwrap();
    assert!(dfa.accept_string(b"abc"));
    assert!(dfa.accept_string(b"abd"));
    assert!(!dfa.accept_string(b"abe"));
    for edges in dfa.fsm().all_edges() {
        assert!(edges.iter().all(|e| !e.is_epsilon()));
    }
}

#[test]
fn to_dfa_state_limit() {
    let f = literal(b"abcdef");
    assert!(f.to_dfa(2).is_err());
}

#[test]
fn minimize_dfa_shrinks() {
    let f = FsmWithStartEnd::union(&[literal(b"a"), literal(b"b")]);
    let dfa = f.to_dfa(DEFAULT_MAX_STATES).unwrap();
    let min = dfa.minimize_dfa(DEFAULT_MAX_STATES).unwrap();
    assert!(min.accept_string(b"a"));
    assert!(min.accept_string(b"b"));
    assert!(!min.accept_string(b"c"));
    assert!(min.num_states() <= dfa.num_states());
}

#[test]
fn not_complements_language() {
    let f = literal(b"a");
    let dfa = f.to_dfa(DEFAULT_MAX_STATES).unwrap();
    let neg = dfa.not(DEFAULT_MAX_STATES).unwrap();
    assert!(!neg.accept_string(b"a"));
    assert!(neg.accept_string(b"b"));
    assert!(neg.accept_string(b"ab"));
}

#[test]
fn intersect_of_overlapping_languages() {
    // [c-f]+  intersect  [d-h]* — common: strings over [d-f], len>=1
    let mut lhs_fsm = Fsm::with_states(2);
    lhs_fsm.add_edge(0, 1, b'c' as i16, b'f' as i16);
    lhs_fsm.add_edge(1, 1, b'c' as i16, b'f' as i16);
    let lhs = FsmWithStartEnd::new(lhs_fsm, 0, vec![false, true], false);

    let mut rhs_fsm = Fsm::with_states(1);
    rhs_fsm.add_edge(0, 0, b'd' as i16, b'h' as i16);
    let rhs = FsmWithStartEnd::new(rhs_fsm, 0, vec![true], false);

    let inter = FsmWithStartEnd::intersect(&lhs, &rhs, DEFAULT_MAX_STATES).unwrap();
    assert!(inter.accept_string(b"de"));
    assert!(inter.accept_string(b"def"));
    assert!(!inter.accept_string(b""));
    assert!(!inter.accept_string(b"cd"));
}

#[test]
fn intersect_rejects_non_leaf() {
    let mut fsm = Fsm::with_states(2);
    fsm.add_rule_edge(0, 1, 1);
    let with_rule = FsmWithStartEnd::new(fsm, 0, vec![false, true], false);
    let lit = literal(b"a");
    assert!(FsmWithStartEnd::intersect(&with_rule, &lit, DEFAULT_MAX_STATES).is_err());
}
