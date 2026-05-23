// SPDX-License-Identifier: AGPL-3.0-only
//
// Unit tests for the sibling module, in a separate file so the
// code file stays under the 250-line cap (included via `#[path]`).

use super::*;

fn literal(bytes: &[u8]) -> FsmWithStartEnd {
    // chain of states accepting exactly `bytes`
    let mut fsm = Fsm::with_states(bytes.len() + 1);
    for (i, &b) in bytes.iter().enumerate() {
        fsm.add_edge(i, i + 1, b as i16, b as i16);
    }
    let mut ends = vec![false; bytes.len() + 1];
    ends[bytes.len()] = true;
    FsmWithStartEnd::new(fsm, 0, ends, false)
}

#[test]
fn accept_exact_literal() {
    let f = literal(b"abc");
    assert!(f.accept_string(b"abc"));
    assert!(!f.accept_string(b"ab"));
    assert!(!f.accept_string(b"abcd"));
    assert!(!f.accept_string(b""));
}

#[test]
fn star_accepts_empty_and_repeats() {
    let f = literal(b"a").star();
    assert!(f.accept_string(b""));
    assert!(f.accept_string(b"a"));
    assert!(f.accept_string(b"aaaa"));
    assert!(!f.accept_string(b"b"));
}

#[test]
fn plus_requires_one() {
    let f = literal(b"a").plus();
    assert!(!f.accept_string(b""));
    assert!(f.accept_string(b"a"));
    assert!(f.accept_string(b"aaa"));
}

#[test]
fn optional_accepts_zero_or_one() {
    let f = literal(b"a").optional();
    assert!(f.accept_string(b""));
    assert!(f.accept_string(b"a"));
    assert!(!f.accept_string(b"aa"));
}

#[test]
fn union_accepts_either() {
    let f = FsmWithStartEnd::union(&[literal(b"abc"), literal(b"xyz")]);
    assert!(f.accept_string(b"abc"));
    assert!(f.accept_string(b"xyz"));
    assert!(!f.accept_string(b"abz"));
}

#[test]
fn concat_joins_languages() {
    let f = FsmWithStartEnd::concat(&[literal(b"ab"), literal(b"cd")]);
    assert!(f.accept_string(b"abcd"));
    assert!(!f.accept_string(b"ab"));
    assert!(!f.accept_string(b"abc"));
}

#[test]
fn single_element_union_and_concat_passthrough() {
    let f = literal(b"q");
    assert!(FsmWithStartEnd::union(std::slice::from_ref(&f)).accept_string(b"q"));
    assert!(FsmWithStartEnd::concat(std::slice::from_ref(&f)).accept_string(b"q"));
}

#[test]
fn compact_roundtrip_accepts() {
    let f = FsmWithStartEnd::concat(&[literal(b"ab"), literal(b"cd")]);
    let c = f.to_compact();
    assert!(c.accept_string(b"abcd"));
    assert!(!c.accept_string(b"abce"));
    let back = c.to_fsm();
    assert!(back.accept_string(b"abcd"));
}

#[test]
fn scanable_and_non_terminal_predicates() {
    let mut fsm = Fsm::with_states(3);
    fsm.add_edge(0, 1, b'a' as i16, b'a' as i16);
    fsm.add_rule_edge(1, 2, 5);
    let f = FsmWithStartEnd::new(fsm, 0, vec![false, false, true], false);
    assert!(f.is_scanable_state(0));
    assert!(!f.is_non_terminal_state(0));
    assert!(f.is_non_terminal_state(1));
    assert!(!f.is_leaf());
}

#[test]
fn leaf_when_no_rule_refs() {
    let f = literal(b"ab");
    assert!(f.is_leaf());
}

#[test]
fn add_state_extends_ends() {
    let mut f = literal(b"a");
    let s = f.add_state();
    assert!(!f.is_end_state(s));
    assert_eq!(f.num_states(), 3);
}

#[test]
fn rebuild_with_mapping_preserves_language() {
    let f = literal(b"ab");
    // identity mapping
    let mapping: Vec<usize> = (0..f.num_states()).collect();
    let r = f.rebuild_with_mapping(&mapping, f.num_states());
    assert!(r.accept_string(b"ab"));
}
