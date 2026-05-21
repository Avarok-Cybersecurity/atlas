// SPDX-License-Identifier: AGPL-3.0-only
//
// Unit tests for the sibling module, in a separate file so the
// code file stays under the 250-line cap (included via `#[path]`).

use super::*;

#[test]
fn walk_basic_trie() {
    let pats: Vec<&[u8]> = vec![b"hello", b"hi", b"good"];
    let res = build_trie(&pats, &[], true, false).unwrap();
    let fsm = res.fsm.fsm();
    // walk "hello"
    let mut s = res.fsm.start() as i32;
    for &c in b"hello" {
        s = fsm.next_state(s as usize, c as i32, edge_type::CHAR_RANGE);
        assert_ne!(s, NO_NEXT_STATE);
    }
    assert!(res.fsm.is_end_state(s as usize));
}

#[test]
fn shared_prefix_states() {
    let pats: Vec<&[u8]> = vec![b"hello", b"hi"];
    let res = build_trie(&pats, &[], true, false).unwrap();
    let fsm = res.fsm.fsm();
    // 'h' transition shared
    let h = fsm.next_state(0, b'h' as i32, edge_type::CHAR_RANGE);
    assert_ne!(h, NO_NEXT_STATE);
    let e = fsm.next_state(h as usize, b'e' as i32, edge_type::CHAR_RANGE);
    let i = fsm.next_state(h as usize, b'i' as i32, edge_type::CHAR_RANGE);
    assert_ne!(e, i);
}

#[test]
fn end_states_in_pattern_order() {
    let pats: Vec<&[u8]> = vec![b"ab", b"cd", b"ef"];
    let res = build_trie(&pats, &[], true, false).unwrap();
    assert_eq!(res.end_states.len(), 3);
}

#[test]
fn walk_failure_returns_no_next_state() {
    let pats: Vec<&[u8]> = vec![b"good"];
    let res = build_trie(&pats, &[], true, false).unwrap();
    let fsm = res.fsm.fsm();
    let g = fsm.next_state(0, b'g' as i32, edge_type::CHAR_RANGE);
    let o = fsm.next_state(g as usize, b'o' as i32, edge_type::CHAR_RANGE);
    // 'e' from 'go' has no edge
    assert_eq!(
        fsm.next_state(o as usize, b'e' as i32, edge_type::CHAR_RANGE),
        NO_NEXT_STATE
    );
}

#[test]
fn no_overlap_rejects_prefix() {
    // "he" is a prefix of "hello"
    let pats: Vec<&[u8]> = vec![b"he", b"hello"];
    assert!(build_trie(&pats, &[], false, false).is_none());
}

#[test]
fn no_overlap_rejects_empty() {
    let pats: Vec<&[u8]> = vec![b""];
    assert!(build_trie(&pats, &[], false, false).is_none());
}

#[test]
fn back_edges_make_aho_corasick() {
    // with back edges, the start state covers all 256 chars
    let pats: Vec<&[u8]> = vec![b"ab"];
    let res = build_trie(&pats, &[], true, true).unwrap();
    let fsm = res.fsm.fsm();
    // any char from start has a transition
    for c in 0..=255u8 {
        assert_ne!(
            fsm.next_state(0, c as i32, edge_type::CHAR_RANGE),
            NO_NEXT_STATE,
            "char {c} uncovered"
        );
    }
}

#[test]
fn unicode_patterns_as_bytes() {
    // "哈" is 3 UTF-8 bytes; trie operates byte-wise
    let ha = "哈".as_bytes();
    let pats: Vec<&[u8]> = vec![ha];
    let res = build_trie(&pats, &[], true, false).unwrap();
    let mut s = 0i32;
    for &c in ha {
        s = res
            .fsm
            .fsm()
            .next_state(s as usize, c as i32, edge_type::CHAR_RANGE);
        assert_ne!(s, NO_NEXT_STATE);
    }
    assert!(res.fsm.is_end_state(s as usize));
}
