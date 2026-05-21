// SPDX-License-Identifier: AGPL-3.0-only
//
// Unit tests for the sibling module, in a separate file so the
// code file stays under the 250-line cap (included via `#[path]`).

use super::*;

#[test]
fn empty_ir_accepts_empty_string() {
    let ir = RegexIr::default();
    let fsm = ir.build().unwrap();
    assert!(fsm.accept_string(b""));
    assert!(!fsm.accept_string(b"a"));
}

#[test]
fn literal_leaf() {
    let fsm = build_leaf_fsm(b"abc");
    assert!(fsm.accept_string(b"abc"));
    assert!(!fsm.accept_string(b"ab"));
}

#[test]
fn dot_matches_any_byte() {
    let fsm = build_leaf_fsm(b"a.c");
    assert!(fsm.accept_string(b"axc"));
    assert!(fsm.accept_string(b"a c"));
}

#[test]
fn class_leaf_range() {
    let fsm = build_leaf_fsm(b"[a-c]");
    assert!(fsm.accept_string(b"a"));
    assert!(fsm.accept_string(b"c"));
    assert!(!fsm.accept_string(b"d"));
}

#[test]
fn class_leaf_negated() {
    let fsm = build_leaf_fsm(b"[^0-9]");
    assert!(fsm.accept_string(b"a"));
    assert!(!fsm.accept_string(b"5"));
}

#[test]
fn class_digit_escape() {
    let fsm = build_leaf_fsm(b"[\\d]");
    for d in b'0'..=b'9' {
        assert!(fsm.accept_string(&[d]));
    }
    assert!(!fsm.accept_string(b"a"));
}

#[test]
fn class_coalesces_duplicates() {
    // [abcdabcd] should produce one [a-d] edge
    let fsm = build_leaf_fsm(b"[abcdabcd]");
    assert_eq!(fsm.fsm().edges(0).len(), 1);
    assert!(fsm.accept_string(b"a"));
    assert!(!fsm.accept_string(b"e"));
}

#[test]
fn handle_escapes_n() {
    let v = handle_escapes(b"\\n", 0);
    assert_eq!(v, vec![('\n' as i32, '\n' as i32)]);
}

#[test]
fn symbol_star_via_ir() {
    let ir = RegexIr {
        states: vec![RegexState::Symbol {
            symbol: RegexSymbol::Star,
            state: Box::new(RegexState::Leaf {
                regex: b"a".to_vec(),
            }),
        }],
    };
    let fsm = ir.build().unwrap();
    assert!(fsm.accept_string(b""));
    assert!(fsm.accept_string(b"aaa"));
}

#[test]
fn repeat_n_m_via_ir() {
    let ir = RegexIr {
        states: vec![RegexState::Repeat {
            states: vec![RegexState::Leaf {
                regex: b"[\\d]".to_vec(),
            }],
            lower_bound: 1,
            upper_bound: 3,
        }],
    };
    let fsm = ir.build().unwrap();
    assert!(fsm.accept_string(b"1"));
    assert!(fsm.accept_string(b"123"));
    assert!(!fsm.accept_string(b"1234"));
    assert!(!fsm.accept_string(b""));
}

#[test]
fn union_requires_two_branches() {
    let bad = RegexState::Union {
        states: vec![RegexState::Leaf {
            regex: b"a".to_vec(),
        }],
    };
    assert!(visit(&bad).is_err());
}
