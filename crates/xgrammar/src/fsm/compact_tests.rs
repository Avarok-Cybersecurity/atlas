// SPDX-License-Identifier: AGPL-3.0-only
//
// Unit tests for the sibling module, in a separate file so the
// code file stays under the 250-line cap (included via `#[path]`).

use super::super::edge::edge_type;
use super::*;

fn sorted(set: &AHashSet<i32>) -> Vec<i32> {
    let mut v: Vec<i32> = set.iter().copied().collect();
    v.sort_unstable();
    v
}

fn sample_fsm() -> Fsm {
    // 0 -a-> 1 -b-> 2 (accept)
    let mut f = Fsm::with_states(3);
    f.add_edge(0, 1, b'a' as i16, b'a' as i16);
    f.add_edge(1, 2, b'b' as i16, b'b' as i16);
    f
}

#[test]
fn roundtrip_fsm_compact_fsm() {
    let f = sample_fsm();
    let c = f.to_compact();
    assert_eq!(c.num_states(), 3);
    let back = c.to_fsm();
    assert_eq!(back.num_states(), 3);
    assert_eq!(back.next_state(0, b'a' as i32, edge_type::CHAR_RANGE), 1);
}

#[test]
fn compact_empty_fsm() {
    let c = Fsm::with_states(0).to_compact();
    assert_eq!(c.num_states(), 0);
    assert_eq!(c.num_edges(), 0);
}

#[test]
fn compact_single_state_self_loop() {
    let mut f = Fsm::with_states(1);
    f.add_edge(0, 0, b'x' as i16, b'x' as i16);
    let c = f.to_compact();
    assert_eq!(c.next_state(0, b'x' as i32, edge_type::CHAR_RANGE), 0);
}

#[test]
fn compact_next_states_collects_all() {
    let mut f = Fsm::with_states(3);
    f.add_edge(0, 1, b'a' as i16, b'a' as i16);
    f.add_edge(0, 2, b'a' as i16, b'a' as i16);
    let c = f.to_compact();
    let mut t = Vec::new();
    c.next_states(0, b'a' as i32, edge_type::CHAR_RANGE, &mut t);
    t.sort_unstable();
    assert_eq!(t, vec![1, 2]);
}

#[test]
fn compact_edges_are_sorted() {
    let mut f = Fsm::with_states(1);
    f.add_edge(0, 0, b'z' as i16, b'z' as i16);
    f.add_edge(0, 0, b'a' as i16, b'a' as i16);
    let c = f.to_compact();
    assert!(c.edges(0)[0].min <= c.edges(0)[1].min);
}

#[test]
fn compact_epsilon_closure() {
    let mut f = Fsm::with_states(3);
    f.add_epsilon_edge(0, 1);
    f.add_epsilon_edge(1, 2);
    let c = f.to_compact();
    let mut set: AHashSet<i32> = AHashSet::from_iter([0]);
    c.epsilon_closure(&mut set);
    assert_eq!(sorted(&set), vec![0, 1, 2]);
}

#[test]
fn compact_advance() {
    let c = sample_fsm().to_compact();
    let from: AHashSet<i32> = AHashSet::from_iter([0]);
    let mut result = AHashSet::new();
    c.advance(
        &from,
        b'a' as i32,
        &mut result,
        edge_type::CHAR_RANGE,
        false,
    );
    assert_eq!(sorted(&result), vec![1]);
}

#[test]
fn compact_reachable_and_aux() {
    let mut f = Fsm::with_states(2);
    f.add_repeat_edge(0, 1, 3, 1, 2);
    let c = f.to_compact();
    let info = c.repeat_edge_info(c.edges(0)[0].aux_index());
    assert_eq!((info.rule_id, info.lower, info.upper), (3, 1, 2));
    let mut reach = AHashSet::new();
    c.reachable_states(&[0], &mut reach);
    assert_eq!(sorted(&reach), vec![0, 1]);
}

#[test]
fn memory_size_grows_with_edges() {
    let small = Fsm::with_states(1).to_compact();
    let big = sample_fsm().to_compact();
    assert!(big.memory_size() > small.memory_size());
}
