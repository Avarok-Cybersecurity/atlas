// SPDX-License-Identifier: AGPL-3.0-only
//
// Unit tests for the sibling module, in a separate file so the
// code file stays under the 250-line cap (included via `#[path]`).

use super::*;
use ahash::AHashSet;

fn collect(set: &AHashSet<i32>) -> Vec<i32> {
    let mut v: Vec<i32> = set.iter().copied().collect();
    v.sort_unstable();
    v
}

#[test]
fn empty_and_single_state() {
    let empty = Fsm::with_states(0);
    assert_eq!(empty.num_states(), 0);
    let one = Fsm::with_states(1);
    assert_eq!(one.num_states(), 1);
    assert!(one.edges(0).is_empty());
}

#[test]
fn add_states_and_edges() {
    let mut f = Fsm::with_states(0);
    let a = f.add_state();
    let b = f.add_state();
    f.add_edge(a, b, b'x' as i16, b'z' as i16);
    assert_eq!(f.num_states(), 2);
    assert_eq!(f.edges(a).len(), 1);
    assert_eq!(
        f.next_state(a, b'y' as i32, edge_type::CHAR_RANGE),
        b as i32
    );
    assert_eq!(
        f.next_state(a, b'w' as i32, edge_type::CHAR_RANGE),
        NO_NEXT_STATE
    );
}

#[test]
fn self_loop_transition() {
    let mut f = Fsm::with_states(1);
    f.add_edge(0, 0, b'a' as i16, b'a' as i16);
    assert_eq!(f.next_state(0, b'a' as i32, edge_type::CHAR_RANGE), 0);
}

#[test]
fn rule_and_eos_edges() {
    let mut f = Fsm::with_states(3);
    f.add_rule_edge(0, 1, 7);
    f.add_eos_edge(0, 2);
    assert_eq!(f.next_state(0, 7, edge_type::RULE_REF), 1);
    assert_eq!(f.next_state(0, 0, edge_type::EOS), 2);
    assert_eq!(f.next_state(0, 9, edge_type::RULE_REF), NO_NEXT_STATE);
}

#[test]
fn repeat_edge_aux_data() {
    let mut f = Fsm::with_states(2);
    f.add_repeat_edge(0, 1, 4, 2, 6);
    let edge = f.edges(0)[0];
    assert!(edge.is_repeat_ref());
    let info = f.repeat_edge_info(edge.aux_index());
    assert_eq!((info.rule_id, info.lower, info.upper), (4, 2, 6));
    assert_eq!(f.next_state(0, 0, edge_type::REPEAT_REF), 1);
}

#[test]
fn epsilon_closure_follows_chains() {
    let mut f = Fsm::with_states(4);
    f.add_epsilon_edge(0, 1);
    f.add_epsilon_edge(1, 2);
    f.add_edge(2, 3, b'a' as i16, b'a' as i16);
    let mut set: AHashSet<i32> = AHashSet::from_iter([0]);
    f.epsilon_closure(&mut set);
    assert_eq!(collect(&set), vec![0, 1, 2]);
}

#[test]
fn advance_uses_epsilon_closure() {
    let mut f = Fsm::with_states(4);
    f.add_epsilon_edge(0, 1);
    f.add_edge(1, 2, b'a' as i16, b'a' as i16);
    f.add_epsilon_edge(2, 3);
    let from: AHashSet<i32> = AHashSet::from_iter([0]);
    let mut result = AHashSet::new();
    f.advance(
        &from,
        b'a' as i32,
        &mut result,
        edge_type::CHAR_RANGE,
        false,
    );
    assert_eq!(collect(&result), vec![2, 3]);
}

#[test]
fn reachable_states_bfs() {
    let mut f = Fsm::with_states(4);
    f.add_edge(0, 1, b'a' as i16, b'a' as i16);
    f.add_edge(1, 2, b'b' as i16, b'b' as i16);
    // state 3 is unreachable
    let mut result = AHashSet::new();
    f.reachable_states(&[0], &mut result);
    assert_eq!(collect(&result), vec![0, 1, 2]);
}

#[test]
fn add_fsm_offsets_states() {
    let mut base = Fsm::with_states(2);
    base.add_edge(0, 1, b'a' as i16, b'a' as i16);
    let mut other = Fsm::with_states(2);
    other.add_edge(0, 1, b'b' as i16, b'b' as i16);
    let mapping = base.add_fsm(&other);
    assert_eq!(mapping, vec![2, 3]);
    assert_eq!(base.num_states(), 4);
    assert_eq!(base.next_state(2, b'b' as i32, edge_type::CHAR_RANGE), 3);
}

#[test]
fn rebuild_drops_epsilon_self_loops_and_dups() {
    let mut f = Fsm::with_states(3);
    f.add_epsilon_edge(0, 1);
    f.add_edge(0, 2, b'a' as i16, b'a' as i16);
    f.add_edge(0, 2, b'a' as i16, b'a' as i16);
    // map states 0 and 1 to the same id
    let rebuilt = f.rebuild_with_mapping(&[0, 0, 1], 2);
    // epsilon 0->1 becomes 0->0 (dropped); duplicate 'a' edges dedup'd
    assert_eq!(rebuilt.edges(0).len(), 1);
}

#[test]
fn sort_edges_orders_rows() {
    let mut f = Fsm::with_states(1);
    f.add_edge(0, 0, b'z' as i16, b'z' as i16);
    f.add_edge(0, 0, b'a' as i16, b'a' as i16);
    f.sort_edges();
    assert_eq!(f.edges(0)[0].min, b'a' as i16);
}

#[test]
fn possible_rules_collects_ids() {
    let mut f = Fsm::with_states(2);
    f.add_rule_edge(0, 1, 3);
    f.add_rule_edge(0, 1, 8);
    f.add_edge(0, 1, b'a' as i16, b'a' as i16);
    let mut rules = AHashSet::new();
    f.possible_rules(0, &mut rules);
    assert_eq!(collect(&rules), vec![3, 8]);
}
