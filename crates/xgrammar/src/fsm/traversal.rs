// SPDX-License-Identifier: AGPL-3.0-only
//
// Shared FSM traversal algorithms — epsilon closure, single-step
// transition, and set-advance. Port of the traversal methods in
// `FSMImplBase` / `FSM::Impl` / `CompactFSM::Impl` from `cpp/fsm.cc`.
//
// These work on any `&[Vec<FsmEdge>]`-shaped adjacency view, so the same
// code backs both the mutable `Fsm` and the immutable `CompactFsm` (the
// latter materializes rows as slices via `CompactFsm::advance`).

use ahash::AHashSet;
use std::collections::VecDeque;

use super::edge::{FsmEdge, edge_type};
use super::fsm::NO_NEXT_STATE;

/// Compute the epsilon closure of `state_set` in place over `edges`.
///
/// The set is *not* cleared — every state already present plus every
/// state reachable from it by epsilon edges ends up in the set.
pub fn epsilon_closure(edges: &[Vec<FsmEdge>], state_set: &mut AHashSet<i32>) {
    let mut queue: VecDeque<i32> = state_set.iter().copied().collect();
    while let Some(cur) = queue.pop_front() {
        for edge in &edges[cur as usize] {
            if !edge.is_epsilon() {
                continue;
            }
            if state_set.insert(edge.target) {
                queue.push_back(edge.target);
            }
        }
    }
}

/// First transition target from `from` for `value` along `edge_type`,
/// or [`NO_NEXT_STATE`]. Panics if `edge_type` is epsilon.
pub fn next_state(edges: &[Vec<FsmEdge>], from: usize, value: i32, edge_type: i16) -> i32 {
    debug_assert_ne!(
        edge_type,
        super::edge::edge_type::EPSILON,
        "Should not call next_state with edge type epsilon."
    );
    for edge in &edges[from] {
        if edge_matches(edge, value, edge_type) {
            return edge.target;
        }
    }
    NO_NEXT_STATE
}

/// True if `edge` accepts `value` under `edge_type`.
pub fn edge_matches(edge: &FsmEdge, value: i32, etype: i16) -> bool {
    match etype {
        edge_type::CHAR_RANGE => {
            edge.is_char_range() && (edge.min as i32) <= value && (edge.max as i32) >= value
        }
        edge_type::RULE_REF => edge.is_rule_ref() && edge.max as i32 == value,
        edge_type::EOS => edge.is_eos(),
        edge_type::REPEAT_REF => edge.is_repeat_ref(),
        _ => false,
    }
}

/// Advance the set `from` by `value` along `edge_type` edges.
///
/// `result` is cleared first, then filled with the epsilon closure of
/// every reachable target. If `from_is_closure` is false, the epsilon
/// closure of `from` is taken first.
pub fn advance(
    edges: &[Vec<FsmEdge>],
    from: &AHashSet<i32>,
    value: i32,
    result: &mut AHashSet<i32>,
    edge_type: i16,
    from_is_closure: bool,
) {
    debug_assert_ne!(
        edge_type,
        super::edge::edge_type::EPSILON,
        "Should not call advance with edge type epsilon."
    );

    let mut closure_tmp;
    let start_closure: &AHashSet<i32> = if from_is_closure {
        from
    } else {
        closure_tmp = from.clone();
        epsilon_closure(edges, &mut closure_tmp);
        &closure_tmp
    };

    result.clear();
    for &state in start_closure {
        for edge in &edges[state as usize] {
            if edge_matches(edge, value, edge_type) {
                result.insert(edge.target);
            }
        }
    }
    epsilon_closure(edges, result);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(set: &AHashSet<i32>) -> Vec<i32> {
        let mut v: Vec<i32> = set.iter().copied().collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn closure_of_disconnected_set_is_identity() {
        let edges: Vec<Vec<FsmEdge>> = vec![Vec::new(), Vec::new()];
        let mut set: AHashSet<i32> = AHashSet::from_iter([0, 1]);
        epsilon_closure(&edges, &mut set);
        assert_eq!(sorted(&set), vec![0, 1]);
    }

    #[test]
    fn closure_handles_epsilon_cycle() {
        let edges = vec![
            vec![FsmEdge::new(edge_type::EPSILON, 0, 1)],
            vec![FsmEdge::new(edge_type::EPSILON, 0, 0)],
        ];
        let mut set: AHashSet<i32> = AHashSet::from_iter([0]);
        epsilon_closure(&edges, &mut set);
        assert_eq!(sorted(&set), vec![0, 1]);
    }

    #[test]
    fn next_state_char_range() {
        let edges = vec![vec![FsmEdge::new(b'a' as i16, b'c' as i16, 1)]];
        assert_eq!(next_state(&edges, 0, b'b' as i32, edge_type::CHAR_RANGE), 1);
        assert_eq!(
            next_state(&edges, 0, b'd' as i32, edge_type::CHAR_RANGE),
            NO_NEXT_STATE
        );
    }

    #[test]
    fn advance_with_precomputed_closure() {
        let edges = vec![vec![FsmEdge::new(b'a' as i16, b'a' as i16, 1)], vec![]];
        let from: AHashSet<i32> = AHashSet::from_iter([0]);
        let mut result = AHashSet::new();
        advance(
            &edges,
            &from,
            b'a' as i32,
            &mut result,
            edge_type::CHAR_RANGE,
            true,
        );
        assert_eq!(sorted(&result), vec![1]);
    }

    #[test]
    fn advance_empty_when_no_match() {
        let edges = vec![vec![FsmEdge::new(b'a' as i16, b'a' as i16, 1)], vec![]];
        let from: AHashSet<i32> = AHashSet::from_iter([0]);
        let mut result = AHashSet::new();
        advance(
            &edges,
            &from,
            b'z' as i32,
            &mut result,
            edge_type::CHAR_RANGE,
            true,
        );
        assert!(result.is_empty());
    }
}
