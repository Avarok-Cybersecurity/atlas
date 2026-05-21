// SPDX-License-Identifier: AGPL-3.0-only
//
// TrieFsmBuilder — builds a trie / Aho-Corasick FSM from a pattern list.
// Port of `TrieFSMBuilder` / `TrieFSMBuilderImpl` from `cpp/fsm_builder.cc`.

use ahash::AHashSet;
use std::collections::BTreeSet;

use crate::fsm::edge::{FsmEdge, cmp_edge_range, edge_type};
use crate::fsm::fsm::{Fsm, NO_NEXT_STATE};
use crate::fsm::with_start_end::FsmWithStartEnd;

/// Result of a trie build — the FSM plus the per-pattern terminal states.
pub struct TrieBuildResult {
    /// The constructed FSM with start/end states.
    pub fsm: FsmWithStartEnd,
    /// Terminal state of each input pattern, in input order.
    pub end_states: Vec<i32>,
}

/// Build a trie-based FSM.
///
/// `patterns` are inserted as a character trie. `excluded_patterns` are
/// only honored when `add_back_edges` is true (the trie becomes an
/// Aho-Corasick automaton with the excluded terminals pruned).
///
/// Returns `None` when `allow_overlap` is false and a pattern is empty
/// or is a prefix of / shares a terminal with another pattern.
pub fn build_trie(
    patterns: &[&[u8]],
    excluded_patterns: &[&[u8]],
    allow_overlap: bool,
    add_back_edges: bool,
) -> Option<TrieBuildResult> {
    let mut fsm = Fsm::with_states(1);
    let start = 0usize;
    let mut ends: AHashSet<i32> = AHashSet::new();
    let mut end_states: Vec<i32> = Vec::new();

    for pattern in patterns {
        if !allow_overlap && pattern.is_empty() {
            return None;
        }
        let mut current = 0i32;
        for &ch in *pattern {
            let ch16 = ch as i16;
            let mut next = fsm.next_state(current as usize, ch16 as i32, edge_type::CHAR_RANGE);
            if next == NO_NEXT_STATE {
                next = fsm.add_state() as i32;
                fsm.add_edge(current as usize, next as usize, ch16, ch16);
            }
            current = next;
            if !allow_overlap && ends.contains(&current) {
                return None;
            }
        }
        if !allow_overlap && !fsm.edges(current as usize).is_empty() {
            return None;
        }
        ends.insert(current);
        end_states.push(current);
    }

    let mut dead_states: AHashSet<i32> = AHashSet::new();

    if add_back_edges {
        for excluded in excluded_patterns {
            if !allow_overlap && excluded.is_empty() {
                return None;
            }
            let mut current = 0i32;
            for &ch in *excluded {
                let ch16 = ch as i16;
                let mut next = fsm.next_state(current as usize, ch16 as i32, edge_type::CHAR_RANGE);
                if next == NO_NEXT_STATE {
                    next = fsm.add_state() as i32;
                    fsm.add_edge(current as usize, next as usize, ch16, ch16);
                }
                current = next;
                if !allow_overlap && ends.contains(&current) {
                    return None;
                }
            }
            if !allow_overlap && !fsm.edges(current as usize).is_empty() {
                return None;
            }
            ends.insert(current);
            dead_states.insert(current);
        }

        add_back_edges_to_fsm(&mut fsm, start, &ends);

        // Drop edges pointing at excluded (dead) terminal states.
        if !dead_states.is_empty() {
            for state in 0..fsm.num_states() {
                let kept: Vec<FsmEdge> = fsm
                    .edges(state)
                    .iter()
                    .copied()
                    .filter(|e| !dead_states.contains(&e.target))
                    .collect();
                *fsm.edges_mut(state) = kept;
            }
        }
    }

    let mut is_end_state = vec![false; fsm.num_states()];
    for &end in &ends {
        is_end_state[end as usize] = true;
    }
    Some(TrieBuildResult {
        fsm: FsmWithStartEnd::new(fsm, start, is_end_state, false),
        end_states,
    })
}

/// Insert Aho-Corasick back edges so a failed match restarts the search.
fn add_back_edges_to_fsm(fsm: &mut Fsm, start: usize, ends: &AHashSet<i32>) {
    for i in 0..fsm.num_states() {
        if i == start || ends.contains(&(i as i32)) {
            continue;
        }
        let mut edge_set: BTreeSet<OrdRange> = fsm.edges(i).iter().map(|e| OrdRange(*e)).collect();

        // Step 1: inherit the start state's edges (for chars not present).
        for root_edge in fsm.edges(start) {
            edge_set.insert(OrdRange(*root_edge));
        }
        // Step 2: route every uncovered char back to `start`.
        fill_range_gaps(&mut edge_set, start);

        let new_edges: Vec<FsmEdge> = edge_set.iter().map(|o| o.0).collect();
        *fsm.edges_mut(i) = new_edges;
    }

    // Finally, complete the start state itself with range edges.
    let mut start_set: BTreeSet<OrdRange> = fsm.edges(start).iter().map(|e| OrdRange(*e)).collect();
    fill_range_gaps(&mut start_set, start);
    *fsm.edges_mut(start) = start_set.iter().map(|o| o.0).collect();
}

/// Insert edges covering every `[0,255]` char not already covered,
/// targeting `start`. Mirrors C++ `f_add_range_edges`.
fn fill_range_gaps(edge_set: &mut BTreeSet<OrdRange>, start: usize) {
    // Sentinels at -1 and 256 bracket the range so gap detection is uniform.
    edge_set.insert(OrdRange(FsmEdge::new(-1, -1, 0)));
    edge_set.insert(OrdRange(FsmEdge::new(256, 256, 0)));

    let ordered: Vec<FsmEdge> = edge_set.iter().map(|o| o.0).collect();
    let mut to_insert: Vec<FsmEdge> = Vec::new();
    for w in ordered.windows(2) {
        let prev = w[0];
        let cur = w[1];
        if prev.max + 1 != cur.min {
            to_insert.push(FsmEdge::new(prev.max + 1, cur.min - 1, start as i32));
        }
    }
    for e in to_insert {
        edge_set.insert(OrdRange(e));
    }
    edge_set.remove(&OrdRange(FsmEdge::new(-1, -1, 0)));
    edge_set.remove(&OrdRange(FsmEdge::new(256, 256, 0)));
}

/// Wrapper that orders [`FsmEdge`] by `(min, max)` only, for the
/// `BTreeSet` used during back-edge construction (the C++ uses a
/// `std::set<FSMEdge, FSMEdgeRangeComparator>`).
#[derive(Clone, Copy)]
struct OrdRange(FsmEdge);
impl PartialEq for OrdRange {
    fn eq(&self, other: &Self) -> bool {
        cmp_edge_range(&self.0, &other.0) == std::cmp::Ordering::Equal
    }
}
impl Eq for OrdRange {}
impl PartialOrd for OrdRange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdRange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        cmp_edge_range(&self.0, &other.0)
    }
}

#[cfg(test)]
#[path = "trie_tests.rs"]
mod tests;
