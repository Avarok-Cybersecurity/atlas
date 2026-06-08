// SPDX-License-Identifier: AGPL-3.0-only
//
// Traversal methods on the mutable `Fsm` — transition, epsilon
// closure, set-advance, reachability. Split out of `fsm.rs` to keep
// each file under the 250-line cap. The heavy lifting lives in the
// shared `traversal` module; these are thin typed wrappers.

use ahash::AHashSet;
use std::collections::VecDeque;

use super::fsm::Fsm;
use super::traversal;

impl Fsm {
    /// Single-character / typed transition. Returns the first matching
    /// target, or `NO_NEXT_STATE`. `edge_type` must not be epsilon.
    pub fn next_state(&self, from: usize, value: i32, edge_type: i16) -> i32 {
        traversal::next_state(&self.edges, from, value, edge_type)
    }

    /// Epsilon closure (in place); `state_set` is *not* cleared first.
    pub fn epsilon_closure(&self, state_set: &mut AHashSet<i32>) {
        traversal::epsilon_closure(&self.edges, state_set);
    }

    /// Advance a set of states by `value` along `edge_type` edges.
    /// `result` is cleared first; the epsilon closure of the result is
    /// included. Set `from_is_closure` if `from` is already closed.
    pub fn advance(
        &self,
        from: &AHashSet<i32>,
        value: i32,
        result: &mut AHashSet<i32>,
        edge_type: i16,
        from_is_closure: bool,
    ) {
        traversal::advance(&self.edges, from, value, result, edge_type, from_is_closure);
    }

    /// All rule ids referenced by `state`'s outgoing edges.
    pub fn possible_rules(&self, state: usize, rules: &mut AHashSet<i32>) {
        rules.clear();
        for edge in &self.edges[state] {
            if edge.is_rule_ref() {
                rules.insert(edge.ref_rule_id());
            }
        }
    }

    /// All states reachable from `from` (including `from`). `result` is
    /// cleared first.
    pub fn reachable_states(&self, from: &[i32], result: &mut AHashSet<i32>) {
        result.clear();
        let mut queue: VecDeque<i32> = VecDeque::new();
        for &s in from {
            if result.insert(s) {
                queue.push_back(s);
            }
        }
        while let Some(cur) = queue.pop_front() {
            for edge in &self.edges[cur as usize] {
                if result.insert(edge.target) {
                    queue.push_back(edge.target);
                }
            }
        }
    }
}
