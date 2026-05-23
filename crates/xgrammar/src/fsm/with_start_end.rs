// SPDX-License-Identifier: AGPL-3.0-only
//
// FsmWithStartEnd / CompactFsmWithStartEnd — an FSM paired with a start
// state and a set of accepting states. Port of `FSMWithStartEndBase`,
// `FSMWithStartEnd` and `CompactFSMWithStartEnd` from `cpp/fsm.{h,cc}`.
//
// The C++ uses a CRTP base template parameterized over `FSM`/`CompactFSM`.
// Rust has no CRTP; instead this file holds the mutable `FsmWithStartEnd`
// with its rich construction API, and a lighter `CompactFsmWithStartEnd`.
// Shared accepting-state logic is provided by free helpers.

use ahash::AHashSet;

use super::edge::edge_type;
use super::fsm::Fsm;

/// A mutable FSM together with a start state and accepting states.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FsmWithStartEnd {
    pub(crate) fsm: Fsm,
    pub(crate) start: usize,
    /// `ends[s]` is true when state `s` is accepting.
    pub(crate) ends: Vec<bool>,
    /// Cached DFA flag — true once the FSM is known deterministic.
    pub(crate) is_dfa: bool,
}

impl FsmWithStartEnd {
    /// Build from parts.
    pub fn new(fsm: Fsm, start: usize, ends: Vec<bool>, is_dfa: bool) -> Self {
        Self {
            fsm,
            start,
            ends,
            is_dfa,
        }
    }

    /* --------------------- accessors --------------------- */

    /// The underlying FSM.
    pub fn fsm(&self) -> &Fsm {
        &self.fsm
    }

    /// Mutable underlying FSM.
    pub fn fsm_mut(&mut self) -> &mut Fsm {
        &mut self.fsm
    }

    /// The start state id.
    pub fn start(&self) -> usize {
        self.start
    }

    /// The accepting-state bitmap.
    pub fn ends(&self) -> &[bool] {
        &self.ends
    }

    /// True if `state` is accepting.
    pub fn is_end_state(&self, state: usize) -> bool {
        self.ends[state]
    }

    /// Total number of states.
    pub fn num_states(&self) -> usize {
        self.fsm.num_states()
    }

    /// Set the start state.
    pub fn set_start_state(&mut self, state: usize) {
        debug_assert!(state < self.num_states());
        self.start = state;
    }

    /// Mark `state` accepting.
    pub fn add_end_state(&mut self, state: usize) {
        debug_assert!(state < self.num_states());
        self.ends[state] = true;
    }

    /// Replace the accepting-state bitmap.
    pub fn set_end_states(&mut self, ends: Vec<bool>) {
        self.ends = ends;
    }

    /// Append a new (non-accepting) state; returns its id.
    pub fn add_state(&mut self) -> usize {
        self.ends.push(false);
        self.fsm.add_state()
    }

    /// True if `state` has at least one outgoing char-range edge.
    pub fn is_scanable_state(&self, state: usize) -> bool {
        self.fsm.edges(state).iter().any(|e| e.is_char_range())
    }

    /// True if `state` has a rule-ref / epsilon / repeat-ref edge.
    pub fn is_non_terminal_state(&self, state: usize) -> bool {
        self.fsm
            .edges(state)
            .iter()
            .any(|e| e.is_rule_ref() || e.is_epsilon() || e.is_repeat_ref())
    }

    /* --------------------- traversal --------------------- */

    /// True if the FSM accepts the byte string `str`.
    pub fn accept_string(&self, bytes: &[u8]) -> bool {
        let mut start_states: AHashSet<i32> = AHashSet::from_iter([self.start as i32]);
        self.fsm.epsilon_closure(&mut start_states);
        let mut result_states = AHashSet::new();
        for &byte in bytes {
            result_states.clear();
            self.fsm.advance(
                &start_states,
                byte as i32,
                &mut result_states,
                edge_type::CHAR_RANGE,
                false,
            );
            if result_states.is_empty() {
                return false;
            }
            std::mem::swap(&mut start_states, &mut result_states);
        }
        start_states.iter().any(|&s| self.ends[s as usize])
    }

    /// All states reachable from the start state.
    pub fn reachable_states(&self, result: &mut AHashSet<i32>) {
        self.fsm.reachable_states(&[self.start as i32], result);
    }

    /// True if no reachable state has a rule-ref or repeat-ref edge.
    pub fn is_leaf(&self) -> bool {
        let mut reachable = AHashSet::new();
        self.reachable_states(&mut reachable);
        for &s in &reachable {
            for edge in self.fsm.edges(s as usize) {
                if edge.is_rule_ref() || edge.is_repeat_ref() {
                    return false;
                }
            }
        }
        true
    }

    /* --------------------- construction --------------------- */

    /// A deep copy.
    pub fn copy(&self) -> FsmWithStartEnd {
        self.clone()
    }

    /// Rebuild with remapped state ids.
    pub fn rebuild_with_mapping(
        &self,
        state_mapping: &[usize],
        new_num_states: usize,
    ) -> FsmWithStartEnd {
        let new_fsm = self.fsm.rebuild_with_mapping(state_mapping, new_num_states);
        let new_start = state_mapping[self.start];
        let mut new_ends = vec![false; new_num_states];
        for end in 0..self.num_states() {
            if self.is_end_state(end) {
                new_ends[state_mapping[end]] = true;
            }
        }
        FsmWithStartEnd::new(new_fsm, new_start, new_ends, false)
    }

    /// Splice this FSM into `complete_fsm`, returning a view onto it.
    pub fn add_to_complete_fsm(&self, complete_fsm: &mut Fsm) -> FsmWithStartEnd {
        let state_mapping = complete_fsm.add_fsm(&self.fsm);
        let new_start = state_mapping[self.start];
        let mut new_ends = vec![false; complete_fsm.num_states()];
        for end in 0..self.num_states() {
            if self.is_end_state(end) {
                new_ends[state_mapping[end]] = true;
            }
        }
        FsmWithStartEnd::new(complete_fsm.clone(), new_start, new_ends, self.is_dfa)
    }

    /// Pack into the compact form.
    pub fn to_compact(&self) -> CompactFsmWithStartEnd {
        CompactFsmWithStartEnd::new(self.fsm.to_compact(), self.start, self.ends.clone())
    }
}

// Star / Plus / Optional / Union / Concat live in `fsm_ops.rs`;
// CompactFsmWithStartEnd lives in `compact_with_start_end.rs`.
pub use super::compact_with_start_end::CompactFsmWithStartEnd;

#[cfg(test)]
#[path = "with_start_end_tests.rs"]
mod tests;
