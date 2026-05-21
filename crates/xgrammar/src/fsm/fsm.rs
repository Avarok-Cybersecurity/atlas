// SPDX-License-Identifier: AGPL-3.0-only
//
// Fsm — a mutable finite-state machine (NFA or DFA).
// Port of `class FSM` (+ `FSM::Impl`) from xgrammar `cpp/fsm.{h,cc}`.
//
// Representation: an adjacency list `edges[state] = Vec<FsmEdge>`, plus a
// flat `edge_aux_data: Vec<i32>` holding `[rule_id, lower, upper]` triples
// for repeat-reference edges. The C++ pimpl/shared_ptr indirection is
// dropped — Rust callers `clone()` explicitly when they need a copy.

use super::edge::{FsmEdge, RepeatEdgeRef, edge_type};

/// Returned by [`Fsm::next_state`] when no transition exists.
pub const NO_NEXT_STATE: i32 = -1;

/// A mutable finite-state machine.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Fsm {
    /// `edges[s]` is the list of outgoing edges from state `s`.
    pub(crate) edges: Vec<Vec<FsmEdge>>,
    /// Flat auxiliary data for repeat-reference edges.
    pub(crate) edge_aux_data: Vec<i32>,
}

impl Fsm {
    /// Create an FSM with `num_states` states and no edges.
    pub fn with_states(num_states: usize) -> Self {
        Self {
            edges: vec![Vec::new(); num_states],
            edge_aux_data: Vec::new(),
        }
    }

    /// Create an FSM from a ready-made adjacency list.
    pub fn from_edges(edges: Vec<Vec<FsmEdge>>, edge_aux_data: Vec<i32>) -> Self {
        Self {
            edges,
            edge_aux_data,
        }
    }

    /* ----------------------- visitors ----------------------- */

    /// Number of states.
    pub fn num_states(&self) -> usize {
        self.edges.len()
    }

    /// Total number of edges across all states.
    pub fn num_edges(&self) -> usize {
        self.edges.iter().map(Vec::len).sum()
    }

    /// All adjacency rows.
    pub fn all_edges(&self) -> &[Vec<FsmEdge>] {
        &self.edges
    }

    /// Outgoing edges of `state`.
    pub fn edges(&self, state: usize) -> &[FsmEdge] {
        &self.edges[state]
    }

    /// Mutable outgoing edges of `state`.
    pub fn edges_mut(&mut self, state: usize) -> &mut Vec<FsmEdge> {
        &mut self.edges[state]
    }

    /// The repeat-edge auxiliary data buffer.
    pub fn edge_aux_data(&self) -> &[i32] {
        &self.edge_aux_data
    }

    /// Replace the repeat-edge auxiliary data buffer.
    pub fn set_edge_aux_data(&mut self, data: Vec<i32>) {
        self.edge_aux_data = data;
    }

    /// Decode the repeat-edge info stored at aux index `idx`.
    pub fn repeat_edge_info(&self, idx: i16) -> RepeatEdgeRef {
        RepeatEdgeRef::from_aux(&self.edge_aux_data, idx as usize)
    }

    /* ----------------------- mutators ----------------------- */

    /// Append a fresh state; returns its id.
    pub fn add_state(&mut self) -> usize {
        self.edges.push(Vec::new());
        self.edges.len() - 1
    }

    /// Add a character-range edge `from -[min,max]-> to`.
    pub fn add_edge(&mut self, from: usize, to: usize, min: i16, max: i16) {
        debug_assert!(from < self.edges.len());
        self.edges[from].push(FsmEdge::new(min, max, to as i32));
    }

    /// Add a raw typed edge (`type` is one of [`edge_type`], `value`
    /// goes into `max`).
    pub fn add_typed_edge(&mut self, from: usize, to: usize, edge_type: i16, value: i16) {
        self.add_edge(from, to, edge_type, value);
    }

    /// Add an epsilon transition.
    pub fn add_epsilon_edge(&mut self, from: usize, to: usize) {
        self.add_typed_edge(from, to, edge_type::EPSILON, 0);
    }

    /// Add a rule-reference edge for `rule_id`.
    pub fn add_rule_edge(&mut self, from: usize, to: usize, rule_id: i16) {
        self.add_typed_edge(from, to, edge_type::RULE_REF, rule_id);
    }

    /// Add an EOS edge.
    pub fn add_eos_edge(&mut self, from: usize, to: usize) {
        self.add_typed_edge(from, to, edge_type::EOS, 0);
    }

    /// Add a repeat-reference edge, allocating `[rule_id, lower, upper]`
    /// into `edge_aux_data`. The source state must have no other edges.
    pub fn add_repeat_edge(
        &mut self,
        from: usize,
        to: usize,
        rule_id: i32,
        lower: i32,
        upper: i32,
    ) {
        debug_assert!(
            self.edges[from].is_empty(),
            "A state with a kRepeatRef edge must have no other outgoing edges."
        );
        let aux_index = self.edge_aux_data.len() as i16;
        self.edge_aux_data.push(rule_id);
        self.edge_aux_data.push(lower);
        self.edge_aux_data.push(upper);
        self.add_typed_edge(from, to, edge_type::REPEAT_REF, aux_index);
    }

    /// Splice `other` into `self`, offsetting all of `other`'s state ids.
    /// Returns the state mapping `other_id -> new_id`.
    pub fn add_fsm(&mut self, other: &Fsm) -> Vec<usize> {
        let old_num_states = self.num_states();
        let aux_offset = self.edge_aux_data.len() as i16;
        self.edge_aux_data.extend_from_slice(&other.edge_aux_data);

        let state_mapping: Vec<usize> = (0..other.num_states())
            .map(|i| i + old_num_states)
            .collect();

        self.edges
            .resize(self.edges.len() + other.num_states(), Vec::new());

        for (i, row) in other.edges.iter().enumerate() {
            for edge in row {
                let max_val = if edge.is_aux_edge() && aux_offset > 0 {
                    edge.max + aux_offset
                } else {
                    edge.max
                };
                self.add_edge(
                    i + old_num_states,
                    edge.target as usize + old_num_states,
                    edge.min,
                    max_val,
                );
            }
        }
        state_mapping
    }

    /* --------------------- construction ---------------------- */

    /// Sort every state's edges by `(min, max, target)`.
    pub fn sort_edges(&mut self) {
        for row in &mut self.edges {
            row.sort_unstable();
        }
    }

    /// Rebuild with remapped state ids. Epsilon self-loops created by the
    /// mapping are dropped; duplicate edges per row are removed.
    pub fn rebuild_with_mapping(&self, state_mapping: &[usize], new_num_states: usize) -> Fsm {
        let mut new_edges: Vec<Vec<FsmEdge>> = vec![Vec::new(); new_num_states];
        for (i, row) in self.edges.iter().enumerate() {
            for edge in row {
                if edge.is_epsilon() && state_mapping[i] == state_mapping[edge.target as usize] {
                    continue; // skip epsilon self-loops
                }
                new_edges[state_mapping[i]].push(FsmEdge::new(
                    edge.min,
                    edge.max,
                    state_mapping[edge.target as usize] as i32,
                ));
            }
        }
        for row in &mut new_edges {
            row.sort_unstable();
            row.dedup();
        }
        Fsm::from_edges(new_edges, self.edge_aux_data.clone())
    }
}

// Traversal methods (next_state / epsilon_closure / advance /
// possible_rules / reachable_states) live in `fsm_traversal.rs`.

#[cfg(test)]
#[path = "fsm_tests.rs"]
mod tests;
