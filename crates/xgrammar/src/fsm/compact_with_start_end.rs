// SPDX-License-Identifier: AGPL-3.0-only
//
// CompactFsmWithStartEnd — the compact-FSM counterpart of
// FsmWithStartEnd. Split out of `with_start_end.rs` to keep each
// file under the 250-line cap. Port of `CompactFSMWithStartEnd`
// from `cpp/fsm.{h,cc}`.

use ahash::AHashSet;

use super::compact::CompactFsm;
use super::edge::edge_type;
use super::with_start_end::FsmWithStartEnd;

/// A compact FSM paired with start/accepting states.
///
/// `node_num` / `edge_num` record the size of the *logical* FSM this
/// value represents. For a standalone FSM that equals the backing
/// `fsm`'s own counts; for a per-rule view spliced into a shared
/// `complete_fsm`, they record the sub-FSM's size — not the whole
/// completed FSM. Distinguishing the two is the upstream #600 fix
/// (`CompactFSMWithStartEndWithSize`, commit 58494db).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactFsmWithStartEnd {
    pub(crate) fsm: CompactFsm,
    pub(crate) start: usize,
    pub(crate) ends: Vec<bool>,
    pub(crate) is_dfa: bool,
    node_num: usize,
    edge_num: usize,
}

impl CompactFsmWithStartEnd {
    /// Build a standalone compact FSM; `node_num`/`edge_num` are the
    /// FSM's own counts.
    pub fn new(fsm: CompactFsm, start: usize, ends: Vec<bool>) -> Self {
        let node_num = fsm.num_states();
        let edge_num = fsm.num_edges();
        Self {
            fsm,
            start,
            ends,
            is_dfa: false,
            node_num,
            edge_num,
        }
    }

    /// Build a per-rule view onto a shared `complete_fsm`. `node_num`
    /// and `edge_num` are the spliced-in sub-FSM's counts, *not* the
    /// backing FSM's totals (upstream commit 58494db, #600).
    pub fn new_view(
        fsm: CompactFsm,
        start: usize,
        ends: Vec<bool>,
        node_num: usize,
        edge_num: usize,
    ) -> Self {
        Self {
            fsm,
            start,
            ends,
            is_dfa: false,
            node_num,
            edge_num,
        }
    }

    /// The underlying compact FSM.
    pub fn fsm(&self) -> &CompactFsm {
        &self.fsm
    }

    /// Start state id.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Accepting-state bitmap.
    pub fn ends(&self) -> &[bool] {
        &self.ends
    }

    /// True if `state` is accepting.
    pub fn is_end_state(&self, state: usize) -> bool {
        self.ends[state]
    }

    /// Number of states in the logical (sub-)FSM. For a per-rule view
    /// this is the spliced-in sub-FSM's count, not the backing
    /// `complete_fsm`'s total (upstream commit 58494db, #600).
    pub fn num_states(&self) -> usize {
        self.node_num
    }

    /// Number of edges in the logical (sub-)FSM. For a per-rule view
    /// this is the spliced-in sub-FSM's count, not the backing
    /// `complete_fsm`'s total (upstream commit 58494db, #600).
    pub fn num_edges(&self) -> usize {
        self.edge_num
    }

    /// Number of states in the backing FSM storage — for a per-rule
    /// view this is the whole shared `complete_fsm`. Use this (not
    /// `num_states`) when indexing into the backing edge table.
    pub fn backing_num_states(&self) -> usize {
        self.fsm.num_states()
    }

    /// True if the FSM accepts the byte string.
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

    /// Expand into the mutable form.
    pub fn to_fsm(&self) -> FsmWithStartEnd {
        FsmWithStartEnd::new(
            self.fsm.to_fsm(),
            self.start,
            self.ends.clone(),
            self.is_dfa,
        )
    }

    /// Approximate heap footprint.
    pub fn memory_size(&self) -> usize {
        self.fsm.memory_size() + self.ends.len()
    }
}
