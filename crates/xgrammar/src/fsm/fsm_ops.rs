// SPDX-License-Identifier: AGPL-3.0-only
//
// Regular-operation constructors on FsmWithStartEnd — Star, Plus,
// Optional, Union, Concat. Split out of `with_start_end.rs` to keep
// each file under the 250-line cap. Port of the corresponding
// `FSMWithStartEnd` methods from `cpp/fsm.cc`.

use super::fsm::Fsm;
use super::with_start_end::FsmWithStartEnd;

impl FsmWithStartEnd {
    /// `self*` — Kleene star.
    pub fn star(&self) -> FsmWithStartEnd {
        let mut fsm = self.fsm.clone();
        let new_start = fsm.add_state();
        for end in 0..self.num_states() {
            if self.is_end_state(end) {
                fsm.add_epsilon_edge(end, new_start);
            }
        }
        fsm.add_epsilon_edge(new_start, self.start);
        let mut is_end = vec![false; self.num_states() + 1];
        is_end[new_start] = true;
        FsmWithStartEnd::new(fsm, new_start, is_end, false)
    }

    /// `self+` — one or more.
    pub fn plus(&self) -> FsmWithStartEnd {
        let mut fsm = self.fsm.clone();
        for end in 0..self.num_states() {
            if self.is_end_state(end) {
                fsm.add_epsilon_edge(end, self.start);
            }
        }
        FsmWithStartEnd::new(fsm, self.start, self.ends.clone(), false)
    }

    /// `self?` — optional.
    pub fn optional(&self) -> FsmWithStartEnd {
        let mut fsm = self.fsm.clone();
        for end in 0..self.num_states() {
            if self.is_end_state(end) {
                fsm.add_epsilon_edge(self.start, end);
                break;
            }
        }
        FsmWithStartEnd::new(fsm, self.start, self.ends.clone(), false)
    }

    /// Parallel union of several FSMs — accepts any of their languages.
    pub fn union(fsms: &[FsmWithStartEnd]) -> FsmWithStartEnd {
        if fsms.len() == 1 {
            return fsms[0].clone();
        }
        debug_assert!(fsms.len() > 1, "Union of 0 FSMs is not allowed.");
        let mut fsm = Fsm::with_states(1);
        let start = 0;
        let mut ends = vec![false];
        for sub in fsms {
            let mapping = fsm.add_fsm(&sub.fsm);
            fsm.add_epsilon_edge(start, mapping[sub.start]);
            for state in 0..sub.num_states() {
                ends.push(sub.is_end_state(state));
            }
        }
        FsmWithStartEnd::new(fsm, start, ends, false)
    }

    /// Sequential concatenation of several FSMs.
    pub fn concat(fsms: &[FsmWithStartEnd]) -> FsmWithStartEnd {
        if fsms.len() == 1 {
            return fsms[0].clone();
        }
        debug_assert!(fsms.len() > 1, "Concatenation of 0 FSMs is not allowed.");
        let mut fsm = Fsm::with_states(0);
        let mut start = 0;
        let mut ends: Vec<bool> = Vec::new();
        let mut previous_ends: Vec<usize> = Vec::new();
        let last = fsms.len() - 1;
        for (i, sub) in fsms.iter().enumerate() {
            let mapping = fsm.add_fsm(&sub.fsm);
            if i == 0 {
                start = mapping[sub.start];
            } else {
                let this_start = mapping[sub.start];
                for &end in &previous_ends {
                    fsm.add_epsilon_edge(end, this_start);
                }
            }
            if i == last {
                ends = vec![false; fsm.num_states()];
                for end in 0..sub.num_states() {
                    if sub.is_end_state(end) {
                        ends[mapping[end]] = true;
                    }
                }
            } else {
                previous_ends.clear();
                for end in 0..sub.num_states() {
                    if sub.is_end_state(end) {
                        previous_ends.push(mapping[end]);
                    }
                }
            }
        }
        FsmWithStartEnd::new(fsm, start, ends, false)
    }
}
