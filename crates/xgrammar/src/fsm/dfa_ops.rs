// SPDX-License-Identifier: AGPL-3.0-only
//
// DFA-level operations — MinimizeDFA, Not, Intersect. Split out of
// `algorithms.rs` to keep each file under the 250-line cap; together
// they port the corresponding `FSMWithStartEnd` methods from
// `cpp/fsm.cc`.

use ahash::{AHashMap, AHashSet};
use std::collections::VecDeque;

use super::algorithms::DEFAULT_MAX_STATES;
use super::fsm::Fsm;
use super::with_start_end::FsmWithStartEnd;

impl FsmWithStartEnd {
    /// Hopcroft DFA minimization. Returns `Err` over the state cap.
    pub fn minimize_dfa(&self, max_num_states: usize) -> Result<FsmWithStartEnd, String> {
        if self.num_states() > max_num_states {
            return Err("The number of states exceeds the limit.".to_string());
        }
        let now_fsm = if self.is_dfa {
            self.copy()
        } else {
            self.to_dfa(max_num_states)?
        };
        let n = now_fsm.num_states();

        // precursors[t] = list of ((min,max), source)
        let mut precursors: Vec<Vec<((i16, i16), usize)>> = vec![Vec::new(); n];
        for i in 0..n {
            for edge in now_fsm.fsm().edges(i) {
                precursors[edge.target as usize].push(((edge.min, edge.max), i));
            }
        }

        let mut final_states: AHashSet<i32> = AHashSet::new();
        let mut non_final: AHashSet<i32> = AHashSet::new();
        for i in 0..n {
            if now_fsm.is_end_state(i) {
                final_states.insert(i as i32);
            } else {
                non_final.insert(i as i32);
            }
        }
        let mut partitions: Vec<AHashSet<i32>> = vec![final_states.clone(), non_final.clone()];
        let mut working: Vec<AHashSet<i32>> = vec![final_states, non_final];

        while let Some(current) = working.pop() {
            // possible_transitions[(min,max)] = set of precursor states
            let mut possible: AHashMap<(i16, i16), AHashSet<i32>> = AHashMap::new();
            for &state in &current {
                for &(label, src) in &precursors[state as usize] {
                    possible.entry(label).or_default().insert(src as i32);
                }
            }
            for (_label, precs) in possible {
                let mut i = 0;
                while i < partitions.len() {
                    let intersection: Vec<i32> = partitions[i]
                        .iter()
                        .copied()
                        .filter(|s| precs.contains(s))
                        .collect();
                    let difference: Vec<i32> = partitions[i]
                        .iter()
                        .copied()
                        .filter(|s| !precs.contains(s))
                        .collect();
                    if !intersection.is_empty() && !difference.is_empty() {
                        let part = partitions[i].clone();
                        let in_working = working.iter().position(|w| *w == part);
                        match in_working {
                            Some(wi) => {
                                working[wi] = intersection.iter().copied().collect();
                                working.push(difference.iter().copied().collect());
                            }
                            None => {
                                let smaller = if difference.len() < intersection.len() {
                                    &difference
                                } else {
                                    &intersection
                                };
                                working.push(smaller.iter().copied().collect());
                            }
                        }
                        partitions[i] = intersection.iter().copied().collect();
                        partitions.push(difference.iter().copied().collect());
                    }
                    i += 1;
                }
            }
        }

        let mut state_mapping = vec![0usize; n];
        for (i, part) in partitions.iter().enumerate() {
            for &state in part {
                state_mapping[state as usize] = i;
            }
        }
        Ok(now_fsm.rebuild_with_mapping(&state_mapping, partitions.len()))
    }

    /// The complement of the FSM's language. Only valid for leaf FSMs
    /// (no rule references).
    pub fn not(&self, max_result_num_states: usize) -> Result<FsmWithStartEnd, String> {
        if !self.is_leaf() {
            return Err("Not operation is not supported for FSM with rule references.".to_string());
        }
        let mut result = if self.is_dfa {
            self.copy()
        } else {
            self.to_dfa(max_result_num_states)?
        };

        let mut new_final = vec![false; result.num_states() + 1];
        for i in 0..result.num_states() {
            if !result.is_end_state(i) {
                new_final[i] = true;
            }
        }
        let accept_all = result.add_state();
        new_final[accept_all] = true;

        for i in 0..result.num_states() {
            if i == accept_all {
                continue;
            }
            let mut char_set = [false; 256];
            for edge in result.fsm().edges(i) {
                if edge.is_char_range() {
                    for c in edge.min..=edge.max {
                        char_set[c as usize] = true;
                    }
                }
            }
            let mut left = 0usize;
            while left < 256 {
                if char_set[left] {
                    left += 1;
                    continue;
                }
                let mut right = left + 1;
                while right < 256 && !char_set[right] {
                    right += 1;
                }
                result
                    .fsm_mut()
                    .add_edge(i, accept_all, left as i16, (right - 1) as i16);
                left = right;
            }
        }
        result.set_end_states(new_final);
        Ok(result)
    }

    /// Intersection of two leaf FSMs (product construction over DFAs).
    pub fn intersect(
        lhs: &FsmWithStartEnd,
        rhs: &FsmWithStartEnd,
        _max_result_num_states: usize,
    ) -> Result<FsmWithStartEnd, String> {
        if !lhs.is_leaf() || !rhs.is_leaf() {
            return Err("Intersect only support leaf fsm!".to_string());
        }
        let lhs_dfa = lhs.to_dfa(DEFAULT_MAX_STATES)?;
        let rhs_dfa = rhs.to_dfa(DEFAULT_MAX_STATES)?;

        let mut result = FsmWithStartEnd::new(Fsm::with_states(0), 0, Vec::new(), true);
        let mut state_map: AHashMap<(usize, usize), usize> = AHashMap::new();
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

        queue.push_back((lhs_dfa.start(), rhs_dfa.start()));
        result.add_state();
        state_map.insert((lhs_dfa.start(), rhs_dfa.start()), 0);

        while let Some((ls, rs)) = queue.pop_front() {
            if lhs_dfa.is_end_state(ls) && rhs_dfa.is_end_state(rs) {
                let id = state_map[&(ls, rs)];
                result.add_end_state(id);
            }
            for le in lhs_dfa.fsm().edges(ls) {
                for re in rhs_dfa.fsm().edges(rs) {
                    if !le.is_char_range() || !re.is_char_range() {
                        continue;
                    }
                    if le.min > re.max || re.min > le.max {
                        continue;
                    }
                    let min_v = le.min.max(re.min);
                    let max_v = le.max.min(re.max);
                    let key = (le.target as usize, re.target as usize);
                    let target = match state_map.get(&key) {
                        Some(&t) => t,
                        None => {
                            let t = result.add_state();
                            state_map.insert(key, t);
                            queue.push_back(key);
                            t
                        }
                    };
                    let src = state_map[&(ls, rs)];
                    result.fsm_mut().add_edge(src, target, min_v, max_v);
                }
            }
        }
        Ok(result)
    }
}
