// SPDX-License-Identifier: AGPL-3.0-only
//
// FSM language algorithms — `IsDFA`, `ToDFA`, `MinimizeDFA`, `Not`,
// `Intersect`. Port of the corresponding `FSMWithStartEnd` methods from
// `cpp/fsm.cc`.

use ahash::AHashSet;

use super::edge::edge_type;
use super::fsm::Fsm;
use super::with_start_end::FsmWithStartEnd;

/// Default cap on result-state count for DFA / minimize / not.
pub const DEFAULT_MAX_STATES: usize = 1_000_000;

impl FsmWithStartEnd {
    /// True if the FSM is a DFA: no epsilon edges and no two char-range
    /// (or rule-ref) edges from one state overlap. Caches the result.
    pub fn is_dfa(&mut self) -> bool {
        if self.is_dfa {
            return true;
        }
        for edges in self.fsm.all_edges() {
            let mut char_seen = [false; 256];
            let mut rule_seen: AHashSet<i32> = AHashSet::new();
            for edge in edges {
                if edge.is_epsilon() {
                    return false;
                }
                if edge.is_char_range() {
                    for c in edge.min..=edge.max {
                        if char_seen[c as usize] {
                            return false;
                        }
                        char_seen[c as usize] = true;
                    }
                    continue;
                }
                if edge.is_rule_ref() && !rule_seen.insert(edge.ref_rule_id()) {
                    return false;
                }
            }
        }
        self.is_dfa = true;
        true
    }

    /// Subset-construction NFA -> DFA. Returns `Err` if the result would
    /// exceed `max_num_states`.
    pub fn to_dfa(&self, max_num_states: usize) -> Result<FsmWithStartEnd, String> {
        if self.num_states() > max_num_states {
            return Err("The number of states exceeds the limit.".to_string());
        }
        let mut dfa = FsmWithStartEnd::new(Fsm::with_states(0), 0, Vec::new(), true);
        let mut closures: Vec<AHashSet<i32>> = Vec::new();

        let mut closure: AHashSet<i32> = AHashSet::from_iter([self.start as i32]);
        self.fsm.epsilon_closure(&mut closure);
        closures.push(closure);

        let mut now_process = 0;
        while now_process < closures.len() {
            let mut rules: AHashSet<i32> = AHashSet::new();
            let mut repeat_aux: AHashSet<i16> = AHashSet::new();
            let mut interval_ends: std::collections::BTreeSet<i32> =
                std::collections::BTreeSet::new();
            let mut allowed = [false; 256];
            dfa.add_state();

            for &state in &closures[now_process] {
                if self.is_end_state(state as usize) {
                    dfa.add_end_state(now_process);
                }
                for edge in self.fsm.edges(state as usize) {
                    if edge.is_char_range() {
                        interval_ends.insert(edge.min as i32);
                        interval_ends.insert(edge.max as i32 + 1);
                        for c in edge.min..=edge.max {
                            allowed[c as usize] = true;
                        }
                    } else if edge.is_rule_ref() {
                        rules.insert(edge.ref_rule_id());
                    } else if edge.is_repeat_ref() {
                        repeat_aux.insert(edge.aux_index());
                    }
                }
            }

            // Partition the char axis into maximal fully-allowed intervals.
            let mut intervals: Vec<(i32, i32)> = Vec::new();
            let mut last: i32 = -1;
            for &end in &interval_ends {
                if last == -1 {
                    last = end;
                    continue;
                }
                let ok = (last..end).all(|i| allowed[i as usize]);
                if ok {
                    intervals.push((last, end - 1));
                }
                last = end;
            }

            for (lo, hi) in intervals {
                let mut next_closure: AHashSet<i32> = AHashSet::new();
                for &state in &closures[now_process] {
                    for edge in self.fsm.edges(state as usize) {
                        if edge.is_char_range()
                            && lo >= edge.min as i32
                            && hi <= edge.max as i32
                            && !next_closure.contains(&edge.target)
                        {
                            let mut ec: AHashSet<i32> = AHashSet::from_iter([edge.target]);
                            self.fsm.epsilon_closure(&mut ec);
                            next_closure.extend(ec);
                        }
                    }
                }
                let idx = closures.iter().position(|c| *c == next_closure);
                match idx {
                    Some(j) => dfa.fsm_mut().add_edge(now_process, j, lo as i16, hi as i16),
                    None => {
                        dfa.fsm_mut()
                            .add_edge(now_process, closures.len(), lo as i16, hi as i16);
                        closures.push(next_closure);
                    }
                }
            }

            for rule in &rules {
                let next_closure = self.rule_target_closure(&closures[now_process], *rule);
                let idx = closures.iter().position(|c| *c == next_closure);
                match idx {
                    Some(j) => dfa.fsm_mut().add_rule_edge(now_process, j, *rule as i16),
                    None => {
                        dfa.fsm_mut()
                            .add_rule_edge(now_process, closures.len(), *rule as i16);
                        closures.push(next_closure);
                    }
                }
            }

            for aux_idx in &repeat_aux {
                let next_closure = self.repeat_target_closure(&closures[now_process], *aux_idx);
                let idx = closures.iter().position(|c| *c == next_closure);
                match idx {
                    Some(j) => dfa.fsm_mut().add_typed_edge(
                        now_process,
                        j,
                        edge_type::REPEAT_REF,
                        *aux_idx,
                    ),
                    None => {
                        dfa.fsm_mut().add_typed_edge(
                            now_process,
                            closures.len(),
                            edge_type::REPEAT_REF,
                            *aux_idx,
                        );
                        closures.push(next_closure);
                    }
                }
            }
            now_process += 1;
        }
        dfa.fsm_mut()
            .set_edge_aux_data(self.fsm.edge_aux_data().to_vec());
        dfa.is_dfa = true;
        Ok(dfa)
    }

    /// Epsilon closure of all rule-ref targets matching `rule`.
    fn rule_target_closure(&self, from: &AHashSet<i32>, rule: i32) -> AHashSet<i32> {
        let mut next: AHashSet<i32> = AHashSet::new();
        for &state in from {
            for edge in self.fsm.edges(state as usize) {
                if edge.is_rule_ref() && edge.ref_rule_id() == rule && !next.contains(&edge.target)
                {
                    let mut ec: AHashSet<i32> = AHashSet::from_iter([edge.target]);
                    self.fsm.epsilon_closure(&mut ec);
                    next.extend(ec);
                }
            }
        }
        next
    }

    /// Epsilon closure of all repeat-ref targets at `aux_idx`.
    fn repeat_target_closure(&self, from: &AHashSet<i32>, aux_idx: i16) -> AHashSet<i32> {
        let mut next: AHashSet<i32> = AHashSet::new();
        for &state in from {
            for edge in self.fsm.edges(state as usize) {
                if edge.is_repeat_ref()
                    && edge.aux_index() == aux_idx
                    && !next.contains(&edge.target)
                {
                    let mut ec: AHashSet<i32> = AHashSet::from_iter([edge.target]);
                    self.fsm.epsilon_closure(&mut ec);
                    next.extend(ec);
                }
            }
        }
        next
    }
}

#[cfg(test)]
#[path = "algorithms_tests.rs"]
mod tests;
