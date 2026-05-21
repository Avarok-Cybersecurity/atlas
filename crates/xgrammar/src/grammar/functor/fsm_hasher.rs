// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarFsmHasher — port of `GrammarFSMHasherImpl` from
// `cpp/grammar_functor.cc`.
//
// Computes a structural hash for each rule's FSM (used by the matcher's
// rule-level token-mask cache) plus a canonical state renumbering.
// Hashing proceeds from terminal FSMs inward, breaking simple cycles.

use crate::grammar::data::GrammarData;
use crate::grammar::expr::GrammarExprType;
use crate::support::hash::hash_combine_binary;

use super::analyzer::RuleRefGraphFinder;
use super::hash_walk::{flag, hash_fsm};

/// Per-rule FSM structural hashing pass.
pub struct GrammarFsmHasher {
    visited: Vec<bool>,
    /// `referrer_to_referee[r]` lists the rules that rule `r` references.
    referrer_to_referee: Vec<Vec<i32>>,
    /// `referee_to_referrer[r]` lists the rules that reference rule `r`.
    referee_to_referrer: Vec<Vec<i32>>,
}

impl GrammarFsmHasher {
    /// Build `per_rule_fsm_hashes` and `per_rule_fsm_new_state_ids`.
    pub fn apply(grammar: &mut GrammarData) {
        let num_rules = grammar.num_rules() as usize;
        grammar.per_rule_fsm_hashes = vec![None; num_rules];
        grammar.per_rule_fsm_new_state_ids = vec![Vec::new(); num_rules];

        let referee_to_referrer = RuleRefGraphFinder::apply(grammar);
        let mut referrer_to_referee: Vec<Vec<i32>> = vec![Vec::new(); num_rules];
        for (referee, referrers) in referee_to_referrer.iter().enumerate() {
            for &referrer in referrers {
                referrer_to_referee[referrer as usize].push(referee as i32);
            }
        }

        let mut visited = vec![false; num_rules];
        for (i, fsm) in grammar.per_rule_fsms.iter().enumerate() {
            if fsm.is_none() {
                visited[i] = true;
            }
        }

        let mut hasher = Self {
            visited,
            referrer_to_referee,
            referee_to_referrer,
        };

        // Hash terminal / self-recursive FSMs first, breaking cycles.
        loop {
            let idx = hasher.find_terminal();
            if idx != -1 {
                hasher.visited[idx as usize] = true;
                let (h, mapping) = hash_fsm(grammar, idx);
                grammar.per_rule_fsm_hashes[idx as usize] = Some(h);
                grammar.per_rule_fsm_new_state_ids[idx as usize] = mapping;
                hasher.detach(idx);
                continue;
            }
            if !hasher.break_one_cycle(grammar) {
                break;
            }
        }

        // Remaining FSMs: partial-hash those whose start has no inward
        // edge and whose unhashed rule refs are all at the start state.
        let mut partial: Vec<(i32, u64)> = Vec::new();
        for i in 0..num_rules as i32 {
            if grammar.per_rule_fsm_hashes[i as usize].is_some() {
                continue;
            }
            let Some(fsm) = &grammar.per_rule_fsms[i as usize] else {
                continue;
            };
            if has_inward_edge(grammar, fsm.start()) {
                continue;
            }
            if let Some((h, mapping)) = super::hash_walk::partial_hash_fsm(grammar, i) {
                grammar.per_rule_fsm_new_state_ids[i as usize] = mapping;
                partial.push((i, h));
            }
        }
        for (rule_id, h) in partial {
            grammar.per_rule_fsm_hashes[rule_id as usize] = Some(h);
        }
    }

    /// Find a terminal (no outgoing refs) or self-recursive FSM that can
    /// be hashed now. `-1` if none.
    fn find_terminal(&self) -> i32 {
        for i in 0..self.referrer_to_referee.len() {
            if self.visited[i] {
                continue;
            }
            if self.referrer_to_referee[i].is_empty() {
                return i as i32;
            }
            if self.referrer_to_referee[i].len() == 1 && self.referrer_to_referee[i][0] == i as i32
            {
                return i as i32;
            }
        }
        -1
    }

    /// Detach `idx` from the reference graph.
    fn detach(&mut self, idx: i32) {
        for referer in self.referee_to_referrer[idx as usize].clone() {
            self.referrer_to_referee[referer as usize].retain(|&r| r != idx);
        }
    }

    /// Find one simple cycle, hash its members, detach it. Returns
    /// `false` when no cycle remains.
    fn break_one_cycle(&mut self, grammar: &mut GrammarData) -> bool {
        let n = self.referee_to_referrer.len();
        let mut not_simple = self.visited.clone();
        for i in 0..n {
            if not_simple[i] {
                continue;
            }
            let mut dfs_stack: Vec<i32> = vec![i as i32];
            let mut in_stack = vec![false; n];
            in_stack[i] = true;
            let mut cur = i as i32;
            let mut cycle: Vec<i32> = Vec::new();
            while self.referrer_to_referee[cur as usize].len() == 1 && !not_simple[cur as usize] {
                assert_ne!(
                    cur, self.referrer_to_referee[cur as usize][0],
                    "self-recursion cycle not allowed"
                );
                not_simple[cur as usize] = true;
                cur = self.referrer_to_referee[cur as usize][0];
                if in_stack[cur as usize] {
                    cycle.push(cur);
                    while *dfs_stack.last().unwrap() != cur {
                        cycle.push(dfs_stack.pop().unwrap());
                    }
                    break;
                }
                dfs_stack.push(cur);
                in_stack[cur as usize] = true;
            }
            if !cycle.is_empty() {
                self.hash_cycle(grammar, &cycle);
                return true;
            }
        }
        false
    }

    /// Hash a simple cycle: hash each member, rotation-combine, detach.
    fn hash_cycle(&mut self, grammar: &mut GrammarData, cycle: &[i32]) {
        for &id in cycle {
            self.visited[id as usize] = true;
            grammar.per_rule_fsm_hashes[id as usize] = Some(flag::SIMPLE_CYCLE);
        }
        let mut local: Vec<u64> = Vec::with_capacity(cycle.len());
        for &id in cycle {
            let (h, mapping) = hash_fsm(grammar, id);
            grammar.per_rule_fsm_new_state_ids[id as usize] = mapping;
            local.push(h);
        }
        let len = local.len();
        for (i, &id) in cycle.iter().enumerate() {
            let mut seed = 0u64;
            for j in 0..len {
                hash_combine_binary(&mut seed, local[(i + j) % len]);
            }
            grammar.per_rule_fsm_hashes[id as usize] = Some(seed);
        }
        for &id in cycle {
            self.detach(id);
        }
    }
}

/// True if any state in the complete FSM has an edge into `state`.
fn has_inward_edge(grammar: &GrammarData, state: usize) -> bool {
    let fsm = &grammar.complete_fsm;
    for s in 0..fsm.num_states() {
        for edge in fsm.edges(s) {
            if edge.target as usize == state {
                return true;
            }
        }
    }
    false
}

/// Hash a normalized `Sequence` expr — used by the matcher to key its
/// per-sequence cache. `None` when the sequence is not hashable.
pub fn hash_sequence(grammar: &GrammarData, sequence_id: i32) -> Option<u64> {
    if sequence_id == -1 {
        return None;
    }
    let seq = grammar.expr(sequence_id);
    debug_assert_eq!(seq.kind, GrammarExprType::Sequence);
    let mut seed = 0u64;
    for &eid in seq.data {
        let e = grammar.expr(eid);
        hash_combine_binary(&mut seed, e.kind as i32 as u64);
        match e.kind {
            GrammarExprType::ByteString
            | GrammarExprType::CharacterClass
            | GrammarExprType::CharacterClassStar
            | GrammarExprType::EmptyStr => {
                for &x in e.data {
                    hash_combine_binary(&mut seed, x as i64 as u64);
                }
            }
            GrammarExprType::RuleRef => {
                let h = grammar.per_rule_fsm_hashes[e.data[0] as usize]?;
                hash_combine_binary(&mut seed, h);
            }
            GrammarExprType::Repeat => {
                let h = grammar.per_rule_fsm_hashes[e.data[0] as usize]?;
                hash_combine_binary(&mut seed, h);
                hash_combine_binary(&mut seed, e.data[1] as i64 as u64);
                hash_combine_binary(&mut seed, e.data[2] as i64 as u64);
            }
            GrammarExprType::Sequence | GrammarExprType::Choices => return None,
            GrammarExprType::TagDispatch => return None,
        }
    }
    Some(seed)
}

#[cfg(test)]
#[path = "fsm_hasher_tests.rs"]
mod tests;
