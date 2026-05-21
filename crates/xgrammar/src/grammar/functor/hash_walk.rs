// SPDX-License-Identifier: AGPL-3.0-only
//
// Per-FSM BFS hashing — the `HashFsm` / `IsPartialHashable` helpers of
// `GrammarFSMHasherImpl` (`cpp/grammar_functor.cc`).
//
// Each rule's FSM is walked breadth-first; states are renumbered in
// discovery order and every edge (char range / rule ref / repeat ref)
// is folded into a structural hash.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::grammar::data::GrammarData;
use crate::support::hash::hash_combine_binary;

/// Sentinel flag values mixed into structural hashes.
pub mod flag {
    /// State is not accepting.
    pub const NOT_END: u64 = (-0x100_i64) as u64;
    /// State is accepting.
    pub const END: u64 = (-0x200_i64) as u64;
    /// A self-recursion rule-ref edge.
    pub const SELF_RECURSION: u64 = (-0x300_i64) as u64;
    /// Placeholder hash for a rule on a simple cycle.
    pub const SIMPLE_CYCLE: u64 = (-0x400_i64) as u64;
    /// An as-yet-unhashed referenced rule (only valid at the start).
    pub const UNKNOWN: u64 = (-0x500_i64) as u64;
}

/// Mix several values into `seed` (incremental `HashCombine`).
fn mix(seed: u64, values: &[u64]) -> u64 {
    let mut s = seed;
    for &v in values {
        hash_combine_binary(&mut s, v);
    }
    s
}

/// Sorted edges of complete-FSM state `s` (rule/repeat/char all together).
fn sorted_edges(grammar: &GrammarData, s: usize) -> Vec<crate::fsm::FsmEdge> {
    let mut e: Vec<_> = grammar.complete_fsm.edges(s).to_vec();
    e.sort_unstable();
    e
}

/// Hash a fully-hashable FSM (all referenced rules already hashed, or
/// self-recursive). Returns `(hash, new_state_id_mapping)`.
pub fn hash_fsm(grammar: &GrammarData, fsm_index: i32) -> (u64, Vec<(i32, i32)>) {
    let fsm = grammar.per_rule_fsms[fsm_index as usize]
        .as_ref()
        .expect("fsm present");
    let mut old_to_new: BTreeMap<i32, i32> = BTreeMap::new();
    old_to_new.insert(fsm.start() as i32, 0);
    let mut queue: VecDeque<i32> = VecDeque::new();
    queue.push_back(fsm.start() as i32);
    let mut hash = 0u64;

    while let Some(old) = queue.pop_front() {
        let new_id = old_to_new[&old];
        hash = if fsm.is_end_state(old as usize) {
            mix(hash, &[new_id as u64, flag::END, flag::END, new_id as u64])
        } else {
            mix(
                hash,
                &[new_id as u64, flag::NOT_END, flag::NOT_END, new_id as u64],
            )
        };

        let edges = sorted_edges(grammar, old as usize);
        // First: rule/repeat-ref edges, ordered by (ref-hash, target).
        let mut hash_and_target: BTreeSet<(i64, i32)> = BTreeSet::new();
        for edge in &edges {
            if edge.is_rule_ref() {
                let rid = edge.ref_rule_id();
                let h = ref_hash(grammar, fsm_index, rid);
                hash_and_target.insert((h as i64, edge.target));
            } else if edge.is_repeat_ref() {
                let info = grammar.complete_fsm.repeat_edge_info(edge.aux_index());
                let rid = info.rule_id as i32;
                let base = ref_hash(grammar, fsm_index, rid);
                let rh = mix(base, &[info.lower as i64 as u64, info.upper as i64 as u64]);
                hash_and_target.insert((rh as i64, edge.target));
            }
        }
        for (h, target) in hash_and_target {
            assign(&mut old_to_new, &mut queue, target);
            let tnew = old_to_new[&target];
            hash = mix(hash, &[new_id as u64, h as u64, tnew as u64]);
        }
        // Then: char-range edges.
        for edge in &edges {
            assign(&mut old_to_new, &mut queue, edge.target);
            let tnew = old_to_new[&edge.target];
            if edge.is_rule_ref() || edge.is_repeat_ref() {
                continue;
            }
            hash = mix(
                hash,
                &[
                    new_id as u64,
                    edge.min as i64 as u64,
                    edge.max as i64 as u64,
                    tnew as u64,
                ],
            );
        }
    }
    (hash, old_to_new.into_iter().collect())
}

/// The hash of a referenced rule (or the self-recursion flag).
fn ref_hash(grammar: &GrammarData, fsm_index: i32, ref_rule_id: i32) -> u64 {
    if ref_rule_id == fsm_index {
        flag::SELF_RECURSION
    } else {
        grammar.per_rule_fsm_hashes[ref_rule_id as usize]
            .expect("referenced rule must already be hashed")
    }
}

/// Renumber `target` in discovery order if not yet seen.
fn assign(map: &mut BTreeMap<i32, i32>, queue: &mut VecDeque<i32>, target: i32) {
    if !map.contains_key(&target) {
        let new_id = map.len() as i32;
        map.insert(target, new_id);
        queue.push_back(target);
    }
}

/// Partially hash an FSM: tolerated when up to one referenced rule is
/// unhashed *and* that edge starts at the start state. `None` otherwise.
pub fn partial_hash_fsm(grammar: &GrammarData, fsm_index: i32) -> Option<(u64, Vec<(i32, i32)>)> {
    let fsm = grammar.per_rule_fsms[fsm_index as usize].as_ref()?;
    let start = fsm.start() as i32;
    let mut old_to_new: BTreeMap<i32, i32> = BTreeMap::new();
    old_to_new.insert(start, 0);
    let mut queue: VecDeque<i32> = VecDeque::new();
    queue.push_back(start);
    let mut hash = 0u64;

    while let Some(old) = queue.pop_front() {
        let is_start = old == start;
        let new_id = old_to_new[&old];
        hash = if fsm.is_end_state(old as usize) {
            mix(hash, &[new_id as u64, flag::END, flag::END, new_id as u64])
        } else {
            mix(
                hash,
                &[new_id as u64, flag::NOT_END, flag::NOT_END, new_id as u64],
            )
        };

        let edges = sorted_edges(grammar, old as usize);
        let mut hash_and_target: BTreeSet<(i64, i32)> = BTreeSet::new();
        let mut unhashed = 0;
        for edge in &edges {
            let rid = if edge.is_rule_ref() {
                edge.ref_rule_id()
            } else if edge.is_repeat_ref() {
                grammar
                    .complete_fsm
                    .repeat_edge_info(edge.aux_index())
                    .rule_id as i32
            } else {
                continue;
            };
            if rid == fsm_index {
                hash_and_target.insert((flag::SELF_RECURSION as i64, edge.target));
                continue;
            }
            match grammar.per_rule_fsm_hashes[rid as usize] {
                Some(h) => {
                    hash_and_target.insert((h as i64, edge.target));
                }
                None => {
                    if !is_start {
                        return None;
                    }
                    unhashed += 1;
                    if unhashed > 1 {
                        return None;
                    }
                    hash_and_target.insert((flag::UNKNOWN as i64, edge.target));
                }
            }
        }
        for (h, target) in hash_and_target {
            assign(&mut old_to_new, &mut queue, target);
            let tnew = old_to_new[&target];
            hash = mix(hash, &[new_id as u64, h as u64, tnew as u64]);
        }
        for edge in &edges {
            assign(&mut old_to_new, &mut queue, edge.target);
            let tnew = old_to_new[&edge.target];
            if edge.is_rule_ref() || edge.is_repeat_ref() {
                continue;
            }
            hash = mix(
                hash,
                &[
                    new_id as u64,
                    edge.min as i64 as u64,
                    edge.max as i64 as u64,
                    tnew as u64,
                ],
            );
        }
    }
    Some((hash, old_to_new.into_iter().collect()))
}
