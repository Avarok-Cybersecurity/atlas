// SPDX-License-Identifier: AGPL-3.0-only
//
// FSM simplification passes — `SimplifyEpsilon` and
// `MergeEquivalentSuccessors`. Port of the corresponding methods of
// `FSMWithStartEnd` from `cpp/fsm.cc`.

use ahash::AHashMap;

use super::edge::FsmEdge;
use super::union_find::UnionFindSet;
use super::with_start_end::FsmWithStartEnd;

impl FsmWithStartEnd {
    /// Merge states linked by removable epsilon transitions.
    ///
    /// `a --eps--> b` is collapsible when either (1) `a` has no other
    /// outgoing edge, or (2) `b` has no other incoming edge.
    pub fn simplify_epsilon(&self) -> FsmWithStartEnd {
        if self.is_dfa {
            return self.clone();
        }
        let n = self.num_states();

        let mut uf = UnionFindSet::new();
        let mut in_degree = vec![0i32; n];
        let mut epsilon_edges: Vec<(usize, usize)> = Vec::new();

        for i in 0..n {
            let edges = self.fsm.edges(i);
            for edge in edges {
                in_degree[edge.target as usize] += 1;
                if edge.is_epsilon() {
                    if edges.len() == 1 {
                        // case 1: `a` has only this outgoing edge
                        uf.add(i as i32);
                        uf.add(edge.target);
                        uf.union(i as i32, edge.target);
                        in_degree[edge.target as usize] -= 1;
                    } else {
                        epsilon_edges.push((i, edge.target as usize));
                    }
                }
            }
        }

        // Build the equivalence representative per node.
        let mut equiv_node = vec![0usize; n];
        for i in 0..n {
            if uf.contains(i as i32) {
                let rep = uf.find(i as i32) as usize;
                equiv_node[i] = rep;
                if rep != i {
                    in_degree[rep] += in_degree[i];
                }
            } else {
                equiv_node[i] = i;
            }
        }

        // case 2: `a --eps--> b`, `b` has no other incoming edge.
        for &(from_raw, to_raw) in &epsilon_edges {
            let from = equiv_node[from_raw];
            let to = equiv_node[to_raw];
            if in_degree[to] == 1 && equiv_node[self.start] != to {
                uf.add(from as i32);
                uf.add(to as i32);
                uf.union(from as i32, to as i32);
            }
        }

        let eq_classes = uf.all_sets();
        if eq_classes.is_empty() {
            return self.clone();
        }

        let mut new_to_old = vec![-1i64; n];
        for (i, class) in eq_classes.iter().enumerate() {
            for &state in class {
                new_to_old[state as usize] = i as i64;
            }
        }
        let mut cnt = eq_classes.len();
        for slot in new_to_old.iter_mut() {
            if *slot == -1 {
                *slot = cnt as i64;
                cnt += 1;
            }
        }
        let mapping: Vec<usize> = new_to_old.iter().map(|&v| v as usize).collect();
        self.rebuild_with_mapping(&mapping, cnt)
    }

    /// Merge states with identical incoming or outgoing transition
    /// structure (`ab | ac | ad` -> `a(b|c|d)`, and the mirror case).
    pub fn merge_equivalent_successors(&self) -> FsmWithStartEnd {
        let mut result = self.copy();
        result.fsm_mut().sort_edges();
        let mut uf = UnionFindSet::new();
        let mut changed = true;

        while changed {
            uf.clear();
            let n = result.num_states();
            // previous_states[t][s] = edges s->t ; next_states[s][t] = edges s->t
            let mut previous: Vec<AHashMap<usize, Vec<FsmEdge>>> =
                vec![AHashMap::new(); n];
            let mut next: Vec<AHashMap<usize, Vec<FsmEdge>>> = vec![AHashMap::new(); n];
            for i in 0..n {
                for edge in result.fsm().edges(i) {
                    let t = edge.target as usize;
                    previous[t].entry(i).or_default().push(*edge);
                    next[i].entry(t).or_default().push(*edge);
                }
            }

            let mut equiv_successor = false;
            // Case 1: ab|ac|ad -> a(b|c|d)
            for i in 0..n {
                if previous[i].len() != 1 || uf.contains(i as i32) {
                    continue;
                }
                let (prev_state, edges_to_i) = previous[i].iter().next().unwrap();
                let prev_state = *prev_state;
                let edges_to_i = edges_to_i.clone();
                let siblings: Vec<(usize, Vec<FsmEdge>)> = next[prev_state]
                    .iter()
                    .map(|(k, v)| (*k, v.clone()))
                    .collect();
                for (sibling, edges_to_sibling) in siblings {
                    if sibling <= i
                        || previous[sibling].len() != 1
                        || result.is_end_state(sibling) != result.is_end_state(i)
                    {
                        continue;
                    }
                    if edges_to_i.len() != edges_to_sibling.len() {
                        break;
                    }
                    let same = edges_to_i.iter().zip(&edges_to_sibling).all(|(a, b)| {
                        a.min == b.min && a.max == b.max
                    });
                    if same {
                        uf.add(i as i32);
                        uf.add(sibling as i32);
                        uf.union(i as i32, sibling as i32);
                        equiv_successor = true;
                    }
                }
            }

            // Case 2: ba|ca|da -> (b|c|d)a, plus dead-end merges.
            let mut equiv_precursor = false;
            let mut no_succ_end: Vec<usize> = Vec::new();
            let mut no_succ_non_end: Vec<usize> = Vec::new();
            for i in 0..n {
                if next[i].is_empty() {
                    if result.is_end_state(i) {
                        no_succ_end.push(i);
                    } else {
                        no_succ_non_end.push(i);
                    }
                    continue;
                }
                if next[i].len() != 1 || uf.contains(i as i32) {
                    continue;
                }
                let next_state = *next[i].keys().next().unwrap();
                let node_edges: Vec<FsmEdge> = result.fsm().edges(i).to_vec();
                let siblings: Vec<usize> = previous[next_state].keys().copied().collect();
                for sibling in siblings {
                    if sibling <= i
                        || next[sibling].len() != 1
                        || result.is_end_state(i) != result.is_end_state(sibling)
                    {
                        continue;
                    }
                    let sibling_edges = result.fsm().edges(sibling);
                    if sibling_edges.len() != node_edges.len() {
                        continue;
                    }
                    let same = sibling_edges
                        .iter()
                        .zip(&node_edges)
                        .all(|(a, b)| a.min == b.min && a.max == b.max);
                    if same {
                        uf.add(i as i32);
                        uf.add(sibling as i32);
                        uf.union(i as i32, sibling as i32);
                        equiv_successor = true;
                    }
                }
            }

            if no_succ_end.len() > 1 {
                for &s in &no_succ_end[1..] {
                    uf.add(no_succ_end[0] as i32);
                    uf.add(s as i32);
                    uf.union(no_succ_end[0] as i32, s as i32);
                    equiv_precursor = true;
                }
            }
            if no_succ_non_end.len() > 1 {
                for &s in &no_succ_non_end[1..] {
                    uf.add(no_succ_non_end[0] as i32);
                    uf.add(s as i32);
                    uf.union(no_succ_non_end[0] as i32, s as i32);
                    equiv_precursor = true;
                }
            }

            changed = equiv_successor || equiv_precursor;
            if changed {
                let eq_classes = uf.all_sets();
                let mut old_to_new = vec![-1i64; n];
                for (idx, class) in eq_classes.iter().enumerate() {
                    for &state in class {
                        old_to_new[state as usize] = idx as i64;
                    }
                }
                let mut cnt = eq_classes.len();
                for slot in old_to_new.iter_mut() {
                    if *slot == -1 {
                        *slot = cnt as i64;
                        cnt += 1;
                    }
                }
                let mapping: Vec<usize> = old_to_new.iter().map(|&v| v as usize).collect();
                result = result.rebuild_with_mapping(&mapping, cnt);
                result.fsm_mut().sort_edges();
            }
        }
        result
    }
}

#[cfg(test)]
#[path = "simplify_tests.rs"]
mod tests;
