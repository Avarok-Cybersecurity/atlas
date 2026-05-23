// SPDX-License-Identifier: AGPL-3.0-only
//
// FSM simplification passes — `SimplifyEpsilon` and
// `MergeEquivalentSuccessors`. Port of the corresponding methods of
// `FSMWithStartEnd` from `cpp/fsm.cc`.

use super::union_find::UnionFindSet;
use super::with_start_end::FsmWithStartEnd;

/// A compact edge endpoint used for the incoming/outgoing CSR rows of
/// [`FsmWithStartEnd::merge_equivalent_successors`]. Port of upstream
/// `EndpointEdge` (xgrammar commit 96ae88b).
///
/// `peer` is the *source* state in incoming rows and the *target* state
/// in outgoing rows. Outgoing rows are sorted by `(peer, min, max)` so
/// successors of one source form contiguous groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EndpointEdge {
    peer: i32,
    min: i16,
    max: i16,
}

/// A CSR (compressed-sparse-row) array of [`EndpointEdge`]: `data`
/// holds all rows contiguously, `indptr[i]..indptr[i + 1]` delimits
/// row `i`. Replaces the per-state `HashMap`s that upstream commit
/// 96ae88b removed: row sizing + filling is two flat passes with no
/// per-edge allocation, and the buffers are reused across iterations.
#[derive(Debug, Default)]
struct EdgeCsr {
    data: Vec<EndpointEdge>,
    indptr: Vec<i32>,
}

impl EdgeCsr {
    /// Resize the CSR so row `i` has `row_sizes[i]` default slots,
    /// reusing existing allocations. Port of `ResetWithRowSizes`.
    fn reset_with_row_sizes(&mut self, row_sizes: &[i32]) {
        self.indptr.clear();
        self.indptr.reserve(row_sizes.len() + 1);
        self.indptr.push(0);
        let mut acc = 0i32;
        for &sz in row_sizes {
            debug_assert!(sz >= 0);
            acc += sz;
            self.indptr.push(acc);
        }
        self.data.clear();
        self.data.resize(
            acc as usize,
            EndpointEdge {
                peer: 0,
                min: 0,
                max: 0,
            },
        );
    }

    /// Immutable view of row `i`.
    fn row(&self, i: usize) -> &[EndpointEdge] {
        let start = self.indptr[i] as usize;
        let end = self.indptr[i + 1] as usize;
        &self.data[start..end]
    }

    /// Mutable view of row `i`.
    fn row_mut(&mut self, i: usize) -> &mut [EndpointEdge] {
        let start = self.indptr[i] as usize;
        let end = self.indptr[i + 1] as usize;
        &mut self.data[start..end]
    }
}

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
    ///
    /// Upstream renamed this `MergeEquivalentStates` (commit 8d22ba0);
    /// the port keeps the original name.
    ///
    /// The incoming/outgoing edge relations are stored in two reused
    /// CSR arrays (`EdgeCsr`) instead of per-state hash maps — port
    /// of upstream commit 96ae88b (#616), which found `build_maps`
    /// dominated the pass through hash-map and small-vector churn.
    pub fn merge_equivalent_successors(&self) -> FsmWithStartEnd {
        // No merge is possible with fewer than 4 states: a Case 1 merge
        // needs >=2 sinks sharing a source, a Case 2 merge needs >=2
        // sources sharing a sink (upstream commit 96ae88b, #616).
        if self.num_states() < 4 {
            return self.copy();
        }
        let mut result = self.copy();
        result.fsm_mut().sort_edges();
        let mut uf = UnionFindSet::new();
        let mut changed = true;

        // Scratch buffers hoisted out of the loop so their capacity is
        // reused across iterations (upstream commit 96ae88b).
        let mut incoming = EdgeCsr::default();
        let mut outgoing = EdgeCsr::default();
        let mut incoming_row_sizes: Vec<i32> = Vec::new();
        let mut outgoing_row_sizes: Vec<i32> = Vec::new();
        let mut incoming_write_pos: Vec<i32> = Vec::new();
        let mut outgoing_write_pos: Vec<i32> = Vec::new();
        // For each state: distinct-peer count and (when count == 1) the
        // single peer; -1 otherwise.
        let mut incoming_distinct: Vec<i32> = Vec::new();
        let mut outgoing_distinct: Vec<i32> = Vec::new();
        let mut single_incoming: Vec<i32> = Vec::new();
        let mut single_outgoing: Vec<i32> = Vec::new();
        let mut no_succ_end: Vec<usize> = Vec::new();
        let mut no_succ_non_end: Vec<usize> = Vec::new();

        while changed {
            uf.clear();
            let n = result.num_states();

            // First pass: count incoming/outgoing edges per state.
            incoming_row_sizes.clear();
            incoming_row_sizes.resize(n, 0);
            outgoing_row_sizes.clear();
            outgoing_row_sizes.resize(n, 0);
            for source in 0..n {
                let edges = result.fsm().edges(source);
                outgoing_row_sizes[source] = edges.len() as i32;
                for edge in edges {
                    incoming_row_sizes[edge.target as usize] += 1;
                }
            }

            // Allocate CSR rows (reusing backing storage).
            incoming.reset_with_row_sizes(&incoming_row_sizes);
            outgoing.reset_with_row_sizes(&outgoing_row_sizes);
            incoming_write_pos.clear();
            incoming_write_pos.resize(n, 0);
            outgoing_write_pos.clear();
            outgoing_write_pos.resize(n, 0);

            // Second pass: fill incoming/outgoing rows. Incoming rows
            // are naturally grouped by source (sources scanned in
            // order); outgoing rows are sorted by `(peer, min, max)`.
            for source in 0..n {
                for edge in result.fsm().edges(source) {
                    let t = edge.target as usize;
                    let ip = incoming_write_pos[t];
                    incoming.row_mut(t)[ip as usize] = EndpointEdge {
                        peer: source as i32,
                        min: edge.min,
                        max: edge.max,
                    };
                    incoming_write_pos[t] = ip + 1;
                    let op = outgoing_write_pos[source];
                    outgoing.row_mut(source)[op as usize] = EndpointEdge {
                        peer: edge.target,
                        min: edge.min,
                        max: edge.max,
                    };
                    outgoing_write_pos[source] = op + 1;
                }
                outgoing.row_mut(source).sort_unstable();
            }

            // Distinct-peer counts. A peer count of 1 with its identity
            // mirrors the old `map.len() == 1` + `keys().next()` usage.
            distinct_peers(&incoming, n, &mut incoming_distinct, &mut single_incoming);
            distinct_peers(&outgoing, n, &mut outgoing_distinct, &mut single_outgoing);

            let mut equiv_successor = false;
            // Case 1: ab|ac|ad -> a(b|c|d)
            for i in 0..n {
                if incoming_distinct[i] != 1 || uf.contains(i as i32) {
                    continue;
                }
                let prev_state = single_incoming[i] as usize;
                let edges_to_i = incoming.row(i);
                let siblings = outgoing.row(prev_state);
                let mut group_begin = 0usize;
                while group_begin < siblings.len() {
                    let sibling = siblings[group_begin].peer as usize;
                    let mut group_end = group_begin + 1;
                    while group_end < siblings.len() && siblings[group_end].peer as usize == sibling
                    {
                        group_end += 1;
                    }
                    let edges_to_sibling = &siblings[group_begin..group_end];
                    group_begin = group_end;
                    if sibling <= i
                        || incoming_distinct[sibling] != 1
                        || result.is_end_state(sibling) != result.is_end_state(i)
                    {
                        continue;
                    }
                    if edges_to_i.len() != edges_to_sibling.len() {
                        continue;
                    }
                    let same = edges_to_i
                        .iter()
                        .zip(edges_to_sibling)
                        .all(|(a, b)| a.min == b.min && a.max == b.max);
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
            no_succ_end.clear();
            no_succ_non_end.clear();
            for i in 0..n {
                if outgoing_distinct[i] == 0 {
                    if result.is_end_state(i) {
                        no_succ_end.push(i);
                    } else {
                        no_succ_non_end.push(i);
                    }
                    continue;
                }
                if outgoing_distinct[i] != 1 || uf.contains(i as i32) {
                    continue;
                }
                let next_state = single_outgoing[i] as usize;
                let node_edges = outgoing.row(i);
                let siblings = incoming.row(next_state);
                let mut group_begin = 0usize;
                while group_begin < siblings.len() {
                    let sibling = siblings[group_begin].peer as usize;
                    while group_begin < siblings.len()
                        && siblings[group_begin].peer as usize == sibling
                    {
                        group_begin += 1;
                    }
                    // Skip a sibling already merged earlier this iteration
                    // (typically by Case 1): chaining a Case 2 merge onto it
                    // can over-merge via transitive closure (upstream 8d22ba0,
                    // #632). The `equiv_precursor` flag below was also fixed
                    // from a wrongly-set `equiv_successor` in the same commit.
                    if sibling <= i
                        || uf.contains(sibling as i32)
                        || outgoing_distinct[sibling] != 1
                        || result.is_end_state(i) != result.is_end_state(sibling)
                    {
                        continue;
                    }
                    let sibling_edges = outgoing.row(sibling);
                    if sibling_edges.len() != node_edges.len() {
                        continue;
                    }
                    let same = sibling_edges
                        .iter()
                        .zip(node_edges)
                        .all(|(a, b)| a.min == b.min && a.max == b.max);
                    if same {
                        uf.add(i as i32);
                        uf.add(sibling as i32);
                        uf.union(i as i32, sibling as i32);
                        equiv_precursor = true;
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

/// Compute, for each of the first `n` rows of `csr`, the number of
/// distinct `peer` values and (when exactly one) that peer's id.
///
/// `distinct[i]` and `single[i]` are resized and overwritten. A row
/// with one distinct peer mirrors the old `map.len() == 1` check; the
/// single peer mirrors `map.keys().next()`. Rows are pre-sorted by
/// `peer`, so distinct peers are counted by scanning for changes.
fn distinct_peers(csr: &EdgeCsr, n: usize, distinct: &mut Vec<i32>, single: &mut Vec<i32>) {
    distinct.clear();
    distinct.resize(n, 0);
    single.clear();
    single.resize(n, -1);
    for (state, slot) in distinct.iter_mut().enumerate().take(n) {
        let row = csr.row(state);
        if row.is_empty() {
            continue;
        }
        let mut count = 1;
        let mut only = row[0].peer;
        for w in row.windows(2) {
            if w[0].peer != w[1].peer {
                count += 1;
                only = -1;
            }
        }
        *slot = count;
        single[state] = only;
    }
}

#[cfg(test)]
#[path = "simplify_tests.rs"]
mod tests;
