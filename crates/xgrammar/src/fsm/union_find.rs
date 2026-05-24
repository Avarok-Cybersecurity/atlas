// SPDX-License-Identifier: AGPL-3.0-only
//
// UnionFindSet — a disjoint-set forest keyed by `i32` state ids.
// Port of `cpp/support/union_find_set.h`, specialized to `i32` (the FSM
// only ever unions state ids). Uses path compression + union-by-size.

use ahash::AHashMap;

/// A disjoint-set (union-find) structure over `i32` elements.
///
/// Elements must be `add`ed before they can be `find`/`union`-ed; this
/// matches the C++ semantics where membership is explicit.
#[derive(Debug, Default)]
pub struct UnionFindSet {
    /// Maps element -> (parent, subtree size).
    parent_size: AHashMap<i32, (i32, usize)>,
}

impl UnionFindSet {
    /// Create an empty union-find set.
    pub fn new() -> Self {
        Self {
            parent_size: AHashMap::new(),
        }
    }

    /// Add `element` as its own singleton set. Returns `false` if it
    /// was already present.
    pub fn add(&mut self, element: i32) -> bool {
        if self.parent_size.contains_key(&element) {
            return false;
        }
        self.parent_size.insert(element, (element, 1));
        true
    }

    /// Remove every element.
    pub fn clear(&mut self) {
        self.parent_size.clear();
    }

    /// True if `element` has been added.
    pub fn contains(&self, element: i32) -> bool {
        self.parent_size.contains_key(&element)
    }

    /// Find the representative of `element`'s set, compressing the path.
    /// Panics if `element` was never added (matches C++ `XGRAMMAR_CHECK`).
    pub fn find(&mut self, element: i32) -> i32 {
        let parent = self
            .parent_size
            .get(&element)
            .expect("Element not found in union-find set.")
            .0;
        if parent != element {
            let root = self.find(parent);
            self.parent_size.get_mut(&element).unwrap().0 = root;
            root
        } else {
            element
        }
    }

    /// Merge the sets containing `a` and `b`. Both must have been added.
    pub fn union(&mut self, a: i32, b: i32) {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return;
        }
        let size_a = self.parent_size[&root_a].1;
        let size_b = self.parent_size[&root_b].1;
        // Keep root_a as the larger set (union by size).
        if size_a < size_b {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent_size.get_mut(&root_b).unwrap().0 = root_a;
        let merged = self.parent_size[&root_a].1 + self.parent_size[&root_b].1;
        self.parent_size.get_mut(&root_a).unwrap().1 = merged;
    }

    /// Collect all sets, each sorted ascending, ordered by their first
    /// element — deterministic, matching the C++ `GetAllSets`.
    pub fn all_sets(&mut self) -> Vec<Vec<i32>> {
        let elements: Vec<i32> = self.parent_size.keys().copied().collect();
        let mut root_to_idx: AHashMap<i32, usize> = AHashMap::new();
        let mut result: Vec<Vec<i32>> = Vec::new();
        for value in elements {
            let root = self.find(value);
            let idx = *root_to_idx.entry(root).or_insert_with(|| {
                result.push(Vec::new());
                result.len() - 1
            });
            result[idx].push(value);
        }
        for vec in &mut result {
            vec.sort_unstable();
        }
        result.sort_by(|v1, v2| v1[0].cmp(&v2[0]));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_is_idempotent() {
        let mut uf = UnionFindSet::new();
        assert!(uf.add(1));
        assert!(!uf.add(1));
        assert!(uf.contains(1));
        assert!(!uf.contains(2));
    }

    #[test]
    fn union_merges_sets() {
        let mut uf = UnionFindSet::new();
        for i in 0..4 {
            uf.add(i);
        }
        uf.union(0, 1);
        uf.union(2, 3);
        assert_eq!(uf.find(0), uf.find(1));
        assert_eq!(uf.find(2), uf.find(3));
        assert_ne!(uf.find(0), uf.find(2));
        uf.union(1, 2);
        assert_eq!(uf.find(0), uf.find(3));
    }

    #[test]
    fn all_sets_deterministic() {
        let mut uf = UnionFindSet::new();
        for i in 0..6 {
            uf.add(i);
        }
        uf.union(4, 1);
        uf.union(5, 2);
        let sets = uf.all_sets();
        // Sets: {1,4}, {2,5}, {0}, {3} — ordered by first element.
        assert_eq!(sets, vec![vec![0], vec![1, 4], vec![2, 5], vec![3]]);
    }

    #[test]
    fn clear_empties() {
        let mut uf = UnionFindSet::new();
        uf.add(7);
        uf.clear();
        assert!(!uf.contains(7));
    }

    #[test]
    #[should_panic(expected = "not found")]
    fn find_missing_panics() {
        let mut uf = UnionFindSet::new();
        uf.find(99);
    }
}
