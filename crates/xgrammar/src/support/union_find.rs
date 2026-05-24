// SPDX-License-Identifier: AGPL-3.0-only
//
// Union-find / disjoint-set — port of `cpp/support/union_find_set.h`.
//
// Uses path compression on `find` and union-by-size on `union_sets`,
// keyed by an arbitrary hashable + ordered element type.

use std::collections::HashMap;
use std::hash::Hash;

/// A disjoint-set forest over elements of type `T`.
///
/// `T` must be `Clone + Eq + Hash` for storage and `Ord` so that
/// [`UnionFindSet::all_sets`] can produce deterministic output.
#[derive(Debug, Default, Clone)]
pub struct UnionFindSet<T> {
    /// Maps each element to `(parent, subtree_size)`.
    parent_and_size: HashMap<T, (T, usize)>,
}

impl<T: Clone + Eq + Hash + Ord> UnionFindSet<T> {
    /// Create an empty union-find set.
    pub fn new() -> Self {
        Self {
            parent_and_size: HashMap::new(),
        }
    }

    /// Add `element` as its own singleton set.
    ///
    /// Returns `true` if inserted, `false` if it already existed.
    pub fn add(&mut self, element: T) -> bool {
        if self.parent_and_size.contains_key(&element) {
            return false;
        }
        self.parent_and_size.insert(element.clone(), (element, 1));
        true
    }

    /// Remove every element from the set.
    pub fn clear(&mut self) {
        self.parent_and_size.clear();
    }

    /// Whether `element` is present.
    pub fn contains(&self, element: &T) -> bool {
        self.parent_and_size.contains_key(element)
    }

    /// Number of stored elements.
    pub fn len(&self) -> usize {
        self.parent_and_size.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.parent_and_size.is_empty()
    }

    /// Find the representative of the set containing `element`,
    /// applying path compression.
    ///
    /// # Panics
    /// Panics if `element` is not present — mirrors the C++
    /// `XGRAMMAR_CHECK`.
    pub fn find(&mut self, element: &T) -> T {
        let parent = self
            .parent_and_size
            .get(element)
            .expect("Element not found in union-find set.")
            .0
            .clone();
        if &parent == element {
            return parent;
        }
        let root = self.find(&parent);
        // Path compression: point `element` straight at the root.
        if let Some(entry) = self.parent_and_size.get_mut(element) {
            entry.0 = root.clone();
        }
        root
    }

    /// Merge the sets containing `a` and `b` (union by size).
    ///
    /// # Panics
    /// Panics if either element is absent.
    pub fn union_sets(&mut self, a: &T, b: &T) {
        assert!(
            self.parent_and_size.contains_key(a),
            "Element not found in union-find set."
        );
        assert!(
            self.parent_and_size.contains_key(b),
            "Element not found in union-find set."
        );
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return;
        }
        let size_a = self.parent_and_size[&root_a].1;
        let size_b = self.parent_and_size[&root_b].1;
        // Ensure root_a is the larger set.
        if size_a < size_b {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        let combined = self.parent_and_size[&root_a].1 + self.parent_and_size[&root_b].1;
        self.parent_and_size.get_mut(&root_b).unwrap().0 = root_a.clone();
        self.parent_and_size.get_mut(&root_a).unwrap().1 = combined;
    }

    /// Collect every disjoint set.
    ///
    /// Each inner vector is sorted ascending, and the outer vector is
    /// sorted by each set's smallest element — deterministic output.
    pub fn all_sets(&mut self) -> Vec<Vec<T>> {
        let elements: Vec<T> = self.parent_and_size.keys().cloned().collect();
        let mut root_to_idx: HashMap<T, usize> = HashMap::new();
        let mut result: Vec<Vec<T>> = Vec::new();
        for value in elements {
            let root = self.find(&value);
            let idx = *root_to_idx.entry(root).or_insert_with(|| {
                result.push(Vec::new());
                result.len() - 1
            });
            result[idx].push(value);
        }
        for vec in result.iter_mut() {
            vec.sort();
        }
        result.sort_by(|v1, v2| v1[0].cmp(&v2[0]));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_returns_false_on_duplicate() {
        let mut uf: UnionFindSet<i32> = UnionFindSet::new();
        assert!(uf.add(1));
        assert!(!uf.add(1));
        assert_eq!(uf.len(), 1);
    }

    #[test]
    fn singleton_is_its_own_root() {
        let mut uf = UnionFindSet::new();
        uf.add(5);
        assert_eq!(uf.find(&5), 5);
    }

    #[test]
    fn union_merges_sets() {
        let mut uf = UnionFindSet::new();
        for i in 1..=4 {
            uf.add(i);
        }
        uf.union_sets(&1, &2);
        uf.union_sets(&3, &4);
        assert_eq!(uf.find(&1), uf.find(&2));
        assert_eq!(uf.find(&3), uf.find(&4));
        assert_ne!(uf.find(&1), uf.find(&3));
        uf.union_sets(&2, &3);
        assert_eq!(uf.find(&1), uf.find(&4));
    }

    #[test]
    fn union_idempotent_on_same_set() {
        let mut uf = UnionFindSet::new();
        uf.add(1);
        uf.add(2);
        uf.union_sets(&1, &2);
        uf.union_sets(&1, &2);
        assert_eq!(uf.find(&1), uf.find(&2));
    }

    #[test]
    fn all_sets_is_deterministic() {
        let mut uf = UnionFindSet::new();
        for i in 1..=6 {
            uf.add(i);
        }
        uf.union_sets(&1, &3);
        uf.union_sets(&3, &5);
        uf.union_sets(&2, &4);
        let sets = uf.all_sets();
        assert_eq!(sets, vec![vec![1, 3, 5], vec![2, 4], vec![6]]);
    }

    #[test]
    fn clear_empties_the_set() {
        let mut uf = UnionFindSet::new();
        uf.add(1);
        uf.add(2);
        uf.clear();
        assert!(uf.is_empty());
        assert!(!uf.contains(&1));
    }

    #[test]
    fn contains_reports_membership() {
        let mut uf = UnionFindSet::new();
        uf.add(42);
        assert!(uf.contains(&42));
        assert!(!uf.contains(&7));
    }

    #[test]
    #[should_panic(expected = "Element not found")]
    fn find_missing_panics() {
        let mut uf: UnionFindSet<i32> = UnionFindSet::new();
        uf.find(&99);
    }

    #[test]
    fn works_with_string_elements() {
        let mut uf: UnionFindSet<String> = UnionFindSet::new();
        uf.add("a".to_string());
        uf.add("b".to_string());
        uf.add("c".to_string());
        uf.union_sets(&"a".to_string(), &"c".to_string());
        let sets = uf.all_sets();
        assert_eq!(
            sets,
            vec![
                vec!["a".to_string(), "c".to_string()],
                vec!["b".to_string()]
            ]
        );
    }
}
