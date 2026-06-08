// SPDX-License-Identifier: AGPL-3.0-only
//
// Integer-set utilities — port of `cpp/support/int_set.h`.
//
// Both operations assume their inputs are sorted ascending and treat
// the vectors as mathematical sets. They mutate `lhs` in place,
// matching the C++ `IntsetUnion` / `IntsetIntersection` signatures.

/// Replace `lhs` with the sorted union of `lhs` and `rhs`.
///
/// Both inputs must already be sorted ascending; duplicates within an
/// input are collapsed in the result. Runs in `O(n + m)`.
pub fn intset_union(lhs: &mut Vec<i32>, rhs: &[i32]) {
    // Merge two sorted slices, then dedup — equivalent in observable
    // behavior to the C++ in-place reverse merge + std::unique.
    let mut merged: Vec<i32> = Vec::with_capacity(lhs.len() + rhs.len());
    let mut i = 0;
    let mut j = 0;
    while i < lhs.len() && j < rhs.len() {
        match lhs[i].cmp(&rhs[j]) {
            std::cmp::Ordering::Less => {
                merged.push(lhs[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                merged.push(rhs[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                merged.push(lhs[i]);
                i += 1;
                j += 1;
            }
        }
    }
    merged.extend_from_slice(&lhs[i..]);
    merged.extend_from_slice(&rhs[j..]);
    merged.dedup();
    *lhs = merged;
}

/// Replace `lhs` with the sorted intersection of `lhs` and `rhs`.
///
/// Both inputs must already be sorted ascending. As a special case,
/// `lhs == [-1]` is treated as the universal set, so the result
/// becomes `rhs`. Runs in `O(n + m)`.
pub fn intset_intersection(lhs: &mut Vec<i32>, rhs: &[i32]) {
    if lhs.len() == 1 && lhs[0] == -1 {
        *lhs = rhs.to_vec();
        return;
    }

    let mut write = 0;
    let mut i = 0;
    let mut j = 0;
    while i < lhs.len() && j < rhs.len() {
        match lhs[i].cmp(&rhs[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                lhs[write] = lhs[i];
                write += 1;
                i += 1;
                j += 1;
            }
        }
    }
    lhs.truncate(write);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_disjoint() {
        let mut lhs = vec![1, 3, 5];
        intset_union(&mut lhs, &[2, 4, 6]);
        assert_eq!(lhs, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn union_overlapping() {
        let mut lhs = vec![1, 2, 3];
        intset_union(&mut lhs, &[2, 3, 4]);
        assert_eq!(lhs, vec![1, 2, 3, 4]);
    }

    #[test]
    fn union_with_empty() {
        let mut lhs = vec![1, 2, 3];
        intset_union(&mut lhs, &[]);
        assert_eq!(lhs, vec![1, 2, 3]);

        let mut empty: Vec<i32> = vec![];
        intset_union(&mut empty, &[4, 5]);
        assert_eq!(empty, vec![4, 5]);
    }

    #[test]
    fn union_identical() {
        let mut lhs = vec![1, 2, 3];
        intset_union(&mut lhs, &[1, 2, 3]);
        assert_eq!(lhs, vec![1, 2, 3]);
    }

    #[test]
    fn union_dedups_input_duplicates() {
        let mut lhs = vec![1, 1, 2];
        intset_union(&mut lhs, &[2, 2, 3]);
        assert_eq!(lhs, vec![1, 2, 3]);
    }

    #[test]
    fn intersection_basic() {
        let mut lhs = vec![1, 2, 3, 4];
        intset_intersection(&mut lhs, &[2, 4, 6]);
        assert_eq!(lhs, vec![2, 4]);
    }

    #[test]
    fn intersection_disjoint() {
        let mut lhs = vec![1, 3, 5];
        intset_intersection(&mut lhs, &[2, 4, 6]);
        assert_eq!(lhs, Vec::<i32>::new());
    }

    #[test]
    fn intersection_universal_set() {
        let mut lhs = vec![-1];
        intset_intersection(&mut lhs, &[7, 8, 9]);
        assert_eq!(lhs, vec![7, 8, 9]);
    }

    #[test]
    fn intersection_with_empty() {
        let mut lhs = vec![1, 2, 3];
        intset_intersection(&mut lhs, &[]);
        assert_eq!(lhs, Vec::<i32>::new());
    }

    #[test]
    fn intersection_negative_one_not_universal_when_longer() {
        // [-1, 2] is a real set, not the universal sentinel.
        let mut lhs = vec![-1, 2];
        intset_intersection(&mut lhs, &[-1, 5]);
        assert_eq!(lhs, vec![-1]);
    }
}
