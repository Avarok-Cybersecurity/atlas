// SPDX-License-Identifier: AGPL-3.0-only
//
// Compact2DArray — port of `cpp/support/compact_2d_array.h`.
//
// A Compressed-Sparse-Row (CSR) 2D array: a sequence of variable-length
// rows packed end-to-end in one backing buffer. Inserted rows are
// immutable. Atlas's grammar AST already uses this pattern inline; this
// is the reusable, standalone version.

use serde::{Deserialize, Serialize};

/// A CSR-style 2D array of `T`.
///
/// `data` holds every row's elements contiguously; `indptr` records the
/// start offset of each row, with a trailing entry equal to
/// `data.len()`. With `n` rows, `indptr` has `n + 1` entries — so
/// `indptr` always starts as `[0]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Compact2DArray<T> {
    /// All row elements packed end-to-end.
    data: Vec<T>,
    /// Row start offsets; `indptr[i]..indptr[i + 1]` is row `i`.
    indptr: Vec<i32>,
}

impl<T> Default for Compact2DArray<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Compact2DArray<T> {
    /// Create an empty array (`indptr` initialized to `[0]`).
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            indptr: vec![0],
        }
    }

    /// Number of rows.
    pub fn len(&self) -> i32 {
        self.indptr.len() as i32 - 1
    }

    /// Whether there are no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow row `i` as a slice.
    ///
    /// # Panics
    /// Panics if `i` is out of `0..len()`.
    pub fn row(&self, i: i32) -> &[T] {
        assert!(
            i >= 0 && i < self.len(),
            "Compact2DArray index {i} is out of bound"
        );
        let start = self.indptr[i as usize] as usize;
        let end = self.indptr[i as usize + 1] as usize;
        &self.data[start..end]
    }

    /// Borrow the last row.
    ///
    /// # Panics
    /// Panics if the array has no rows.
    pub fn back(&self) -> &[T] {
        assert!(!self.is_empty(), "Compact2DArray is empty");
        self.row(self.len() - 1)
    }

    /// Iterate over rows as slices.
    pub fn iter(&self) -> impl Iterator<Item = &[T]> {
        (0..self.len()).map(move |i| self.row(i))
    }

    /// The flat backing data buffer.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// The row-offset (`indptr`) array.
    pub fn indptr(&self) -> &[i32] {
        &self.indptr
    }

    /// Append `new_data` as a fresh row; returns its row index.
    pub fn push_row(&mut self, new_data: &[T]) -> i32
    where
        T: Clone,
    {
        self.data.extend_from_slice(new_data);
        self.indptr.push(self.data.len() as i32);
        self.indptr.len() as i32 - 2
    }

    /// Append `new_data` as a fresh row by value (no clone); returns
    /// the row index.
    pub fn push_row_owned(&mut self, new_data: Vec<T>) -> i32 {
        self.data.extend(new_data);
        self.indptr.push(self.data.len() as i32);
        self.indptr.len() as i32 - 2
    }

    /// Append a row consisting of `first` followed by `rest`; returns
    /// the row index. Mirrors C++ `PushBackNonContiguous`, used by the
    /// grammar AST.
    pub fn push_row_non_contiguous(&mut self, first: T, rest: &[T]) -> i32
    where
        T: Clone,
    {
        self.data.push(first);
        self.data.extend_from_slice(rest);
        self.indptr.push(self.data.len() as i32);
        self.indptr.len() as i32 - 2
    }

    /// Append one element to the most recently inserted row.
    ///
    /// # Panics
    /// Panics if there is no row to append to.
    pub fn push_in_latest_row(&mut self, new_data: T) {
        assert!(
            self.indptr.len() > 1,
            "Cannot push back in an empty Compact2DArray"
        );
        self.data.push(new_data);
        *self.indptr.last_mut().unwrap() += 1;
    }

    /// Remove the last `cnt` rows.
    ///
    /// # Panics
    /// Panics if `cnt` exceeds the number of rows.
    pub fn pop_rows(&mut self, cnt: i32) {
        assert!(
            cnt >= 0 && cnt <= self.len(),
            "Cannot pop {cnt} rows from a Compact2DArray with {} rows",
            self.len()
        );
        let new_indptr_len = self.indptr.len() - cnt as usize;
        self.indptr.truncate(new_indptr_len);
        let new_data_len = *self.indptr.last().unwrap() as usize;
        self.data.truncate(new_data_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_array() {
        let arr: Compact2DArray<i32> = Compact2DArray::new();
        assert_eq!(arr.len(), 0);
        assert!(arr.is_empty());
        assert_eq!(arr.indptr(), &[0]);
    }

    #[test]
    fn push_and_read_rows() {
        let mut arr = Compact2DArray::new();
        assert_eq!(arr.push_row(&[1, 2, 3]), 0);
        assert_eq!(arr.push_row(&[4]), 1);
        assert_eq!(arr.push_row(&[]), 2);
        assert_eq!(arr.push_row(&[5, 6]), 3);
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.row(0), &[1, 2, 3]);
        assert_eq!(arr.row(1), &[4]);
        assert_eq!(arr.row(2), &[] as &[i32]);
        assert_eq!(arr.row(3), &[5, 6]);
        assert_eq!(arr.data(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(arr.indptr(), &[0, 3, 4, 4, 6]);
    }

    #[test]
    fn push_row_owned_matches_push_row() {
        let mut arr = Compact2DArray::new();
        arr.push_row_owned(vec![7, 8]);
        arr.push_row_owned(vec![9]);
        assert_eq!(arr.row(0), &[7, 8]);
        assert_eq!(arr.row(1), &[9]);
    }

    #[test]
    fn back_returns_last_row() {
        let mut arr = Compact2DArray::new();
        arr.push_row(&[1]);
        arr.push_row(&[2, 3]);
        assert_eq!(arr.back(), &[2, 3]);
    }

    #[test]
    fn push_non_contiguous() {
        let mut arr = Compact2DArray::new();
        assert_eq!(arr.push_row_non_contiguous(10, &[20, 30]), 0);
        assert_eq!(arr.row(0), &[10, 20, 30]);
        assert_eq!(arr.push_row_non_contiguous(40, &[]), 1);
        assert_eq!(arr.row(1), &[40]);
    }

    #[test]
    fn push_in_latest_row() {
        let mut arr = Compact2DArray::new();
        arr.push_row(&[1, 2]);
        arr.push_in_latest_row(3);
        arr.push_in_latest_row(4);
        assert_eq!(arr.row(0), &[1, 2, 3, 4]);
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn pop_rows() {
        let mut arr = Compact2DArray::new();
        arr.push_row(&[1, 2]);
        arr.push_row(&[3]);
        arr.push_row(&[4, 5, 6]);
        arr.pop_rows(2);
        assert_eq!(arr.len(), 1);
        assert_eq!(arr.row(0), &[1, 2]);
        assert_eq!(arr.data(), &[1, 2]);
        arr.pop_rows(1);
        assert!(arr.is_empty());
    }

    #[test]
    fn iter_yields_all_rows() {
        let mut arr = Compact2DArray::new();
        arr.push_row(&[1]);
        arr.push_row(&[2, 3]);
        let rows: Vec<Vec<i32>> = arr.iter().map(|r| r.to_vec()).collect();
        assert_eq!(rows, vec![vec![1], vec![2, 3]]);
    }

    #[test]
    #[should_panic(expected = "out of bound")]
    fn row_out_of_bounds_panics() {
        let mut arr = Compact2DArray::new();
        arr.push_row(&[1]);
        arr.row(1);
    }

    #[test]
    #[should_panic(expected = "Cannot push back in an empty")]
    fn push_in_latest_row_empty_panics() {
        let mut arr: Compact2DArray<i32> = Compact2DArray::new();
        arr.push_in_latest_row(1);
    }

    #[test]
    fn serde_round_trip() {
        let mut arr = Compact2DArray::new();
        arr.push_row(&[1, 2, 3]);
        arr.push_row(&[4, 5]);
        let json = serde_json::to_string(&arr).unwrap();
        let back: Compact2DArray<i32> = serde_json::from_str(&json).unwrap();
        assert_eq!(arr, back);
    }
}
