// SPDX-License-Identifier: AGPL-3.0-only
//
// Compact2DArray — a Compressed-Sparse-Row (CSR) 2D array.
// Port of `cpp/support/compact_2d_array.h`, restricted to the FSM use
// case (rows of `FsmEdge`). It stores a jagged 2D array as two flat
// vectors so that all rows are contiguous in memory — this layout is
// load-bearing for the compiled-grammar memory footprint.

/// A CSR-packed jagged 2D array.
///
/// `data` holds every row's elements contiguously; `indptr` holds the
/// start offset of each row, with a trailing entry equal to `data.len()`
/// so row `i` is `data[indptr[i]..indptr[i+1]]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compact2DArray<T> {
    data: Vec<T>,
    indptr: Vec<i32>,
}

impl<T: Clone> Default for Compact2DArray<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Compact2DArray<T> {
    /// Create an empty array (zero rows).
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            indptr: vec![0],
        }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.indptr.len() - 1
    }

    /// True when there are no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total number of stored elements across all rows.
    pub fn total_elems(&self) -> usize {
        self.data.len()
    }

    /// Borrow row `i` as a slice. Panics if `i` is out of bounds.
    pub fn row(&self, i: usize) -> &[T] {
        let start = self.indptr[i] as usize;
        let end = self.indptr[i + 1] as usize;
        &self.data[start..end]
    }

    /// Append a new row from a slice; returns the new row's index.
    pub fn push_row(&mut self, row: &[T]) -> usize {
        self.data.extend_from_slice(row);
        self.indptr.push(self.data.len() as i32);
        self.len() - 1
    }

    /// Iterate over every row as a slice.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[T]> {
        (0..self.len()).map(move |i| self.row(i))
    }

    /// Approximate heap memory size in bytes (mirrors C++ `MemorySize`).
    pub fn memory_size(&self) -> usize {
        self.data.len() * std::mem::size_of::<T>() + self.indptr.len() * std::mem::size_of::<i32>()
    }

    /// Raw element backing store (CSR `data` array).
    pub fn raw_data(&self) -> &[T] {
        &self.data
    }

    /// Raw row-offset array (CSR `indptr` array).
    pub fn raw_indptr(&self) -> &[i32] {
        &self.indptr
    }

    /// Reconstruct from raw CSR parts. Returns `None` if the parts are
    /// inconsistent (used by deserialization).
    pub fn from_raw(data: Vec<T>, indptr: Vec<i32>) -> Option<Self> {
        if indptr.is_empty() || indptr[0] != 0 {
            return None;
        }
        if *indptr.last().unwrap() as usize != data.len() {
            return None;
        }
        if indptr.windows(2).any(|w| w[0] > w[1]) {
            return None;
        }
        Some(Self { data, indptr })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_array() {
        let a: Compact2DArray<i32> = Compact2DArray::new();
        assert_eq!(a.len(), 0);
        assert!(a.is_empty());
        assert_eq!(a.total_elems(), 0);
    }

    #[test]
    fn push_and_read_rows() {
        let mut a: Compact2DArray<i32> = Compact2DArray::new();
        assert_eq!(a.push_row(&[1, 2, 3]), 0);
        assert_eq!(a.push_row(&[]), 1);
        assert_eq!(a.push_row(&[4]), 2);
        assert_eq!(a.len(), 3);
        assert_eq!(a.row(0), &[1, 2, 3]);
        assert_eq!(a.row(1), &[] as &[i32]);
        assert_eq!(a.row(2), &[4]);
        assert_eq!(a.total_elems(), 4);
    }

    #[test]
    fn iter_rows_yields_all() {
        let mut a: Compact2DArray<i32> = Compact2DArray::new();
        a.push_row(&[1]);
        a.push_row(&[2, 3]);
        let collected: Vec<Vec<i32>> = a.iter_rows().map(|r| r.to_vec()).collect();
        assert_eq!(collected, vec![vec![1], vec![2, 3]]);
    }

    #[test]
    fn raw_csr_roundtrip() {
        let mut a: Compact2DArray<i32> = Compact2DArray::new();
        a.push_row(&[1, 2]);
        a.push_row(&[3]);
        let data = a.raw_data().to_vec();
        let indptr = a.raw_indptr().to_vec();
        let b = Compact2DArray::from_raw(data, indptr).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn from_raw_rejects_inconsistent() {
        assert!(Compact2DArray::<i32>::from_raw(vec![1], vec![]).is_none());
        assert!(Compact2DArray::<i32>::from_raw(vec![1], vec![1, 1]).is_none());
        assert!(Compact2DArray::<i32>::from_raw(vec![1, 2], vec![0, 5]).is_none());
        assert!(Compact2DArray::<i32>::from_raw(vec![1], vec![0, 1]).is_some());
    }

    #[test]
    fn memory_size_nonzero() {
        let mut a: Compact2DArray<i32> = Compact2DArray::new();
        a.push_row(&[1, 2, 3]);
        assert!(a.memory_size() >= 3 * 4);
    }
}
