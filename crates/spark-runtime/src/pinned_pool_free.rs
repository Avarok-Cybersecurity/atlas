// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/memory_allocators/tensor_memory_allocator.py (free list with
// coalescing and alignment), commit 3aefae1, Apache-2.0; no code copied. Changes: best-fit by a
// size index, and extents never cross a pinning chunk, by Atlas contributors.

//! Responsibility: the [`crate::pinned_pool::PinnedPool`] free list: best-fit allocation of page
//! runs over pinned chunks, coalescing on free, and the table of live runs.
//!
//! Offsets are relative to the pool base. Every extent and every run lies inside ONE pinning chunk
//! (`[k * chunk, (k + 1) * chunk)`): each chunk is page-locked by its own `host_register` call, so
//! a run that straddled two would be pinned by two registrations, which `pinned_hosts::is_pinned`
//! (one containing region) would not report and a driver need not treat as one locked range.
//! Coalescing therefore stops at chunk boundaries.

use std::collections::{BTreeMap, BTreeSet, HashMap};

pub(crate) struct FreeList {
    chunk: usize,
    /// Free extents: offset -> length.
    by_off: BTreeMap<usize, usize>,
    /// The same extents as `(length, offset)`, for best fit.
    by_size: BTreeSet<(usize, usize)>,
    /// Live runs: offset -> (length, id).
    live: HashMap<usize, (usize, u64)>,
    free_bytes: usize,
}

impl FreeList {
    pub(crate) fn new(chunk: usize) -> Self {
        Self {
            chunk,
            by_off: BTreeMap::new(),
            by_size: BTreeSet::new(),
            live: HashMap::new(),
            free_bytes: 0,
        }
    }

    fn chunk_of(&self, off: usize) -> usize {
        off / self.chunk
    }

    fn insert_extent(&mut self, off: usize, len: usize) {
        self.by_off.insert(off, len);
        self.by_size.insert((len, off));
    }

    fn remove_extent(&mut self, off: usize, len: usize) {
        self.by_off.remove(&off);
        self.by_size.remove(&(len, off));
    }

    /// Make the newly pinned chunk at `off` (a multiple of the chunk size) allocatable.
    pub(crate) fn add_chunk(&mut self, off: usize) {
        debug_assert!(off.is_multiple_of(self.chunk));
        self.insert_extent(off, self.chunk);
        self.free_bytes += self.chunk;
    }

    /// Best fit for `need` bytes (a page multiple, at most one chunk): the smallest free extent
    /// that holds it, lowest offset first among equals. Returns the run's offset.
    pub(crate) fn alloc(&mut self, need: usize, id: u64) -> Option<usize> {
        if need == 0 || need > self.chunk {
            return None;
        }
        let &(len, off) = self.by_size.range((need, 0)..).next()?;
        self.remove_extent(off, len);
        if len > need {
            self.insert_extent(off + need, len - need);
        }
        self.live.insert(off, (need, id));
        self.free_bytes -= need;
        Some(off)
    }

    /// Length of the live run at `off` when its id is `id`.
    pub(crate) fn live_len(&self, off: usize, id: u64) -> Option<usize> {
        match self.live.get(&off) {
            Some(&(len, live_id)) if live_id == id => Some(len),
            _ => None,
        }
    }

    /// Return the run at `off` (id `id`) and coalesce it with free neighbors in the same chunk.
    /// `None` when no such live run exists (a double free or a foreign run), changing nothing.
    pub(crate) fn free(&mut self, off: usize, id: u64) -> Option<usize> {
        let len = self.live_len(off, id)?;
        self.live.remove(&off);
        self.free_bytes += len;
        let (mut start, mut end) = (off, off + len);
        let prev = self
            .by_off
            .range(..start)
            .next_back()
            .map(|(&o, &l)| (o, l));
        if let Some((p, pl)) = prev
            && p + pl == start
            && self.chunk_of(p) == self.chunk_of(start)
        {
            self.remove_extent(p, pl);
            start = p;
        }
        let next = self.by_off.get(&end).copied();
        if let Some(nl) = next
            && self.chunk_of(end) == self.chunk_of(off)
        {
            self.remove_extent(end, nl);
            end += nl;
        }
        self.insert_extent(start, end - start);
        Some(len)
    }

    pub(crate) fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    pub(crate) fn largest_free(&self) -> usize {
        self.by_size.iter().next_back().map_or(0, |&(len, _)| len)
    }

    pub(crate) fn live_runs(&self) -> usize {
        self.live.len()
    }

    /// Free extents in offset order (tests and invariant checks).
    #[cfg(test)]
    pub(crate) fn extents(&self) -> Vec<(usize, usize)> {
        self.by_off.iter().map(|(&o, &l)| (o, l)).collect()
    }
}
