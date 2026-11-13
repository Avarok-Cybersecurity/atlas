// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the host-side allocator of the retro device arena. Every
//! retro device byte (live indexes, the index cache, build scratch, zone lists,
//! the exact buffer) comes out of ONE allocation of `--retro-index-cache-mb`
//! MiB, made on the first eager hook after the model build and charged to the
//! KV-block budget at registration, so `--gpu-memory-utilization` stays honest
//! and nothing is allocated while a CUDA graph is captured.
//!
//! First fit over a sorted free list of 256-byte aligned ranges; frees merge
//! with their neighbors. Device work that reads a freed range was queued on the
//! same stream as any later writer of it, so reuse is stream-ordered.

use std::collections::BTreeMap;

use super::state::ALIGN;

/// A range of the arena: byte offset and length (both `ALIGN` multiples).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Region {
    pub offset: usize,
    pub bytes: usize,
}

/// The allocator. Offsets are relative to the arena's base pointer.
#[derive(Debug)]
pub struct Arena {
    size: usize,
    /// offset -> length of every free range, non-adjacent.
    free: BTreeMap<usize, usize>,
    used: usize,
    peak: usize,
}

impl Arena {
    /// An empty arena of `size` bytes (rounded down to `ALIGN`).
    pub fn new(size: usize) -> Self {
        let size = size / ALIGN * ALIGN;
        let mut free = BTreeMap::new();
        if size > 0 {
            free.insert(0, size);
        }
        Self {
            size,
            free,
            used: 0,
            peak: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Bytes handed out and not yet freed.
    pub fn used(&self) -> usize {
        self.used
    }

    pub fn peak(&self) -> usize {
        self.peak
    }

    /// The largest single allocation that would succeed now.
    pub fn largest_free(&self) -> usize {
        self.free.values().copied().max().unwrap_or(0)
    }

    /// Allocate `bytes` (rounded up to `ALIGN`); `None` when no free range fits.
    pub fn alloc(&mut self, bytes: usize) -> Option<Region> {
        let bytes = bytes.max(1).div_ceil(ALIGN) * ALIGN;
        let (&off, &len) = self.free.iter().find(|(_, len)| **len >= bytes)?;
        self.free.remove(&off);
        if len > bytes {
            self.free.insert(off + bytes, len - bytes);
        }
        self.used += bytes;
        self.peak = self.peak.max(self.used);
        Some(Region { offset: off, bytes })
    }

    /// Return a region. Double frees and foreign regions are refused (a bug
    /// in the caller), so the free list never overlaps.
    pub fn free(&mut self, r: Region) -> bool {
        if r.bytes == 0 || r.offset + r.bytes > self.size || !r.offset.is_multiple_of(ALIGN) {
            return false;
        }
        // Must not overlap any free range.
        if let Some((&o, &l)) = self.free.range(..=r.offset).next_back()
            && o + l > r.offset
        {
            return false;
        }
        if let Some((&o, _)) = self.free.range(r.offset..).next()
            && o < r.offset + r.bytes
        {
            return false;
        }
        let mut off = r.offset;
        let mut len = r.bytes;
        if let Some((&o, &l)) = self.free.range(..off).next_back()
            && o + l == off
        {
            self.free.remove(&o);
            off = o;
            len += l;
        }
        if let Some(&l) = self.free.get(&(r.offset + r.bytes)) {
            self.free.remove(&(r.offset + r.bytes));
            len += l;
        }
        self.free.insert(off, len);
        self.used -= r.bytes;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_rounds_and_tracks_use() {
        let mut a = Arena::new(10 * ALIGN + 17);
        assert_eq!(a.size(), 10 * ALIGN);
        let r = a.alloc(1).unwrap();
        assert_eq!(
            r,
            Region {
                offset: 0,
                bytes: ALIGN
            }
        );
        let s = a.alloc(ALIGN + 1).unwrap();
        assert_eq!(s.offset, ALIGN);
        assert_eq!(s.bytes, 2 * ALIGN);
        assert_eq!(a.used(), 3 * ALIGN);
        assert!(a.alloc(8 * ALIGN).is_none());
        assert_eq!(a.largest_free(), 7 * ALIGN);
    }

    #[test]
    fn frees_merge_and_refuse_double_free() {
        let mut a = Arena::new(8 * ALIGN);
        let r: Vec<Region> = (0..4).map(|_| a.alloc(2 * ALIGN).unwrap()).collect();
        assert!(a.alloc(1).is_none());
        assert!(a.free(r[1]));
        assert!(!a.free(r[1]), "double free refused");
        assert!(a.free(r[2]));
        assert_eq!(a.largest_free(), 4 * ALIGN, "neighbors merged");
        assert!(a.free(r[0]));
        assert!(a.free(r[3]));
        assert_eq!(a.used(), 0);
        assert_eq!(a.largest_free(), 8 * ALIGN);
        assert_eq!(a.peak(), 8 * ALIGN);
    }

    #[test]
    fn foreign_regions_are_refused() {
        let mut a = Arena::new(4 * ALIGN);
        assert!(!a.free(Region {
            offset: 8 * ALIGN,
            bytes: ALIGN
        }));
        assert!(!a.free(Region {
            offset: 3,
            bytes: ALIGN
        }));
        assert!(
            !a.free(Region {
                offset: 0,
                bytes: ALIGN
            }),
            "range is still free"
        );
    }
}
