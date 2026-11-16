// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the host store of Quest page records for the window desk's scorer (port lane
//! L30, request R3 of lane L36): one record per demoted page, per sequence, bounded in bytes with
//! least-recently-used eviction. A sequence is keyed by its first physical block, the desk's own
//! key. Pure bookkeeping (no GPU), tested in `scorer_tests.rs`.

use std::collections::{BTreeMap, HashMap};

use super::scorer_rank::PageRecord;

/// Host bytes of page summaries kept before the least recently used sequence is dropped.
pub const HOST_CAP_BYTES: usize = 1 << 30;

/// The records of one sequence, by logical page index.
#[derive(Default)]
struct SeqPages {
    touched: u64,
    pages: BTreeMap<u32, PageRecord>,
}

/// Records of every sequence the desk demoted pages of.
pub(super) struct HostStore {
    generation: u64,
    clock: u64,
    bytes: usize,
    cap: usize,
    seqs: HashMap<u32, SeqPages>,
}

impl HostStore {
    pub(super) fn new(cap: usize) -> Self {
        Self {
            generation: 0,
            clock: 0,
            bytes: 0,
            cap,
            seqs: HashMap::new(),
        }
    }

    /// Bytes of records held.
    #[cfg(test)]
    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }

    /// A new model build invalidates every record.
    pub(super) fn reset_for(&mut self, generation: u64) {
        if self.generation != generation {
            *self = Self::new(self.cap);
            self.generation = generation;
        }
    }

    /// Store (or, with `None`, forget) the record of `page` of the sequence keyed `block0`.
    pub(super) fn put(&mut self, block0: u32, page: u32, rec: Option<PageRecord>) {
        self.clock += 1;
        let clock = self.clock;
        let seq = self.seqs.entry(block0).or_default();
        seq.touched = clock;
        let old = match rec {
            Some(r) => {
                self.bytes += r.len() * 2;
                seq.pages.insert(page, r)
            }
            None => seq.pages.remove(&page),
        };
        if let Some(o) = old {
            self.bytes -= o.len() * 2;
        }
    }

    /// The stored records among the `wanted` pages, in the order asked.
    pub(super) fn records(&mut self, block0: u32, wanted: &[u32]) -> Vec<(u32, PageRecord)> {
        self.clock += 1;
        let clock = self.clock;
        let Some(seq) = self.seqs.get_mut(&block0) else {
            return Vec::new();
        };
        seq.touched = clock;
        wanted
            .iter()
            .filter_map(|p| seq.pages.get(p).map(|r| (*p, r.clone())))
            .collect()
    }

    /// Drop least recently used sequences (never `keep`) until under the cap.
    pub(super) fn evict(&mut self, keep: u32) {
        while self.bytes > self.cap {
            let victim = self
                .seqs
                .iter()
                .filter(|(k, _)| **k != keep)
                .min_by_key(|(_, s)| s.touched)
                .map(|(k, _)| *k);
            let Some(v) = victim else {
                break;
            };
            if let Some(s) = self.seqs.remove(&v) {
                self.bytes -= s.pages.values().map(|p| p.len() * 2).sum::<usize>();
            }
        }
    }
}
