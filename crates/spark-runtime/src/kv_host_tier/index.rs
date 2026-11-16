// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/distributed/l1_manager/base.py:132-188 (commit 3aefae1, Apache-2.0; object states staging -> ready -> read-locked); no code copied.
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the host tier's entry table. One [`Entry`] per stored block: its key, its
//! tokens (compared on every hit, so a forged or colliding key never serves another prefix's
//! KV), the host run holding its bytes, its state (STAGING while the D2H copy may still be
//! writing the run, READY after the copy's event completed) and its read-lease count (restores
//! whose H2D copy may still be reading the run). The table keeps the byte totals the watermark,
//! the in-flight cap and the gauges need, so none of them scans.
//!
//! Rules enforced here: only READY entries are visible to a probe; an entry with a lease, or a
//! STAGING one, can never be removed through [`Index::remove_evictable`]; a lease can only be
//! taken on a READY entry.

use std::collections::HashMap;

use super::arena::HostRun;
use super::fingerprint::BlockKey;

/// Where an entry is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EntryState {
    /// The D2H copy into the run may still be in flight (write-back batch `batch`).
    Staging { batch: u64 },
    /// The bytes in the run are complete and immutable.
    Ready,
}

/// One stored block.
#[derive(Debug)]
pub(crate) struct Entry {
    pub(crate) tokens: Box<[u32]>,
    pub(crate) run: HostRun,
    /// Bytes this entry charges against the budget (the tier's block size).
    pub(crate) bytes: u64,
    pub(crate) state: EntryState,
    /// Restores whose H2D copy may still read the run.
    pub(crate) leases: u32,
}

impl Entry {
    pub(crate) fn evictable(&self) -> bool {
        self.state == EntryState::Ready && self.leases == 0
    }
}

/// What a probe found for one key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Probe {
    /// READY and the stored tokens equal the probe's tokens: a hit.
    Ready,
    /// Present but still STAGING (its copy has not completed): a miss for now.
    Staging,
    /// Present but with other tokens (a forged or colliding key): a miss.
    Mismatch,
    /// Not in host RAM.
    Absent,
}

/// The entry table and its byte totals.
#[derive(Default)]
pub(crate) struct Index {
    map: HashMap<BlockKey, Entry>,
    used: u64,
    staging: u64,
    leased: u64,
}

impl Index {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }

    /// Bytes held by every entry (staging and ready).
    pub(crate) fn used_bytes(&self) -> u64 {
        self.used
    }

    /// Bytes of STAGING entries (in-flight D2H).
    pub(crate) fn staging_bytes(&self) -> u64 {
        self.staging
    }

    /// Bytes of entries with at least one lease (in-flight H2D).
    pub(crate) fn leased_bytes(&self) -> u64 {
        self.leased
    }

    pub(crate) fn get(&self, k: &BlockKey) -> Option<&Entry> {
        self.map.get(k)
    }

    /// Classify `k` against the probe's `tokens`.
    pub(crate) fn probe(&self, k: &BlockKey, tokens: &[u32]) -> Probe {
        match self.map.get(k) {
            None => Probe::Absent,
            Some(e) if *e.tokens != *tokens => Probe::Mismatch,
            Some(e) => match e.state {
                EntryState::Ready => Probe::Ready,
                EntryState::Staging { .. } => Probe::Staging,
            },
        }
    }

    /// Add a STAGING entry. Returns the run back when `k` is already present (nothing changed).
    pub(crate) fn insert_staging(
        &mut self,
        k: BlockKey,
        tokens: &[u32],
        run: HostRun,
        bytes: u64,
        batch: u64,
    ) -> Result<(), HostRun> {
        self.insert(k, tokens, run, bytes, EntryState::Staging { batch })
    }

    /// Add a READY entry (a block promoted from the cold store).
    pub(crate) fn insert_ready(
        &mut self,
        k: BlockKey,
        tokens: &[u32],
        run: HostRun,
        bytes: u64,
    ) -> Result<(), HostRun> {
        self.insert(k, tokens, run, bytes, EntryState::Ready)
    }

    fn insert(
        &mut self,
        k: BlockKey,
        tokens: &[u32],
        run: HostRun,
        bytes: u64,
        state: EntryState,
    ) -> Result<(), HostRun> {
        if self.map.contains_key(&k) {
            return Err(run);
        }
        self.used += bytes;
        if matches!(state, EntryState::Staging { .. }) {
            self.staging += bytes;
        }
        self.map.insert(
            k,
            Entry {
                tokens: tokens.into(),
                run,
                bytes,
                state,
                leases: 0,
            },
        );
        Ok(())
    }

    /// STAGING -> READY for the entry of batch `batch`. False when `k` is not that batch's
    /// staging entry (nothing changed).
    pub(crate) fn publish(&mut self, k: &BlockKey, batch: u64) -> bool {
        match self.map.get_mut(k) {
            Some(e) if e.state == (EntryState::Staging { batch }) => {
                e.state = EntryState::Ready;
                self.staging -= e.bytes;
                true
            }
            _ => false,
        }
    }

    /// Remove the STAGING entry of batch `batch` (its copy failed or was abandoned after a
    /// synchronize). The caller frees the returned run.
    pub(crate) fn remove_staging(&mut self, k: &BlockKey, batch: u64) -> Option<Entry> {
        match self.map.get(k) {
            Some(e) if e.state == (EntryState::Staging { batch }) => {}
            _ => return None,
        }
        let e = self.map.remove(k)?;
        self.used -= e.bytes;
        self.staging -= e.bytes;
        Some(e)
    }

    /// Remove a READY entry without leases (eviction). None for any other entry.
    pub(crate) fn remove_evictable(&mut self, k: &BlockKey) -> Option<Entry> {
        if !self.map.get(k)?.evictable() {
            return None;
        }
        let e = self.map.remove(k)?;
        self.used -= e.bytes;
        Some(e)
    }

    /// Whether `k` is READY without leases.
    pub(crate) fn is_evictable(&self, k: &BlockKey) -> bool {
        self.map.get(k).is_some_and(Entry::evictable)
    }

    /// Take a read lease on a READY entry. False (nothing changed) for any other entry.
    pub(crate) fn lease(&mut self, k: &BlockKey) -> bool {
        match self.map.get_mut(k) {
            Some(e) if e.state == EntryState::Ready => {
                if e.leases == 0 {
                    self.leased += e.bytes;
                }
                e.leases += 1;
                true
            }
            _ => false,
        }
    }

    /// Drop one lease taken by [`Self::lease`].
    pub(crate) fn unlease(&mut self, k: &BlockKey) {
        let Some(e) = self.map.get_mut(k) else {
            return;
        };
        if e.leases > 0 {
            e.leases -= 1;
            if e.leases == 0 {
                self.leased -= e.bytes;
            }
        }
    }

    /// Remove every entry (teardown after a full quiesce). The caller frees the runs.
    pub(crate) fn drain_all(&mut self) -> Vec<(BlockKey, Entry)> {
        self.used = 0;
        self.staging = 0;
        self.leased = 0;
        self.map.drain().collect()
    }

    /// Remove every entry no DMA can touch (READY, no lease) and return them; the rest stay.
    pub(crate) fn drain_quiescent(&mut self) -> Vec<(BlockKey, Entry)> {
        let keys: Vec<BlockKey> = self
            .map
            .iter()
            .filter(|(_, e)| e.evictable())
            .map(|(k, _)| *k)
            .collect();
        keys.into_iter()
            .filter_map(|k| self.remove_evictable(&k).map(|e| (k, e)))
            .collect()
    }
}
