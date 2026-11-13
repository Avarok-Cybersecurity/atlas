// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: which RetroInfer index belongs to which sequence, and the
//! index cache that lets a new sequence reuse the index of a prefix it shares
//! (the regime the method pays in: one long document prefilled once, many
//! questions decoded against it). Host bookkeeping only; the device regions
//! live in the arena (`arena.rs`) and are freed by the caller with the
//! regions these functions hand back.
//!
//! Identity without scheduler help: a decode row is matched to a live
//! sequence by its host block table. A live sequence owns its writable tail
//! block, so the table it was last seen with is a prefix of exactly one row's
//! table (or of none). When the scheduler publishes sequence ids
//! (`sparse_seam::publish_step`), ids win and `on_seq_free` releases at once;
//! otherwise a sequence not seen for [`STALE_STEPS`] steps is released. A KV
//! write into a block some other sequence's index covers proves that sequence
//! ended (its blocks were reused), so it is dropped at once. A sequence seen
//! again after [`GAP_STEPS`], after a rollback deeper than a verify, or after a
//! KV write the host could not map to blocks (batched prefill), gives up its
//! indexes and re-installs them from the cache, whose entries are checked
//! against their content fingerprint before use (the caller does the device
//! read).
//!
//! The cache: every initial build is published as an immutable
//! [`IndexEntry`] behind an `Arc`, keyed by the model (pool address and
//! geometry), the attention layer, the build-parameter hash, and the physical
//! block chain of the indexed positions with its end. Physical blocks of a
//! live or prefix-cached sequence are immutable and shared only between
//! sequences with identical prefixes, so a key whose blocks prefix a new row's
//! table names the same keys; a block freed and reallocated in between is
//! caught by the content fingerprint. A sequence reads the shared entry until
//! its first update or append, which copies it into a private index
//! (copy-on-write). Entries are evicted least recently used when the arena
//! needs room and nobody holds them.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use super::arena::Region;
use super::state::{IndexCounts, LayerIndex};

/// Steps a live sequence may go unseen before it is released.
pub const STALE_STEPS: u64 = 4096;

/// Steps a live sequence may go unseen and still be trusted without a
/// fingerprint check (batches that leave a sequence out for a step or two).
pub const GAP_STEPS: u64 = 8;

/// Live sequences tracked at once (oldest released first beyond it).
pub const MAX_LIVE: usize = 256;

/// What an index-cache entry is valid for.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IndexKey {
    /// The serving model instance (pool address and geometry).
    pub model: u64,
    pub attn_layer: u32,
    /// `RetroParams::build_hash`.
    pub params: u64,
    /// Physical blocks covering `[0, end)`, in order.
    pub blocks: Vec<u32>,
    /// The index covers `[sink, end)`; block aligned.
    pub end: u32,
}

impl IndexKey {
    /// Whether this entry's blocks are a prefix of `table` and it ends by `max_end`.
    pub fn serves(
        &self,
        model: u64,
        attn_layer: u32,
        params: u64,
        table: &[u32],
        max_end: u32,
    ) -> bool {
        self.model == model
            && self.attn_layer == attn_layer
            && self.params == params
            && self.end <= max_end
            && table.len() >= self.blocks.len()
            && table[..self.blocks.len()] == self.blocks[..]
    }
}

/// Where a content fingerprint was read: the K bytes of one position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fingerprint {
    pub pos: u32,
    pub bytes: Vec<u8>,
}

/// An immutable, shareable index.
#[derive(Debug)]
pub struct IndexEntry {
    pub key: IndexKey,
    pub index: LayerIndex,
    pub region: Region,
    pub fingerprint: Fingerprint,
    pub build_secs: f64,
}

/// The index cache.
#[derive(Default)]
pub struct IndexStore {
    entries: Vec<(Arc<IndexEntry>, u64)>,
    tick: u64,
}

impl IndexStore {
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Device bytes of every entry.
    pub fn bytes(&self) -> usize {
        self.entries.iter().map(|(e, _)| e.region.bytes).sum()
    }
    /// Add an entry (an equal key replaces the older one, whose region is returned
    /// when nobody else holds it).
    pub fn insert(&mut self, e: Arc<IndexEntry>) -> Option<Region> {
        self.tick += 1;
        let mut freed = None;
        if let Some(i) = self.entries.iter().position(|(o, _)| o.key == e.key) {
            let (old, _) = self.entries.remove(i);
            if Arc::strong_count(&old) == 1 {
                freed = Some(old.region);
            }
        }
        self.entries.push((e, self.tick));
        freed
    }
    /// Entries that serve `table` up to `max_end`, longest first. Marks them used.
    pub fn candidates(
        &mut self,
        model: u64,
        attn_layer: u32,
        params: u64,
        table: &[u32],
        max_end: u32,
    ) -> Vec<Arc<IndexEntry>> {
        self.tick += 1;
        let tick = self.tick;
        let mut out: Vec<Arc<IndexEntry>> = Vec::new();
        for (e, t) in &mut self.entries {
            if e.key.serves(model, attn_layer, params, table, max_end) {
                *t = tick;
                out.push(e.clone());
            }
        }
        out.sort_by(|a, b| b.key.end.cmp(&a.key.end));
        out
    }
    /// Drop `e` (a failed fingerprint). Its region when nobody else holds it.
    pub fn discard(&mut self, e: &Arc<IndexEntry>) -> Option<Region> {
        let i = self.entries.iter().position(|(o, _)| Arc::ptr_eq(o, e))?;
        let (old, _) = self.entries.remove(i);
        drop(old);
        (Arc::strong_count(e) == 1).then_some(e.region)
    }
    /// Evict the least recently used entry nobody else holds; its region.
    pub fn evict_one(&mut self) -> Option<Region> {
        let i = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, (e, _))| Arc::strong_count(e) == 1)
            .min_by_key(|(_, (_, t))| *t)
            .map(|(i, _)| i)?;
        Some(self.entries.remove(i).0.region)
    }
    /// Forget everything (a new backend: the old arena is gone with it).
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl IndexStore {
    /// Drop every entry covering one of `blocks` (a KV write into a block
    /// proves its old content is gone). Regions of entries nobody holds.
    pub fn invalidate_blocks(&mut self, blocks: &[u32]) -> Vec<Region> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.entries.len() {
            let hit = blocks
                .iter()
                .any(|b| self.entries[i].0.key.blocks.contains(b));
            if hit {
                let (e, _) = self.entries.swap_remove(i);
                if Arc::strong_count(&e) == 1 {
                    out.push(e.region);
                }
            } else {
                i += 1;
            }
        }
        out
    }
}

/// One attention layer's index state of a live sequence.
#[derive(Debug, Default)]
pub struct LayerSlot {
    /// The shared index it reads until its first update or append.
    pub base: Option<Arc<IndexEntry>>,
    /// Its private index after copy-on-write, with the region to free.
    pub own: Option<(LayerIndex, Region)>,
    /// The content fingerprint of the slot's index (K at one indexed position).
    pub fingerprint: Option<Fingerprint>,
}

impl LayerSlot {
    /// The index decode reads.
    pub fn index(&self) -> Option<&LayerIndex> {
        self.own
            .as_ref()
            .map(|(i, _)| i)
            .or(self.base.as_ref().map(|e| &e.index))
    }
    pub fn counts(&self) -> Option<IndexCounts> {
        self.index().map(|i| i.counts)
    }
    /// Release the slot: the private region, and the shared entry's region when
    /// this was its last holder outside the store (the store keeps a reference
    /// while the entry is cached, so that is only for discarded entries).
    pub fn release(self) -> Vec<Region> {
        let mut out = Vec::new();
        if let Some((_, r)) = self.own {
            out.push(r);
        }
        if let Some(e) = self.base
            && Arc::strong_count(&e) == 1
        {
            out.push(e.region);
        }
        out
    }
}

/// A sequence retro has seen.
#[derive(Debug, Default)]
pub struct LiveSeq {
    pub uid: u64,
    pub seq_id: Option<u64>,
    /// Host block table at the last sighting.
    pub table: Vec<u32>,
    pub len: u32,
    pub last_step: u64,
    pub layers: BTreeMap<u32, LayerSlot>,
    /// Physical blocks its indexes cover (positions below their `indexed_end`).
    pub covered: HashSet<u32>,
    /// A KV write the host could not map to blocks happened since it was last
    /// seen: its indexes are re-checked against their fingerprints.
    pub suspect: bool,
    /// No index can be built for it (no room); it runs dense until it ends.
    pub dense_only: bool,
}

impl LiveSeq {
    /// Release every layer's regions.
    pub fn release(&mut self) -> Vec<Region> {
        std::mem::take(&mut self.layers)
            .into_values()
            .flat_map(LayerSlot::release)
            .collect()
    }
}

/// The live sequences.
#[derive(Default)]
pub struct SeqTable {
    seqs: Vec<LiveSeq>,
    next_uid: u64,
}

/// How a row was matched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sighting {
    /// Seen in the previous step (or matched by id): trusted.
    Continuing(usize),
    /// Matched after a gap: the caller re-checks its fingerprint.
    Returning(usize),
    /// A sequence retro has not seen.
    New(usize),
}

impl Sighting {
    pub fn index(self) -> usize {
        match self {
            Sighting::Continuing(i) | Sighting::Returning(i) | Sighting::New(i) => i,
        }
    }
}

impl SeqTable {
    pub fn len(&self) -> usize {
        self.seqs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.seqs.is_empty()
    }
    pub fn get(&self, i: usize) -> &LiveSeq {
        &self.seqs[i]
    }
    pub fn get_mut(&mut self, i: usize) -> &mut LiveSeq {
        &mut self.seqs[i]
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut LiveSeq> {
        self.seqs.iter_mut()
    }
    /// The current index of the entry with internal id `uid` (indexes move when
    /// entries are removed; step plans hold uids).
    pub fn by_uid(&self, uid: u64) -> Option<usize> {
        self.seqs.iter().position(|s| s.uid == uid)
    }
    pub fn uid_mut(&mut self, uid: u64) -> Option<&mut LiveSeq> {
        self.seqs.iter_mut().find(|s| s.uid == uid)
    }

    fn find(&self, seq_id: Option<u64>, table: &[u32]) -> Option<usize> {
        if let Some(id) = seq_id
            && let Some(i) = self.seqs.iter().position(|s| s.seq_id == Some(id))
        {
            return Some(i);
        }
        self.seqs
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                (seq_id.is_none() || s.seq_id.is_none())
                    && !s.table.is_empty()
                    && table.len() >= s.table.len()
                    && table[..s.table.len()] == s.table[..]
            })
            .max_by_key(|(_, s)| (s.table.len(), s.last_step))
            .map(|(i, _)| i)
    }

    /// Match a row (or create its entry) at `step`, and record the sighting.
    /// Two rows of one step never share an entry: `taken` lists the entries the
    /// step already matched, and a row that would match one of them is new.
    pub fn sight(
        &mut self,
        seq_id: Option<u64>,
        table: &[u32],
        len: u32,
        step: u64,
        taken: &[usize],
    ) -> Sighting {
        let found = self.find(seq_id, table).filter(|i| !taken.contains(i));
        let s = match found {
            Some(i) => {
                let s = &self.seqs[i];
                let gap = step.saturating_sub(s.last_step) > GAP_STEPS && seq_id.is_none();
                let rolled = len + super::config::MAX_VERIFY_ROWS < s.len && seq_id.is_none();
                if gap || rolled || s.suspect {
                    Sighting::Returning(i)
                } else {
                    Sighting::Continuing(i)
                }
            }
            None => {
                self.next_uid += 1;
                self.seqs.push(LiveSeq {
                    uid: self.next_uid,
                    seq_id,
                    ..Default::default()
                });
                Sighting::New(self.seqs.len() - 1)
            }
        };
        let e = &mut self.seqs[s.index()];
        e.table = table.to_vec();
        e.len = len;
        e.last_step = step;
        if seq_id.is_some() {
            e.seq_id = seq_id;
        }
        s
    }

    /// Remove every entry other than `writer` whose indexes cover one of
    /// `blocks`: a KV write into a covered block means that sequence's blocks
    /// were freed and reused, so it has ended.
    pub fn invalidate_blocks(&mut self, blocks: &[u32], writer: Option<u64>) -> Vec<LiveSeq> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.seqs.len() {
            let s = &self.seqs[i];
            if Some(s.uid) != writer && blocks.iter().any(|b| s.covered.contains(b)) {
                out.push(self.seqs.swap_remove(i));
            } else {
                i += 1;
            }
        }
        out
    }

    /// Mark every entry except `keep` suspect (a KV write the host could not map).
    pub fn mark_suspect_except(&mut self, keep: &[u64]) {
        for s in &mut self.seqs {
            if !keep.contains(&s.uid) && !s.layers.is_empty() {
                s.suspect = true;
            }
        }
    }

    /// Remove entry `i` (its regions are the caller's to free).
    pub fn remove(&mut self, i: usize) -> LiveSeq {
        self.seqs.swap_remove(i)
    }

    /// Remove the entry with scheduler id `id`.
    pub fn remove_id(&mut self, id: u64) -> Option<LiveSeq> {
        let i = self.seqs.iter().position(|s| s.seq_id == Some(id))?;
        Some(self.remove(i))
    }

    /// Entries unseen since before `step - STALE_STEPS`, plus the oldest ones
    /// beyond [`MAX_LIVE`]; removed and returned.
    pub fn sweep(&mut self, step: u64) -> Vec<LiveSeq> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.seqs.len() {
            if step.saturating_sub(self.seqs[i].last_step) > STALE_STEPS {
                out.push(self.seqs.swap_remove(i));
            } else {
                i += 1;
            }
        }
        while self.seqs.len() > MAX_LIVE {
            let oldest = (0..self.seqs.len())
                .min_by_key(|&i| self.seqs[i].last_step)
                .unwrap_or(0);
            out.push(self.seqs.swap_remove(oldest));
        }
        out
    }

    /// The least recently seen entry not seen at `step` that holds device
    /// regions; removed and returned (the arena needs room).
    pub fn evict_idle(&mut self, step: u64) -> Option<LiveSeq> {
        let i = (0..self.seqs.len())
            .filter(|&i| self.seqs[i].last_step < step && !self.seqs[i].layers.is_empty())
            .min_by_key(|&i| self.seqs[i].last_step)?;
        Some(self.seqs.swap_remove(i))
    }

    /// Forget everything (a new backend).
    pub fn clear(&mut self) {
        self.seqs.clear();
    }
}
