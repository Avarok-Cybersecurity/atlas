// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/cache_engine.py (store the blocks a prefill newly cached) and lmcache/integration/vllm/lazy_offload_policy/eviction_aware.py (lazy stores released near the eviction head) (commit 3aefae1, Apache-2.0); no code copied.
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the host tier's store side. [`Live::store_inserted`] hands the blocks a
//! prefill insert newly cached to the tier and takes one extra block reference per started (or
//! lazily buffered) copy; [`Live::reap`] drops those references when the tier reports the copy
//! finished; [`Live::tick`] runs once per scheduler step (the write-back budget, the reap, and
//! in lazy store mode the drain with this file's [`HeldView`] of the GPU eviction order).
//!
//! Why the extra reference: the D2H reads the block on the tier's copy stream after the
//! prefill returned. Without it, a radix eviction in that window would return the block to
//! the free list and the next allocation could overwrite it while the DMA still reads it. With
//! it, eviction only drops the cache's reference and the block is freed when the copy is done.
//!
//! The lazy mode keeps the same reference while the store is buffered, so a buffered block's
//! device regions stay valid exactly as lane L26a's address rule requires: a block can never
//! be recycled while the tier may still copy it. The reference count then also tells the drain
//! where the block stands: only the glue's reference left means the radix already evicted it
//! (copy it now), the glue's plus the cache's means it is cached and evictable, more means a
//! live sequence uses it.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::kv_host_tier::{BlockKey, DrainSignals, EvictionView, GpuBlockState, StoreMode};

use super::state::{Books, Held, Live};

/// Weight of the newest tick in the smoothed block demand of the lazy drain.
const DEMAND_ALPHA: f64 = 0.25;

impl Live {
    /// Store the full blocks of `tokens` whose radix nodes the insert just created
    /// (`acquired`, from `InsertAcquired::blocks`). `table[i]` holds tokens
    /// `[i * bs, (i + 1) * bs)`. Returns how many copies were started or buffered.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn store_inserted(
        &self,
        gpu: &dyn GpuBackend,
        kv: &mut PagedKvCache,
        tokens: &[u32],
        table: &[u32],
        adapter: u64,
        acquired: &[u32],
        stream: u64,
    ) -> Result<usize> {
        let bs = self.block_size;
        if adapter != 0 || bs == 0 || acquired.is_empty() {
            return Ok(0);
        }
        let picks = new_block_positions(tokens.len(), bs, table, acquired);
        let Some(&last) = picks.last() else {
            return Ok(0);
        };
        let keys = BlockKey::chain_blocks(self.ns, 0, &tokens[..(last + 1) * bs], bs);
        let mut sel_keys = Vec::with_capacity(picks.len());
        let mut sel_tokens: Vec<&[u32]> = Vec::with_capacity(picks.len());
        let mut regions = Vec::with_capacity(picks.len());
        for &i in &picks {
            sel_keys.push(keys[i]);
            sel_tokens.push(&tokens[i * bs..(i + 1) * bs]);
            regions.push(kv.checked_block_regions(table[i])?);
        }
        self.tier.set_source_stream(stream);
        let report = self
            .tier
            .store_detailed(gpu, &sel_keys, &sel_tokens, &regions)?;
        let mut b = self.books.lock();
        let mut taken = 0;
        for (&j, lazy) in report
            .started
            .iter()
            .map(|j| (j, false))
            .chain(report.buffered.iter().map(|j| (j, true)))
        {
            let key = sel_keys[j];
            if b.held.contains_key(&key) {
                continue;
            }
            let block = table[picks[j]];
            kv.inc_ref(block);
            let order = Live::stamp(&mut b);
            b.held.insert(key, Held { block, lazy, order });
            taken += 1;
        }
        Ok(taken)
    }

    /// Drop the block reference of every write-back the tier reports finished.
    pub(crate) fn reap(&self, gpu: &dyn GpuBackend, kv: &mut PagedKvCache) {
        let done = self.tier.poll_writebacks(gpu);
        if done.is_empty() {
            return;
        }
        let mut b = self.books.lock();
        for k in done {
            if let Some(h) = b.held.remove(&k) {
                kv.dec_ref(h.block);
            }
        }
    }

    /// Once per scheduler step: reset the write-back budget, reap, and drain lazy stores.
    pub(crate) fn tick(&self, gpu: &dyn GpuBackend, kv: &mut PagedKvCache) {
        self.tier.begin_step();
        self.reap(gpu, kv);
        if self.tier.options().store_mode == StoreMode::Lazy
            && let Err(e) = self.drain(gpu, kv)
        {
            self.warn_once(&format!("kv_host_tier lazy drain failed: {e:#}"));
        }
    }

    /// Start the buffered stores whose blocks approach eviction, then drop the references of
    /// buffered keys the tier no longer holds (every buffered entry is asked for its state at
    /// each drain, so a held lazy key that was not asked about left the buffer without a copy).
    fn drain(&self, gpu: &dyn GpuBackend, kv: &mut PagedKvCache) -> Result<()> {
        let mut b = self.books.lock();
        let free = kv.num_free_blocks();
        let used = b
            .last_free
            .map(|prev| prev.saturating_sub(free))
            .unwrap_or(0);
        b.last_free = Some(free);
        b.demand = (1.0 - DEMAND_ALPHA) * b.demand + DEMAND_ALPHA * used as f64;
        let signals = DrainSignals {
            blocks_per_step: b.demand.max(1.0),
        };
        if !b.held.values().any(|h| h.lazy) {
            return Ok(());
        }
        let view = HeldView::new(&b, kv);
        let started = self.tier.drain_lazy(gpu, signals, &view)?;
        let asked = view.into_asked();
        for k in &started {
            if let Some(h) = b.held.get_mut(k) {
                h.lazy = false;
            }
        }
        let gone: Vec<BlockKey> = b
            .held
            .iter()
            .filter(|(k, h)| h.lazy && !asked.contains(*k))
            .map(|(k, _)| *k)
            .collect();
        for k in gone {
            if let Some(h) = b.held.remove(&k) {
                kv.dec_ref(h.block);
            }
        }
        Ok(())
    }
}

/// Table positions `i < min(len / bs, table.len())` whose block is in `acquired`, ascending.
pub(crate) fn new_block_positions(
    len: usize,
    bs: usize,
    table: &[u32],
    acquired: &[u32],
) -> Vec<usize> {
    if bs == 0 {
        return Vec::new();
    }
    let want: HashSet<u32> = acquired.iter().copied().collect();
    let full = (len / bs).min(table.len());
    (0..full).filter(|&i| want.contains(&table[i])).collect()
}

/// The GPU eviction order of the held lazy blocks, as the reference counts tell it.
pub(crate) struct HeldView {
    states: HashMap<BlockKey, GpuBlockState>,
    asked: RefCell<HashSet<BlockKey>>,
}

impl HeldView {
    pub(crate) fn new(b: &Books, kv: &PagedKvCache) -> Self {
        let mut queued: Vec<(u64, BlockKey)> = Vec::new();
        let mut states = HashMap::new();
        for (k, h) in b.held.iter().filter(|(_, h)| h.lazy) {
            match kv.ref_count(h.block) {
                // Only the glue's reference: the radix evicted the block; copy it first.
                0 | 1 => {
                    states.insert(*k, GpuBlockState::Queued { rank: 0 });
                }
                2 => queued.push((h.order, *k)),
                _ => {
                    states.insert(*k, GpuBlockState::InUse);
                }
            }
        }
        // Cached and evictable: the radix evicts least recently used first, and an older store
        // is the better guess for a less recently used block (the tree's own LRU order is not
        // exposed). Rank 0 stays reserved for blocks already evicted.
        queued.sort_unstable();
        for (rank, (_, k)) in queued.into_iter().enumerate() {
            states.insert(k, GpuBlockState::Queued { rank: rank + 1 });
        }
        Self {
            states,
            asked: RefCell::new(HashSet::new()),
        }
    }

    pub(crate) fn into_asked(self) -> HashSet<BlockKey> {
        self.asked.into_inner()
    }
}

impl EvictionView for HeldView {
    fn state(&self, key: &BlockKey) -> GpuBlockState {
        self.asked.borrow_mut().insert(*key);
        // A buffered key the glue holds no reference on has no address guarantee: gone.
        self.states.get(key).copied().unwrap_or(GpuBlockState::Gone)
    }
}
