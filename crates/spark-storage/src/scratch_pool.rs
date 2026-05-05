// SPDX-License-Identifier: AGPL-3.0-only
//
// HBM scratch pool for the high-speed-swap path. Holds N "slots", each large
// enough for one block worth of K + V across all kv_heads:
//
//   slot_bytes = 2 * num_kv_heads * group_stride
//
// The pool is laid out so that, for a given slot S, the K bytes for kv_head
// `h` start at `pool_base + S*slot_bytes + h*group_stride`, and V bytes start
// at `pool_base + S*slot_bytes + (num_kv_heads + h)*group_stride`. This
// matches the BHND layout the tiled-attention kernel expects when treating
// the scratch pool itself as the K/V "block pool" and using slot indices as
// block IDs.
//
// The pool maintains:
//   - A `Vec<Option<(u32 layer, u32 block)>>` of slot residents.
//   - A free-list of available slot indices.
//   - A `HashMap<(layer, block), slot_idx>` for lookup.
//
// Phase 2 keeps the API intentionally simple: `assign(layer, block)` returns
// a slot index for a fresh block, evicting the head of the free-list (or, if
// empty, the resident with the lowest predictor score — see eviction.rs).
// No epoch counters yet; threading and fence safety arrive in Phase 3 when
// the I/O thread lives on a separate stream.

use anyhow::{Result, bail};
use std::collections::{HashMap, VecDeque};

use crate::cuda_min::DeviceBuffer;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResidentKey {
    pub layer: u32,
    pub block: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ScratchDims {
    pub num_slots: u32,
    pub num_kv_heads: u16,
    pub group_stride: u64, // bytes per (block, kv_head) stripe
}

impl ScratchDims {
    pub fn slot_bytes(&self) -> usize {
        (2 * self.num_kv_heads as u64 * self.group_stride) as usize
    }
    pub fn pool_bytes(&self) -> usize {
        self.num_slots as usize * self.slot_bytes()
    }
}

pub struct ScratchPool {
    dims: ScratchDims,
    pool: DeviceBuffer,
    residents: Vec<Option<ResidentKey>>, // indexed by slot idx
    lookup: HashMap<ResidentKey, u32>,
    free_list: VecDeque<u32>,
}

impl ScratchPool {
    pub fn new(dims: ScratchDims) -> Result<Self> {
        if dims.num_slots == 0 {
            bail!("ScratchPool requires at least one slot");
        }
        let pool = DeviceBuffer::new(dims.pool_bytes())?;
        let residents = vec![None; dims.num_slots as usize];
        let free_list = (0..dims.num_slots).collect();
        Ok(Self {
            dims,
            pool,
            residents,
            lookup: HashMap::new(),
            free_list,
        })
    }

    pub fn dims(&self) -> ScratchDims {
        self.dims
    }
    pub fn pool_dev_ptr(&self) -> u64 {
        self.pool.ptr
    }
    pub fn slot_dev_ptr(&self, slot: u32) -> u64 {
        self.pool.ptr + (slot as u64) * (self.dims.slot_bytes() as u64)
    }
    /// K stripe device pointer for (slot, kv_head).
    pub fn slot_k_ptr(&self, slot: u32, kv_head: u16) -> u64 {
        self.slot_dev_ptr(slot) + (kv_head as u64) * self.dims.group_stride
    }
    /// V stripe device pointer for (slot, kv_head).
    pub fn slot_v_ptr(&self, slot: u32, kv_head: u16) -> u64 {
        self.slot_dev_ptr(slot)
            + (self.dims.num_kv_heads as u64 + kv_head as u64) * self.dims.group_stride
    }

    pub fn lookup(&self, key: ResidentKey) -> Option<u32> {
        self.lookup.get(&key).copied()
    }

    /// Drop the resident slot for `key` (if any). Returns the slot to the
    /// free list so the next `assign(key, _)` triggers a fresh disk read.
    ///
    /// Used by the offload path to discard the cached copy of a block after
    /// its on-disk image has been overwritten — without this, streaming
    /// attention would keep serving the stale resident copy and never see
    /// the freshly-offloaded K/V (e.g., decode steps re-writing the active
    /// block every step).
    pub fn invalidate(&mut self, key: ResidentKey) {
        if let Some(slot) = self.lookup.remove(&key) {
            self.residents[slot as usize] = None;
            self.free_list.push_back(slot);
        }
    }

    pub fn capacity(&self) -> u32 {
        self.dims.num_slots
    }
    pub fn free_count(&self) -> u32 {
        self.free_list.len() as u32
    }

    /// Reserve a slot for `key`. If the pool is full, picks an evictable slot
    /// from `evict_candidates` (callers pass them in score-ascending order;
    /// the lowest-scoring one is kicked first). Returns the slot index. The
    /// caller is responsible for issuing the disk read into `slot_dev_ptr`.
    pub fn assign(&mut self, key: ResidentKey, evict_candidates: &[u32]) -> Result<u32> {
        if let Some(&slot) = self.lookup.get(&key) {
            return Ok(slot); // already resident
        }
        let slot = match self.free_list.pop_front() {
            Some(s) => s,
            None => {
                // Find the first candidate that is currently resident (still
                // backed by a known key) and is not pinned.
                let mut chosen = None;
                for &c in evict_candidates {
                    if self
                        .residents
                        .get(c as usize)
                        .and_then(|r| r.as_ref())
                        .is_some()
                    {
                        chosen = Some(c);
                        break;
                    }
                }
                let s = chosen.ok_or_else(|| {
                    anyhow::anyhow!("no slot available and no eviction candidate is resident")
                })?;
                if let Some(prev) = self.residents[s as usize].take() {
                    self.lookup.remove(&prev);
                }
                s
            }
        };
        self.residents[slot as usize] = Some(key);
        self.lookup.insert(key, slot);
        Ok(slot)
    }

    /// Return all currently-resident slot indices in arbitrary order (for the
    /// eviction policy to score against the predictor).
    pub fn residents(&self) -> Vec<(u32, ResidentKey)> {
        self.residents
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|k| (i as u32, k)))
            .collect()
    }

    /// Free all slots; use between decode steps that don't share residency.
    pub fn clear(&mut self) {
        self.lookup.clear();
        for r in self.residents.iter_mut() {
            *r = None;
        }
        self.free_list.clear();
        for s in 0..self.dims.num_slots {
            self.free_list.push_back(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> crate::cuda_min::CudaCtx {
        crate::cuda_min::CudaCtx::new(0).expect("cuda init")
    }

    #[test]
    #[ignore = "requires GPU"]
    fn assign_and_lookup() {
        let _ctx = ctx();
        let mut pool = ScratchPool::new(ScratchDims {
            num_slots: 4,
            num_kv_heads: 2,
            group_stride: 4096,
        })
        .unwrap();
        let k0 = ResidentKey { layer: 0, block: 7 };
        let s0 = pool.assign(k0, &[]).unwrap();
        assert_eq!(pool.lookup(k0), Some(s0));
        // Repeated assign returns the same slot.
        let s0_again = pool.assign(k0, &[]).unwrap();
        assert_eq!(s0, s0_again);
        // Fill the pool.
        for b in 8..11 {
            pool.assign(ResidentKey { layer: 0, block: b }, &[])
                .unwrap();
        }
        assert_eq!(pool.free_count(), 0);
        // Evict the lowest-scoring slot (caller passes it in).
        let evicted = pool
            .assign(
                ResidentKey {
                    layer: 0,
                    block: 99,
                },
                &[s0],
            )
            .unwrap();
        assert_eq!(evicted, s0); // s0 was the eviction candidate
        assert_eq!(pool.lookup(k0), None); // k0 displaced
    }

    #[test]
    #[ignore = "requires GPU"]
    fn slot_pointer_layout() {
        let _ctx = ctx();
        let pool = ScratchPool::new(ScratchDims {
            num_slots: 2,
            num_kv_heads: 4,
            group_stride: 4096,
        })
        .unwrap();
        let base = pool.pool_dev_ptr();
        assert_eq!(pool.slot_dev_ptr(0), base);
        assert_eq!(pool.slot_dev_ptr(1), base + 8 * 4096);
        assert_eq!(pool.slot_k_ptr(0, 2), base + 2 * 4096);
        assert_eq!(pool.slot_v_ptr(0, 2), base + (4 + 2) * 4096);
    }
}
