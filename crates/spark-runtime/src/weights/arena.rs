// SPDX-License-Identifier: AGPL-3.0-only

//! Arena (pooled) ownership for [`WeightStore`] tensors, and the store's
//! teardown. Child module of `weights.rs` (≤500 LoC split).
//!
//! # Ownership invariant
//!
//! Every `WeightTensor.ptr` in the store is ONE of:
//!
//!  * an allocation BASE the store owns outright — the per-tensor
//!    `gpu.alloc(len)` every loader does by default; freed per entry; or
//!  * an INTERIOR pointer into a [`WeightArena`] the store also owns — a
//!    view produced by the fast loader's EXL3 pool (`fast_weights/pool.rs`),
//!    which uploads the `.trellis/.suh/.svh/.mul1` quartets of every
//!    prefix the materialize pass will keep packed into one allocation per
//!    (shard, class). The arena is freed ONCE, at `ModelResource::release`;
//!    its members are never passed to `gpu.free` (a `cuMemFree` on an
//!    interior pointer is `CUDA_ERROR_INVALID_VALUE`).
//!
//! Anyone who `remove`s or `insert`-displaces a store tensor therefore frees
//! it through [`WeightStore::release_tensor`] / [`WeightStore::release_ptr`]
//! (no-op for arena members, `gpu.free` otherwise) — never a raw `gpu.free`.
//! The two sites that do this for packed EXL3 tensors are the materialize
//! pass (`spark-model/weight_map/exl3_materialize.rs`) and `release` below;
//! the `.weight`-class frees in the NVFP4 / FP8 / fused-expert loaders route
//! through the same helper so they stay correct should those classes ever
//! be pooled.
//!
//! Why pool at all: measured on GB10 (driver 580, `cu_granularity.py`),
//! `cuMemAlloc_v2` sub-allocates requests under 2 MiB from 2 MiB chunks and
//! never splits an object across a chunk — an 800 KiB K=4 expert trellis
//! fits two per chunk and burns 224 KiB each; 73,728 of them cost ~17 GiB
//! of device memory the alloc ledger cannot see. One arena per (shard,
//! class) costs 1.006x its bytes.

use std::collections::BTreeMap;

use super::{WeightStore, WeightTensor};
use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::Result;

/// One pooled allocation the store owns; member tensors are `.offset()`
/// views into `[base, base + bytes)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightArena {
    pub base: DevicePtr,
    pub bytes: usize,
    /// Tensor class the arena holds (`"exl3-trellis"`, `"exl3-aux"`) — for
    /// the load log and `alloc_report` readers.
    pub label: &'static str,
}

impl WeightArena {
    /// True if `p` lies inside this arena.
    pub fn contains(&self, p: DevicePtr) -> bool {
        p.0 >= self.base.0 && p.0 < self.base.0 + self.bytes as u64
    }
}

/// O(log #arenas) membership: the arena whose base is the greatest `<= p`,
/// if `p` falls inside it. Free fn so `release` can query while `weights`
/// is being drained (disjoint field borrows).
fn arena_containing(arenas: &BTreeMap<u64, WeightArena>, p: DevicePtr) -> Option<&WeightArena> {
    arenas
        .range(..=p.0)
        .next_back()
        .map(|(_, a)| a)
        .filter(|a| a.contains(p))
}

impl WeightStore {
    /// Take ownership of an arena. Its member tensors are inserted by the
    /// caller as ordinary entries (`.offset()` views of `base`).
    ///
    /// Panics on a null base or an arena overlapping one already adopted —
    /// both are loader bugs that would otherwise surface as a teardown
    /// double-free.
    pub fn adopt_arena(&mut self, arena: WeightArena) {
        assert!(!arena.base.is_null(), "adopt_arena: null base");
        assert!(arena.bytes > 0, "adopt_arena: empty arena");
        let end = arena.base.0 + arena.bytes as u64;
        let overlaps = self
            .arenas
            .range(..end)
            .next_back()
            .is_some_and(|(_, a)| a.base.0 + a.bytes as u64 > arena.base.0);
        assert!(
            !overlaps,
            "adopt_arena: {arena:?} overlaps an adopted arena"
        );
        self.arenas.insert(arena.base.0, arena);
    }

    /// The arena `p` lives in, if any.
    pub fn arena_of(&self, p: DevicePtr) -> Option<&WeightArena> {
        arena_containing(&self.arenas, p)
    }

    /// True if `p` is an interior pointer of an adopted arena.
    pub fn is_pooled(&self, p: DevicePtr) -> bool {
        self.arena_of(p).is_some()
    }

    /// Number of adopted arenas.
    pub fn arena_count(&self) -> usize {
        self.arenas.len()
    }

    /// Bytes held by adopted arenas (what the alloc ledger sees for them).
    pub fn pooled_bytes(&self) -> usize {
        self.arenas.values().map(|a| a.bytes).sum()
    }

    /// Free a store-derived device pointer the caller has removed or
    /// displaced: `gpu.free` for a per-tensor allocation, a no-op for an
    /// arena member (its bytes stay resident until the arena is released).
    /// Returns whether memory was actually freed.
    pub fn release_ptr(&self, gpu: &dyn GpuBackend, ptr: DevicePtr) -> Result<bool> {
        if ptr.is_null() || self.is_pooled(ptr) {
            return Ok(false);
        }
        gpu.free(ptr)?;
        Ok(true)
    }

    /// [`Self::release_ptr`] for a tensor the caller took out of the store.
    pub fn release_tensor(&self, gpu: &dyn GpuBackend, t: WeightTensor) -> Result<bool> {
        self.release_ptr(gpu, t.ptr)
    }

    /// Remove `name` and free it through [`Self::release_tensor`].
    /// `Ok(None)` if absent; `Ok(Some(freed))` otherwise, `freed == false`
    /// meaning the tensor was an arena member and its bytes are stranded
    /// until release.
    pub fn remove_and_free(&mut self, gpu: &dyn GpuBackend, name: &str) -> Result<Option<bool>> {
        match self.remove(name) {
            Some(t) => Ok(Some(self.release_tensor(gpu, t)?)),
            None => Ok(None),
        }
    }
}

/// Release every weight tensor, then every arena — each exactly once.
///
/// Per-entry frees are safe for the non-pooled entries because every loader
/// allocates those per tensor (`gpu.alloc(meta.len)` before `insert`);
/// pooled entries are skipped and their arena base is freed afterwards.
/// (Fused per-expert views DO exist outside the store — see
/// `weight_loader/step3p7.rs:93` — but they live in the layer structs that
/// own the fused allocation, not here, so this cannot double-free them.)
impl atlas_core::scope::ModelResource<dyn GpuBackend> for WeightStore {
    fn label(&self) -> &'static str {
        "weight store"
    }

    fn release(&mut self, gpu: &dyn GpuBackend) -> anyhow::Result<()> {
        let mut first_error = None;
        // `drain` rather than iterate: the map must not be left holding
        // pointers to memory that is gone, and it makes this idempotent.
        for (name, tensor) in self.weights.drain() {
            if arena_containing(&self.arenas, tensor.ptr).is_some() {
                continue;
            }
            if let Err(e) = gpu.free(tensor.ptr)
                && first_error.is_none()
            {
                first_error = Some(e.context(format!("freeing weight {name}")));
            }
        }
        // Arenas last: a member view must never outlive its base, and the
        // drain above already guarantees the map holds none.
        for (_, arena) in std::mem::take(&mut self.arenas) {
            if let Err(e) = gpu.free(arena.base)
                && first_error.is_none()
            {
                first_error = Some(e.context(format!(
                    "freeing {} weight arena ({} bytes)",
                    arena.label, arena.bytes
                )));
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
