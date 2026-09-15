// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 weight pooling for the fast loader: ONE device allocation per
//! (shard, tensor class) instead of one per tensor.
//!
//! # Why
//!
//! Measured on GB10 (driver 580, `cu_granularity.py`): `cuMemAlloc_v2`
//! sub-allocates requests below 2 MiB from 2 MiB chunks with 512-B rounding
//! and never splits an object across a chunk, so a size `s` costs
//! `2 MiB / floor(2 MiB / roundup(s, 512))`. Every EXL3 expert trellis of the
//! Qwen3.8-Flash-Next family lands in the bad tier: 800 KiB (K=4) fits two
//! per chunk and burns 224 KiB each (1.32x), 600 KiB (K=3) 1.17x, 400 KiB
//! (K=2) 1.05x. On the 4.05bpw checkpoint that is ~17.9 GiB of device memory
//! the alloc ledger never sees (296K allocations × chunk tails), which the
//! KV budget then reports as "co-tenant/page-cache use excluded". A single
//! GiB-scale allocation costs 1.006x.
//!
//! # What is pooled
//!
//! Only the `.trellis / .suh / .svh / .mul1` quartets of prefixes the
//! caller's predicate says the EXL3 materialize pass will KEEP packed
//! (native serving). Non-kept EXL3 tensors are materialized and FREED by
//! that pass; inside an arena they could never be reclaimed (no partial
//! `cuMemFree`), and with the native gates off that is the whole packed
//! checkpoint held next to its NVFP4 rewrite. Tiny non-EXL3 tensors are not
//! pooled either: the driver already packs them at 512 B (63 MiB model-wide
//! saving, measured), and several loaders free `.weight`-class store
//! pointers in place. Everything outside the pooled set takes the
//! byte-identical per-tensor path.
//!
//! # Layout
//!
//! Per shard, two arenas — `exl3-trellis` (the blobs) and `exl3-aux`
//! (suh/svh/mul1) — bump-allocated in file order with 256-B slot alignment
//! (`cuMemAlloc` bases are ≥256-B aligned; the EXL3 kernels take 16-B vector
//! loads on trellis tiles and f16 vectors on suh/svh). The class split is
//! for `alloc_report` legibility and locality; both are plain
//! `WeightArena`s owned by the store. Slot padding is ≤255 B per tensor
//! (~36 MiB on 4.05bpw).

use std::collections::HashSet;

use super::header::TensorMeta;
use crate::gpu::{DevicePtr, GpuBackend};
use crate::weights::WeightArena;

/// Slot alignment inside an arena.
pub(super) const SLOT_ALIGN: usize = 256;

/// The driver's sub-allocation chunk: requests below this pay a chunk-tail
/// tax; requests at or above it get a dedicated allocation (~1.02x).
const DRIVER_CHUNK: usize = 2 * 1024 * 1024;
/// Sub-allocation rounding inside a chunk.
const DRIVER_ROUND: usize = 512;

/// The EXL3 tensor suffixes that form one packed linear.
const EXL3_SUFFIXES: [&str; 4] = [".trellis", ".suh", ".svh", ".mul1"];

/// `Some(prefix)` if `name` is one of the four packed-EXL3 tensors.
pub(super) fn exl3_prefix(name: &str) -> Option<&str> {
    EXL3_SUFFIXES.iter().find_map(|s| name.strip_suffix(s))
}

/// Which arena a pooled tensor goes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PoolClass {
    Trellis,
    Aux,
}

impl PoolClass {
    const ALL: [Self; 2] = [Self::Trellis, Self::Aux];

    fn label(self) -> &'static str {
        match self {
            Self::Trellis => "exl3-trellis",
            Self::Aux => "exl3-aux",
        }
    }

    fn of(name: &str) -> Self {
        if name.ends_with(".trellis") {
            Self::Trellis
        } else {
            Self::Aux
        }
    }
}

/// The per-allocation footprint `cuMemAlloc_v2` was measured to charge on
/// GB10 for a request of `len` bytes (see the module docs). Used only for
/// the "padding avoided" figure in the load log.
pub(super) fn measured_alloc_footprint(len: usize) -> usize {
    if len >= DRIVER_CHUNK {
        return len;
    }
    let rounded = len.max(1).div_ceil(DRIVER_ROUND) * DRIVER_ROUND;
    let per_chunk = (DRIVER_CHUNK / rounded).max(1);
    DRIVER_CHUNK / per_chunk
}

/// One class's bump layout for a shard.
struct ClassPlan {
    class: PoolClass,
    /// `(tensor index, byte offset)` in file order.
    slots: Vec<(usize, usize)>,
    bytes: usize,
}

/// A shard's pooled layout: which tensors go to which arena at which offset.
pub(super) struct PoolPlan {
    classes: Vec<ClassPlan>,
    pub(super) pooled_tensors: usize,
    pub(super) pooled_bytes: usize,
    /// Σ (measured per-tensor footprint − len) over the pooled tensors.
    pub(super) padding_avoided: usize,
}

impl PoolPlan {
    /// Lay out every tensor whose EXL3 prefix is in `pooled_prefixes`.
    /// `tensors` is the shard's upload set (already EP/mtp/index filtered,
    /// file order), so indices refer to it directly.
    pub(super) fn build(tensors: &[TensorMeta], pooled_prefixes: &HashSet<String>) -> Self {
        let mut classes: Vec<ClassPlan> = PoolClass::ALL
            .iter()
            .map(|&class| ClassPlan {
                class,
                slots: Vec::new(),
                bytes: 0,
            })
            .collect();
        let (mut pooled_tensors, mut pooled_bytes, mut padding_avoided) = (0, 0, 0);
        if !pooled_prefixes.is_empty() {
            for (idx, t) in tensors.iter().enumerate() {
                let Some(prefix) = exl3_prefix(&t.name) else {
                    continue;
                };
                if !pooled_prefixes.contains(prefix) {
                    continue;
                }
                let class = PoolClass::of(&t.name);
                let plan = &mut classes[class as usize];
                plan.slots.push((idx, plan.bytes));
                plan.bytes += t.len.div_ceil(SLOT_ALIGN) * SLOT_ALIGN;
                pooled_tensors += 1;
                pooled_bytes += t.len;
                padding_avoided += measured_alloc_footprint(t.len) - t.len;
            }
        }
        classes.retain(|c| c.bytes > 0);
        Self {
            classes,
            pooled_tensors,
            pooled_bytes,
            padding_avoided,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    pub(super) fn arena_count(&self) -> usize {
        self.classes.len()
    }
}

/// A shard's allocated arenas plus the slot pointer of every pooled tensor.
pub(super) struct ShardArenas {
    arenas: Vec<WeightArena>,
    /// Indexed by tensor index; `None` for tensors on the per-tensor path.
    slots: Vec<Option<DevicePtr>>,
}

impl ShardArenas {
    /// Allocate ONE device buffer per class in `plan`. `Ok(None)` when the
    /// plan is empty or an arena allocation failed — the caller then loads
    /// the whole shard per tensor (every arena allocated so far is freed
    /// first, so a failure leaves nothing behind).
    pub(super) fn alloc(
        gpu: &dyn GpuBackend,
        plan: &PoolPlan,
        tensor_count: usize,
        fallback_logged: &mut bool,
    ) -> anyhow::Result<Option<Self>> {
        if plan.is_empty() {
            return Ok(None);
        }
        let mut arenas = Vec::with_capacity(plan.classes.len());
        let mut slots = vec![None; tensor_count];
        for c in &plan.classes {
            // The ledger records this call site: `alloc_report` shows the
            // pool as `fast_weights/pool.rs:<line>` with one record per arena.
            let base = match gpu.alloc(c.bytes) {
                Ok(p) => p,
                Err(e) => {
                    if !*fallback_logged {
                        tracing::warn!(
                            "EXL3 weight pool: {} arena alloc of {:.1} MB failed ({e}) — loading \
                             this shard per tensor (allocator chunk padding not avoided)",
                            c.class.label(),
                            c.bytes as f64 / (1024.0 * 1024.0),
                        );
                        *fallback_logged = true;
                    }
                    let partial = Self { arenas, slots };
                    partial.free_all(gpu);
                    return Ok(None);
                }
            };
            for &(idx, off) in &c.slots {
                slots[idx] = Some(base.offset(off));
            }
            arenas.push(WeightArena {
                base,
                bytes: c.bytes,
                label: c.class.label(),
            });
        }
        Ok(Some(Self { arenas, slots }))
    }

    /// The pooled slot for tensor `idx`, if it has one.
    pub(super) fn slot(&self, idx: usize) -> Option<DevicePtr> {
        self.slots.get(idx).copied().flatten()
    }

    /// Hand the arenas to the caller (for `WeightStore::adopt_arena`).
    pub(super) fn into_arenas(self) -> Vec<WeightArena> {
        self.arenas
    }

    /// Rollback: free every arena once. Used when the shard's copy loop
    /// fails after the arenas were allocated — the member views inserted so
    /// far die with the abandoned map, and the bases must not wait for the
    /// backend's `sweep_unreleased`.
    pub(super) fn free_all(self, gpu: &dyn GpuBackend) {
        for a in self.arenas {
            if let Err(e) = gpu.free(a.base) {
                tracing::warn!(
                    "EXL3 weight pool: freeing {} arena on rollback: {e}",
                    a.label
                );
            }
        }
    }
}

#[cfg(test)]
#[path = "pool_tests.rs"]
mod tests;
