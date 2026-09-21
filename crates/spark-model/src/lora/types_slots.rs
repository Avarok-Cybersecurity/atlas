// SPDX-License-Identifier: AGPL-3.0-only

//! `LoraWeights` SLOT LIFECYCLE: acquiring, releasing, ref-counting and the
//! LRU bookkeeping the victim search reads.
//!
//! Split from `types.rs` for the 500-LoC cap, which the file crossed when the
//! GDN `out_proj` module joined `LoraModule`. Exact piecewise copy — no method
//! changed in the move. The pool LAYOUT stays in `types.rs`; this is the half
//! that mutates which adapter is resident.

use std::sync::atomic::Ordering;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::types::{LoraLayerWeights, LoraModule, LoraWeights, SlotView};

impl LoraWeights {
    /// Task #25: resolve `slot` (`>= 0` → that slot, `-1` → active) to a concrete
    /// pool index and `+1` its ref_count, returning the RESOLVED index so the
    /// caller can release EXACTLY that index later (immune to an intervening
    /// rotate changing `active`). Returns `-1` — "nothing acquired" — when the
    /// resolved index is out of range (bad request slot); the active slot is
    /// always in range so `-1 -> active` never no-ops here for a loaded pool.
    pub fn acquire_slot(&self, slot: i32) -> i32 {
        let resolved = if slot >= 0 {
            slot as usize
        } else {
            self.active
        };
        match self.ref_counts.get(resolved) {
            Some(rc) => {
                rc.fetch_add(1, Ordering::AcqRel);
                // Task #27: stamp the RESOLVED slot as most-recently-used so the
                // LRU victim policy ages the slot a request actually touched
                // (including `-1 -> active`). Ticks are strictly increasing.
                if let Some(lu) = self.last_used.get(resolved) {
                    let t = self.lru_tick.fetch_add(1, Ordering::Relaxed) + 1;
                    lu.store(t, Ordering::Relaxed);
                }
                resolved as i32
            }
            None => -1,
        }
    }

    /// Task #27: stamp `slot` as most-recently-used WITHOUT taking a ref. Called
    /// right after a promote so a freshly-staged (ref_count==0) slot is NOT the
    /// immediate LRU victim of a back-to-back promote before its own request has
    /// acquired — otherwise two distinct cold adapters promoted in quick
    /// succession would collide on the same slot (the second evicting the first).
    pub fn touch_slot(&self, slot: usize) {
        if let Some(lu) = self.last_used.get(slot) {
            let t = self.lru_tick.fetch_add(1, Ordering::Relaxed) + 1;
            lu.store(t, Ordering::Relaxed);
        }
    }

    /// Task #27: current LRU tick of pool `slot` (larger = more recently
    /// acquired). Out-of-range → 0 (never used).
    pub fn slot_last_used(&self, slot: usize) -> u64 {
        self.last_used
            .get(slot)
            .map(|lu| lu.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Task #26: refresh `slot`'s cell in the `[max_loras]` a/b pointer tables +
    /// the per-slot scale table from `layers` (the just-staged adapter's actual
    /// per-module coverage). A re-staged adapter whose module coverage DIFFERS
    /// from the evicted one would otherwise keep a STALE table entry: the
    /// bgmv-routed path would SKIP a module the new adapter adds (`a_table[slot]`
    /// stale-NULL → missed delta), keep applying an evicted module (stale non-NULL
    /// → wrong delta), or use the wrong per-slot scale. Shared by BOTH the disk
    /// swap (`pack_store_into_slot`) and the RDMA swap (`swap_lora_slot_from_peer`).
    /// Only the `[slot]` cell of each fixed-address device array is rewritten.
    pub fn refresh_slot_tables(
        &self,
        slot: usize,
        layers: &[Option<LoraLayerWeights>],
        scale: f32,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        for ((layer, module), (a_dev, b_dev)) in &self.tables {
            let pair = layers
                .get(*layer)
                .and_then(|o| o.as_ref())
                .and_then(|lw| match module {
                    LoraModule::QProj => lw.q_proj.as_ref(),
                    LoraModule::KProj => lw.k_proj.as_ref(),
                    LoraModule::VProj => lw.v_proj.as_ref(),
                    LoraModule::OProj => lw.o_proj.as_ref(),
                    LoraModule::GateProj => lw.gate_proj.as_ref(),
                    LoraModule::UpProj => lw.up_proj.as_ref(),
                    LoraModule::DownProj => lw.down_proj.as_ref(),
                    LoraModule::OutProj => lw.out_proj.as_ref(),
                });
            let (a_ptr, b_ptr) = pair.map(|p| (p.a.weight.0, p.b.weight.0)).unwrap_or((0, 0));
            gpu.copy_h2d(&a_ptr.to_le_bytes(), DevicePtr(a_dev.0 + (slot * 8) as u64))?;
            gpu.copy_h2d(&b_ptr.to_le_bytes(), DevicePtr(b_dev.0 + (slot * 8) as u64))?;
        }
        if self.scale_table.0 != 0 {
            gpu.copy_h2d(
                &scale.to_le_bytes(),
                DevicePtr(self.scale_table.0 + (slot * 4) as u64),
            )?;
        }
        Ok(())
    }

    /// Task #27: snapshot the CACHE region `[pinned, max_loras)` as
    /// `(slot_index, SlotView)` for `select_victim_slot`. `filled` = the slot
    /// holds a non-placeholder adapter (non-empty name). Read on the model
    /// thread at a quiescent point.
    pub fn cache_slot_views(&self) -> Vec<(usize, SlotView)> {
        (self.pinned..self.max_loras)
            .map(|k| {
                let filled = self.slots.get(k).is_some_and(|s| !s.name.is_empty());
                (
                    k,
                    SlotView {
                        filled,
                        ref_count: self.slot_ref_count(k),
                        last_used: self.slot_last_used(k),
                    },
                )
            })
            .collect()
    }

    /// Task #25: release a ref previously taken by [`Self::acquire_slot`], by the
    /// RESOLVED index it returned. `-1` (nothing acquired) is a no-op. Saturating
    /// so a stray double-release can never wrap the counter below 0.
    pub fn release_slot(&self, resolved: i32) {
        if resolved < 0 {
            return;
        }
        if let Some(rc) = self.ref_counts.get(resolved as usize) {
            let _ = rc.fetch_update(Ordering::Release, Ordering::Acquire, |v| {
                Some(v.saturating_sub(1))
            });
        }
    }

    /// Task #25: current in-flight ref_count of pool `slot` (the exact read the
    /// swap busy-slot gate branches on). Out-of-range → 0.
    pub fn slot_ref_count(&self, slot: usize) -> usize {
        self.ref_counts
            .get(slot)
            .map(|rc| rc.load(Ordering::Acquire))
            .unwrap_or(0)
    }
}
