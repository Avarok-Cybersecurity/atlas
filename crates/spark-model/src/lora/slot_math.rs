// SPDX-License-Identifier: AGPL-3.0-only

//! LoRA slot/offset math: the pure, GPU-free functions that place adapters in
//! the fixed-address rank-padded pool and route requests to slots — victim
//! selection, per-step `seq_slot` build, scale-table values, routed-prefill
//! predicate + pair selection, and the frozen per-slot byte-offset layout.
//! Split out of the former monolithic `lora/mod.rs` (SDD seam: SLOT/OFFSET
//! MATH) — visibility unchanged.

use atlas_core::config::ModelConfig;

use super::*;
use crate::layers::ops::lora_delta::LoraPair;

pub(crate) const BF16_BYTES: usize = 2;

/// Task #27 pure victim-selection policy over the CACHE region only (the caller
/// passes `(slot_index, view)` for slots `[pinned, max_loras)` — pinned startup
/// adapters are never candidates, so the resident set and its position-based
/// resolver can never desync). Tiers:
///   1. FREE-FIRST: the first `!filled` (never-promoted) placeholder slot.
///   2. LRU-IDLE: else the `ref_count == 0` slot with the smallest `last_used`.
///   3. POOL-FULL: else every cache slot is busy → `Err(PoolFull)` (retryable);
///      a `ref_count > 0` slot is NEVER returned.
pub fn select_victim_slot(cache: &[(usize, SlotView)]) -> Result<usize, VictimError> {
    // Tier 1: a never-filled placeholder is the cheapest victim (no eviction).
    if let Some((idx, _)) = cache.iter().find(|(_, v)| !v.filled) {
        return Ok(*idx);
    }
    // Tier 2: LRU among the idle (ref_count == 0) filled slots.
    cache
        .iter()
        .filter(|(_, v)| v.ref_count == 0)
        .min_by_key(|(_, v)| v.last_used)
        .map(|(idx, _)| *idx)
        // Tier 3: all cache slots busy — retryable, never evict a busy slot.
        .ok_or(VictimError::PoolFull)
}

/// Build the per-step `seq_slot[padded_n]` host buffer the batched bgmv reads,
/// from each real sequence's `adapter_slot`. Resolution rules (graph-safe:
/// contents vary per step, buffer address is fixed):
///   real row i (< n): `adapter_slots[i]` if `>= 0`, else `active` — a request
///     with no `adapter` field carries `-1` and DEFERS to the installed active
///     adapter, so a single global adapter (or a rotate re-point) applies to
///     every default row exactly like the n==1 path.
///   pad row i (n..padded_n): `-1` — base / no delta (bgmv early-returns).
/// A row that explicitly names the base model (some future `-1`-means-base
/// convention) is out of scope here; `-1` uniformly means "defer to active".
pub fn build_seq_slot_host(adapter_slots: &[i32], padded_n: usize, active: i32) -> Vec<i32> {
    let n = adapter_slots.len();
    (0..padded_n)
        .map(|i| {
            if i < n {
                let s = adapter_slots[i];
                if s >= 0 { s } else { active }
            } else {
                -1
            }
        })
        .collect()
}

/// Pure per-slot scale vector for the `[max_loras]` f32 scale table: entry `k`
/// = adapter `k`'s `scaling()` (alpha/r, or alpha/√r under rsLoRA — read per
/// adapter, never defaulted), 0.0 for unpacked slots `k >= adapters.len()`.
/// Split out for unit testing (the device upload is a thin wrapper).
pub(crate) fn scale_table_values(adapters: &[LoraAdapterInput<'_>], max_loras: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; max_loras];
    for (k, a) in adapters.iter().enumerate() {
        v[k] = a.peft.scaling();
    }
    v
}

/// #30 (routed-prefill precision): the pure predicate behind
/// [`LoraWeights::routed_prefill_slot`], split out for unit testing. Resolves a
/// request's `adapter_slot` (`>= 0` → that slot, `-1` → active) and returns
/// `Some(resolved)` ONLY when it routes to a NON-active, in-range slot. Returns
/// `None` for an active/base request (byte-identical installed-pair path) and for
/// out-of-range slots. Kept in exact lockstep with `upload_seq_slot_uniform`
/// (`resolved == active` → `DevicePtr(0)`).
pub fn routed_prefill_slot_of(adapter_slot: i32, active: usize, num_slots: usize) -> Option<usize> {
    let resolved = if adapter_slot >= 0 {
        adapter_slot as usize
    } else {
        active
    };
    (resolved != active && resolved < num_slots).then_some(resolved)
}

/// #30 (routed-prefill precision): pure selector for a routed prefill's
/// (global_layer, module) [`LoraPair`] out of a request slot's GLOBAL-layer-indexed
/// `layers`. `None` when the index is out of range, the layer is unadapted, or the
/// routed adapter does not adapt that module (the caller then falls back to the
/// bgmv/installed path — no delta if the slot's a_table cell is base). GPU-free +
/// unit-tested so the (layer, module) indexing is verifiable without hardware.
pub fn select_routed_pair(
    layers: &[Option<LoraLayerWeights>],
    global_layer_idx: usize,
    module: LoraModule,
) -> Option<&LoraPair> {
    layers
        .get(global_layer_idx)
        .and_then(|o| o.as_ref())
        .and_then(|l| l.module_pair(module))
}

/// Padded per-slot bytes: Σ over (full-attn layers × 6 modules) of
/// (max_rank·in + out·max_rank)·2. Holo @ max_rank=64: ≈ 2.44 MiB/layer
/// × 6 = ~14.6 MiB/slot; × max_loras=8 ≈ 117 MiB total.
pub(crate) fn pool_slot_bytes(cfg: &ModelConfig, max_rank: usize) -> usize {
    full_attention_layers(cfg)
        .iter()
        .map(|_| {
            LoraModule::ALL
                .iter()
                .map(|m| {
                    let (out, inp) = m.dims(cfg);
                    (max_rank * inp + out * max_rank) * BF16_BYTES
                })
                .sum::<usize>()
        })
        .sum()
}

/// Byte offset of slot `k`'s base within the pool. Slots are equal fixed size,
/// so slot `k` starts at `k * pool_slot_bytes`. Slot 0 → 0 (byte-identical to
/// the single-adapter path).
// Only the RDMA landing path (`rdma_stage`) and the unit tests call this. That
// path is cuda AND unix (it lands through spark-storage's RDMA weight loader),
// so the dead-code allowance must match: `not(all(cuda, unix))`, not
// `not(cuda)`. It stays defined everywhere for the offset unit tests.
#[cfg_attr(not(all(feature = "cuda", unix)), allow(dead_code))]
pub(crate) fn slot_base_offset(slot: usize, cfg: &ModelConfig, max_rank: usize) -> usize {
    slot * pool_slot_bytes(cfg, max_rank)
}

/// The (a_off, b_off) of a given (layer, module) WITHIN a slot — the exact
/// running offsets the pack loop computes (layer asc × [`LoraModule::ALL`] ×
/// A-then-B). `None` if `target_layer` is not a full-attention layer. Used by
/// the pack loop, the RDMA landing path, and the offset unit tests so all three
/// agree on the one frozen layout.
#[cfg_attr(not(all(feature = "cuda", unix)), allow(dead_code))]
pub(crate) fn module_slot_offsets(
    cfg: &ModelConfig,
    max_rank: usize,
    target_layer: usize,
    target_module: LoraModule,
) -> Option<(usize, usize)> {
    let mut off = 0usize;
    for layer_idx in full_attention_layers(cfg) {
        for module in LoraModule::ALL {
            let (out_dim, in_dim) = module.dims(cfg);
            let a_off = off;
            let b_off = off + max_rank * in_dim * BF16_BYTES;
            off = b_off + out_dim * max_rank * BF16_BYTES;
            if layer_idx == target_layer && module == target_module {
                return Some((a_off, b_off));
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "slot_math_tests.rs"]
mod tests;
