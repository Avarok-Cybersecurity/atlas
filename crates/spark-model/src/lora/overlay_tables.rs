// SPDX-License-Identifier: AGPL-3.0-only

//! Token-overlay device pointer tables (Feature 2). The per-adapter-slot
//! overlays are addressed by `[max_loras]` device tables of pointers/counts at
//! LOAD-TIME-FIXED addresses, exactly like the attention BGMV `a_table`/
//! `b_table` (`ops::lora_delta::LoraRoute`): the only per-step kernel argument
//! is `seq_slot` (device i32, contents re-uploaded each step), so the tables are
//! stable kernel args across CUDA-graph capture/replay and the overlay launches
//! capture cleanly. Cell `k == 0` (null pointer) or `n_override[k] == 0` ⇒ that
//! slot's overlay is skipped by the kernel — zero-overhead for base requests.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::overlay_build::EmbedOverlay;

/// The resolved overlay set for the whole adapter pool. Built once in
/// `set_lora_weights` (Stage 2) from the per-slot [`EmbedOverlay`]s. `None` on
/// the model ⇒ overlay feature OFF ⇒ every forward hook early-returns
/// (byte-identical to a no-overlay build).
pub struct TokenOverlaySet {
    /// Per-slot compact overlays (`len == max_loras`), retained so the device
    /// buffers they own outlive the tables that point into them.
    pub overlays: Vec<Option<EmbedOverlay>>,
    /// `u64[max_loras]` → `i32*[vocab]` embed slot_map (0 = slot has no overlay).
    pub embed_slot_map_table: DevicePtr,
    /// `u64[max_loras]` → `bf16*[n,h]` embed override rows.
    pub embed_rows_table: DevicePtr,
    /// `u32[max_loras]` EMBED n_override per slot — the row count of each
    /// slot's `embed_rows_table` cell, the embed kernel's `slot < n` bound
    /// (CWE-125 guard). Distinct from `n_override_table`, which is the
    /// LM_HEAD count (0 for an untied slot with no lm_head overlay).
    pub embed_n_table: DevicePtr,
    /// `slot_map` length shared by every resident overlay (each slot records
    /// the served vocab it was built against; `from_slots` REFUSES a mix) —
    /// the embed kernel's `ids[r] < vocab` bound (CWE-125 guard).
    pub vocab: u32,
    /// `u64[max_loras]` → `bf16*[n,h]` lm_head override rows (== embed cell when tied).
    pub lmhead_rows_table: DevicePtr,
    /// `u64[max_loras]` → `u32*[n]` lm_head override ids (== embed cell when tied).
    pub lmhead_ids_table: DevicePtr,
    /// `u32[max_loras]` lm_head n_override per slot (0 ⇒ lm_head skip).
    pub n_override_table: DevicePtr,
    /// `max(lmhead n_override)` across slots — the lm_head kernel's `grid.y`.
    pub max_n_override: u32,
}

/// Pack a `[max_loras]` u64 pointer array to device (le bytes → alloc → h2d),
/// mirroring the `mk` closure in `loading.rs`.
fn mk_u64(gpu: &dyn GpuBackend, tab: &[u64]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = tab.iter().flat_map(|p| p.to_le_bytes()).collect();
    let d = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, d)?;
    Ok(d)
}

fn mk_u32(gpu: &dyn GpuBackend, tab: &[u32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = tab.iter().flat_map(|p| p.to_le_bytes()).collect();
    let d = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, d)?;
    Ok(d)
}

impl TokenOverlaySet {
    /// Build the device tables from per-slot overlays. `tied` ⇒ the lm_head
    /// reuses each slot's embed override rows/ids (tied vocab head, or a
    /// quantized head derived from embed): the logit recompute `dot(hidden,
    /// embed_row)` is exactly the tied output projection. An untied slot that
    /// ships its own lm_head overlay uses its distinct rows/ids; an untied slot
    /// that does NOT ⇒ `n_override[k] = 0` (embed-only correction).
    pub fn from_slots(
        gpu: &dyn GpuBackend,
        overlays: Vec<Option<EmbedOverlay>>,
        max_loras: usize,
        tied: bool,
    ) -> Result<Self> {
        let mut slot_map_tab = vec![0u64; max_loras];
        let mut embed_rows_tab = vec![0u64; max_loras];
        let mut embed_n_tab = vec![0u32; max_loras];
        let mut lmhead_rows_tab = vec![0u64; max_loras];
        let mut lmhead_ids_tab = vec![0u64; max_loras];
        let mut n_override_tab = vec![0u32; max_loras];
        let mut max_n_override = 0u32;
        let mut vocab = 0u32;

        for (k, ov) in overlays.iter().enumerate() {
            let Some(ov) = ov else { continue };
            slot_map_tab[k] = ov.slot_map.0;
            embed_rows_tab[k] = ov.rows.0;
            embed_n_tab[k] = ov.n_override;
            // One build pass sizes every slot_map to the same served vocab; a
            // mix would make the kernel's single `ids[r] < vocab` bound wrong
            // for some slot, so REFUSE it rather than guess a bound.
            anyhow::ensure!(
                vocab == 0 || vocab == ov.vocab,
                "token-overlay: slot {k} was built against vocab {} but an \
                 earlier slot against {vocab}; refusing mixed-vocab overlay tables",
                ov.vocab
            );
            vocab = ov.vocab;
            let (rows, ids, n) = match (&ov.lmhead, tied) {
                (Some(lm), _) => (lm.rows.0, lm.ids_dev.0, lm.n_override),
                (None, true) => (ov.rows.0, ov.ids_dev.0, ov.n_override),
                (None, false) => (0, 0, 0),
            };
            lmhead_rows_tab[k] = rows;
            lmhead_ids_tab[k] = ids;
            n_override_tab[k] = n;
            max_n_override = max_n_override.max(n);
        }

        Ok(Self {
            embed_slot_map_table: mk_u64(gpu, &slot_map_tab)?,
            embed_rows_table: mk_u64(gpu, &embed_rows_tab)?,
            embed_n_table: mk_u32(gpu, &embed_n_tab)?,
            vocab,
            lmhead_rows_table: mk_u64(gpu, &lmhead_rows_tab)?,
            lmhead_ids_table: mk_u64(gpu, &lmhead_ids_tab)?,
            n_override_table: mk_u32(gpu, &n_override_tab)?,
            max_n_override,
            overlays,
        })
    }

    /// True when at least one slot has a resident overlay (else the set is inert
    /// and the caller can drop it to keep the hooks byte-identical to off).
    pub fn any_active(&self) -> bool {
        self.overlays.iter().any(|o| o.is_some())
    }
}

#[cfg(test)]
#[path = "overlay_tables_tests.rs"]
mod tests;
