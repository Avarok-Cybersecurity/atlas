// SPDX-License-Identifier: AGPL-3.0-only

//! Token-overlay GPU build (Feature 2, Stage 1 + Stage 2). Splits the overlay
//! materialization across the model-build ordering gap: the RAW adapter tensors
//! live only while the adapter [`WeightStore`] is alive (loader time), but the
//! served embed/lm_head tables the row-diff needs only exist after weight load.
//! So:
//!
//! - **Stage 1** ([`stage_overlay_raw`], loader): copy the raw overlay tensors
//!   into owned device scratch while the store is alive; stash an [`OverlayRaw`].
//! - **Stage 2** ([`build_overlay`], `set_lora_weights`): served tables now
//!   exist → row-diff kernel → [`build_override_set`] → compact override rows +
//!   `slot_map` + `ids` → [`EmbedOverlay`]; free the raw scratch.
//!
//! The pure row-selection math ([`clamp_trainable_to_vocab`],
//! [`build_override_set`], [`override_source`], [`ROWDIFF_THRESH`]) lives in
//! `super::overlay`; the device pointer tables in `super::overlay_tables`; the
//! forward hooks in `crate::model::token_overlay`.

use anyhow::{Result, bail};
use atlas_core::config::PeftAdapterConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use super::overlay::{
    OverlayTensors, ROWDIFF_THRESH, build_override_set, clamp_trainable_to_vocab, override_source,
};
use crate::layers::ops::token_overlay::{self, OverlayKernels};

const BF16_BYTES: usize = 2;
const F32_BYTES: usize = 4;

/// Stage-1 raw upload: the adapter's own overlay tensors copied into owned
/// device scratch (the source `WeightStore` is freed after the loader). Dims
/// are recorded so Stage 2 can shape the row-diff and compaction without the
/// store. `embed_base`/`lmhead_base` also carry a `modules_to_save` full-weight
/// replacement (row-diff finds the changed rows either way).
#[derive(Debug, Clone, Copy, Default)]
pub struct OverlayRaw {
    pub embed_base: Option<DevicePtr>,  // [embed_r, h] bf16
    pub embed_delta: Option<DevicePtr>, // [embed_t, h] f32
    pub embed_r: u32,
    pub embed_t: u32,
    pub lmhead_base: Option<DevicePtr>,  // [lmhead_r, h] bf16
    pub lmhead_delta: Option<DevicePtr>, // [lmhead_t, h] f32
    pub lmhead_r: u32,
    pub lmhead_t: u32,
}

/// Stage-1 raw upload + the (host) clamped trainable ids the delta rows align to.
pub struct OverlayRawSlot {
    pub raw: OverlayRaw,
    /// PEFT `trainable_token_indices` VERBATIM (clamped to served vocab in
    /// Stage 2 so the loader stays independent of the served geometry).
    pub trainable: Vec<u32>,
}

/// Stage-2 compact result for one adapter slot's embed overlay. `rows` holds the
/// `n_override` replacement embedding rows in ascending-`ids` order; `slot_map`
/// is the `[vocab]` i32 lookup (`-1` default, `[id] = compact-row-index`) the
/// embed kernel indexes by token id. `lmhead` is `Some` only for an UNTIED head
/// that ships its own overlay tensors.
#[derive(Debug)]
pub struct EmbedOverlay {
    pub rows: DevicePtr,     // [n_override, h] bf16
    pub ids_dev: DevicePtr,  // u32[n_override] ascending vocab ids
    pub slot_map: DevicePtr, // i32[vocab], -1 default
    pub n_override: u32,
    /// `slot_map` length — the served vocab this overlay was built against.
    /// Recorded HERE (where `slot_map` is allocated) as the SSOT bound the
    /// embed kernel's `ids[r] < vocab` guard checks against (CWE-125).
    pub vocab: u32,
    pub lmhead: Option<LmHeadOverlay>,
}

/// Distinct output-projection overlay for an untied lm_head. Rows are recomputed
/// into logits by a `dot(hidden, row)` per overridden id (no `slot_map`).
#[derive(Debug)]
pub struct LmHeadOverlay {
    pub rows: DevicePtr,    // [n_override, h] bf16
    pub ids_dev: DevicePtr, // u32[n_override]
    pub n_override: u32,
}

/// Round-to-nearest-even f32 → bf16 bit pattern (build-time host cast for the
/// f32 `trainable_tokens_delta` replacement rows).
fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = ((bits >> 16) & 1) + 0x7fff;
    (bits.wrapping_add(round) >> 16) as u16
}

/// Copy one on-device tensor into a fresh owned device buffer, returning the
/// buffer and its leading dim (`shape[0]`). Validates the trailing dim == `h`.
fn stage_tensor(
    store: &WeightStore,
    name: &str,
    h: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, u32)> {
    let t = store.get(name)?;
    if t.shape.len() != 2 || t.shape[1] != h {
        bail!(
            "REJECT[overlay-shape]: '{name}' is {:?}, expected [rows, {h}] (hidden)",
            t.shape
        );
    }
    let bytes: usize = t.shape.iter().product::<usize>() * t.dtype.byte_size();
    let dst = gpu.alloc(bytes)?;
    gpu.copy_d2d(t.ptr, dst, bytes)?;
    Ok((dst, t.shape[0] as u32))
}

/// Stage 1 (loader): upload the classified overlay tensors of one adapter into
/// owned device scratch. `Ok(None)` when the adapter ships no overlay tensors.
/// `embed_full`/`lmhead_full` (`modules_to_save`) map onto the `*_base` slot.
pub fn stage_overlay_raw(
    store: &WeightStore,
    overlay: &OverlayTensors,
    peft: &PeftAdapterConfig,
    h: usize,
    gpu: &dyn GpuBackend,
) -> Result<Option<OverlayRawSlot>> {
    if overlay.is_empty() || overlay.lora_embedding_seen {
        // lora_embedding is a Tier-2 reject handled at audit; never staged.
        return Ok(None);
    }
    let mut raw = OverlayRaw::default();
    let embed_base_name = overlay.embed_base.as_ref().or(overlay.embed_full.as_ref());
    if let Some(name) = embed_base_name {
        let (p, r) = stage_tensor(store, name, h, gpu)?;
        raw.embed_base = Some(p);
        raw.embed_r = r;
    }
    if let Some(name) = &overlay.embed_delta {
        let (p, t) = stage_tensor(store, name, h, gpu)?;
        raw.embed_delta = Some(p);
        raw.embed_t = t;
    }
    let lmhead_base_name = overlay
        .lmhead_base
        .as_ref()
        .or(overlay.lmhead_full.as_ref());
    if let Some(name) = lmhead_base_name {
        let (p, r) = stage_tensor(store, name, h, gpu)?;
        raw.lmhead_base = Some(p);
        raw.lmhead_r = r;
    }
    if let Some(name) = &overlay.lmhead_delta {
        let (p, t) = stage_tensor(store, name, h, gpu)?;
        raw.lmhead_delta = Some(p);
        raw.lmhead_t = t;
    }
    if raw.embed_base.is_none() && raw.lmhead_base.is_none() {
        // A bare delta with no base to diff against is meaningless; a real
        // trainable_tokens adapter always ships base_layer.weight.
        bail!(
            "REJECT[overlay-no-base]: overlay tensors present but no embed/lm_head base row table"
        );
    }
    Ok(Some(OverlayRawSlot {
        raw,
        trainable: peft.trainable_token_indices.clone(),
    }))
}

/// The compact-override intermediate shared by the embed and lm_head builds:
/// the ascending overridden ids and their `[n, h]` bf16 replacement rows on
/// device. `None` when nothing is overridden.
struct Compact {
    ids: Vec<u32>,
    rows_dev: DevicePtr,
    ids_dev: DevicePtr,
    n: u32,
}

/// Row-diff `base` vs `served`, union with `kept` trainable ids, and materialize
/// the compact bf16 replacement rows (delta wins over base per `override_source`).
#[allow(clippy::too_many_arguments)]
fn compact_override(
    gpu: &dyn GpuBackend,
    kernels: &OverlayKernels,
    base: DevicePtr,
    delta: Option<DevicePtr>,
    r: u32,
    served: DevicePtr,
    vocab: usize,
    h: usize,
    kept: &[u32],
    stream: u64,
) -> Result<Option<Compact>> {
    let r_eff = (r as usize).min(vocab);
    // Row-diff kernel: flags[row] = max_i |base-served| > thresh.
    let flags_dev = gpu.alloc(r_eff.max(1))?;
    if r_eff > 0 {
        token_overlay::embed_rowdiff(
            gpu,
            kernels.rowdiff,
            base,
            served,
            flags_dev,
            r_eff as u32,
            h as u32,
            ROWDIFF_THRESH,
            stream,
        )?;
        gpu.synchronize(stream)?;
    }
    let mut flags = vec![0u8; r_eff];
    if r_eff > 0 {
        gpu.copy_d2h(flags_dev, &mut flags)?;
    }
    let _ = gpu.free(flags_dev);
    let row_diff: Vec<bool> = flags.iter().map(|&b| b != 0).collect();
    let ids = build_override_set(&row_diff, kept);
    if ids.is_empty() {
        return Ok(None);
    }
    let n = ids.len();
    // Materialize compact rows host-side (build-time, n is small): delta rows
    // f32→bf16, base rows copied verbatim bf16.
    let mut compact = vec![0u8; n * h * BF16_BYTES];
    let mut frow = vec![0u8; h * F32_BYTES];
    for (ci, &id) in ids.iter().enumerate() {
        let dst = &mut compact[ci * h * BF16_BYTES..(ci + 1) * h * BF16_BYTES];
        match override_source(id, kept) {
            Some(k) => {
                let d = delta.ok_or_else(|| {
                    anyhow::anyhow!(
                        "REJECT[overlay-delta-missing]: trainable id {id} but no delta tensor"
                    )
                })?;
                gpu.copy_d2h(d.offset(k * h * F32_BYTES), &mut frow)?;
                for i in 0..h {
                    let x = f32::from_le_bytes([
                        frow[i * 4],
                        frow[i * 4 + 1],
                        frow[i * 4 + 2],
                        frow[i * 4 + 3],
                    ]);
                    dst[i * 2..i * 2 + 2].copy_from_slice(&f32_to_bf16(x).to_le_bytes());
                }
            }
            None => {
                gpu.copy_d2h(base.offset(id as usize * h * BF16_BYTES), dst)?;
            }
        }
    }
    let rows_dev = gpu.alloc(compact.len())?;
    gpu.copy_h2d(&compact, rows_dev)?;
    let ids_bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
    let ids_dev = gpu.alloc(ids_bytes.len())?;
    gpu.copy_h2d(&ids_bytes, ids_dev)?;
    Ok(Some(Compact {
        ids,
        rows_dev,
        ids_dev,
        n: n as u32,
    }))
}

/// Stage 2 (`set_lora_weights`): row-diff against the served tables, compact the
/// override rows, build the `slot_map`, and free the Stage-1 raw scratch.
/// `Ok(None)` when the slot overrides nothing (silently inert = correct).
///
/// `tied` means the lm_head aliases the embed table (buffer aliasing OR a
/// quantized head derived from embed) — the caller then reuses the embed rows
/// for the logit recompute (`overlay_tables`), so no distinct lm_head build.
#[allow(clippy::too_many_arguments)]
pub fn build_overlay(
    gpu: &dyn GpuBackend,
    kernels: &OverlayKernels,
    slot: &OverlayRawSlot,
    served_embed: DevicePtr,
    served_lmhead: DevicePtr,
    vocab: usize,
    h: usize,
    tied: bool,
    stream: u64,
) -> Result<Option<EmbedOverlay>> {
    if kernels.rowdiff.0 == 0 || kernels.embed_overlay.0 == 0 {
        bail!(
            "REJECT[overlay-kernels-missing]: adapter ships token-overlay tensors but the \
             token_overlay CUDA kernels are not loaded (rebuild with the kernel image)"
        );
    }
    let raw = &slot.raw;
    let Some(embed_base) = raw.embed_base else {
        return Ok(None); // lm_head-only overlay without an embed base: not supported here.
    };
    let (kept, skipped) = clamp_trainable_to_vocab(&slot.trainable, raw.embed_r as usize, vocab)?;
    if skipped > 0 {
        tracing::warn!(
            "LoRA overlay: dropped {skipped} vocab-extension trainable id(s) beyond served vocab {vocab}"
        );
    }
    let Some(embed) = compact_override(
        gpu,
        kernels,
        embed_base,
        raw.embed_delta,
        raw.embed_r,
        served_embed,
        vocab,
        h,
        &kept,
        stream,
    )?
    else {
        return Ok(None);
    };
    // Host slot_map: [vocab] i32, -1 default, [id] = compact index.
    let mut slot_map = vec![-1i32; vocab];
    for (ci, &id) in embed.ids.iter().enumerate() {
        slot_map[id as usize] = ci as i32;
    }
    let sm_bytes: Vec<u8> = slot_map.iter().flat_map(|i| i.to_le_bytes()).collect();
    let slot_map_dev = gpu.alloc(sm_bytes.len())?;
    gpu.copy_h2d(&sm_bytes, slot_map_dev)?;

    // lm_head branch: distinct build only for an untied head that ships its own
    // overlay tensors; tied heads reuse the embed rows (handled in the tables).
    let lmhead = if let Some(base) = raw.lmhead_base.filter(|_| !tied) {
        compact_override(
            gpu,
            kernels,
            base,
            raw.lmhead_delta,
            raw.lmhead_r,
            served_lmhead,
            vocab,
            h,
            &kept,
            stream,
        )?
        .map(|c| LmHeadOverlay {
            rows: c.rows_dev,
            ids_dev: c.ids_dev,
            n_override: c.n,
        })
    } else {
        None
    };

    // Free Stage-1 raw scratch (build-time; on error the buffers leak, which
    // matches the existing adapter-load leak tolerance).
    for p in [
        raw.embed_base,
        raw.embed_delta,
        raw.lmhead_base,
        raw.lmhead_delta,
    ]
    .into_iter()
    .flatten()
    {
        let _ = gpu.free(p);
    }

    Ok(Some(EmbedOverlay {
        rows: embed.rows_dev,
        ids_dev: embed.ids_dev,
        slot_map: slot_map_dev,
        n_override: embed.n,
        vocab: vocab as u32,
        lmhead,
    }))
}

#[cfg(test)]
#[path = "overlay_build_tests.rs"]
mod tests;
