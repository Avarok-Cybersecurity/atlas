// SPDX-License-Identifier: AGPL-3.0-only

//! Model side of the GDN write-on-accept K=4 verify: the one-time stash
//! binding and the post-verdict fold gate.
//!
//! The batched verify asks for write-on-accept per call
//! (`VerifyBatchedOpts::write_on_accept`: the DFlash batched step, and the
//! MTP batched step with the port row `gdn_woa_mtp` on). This file owns the
//! model-level facts the fold needs and the layer cannot know:
//!
//! * whether the verify that JUST completed ran under the request with its
//!   pointer tables staged (`gdn_woa_eligible`, set at the end of that
//!   verify, cleared at the start of every verify, consumed here),
//! * the stash memory, allocated on the first request and bound into every
//!   GDN layer before any capture bakes its address, sized for the widest
//!   width this model's batched verify can ask for (four rows with the row
//!   off, exactly as before; up to the DFlash band with it on, for the Ada
//!   K-generic twin), and
//! * the folded-slot record's lifetime: the MTP step releases it once its
//!   verdicts are applied (`gdn_woa_release_folded_dispatch`).
//!
//! provenance-id: 526f6e616c6420522e205374657369616b

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::TransformerModel;
use crate::layer::{VERIFY_WY_LAYER_STRIDE_BYTES, VERIFY_WY_TABLE_SEQS};

impl TransformerModel {
    /// Bind the write-on-accept stash into every GDN layer, allocating it on
    /// the first call. Returns whether at least one layer is bound. Runs
    /// pre-capture from the batched verify, so the allocation never lands
    /// inside a graph capture and the baked addresses never move.
    pub(super) fn gdn_woa_bind(&self) -> Result<bool> {
        let mut bound = self.gdn_woa_bound.lock();
        if !bound.1.is_null() {
            return Ok(bound.2 > 0);
        }
        // Per-layer stash width comes from the layer (its dims and whether
        // its kernels linked); a model with no capable layer binds nothing
        // and pays nothing. The layer reports the K=4 (four-row) width.
        let k4_floats = self
            .layers
            .iter()
            .filter_map(|l| l.gdn_woa_stash_seq_floats())
            .max()
            .unwrap_or(0);
        let n_ssm = self.config.num_ssm_layers();
        if k4_floats == 0 || n_ssm == 0 {
            // Mark as probed so the next request returns without the walk.
            *bound = (DevicePtr::NULL, DevicePtr(u64::MAX), 0);
            return Ok(false);
        }
        // Rows per sequence: 4 with the row off (the shipped stash, byte for
        // byte); with it on, the widest batched-verify width this model can
        // ask for, so every K-generic twin width fits its stash slab.
        let row_on = crate::layers::ops::port_rows::gdn_woa_mtp::row();
        let dflash_k = self
            .dflash_hidden_save
            .is_some()
            .then_some(self.dflash_kgamma);
        let rows = crate::port::gdn_woa::stash_rows_cap(row_on, dflash_k);
        let seq_floats = crate::port::gdn_woa::scale_k4_stash_floats(k4_floats, rows);
        // The batched verify never exceeds the pointer-table width, and a
        // serve never batches more than its configured sequence pool.
        let seqs = VERIFY_WY_TABLE_SEQS.min((self.levers.max_decode_seqs as usize).max(2));
        let stash_bytes = n_ssm * seqs * seq_floats * 4;
        let flag_bytes = n_ssm * 4;
        let (flags, stash) = if row_on {
            // A late allocation can fail on a full card. With the row on the
            // request is declined (parent kernels, host restore) instead of
            // failing the verify and finishing every sequence in the batch.
            match self.gdn_woa_alloc(flag_bytes, stash_bytes) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(
                        "GDN write-on-accept: stash of {:.1} MB not allocated ({e:#}); \
                         write-on-accept stays off for this model",
                        stash_bytes as f64 / 1e6
                    );
                    *bound = (DevicePtr::NULL, DevicePtr(u64::MAX), 0);
                    return Ok(false);
                }
            }
        } else {
            let flags = self.gpu.alloc(flag_bytes)?;
            self.gpu.memset(flags, 0, flag_bytes)?;
            let stash = self.gpu.alloc(stash_bytes)?;
            (flags, stash)
        };
        if row_on {
            tracing::info!(
                "GDN write-on-accept: bound {:.1} MB stash ({} GDN layers x {} seqs x {} rows, \
                 gdn_woa_mtp) on first request",
                stash_bytes as f64 / 1e6,
                n_ssm,
                seqs,
                rows
            );
        } else {
            tracing::info!(
                "GDN write-on-accept: bound {:.1} MB stash ({} GDN layers x {} seqs) on first request",
                stash_bytes as f64 / 1e6,
                n_ssm,
                seqs
            );
        }
        let mut ssm_idx = 0usize;
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) != avarok_core::config::LayerType::LinearAttention {
                continue;
            }
            layer.gdn_woa_bind(
                flags.offset(ssm_idx * 4),
                stash.offset(ssm_idx * seqs * seq_floats * 4),
                seqs,
            );
            ssm_idx += 1;
        }
        *bound = (flags, stash, seqs);
        Ok(true)
    }

    /// Flag words (zeroed) then stash, freeing the flags if the stash does
    /// not fit, so a declined bind leaves nothing allocated.
    fn gdn_woa_alloc(
        &self,
        flag_bytes: usize,
        stash_bytes: usize,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let flags = self.gpu.alloc(flag_bytes)?;
        let stash = self
            .gpu
            .memset(flags, 0, flag_bytes)
            .and_then(|()| self.gpu.alloc(stash_bytes));
        match stash {
            Ok(stash) => Ok((flags, stash)),
            Err(e) => {
                if let Err(free_err) = self.gpu.free(flags) {
                    tracing::warn!("GDN write-on-accept: freeing the flag words: {free_err:#}");
                }
                Err(e)
            }
        }
    }

    /// Drop every slot still recorded as folded. The MTP batched step calls
    /// this once its verdicts are applied: a sequence that finishes while
    /// emitting a full accept returns before its `commit_accepted_prefix`,
    /// which is what normally consumes its entry, and slot indices are reused
    /// across requests, so a leftover entry would make the slot's next
    /// occupant skip an `h` restore it needs (the failure class the
    /// unconditional clear in `gdn_fold_accepted_dispatch` closed).
    pub(super) fn gdn_woa_release_folded_dispatch(&self) {
        self.gdn_woa_folded_slots.lock().clear();
    }

    /// The post-verdict fold. Declines (`Ok(false)`, host h restore runs)
    /// unless the batched verify that just completed ran under a
    /// write-on-accept request with its pointer tables staged; the flag is
    /// consumed so a second call, or a later per-sequence step, folds
    /// nothing. `gdn_woa_folded_slots` is cleared UNCONDITIONALLY at entry:
    /// a stale entry would make a later `commit_accepted_prefix` for a
    /// re-claimed slot skip its h restore.
    ///
    /// Why unconditional (main, 2026-09-18): this was `if any { clear();
    /// extend() }`, so a fold that folded NOTHING left the PREVIOUS batch's
    /// slot list in place. The reader (`async_chkpt.rs`) looks its own
    /// `slot_idx` up in that list and destructively consumes the entry to
    /// decide whether `h` still needs restoring from the checkpoint, and
    /// slot indices are reused across requests — so a sequence could find a
    /// stale entry from an earlier batch and skip the restore, carrying a
    /// wrong GDN state forward. Clearing on every call makes the list mean
    /// "the slots folded by the MOST RECENT fold", the only claim the reader
    /// can safely consume. (The C=4..8 SimHash-watchdog fires measured on
    /// gb10 that motivated the OPT-IN default are documented at
    /// `gdn_flags::gdn_woa_enabled`; that scope fix alone did not clear them.)
    pub(super) fn gdn_fold_accepted_dispatch(
        &self,
        slots: &[usize],
        accepted_rows: &[u32],
        k_rows: usize,
    ) -> Result<bool> {
        self.gdn_woa_folded_slots.lock().clear();
        let eligible = self
            .gdn_woa_eligible
            .swap(false, std::sync::atomic::Ordering::AcqRel);
        if !eligible
            || self.gdn_woa_na_tab.is_null()
            || self.verify_wy_tables.is_null()
            || accepted_rows.is_empty()
            || accepted_rows.len() > VERIFY_WY_TABLE_SEQS
        {
            return Ok(false);
        }
        let mut host = [0u32; VERIFY_WY_TABLE_SEQS];
        host[..accepted_rows.len()].copy_from_slice(accepted_rows);
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let stream = self.gpu.default_stream();
        self.gpu
            .copy_h2d_async(&bytes, self.gdn_woa_na_tab, stream)?;
        let mut ssm_idx = 0usize;
        let mut any = false;
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) != avarok_core::config::LayerType::LinearAttention {
                continue;
            }
            let h_table = self
                .verify_wy_tables
                .offset(ssm_idx * VERIFY_WY_LAYER_STRIDE_BYTES);
            any |= layer.gdn_fold_accepted(
                self.gpu.as_ref(),
                h_table,
                self.gdn_woa_na_tab,
                k_rows,
                accepted_rows.len(),
                stream,
            )?;
            ssm_idx += 1;
        }
        if any {
            self.gdn_woa_folded_slots.lock().extend_from_slice(slots);
            crate::port::gdn_woa::record_fold(k_rows);
        }
        Ok(any)
    }
}
