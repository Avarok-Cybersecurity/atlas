// SPDX-License-Identifier: AGPL-3.0-only

//! Model side of the GDN write-on-accept K=4 verify: the one-time stash
//! binding and the post-verdict fold gate.
//!
//! The batched verify asks for write-on-accept per call
//! (`VerifyBatchedOpts::write_on_accept`, set only by the DFlash batched
//! step). This file owns the two model-level facts the fold needs and the
//! layer cannot know:
//!
//! * whether the verify that JUST completed ran under the request with its
//!   pointer tables staged (`gdn_woa_eligible`, set at the end of that
//!   verify, cleared at the start of every verify, consumed here), and
//! * the stash memory, allocated on the first request and bound into every
//!   GDN layer before any capture bakes its address.
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
        // and pays nothing.
        let seq_floats = self
            .layers
            .iter()
            .filter_map(|l| l.gdn_woa_stash_seq_floats())
            .max()
            .unwrap_or(0);
        let n_ssm = self.config.num_ssm_layers();
        if seq_floats == 0 || n_ssm == 0 {
            // Mark as probed so the next request returns without the walk.
            *bound = (DevicePtr::NULL, DevicePtr(u64::MAX), 0);
            return Ok(false);
        }
        // The batched verify never exceeds the pointer-table width, and a
        // serve never batches more than its configured sequence pool.
        let seqs = VERIFY_WY_TABLE_SEQS.min((self.levers.max_decode_seqs as usize).max(2));
        let stash_bytes = n_ssm * seqs * seq_floats * 4;
        let flag_bytes = n_ssm * 4;
        let flags = self.gpu.alloc(flag_bytes)?;
        self.gpu.memset(flags, 0, flag_bytes)?;
        let stash = self.gpu.alloc(stash_bytes)?;
        tracing::info!(
            "GDN write-on-accept: bound {:.1} MB stash ({} GDN layers x {} seqs) on first request",
            stash_bytes as f64 / 1e6,
            n_ssm,
            seqs
        );
        let mut ssm_idx = 0usize;
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) != atlas_core::config::LayerType::LinearAttention {
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

    /// The post-verdict fold. Declines (`Ok(false)`, host h restore runs)
    /// unless the batched verify that just completed ran under a
    /// write-on-accept request with its pointer tables staged; the flag is
    /// consumed so a second call, or a later per-sequence step, folds
    /// nothing. `gdn_woa_folded_slots` is cleared UNCONDITIONALLY at entry:
    /// a stale entry would make a later `commit_accepted_prefix` for a
    /// re-claimed slot skip its h restore.
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
            if self.config.layer_type(i) != atlas_core::config::LayerType::LinearAttention {
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
        }
        Ok(any)
    }
}
