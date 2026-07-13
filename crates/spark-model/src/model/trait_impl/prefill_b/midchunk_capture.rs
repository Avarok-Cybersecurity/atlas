// SPDX-License-Identifier: AGPL-3.0-only

//! MID-CHUNK SSM tail capture (`ATLAS_SSM_TAIL_MIDCHUNK=1`).
//!
//! The clamp-based `ATLAS_SSM_TAIL_CKPT` path lands a chunk boundary on
//! `ssm_tail_boundary(tb)` and saves the SSM snapshot there via an extra
//! forward pass over the trailing tokens (~868 ms — cancels the replay win).
//! This path instead lets the prefill chunk run its natural full span and
//! captures each GDN layer's recurrent + conv state exactly at `tb` by
//! splitting only the two cheap per-token GDN kernels (split4 recurrence +
//! conv1d) — no extra pass, no projection/FFN re-run.
//!
//! Flow:
//!   1. [`TransformerModel::prepare_midchunk_capture`] (before `forward_layers`)
//!      decides whether this pass spans `tb`, reserves a snapshot slot, and
//!      precomputes the per-SSM-layer destination pointers.
//!   2. `forward_layers` threads the plan into `ForwardContext::midchunk_capture`;
//!      each SSM layer's `prefill_inner` splits its h_state/conv_state kernels at
//!      `cap_local` and D2D-copies the @tb state into the reserved slot.
//!   3. [`TransformerModel::finalize_midchunk_capture`] (after `forward_layers`)
//!      registers the reserved slot as the session's tail snapshot in the index.
//!
//! All behavior is gated on `ssm_tail_midchunk_enabled()` — flag off is a no-op
//! (returns `None`) and byte-identical to prior behavior.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// Per-pass plan for a mid-chunk tail capture. Owns the per-SSM-layer
/// destination pointer vectors that `ForwardContext::midchunk_capture`
/// borrows for the duration of `forward_layers`.
pub(in crate::model) struct MidCapturePlan {
    /// Split point in local (chunk) token coordinates (`tb - proc_start`).
    pub cap_local: usize,
    /// Reserved snapshot slot (== snapshot id used for registration).
    pub snap_slot: usize,
    /// Block-floored matched-prefix boundary the snapshot represents.
    pub tb: usize,
    /// Per-SSM-layer h_state destination (offset to `snap_slot`).
    pub h_dsts: Vec<DevicePtr>,
    /// Per-SSM-layer conv_state destination (offset to `snap_slot`).
    pub conv_dsts: Vec<DevicePtr>,
    /// Bytes per layer of h_state.
    pub h_bytes: usize,
    /// Bytes per layer of conv_state.
    pub conv_bytes: usize,
}

impl TransformerModel {
    /// Decide + set up an in-pass mid-chunk tail capture for the prefill pass
    /// over local token range `[proc_start, proc_start + proc_count)`.
    ///
    /// Returns `None` (=> no capture, byte-identical behavior) when the flag is
    /// off, the snapshot pool is disabled, there is no `tb`, the pass does not
    /// strictly span `tb`, or no snapshot slot can be reserved (even after a
    /// cache reclaim). Never fails the prefill — capture is best-effort.
    pub(in crate::model) fn prepare_midchunk_capture(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        proc_start: usize,
        proc_count: usize,
    ) -> Option<MidCapturePlan> {
        if !spark_runtime::ssm_tail_midchunk_enabled() || !self.ssm_snapshots.is_enabled() {
            return None;
        }
        let bs = kv_cache.block_size();
        let tb = spark_runtime::ssm_tail_boundary(tokens.len(), bs)?;
        // Only capture when this pass strictly crosses tb.
        if !(proc_start < tb && tb < proc_start + proc_count) {
            return None;
        }
        let cap_local = tb - proc_start;

        // Reserve a slot; on exhaustion reclaim one from the cache and retry.
        let snap_slot = match self.ssm_snapshots.reserve_tail_slot(seq.session_hash) {
            Some(s) => s,
            None => {
                if self
                    .ssm_snapshots
                    .reclaim_from_cache(self.prefix_cache.as_ref(), kv_cache)
                {
                    self.ssm_snapshots.reserve_tail_slot(seq.session_hash)?
                } else {
                    return None;
                }
            }
        };

        let n = self.ssm_snapshots.num_ssm_layers();
        let mut h_dsts = Vec::with_capacity(n);
        let mut conv_dsts = Vec::with_capacity(n);
        for l in 0..n {
            h_dsts.push(self.ssm_snapshots.tail_h_dst(l, snap_slot));
            conv_dsts.push(self.ssm_snapshots.tail_conv_dst(l, snap_slot));
        }
        Some(MidCapturePlan {
            cap_local,
            snap_slot,
            tb,
            h_dsts,
            conv_dsts,
            h_bytes: self.ssm_snapshots.h_bytes(),
            conv_bytes: self.ssm_snapshots.conv_bytes(),
        })
    }

    /// Register the reserved slot as this session's tail snapshot after the
    /// full forward pass has captured the @tb state into it. Frees any
    /// snapshot id displaced by the supersede.
    pub(in crate::model) fn finalize_midchunk_capture(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        plan: &MidCapturePlan,
    ) {
        for old in
            self.prefix_cache
                .insert_tail_snapshot(&tokens[..plan.tb], plan.snap_slot, seq.session_hash)
        {
            self.ssm_snapshots.free(old);
        }
        tracing::info!(
            "midchunk tail SSM capture at token {} (snap {})",
            plan.tb,
            plan.snap_slot
        );
    }
}
