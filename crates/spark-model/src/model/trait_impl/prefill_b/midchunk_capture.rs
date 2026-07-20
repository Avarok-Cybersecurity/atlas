// SPDX-License-Identifier: AGPL-3.0-only

//! MID-CHUNK SSM tail capture (default-on, opt-out `ATLAS_SSM_TAIL_MIDCHUNK=0`).
//!
//! The clamp-based `ATLAS_SSM_TAIL_CKPT` path lands a chunk boundary on
//! `ssm_tail_boundary(tb)` and saves the SSM snapshot there via an extra
//! forward pass over the trailing tokens (~868 ms — cancels the replay win).
//! This path instead lets the prefill chunk run its natural full span and
//! captures each GDN layer's recurrent + conv state exactly at `tb` by
//! splitting only the two cheap per-token GDN kernels (split4 recurrence +
//! conv1d) — no extra pass, no projection/FFN re-run.
//!
//! Two capture points per pass:
//!   * `tb` — the block-floored matched-prefix boundary, registered as the
//!     session TAIL snapshot.
//!   * `tb - block_size` — one KV block earlier, registered as a NON-TAIL
//!     intermediate. On ~5/19 warm turns the next turn's block-floored
//!     `matched_tokens` lands exactly `tb - block_size` (the chat-template
//!     generation-suffix / detokenize-retokenize divergence puts the longest
//!     common prefix one block short of the tail). The tail@`tb` is then
//!     filtered by `token_count > matched_tokens`, so without this earlier
//!     restore point those turns fall back to a coarse checkpoint and replay
//!     up to ~479 SSM tokens. With it, `matched ∈ {tb, tb-bs}` are both exact.
//!
//! Flow:
//!   1. [`TransformerModel::prepare_midchunk_capture`] (before `forward_layers`)
//!      decides whether this pass spans `tb` (and optionally `tb-bs`), reserves
//!      one or two snapshot slots, and precomputes the per-SSM-layer dst pointers.
//!   2. `forward_layers` threads the plan into `ForwardContext::midchunk_capture`;
//!      each SSM layer's `prefill_inner` splits its h_state/conv_state kernels at
//!      `cap_local` (and `cap_local - bs` when present) and D2D-copies each
//!      captured state into its reserved slot.
//!   3. [`TransformerModel::finalize_midchunk_capture`] (after `forward_layers`)
//!      registers the tail slot as the session tail and the earlier slot as an
//!      intermediate snapshot in the index.
//!
//! All behavior is gated on `ssm_tail_midchunk_enabled()` — opt-out
//! (`ATLAS_SSM_TAIL_MIDCHUNK=0`) is a no-op (returns `None`) and byte-identical
//! to prior behavior.

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
    /// Reserved TAIL snapshot slot (== snapshot id used for registration).
    pub snap_slot: usize,
    /// Block-floored matched-prefix boundary the tail snapshot represents.
    pub tb: usize,
    /// Per-SSM-layer h_state destination (offset to `snap_slot`).
    pub h_dsts: Vec<DevicePtr>,
    /// Per-SSM-layer conv_state destination (offset to `snap_slot`).
    pub conv_dsts: Vec<DevicePtr>,
    /// Bytes per layer of h_state.
    pub h_bytes: usize,
    /// Bytes per layer of conv_state.
    pub conv_bytes: usize,
    /// Block size (`ssm_tail_boundary`'s grid) — used to register the earlier
    /// intermediate and to derive `tb - bs`.
    pub bs: usize,
    /// EARLIER capture at `tb - bs`: split point in local coords
    /// (`cap_local - bs`). `Some` only when the pass also covers it AND a
    /// second slot could be reserved.
    pub cap_local_early: Option<usize>,
    /// Reserved intermediate slot for the `tb - bs` snapshot.
    pub snap_slot_early: Option<usize>,
    /// Token boundary the earlier snapshot represents (`tb - bs`).
    pub tb_early: Option<usize>,
    /// Per-SSM-layer h_state destination for the `tb - bs` slot.
    pub h_dsts_early: Vec<DevicePtr>,
    /// Per-SSM-layer conv_state destination for the `tb - bs` slot.
    pub conv_dsts_early: Vec<DevicePtr>,
}

impl TransformerModel {
    /// Reserve a Marconi snapshot slot, reclaiming one from the cache on
    /// exhaustion. Returns `None` only when the pool is full and nothing is
    /// evictable — the caller then degrades gracefully (fewer/no captures).
    fn reserve_snapshot_slot(
        &self,
        session_hash: u64,
        kv_cache: &mut PagedKvCache,
    ) -> Option<usize> {
        match self.ssm_snapshots.reserve_tail_slot(session_hash) {
            Some(s) => Some(s),
            None => {
                if self.ssm_snapshots.reclaim_from_cache(
                    self.prefix_cache.as_ref(),
                    kv_cache,
                    self.ssm_tier_store.as_deref(),
                    self.gpu.as_ref(),
                ) {
                    self.ssm_snapshots.reserve_tail_slot(session_hash)
                } else {
                    None
                }
            }
        }
    }

    /// Decide + set up an in-pass mid-chunk tail capture for the prefill pass
    /// over local token range `[proc_start, proc_start + proc_count)`.
    ///
    /// Returns `None` (=> no capture, byte-identical behavior) when the flag is
    /// off, the snapshot pool is disabled, there is no `tb`, the pass does not
    /// strictly span `tb`, or no snapshot slot can be reserved (even after a
    /// cache reclaim). The earlier `tb - bs` capture is best-effort on top: if a
    /// second slot cannot be reserved, only the tail is captured.
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
        // Reuse gate: capture costs a per-prefill kernel split + D2D copy and
        // is only ever consumable by a LATER request of the SAME session (the
        // snapshot lookup is session-gated). A session seen for the FIRST
        // time — which includes ALL single-turn traffic, whose ≤1024-token
        // prefix hash is unique per request — can never reuse this capture,
        // so skip it entirely; sessions gain capture from their second
        // request onward (the first request's leaf save tags the session),
        // exactly when reuse begins. Measured (35B FP8 webserver_ok N=10,
        // 2026-07-20): capture-always Σ1846s/9-of-10 + 257 tok/turn vs
        // capture-off Σ1495s/10-of-10 + 203 tok/turn on sessionless agentic
        // traffic; multi-turn warm-TTFT (-54% max) is preserved from turn 3.
        if !self.ssm_snapshots.session_has_history(seq.session_hash) {
            return None;
        }
        let bs = kv_cache.block_size();
        let tb = spark_runtime::ssm_tail_boundary(tokens.len(), bs)?;
        // Only capture when this pass strictly crosses tb.
        if !(proc_start < tb && tb < proc_start + proc_count) {
            return None;
        }
        let cap_local = tb - proc_start;
        let n = self.ssm_snapshots.num_ssm_layers();

        // TAIL slot @ tb — required. Reserve first (reclaim on exhaustion).
        let snap_slot = self.reserve_snapshot_slot(seq.session_hash, kv_cache)?;
        let mut h_dsts = Vec::with_capacity(n);
        let mut conv_dsts = Vec::with_capacity(n);
        for l in 0..n {
            h_dsts.push(self.ssm_snapshots.tail_h_dst(l, snap_slot));
            conv_dsts.push(self.ssm_snapshots.tail_conv_dst(l, snap_slot));
        }

        // EARLIER slot @ tb - bs — optional. Only when the pass covers that
        // point (cap_local - bs > 0, equivalently proc_start < tb - bs) AND a
        // second slot is available. Best-effort: degrade to tail-only on miss.
        let mut cap_local_early = None;
        let mut snap_slot_early = None;
        let mut tb_early = None;
        let mut h_dsts_early = Vec::new();
        let mut conv_dsts_early = Vec::new();
        if bs > 0
            && cap_local > bs
            && tb > bs
            && let Some(slot2) = self.reserve_snapshot_slot(seq.session_hash, kv_cache)
        {
            cap_local_early = Some(cap_local - bs);
            snap_slot_early = Some(slot2);
            tb_early = Some(tb - bs);
            h_dsts_early = Vec::with_capacity(n);
            conv_dsts_early = Vec::with_capacity(n);
            for l in 0..n {
                h_dsts_early.push(self.ssm_snapshots.tail_h_dst(l, slot2));
                conv_dsts_early.push(self.ssm_snapshots.tail_conv_dst(l, slot2));
            }
        }

        Some(MidCapturePlan {
            cap_local,
            snap_slot,
            tb,
            h_dsts,
            conv_dsts,
            h_bytes: self.ssm_snapshots.h_bytes(),
            conv_bytes: self.ssm_snapshots.conv_bytes(),
            bs,
            cap_local_early,
            snap_slot_early,
            tb_early,
            h_dsts_early,
            conv_dsts_early,
        })
    }

    /// Register the captured slots after the full forward pass has copied the
    /// @tb (and, when present, @tb-bs) state into them:
    ///
    /// * TAIL slot -> `insert_tail_snapshot` (supersedes the session tail).
    /// * EARLIER slot -> `insert_intermediate_snapshot` (a NON-tail entry, so
    ///   it does NOT evict the tail; both become valid restore points).
    ///
    /// Frees any snapshot id displaced by either insert.
    pub(in crate::model) fn finalize_midchunk_capture(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        plan: &MidCapturePlan,
    ) {
        for old in self.prefix_cache.insert_tail_snapshot(
            &tokens[..plan.tb],
            plan.snap_slot,
            seq.session_hash,
            seq.adapter_id,
        ) {
            self.ssm_snapshots.free(old);
        }
        tracing::info!(
            "midchunk tail SSM capture at token {} (snap {})",
            plan.tb,
            plan.snap_slot
        );

        if let (Some(tb_early), Some(slot2)) = (plan.tb_early, plan.snap_slot_early) {
            // Intermediate (non-tail): the radix-tree nodes for [0, tb_early)
            // are laid down by this turn's finalize_last `insert([0, total))`,
            // so this is index-only (block_table/disk are ignored by the impl).
            if let Some(old) = self.prefix_cache.insert_intermediate_snapshot(
                &tokens[..tb_early],
                &[],
                &[],
                plan.bs,
                slot2,
                seq.session_hash,
                tb_early,
                seq.adapter_id,
            ) {
                self.ssm_snapshots.free(old);
            }
            tracing::info!(
                "midchunk EARLY tail SSM capture at token {} (snap {})",
                tb_early,
                slot2
            );
        }
    }
}
