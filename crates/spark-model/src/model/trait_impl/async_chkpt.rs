// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Kill-switch for the batched SSM state copy: `ATLAS_SSM_BULK_COPY=0`
/// restores the per-layer `copy_d2d_async` loops for an A/B. Read once —
/// this sits on the per-step path and `env::var` is not free there.
fn ssm_bulk_copy_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("ATLAS_SSM_BULK_COPY").ok().as_deref(),
            Some("0")
        )
    })
}

impl TransformerModel {
    /// Whether `bulk_state_copy` can be used for a copy of `bytes` per layer.
    ///
    /// Does NOT check pointer provenance — pair it with
    /// [`Self::ssm_state_is_pool_backed`], which is what makes the base-pointer
    /// arithmetic in the kernel equal to the pointers the loop would have used.
    fn bulk_copy_available(&self, bytes: usize) -> bool {
        ssm_bulk_copy_enabled()
            && self.ssm_bulk_copy_kernel.0 != 0
            && self.ssm_pool.num_ssm_layers > 0
            && bytes % 16 == 0
    }

    /// True when every SSM layer's live + checkpoint pointers are exactly the
    /// pool addresses for `seq.slot_idx`, which is the layout
    /// `ssm_state_bulk_copy` computes from the uploaded base arrays.
    ///
    /// They are, in the normal MTP/DFlash path — `meta.rs` and `sequence.rs`
    /// wire them straight from the pool. But `async_chkpt.rs:51` and
    /// `verify_a.rs:275` can lazily `gpu.alloc` a checkpoint when one is
    /// missing, and those allocations are NOT in the pool. Falling back to the
    /// loop in that case is the difference between a fast path and silent SSM
    /// corruption, so this is checked on every call. Pure host pointer
    /// comparison, no driver calls.
    fn ssm_state_is_pool_backed(&self, seq: &SequenceState) -> bool {
        use crate::layer::SsmLayerState;

        let pool = &self.ssm_pool;
        if pool.h_state_bases_dev.0 == 0
            || pool.conv_state_bases_dev.0 == 0
            || pool.h_checkpoint_bases_dev.0 == 0
            || pool.conv_checkpoint_bases_dev.0 == 0
        {
            return false;
        }
        let slot = seq.slot_idx;
        let mut ssm_layer_idx = 0usize;
        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let Some(ssm) = layer_state.as_any().downcast_ref::<SsmLayerState>() else {
                return false;
            };
            if ssm.h_state != pool.h_state(ssm_layer_idx, slot)
                || ssm.conv_state != pool.conv_state(ssm_layer_idx, slot)
                || ssm.h_state_checkpoint != Some(pool.h_checkpoint(ssm_layer_idx, slot))
                || ssm.conv_state_checkpoint != Some(pool.conv_checkpoint(ssm_layer_idx, slot))
            {
                return false;
            }
            ssm_layer_idx += 1;
        }
        ssm_layer_idx == pool.num_ssm_layers
    }

    /// Additionally require the intermediate pools, for the partial-accept
    /// commit branch which reads `h_intermediate` / `conv_intermediate`.
    fn ssm_intermediates_are_pool_backed(&self, seq: &SequenceState) -> bool {
        self.ssm_pool.h_intermediate_bases_dev.0 != 0
            && self.ssm_pool.conv_intermediate_bases_dev.0 != 0
            && self.ssm_state_is_pool_backed(seq)
    }

    /// One launch covering all SSM layers, replacing a per-layer
    /// `copy_d2d_async` loop. `src_off`/`dst_off` are byte offsets applied to
    /// every layer's pool base. Bit-identical to the loop: same bytes, same
    /// direction, same stream.
    ///
    /// Callers must have checked [`Self::bulk_copy_available`] and the
    /// matching pool-backed predicate first.
    fn bulk_state_copy(
        &self,
        src_bases: DevicePtr,
        dst_bases: DevicePtr,
        src_off: usize,
        dst_off: usize,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        // One-shot path confirmation: the pool-backed guard falls back to the
        // per-layer loop silently, so without this line no boot log can prove
        // the batched path ever fired.
        static FIRED: std::sync::Once = std::sync::Once::new();
        FIRED.call_once(|| {
            tracing::info!(
                "SSM bulk copy: batched path ACTIVE ({} layers/launch)",
                self.ssm_pool.num_ssm_layers
            );
        });
        const BLOCK: u32 = 256;
        let n_u4 = (bytes / 16) as u32;
        // Cap the x-grid so wide layer counts do not explode the grid; the
        // kernel strides, so any block count is correct.
        let blocks = n_u4.div_ceil(BLOCK).clamp(1, 512);
        KernelLaunch::new(self.gpu.as_ref(), self.ssm_bulk_copy_kernel)
            .grid([blocks, self.ssm_pool.num_ssm_layers as u32, 1])
            .block([BLOCK, 1, 1])
            .arg_ptr(src_bases)
            .arg_ptr(dst_bases)
            .arg_u64(src_off as u64)
            .arg_u64(dst_off as u64)
            .arg_u32(n_u4)
            .launch(stream)
    }

    pub(super) fn start_checkpoint_async_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        use crate::layer::SsmLayerState;

        let stream = self.secondary_stream;
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                let h_bytes = nv * vd * kd * 4;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                if ssm.h_state_checkpoint.is_none() {
                    ssm.h_state_checkpoint = Some(self.gpu.alloc(h_bytes)?);
                }
                if ssm.conv_state_checkpoint.is_none() {
                    ssm.conv_state_checkpoint = Some(self.gpu.alloc(conv_bytes)?);
                }

                self.gpu.copy_d2d_async(
                    ssm.h_state,
                    ssm.h_state_checkpoint.unwrap(),
                    h_bytes,
                    stream,
                )?;
                self.gpu.copy_d2d_async(
                    ssm.conv_state,
                    ssm.conv_state_checkpoint.unwrap(),
                    conv_bytes,
                    stream,
                )?;
            }
        }
        // Record event so default stream can wait (GPU-side, no CPU block).
        self.gpu.record_event(self.secondary_event, stream)?;
        Ok(())
    }

    pub(super) fn start_rollback_and_checkpoint_async_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        use crate::layer::SsmLayerState;

        let stream = self.secondary_stream;
        let mut ssm_layer_idx = 0usize;

        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                let h_bytes = nv * vd * kd * 4;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                // Rollback: restore h_state and conv_state from the appropriate source.
                if num_accepted == 0 {
                    // No tokens accepted: restore from checkpoint (pre-verify state).
                    if let Some(ckpt) = ssm.h_state_checkpoint {
                        self.gpu
                            .copy_d2d_async(ckpt, ssm.h_state, h_bytes, stream)?;
                    }
                    if let Some(ckpt) = ssm.conv_state_checkpoint {
                        self.gpu
                            .copy_d2d_async(ckpt, ssm.conv_state, conv_bytes, stream)?;
                    }
                } else {
                    // Partial acceptance: restore from intermediate[num_accepted - 1].
                    let slot = seq.slot_idx;
                    let inter_idx = num_accepted - 1;
                    let h_inter = self.ssm_pool.h_intermediate(ssm_layer_idx, slot, inter_idx);
                    let conv_inter =
                        self.ssm_pool
                            .conv_intermediate(ssm_layer_idx, slot, inter_idx);
                    self.gpu
                        .copy_d2d_async(h_inter, ssm.h_state, h_bytes, stream)?;
                    self.gpu
                        .copy_d2d_async(conv_inter, ssm.conv_state, conv_bytes, stream)?;
                }

                // Checkpoint the (now rolled-back) state for the next verify.
                if let Some(ckpt) = ssm.h_state_checkpoint {
                    self.gpu
                        .copy_d2d_async(ssm.h_state, ckpt, h_bytes, stream)?;
                }
                if let Some(ckpt) = ssm.conv_state_checkpoint {
                    self.gpu
                        .copy_d2d_async(ssm.conv_state, ckpt, conv_bytes, stream)?;
                }

                ssm_layer_idx += 1;
            }
        }
        // Record event so default stream can wait (GPU-side, no CPU block).
        self.gpu.record_event(self.secondary_event, stream)?;
        Ok(())
    }

    pub(super) fn sync_secondary_dispatch(&self) -> Result<()> {
        // GPU-side event sync: make the default stream wait for the secondary
        // event. Zero CPU cost — the GPU scheduler handles the dependency.
        self.gpu
            .stream_wait_event(self.gpu.default_stream(), self.secondary_event)
    }

    /// Record the snapshot-ordering event on `save_stream` AFTER an SSM-snapshot
    /// save's D2D copies have been enqueued. A later warm Marconi restore on the
    /// prefill stream waits on this event ([`Self::wait_snapshot_saves_dispatch`])
    /// so it never reads a snapshot slot whose save copy is still in flight on
    /// another stream. See the `snapshot_event` doc (types.rs) for the race.
    pub(super) fn record_snapshot_save_dispatch(&self, save_stream: u64) -> Result<()> {
        self.gpu.record_event(self.snapshot_event, save_stream)
    }

    /// Order `restore_stream` after all SSM-snapshot saves recorded so far:
    /// make it wait on the snapshot-ordering event before reading the snapshot
    /// region. GPU-side, zero CPU cost. No-op if no save has been recorded yet
    /// (the event is empty → wait returns immediately).
    pub(super) fn wait_snapshot_saves_dispatch(&self, restore_stream: u64) -> Result<()> {
        self.gpu
            .stream_wait_event(restore_stream, self.snapshot_event)
    }

    pub(super) fn pre_verify_copy_async_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        use crate::layer::SsmLayerState;

        let stream = self.gpu.default_stream();

        // Bulk path: 2 launches for all SSM layers instead of 2
        // cuMemcpyDtoDAsync per layer. On the 35B-A3B (30 SSM layers) this is
        // 60 driver calls per step at ~8.78 us each, on the DEFAULT stream at
        // the head of every verify, so it serializes directly against the
        // verify kernels rather than hiding behind them.
        let h_bytes = self.ssm_pool.h_bytes;
        let conv_bytes = self.ssm_pool.conv_bytes;
        if self.bulk_copy_available(h_bytes)
            && self.bulk_copy_available(conv_bytes)
            && self.ssm_state_is_pool_backed(seq)
        {
            let slot = seq.slot_idx;
            // canonical → scratch, same as the loop below.
            self.bulk_state_copy(
                self.ssm_pool.h_checkpoint_bases_dev,
                self.ssm_pool.h_state_bases_dev,
                slot * h_bytes,
                slot * h_bytes,
                h_bytes,
                stream,
            )?;
            self.bulk_state_copy(
                self.ssm_pool.conv_checkpoint_bases_dev,
                self.ssm_pool.conv_state_bases_dev,
                slot * conv_bytes,
                slot * conv_bytes,
                conv_bytes,
                stream,
            )?;
            return Ok(());
        }

        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                // No-op if checkpoint isn't populated (non-MTP path).
                let Some(h_ckpt) = ssm.h_state_checkpoint else {
                    continue;
                };
                let Some(conv_ckpt) = ssm.conv_state_checkpoint else {
                    continue;
                };

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                let h_bytes = nv * vd * kd * 4;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                // canonical → scratch (live → kernel input/output).
                self.gpu
                    .copy_d2d_async(h_ckpt, ssm.h_state, h_bytes, stream)?;
                self.gpu
                    .copy_d2d_async(conv_ckpt, ssm.conv_state, conv_bytes, stream)?;
            }
        }
        Ok(())
    }

    /// STree-style in-place verify commit (item #2): the verify kernel
    /// writes directly onto the canonical `h_state`/`conv_state`, so the
    /// surviving prefix is already live and "commit" reduces to a single
    /// index-select on a partial accept (and nothing on a full accept).
    ///
    /// - `num_accepted == k` (full accept): the kernel's final `h_state`
    ///   is the committed state → no-op.
    /// - `0 < num_accepted < k` (partial accept): copy
    ///   `h_state_intermediates[num_accepted - 1]` (state after the last
    ///   accepted token) → `h_state` (+ conv intermediate).
    ///
    /// All verify paths (K=2, K=3, K=4, DFlash) run the kernel directly
    /// on the canonical `h_state` (no `pre_verify_copy_async` scratch-seed),
    /// so on a full accept the live state is already committed and on a
    /// partial accept the single index-select below leaves `h_state`
    /// canonical for every successor (bootstrap decode, gate-flip decode,
    /// concurrent request). No `*_checkpoint` write is needed — the next
    /// `start_checkpoint_async` syncs h_state → checkpoint at prefill time.
    ///
    /// Runs on `secondary_stream`; pair with `sync_secondary`.
    pub(super) fn commit_accepted_prefix_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        use crate::layer::SsmLayerState;

        // Width invariant. Together with the `num_accepted == 0` guard below
        // this pins the reachable intermediate index to exactly [0, k-2],
        // which is the invariant the fused K=2/3/4 verify paths rely on when
        // they skip writing `conv_state_intermediates[k-1]`
        // (qwen3_ssm/trait_decode_batched_conv_gdn.rs). Enforcing it here
        // turns that from a global argument about callers into a locally
        // checked precondition: if a caller ever passes a width that would
        // reach index k-1, it errors here instead of silently reading a slot
        // the kernel no longer writes.
        //
        // `num_accepted > k` is nonsense (more tokens committed than
        // verified) and is the shape a bonus-token off-by-one would take —
        // e.g. DFlash passing `gamma` instead of `gamma + 1` as `k` while
        // passing `num_accepted + 1`. Today `verify_dflash_step.rs` passes
        // `k_verify = drafts.len() + 1` against `total_accepted =
        // num_accepted + 1`, so the two agree; this catches the day they
        // stop agreeing.
        if num_accepted > k {
            anyhow::bail!(
                "commit_accepted_prefix: num_accepted ({num_accepted}) > k ({k}) — more \
                 tokens committed than were verified. Check that the caller's `k` is the \
                 VERIFY WIDTH (drafts + 1), not the draft count."
            );
        }

        // Full accept: the verify kernel's final h_state/conv_state is
        // already the canonical committed state — nothing to do.
        if num_accepted == k {
            return Ok(());
        }

        // `num_accepted == 0` has no representable rewind target here: the
        // per-token intermediates are indexed `num_accepted - 1`, and this is
        // `usize` arithmetic in a release build (overflow-checks off), so a 0
        // would wrap to `usize::MAX` and hand `h_intermediate()` an
        // out-of-range index — a wild device pointer straight into
        // `copy_d2d_async`. Every scheduler caller passes >= 1 today (position
        // 0 of a verify batch is accepted by construction; DFlash adds the
        // bonus token via `num_accepted + 1`), but that is a caller
        // convention, not an invariant this function can see. Fail fast so a
        // future caller change surfaces as an error instead of silent memory
        // corruption. A genuine "nothing accepted" rewind belongs in
        // `rollback_ssm_states`, which restores the pre-verify checkpoint.
        if num_accepted == 0 {
            anyhow::bail!(
                "commit_accepted_prefix: num_accepted == 0 (k={k}) has no intermediate to \
                 rewind to — position 0 of a verify batch is accepted by construction. \
                 Use rollback_ssm_states() for a full-reject rewind to the pre-verify \
                 checkpoint."
            );
        }

        let stream = self.secondary_stream;
        let mut ssm_layer_idx = 0usize;
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = layer_state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

            let nv = self.config.linear_num_value_heads;
            let vd = self.config.linear_value_head_dim;
            let nk = self.config.linear_num_key_heads;
            let kd = self.config.linear_key_head_dim;
            let h_bytes = nv * vd * kd * 4;
            let conv_bytes = (nk * kd * 2 + nv * vd) * self.config.linear_conv_kernel_dim * 4;

            // Partial accept: rewind live state to the last accepted token's
            // intermediate (state after token `num_accepted-1`).
            let slot = seq.slot_idx;
            let inter_idx = num_accepted - 1;
            let h_inter = self.ssm_pool.h_intermediate(ssm_layer_idx, slot, inter_idx);
            let conv_inter = self
                .ssm_pool
                .conv_intermediate(ssm_layer_idx, slot, inter_idx);
            self.gpu
                .copy_d2d_async(h_inter, ssm.h_state, h_bytes, stream)?;
            self.gpu
                .copy_d2d_async(conv_inter, ssm.conv_state, conv_bytes, stream)?;

            ssm_layer_idx += 1;
        }
        self.gpu.record_event(self.secondary_event, stream)?;
        Ok(())
    }

    pub(super) fn commit_verify_state_async_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        use crate::layer::SsmLayerState;

        // Bulk path: 2 launches (full accept / full reject) or 4 (partial)
        // instead of 2-4 cuMemcpyDtoDAsync PER LAYER. This is the copy burst
        // that owns the ~2.35 ms post-argmax stall measured every step in
        // nsys-moe-cap4-0931 — 43.9 h_state + 43.9 conv copies per step on the
        // secondary stream, 93 driver calls, of which only ~0.77 ms is the GPU
        // actually moving bytes.
        //
        // Every branch below mirrors the loop's source/destination exactly,
        // including the partial-accept case writing the intermediate to BOTH
        // the checkpoint and the live buffer (the live-state invariant noted
        // under the loop). Same bytes, same order of families, same stream.
        {
            let pool = &self.ssm_pool;
            let h_bytes = pool.h_bytes;
            let conv_bytes = pool.conv_bytes;
            let slot = seq.slot_idx;
            let ni = pool.num_intermediates;
            let stream = self.secondary_stream;

            let bulk_ok = self.bulk_copy_available(h_bytes) && self.bulk_copy_available(conv_bytes);
            let partial = num_accepted != 0 && num_accepted != k;

            let usable = bulk_ok
                && if partial {
                    self.ssm_intermediates_are_pool_backed(seq)
                } else {
                    self.ssm_state_is_pool_backed(seq)
                };

            if usable {
                if num_accepted == 0 {
                    // Full reject: checkpoint → live.
                    self.bulk_state_copy(
                        pool.h_checkpoint_bases_dev,
                        pool.h_state_bases_dev,
                        slot * h_bytes,
                        slot * h_bytes,
                        h_bytes,
                        stream,
                    )?;
                    self.bulk_state_copy(
                        pool.conv_checkpoint_bases_dev,
                        pool.conv_state_bases_dev,
                        slot * conv_bytes,
                        slot * conv_bytes,
                        conv_bytes,
                        stream,
                    )?;
                } else if num_accepted == k {
                    // Full accept: live → checkpoint (commit verify result).
                    self.bulk_state_copy(
                        pool.h_state_bases_dev,
                        pool.h_checkpoint_bases_dev,
                        slot * h_bytes,
                        slot * h_bytes,
                        h_bytes,
                        stream,
                    )?;
                    self.bulk_state_copy(
                        pool.conv_state_bases_dev,
                        pool.conv_checkpoint_bases_dev,
                        slot * conv_bytes,
                        slot * conv_bytes,
                        conv_bytes,
                        stream,
                    )?;
                } else {
                    // Partial accept: intermediate[num_accepted-1] → checkpoint
                    // AND → live.
                    let inter_idx = num_accepted - 1;
                    let h_inter_off = (slot * ni + inter_idx) * h_bytes;
                    let conv_inter_off = (slot * ni + inter_idx) * conv_bytes;
                    self.bulk_state_copy(
                        pool.h_intermediate_bases_dev,
                        pool.h_checkpoint_bases_dev,
                        h_inter_off,
                        slot * h_bytes,
                        h_bytes,
                        stream,
                    )?;
                    self.bulk_state_copy(
                        pool.conv_intermediate_bases_dev,
                        pool.conv_checkpoint_bases_dev,
                        conv_inter_off,
                        slot * conv_bytes,
                        conv_bytes,
                        stream,
                    )?;
                    self.bulk_state_copy(
                        pool.h_intermediate_bases_dev,
                        pool.h_state_bases_dev,
                        h_inter_off,
                        slot * h_bytes,
                        h_bytes,
                        stream,
                    )?;
                    self.bulk_state_copy(
                        pool.conv_intermediate_bases_dev,
                        pool.conv_state_bases_dev,
                        conv_inter_off,
                        slot * conv_bytes,
                        conv_bytes,
                        stream,
                    )?;
                }
                self.gpu.record_event(self.secondary_event, stream)?;
                return Ok(());
            }
        }

        // Live-state invariant (2026-06-10 MTP×warm stutter fix): the live
        // h_state/conv_state MUST be canonical after every commit, not just
        // the checkpoint. Leaving live dirty (holding the rejected draft)
        // was safe only when the next op was guaranteed to be another verify
        // (pre_verify copies checkpoint→live). Three real paths run a plain
        // decode() on the live buffer with no restore — spontaneous <think>
        // flipping the scheduler MTP gate, a second concurrent request, and
        // the MTP bootstrap after empty drafts (which then BAKES the dirty
        // live state into the checkpoint via start_checkpoint_async). The
        // phantom rejected token in the GDN memory garbles subsequent
        // decode (token-stutter), and with prefix caching the poisoned
        // decode-KV is immortalized in shared blocks across agentic turns.
        // Cost: one extra D2D pair per SSM layer per reject — same as the
        // pre-verify copy.
        if num_accepted == 0 {
            // Full reject: canonical state untouched; restore live from the
            // checkpoint so any non-verify successor reads canonical state.
            let stream = self.secondary_stream;
            for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
                if self.config.layer_type(i) == LayerType::LinearAttention {
                    let ssm = layer_state
                        .as_any_mut()
                        .downcast_mut::<SsmLayerState>()
                        .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
                    let (Some(h_ckpt), Some(conv_ckpt)) =
                        (ssm.h_state_checkpoint, ssm.conv_state_checkpoint)
                    else {
                        continue;
                    };
                    let nv = self.config.linear_num_value_heads;
                    let vd = self.config.linear_value_head_dim;
                    let nk = self.config.linear_num_key_heads;
                    let kd = self.config.linear_key_head_dim;
                    let h_bytes = nv * vd * kd * 4;
                    let conv_bytes =
                        (nk * kd * 2 + nv * vd) * self.config.linear_conv_kernel_dim * 4;
                    self.gpu
                        .copy_d2d_async(h_ckpt, ssm.h_state, h_bytes, stream)?;
                    self.gpu
                        .copy_d2d_async(conv_ckpt, ssm.conv_state, conv_bytes, stream)?;
                }
            }
            self.gpu
                .record_event(self.secondary_event, self.secondary_stream)?;
            // Ordering: the verify path syncs at entry (verify_*_step
            // sync_secondary); the non-verify successors (gate flip,
            // bootstrap) sync at THEIR entry — see scheduler/mod.rs and
            // mtp_step.rs. No wait here: a commit-side wait would serialize
            // this copy against the next draft and cost ~25% decode wall.
            return Ok(());
        }

        let stream = self.secondary_stream;
        let mut ssm_layer_idx = 0usize;

        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let Some(h_ckpt) = ssm.h_state_checkpoint else {
                    ssm_layer_idx += 1;
                    continue;
                };
                let Some(conv_ckpt) = ssm.conv_state_checkpoint else {
                    ssm_layer_idx += 1;
                    continue;
                };

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                let h_bytes = nv * vd * kd * 4;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                if num_accepted == k {
                    // Full accept: scratch → live (commit verify result).
                    self.gpu
                        .copy_d2d_async(ssm.h_state, h_ckpt, h_bytes, stream)?;
                    self.gpu
                        .copy_d2d_async(ssm.conv_state, conv_ckpt, conv_bytes, stream)?;
                } else {
                    // Partial accept: intermediate[num_accepted-1] → checkpoint
                    // AND → live. The live buffer holds state through ALL k
                    // verify tokens (including the rejected draft); restoring
                    // it here keeps live canonical for any non-verify
                    // successor (see the live-state invariant note above).
                    let slot = seq.slot_idx;
                    let inter_idx = num_accepted - 1;
                    let h_inter = self.ssm_pool.h_intermediate(ssm_layer_idx, slot, inter_idx);
                    let conv_inter =
                        self.ssm_pool
                            .conv_intermediate(ssm_layer_idx, slot, inter_idx);
                    self.gpu.copy_d2d_async(h_inter, h_ckpt, h_bytes, stream)?;
                    self.gpu
                        .copy_d2d_async(conv_inter, conv_ckpt, conv_bytes, stream)?;
                    self.gpu
                        .copy_d2d_async(h_inter, ssm.h_state, h_bytes, stream)?;
                    self.gpu
                        .copy_d2d_async(conv_inter, ssm.conv_state, conv_bytes, stream)?;
                }

                ssm_layer_idx += 1;
            }
        }

        self.gpu.record_event(self.secondary_event, stream)?;
        // Ordering: verify_*_step calls sync_secondary at entry; the
        // non-verify successors that read the live state (MTP gate flip →
        // step_decode_only, bootstrap decode) call sync_secondary at THEIR
        // entry (scheduler/mod.rs, mtp_step.rs). A commit-side wait here
        // would serialize this 250MB copy against the next draft kernels
        // that used to overlap it (~25% decode wall, tq11 360s cap-riders).
        Ok(())
    }
}
