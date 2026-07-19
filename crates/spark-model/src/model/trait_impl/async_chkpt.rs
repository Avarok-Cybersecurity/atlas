// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
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

impl TransformerModel {
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

        // Full accept: the verify kernel's final h_state/conv_state is
        // already the canonical committed state — nothing to do.
        if num_accepted == k {
            return Ok(());
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
