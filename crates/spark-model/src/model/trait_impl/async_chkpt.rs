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
mod commit;

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
            && bytes.is_multiple_of(16)
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

}
