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
use super::super::ssm_batched_copy::{StateCopy, run_ssm_state_copies};
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
        let mut h_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        let mut conv_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
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
                // Pool h STORAGE width (SSOT: ssm_reserve::ssm_h_stored_bytes).
                let h_bytes = self.ssm_pool.h_stored_bytes;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                if ssm.h_state_checkpoint.is_none() {
                    ssm.h_state_checkpoint = Some(self.gpu.alloc(h_bytes)?);
                }
                if ssm.conv_state_checkpoint.is_none() {
                    ssm.conv_state_checkpoint = Some(self.gpu.alloc(conv_bytes)?);
                }

                h_plan.push(StateCopy {
                    src: ssm.h_state,
                    dst: ssm.h_state_checkpoint.unwrap(),
                    bytes: h_bytes,
                });
                conv_plan.push(StateCopy {
                    src: ssm.conv_state,
                    dst: ssm.conv_state_checkpoint.unwrap(),
                    bytes: conv_bytes,
                });
            }
        }
        run_ssm_state_copies(self.gpu.as_ref(), &h_plan, &conv_plan, stream)?;
        // Record event so default stream can wait (GPU-side, no CPU block).
        self.gpu.record_event(self.secondary_event, stream)?;
        Ok(())
    }

    /// Advance a replay-mode rollback over the accepted rows.
    ///
    /// The copies above put every layer back to its PRE-VERIFY checkpoint —
    /// correct as-is for a full reject, and the reconstruction base for a
    /// partial accept. This walks the recurrent layers and re-runs each one's
    /// `num_accepted` cached rows forward from that base.
    ///
    /// Runs on the DEFAULT stream, not the secondary rollback stream: the
    /// state copies are pure d2d and safe to overlap, but this is KERNEL work
    /// reading the shared decode scratch, so overlapping it with the next
    /// forward would race on those buffers. Replay is the capacity mode, not
    /// the fast one.
    /// Minimal [`ForwardContext`] for a replay-mode rollback.
    ///
    /// See the module note: `comm` MUST stay `None` (this pass runs the
    /// recurrence only and must issue no collective, or the EP ranks
    /// desynchronise), and `gdn_exact_replay` stays `false` so the arm is
    /// chosen by the replay gate rather than by the pass-scoped exact leg.
    fn replay_forward_ctx(&self) -> crate::layer::ForwardContext<'_> {
        crate::layer::ForwardContext {
            decode_step: false,
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            moe_lora_route: self.decode_moe_route(),
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
        }
    }

    /// Partial-accept rollback under replay: checkpoint, then replay.
    ///
    /// Mirrors the snapshot path's contract — on return the live state is the
    /// state after the committed prefix, and the checkpoint holds that same
    /// state ready for the next verify.
    fn replay_commit_accepted_prefix(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        let stream = self.gpu.default_stream();
        let mut h_back: Vec<StateCopy> = Vec::new();
        let mut conv_back: Vec<StateCopy> = Vec::new();
        let mut h_ckpt: Vec<StateCopy> = Vec::new();
        let mut conv_ckpt: Vec<StateCopy> = Vec::new();
        let h_bytes = self.ssm_pool.h_stored_bytes;
        let conv_bytes = (self.config.linear_num_key_heads * self.config.linear_key_head_dim * 2
            + self.config.linear_num_value_heads * self.config.linear_value_head_dim)
            * self.config.linear_conv_kernel_dim
            * 4;
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let Some(ssm) = layer_state.as_any_mut().downcast_mut::<SsmLayerState>() else {
                continue;
            };
            if let Some(ckpt) = ssm.h_state_checkpoint {
                h_back.push(StateCopy {
                    src: ckpt,
                    dst: ssm.h_state,
                    bytes: h_bytes,
                });
                h_ckpt.push(StateCopy {
                    src: ssm.h_state,
                    dst: ckpt,
                    bytes: h_bytes,
                });
            }
            if let Some(ckpt) = ssm.conv_state_checkpoint {
                conv_back.push(StateCopy {
                    src: ckpt,
                    dst: ssm.conv_state,
                    bytes: conv_bytes,
                });
                conv_ckpt.push(StateCopy {
                    src: ssm.conv_state,
                    dst: ckpt,
                    bytes: conv_bytes,
                });
            }
        }
        // Restore -> advance -> re-checkpoint, all on the default stream: the
        // advance is kernel work on shared decode scratch and must not overlap
        // the next forward.
        run_ssm_state_copies(self.gpu.as_ref(), &h_back, &conv_back, stream)?;
        self.replay_rollback_rows(seq, num_accepted, stream)?;
        run_ssm_state_copies(self.gpu.as_ref(), &h_ckpt, &conv_ckpt, stream)?;
        self.commit_verify_aux_rows(seq, num_accepted, stream)
    }

    fn replay_rollback_rows(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        if num_accepted == 0 {
            return Ok(());
        }
        let stream = self.gpu.default_stream();
        let ctx = self.replay_forward_ctx();
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            self.layers[i].replay_verify_rows(layer_state.as_mut(), num_accepted, &ctx, stream)?;
        }
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
        // Four plans in the ORIGINAL issue order: rollback h, rollback conv,
        // then checkpoint h, checkpoint conv. The checkpoint reads the state
        // the rollback just wrote, and both land on `stream` in that order,
        // so the read-after-write is ordered exactly as the per-layer loop
        // ordered it. Re-ordering ACROSS layers within one plan is sound:
        // layer L's blobs are disjoint from layer M's.
        let n_ssm = self.ssm_pool.num_ssm_layers;
        let mut h_back = Vec::with_capacity(n_ssm);
        let mut conv_back = Vec::with_capacity(n_ssm);
        let mut h_ckpt = Vec::with_capacity(n_ssm);
        let mut conv_ckpt = Vec::with_capacity(n_ssm);

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
                // Pool h STORAGE width (SSOT: ssm_reserve::ssm_h_stored_bytes).
                let h_bytes = self.ssm_pool.h_stored_bytes;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                // Rollback: restore h_state and conv_state from the appropriate source.
                //
                // Under `--ssm-rollback-mode replay` there are no per-token
                // intermediates, so EVERY accept count restores the pre-verify
                // checkpoint here and a partial accept is then advanced by
                // `replay_verify_rows` below. `num_accepted == 0` is already
                // that same restore, so the two modes agree on the reject path
                // and differ only in how a partial accept gets forward again.
                let replay_rollback = !ssm.replay_inputs.is_empty();
                if num_accepted == 0 || replay_rollback {
                    // No tokens accepted: restore from checkpoint (pre-verify state).
                    if let Some(ckpt) = ssm.h_state_checkpoint {
                        h_back.push(StateCopy {
                            src: ckpt,
                            dst: ssm.h_state,
                            bytes: h_bytes,
                        });
                    }
                    if let Some(ckpt) = ssm.conv_state_checkpoint {
                        conv_back.push(StateCopy {
                            src: ckpt,
                            dst: ssm.conv_state,
                            bytes: conv_bytes,
                        });
                    }
                } else {
                    // Partial acceptance: restore from intermediate[num_accepted - 1].
                    let slot = seq.slot_idx;
                    let inter_idx = num_accepted - 1;
                    h_back.push(StateCopy {
                        src: self.ssm_pool.h_intermediate(ssm_layer_idx, slot, inter_idx),
                        dst: ssm.h_state,
                        bytes: h_bytes,
                    });
                    conv_back.push(StateCopy {
                        src: self
                            .ssm_pool
                            .conv_intermediate(ssm_layer_idx, slot, inter_idx),
                        dst: ssm.conv_state,
                        bytes: conv_bytes,
                    });
                }

                // Checkpoint the (now rolled-back) state for the next verify.
                if let Some(ckpt) = ssm.h_state_checkpoint {
                    h_ckpt.push(StateCopy {
                        src: ssm.h_state,
                        dst: ckpt,
                        bytes: h_bytes,
                    });
                }
                if let Some(ckpt) = ssm.conv_state_checkpoint {
                    conv_ckpt.push(StateCopy {
                        src: ssm.conv_state,
                        dst: ckpt,
                        bytes: conv_bytes,
                    });
                }

                ssm_layer_idx += 1;
            }
        }
        run_ssm_state_copies(self.gpu.as_ref(), &h_back, &conv_back, stream)?;
        // Replay: the checkpoint is only the BASE. Advance it over the
        // accepted rows before re-checkpointing, or the next verify would
        // start from the pre-verify state and silently drop the accepted
        // tokens from the recurrence.
        self.replay_rollback_rows(seq, num_accepted, stream)?;
        run_ssm_state_copies(self.gpu.as_ref(), &h_ckpt, &conv_ckpt, stream)?;
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
}

/// The verify-intermediate slot `commit_accepted_prefix` rewinds the live SSM
/// state to for `num_accepted` committed rows: "state after token
/// `num_accepted - 1`".
///
/// Named, and paired with `verify_hc::hc_publish_rows`, because the two halves
/// of this contract live in different files and a verify path that publishes
/// FEWER rows than this can read rewinds onto never-written pool memory —
/// which is exactly what the mHC verify did before 2026-09-03 (36 GDN layers
/// of live `h_state`/`conv_state` overwritten with garbage on every K=3 step).
/// Callers guarantee `num_accepted >= 1`.
pub(super) const fn commit_rewind_index(num_accepted: usize) -> usize {
    num_accepted - 1
}

#[path = "async_chkpt_commit.rs"]
mod async_chkpt_commit;
