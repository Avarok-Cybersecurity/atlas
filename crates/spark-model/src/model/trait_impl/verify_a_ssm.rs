// SPDX-License-Identifier: AGPL-3.0-only

//! SSM state checkpoint / rollback / snapshot dispatch for the verify path.
//!
//! Split verbatim out of `verify_a.rs`, which crossed the 500-LoC cap when
//! per-token verify metadata landed (#745: 484 -> 638). Same idiom as
//! `vision_encoder/enc_impl/`: each sibling adds methods to the same
//! inherent impl, so nothing about dispatch or visibility changes.
//!
//! The bodies are an EXACT copy. If this file and `verify_a.rs` ever
//! disagree about a method, this one was edited by mistake.

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
    pub(super) fn checkpoint_ssm_states_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        use crate::layer::SsmLayerState;

        let stream = self.gpu.default_stream();
        let mut h_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        let mut conv_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == atlas_core::config::LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                // Determine sizes from config
                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                // STORAGE width of pool h regions (SSOT: ssm_pool /
                // ssm_reserve::ssm_h_stored_bytes) — FP32 today; halves
                // under the stage-3 f16-sized pool so these copies can
                // never overrun a narrow slot.
                let h_bytes = self.ssm_pool.h_stored_bytes;
                let conv_dim = nk * kd * 2 + nv * vd; // 8192
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4; // FP32

                // Lazy alloc checkpoint buffers
                if ssm.h_state_checkpoint.is_none() {
                    ssm.h_state_checkpoint = Some(self.gpu.alloc(h_bytes)?);
                }
                if ssm.conv_state_checkpoint.is_none() {
                    ssm.conv_state_checkpoint = Some(self.gpu.alloc(conv_bytes)?);
                }

                // D2D copy: state → checkpoint
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
        self.gpu.synchronize(stream)?;
        Ok(())
    }

    pub(super) fn rollback_ssm_states_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        use crate::layer::SsmLayerState;

        // PRE-VALIDATION PASS — no GPU work is enqueued until every SSM layer
        // is known to be restorable. Bailing part-way through the copy loop
        // below would leave the first N layers rewound and the rest advanced
        // past the accepted boundary: a MIXED state, which is strictly worse
        // than the uniform corruption it is meant to prevent and much harder
        // to reason about. Validate first, then copy unconditionally.
        if num_accepted > 0 {
            for (i, layer_state) in seq.layer_states.iter().enumerate() {
                if self.config.layer_type(i) != atlas_core::config::LayerType::LinearAttention {
                    continue;
                }
                let ssm = layer_state
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
                if num_accepted > ssm.h_state_intermediates.len() {
                    anyhow::bail!(
                        "rollback_ssm_states: cannot restore SSM to N={num_accepted} \
                         (layer {i}): only {} per-token intermediate(s) available. \
                         With no intermediates this is the self-speculative / ngram \
                         path — use --speculative (MTP) or --num-drafts 1 for SSM \
                         models. With too few, the MTP h-intermediate pool \
                         (num_drafts per slot, tiered — K-1 snapshots for a \
                         K-row verify) is smaller than this rollback target. \
                         No rollback copies were enqueued.",
                        ssm.h_state_intermediates.len(),
                    );
                }
            }
        }

        let stream = self.gpu.default_stream();
        let mut h_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        let mut conv_plan = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == atlas_core::config::LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let kd = self.config.linear_key_head_dim;
                let nk = self.config.linear_num_key_heads;
                // Pool h STORAGE width (SSOT: ssm_reserve::ssm_h_stored_bytes).
                let h_bytes = self.ssm_pool.h_stored_bytes;
                let conv_dim = nk * kd * 2 + nv * vd; // 8192
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                if num_accepted == 0 {
                    // Restore to pre-verification checkpoint
                    if let Some(ckpt) = ssm.h_state_checkpoint {
                        h_plan.push(StateCopy {
                            src: ckpt,
                            dst: ssm.h_state,
                            bytes: h_bytes,
                        });
                    }
                    if let Some(ckpt) = ssm.conv_state_checkpoint {
                        conv_plan.push(StateCopy {
                            src: ckpt,
                            dst: ssm.conv_state,
                            bytes: conv_bytes,
                        });
                    }
                } else if num_accepted <= ssm.h_state_intermediates.len() {
                    // Restore to intermediate checkpoint after the last accepted token
                    let idx = num_accepted - 1;
                    h_plan.push(StateCopy {
                        src: ssm.h_state_intermediates[idx],
                        dst: ssm.h_state,
                        bytes: h_bytes,
                    });
                    conv_plan.push(StateCopy {
                        src: ssm.conv_state_intermediates[idx],
                        dst: ssm.conv_state,
                        bytes: conv_bytes,
                    });
                } else {
                    // Unreachable: the pre-validation pass above already
                    // bailed for every `num_accepted > intermediates.len()`,
                    // and `num_accepted == 0` took the first branch. Kept as
                    // a hard error rather than a silent fallthrough — the
                    // original code returned Ok(()) here, leaving h_state and
                    // conv_state ADVANCED past the last accepted token with
                    // no error and no log line, which corrupts every
                    // subsequent decode and surfaces much later as gibberish.
                    unreachable!(
                        "rollback_ssm_states: layer {i} passed pre-validation but \
                         num_accepted={num_accepted} exceeds {} intermediates",
                        ssm.h_state_intermediates.len(),
                    );
                }
                // `num_accepted == num_tokens` (full accept) never reaches
                // here: callers guard it (`seq.seq_len > expected_seq_len`),
                // and it would otherwise be swallowed by the branch above.
            }
        }
        // Enqueued only after every layer validated AND planned — the
        // pre-validation pass above already guarantees no bail can happen
        // here, and building the plan first makes that structural rather
        // than argued: a partially-rewound MIXED state is unrepresentable.
        run_ssm_state_copies(self.gpu.as_ref(), &h_plan, &conv_plan, stream)?;
        // No synchronize needed: rollback copies and subsequent operations
        // are on the same CUDA stream, so ordering is guaranteed.
        Ok(())
    }

    /// Phase-C decode-time boundary snapshot save.
    ///
    /// Copies the sequence's live SSM state (the active `SsmStatePool`
    /// slot `seq.slot_idx`) into the decode-rollback ring slot
    /// `(seq.slot_idx, ring_slot)` of [`SsmSnapshotPool`]. Reuses the
    /// same D2D copy primitive Marconi and MTP verify use (SSOT). The
    /// copies run on the default stream so they are ordered after the
    /// decode that produced this boundary token and before any later
    /// decode that would overwrite the pool slot.
    pub(super) fn save_decode_ssm_snapshot_dispatch(
        &self,
        seq: &SequenceState,
        ring_slot: usize,
    ) -> Result<()> {
        if !self.ssm_snapshots.decode_rollback_enabled() {
            anyhow::bail!("save_decode_ssm_snapshot: decode-rollback region not allocated");
        }
        let stream = self.gpu.default_stream();
        self.ssm_snapshots.save_decode(
            seq.slot_idx,
            ring_slot,
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        )
    }

    /// Phase-C decode-time boundary snapshot restore.
    ///
    /// Inverse of [`Self::save_decode_ssm_snapshot_dispatch`]: copies the
    /// ring snapshot `(seq.slot_idx, ring_slot)` back into the live
    /// `SsmStatePool` slot, undoing every recurrent update the dropped
    /// degenerate tail applied.
    pub(super) fn restore_decode_ssm_snapshot_dispatch(
        &self,
        seq: &SequenceState,
        ring_slot: usize,
    ) -> Result<()> {
        if !self.ssm_snapshots.decode_rollback_enabled() {
            anyhow::bail!("restore_decode_ssm_snapshot: decode-rollback region not allocated");
        }
        let stream = self.gpu.default_stream();
        self.ssm_snapshots.restore_decode(
            seq.slot_idx,
            ring_slot,
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        )
    }
}
