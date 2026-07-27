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
    pub(super) fn decode_verify_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        if k == 1 {
            let logits = self.decode(tokens[0], seq, stream)?;
            let tok = self.argmax_on_device(logits, stream)?;
            return Ok(vec![tok]);
        }

        // GEMM-batched verification: process all K tokens per-layer,
        // using GEMM for weight-heavy projections to amortize bandwidth.
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize;

        let hidden = self.buffers.hidden_states(); // [K, H]
        let residual = self.buffers.residual(); // [K, H]

        let mut kv_cache = self.kv_cache.lock();

        // ── Embed all K tokens into hidden[K, H] ──
        for (t, &token) in tokens.iter().enumerate() {
            let h_t = hidden.offset(t * h * fp32);
            self.embed(token, h_t, stream)?;
        }

        // ── Per-layer processing ──
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_type = self.config.layer_type(i);

            if layer_type == atlas_core::config::LayerType::FullAttention {
                // Attention layers: sequential per-token (need per-token metadata)
                for t in 0..k {
                    let pos = seq.seq_len + t;
                    let bs = kv_cache.block_size();
                    let blocks_needed = (pos / bs) + 1;
                    ensure_blocks_through_decode(
                        seq,
                        blocks_needed - 1,
                        &mut kv_cache,
                        self.prefix_cache.as_ref(),
                        self.gpu.as_ref(),
                        stream,
                    )?;

                    // Upload per-token attention metadata
                    let meta_base = self.buffers.scratch().offset(32768);
                    let max_blocks = seq.block_table.len() as u32;
                    let pos_val = pos as u32;
                    self.gpu
                        .copy_h2d_async(&pos_val.to_le_bytes(), meta_base, stream)?;
                    let block_idx = seq
                        .physical_block_for(pos / bs)
                        .unwrap_or(self.dummy_kv_block);
                    let global_slot = (block_idx as i64) * (bs as i64) + ((pos % bs) as i64);
                    self.gpu.copy_h2d_async(
                        &global_slot.to_le_bytes(),
                        meta_base.offset(8),
                        stream,
                    )?;
                    let actual_seq_len = (pos + 1) as i32;
                    self.gpu.copy_h2d_async(
                        &actual_seq_len.to_le_bytes(),
                        meta_base.offset(16),
                        stream,
                    )?;
                    let bt_i32: Vec<i32> = seq.block_table.iter().map(|&b| b as i32).collect();
                    let bt_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(bt_i32.as_ptr() as *const u8, bt_i32.len() * 4)
                    };
                    self.gpu
                        .copy_h2d_async(bt_bytes, meta_base.offset(256), stream)?;

                    // Request-scoped LoRA routing (eager per-token verify). One
                    // sequence → one adapter for all K tokens; 1-elem buffer at
                    // the free +128 gap (layout slot@+8/seq_len@+16/bt@+256).
                    // Eager path, so no capture concern. `DevicePtr(0)` (no
                    // pool) → installed-pair fallback.
                    let seq_slot = self.upload_seq_slot_uniform(
                        seq.adapter_slot,
                        1,
                        meta_base.offset(128),
                        stream,
                    )?;

                    let attn_metadata = AttnMetadataDev {
                        positions: meta_base,
                        positions_h: meta_base,
                        positions_w: meta_base,
                        slot: meta_base.offset(8),
                        seq_len: meta_base.offset(16),
                        block_table: meta_base.offset(256),
                        max_blocks_per_seq: max_blocks,
                        num_seqs: 1,
                        seq_slot,
                    };

                    let ctx = ForwardContext {
                        buffers: &self.buffers,
                        gpu: self.gpu.as_ref(),
                        config: &self.config,
                        attn_metadata: Some(attn_metadata),
                        profile: false,
                        comm: self.comm_ref(),
                        graph_capture: false,
                        gdn_exact_replay: false,
                        token_ids: None,
                        routed_lora_layers: None, // #30: verify decode; no prefill route.
                        midchunk_capture: None,
                    };

                    let h_t = hidden.offset(t * h * fp32);
                    let r_t = residual.offset(t * h * fp32);
                    layer.decode(
                        h_t,
                        r_t,
                        seq.layer_states[i].as_mut(),
                        &mut kv_cache,
                        pos,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )?;
                }
            } else {
                // SSM layers: GEMM-batched via decode_batched override
                let ctx = ForwardContext {
                    buffers: &self.buffers,
                    gpu: self.gpu.as_ref(),
                    config: &self.config,
                    attn_metadata: None,
                    profile: false,
                    comm: self.comm_ref(),
                    graph_capture: false,
                    gdn_exact_replay: false,
                    token_ids: None,
                    routed_lora_layers: None, // #30: verify decode; no prefill route.
                    midchunk_capture: None,
                };

                layer.decode_batched(
                    hidden,
                    residual,
                    k,
                    seq.layer_states[i].as_mut(),
                    &mut kv_cache,
                    seq.seq_len,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    &ctx,
                    stream,
                )?;
            }
        }

        // ── Final norm for K tokens ──
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            hidden,
            &self.final_norm,
            normed,
            k as u32,
            h as u32,
            eps,
            stream,
        )?;

        // ── LM head for K tokens → logits[K, vocab] ──
        self.lm_head_batched(normed, k as u32, self.buffers.logits(), stream)?;

        // ── Argmax per token ──
        let vocab = self.config.vocab_size;
        let mut results = Vec::with_capacity(k);
        for t in 0..k {
            let logits_t = self.buffers.logits().offset(t * vocab * bf16);
            let out_ptr = self.buffers.scratch().offset(t * 4);
            ops::argmax_bf16(
                self.gpu.as_ref(),
                self.argmax_kernel,
                logits_t,
                out_ptr,
                vocab as u32,
                stream,
            )?;
        }
        // D2H: copy all K argmax results at once
        let mut buf = vec![0u8; k * 4];
        self.gpu.copy_d2h(self.buffers.scratch(), &mut buf)?;
        for t in 0..k {
            let tok =
                u32::from_le_bytes([buf[t * 4], buf[t * 4 + 1], buf[t * 4 + 2], buf[t * 4 + 3]]);
            results.push(tok);
        }

        // Push ALL tokens and advance seq_len by K. See `decode_verify_graphed`
        // for rationale — the prior `seq_len += k-1 / skip tokens[0]` semantic
        // had an off-by-one that misaligned positions across iterations and
        // broke 80B-nvfp4-mtp fib on the argmax-edge final token.
        for &token in tokens {
            seq.tokens.push(token);
        }
        seq.seq_len += k;

        Ok(results)
    }

    pub(super) fn checkpoint_ssm_states_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        use crate::layer::SsmLayerState;

        let stream = self.gpu.default_stream();
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
                let h_bytes = nv * vd * kd * 4; // FP32
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
                         models. With too few, the MTP intermediate pool \
                         (num_drafts + 1) is smaller than the verify width K. \
                         No rollback copies were enqueued.",
                        ssm.h_state_intermediates.len(),
                    );
                }
            }
        }

        let stream = self.gpu.default_stream();
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
                let h_bytes = nv * vd * kd * 4;
                let conv_dim = nk * kd * 2 + nv * vd; // 8192
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                if num_accepted == 0 {
                    // Restore to pre-verification checkpoint
                    if let Some(ckpt) = ssm.h_state_checkpoint {
                        self.gpu
                            .copy_d2d_async(ckpt, ssm.h_state, h_bytes, stream)?;
                    }
                    if let Some(ckpt) = ssm.conv_state_checkpoint {
                        self.gpu
                            .copy_d2d_async(ckpt, ssm.conv_state, conv_bytes, stream)?;
                    }
                } else if num_accepted <= ssm.h_state_intermediates.len() {
                    // Restore to intermediate checkpoint after the last accepted token
                    let idx = num_accepted - 1;
                    self.gpu.copy_d2d_async(
                        ssm.h_state_intermediates[idx],
                        ssm.h_state,
                        h_bytes,
                        stream,
                    )?;
                    self.gpu.copy_d2d_async(
                        ssm.conv_state_intermediates[idx],
                        ssm.conv_state,
                        conv_bytes,
                        stream,
                    )?;
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
