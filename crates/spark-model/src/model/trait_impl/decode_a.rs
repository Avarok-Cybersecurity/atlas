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
    pub(super) fn decode_dispatch(
        &self,
        token: u32,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<DevicePtr> {
        // Use backend's own stream (non-default, required for CUDA graph capture).
        let stream = self.gpu.default_stream();
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: Operations OUTSIDE graph (vary per token) ──

        // MLA models: zero buffers reused for Q_absorbed computation.
        // Without this, stale prefill data in expert_up_out / ssm_conv_out_f32 /
        // ssm_ba contaminates the absorbed attention → generic/wrong output.
        if self.config.kv_lora_rank > 0 {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        }

        // 1. Embedding lookup
        self.embed(token, hidden, stream)?;

        // 2. Pre-allocate KV cache blocks + upload attention metadata
        let bs = kv_cache.block_size();
        let blocks_needed = (seq.seq_len / bs) + 1;
        ensure_blocks_through_decode(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
        )?;

        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = seq.block_table.len() as u32;

        let pos_val = seq.seq_len as u32;
        self.gpu
            .copy_h2d_async(&pos_val.to_le_bytes(), meta_base, stream)?;

        let block_idx = seq
            .physical_block_for(seq.seq_len / bs)
            .unwrap_or(self.dummy_kv_block);
        let global_slot = (block_idx as i64) * (bs as i64) + ((seq.seq_len % bs) as i64);
        self.gpu
            .copy_h2d_async(&global_slot.to_le_bytes(), meta_base.offset(8), stream)?;

        let actual_seq_len = (seq.seq_len + 1) as i32;
        self.gpu
            .copy_h2d_async(&actual_seq_len.to_le_bytes(), meta_base.offset(16), stream)?;

        let bt_i32: Vec<i32> = seq.block_table.iter().map(|&b| b as i32).collect();
        let bt_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(bt_i32.as_ptr() as *const u8, bt_i32.len() * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(256), stream)?;

        let attn_metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: max_blocks,
            num_seqs: 1,
        };

        // CUDA graphs cannot capture NCCL all-reduce (it runs on a separate
        // stream) or cuStreamSynchronize calls. Suppress for EP and profile.
        // Re-enable graphs once FP8 calibration is frozen.
        if self.config.fp8_kv_calibration_tokens > 0
            && self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && seq.seq_len > self.config.fp8_kv_calibration_tokens + 10
        {
            self.suppress_graphs
                .store(false, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("FP8 calibration frozen — re-enabling CUDA graphs");
        }
        // Phase 6.2.c — `--high-speed-swap` paths do host-side D2H + dequant
        // + per-step disk I/O which is illegal under CUDA graph capture
        // (cuStreamSynchronize fails with status 900 = CAPTURE_UNSUPPORTED).
        // Capture isn't a useful win for HSS anyway: per-layer launch overhead
        // is small relative to the per-step disk I/O on the critical path.
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        let use_graphs = self.comm.is_none()
            && !self.profile
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(attn_metadata),
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
        };

        // Profile mode: use per-layer sync decode for timing breakdown.
        if self.profile {
            return self.decode_profiled(token, hidden, residual, seq, &mut kv_cache, &ctx, stream);
        }

        // ── Phase 2: Try CUDA graph replay ──

        let mut graph_cache = if use_graphs {
            Some(self.decode_graph.lock())
        } else {
            None
        };

        // For batch=1, the captured graph works for any max_blocks because
        // max_blocks_per_seq is only used as block_table stride (seq_idx * stride),
        // and seq_idx=0 makes the stride irrelevant. All dynamic data (seq_len,
        // block_table, positions, slots) is read from device memory uploaded
        // before each graph replay.
        // SLOT-KEYED LOOKUP: only replay if this seq's slot matches a captured graph.
        if let Some(ref cache) = graph_cache
            && let Some(graph) = cache.get(&seq.slot_idx)
            && graph.0 != 0
        {
            self.gpu.launch_graph(*graph, stream)?;
            seq.tokens.push(token);
            seq.seq_len += 1;
            return Ok(self.decode_logits_ptr());
        }

        // ── Phase 3: Capture new CUDA graph (or run eagerly for EP) ──

        if use_graphs {
            tracing::info!(
                "CUDA graph capture: starting for {} layers",
                self.layers.len()
            );
            self.gpu.begin_capture(stream)?;
        }

        for (i, layer) in self.layers.iter().enumerate() {
            layer.decode(
                hidden,
                residual,
                seq.layer_states[i].as_mut(),
                &mut kv_cache,
                seq.seq_len,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                &ctx,
                stream,
            )?;
            // DFlash 5-layer hidden capture (no-op when proposer is not DFlash).
            // Single-token decode: row 0 of `hidden_states()` holds the post-layer
            // activation. Cheap d2d when the layer index matches; otherwise a
            // hashmap-free position() probe over a 5-element vec.
            self.try_dflash_capture(i, 0, stream)?;
        }
        // MLA absorbed attention: defensive sync before final norm in eager
        // mode. Skipped under graph capture because cuStreamSynchronize is
        // illegal inside a capture region (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
        // status 900). The sync is redundant when all kernels run on the same
        // stream — they are already sequenced — so the removal is safe for
        // both eager (retains sync as paranoia) and graph mode.
        if self.config.kv_lora_rank > 0 && !use_graphs {
            self.gpu.synchronize(stream)?;
        }

        // Periodic SSM state normalization during decode.
        // Mamba-2 has no per-token gate clamping (unlike GDN), so state can drift
        // from accumulated BF16 input truncation. Normalize every 64 tokens.
        if self.config.mamba_num_heads > 0
            && seq.seq_len > 0
            && seq.seq_len.is_multiple_of(64)
            && let Err(e) = self.normalize_ssm_states(seq, stream)
        {
            tracing::warn!("Periodic SSM state normalization failed: {e:#}");
        }

        let normed = self.buffers.norm_output();
        let h = self.config.hidden_size as u32;
        let eps = self.config.rms_norm_eps as f32;
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            hidden,
            &self.final_norm,
            normed,
            1,
            h,
            eps,
            stream,
        )?;

        // LM head reads from normed directly (no D2D copy needed)
        self.lm_head(normed, stream)?;

        // Decode-step diagnostic for Gemma-4 degeneration analysis. Only fires
        // when ATLAS_DIAG_GEMMA4=1 (which also disables CUDA graphs upstream,
        // so the d2h sync below is safe). Reads top-5 tokens by logit so we
        // can see whether the LM head produced a near-tie or a confident bad
        // pick. (B4 — Creative haiku degeneration loop diagnostic.)
        if std::env::var("ATLAS_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true") {
            self.gpu.synchronize(stream)?;
            let n_logits = self.config.vocab_size;
            // Read the buffer the LM head actually wrote to. With Gemma-4
            // dense the single-token decode lm_head produces FP32 in
            // `logits_fp32_buf`; the BF16 buffer would be all zeros there.
            let logit_vals: Vec<f32> = if self.use_fp32_logits {
                let mut buf = vec![0u8; n_logits * 4];
                if let Err(e) = self.gpu.copy_d2h(self.logits_fp32_buf, &mut buf) {
                    tracing::error!("ATLAS_DIAG_GEMMA4: copy_d2h(logits_fp32_buf): {e:#}");
                }
                buf.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            } else {
                let mut buf = vec![0u8; n_logits * 2];
                if let Err(e) = self.gpu.copy_d2h(self.buffers.logits(), &mut buf) {
                    tracing::error!("ATLAS_DIAG_GEMMA4: copy_d2h(logits BF16): {e:#}");
                }
                buf.chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect()
            };
            let max = logit_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = logit_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let mut idx: Vec<usize> = (0..logit_vals.len()).collect();
            idx.sort_by(|&a, &b| {
                logit_vals[b]
                    .partial_cmp(&logit_vals[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top5: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, logit_vals[i])).collect();
            tracing::warn!(
                "DIAG decode logits: max={max:.4} min={min:.4} prev_token={token} top5={top5:?}",
            );
        }

        if use_graphs {
            let graph = self.gpu.end_capture(stream)?;
            if graph.0 != 0 {
                tracing::info!(
                    "CUDA graph captured successfully for slot={} (handle={:?})",
                    seq.slot_idx,
                    graph.0
                );
                if let Some(ref mut cache) = graph_cache {
                    cache.insert(seq.slot_idx, graph);
                }
                self.gpu.launch_graph(graph, stream)?;
            } else {
                tracing::warn!("CUDA graph capture returned null handle — running eagerly");
            }
            // If graph.0 == 0 (mock): operations already executed during capture
        }

        seq.tokens.push(token);
        seq.seq_len += 1;

        Ok(self.decode_logits_ptr())
    }
}
