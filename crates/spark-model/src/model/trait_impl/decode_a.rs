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

        // CBD probe: at the FIRST decode step (seq_len still == prompt_len,
        // before this token is appended) checksum every reusable scratch
        // buffer + per-slot SSM state BEFORE any compute. On the prefix-cache
        // skip path a buffer that the cold full-prefill writes but the skip
        // path bypasses will show (a) a different fingerprint cold-vs-ON or
        // (b) a different fingerprint between two ON runs (leftover from the
        // prior pool occupant) — that is the stale-scratch culprit.
        if seq.seq_len == seq.prompt_len && std::env::var("ATLAS_SSM_SAVE_DUMP").is_ok() {
            self.buffers
                .debug_buffer_checksum(self.gpu.as_ref(), stream, "decode_step0_pre");
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                "decode_step0_pre",
            );
            // Per-block KV fingerprint over the WHOLE block table for the
            // first attention layer (idx 0 = L3) — the layer where the
            // per-layer hidden first diverges. Compares on1-vs-on2 to pin the
            // physical block whose K/V the skip path left stale.
            kv_cache.debug_kv_per_block(
                0,
                &seq.block_table,
                self.gpu.as_ref(),
                stream,
                "decode_step0_pre",
            );
        }

        // ── Phase 1: Operations OUTSIDE graph (vary per token) ──

        // MLA models: zero buffers reused for Q_absorbed computation.
        // Without this, stale prefill data in expert_up_out / ssm_conv_out_f32 /
        // ssm_ba contaminates the ABSORBED attention → generic/wrong output.
        // DeepSeek-V4-Flash (o_lora_rank > 0) uses the DIRECT V=K attention path
        // (not absorbed) and writes-before-reads those scratch buffers, so the
        // full-arena zero (~1.7GB memset/step, sized for max prefill tokens) is
        // unnecessary — skip it for V4 to reclaim that decode-step memset
        // bandwidth. (Other MLA models keep the zero.)
        if self.config.kv_lora_rank > 0 && self.config.o_lora_rank == 0 {
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

        // Upload the decode token ID into the STABLE token_ids buffer (uploaded
        // every step, BEFORE any CUDA-graph replay, so DeepSeek-V4 hash-MoE
        // layers read the correct `tid2eid[token_id]`). Single token at offset 0.
        self.gpu
            .copy_h2d_async(&token.to_le_bytes(), self.buffers.token_ids(), stream)?;

        // ── M2 request-scoped LoRA routing (single-seq decode). Upload this
        // request's 1-elem adapter-slot buffer to the free +128 gap (positions
        // @+0..+4, slot @+8..+16, seq_len @+16..+20, block_table @+256 — +128
        // is clear). Fixed address + per-step contents = graph-safe (same
        // phasing as positions), uploaded pre-`begin_capture`. `DevicePtr(0)`
        // when no adapter pool is resident → the K/V/O apply sites take the
        // byte-identical installed-pair path. `seq.adapter_slot == -1` (no
        // per-request `adapter` field) resolves to the active slot.
        let seq_slot =
            self.upload_seq_slot_uniform(seq.adapter_slot, 1, meta_base.offset(128), stream)?;

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
        // CBD: run the FIRST decode step eagerly when dumping so per-layer
        // probes can sync (illegal under graph capture). Subsequent steps
        // still capture/replay normally.
        let dump_step0 =
            seq.seq_len == seq.prompt_len && std::env::var("ATLAS_SSM_SAVE_DUMP").is_ok();
        // EXPERIMENT (ATLAS_EP_GRAPHS=1): allow CUDA-graph capture under EP. The
        // EP all-reduce queues ncclSend/Recv + local-add on the compute (capture)
        // stream; NCCL ≥2.9 supports graph capture, so this MAY capture cleanly
        // and remove per-kernel launch overhead. Env-gated so it can be toggled
        // off at deploy time (instant revert) if capture crashes / replay hangs.
        let ep_graphs = std::env::var("ATLAS_EP_GRAPHS").is_ok_and(|v| v == "1" || v == "true");
        // GDN HeadParallel TP decode graphs (ATLAS_GDN_DECODE_GRAPH=1, default
        // OFF): capture the whole single-token decode forward — ~130 kernels
        // plus the per-layer TP all-reduces (48 GDN SSM out_proj + 16
        // attention o_proj on Qwen3.6) — into one replayable graph. The
        // collectives go through `all_reduce_async` (event fork/join onto the
        // dedicated NCCL comm stream), which stream capture pulls into the
        // graph as cross-stream nodes; capture runs in RELAXED mode (see
        // `begin_capture`) as NCCL requires, and the events are
        // CU_EVENT_DISABLE_TIMING (capture-legal). All per-token inputs
        // (token embedding, positions, slot, seq_len, block_table) are
        // uploaded to STABLE device buffers in Phase 1 before replay, and the
        // per-slot SSM conv/h states are updated in place at stable pointers,
        // so replay is shape/pointer-static. This removes the per-token host
        // launch cost that dominates 2-node GDN HeadParallel decode. Capture
        // failure falls back to eager execution (graphs then stay disabled).
        let gdn_graphs =
            std::env::var("ATLAS_GDN_DECODE_GRAPH").is_ok_and(|v| v == "1" || v == "true");
        // LoRA debugging hatch (ATLAS_LORA_EAGER=1): force eager decode when an
        // adapter is active so graph-vs-eager delta parity can be compared.
        // Default (unset) keeps graphs ON — the LoRA delta launches are
        // capture-safe (pool weights / arena scratch / f32 scale are all
        // load-time-fixed). Folded in as one more suppressor.
        let lora_eager = self.lora.is_some() && crate::lora::lora_eager_env();
        let use_graphs = (self.comm.is_none() || ep_graphs || gdn_graphs)
            && !self.profile
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged
            && !dump_step0
            && !lora_eager;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(attn_metadata),
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            gdn_exact_replay: false,
            // Hash-MoE: the single decode token ID (uploaded above every step
            // before graph replay). MoE reads it at offset 0.
            token_ids: Some(self.buffers.token_ids()),
            routed_lora_layers: None, // #30: single-seq decode never routes prefill.
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

        // Track whether a capture is actually recording: a begin_capture
        // failure falls back to eager execution (and disables graphs for the
        // rest of the run) instead of failing the decode step.
        let mut capture_active = false;
        if use_graphs {
            tracing::info!(
                "CUDA graph capture: starting for {} layers",
                self.layers.len()
            );
            match self.gpu.begin_capture(stream) {
                Ok(()) => capture_active = true,
                Err(e) => {
                    tracing::warn!(
                        "CUDA graph begin_capture failed ({e:#}) — \
                         running eagerly and disabling graph capture"
                    );
                    self.suppress_graphs
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        let probe_layers = !use_graphs
            && seq.seq_len == seq.prompt_len
            && std::env::var("ATLAS_SSM_SAVE_DUMP").is_ok();
        self.decode_forward_body(
            hidden,
            residual,
            seq,
            &mut kv_cache,
            &ctx,
            probe_layers,
            use_graphs,
            stream,
        )?;

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

        if capture_active {
            match self.gpu.end_capture(stream) {
                Ok(graph) if graph.0 != 0 => {
                    tracing::info!(
                        "CUDA graph captured successfully for slot={} (handle={:?})",
                        seq.slot_idx,
                        graph.0
                    );
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(seq.slot_idx, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
                Ok(_) => {
                    tracing::warn!("CUDA graph capture returned null handle — running eagerly");
                    // If graph.0 == 0 (mock): operations already executed during capture
                }
                Err(e) => {
                    // Capture RECORDS without executing, so nothing has run
                    // for this token yet — re-run the whole forward body
                    // eagerly (end_capture failure terminates the capture, so
                    // the stream is back in normal mode) and disable graphs
                    // for the rest of the run.
                    tracing::warn!(
                        "CUDA graph end_capture failed ({e:#}) — \
                         re-running decode step eagerly and disabling graph capture"
                    );
                    self.suppress_graphs
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    self.decode_forward_body(
                        hidden,
                        residual,
                        seq,
                        &mut kv_cache,
                        &ctx,
                        false,
                        false,
                        stream,
                    )?;
                }
            }
        }

        seq.tokens.push(token);
        seq.seq_len += 1;

        Ok(self.decode_logits_ptr())
    }

    /// Single-token decode forward body: per-layer decode + periodic SSM
    /// state normalization + final RMS norm + LM head.
    ///
    /// Runs once per decode step — either eagerly or inside a CUDA graph
    /// capture region — and a SECOND time as the eager fallback when
    /// `end_capture` fails (capture records without executing, so a re-run
    /// performs the step exactly once). Everything here is stream-ordered on
    /// `stream`; the only host syncs are gated on `!use_graphs` /
    /// `probe_layers` (both false under capture).
    fn decode_forward_body(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        probe_layers: bool,
        use_graphs: bool,
        stream: u64,
    ) -> Result<()> {
        for (i, layer) in self.layers.iter().enumerate() {
            layer.decode(
                hidden,
                residual,
                seq.layer_states[i].as_mut(),
                kv_cache,
                seq.seq_len,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;
            // CBD per-layer hidden fingerprint at decode step 0 (eager only).
            // Localizes the FIRST layer whose post-layer hidden diverges
            // cold-vs-ON / ON-vs-ON → pins the bug to that layer's read set.
            if probe_layers {
                self.gpu.synchronize(stream).ok();
                let mut hb = vec![0u8; self.config.hidden_size * 2];
                if self.gpu.copy_d2h(hidden, &mut hb).is_ok() {
                    let mut s = 0f64;
                    for c in hb.chunks_exact(2) {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        let v = f32::from_bits((bits as u32) << 16) as f64;
                        if v.is_finite() {
                            s += v.abs();
                        }
                    }
                    tracing::warn!("ATLAS_LAYER_H[step0] L{i} hidden_sabs={s:.6}");
                }
            }
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
        Ok(())
    }
}
