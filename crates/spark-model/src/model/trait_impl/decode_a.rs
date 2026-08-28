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
    /// Whether the online FP8-KV calibration has frozen its scale, model-wide.
    ///
    /// Every calibrating attention layer freezes on ITS first observe within
    /// the same first forward pass. `true` when NO layer runs online
    /// calibration — there is nothing to wait for, and the graph-suppression
    /// gate below is additionally guarded by `fp8_kv_calibration_tokens > 0`.
    /// Aggregation is all-frozen, not `find_map`: a BF16 boundary layer that
    /// never observes must not shadow later FP8 layers that already froze.
    pub(in crate::model) fn fp8_calibration_frozen(&self) -> bool {
        crate::layers::fp8_calibration::graphs_ready_after_fp8_kv_cal(
            self.layers.iter().map(|l| l.fp8_calibration_frozen()),
        )
    }

    pub(super) fn decode_dispatch(
        &self,
        token: u32,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<DevicePtr> {
        // Use backend's own stream (non-default, required for CUDA graph capture).
        let stream = self.gpu.default_stream();
        // ATLAS_SSM_H_FP16: narrow this sequence's SSM h-state to FP16 exactly
        // once, HERE — outside the CUDA-graph region. No-op without the flag.
        self.ssm_h_to_f16_dispatch(seq)?;
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

        // 1. Embedding lookup. `seq.tokens` is the history WITHOUT `token`
        // (it is pushed after the forward), which is exactly the n-gram
        // contract: preceding context, then the token being embedded.
        self.embed_ctx(&seq.tokens, token, hidden, stream)?;

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
            self.levers.kv_poison,
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
        // SAFETY: length derived from `bt_i32` itself — `len() * size_of::<i32>()` over the `collect`ed Vec above.
        let bt_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(bt_i32.as_ptr() as *const u8, bt_i32.len() * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(256), stream)?;

        // Upload the decode token ID into the STABLE token_ids buffer (uploaded
        // every step, BEFORE any CUDA-graph replay, so DeepSeek-V4 hash-MoE
        // layers read the correct `tid2eid[token_id]`). Single token at offset 0.
        self.gpu
            .copy_h2d_async(&token.to_le_bytes(), self.buffers.token_ids(), stream)?;

        // Hoisted host-side per-step layer work (PLE n-gram hash + NVMe
        // fault-in + slot upload into its stable slots buffer) — BEFORE any
        // graph replay/capture, same phasing as the token_ids upload above.
        // A replayed graph then only reads buffers this call refreshed.
        for (li, l) in self.layers.iter().enumerate() {
            l.decode_prestage(
                token,
                seq.layer_states[li].as_mut(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

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
            moe_row_adapter: spark_runtime::gpu::DevicePtr::NULL,
        };

        // CUDA graphs cannot capture NCCL all-reduce (it runs on a separate
        // stream) or cuStreamSynchronize calls. Suppress for EP and profile.
        // Re-enable graphs once FP8 calibration is frozen — keyed on the
        // ACTUAL frozen flag, not a token count: the scale freezes on the
        // first observe, and the old `seq_len > calibration_tokens + 10` gate
        // kept every process eager for ~266 tokens waiting on a calibration
        // that had already finished.
        if self.config.fp8_kv_calibration_tokens > 0
            && self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && self.fp8_calibration_frozen()
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
        let lora_eager = self.lora.is_some() && self.levers.lora_eager;
        // A layer that can never be captured (QSA's host top-k) vetoes
        // graphs for the whole model — a graph captured on the dense path
        // would silently replay WRONG attention once selection activates.
        let layer_veto = self.layers.iter().any(|l| l.decode_graph_unsupported());
        let use_graphs = (self.comm.is_none() || ep_graphs || gdn_graphs)
            && !self.profile
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged
            && !dump_step0
            && !lora_eager
            && !layer_veto;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(attn_metadata),
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            gdn_exact_replay: false,
            // Hash-MoE: the single decode token ID (uploaded above every step
            // before graph replay). MoE reads it at offset 0.
            token_ids: Some(self.buffers.token_ids()),
            // The very value uploaded into `token_ids` above — PLE hashes on
            // the host and must not round-trip it back off the device.
            host_token_ids: Some(std::slice::from_ref(&token)),
            routed_lora_layers: None, // #30: single-seq decode never routes prefill.
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(), // route-aware: base(Skip) decodes; adapter refuses
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
        if let Err(e) = self.decode_forward_body(
            hidden,
            residual,
            seq,
            &mut kv_cache,
            &ctx,
            probe_layers,
            use_graphs,
            stream,
        ) {
            // If the body errored WHILE a capture is recording (e.g. a MoE LoRA
            // refuse — router/non-active adapter/mixed batch — bails out of a
            // captured decode step), the stream is left in the capturing state.
            // Left as-is, the caller's cleanup (`free_sequence` → `zero_slot` /
            // `synchronize`) fails with STREAM_CAPTURE_UNSUPPORTED and every
            // subsequent op on this stream is poisoned — a single refused request
            // bricks the whole server. Release the stream (discarding the partial
            // graph) before propagating; no-op when not capturing. Graphs stay
            // enabled: the next decode step begins a fresh capture.
            self.gpu.abort_capture_if_active(stream);
            // A capture-poison error (900 CAPTURE_UNSUPPORTED / 901
            // CAPTURE_INVALIDATED) is a property of the graph attempt, not of
            // the request: capture RECORDS without executing, so nothing has
            // run for this token — re-run the step eagerly and disable graphs,
            // mirroring the end_capture-failure arm below, instead of failing
            // the request outright. Other errors (LoRA refusals, OOM) would
            // fail eagerly too, so they still propagate.
            let msg = format!("{e:#}");
            let capture_poison = capture_active
                && (msg.contains("status 901")
                    || msg.contains("status 900")
                    || msg.contains("STREAM_CAPTURE"));
            if !capture_poison {
                return Err(e);
            }
            tracing::warn!(
                "decode body failed under CUDA graph capture ({msg}) — \
                 re-running eagerly and disabling graph capture"
            );
            self.suppress_graphs
                .store(true, std::sync::atomic::Ordering::Relaxed);
            capture_active = false;
            // The recorded attempt consumed per-step prestaged state (PLE's
            // parked table VA); restore it without recomputing.
            for (li, l) in self.layers.iter().enumerate() {
                l.decode_prestage_rearm(seq.layer_states[li].as_mut());
            }
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

        // Decode-step diagnostic for Gemma-4 degeneration analysis (no-op unless
        // ATLAS_DIAG_GEMMA4=1). Split into decode_a_diag.rs for the LoC budget.
        self.diag_gemma4_decode_logits(token, stream)?;

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
                    // Restore per-step prestaged state the recorded attempt
                    // consumed (see the mid-body arm above).
                    for (li, l) in self.layers.iter().enumerate() {
                        l.decode_prestage_rearm(seq.layer_states[li].as_mut());
                    }
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

        // ── MTP SHADOW (ATLAS_QWEN4EXP_MTP_SHADOW=1, default OFF) ──
        // THIS is the path a qwen4_exp decode actually takes: decode_batch_dispatch
        // delegates every n == 1 step straight to decode(), so a hook in the
        // batched path never fires.
        //
        // Draft the next token and score the PREVIOUS draft against what the
        // target just produced. Feeds nothing back — it measures whether the
        // combiner reading is right before the verify path is built. `hc_streams`
        // still holds this row's four-stream highway: the last layer's hc_head
        // READ it and wrote the collapsed hidden elsewhere.
        if self.qwen4_exp_mtp_head.is_some() {
            match self.argmax_on_device(self.decode_logits_ptr(), 0) {
                Ok(actual) => self.qwen4_exp_mtp_shadow_step(
                    actual,
                    self.buffers.hc_streams(),
                    seq.seq_len,
                    &ctx,
                    stream,
                ),
                Err(e) => tracing::warn!("qwen4_exp MTP shadow: argmax failed: {e:#}"),
            }
        }

        seq.tokens.push(token);
        seq.seq_len += 1;

        Ok(self.decode_logits_ptr())
    }
}
