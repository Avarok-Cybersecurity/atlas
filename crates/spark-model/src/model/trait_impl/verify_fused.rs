// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash decode+verify single-pass fusion (M = 1 + num_drafts).
//!
//! Replaces the two separate weight sweeps on the DFlash path:
//!   - decode_a.rs  : M=1, accepted token → `decode_graph`
//!   - verify_b/c.rs: M=k, draft block   → `verify{k}_graph`
//!
//! with one M=(1+k) forward that reads every weight once. Row 0 is the
//! accepted token (decode semantics); rows 1..=k are the draft block
//! (verify semantics). `try_dflash_capture` fires at row 0 so the DFlash
//! drafter always conditions on a confirmed-accepted token's per-layer
//! hidden instead of a potentially-rejected draft's hidden.
//!
//! ## Safety contract (same as verify_b.rs)
//! `unsafe { from_raw_parts(...) }` reinterprets stack arrays of POD
//! integers as byte slices for H2D upload. All source types are POD with
//! no padding; lengths are exact; sources outlive the async copy.

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
    /// DFlash decode+verify fused dispatch.
    ///
    /// `tokens`: `[accepted_token, draft_0, ..., draft_{k-1}]`.
    /// Returns `Vec<u32>` of length `tokens.len()` — the in-graph argmax at
    /// each position. Row 0 is the "decode" result (what follows the accepted
    /// token); rows 1..k are the "verify" results for each draft.
    pub(super) fn decode_and_verify_fused_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize; // bytes per BF16 element
        let m = tokens.len(); // 1 + k
        let vocab = self.config.vocab_size;

        // SSM dual-buffer pre-verify copy (same as verify_b/c).
        self.pre_verify_copy_async(seq)?;

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: Pre-graph (varies per step, NOT captured) ──

        // 1a. Embed M tokens into consecutive rows of hidden_states.
        for (t, &tok) in tokens.iter().enumerate() {
            self.embed(tok, hidden.offset(t * h * bf16), stream)?;
        }

        // 1b. Allocate KV blocks for all M positions.
        let bs = kv_cache.block_size();
        for t in 0..m {
            let pos = seq.seq_len + t;
            let blocks_needed = (pos / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // 1c. Upload M-entry attention metadata.
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;

        // Positions: [seq_len, seq_len+1, ..., seq_len+m-1]
        let positions: Vec<u32> = (0..m).map(|t| (seq.seq_len + t) as u32).collect();
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, m * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;

        // KV write slots.
        let mut slots = vec![0i64; m];
        for t in 0..m {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = seq.physical_block_for(block_idx).unwrap_or(0);
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, m * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;

        // Seq-lens for multi-seq attention (staggered: row t sees seq_len+t
        // prior keys). Mirrors verify_b.rs / verify_c.rs convention.
        let seq_lens_meta: Vec<i32> = (0..m).map(|t| (seq.seq_len + t + 1) as i32).collect();
        let sl_bytes =
            unsafe { std::slice::from_raw_parts(seq_lens_meta.as_ptr() as *const u8, m * 4) };
        self.gpu
            .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

        // Block table: M identical rows (same physical sequence).
        let mb = max_blocks as usize;
        let needed = m * mb;
        let mut bt_buf_vec;
        let mut bt_buf_stack = [0i32; 1024];
        let bt_buf: &mut [i32] = if needed <= 1024 {
            &mut bt_buf_stack[..needed]
        } else {
            bt_buf_vec = vec![0i32; needed];
            &mut bt_buf_vec
        };
        for row in 0..m {
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                bt_buf[row * mb + j] = block as i32;
            }
        }
        let bt_bytes =
            unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

        let metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: m as u32,
        };

        // FP8 calibration re-enable (mirrors verify_b.rs).
        if self
            .suppress_graphs
            .load(std::sync::atomic::Ordering::Relaxed)
            && seq.seq_len > self.config.fp8_kv_calibration_tokens + 10
        {
            self.suppress_graphs
                .store(false, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("FP8 calibration frozen — re-enabling CUDA graphs (DFlash fused)");
        }

        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        let use_graphs = self.comm.is_none()
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            gdn_exact_replay: false,
            token_ids: None,
        };

        // ── Phase 2: CUDA graph capture / replay ──

        let cache_key = (seq.slot_idx, m);
        let mut graph_cache = if use_graphs {
            Some(self.fused_graph.lock())
        } else {
            None
        };

        let cached_for_slot = graph_cache
            .as_ref()
            .and_then(|c| c.get(&cache_key).copied());
        if let Some(graph) = cached_for_slot
            && graph.0 != 0
        {
            self.gpu.launch_graph(graph, stream)?;
        }
        let need_run = cached_for_slot.is_none();
        if need_run {
            // Staggered seq-lens for decode_multi_seq (same semantics as verify_b/c).
            let seq_lens_vec: Vec<usize> = (0..m).map(|t| seq.seq_len + t).collect();
            let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); m];

            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_type = self.config.layer_type(layer_idx);

                if layer_type == LayerType::FullAttention {
                    if hss_engaged {
                        // HSS: paged-decode kernel only reads HBM blocks, missing
                        // long-context disk history. Fall back to sequential
                        // decode_batched (same rationale as verify_b.rs).
                        layer.decode_batched(
                            hidden,
                            residual,
                            m,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &ctx,
                            stream,
                        )?;
                    } else {
                        let mut dummy_states: Vec<Box<dyn LayerState>> = (0..m)
                            .map(|_| layer.alloc_state(self.gpu.as_ref()))
                            .collect::<Result<_>>()?;
                        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                            dummy_states.iter_mut().map(|s| s.as_mut()).collect();
                        layer.decode_multi_seq(
                            hidden,
                            residual,
                            m,
                            &mut refs,
                            &mut kv_cache,
                            &seq_lens_vec,
                            &block_tables_vec,
                            &ctx,
                            stream,
                        )?;
                    }
                } else {
                    // SSM: batch all M tokens together (M=2/3 fast paths exist).
                    layer.decode_batched(
                        hidden,
                        residual,
                        m,
                        seq.layer_states[layer_idx].as_mut(),
                        &mut kv_cache,
                        seq.seq_len,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )?;
                }
                // DFlash hidden capture at row 0 (the accepted/decode token).
                // Row 0 = tokens[0] = a.last_token = the confirmed-accepted bonus
                // from the previous step. Capturing here ensures the DFlash drafter
                // always conditions on an accepted token's per-layer hidden, never
                // on a potentially-rejected draft's hidden.
                self.try_dflash_capture(layer_idx, 0, stream)?;
            }

            // Final norm over all M rows.
            let normed = self.buffers.norm_output();
            ops::rms_norm(
                self.gpu.as_ref(),
                self.rms_norm_kernel,
                hidden,
                &self.final_norm,
                normed,
                m as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;

            // LM head for M tokens (weights read once).
            self.lm_head_batched(normed, m as u32, self.buffers.logits(), stream)?;

            // In-graph argmax for all M rows (fixed scratch addresses — graph-safe).
            let argmax_out = self.buffers.scratch();
            for t in 0..m {
                let logits_t = self.buffers.logits().offset(t * vocab * bf16);
                let out_t = argmax_out.offset(t * 4);
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits_t,
                    out_t,
                    vocab as u32,
                    stream,
                )?;
            }

            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!(
                        "DFlash fused CUDA graph captured (slot={}, M={})",
                        seq.slot_idx,
                        m
                    );
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(cache_key, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        // ── Phase 3: Post-graph D2H ──

        let out_ptr = self.buffers.scratch();
        let mut buf = vec![0u8; m * 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let result: Vec<u32> = (0..m)
            .map(|t| {
                let b = t * 4;
                u32::from_le_bytes([buf[b], buf[b + 1], buf[b + 2], buf[b + 3]])
            })
            .collect();

        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += m;

        Ok(result)
    }
}
