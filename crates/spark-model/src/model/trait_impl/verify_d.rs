// SPDX-License-Identifier: AGPL-3.0-only

//! K=γ (DFlash) verify path.
//!
//! ## Safety
//!
//! `unsafe { from_raw_parts(...) }` blocks reinterpret stack arrays
//! / `Vec`s of POD integers (`u32`, `i32`, `i64`, `usize`) as byte
//! slices for H2D upload. See `verify_c.rs` module docs for the full
//! safety contract — same pattern, same invariants here.

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
    pub(super) fn decode_verify_graphed_kgamma_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize; // hidden states = BF16; matches verify_c.rs

        // F62 (2026-04-27): SpecMamba dual-buffer pre-verify copy.
        self.pre_verify_copy_async(seq)?;

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: Pre-graph (varies per step, NOT captured) ──

        // 1a. Embed K tokens
        for t in 0..k {
            self.embed(tokens[t], hidden.offset(t * h * fp32), stream)?;
        }

        // 1b. Allocate KV blocks for all K positions
        let bs = kv_cache.block_size();
        for t in 0..k {
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

        // 1c. Upload attention metadata to scratch (offset 32768).
        // Two layouts depending on whether prefill-mode attention is active:
        //   decode:   K-row format (num_seqs=K, K copies of block_table)
        //   prefill:  1-row format (num_seqs=1, single block_table row)
        // Positions and slots are identical in both layouts.
        let use_prefill_attn = std::env::var("ATLAS_VERIFY_PREFILL_ATTN")
            .ok()
            .as_deref()
            == Some("1");

        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;

        // positions[K×4] — same for both layouts
        let positions: Vec<u32> = (0..k).map(|t| (seq.seq_len + t) as u32).collect();
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, k * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;

        // slots[K×8] — same for both layouts
        let mut slots = vec![0i64; k];
        for t in 0..k {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = seq.physical_block_for(block_idx).unwrap_or(0);
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, k * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;

        let metadata = if use_prefill_attn {
            // Prefill layout: single seq_len value, single block_table row.
            // prefill_attention_paged computes kv_len from seq_len_start + num_tokens
            // and reads block_table as a flat 1D array — no seq_idx stride.
            let seq_len_val = [(seq.seq_len + k) as u32];
            let sl_bytes = unsafe {
                std::slice::from_raw_parts(seq_len_val.as_ptr() as *const u8, 4)
            };
            self.gpu
                .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

            let mb = max_blocks as usize;
            let mut bt_buf = vec![0i32; mb];
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                bt_buf[j] = block as i32;
            }
            let bt_bytes =
                unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, mb * 4) };
            self.gpu
                .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

            AttnMetadataDev {
                positions: meta_base,
                positions_h: meta_base,
                positions_w: meta_base,
                slot: meta_base.offset(256),
                seq_len: meta_base.offset(512),
                block_table: meta_base.offset(768),
                max_blocks_per_seq: seq.block_table.len() as u32,
                num_seqs: 1,
            }
        } else {
            // Decode layout: K rows of seq_lens and block_table.
            let seq_lens: Vec<i32> = (0..k).map(|t| (seq.seq_len + t + 1) as i32).collect();
            let sl_bytes =
                unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, k * 4) };
            self.gpu
                .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

            let mb = max_blocks as usize;
            let needed = k * mb;
            let mut bt_buf = vec![0i32; needed];
            for row in 0..k {
                for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                    bt_buf[row * mb + j] = block as i32;
                }
            }
            let bt_bytes =
                unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
            self.gpu
                .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

            AttnMetadataDev {
                positions: meta_base,
                positions_h: meta_base,
                positions_w: meta_base,
                slot: meta_base.offset(256),
                seq_len: meta_base.offset(512),
                block_table: meta_base.offset(768),
                max_blocks_per_seq: max_blocks,
                num_seqs: k as u32,
            }
        };

        // Phase 6.2.c — HSS host I/O is illegal under CUDA graph capture.
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        // ATLAS_DFLASH_DEBUG_NO_GRAPH=1 forces eager (no graph capture) so
        // CUDA_LAUNCH_BLOCKING=1 reports the exact failing kernel — used
        // to localize K=γ illegal-address crashes downstream of SSM.
        let force_eager = std::env::var("ATLAS_DFLASH_DEBUG_NO_GRAPH").ok().as_deref() == Some("1");
        let use_graphs = self.comm.is_none()
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged
            && !force_eager;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            gdn_exact_replay: false,
        };

        // ── Phase 2: CUDA graph capture / replay ──

        let mut graph_cache = if use_graphs {
            Some(self.verify_kgamma_graph.lock())
        } else {
            None
        };

        let cache_key = (seq.slot_idx, k);
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
            let seq_lens_vec: Vec<usize> = (0..k).map(|t| seq.seq_len + t).collect();
            let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); k];

            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            // SSM three-phase parallel verify (ATLAS_VERIFY_PREFILL_SSM=1).
            // When enabled, SSM layers use prefill_phase1/gdn_full/phase3
            // instead of sequential decode_batched(k), reducing GDN work from
            // K serial h_state updates to one WY4 batch recurrence.
            let use_prefill_ssm = std::env::var("ATLAS_VERIFY_PREFILL_SSM")
                .ok()
                .as_deref()
                == Some("1");
            // EAGLE-fix (K=γ): capture ALL k verify rows so the scheduler can
            // append rows 0..=num_accepted to ctx (fixes ctx-undercount + EAGLE
            // shift). Flag off → legacy single row-0 capture.
            let eagle_fix =
                std::env::var("ATLAS_DFLASH_EAGLE_FIX").ok().as_deref() == Some("1");
            // Precompute SSM buffer dimensions (same as prefill_c.rs / phase1_inner).
            let nk = self.config.linear_num_key_heads;
            let kd = self.config.linear_key_head_dim;
            let nv = self.config.linear_num_value_heads;
            let vd = self.config.linear_value_head_dim;
            let ssm_key_dim = nk * kd;
            let ssm_value_dim = nv * vd;
            let ssm_conv_dim = ssm_key_dim * 2 + ssm_value_dim;
            let ssm_gate_stride = nv * 2; // elements (FP32)
            let gdn_bufs = GdnPrefillBuffers {
                qkv:       self.gdn_buf_qkv,
                gate_beta: self.gdn_buf_gate_beta,
                output:    self.gdn_buf_out,
                z:         self.gdn_buf_z,
                total_len: k,
            };

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_type = self.config.layer_type(layer_idx);

                if layer_type == LayerType::FullAttention {
                    if hss_engaged {
                        layer.decode_batched(
                            hidden,
                            residual,
                            k,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &ctx,
                            stream,
                        )?;
                    } else if use_prefill_attn {
                        debug_assert!(seq.seq_len > 0, "prefill-verify requires non-empty context");
                        layer.prefill(
                            hidden,
                            residual,
                            k,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            0,
                            &ctx,
                            stream,
                        )?;
                    } else {
                        let mut dummy_states: Vec<Box<dyn LayerState>> = (0..k)
                            .map(|_| layer.alloc_state(self.gpu.as_ref()))
                            .collect::<Result<_>>()?;
                        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                            dummy_states.iter_mut().map(|s| s.as_mut()).collect();
                        layer.decode_multi_seq(
                            hidden,
                            residual,
                            k,
                            &mut refs,
                            &mut kv_cache,
                            &seq_lens_vec,
                            &block_tables_vec,
                            &ctx,
                            stream,
                        )?;
                    }
                } else if layer.is_ssm_layer() && use_prefill_ssm {
                    // Three-phase parallel SSM verify (see SSM_VERIFY_PLAN.md).
                    //
                    // Phase 1: K serial single-token calls fill gdn_bufs at
                    // token_offset=t and slide conv_state one step each, so
                    // conv_state_intermediates[t] can be saved after each call.
                    for t in 0..k {
                        layer.prefill_phase1(
                            hidden.offset(t * h * bf16),
                            residual.offset(t * h * bf16),
                            1,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len + t,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            0,
                            &gdn_bufs,
                            t,
                            &ctx,
                            stream,
                        )?;
                        let s = seq.layer_states[layer_idx]
                            .as_any_mut()
                            .downcast_mut::<SsmLayerState>()
                            .ok_or_else(|| {
                                anyhow::anyhow!("expected SsmLayerState at layer {layer_idx}")
                            })?;
                        if t < s.conv_state_intermediates.len() {
                            self.gpu.copy_d2d_async(
                                s.conv_state,
                                s.conv_state_intermediates[t],
                                self.ssm_pool.conv_bytes,
                                stream,
                            )?;
                        }
                    }
                    // Phase 2: one WY4 batch GDN recurrence over all K tokens.
                    layer.prefill_gdn_full(
                        seq.layer_states[layer_idx].as_mut(),
                        &gdn_bufs,
                        &ctx,
                        stream,
                    )?;
                    // Fill h_state_intermediates via per-token replay from checkpoint (R7).
                    // Extract DevicePtr copies before the replay loop to avoid borrow
                    // conflicts with the trait method calls inside the loop.
                    let (h_state_ptr, h_ckpt_opt, h_ints_vec) = {
                        let s = seq.layer_states[layer_idx]
                            .as_any_mut()
                            .downcast_mut::<SsmLayerState>()
                            .ok_or_else(|| {
                                anyhow::anyhow!("expected SsmLayerState at layer {layer_idx}")
                            })?;
                        (
                            s.h_state,
                            s.h_state_checkpoint,
                            s.h_state_intermediates.clone(),
                        )
                    };
                    let h_bytes = self.ssm_pool.h_bytes;
                    // Save WY4 final h_state to scratch
                    self.gpu.copy_d2d_async(
                        h_state_ptr,
                        self.ssm_verify_h_tmp,
                        h_bytes,
                        stream,
                    )?;
                    // Restore pre-verify checkpoint so replay starts from correct state
                    if let Some(ckpt) = h_ckpt_opt {
                        self.gpu.copy_d2d_async(ckpt, h_state_ptr, h_bytes, stream)?;
                    }
                    // K serial single-token GDN steps; after each step h_state_ptr
                    // device memory holds h_state after tokens 0..=t.
                    for t in 0..k.min(h_ints_vec.len()) {
                        let tok_gdn = GdnPrefillBuffers {
                            qkv:      gdn_bufs.qkv.offset(t * ssm_conv_dim * bf16),
                            gate_beta: gdn_bufs.gate_beta.offset(t * ssm_gate_stride * 4),
                            output:   gdn_bufs.output.offset(t * ssm_value_dim * bf16),
                            z:        gdn_bufs.z.offset(t * ssm_value_dim * bf16),
                            total_len: 1,
                        };
                        layer.prefill_gdn_full(
                            seq.layer_states[layer_idx].as_mut(),
                            &tok_gdn,
                            &ctx,
                            stream,
                        )?;
                        self.gpu.copy_d2d_async(
                            h_state_ptr,
                            h_ints_vec[t],
                            h_bytes,
                            stream,
                        )?;
                    }
                    // Restore WY4 final h_state
                    self.gpu.copy_d2d_async(
                        self.ssm_verify_h_tmp,
                        h_state_ptr,
                        h_bytes,
                        stream,
                    )?;
                    // Phase 3: gated RMS norm + output projection + FFN
                    layer.prefill_phase3(
                        hidden,
                        residual,
                        k,
                        &gdn_bufs,
                        0,
                        &ctx,
                        stream,
                    )?;
                } else {
                    layer.decode_batched(
                        hidden,
                        residual,
                        k,
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

                // DFlash per-layer hidden capture into dflash_hidden_save.
                // EAGLE-fix: capture all k verify rows (row-major) so the
                // scheduler can append one ctx slot per accepted position.
                // Legacy: capture only row 0 (last_token), appended via propose.
                if eagle_fix {
                    self.try_dflash_capture_all(layer_idx, k, stream)?;
                } else {
                    self.try_dflash_capture(layer_idx, 0, stream)?;
                }
            }

            // Final norm [K, H]
            let normed = self.buffers.norm_output();
            ops::rms_norm(
                self.gpu.as_ref(),
                self.rms_norm_kernel,
                hidden,
                &self.final_norm,
                normed,
                k as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;

            // LM head for K tokens
            self.lm_head_batched(normed, k as u32, stream)?;

            // Argmax inside graph (fixed scratch addresses — graph-safe)
            let vocab = self.config.vocab_size;
            let argmax_out = self.buffers.scratch();
            for t in 0..k {
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
                        "Captured CUDA graph for K=γ verify (slot={} K={})",
                        seq.slot_idx,
                        k
                    );
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(cache_key, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        // ── Phase 3: Post-graph (D2H copy only) ──

        let out_ptr = self.buffers.scratch();
        let mut buf = vec![0u8; k * 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let mut out = Vec::with_capacity(k);
        for t in 0..k {
            let off = t * 4;
            out.push(u32::from_le_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
            ]));
        }

        // See decode_verify_graphed for rationale on `seq_len += k` fix.
        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        Ok(out)
    }
}
