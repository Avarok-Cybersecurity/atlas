// SPDX-License-Identifier: AGPL-3.0-only

//! K=2 verify path.
//!
//! ## Safety
//!
//! `unsafe { from_raw_parts(...) }` blocks reinterpret stack arrays
//! / `Vec`s of POD integers as byte slices for H2D upload.
//! See `verify_c.rs` module docs for the full safety contract.

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
    pub(super) fn decode_verify_graphed_dispatch(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 2]> {
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize;
        let k = 2usize;

        // Item #2 (STree-style in-place K=2 verify): `h_state` IS canonical
        // — the verify kernel reads/writes it directly and the commit
        // (`commit_accepted_prefix`) rewinds it in place on reject. There is
        // no scratch/canonical split to seed, so the legacy SpecMamba
        // dual-buffer pre-verify copy (~60 MB h_state + conv D2D per K=2
        // step) is gone. The CUDA-graph capture below is unaffected: the
        // captured nodes take the same `h_state` pointer, which never moves.
        // (K=3/K=4/DFlash verify still run pre_verify_copy_async.)

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: Pre-graph (varies per step, NOT captured) ──

        // 1a. Embed 2 tokens
        self.embed(tokens[0], hidden, stream)?;
        self.embed(tokens[1], hidden.offset(h * fp32), stream)?;

        // 1b. Allocate KV blocks for both positions
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

        // 1c. Upload 2-entry attention metadata (same layout as batch decode)
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;

        // Zero-alloc metadata upload: cast stack arrays to byte slices directly.
        let positions = [seq.seq_len as u32, (seq.seq_len + 1) as u32];
        let pos_bytes = unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, 8) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;

        let mut slots = [0i64; 2];
        for t in 0..k {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            // Fall back to the dummy scratch block, NOT block 0: physical
            // block 0 is a live (likely shared prefix-cache) block, and a
            // silent write there corrupts cached KV for every reader.
            let physical_block = seq.physical_block_for(block_idx).unwrap_or_else(|| {
                tracing::error!(
                    "verify_k2: no physical block for pos {pos} (block_table len {}); \
                     writing KV to dummy block",
                    seq.block_table.len(),
                );
                self.dummy_kv_block
            });
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, 16) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;

        let seq_lens = [(seq.seq_len + 1) as i32, (seq.seq_len + 2) as i32];
        let sl_bytes = unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, 8) };
        self.gpu
            .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

        // Block table: K rows with same content (same physical sequence).
        let mb = max_blocks as usize;
        let needed = k * mb;
        let mut bt_buf_vec;
        let mut bt_buf_stack = [0i32; 1024];
        let bt_buf: &mut [i32] = if needed <= 1024 {
            &mut bt_buf_stack[..needed]
        } else {
            bt_buf_vec = vec![0i32; needed];
            &mut bt_buf_vec
        };
        for row in 0..k {
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
            num_seqs: k as u32,
        };

        // CUDA graphs cannot capture NCCL all-reduce (disabled for EP).
        // Also disable for FP8 native: w8a16_gemv kernel's __shared__ LUT load
        // has CUDA graph capture compatibility issues.
        //
        // Honor `suppress_graphs` so FP8 KV calibration runs eagerly during
        // warmup (its observe() does host syncs that are illegal inside CUDA
        // graph capture — CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, status 900).
        // With MTP, every decode step lands here (regular `decode` is never
        // called), so we also drive the same auto-unsuppress trigger that
        // `decode` uses: once `seq_len > calibration_tokens + 10` the scales
        // are frozen and graphs become safe.
        if self
            .suppress_graphs
            .load(std::sync::atomic::Ordering::Relaxed)
            && seq.seq_len > self.config.fp8_kv_calibration_tokens + 10
        {
            self.suppress_graphs
                .store(false, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("FP8 calibration frozen — re-enabling CUDA graphs (MTP verify)");
        }
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        let use_graphs = self.comm.is_none()
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            // Phase 6.2.c — see decode() for rationale: HSS path's host I/O is
            // illegal under CUDA graph capture.
            && !hss_engaged;

        // DeepSeek-V4 hash-MoE (first `num_hash_layers`) routes experts by token
        // id via the static tid2eid table, so the verify forward needs the 2
        // verify tokens in the stable `token_ids` device buffer. Uploaded
        // pre-graph (host→device is illegal under CUDA-graph capture); the graph
        // then reads the stable device buffer. Mirrors `decode()`'s token_ids.
        let tid_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        self.gpu
            .copy_h2d_async(&tid_bytes, self.buffers.token_ids(), stream)?;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            gdn_exact_replay: false,
            token_ids: Some(self.buffers.token_ids()),
        };

        // ── Phase 2: CUDA graph capture / replay ──

        let mut graph_cache = if use_graphs {
            Some(self.verify2_graph.lock())
        } else {
            None
        };

        // SLOT-KEYED LOOKUP: only replay if this seq's slot has a captured graph.
        let cached_for_slot = graph_cache
            .as_ref()
            .and_then(|c| c.get(&seq.slot_idx).copied());
        if let Some(graph) = cached_for_slot
            && graph.0 != 0
        {
            self.gpu.launch_graph(graph, stream)?;
        }
        let need_run = cached_for_slot.is_none();
        if need_run {
            let seq_lens_vec: Vec<usize> = (0..k).map(|t| seq.seq_len + t).collect();
            let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); k];

            // Extract layer states. Attention layers use EmptyLayerState (no actual
            // state), so sharing the same alloc is safe. For SSM layers, only one
            // sequence's state exists — pass it to decode_batched directly.
            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_type = self.config.layer_type(layer_idx);

                if layer_type == LayerType::FullAttention {
                    if hss_engaged {
                        // HSS path: `decode_multi_seq` calls the production
                        // paged-decode kernel which reads K/V from HBM only
                        // (`meta.block_table`). Under HSS, HBM is capped to
                        // `cache_blocks_per_seq` blocks, so older context
                        // lives only on disk and is unreachable from the
                        // multi-Q kernel — Q/V attends only over the recent
                        // ~cap×bs tokens, missing the long-context history.
                        // The single-token `decode` path routes through the
                        // HSS orchestrator (`attend_layer_on_stream`) which
                        // reads the full history from disk. Fall back to
                        // `decode_batched` (N sequential single-token
                        // decodes via the orchestrator) at the cost of
                        // ~k× attention launches per verify step. Mirrors
                        // the SSM branch below which already uses
                        // decode_batched for the same correctness reason.
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
                    } else {
                        // Attention: treat 2 tokens as 2 virtual sequences via
                        // decode_multi_seq. EmptyLayerState has no actual state.
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
                } else {
                    // SSM: process K=2 tokens for one sequence via decode_batched.
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
                // DFlash hidden capture for ctx conditioning. Capture from
                // the LAST verified position (K-1) — the bonus token in
                // K=2. This populates `dflash_hidden_save` so the next
                // `propose()` has fresh target hiddens. No-op when DFlash
                // is disabled.
                self.try_dflash_capture(layer_idx, k - 1, stream)?;
            }

            // Final norm [2, H]
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

            // LM head for 2 tokens (GEMM: weights loaded once)
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
                    tracing::info!("Captured CUDA graph for K=2 verify (slot={})", seq.slot_idx);
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(seq.slot_idx, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        // ── Phase 3: Post-graph (D2H copy only) ──

        let out_ptr = self.buffers.scratch();
        let mut buf = [0u8; 8];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let tok0 = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let tok1 = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);

        // EXPERIMENTAL: push ALL tokens (including tokens[0]) and advance
        // seq_len by K. Prior logic (`seq_len += k-1`, push only tokens[1..])
        // assumed tokens[0] was ALREADY in seq.tokens from a prior decode,
        // but that precondition is VIOLATED in the MTP flow: scheduler's
        // step_verify_k2 calls decode_verify_graphed([a.last_token, draft])
        // where a.last_token = sampled-but-not-pushed token from prior
        // bootstrap. Off-by-one accumulates across iterations and likely
        // underlies 80B-nvfp4-mtp fib drift (positions misaligned → wrong
        // RoPE → different logits → argmax flip on edge-case tokens).
        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        Ok([tok0, tok1])
    }
}
