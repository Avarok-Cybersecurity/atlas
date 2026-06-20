// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! `TransformerModel::decode_batch_dispatch` — hoisted from `decode_a.rs`
//! to keep that file under the 500 LoC cap.
//!
//! Single entry point preserves the original control flow 1:1: special-case
//! n=1 and EP, otherwise pad to the nearest captured graph size, build a
//! `ForwardContext`, dispatch through `decode_multi_seq` for each layer,
//! and run final norm + per-seq LM-head GEMVs.

use anyhow::Result;
use atlas_core::config::LayerType;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::block_mgmt::{ensure_blocks_through_decode, extract_layer_refs};
use super::super::types::TransformerModel;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;
use crate::traits::{Model, SequenceState};

impl TransformerModel {
    pub(super) fn decode_batch_dispatch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        let n = tokens.len();
        assert_eq!(n, seqs.len(), "tokens.len() must equal seqs.len()");

        // Single-sequence: delegate to decode() which uses CUDA graphs.
        // decode_batch disables graphs for n≥2 (SSM state pointer staleness),
        // but n=1 is safe and benefits from graph replay (2x throughput).
        //
        // Broadcast the seq_id preamble + cmd here (rather than in the
        // scheduler) so the EP n>1 branch below can interleave broadcasts
        // with decode() calls — see that branch for the rationale.
        if n == 1 {
            self.ep_broadcast_cmd_for_seq(seqs[0].slot_idx as u32, tokens[0])?;
            self.decode(tokens[0], seqs[0], stream)?;
            return Ok(self.decode_logits_ptr());
        }

        // EP mode + n > 1: one batched forward pass per rank.
        //
        // Both ranks must call the same `decode_multi_seq` per-layer with
        // the same N tokens so the per-token NCCL all_reduces inside the
        // MoE forward match in shape and submission order across ranks.
        // The head announces the batch up-front via the `0xFFFFFFE0`
        // protocol primitive (seq_ids[N] + tokens[N] in one shot), then
        // both ranks run `decode_batch_compute_main` — the worker reaches
        // it via the matching handler in `ep_worker_step_impl`.
        //
        // Comm-stream op order on both ranks per step:
        //   B(0) B(0xFFFFFFE0) B(N) B*N(seq_ids) B*N(tokens)
        //   then per layer: per-token AR*N (forward_batched's inner loop)
        //
        // Single batched forward amortises weight loads + kernel launches
        // across N tokens. Per-token all_reduces (forward.rs:445,
        // forward_batched.rs:269) remain at shape `h * elem` per call —
        // batching the comm shape would need new MoE kernel work and is
        // deliberately out of scope here.
        if self.comm.is_some() {
            let seq_ids: Vec<u32> = seqs.iter().map(|s| s.slot_idx as u32).collect();
            self.ep_broadcast_decode_batch_dispatch(&seq_ids, tokens)?;
            return self.decode_batch_compute_main(tokens, seqs, stream);
        }

        // MLA models: as of issue #84 the batched `decode_multi_seq` path
        // HAS a genuine MLA branch (`ms_mla_decode` in
        // `qwen3_attention/trait_impl/multi_seq/mla.rs`) — the batched
        // analogue of `attention_forward_mla`. It reads `self.mla`'s
        // projections (not the NULL `attn.q_proj` stub the Mistral loader
        // installs) and isolates each sequence's compressed latent-KV via
        // per-sequence metadata. Concurrent MLA decode therefore takes the
        // normal batched path below — no host round-trip, no cross-seq
        // contamination.
        //
        // The legacy per-sequence `decode()` fallback (host-staged logits +
        // CUDA-graph suppression) is retained ONLY behind the
        // `ATLAS_MLA_PERSEQ_FALLBACK` escape hatch, as a guarded safety net
        // should a regression surface in the batched MLA path. It does NOT
        // fully isolate concurrent sequences (each `decode()`'s
        // `Buffers::zero_all` wipes the shared `logits` buffer), so it is
        // not the default.
        let mla_perseq_fallback = self.is_mla_dispatch()
            && std::env::var("ATLAS_MLA_PERSEQ_FALLBACK").is_ok_and(|v| v == "1" || v == "true");
        if mla_perseq_fallback {
            use std::sync::atomic::Ordering;
            let logits = self.decode_logits_ptr();
            let v = self.config.vocab_size;
            let elem = if self.decode_logits_fp32() { 4 } else { 2 };
            let row_bytes = v * elem;
            // Suppress CUDA graphs for the loop: `decode()`'s graph cache is
            // slot-keyed; capturing a graph for one slot inside the same
            // stream-capture window as another slot's replay corrupts both.
            let prev_suppress = self.suppress_graphs.swap(true, Ordering::Relaxed);
            let result = (|| -> Result<()> {
                let mut staged = vec![0u8; n * row_bytes];
                for i in 0..n {
                    self.decode(tokens[i], seqs[i], stream)?;
                    // `decode()` wrote this sequence's logits to row 0.
                    // Pull them to the host before the next `decode()`'s
                    // `zero_all` wipes the buffer. `copy_d2h_on_stream`
                    // syncs `stream` first, so the eager lm_head GEMV has
                    // fully landed before the copy reads it.
                    self.gpu.copy_d2h_on_stream(
                        logits,
                        &mut staged[i * row_bytes..(i + 1) * row_bytes],
                        stream,
                    )?;
                }
                // Upload the assembled [n, vocab] batch back to the device.
                self.gpu.copy_h2d_async(&staged, logits, stream)?;
                self.gpu.synchronize(stream)?;
                Ok(())
            })();
            self.suppress_graphs.store(prev_suppress, Ordering::Relaxed);
            result?;
            return Ok(logits);
        }

        self.decode_batch_compute_main(tokens, seqs, stream)
    }

    /// Shared batched-compute path used by both the head's EP branch and
    /// the worker's `0xFFFFFFE0` handler. Contains the per-step embed +
    /// KV-block alloc + metadata upload + per-layer `decode_multi_seq` +
    /// final norm + per-row LM-head GEMV pipeline. No EP broadcasts here
    /// — the head emits the protocol primitive before calling this; the
    /// worker reads the matching payload and dispatches into this from
    /// `ep_worker_decode_batch`. Both ranks then submit identical
    /// per-token `comm.all_reduce(h * elem)` ops on every MoE layer in
    /// the same order.
    pub(crate) fn decode_batch_compute_main(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<DevicePtr> {
        let n = tokens.len();
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize;
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        // Pad to nearest captured graph size [2, 4, 8]
        let padded_n = [2, 4, 8].iter().copied().find(|&s| s >= n).unwrap_or(n);

        // ── Phase 1: Pre-graph (runs every step, NOT captured) ──

        // 1a. Embed active tokens into hidden[0..n)
        for (i, &tok) in tokens.iter().enumerate() {
            self.embed(tok, hidden.offset(i * h * fp32), stream)?;
        }

        // 1b. Zero padding hidden[n..padded_n)
        for i in n..padded_n {
            self.gpu.memset(hidden.offset(i * h * fp32), 0, h * fp32)?;
        }

        // 1c. Allocate KV blocks for active sequences
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        for seq in seqs.iter_mut() {
            let blocks_needed = (seq.seq_len / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // 1d. Upload metadata with fixed stride (active + padding)
        let metadata = self.upload_batch_metadata_fixed(seqs, padded_n, &mut kv_cache, stream)?;

        // CUDA graphs DISABLED for multi-sequence decode: SSM state pointers
        // (h_state, conv_state) are baked into per-seq kernel args of
        // gdn_decode/conv1d_update at capture time. When batch composition
        // changes (sequences finish, swap_remove reorders), the graph
        // replays with stale pointers and corrupts SSM state across seqs.
        // Attention metadata uses fixed device addresses and is safe.
        // n==1 still uses self.decode()'s correct graph cache.
        let use_graphs = false;

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

        // ── Phase 2: CUDA graph lookup / capture ──
        let mut graphs = if use_graphs {
            Some(self.batch_decode_graphs.lock())
        } else {
            None
        };

        if let Some(ref graphs) = graphs
            && let Some(&graph) = graphs.get(&padded_n)
        {
            // Graph exists — replay (kernels use updated metadata + SSM pool addresses)
            if graph.0 != 0 {
                self.gpu.launch_graph(graph, stream)?;
            }

            // ── Phase 3: Post-graph (update sequence state) ──
            for (i, seq) in seqs.iter_mut().enumerate() {
                seq.tokens.push(tokens[i]);
                seq.seq_len += 1;
            }
            return Ok(self.decode_logits_ptr());
        }
        {
            // First time for this padded_n — capture a new graph (or run eagerly for EP).
            // Build layer states for all padded_n sequences (real + dummy padding).
            let seq_lens: Vec<usize> = (0..padded_n)
                .map(|i| if i < n { seqs[i].seq_len } else { 0 })
                .collect();
            let block_tables: Vec<Vec<u32>> = (0..padded_n)
                .map(|i| {
                    if i < n {
                        seqs[i].block_table.clone()
                    } else {
                        vec![self.dummy_kv_block]
                    }
                })
                .collect();

            // Extract real layer_states from sequences
            let mut all_layer_states: Vec<Vec<Box<dyn LayerState>>> = seqs
                .iter_mut()
                .map(|s| std::mem::take(&mut s.layer_states))
                .collect();

            // Build dummy layer_states for padding positions. Use the
            // dedicated `dummy_slot()` so pad SSM kernel writes can never
            // collide with another claimed sequence's pool memory if the
            // scheduler invariant ("active occupies contiguous slots
            // [0..n)") ever drifts.
            let dummy_ssm_slot = self.ssm_pool.dummy_slot();
            for _pad_pos in n..padded_n {
                let mut dummy: Vec<Box<dyn LayerState>> = Vec::with_capacity(self.layers.len());
                let mut ssm_idx = 0usize;
                for (li, layer) in self.layers.iter().enumerate() {
                    if self.config.layer_type(li) == LayerType::LinearAttention {
                        dummy.push(Box::new(SsmLayerState {
                            h_state: self.ssm_pool.h_state(ssm_idx, dummy_ssm_slot),
                            conv_state: self.ssm_pool.conv_state(ssm_idx, dummy_ssm_slot),
                            h_state_checkpoint: None,
                            conv_state_checkpoint: None,
                            h_state_intermediates: Vec::new(),
                            conv_state_intermediates: Vec::new(),
                        }));
                        ssm_idx += 1;
                    } else {
                        dummy.push(layer.alloc_state(self.gpu.as_ref())?);
                    }
                }
                all_layer_states.push(dummy);
            }

            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            // CONC_HSD: per-seq hidden-state dump diagnostic. Logs first 4 FP32
            // hidden values for each seq after each layer to localize where
            // pos>=1 diverges from pos 0 in concurrent batched decode.
            let conc_hsd = std::env::var("ATLAS_CONC_HSD").is_ok_and(|v| v == "1" || v == "true")
                && padded_n >= 2
                && self.comm.is_none();
            let dump_hidden = |label: &str, stream: u64| -> Result<()> {
                if !conc_hsd {
                    return Ok(());
                }
                self.gpu.synchronize(stream)?;
                let mut bufs: Vec<Vec<f32>> = Vec::with_capacity(padded_n);
                for i in 0..padded_n {
                    let mut buf = vec![0u8; 4 * 4]; // 4 FP32 values
                    let _ = self.gpu.copy_d2h(hidden.offset(i * h * fp32), &mut buf);
                    let vals: Vec<f32> = buf
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    bufs.push(vals);
                }
                let pretty: Vec<String> = bufs
                    .iter()
                    .enumerate()
                    .map(|(i, v)| format!("s{i}=[{:.4},{:.4},{:.4},{:.4}]", v[0], v[1], v[2], v[3]))
                    .collect();
                tracing::info!("CONC_HSD {label}: {}", pretty.join(" "));
                Ok(())
            };

            dump_hidden("post_embed", stream)?;

            // Layer loop for padded_n sequences
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let mut layer_state_refs = extract_layer_refs(&mut all_layer_states, layer_idx);
                layer.decode_multi_seq(
                    hidden,
                    residual,
                    padded_n,
                    &mut layer_state_refs,
                    &mut kv_cache,
                    &seq_lens,
                    &block_tables,
                    &ctx,
                    stream,
                )?;
                if conc_hsd {
                    let _ = dump_hidden(&format!("after_L{:02}", layer_idx), stream);
                }
            }

            // Final norm [padded_n, H]
            let normed = self.buffers.norm_output();
            ops::rms_norm(
                self.gpu.as_ref(),
                self.rms_norm_kernel,
                hidden,
                &self.final_norm,
                normed,
                padded_n as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;

            // LM head: padded_n sequential GEMVs (weights in L2 after first)
            let logits = self.buffers.logits();
            let v = self.config.vocab_size;
            for i in 0..padded_n {
                let normed_i = normed.offset(i * h * bf16);
                let logits_i = logits.offset(i * v * bf16);
                if let Some(ref fp8) = self.lm_head_fp8 {
                    ops::dense_gemv_fp8w(
                        self.gpu.as_ref(),
                        self.dense_gemv_fp8w_kernel,
                        normed_i,
                        fp8,
                        logits_i,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                } else if let Some(ref nvfp4) = self.lm_head_nvfp4 {
                    ops::w4a16_gemv(
                        self.gpu.as_ref(),
                        self.w4a16_gemv_kernel,
                        normed_i,
                        nvfp4,
                        logits_i,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                } else {
                    ops::dense_gemv(
                        self.gpu.as_ref(),
                        self.dense_gemv_kernel,
                        normed_i,
                        &self.lm_head_weight,
                        logits_i,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                }
            }

            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!("Captured CUDA graph for batch size {padded_n}");
                    if let Some(ref mut g) = graphs {
                        g.insert(padded_n, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }

            // Restore real layer_states to sequences (dummy states dropped)
            for (seq, ls) in seqs.iter_mut().zip(all_layer_states.drain(..n)) {
                seq.layer_states = ls;
            }
        }

        // ── Phase 3: Post-graph (update sequence state) ──
        for (i, seq) in seqs.iter_mut().enumerate() {
            seq.tokens.push(tokens[i]);
            seq.seq_len += 1;
        }

        Ok(self.decode_logits_ptr())
    }
}
