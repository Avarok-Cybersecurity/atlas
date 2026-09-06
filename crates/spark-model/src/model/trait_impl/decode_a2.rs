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

/// Multi-seq decode CUDA graphs: **ON by default**, disabled by
/// `ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1`.
///
/// Strict `== "1"` on an `ATLAS_NO_*` name rather than a presence check —
/// presence-checked flags here are ENABLED by `=0`. Read once per process.
fn multiseq_graphs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_NO_DECODE_GRAPHS_MULTISEQ").as_deref() != Ok("1"))
}

impl TransformerModel {
    pub(super) fn decode_batch_dispatch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        let n = tokens.len();
        assert_eq!(n, seqs.len(), "tokens.len() must equal seqs.len()");
        // ATLAS_SSM_H_FP16: narrow this sequence's SSM h-state to FP16 exactly
        // once, HERE — outside the CUDA-graph region. No-op without the flag.
        for s in seqs.iter_mut() {
            self.ssm_h_to_f16_dispatch(s)?;
        }

        // Single-sequence: delegate to decode() which uses its own slot-keyed
        // graph cache. (Stale comment removed: n>=2 is ALSO graphed — see the
        // slot-vector-keyed `batch_decode_graphs` in decode_batch_compute_main,
        // default-ON since 2026-07-27.)
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
        let mla_perseq_fallback = self.is_mla_dispatch()
            && std::env::var("ATLAS_MLA_PERSEQ_FALLBACK").is_ok_and(|v| v == "1" || v == "true");
        let qsa_active = self.config.index_topk > 0 && {
            // Mirrors QsaIndexer::inert_bound: index_topk IS the selection
            // budget in tokens (2048 on this card); at or below
            // budget + ratio - 1 visible tokens every block is selected and
            // selection is inert.
            let bound = self.config.index_topk + self.config.index_compress_ratio - 1;
            seqs.iter().any(|s| s.seq_len >= bound)
        };
        let hc_perseq = self.config.hc_mult > 0
            && (qsa_active || std::env::var("ATLAS_HC_PERSEQ_DECODE").as_deref() == Ok("1"));
        // ★ The per-seq routing decision is resolved ABOVE the EP branch on
        // purpose. It used to sit below, so under EP a QSA-active batch
        // returned at `decode_batch_compute_main` before ever reaching the
        // gate, landed on the batched multi-seq path, and died on its guard
        // ("QSA selection active for seq 0 on the batched ms path" — measured
        // on the 2-node EP=2 bring-up, 2026-08-27). EP does not change WHICH
        // path is correct for a sequence; it only changes how the worker is
        // told about it.
        if self.comm.is_some() && !(mla_perseq_fallback || hc_perseq) {
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
        // mHC highway models (#753 item B): the batched GDN paths are UNWIRED
        // (they carry their own residual, which the highway replaces), so the
        // per-seq loop is the DEFAULT here, not a fallback — each sequence
        // runs the proven single-row highway decode against its own per-seq
        // PLE/QSA state, and the host staging below isolates the logits rows.
        // Batched-highway kernels are the perf follow-up.
        // Highway models: the BATCHED multi-seq path (per-layer hc-bracketed
        // decode, weight reads amortized at the GEMM level next increment)
        // is the default. Fall back to the per-seq staging loop when
        //   * ATLAS_HC_PERSEQ_DECODE=1 (A/B escape hatch), or
        //   * QSA would be ACTIVE for any sequence (the ms attention path has
        //     no per-seq indexer hook yet — dense past the budget is NOT the
        //     reference model, so keep those batches on the proven loop).
        if mla_perseq_fallback || hc_perseq {
            use std::sync::atomic::Ordering;
            let logits = self.decode_logits_ptr();
            let v = self.config.vocab_size;
            let elem = if self.decode_logits_fp32() { 4 } else { 2 };
            let row_bytes = v * elem;
            // Suppress CUDA graphs for the loop: `decode()`'s graph cache is
            // slot-keyed; capturing a graph for one slot inside the same
            // stream-capture window as another slot's replay corrupts both.
            let prev_suppress = self.suppress_graphs.swap(true, Ordering::Relaxed);
            // The scheduler passes stream 0 (legacy) here, but `decode()`
            // runs its kernels on the BACKEND default stream — staging the
            // rows on the caller's stream orders the copies against nothing:
            // all n copies can execute after the LAST decode and read the
            // same final row 0 (measured: a clean two-way row swap at every
            // joint C=2 step, '#753 item B' bring-up). Stage on the stream
            // the kernels actually use.
            let copy_stream = self.gpu.default_stream();
            let result = (|| -> Result<()> {
                let mut staged = vec![0u8; n * row_bytes];
                for i in 0..n {
                    // Same announcement the `n == 1` path makes: under EP the
                    // worker must run THIS sequence's single-seq forward, so
                    // the comm-stream op order stays B(seq)B(cmd) per row on
                    // both ranks. No-ops without a communicator.
                    self.ep_broadcast_cmd_for_seq(seqs[i].slot_idx as u32, tokens[i])?;
                    self.decode(tokens[i], seqs[i], stream)?;
                    // `decode()` wrote this sequence's logits to row 0.
                    // Pull them to the host before the next `decode()`'s
                    // `zero_all` wipes the buffer. `copy_d2h_on_stream`
                    // syncs `copy_stream` first, so the eager lm_head GEMV
                    // has fully landed before the copy reads it.
                    self.gpu.copy_d2h_on_stream(
                        logits,
                        &mut staged[i * row_bytes..(i + 1) * row_bytes],
                        copy_stream,
                    )?;
                }
                // Upload the assembled [n, vocab] batch back to the device.
                self.gpu.copy_h2d_async(&staged, logits, copy_stream)?;
                self.gpu.synchronize(copy_stream)?;
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
        // SOLID Incr-4 pre-lookup guard (the bail `forward_batched.rs` and
        // `build_moe_row_adapter_decode` document): a batch with a row routed
        // to a NON-active adapter cannot be served by the single-active fold —
        // the per-row map defensively writes such rows as base, so proceeding
        // would SILENTLY serve base weights for an adapter-routed request.
        // Bail before ANY per-step work (embed / metadata / row-map upload /
        // graph lookup), keeping the captured padded_n graphs route-agnostic.
        // The route is stamped per batch at the `Model` entry
        // (`stamp_decode_moe_batch`).
        crate::lora::ensure_decode_route_servable(
            self.decode_moe_route(),
            "decode_batch_compute_main",
        )?;
        // ATLAS_SSM_H_FP16: narrow this sequence's SSM h-state to FP16 exactly
        // once, HERE — outside the CUDA-graph region. No-op without the flag.
        for s in seqs.iter_mut() {
            self.ssm_h_to_f16_dispatch(s)?;
        }
        if std::env::var("ATLAS_DECODE_BATCH_LOG").ok().as_deref() == Some("1") {
            let slots: Vec<i64> = seqs
                .iter()
                .map(|s| {
                    s.ssm_slot
                        .as_ref()
                        .and_then(|g| g.idx())
                        .map(|x| x as i64)
                        .unwrap_or(-1)
                })
                .collect();
            let contiguous = slots.iter().enumerate().all(|(i, &s)| s == i as i64);
            tracing::info!(
                "ATLAS_DECODE_BATCH: n={n} slots={slots:?} contiguous_0..n={contiguous}"
            );
        }
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize;
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        // Pad to the nearest captured graph size — SSOT ladder in
        // `traits::padded_batch_n` (now includes 12 and 16 for the C-sweep).
        let padded_n = crate::traits::padded_batch_n(n);

        // CUDA graphs for multi-sequence decode (ATLAS_DECODE_GRAPHS_MULTISEQ=1).
        //
        // SSM h_state/conv_state pointers ARE baked into per-seq kernel args at
        // capture, so the cache is keyed by the per-row SSM slot VECTOR — see
        // `decode_graph_key.rs` for why the former `padded_n` key was unsound.
        // Everything else captured (metadata, block tables, embed, scratch) is
        // a fixed address refreshed every step BEFORE replay. This is the
        // dominant lever for n>=2 decode (eliminates ~1500 launches/step).
        //
        // DEFAULT-ON since 2026-07-27; disable with
        // ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1. Measurements + the rewrite this
        // retired: `decode_graph_key.rs`.
        let ms_profile = std::env::var("ATLAS_MS_PROFILE").ok().as_deref() == Some("1");
        // ATLAS_MS_PROFILE forces eager (graphs off) so per-phase syncs are legal.
        // ATLAS_LORA_EAGER: same LoRA graph-vs-eager debugging hatch as decode_a.
        let lora_eager = self.lora.is_some() && self.levers.lora_eager;
        // Per-layer graph veto (QSA's mid-decode top-k D2H, PLE's per-seq
        // host hash on the hc multi-seq path) — the single-decode path
        // consults it (decode_a `layer_veto`); the batched path must too, or
        // capture hits 'PLE: un-prestaged forward inside CUDA graph capture'
        // on the first joint hc step.
        let layer_veto = self.layers.iter().any(|l| l.decode_graph_unsupported());
        let graph_key = if !ms_profile && !lora_eager && !layer_veto && multiseq_graphs_enabled() {
            self.batch_decode_graph_key(&*seqs, padded_n)
        } else {
            None
        };
        let use_graphs = graph_key.is_some();

        // Lock order: kv_cache BEFORE the graph cache, matching verify_e.
        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 2 (decision): exact CUDA-graph hit, or drain-tail borrow ──
        let mut graphs = if use_graphs {
            Some(self.batch_decode_graphs.lock())
        } else {
            None
        };

        // Exact hit (LRU-touched), else borrow a WIDER captured graph
        // (`graph_borrow.rs`): a drain batch's slot vector is a prefix of the
        // steady-state canonical vector, so the wider graph's active rows are
        // exactly this batch's rows and its tail rows pad on the dummy slot
        // or currently-free slots. `dispatch_n` is the width Phase 1 must
        // prepare (embeds/metadata) — the borrowed graph's captured width on
        // a borrow, `padded_n` otherwise.
        let mut replay: Option<spark_runtime::gpu::GraphHandle> = None;
        let mut dispatch_n = padded_n;
        if let (Some(g), Some(key)) = (&mut graphs, &graph_key) {
            g.1 += 1;
            let tick = g.1;
            if let Some(e) = g.0.get_mut(key) {
                e.1 = tick;
                replay = Some(e.0);
            } else if super::graph_borrow::graph_borrow_enabled()
                && self.comm.is_none()
                && self.config.num_ssm_layers() > 0
            {
                let dummy = self.ssm_pool.dummy_slot() as u32;
                let borrowed =
                    super::graph_borrow::find_borrowable_decode_key(&key[..n], g.0.keys(), |s| {
                        s == dummy || self.ssm_pool.slot_is_free(s as usize)
                    });
                if let Some(bk) = borrowed {
                    dispatch_n = bk.len();
                    let e =
                        g.0.get_mut(&bk)
                            .expect("borrowed key comes from this cache");
                    e.1 = tick;
                    replay = Some(e.0);
                    // INFO once per transition (same cardinality as the
                    // captures this replaces); repeats of the same pair
                    // stay silent. Provable engagement: grep "graph borrow".
                    if super::graph_borrow::DECODE_BORROW_LOG.should_log(key, &bk) {
                        tracing::info!(
                            "decode graph borrow: n={n} padded_n={padded_n} -> replaying \
                             captured width {dispatch_n}"
                        );
                    }
                }
            }
        }

        // ── Phase 1: Pre-graph (runs every step, NOT captured) ──

        // 1a. Embed active tokens into hidden[0..n)
        for (i, &tok) in tokens.iter().enumerate() {
            // Each batch slot is a DIFFERENT sequence: the n-gram context must
            // come from that sequence's own history, never the batch's.
            self.embed_ctx(&seqs[i].tokens, tok, hidden.offset(i * h * fp32), stream)?;
        }

        // 1b. Zero padding hidden[n..dispatch_n)
        for i in n..dispatch_n {
            self.gpu.memset(hidden.offset(i * h * fp32), 0, h * fp32)?;
        }

        // 1c. Allocate KV blocks for active sequences
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
                self.levers.kv_poison,
            )?;
        }

        // 1d. Upload metadata with fixed stride (active + padding)
        let metadata = self.upload_batch_metadata_fixed(seqs, dispatch_n, &mut kv_cache, stream)?;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            // SOLID Incr-4: batched decode FOLDS. `Refuse` was bailed by the
            // pre-lookup guard above; the metadata carries the per-row
            // `moe_row_adapter` map (`upload_batch_metadata_fixed`), and the
            // MoE decode ladder's token-major arm delegates to
            // `forward_batched`'s per-row router/gate-up/down folds whenever
            // an adapter is resident (presence gate, like forward_k2/k3).
            // Base (Skip) batches still pay nothing.
            moe_lora_route: self.decode_moe_route(),
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            // The batch's token ids: the hc multi-seq PLE rows read their
            // per-seq id from this slice.
            host_token_ids: Some(tokens),
            routed_lora_layers: None, // #30: batched decode never routes prefill.
            midchunk_capture: None,
        };

        if let Some(graph) = replay {
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
                            // Padding rows point at the write-only dummy slot;
                            // tag them with the active mode so the decode mixer
                            // does not re-convert scratch on every single step.
                            h_is_f16: crate::layers::qwen3_ssm::ssm_h_fp16_enabled(),
                            // Decode-only rows: never prefilled. Carried
                            // anyway so the dummy slot's geometry matches a
                            // real one and a stray prefill over it stages
                            // rather than overruns.
                            h_prefill_stage: self.ssm_pool.h_prefill_stage(dummy_ssm_slot),
                            ple: None,
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
            let mut ssm_us: u128 = 0;
            let mut attn_us: u128 = 0;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let mut layer_state_refs = extract_layer_refs(&mut all_layer_states, layer_idx);
                let t0 = if ms_profile {
                    self.gpu.synchronize(stream).ok();
                    Some(std::time::Instant::now())
                } else {
                    None
                };
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
                if let Some(t0) = t0 {
                    self.gpu.synchronize(stream).ok();
                    let dt = t0.elapsed().as_micros();
                    if self.config.layer_type(layer_idx) == LayerType::LinearAttention {
                        ssm_us += dt;
                    } else {
                        attn_us += dt;
                    }
                }
                // DFlash multi-row hidden capture: batched decode advances
                // EVERY sequence one position per step, so every batch row's
                // per-layer hidden must reach the capture scratch (row i =
                // seq i) for the scheduler's per-seq `commit_ctx(.., i)`.
                // Captured inside the graph region — src (shared hidden
                // buffer) and dst (scratch) are fixed addresses, so replays
                // stay correct; a borrowed wider graph writes garbage into
                // rows n..padded_n, which the scheduler never commits.
                // No-op unless DFlash is on and this is a capture layer.
                self.try_dflash_capture_all(layer_idx, padded_n, stream)?;
                if conc_hsd {
                    let _ = dump_hidden(&format!("after_L{:02}", layer_idx), stream);
                }
            }
            if ms_profile {
                self.gpu.synchronize(stream).ok();
            }
            let lmhead_t0 = if ms_profile {
                Some(std::time::Instant::now())
            } else {
                None
            };

            // Final norm [padded_n, H]
            let normed = self.buffers.norm_output();
            self.final_norm_apply(
                hidden,
                normed,
                padded_n as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;

            // LM head: ONE batched [padded_n, vocab] GEMM so the ~254 MB
            // vocab weight is read ONCE per step instead of once per sequence
            // (the per-row GEMV loop re-read it N times — a major C>=2 cost:
            // ~N×254 MB/step). nvfp4/dense are batched here; FP8 single-scale
            // keeps the per-row path (no batched single-scale FP8 GEMM handle
            // on the model, and Holo's lm_head is NVFP4 anyway).
            // The ladder itself lives in `lm_head_batched.rs` — the mixed
            // co-dispatch head (`decode_b2::mixed_final_norm_lm_head`) calls
            // the same function, so the two heads cannot pick different
            // kernels for the same `padded_n`.
            // The returned pointer is discarded here: this path reports its
            // logits through `self.decode_logits_ptr()` at the end of the
            // function, which reads the same buffer.
            self.lm_head_project_batched(normed, padded_n, h, bf16, stream)?;
            if let Some(t0) = lmhead_t0 {
                self.gpu.synchronize(stream).ok();
                let head_us = t0.elapsed().as_micros();
                let total = ssm_us + attn_us + head_us;
                tracing::info!(
                    "ATLAS_MS_PROFILE n={n} padded_n={padded_n}: total={}us  ssm={}us({}L)  attn={}us({}L)  head={}us  [per-tok {:.2}ms]",
                    total,
                    ssm_us,
                    self.config.num_ssm_layers(),
                    attn_us,
                    self.layers.len() - self.config.num_ssm_layers(),
                    head_us,
                    total as f64 / 1000.0 / padded_n as f64,
                );
            }

            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!(
                        "Captured CUDA graph for batch size {padded_n} (n={n}, slots={graph_key:?})"
                    );
                    if let (Some(g), Some(key)) = (graphs.as_mut(), graph_key.clone()) {
                        self.insert_batch_decode_graph(g, key, graph);
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
