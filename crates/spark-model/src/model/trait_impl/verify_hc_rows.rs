// SPDX-License-Identifier: AGPL-3.0-only

//! The K-row mini-prefill itself, split out of `verify_hc.rs` to keep
//! that file under the 500-line cap. Declared there with `#[path]`.

use super::*;

impl TransformerModel {
    /// One K-row mini-prefill. Advances sequence state by K rows.
    pub(super) fn verify_hc_rows(
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
        let vocab = self.config.vocab_size;
        let bf16 = 2usize;

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();
        let mut kv_cache = self.kv_cache.lock();

        // ── KV blocks for every position this verify will write ──
        let bs = kv_cache.block_size();
        let last_pos = seq.seq_len + k - 1;
        let blocks_needed = (last_pos / bs) + 1;
        while seq.block_table.len() < blocks_needed {
            // ★ EVICT BEFORE GIVING UP — the prefill path always did, this one
            // did not, and that asymmetry is what took two GB10s down.
            //
            // A long run fills the pool with prefix-cache blocks that are
            // EVICTABLE: a finished sequence hands its blocks to the cache
            // (`cache_sequence`) and then drops its own ref (`free_sequence`),
            // leaving them at ref 1 — exactly what `evict` reclaims. Prefill
            // goes through `alloc_block_evicting` and keeps serving. This site
            // called the raw allocator, so it failed the instant the free list
            // emptied, with the pool full of blocks it was entitled to take.
            //
            // Reproduced 2026-09-17: ~250 distinct prompts filled a
            // 77,808-block pool, then every request died here with the bare
            // "KV cache exhausted: no free blocks" while the gauge read
            // used=77808 free=0 — and the server never recovered, because
            // nothing on this path ever asks for a reclaim. It is also the
            // same bare message rank 1 logged when it wedged the pair.
            //
            // No new staleness risk: `return_evicted_block` puts evicted
            // blocks back on the free list, so `alloc_block` was already
            // handing out previously-used blocks.
            let blk = crate::model::block_mgmt::alloc_block_evicting(
                &mut kv_cache,
                self.prefix_cache.as_ref(),
            )
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "KV cache exhausted in verify after eviction (blocks total={} free={}, \
                     this seq holds {} of {} needed) — every prefix-cache node left is \
                     still referenced by a live sequence",
                    kv_cache.num_blocks(),
                    kv_cache.num_free_blocks(),
                    seq.block_table.len(),
                    blocks_needed,
                )
            })?;
            seq.block_table.push(blk);
        }

        // ── Align the QSA carry with the rows about to be replayed ──
        // The graphed K=2 verify re-processes the CURRENT position: row 0 is
        // token_0, which the bootstrap decode already emitted — and already
        // ingested into the indexer. Replaying it without rewinding trips
        // `QSA: prefill chunk starts at 367 but 368 tokens are ingested`.
        // ALIGN to seq_len — absolute, not a fixed rewind. The overlap is not
        // constant: rewinding by 1 unconditionally produced the mirror-image
        // failure ("starts at 366 but 365 ingested"). Aligning never advances
        // the mark, so a carry that is already correct is untouched.
        for (i, layer) in self.layers.iter().enumerate() {
            layer.align_aux(
                seq.layer_states[i].as_mut(),
                seq.seq_len,
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // ── Embed the K candidates into hidden[K, H] ──
        // FP32 stride: `hidden_states` is the FP32 residual-stream buffer on
        // this path, matching verify_a.
        for (t, &token) in tokens.iter().enumerate() {
            self.embed(token, hidden.offset(t * h * 2), stream)?;
        }

        // ── Prefill-shaped metadata for K rows at [seq_len, seq_len+K) ──
        // Reuses the prefill packer rather than hand-rolling: it owns the
        // MRoPE stream layout and bounds the write against the scratch region.
        let meta_base = self.buffers.scratch().offset(32768);
        let meta_region = self.buffers.scratch_bytes().saturating_sub(32768);
        let all_tokens: Vec<u32> = seq
            .tokens
            .iter()
            .copied()
            .chain(tokens.iter().copied())
            .collect();
        let chunk_start = seq.tokens.len();
        let meta = self.prefill_b_upload_meta_at(
            &all_tokens,
            seq,
            chunk_start,
            k,
            seq.seq_len,
            k,
            seq.seq_len,
            &kv_cache,
            meta_base,
            meta_region,
            stream,
        )?;

        // Paged metadata (block table delta + seq_len) — the same helper the
        // chunked-prefill path uses. `needs_paged` is always true here: verify
        // only ever runs at seq_len_start > 0.
        if meta.needs_paged {
            // GROW the paged metadata. It was allocated for the ORIGINAL
            // prefill and verify extends past it — measured: "chunked prefill
            // metadata capacity 4 < required 7 blocks". `ensure_...` BAILS on a
            // short capacity rather than growing, so drop the old one first and
            // let it allocate at the size this verify needs. The old device
            // buffers are freed explicitly: `DevicePtr` has no Drop.
            let bs_meta = kv_cache.block_size();
            let need_blocks = all_tokens.len().saturating_sub(1) / bs_meta + 1;
            let too_small = seq
                .chunked_prefill_meta
                .as_ref()
                .is_some_and(|m| m.block_capacity < need_blocks);
            if too_small && let Some(old) = seq.chunked_prefill_meta.take() {
                let _ = self.gpu.free(old.block_table);
                let _ = self.gpu.free(old.seq_len);
            }
            self.ensure_chunked_prefill_meta(seq, all_tokens.len(), bs_meta)?;
            self.prefill_b_upload_paged(
                seq,
                all_tokens.len(),
                seq.seq_len,
                k,
                meta_base,
                meta.slot_offset,
                &kv_cache,
                stream,
            )?;
        }
        let (block_table_dev, seq_len_dev) = if meta.needs_paged {
            let pm = seq
                .chunked_prefill_meta
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("verify_hc: paged meta missing after upload"))?;
            (pm.block_table, pm.seq_len)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        let seq_slot = self.upload_seq_slot_uniform(
            seq.adapter_slot,
            k,
            self.buffers.lora_seq_slot(),
            stream,
        )?;

        // Field-for-field as prefill_c builds it. Pointing these at `meta_base`
        // wholesale (an earlier cut of this file) makes attention read the
        // position stream as its slot/seq_len/block table — silently wrong.
        let attn_metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(meta.slot_offset),
            seq_len: seq_len_dev,
            block_table: block_table_dev,
            max_blocks_per_seq: seq.block_table.len() as u32,
            num_seqs: 1,
            seq_slot,
            moe_row_adapter: DevicePtr::NULL,
        };

        // ── Per-row `seq_len` for the decode-shaped attention replay ──
        //
        // `prefill_b_upload_paged` writes ONE `i32` (the chunk end,
        // `seq_len + K`) because a prefill's paged attention is causal-masked
        // over the whole chunk. A DECODE step instead reads
        // `seq_lens[0]` as "how many keys are visible", so row `t` needs its
        // own value `seq_len + t + 1`. Parked immediately past the i64 slot
        // table inside the SAME metadata region, which nothing else writes.
        let attn_rows = verify_attn_decode_enabled();
        let row_seq_lens = meta_base.offset(verify_row_seq_len_offset(meta.slot_offset, k));
        if attn_rows {
            let need = verify_row_seq_len_offset(meta.slot_offset, k) + k * VERIFY_SEQ_LEN_STRIDE;
            anyhow::ensure!(
                need <= meta_region,
                "verify_hc: metadata region {meta_region} B cannot hold the per-row \
                 seq_len array ({need} B needed)"
            );
            let vals: Vec<i32> = (0..k)
                .map(|t| verify_row_seq_len_value(seq.seq_len, t))
                .collect();
            // SAFETY: exactly `k * 4` bytes over the live, fully initialised
            // `vals` Vec built on the lines above.
            let bytes = unsafe {
                std::slice::from_raw_parts(vals.as_ptr() as *const u8, k * VERIFY_SEQ_LEN_STRIDE)
            };
            self.gpu.copy_h2d(bytes, row_seq_lens)?;
        }

        let ctx = ForwardContext {
            decode_step: false,
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(attn_metadata),
            profile: false,
            comm: self.comm_ref(),
            // Host-built metadata: capture is illegal here.
            graph_capture: false,
            // The verify must agree TOKEN-FOR-TOKEN with serial decode: row 0
            // re-processes a row the decode already committed. `prefill()`
            // would otherwise pick the FLA chunked scan while `decode()`
            // carries H forward one token at a time — equivalent in exact
            // arithmetic, not in bf16 (measured on this model: chunked 110428
            // / 2097152 words differ, relL2 1.212e-3; sequential 0). That ~1%
            // per layer compounds over 48 layers and flips greedy argmaxes
            // wherever the top-2 margin is under ~0.9 logit units.
            // `AVAROK_QWEN4EXP_MTP_VERIFY_FLA=1` restores the chunked scan.
            gdn_exact_replay: !verify_uses_fla_scan(),
            token_ids: None,
            // PLE reads HOST ids for the rows it is about to process.
            host_token_ids: Some(tokens),
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        };

        // ── Every layer over the same K rows ──
        //
        // The 12 full-attention layers always take the K-row mHC prefill path
        // — that is also what advances the QSA `ingested`/`pooled` marks, the
        // third of the three per-row carries.
        //
        // The 36 GDN layers take one of two bodies:
        //   * DEFAULT — `prefill()` -> `prefill_inner_hc` -> `prefill_block`,
        //     the CHUNK SCAN. It writes no `h_state_intermediates`, so the
        //     caller must run one row per pass and publish them by hand
        //     (`publish_verify_row_state`).
        //   * `AVAROK_QWEN4EXP_MTP_HC_BATCHED=1` — `decode_batched()` ->
        //     `decode_batched_inner_hc` -> `decode_batched_block`, the fused
        //     conv+GDN verify kernels. They advance the recurrence over all K
        //     rows in ONE pass and write the per-row intermediates natively,
        //     which is the whole point: one pass instead of K, and the state
        //     published by the kernel that owns it.
        //
        // Both bodies are K-row and both sit at `hc_row_offset = 0`, so the
        // `[T, hc, H]` highway layout is uniform either way.
        let batched_gdn =
            crate::layers::qwen3_ssm::trait_decode_batched_hc::hc_batched_verify_enabled();
        // LIVENESS, once per process. The switch is an env read; this line is
        // the proof the batched body actually ran, which a flag is not.
        if batched_gdn {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::info!(
                    "mHC verify: K-row BATCHED GDN armed (AVAROK_QWEN4EXP_MTP_HC_BATCHED=1),                      first pass k={k} over {} layers",
                    self.layers.len()
                );
            });
        }
        if attn_rows {
            static SAID_ATTN: std::sync::Once = std::sync::Once::new();
            SAID_ATTN.call_once(|| {
                tracing::info!(
                    "mHC verify: attention layers replayed as K sequential one-row \
                     DECODE bodies (kill switch AVAROK_QWEN4EXP_MTP_HC_ATTN_DECODE=0)"
                );
            });
        }
        // EXPERIMENTAL, opt-in (`AVAROK_QWEN4EXP_MTP_HC_SSM_DECODE=1`): send the
        // GDN layers down the same one-row decode body. Legal ONLY at k == 1 --
        // the per-row reference arm's pass width -- because a plain `decode()`
        // writes no `h_state_intermediates`, and at k > 1 the commit rewind
        // would read never-written pool memory. At k == 1 `hc_publish_rows(1)`
        // is empty and the CALLER publishes the row state after the pass
        // (`publish_verify_row_state`), so the contract still holds.
        let ssm_rows = attn_rows && k == 1 && verify_ssm_decode_enabled();
        anyhow::ensure!(
            !(verify_ssm_decode_enabled() && k > 1),
            "AVAROK_QWEN4EXP_MTP_HC_SSM_DECODE=1 needs the per-row verify arm \
             (one row per pass); this pass is k={k}. Unset \
             AVAROK_QWEN4EXP_MTP_HC_BATCHED."
        );
        let base_seq_len = seq.seq_len;
        for (i, layer) in self.layers.iter().enumerate() {
            // ── Attention: K sequential one-row decode bodies ──
            //
            // Row 0 re-processes a token a serial decode already committed, so
            // its logits MUST equal that decode's. `prefill()` cannot give
            // that: prefill attention is the chunked/flash paged kernel over K
            // queries with a causal mask, decode attention is the paged-decode
            // GEMV against the KV cache at M=1 -- different reduction order,
            // and on this checkpoint different enough to flip greedy argmaxes.
            // The same split applies to the projections (GEMM at M=K vs GEMV
            // at M=1), to the MoE (grouped/fused-small-M vs the decode
            // expert path) and to QSA (`prefill_ingest` vs `decode_select`).
            //
            // LAYOUT RECONCILIATION -- the reason the module doc above said
            // this could not be done. The highway is `[T, hc_mult, H]` FP32
            // and `prefill_inner_hc` addresses it at
            // `hc_row_offset * hc_mult * H * 4` (prefill_inner.rs:565).
            // `decode_inner_hc` used to hard-code row 0. It now applies the
            // IDENTICAL arithmetic (decode_inner.rs:467), so a one-row decode
            // body at `hc_row_offset = t` reads and writes exactly the row the
            // K-row GDN body would have. `hidden`/`residual` are offset by the
            // same row in BF16 stride, and `hc_post`/`hc_comb`/`norm_output`
            // are per-pass scratch that a one-row body uses at its base. So
            // the 1-row attention path and the K-row SSM path agree about
            // which row a stream belongs to.
            //
            // The metadata is re-pointed per row rather than rebuilt: the
            // K-row pack already holds `[K]` u32 positions and `[K]` i64
            // slots, so row `t` is a pointer bump of `t*4` / `t*8`. Only the
            // device `seq_len` differs in KIND between the two shapes, and it
            // is uploaded above.
            // K-ROW ATTENTION BODY (default on; `AVAROK_QWEN4EXP_MTP_HC_ATTN_ROWS=0` disables): the
            // hyper-connection sites, the norms and the FFN run once at T=K
            // (the GDN layers' dispatch); only the attention core stays per
            // row. Same rows, same metadata, same highway rows as the loop
            // below; see qwen3_attention/trait_impl/verify_rows_hc.rs.
            if attn_rows
                && k > 1
                && !layer.is_ssm_layer()
                && crate::layers::qwen3_attention::verify_attn_rows_enabled()
                && let Some(attn) = layer.as_any().and_then(|a| {
                    a.downcast_ref::<crate::layers::qwen3_attention::Qwen3AttentionLayer>()
                })
                && attn.verify_rows_hc_ok()
            {
                static SAID_ROWS: std::sync::Once = std::sync::Once::new();
                SAID_ROWS.call_once(|| {
                    tracing::info!(
                        "mHC verify: attention layers run the K-ROW body \
                         (default on; AVAROK_QWEN4EXP_MTP_HC_ATTN_ROWS=0 disables), first pass k={k}"
                    );
                });
                let row_metas: Vec<AttnMetadataDev> = (0..k)
                    .map(|t| AttnMetadataDev {
                        positions: attn_metadata.positions.offset(t * VERIFY_POS_STRIDE),
                        positions_h: attn_metadata.positions_h.offset(t * VERIFY_POS_STRIDE),
                        positions_w: attn_metadata.positions_w.offset(t * VERIFY_POS_STRIDE),
                        slot: attn_metadata.slot.offset(t * VERIFY_SLOT_STRIDE),
                        seq_len: row_seq_lens.offset(t * VERIFY_SEQ_LEN_STRIDE),
                        block_table: attn_metadata.block_table,
                        max_blocks_per_seq: attn_metadata.max_blocks_per_seq,
                        num_seqs: 1,
                        seq_slot: attn_metadata.seq_slot,
                        moe_row_adapter: attn_metadata.moe_row_adapter,
                    })
                    .collect();
                let row_lens: Vec<usize> = (0..k)
                    .map(|t| verify_row_decode_seq_len(base_seq_len, t))
                    .collect();
                attn.decode_verify_rows_hc(
                    hidden,
                    k,
                    seq.layer_states[i].as_mut(),
                    &mut kv_cache,
                    &row_metas,
                    &row_lens,
                    &tokens[..k],
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    &ctx,
                    stream,
                )?;
                self.hidden_probe_layer("verify_hc", i, 0, hidden, stream);
                continue;
            }
            if attn_rows && (ssm_rows || !layer.is_ssm_layer()) {
                for t in 0..k {
                    let row_meta = AttnMetadataDev {
                        positions: attn_metadata.positions.offset(t * VERIFY_POS_STRIDE),
                        positions_h: attn_metadata.positions_h.offset(t * VERIFY_POS_STRIDE),
                        positions_w: attn_metadata.positions_w.offset(t * VERIFY_POS_STRIDE),
                        slot: attn_metadata.slot.offset(t * VERIFY_SLOT_STRIDE),
                        seq_len: row_seq_lens.offset(t * VERIFY_SEQ_LEN_STRIDE),
                        block_table: attn_metadata.block_table,
                        max_blocks_per_seq: attn_metadata.max_blocks_per_seq,
                        num_seqs: 1,
                        seq_slot: attn_metadata.seq_slot,
                        moe_row_adapter: attn_metadata.moe_row_adapter,
                    };
                    let row_ctx = ForwardContext {
                        decode_step: false,
                        buffers: &self.buffers,
                        hc_row_offset: t,
                        gpu: self.gpu.as_ref(),
                        config: &self.config,
                        dispatch: &self.dispatch,
                        derived: &self.derived,
                        levers: &self.levers,
                        stats: &self.stats,
                        attn_metadata: Some(row_meta),
                        profile: false,
                        comm: self.comm_ref(),
                        graph_capture: false,
                        // Attention layers never read it; carried so the two
                        // contexts cannot drift.
                        gdn_exact_replay: !verify_uses_fla_scan(),
                        token_ids: None,
                        host_token_ids: Some(&tokens[t..t + 1]),
                        routed_lora_layers: None,
                        midchunk_capture: None,
                        moe_lora_route: self.decode_moe_route(),
                    };
                    layer.decode(
                        hidden.offset(t * h * 2),
                        residual.offset(t * h * 2),
                        seq.layer_states[i].as_mut(),
                        &mut kv_cache,
                        // PRE-APPEND length, the decode convention: the token
                        // being processed sits at absolute position
                        // `base_seq_len + t`.
                        verify_row_decode_seq_len(base_seq_len, t),
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &row_ctx,
                        stream,
                    )?;
                }
                self.hidden_probe_layer("verify_hc", i, 0, hidden, stream);
                // INVARIANT: every exit from this loop body applies the
                // control vector exactly once, over all K rows at
                // `hc_row_offset = 0`. The per-row bodies above steer nothing
                // themselves — they run at `row_ctx`, one row each, and the
                // highway is only complete for this layer once they all have.
                self.cvec_after_layer(&ctx, "verify_rows", i, k, stream)?;
                continue;
            }
            if batched_gdn && layer.is_ssm_layer() {
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
                // See the INVARIANT above: this exit steers too.
                self.cvec_after_layer(&ctx, "verify_rows", i, k, stream)?;
                continue;
            }
            layer.prefill(
                hidden,
                residual,
                k,
                seq.layer_states[i].as_mut(),
                &mut kv_cache,
                seq.seq_len,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                seq.seq_len,
                &ctx,
                stream,
            )?;
            self.hidden_probe_layer("verify_hc", i, 0, hidden, stream);
            // See the INVARIANT above: the fallthrough K-row body steers too.
            self.cvec_after_layer(&ctx, "verify_rows", i, k, stream)?;
        }
        drop(kv_cache);

        // ── K-row head: same tail as the non-hc verify ──
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        // AVAROK_LOGIT_PROBE=1: the VERIFY side of the hidden-state A/B against
        // `decode_forward_body`. Same point in the pipeline (pre-final-norm),
        // same row stride, so an equal fingerprint blames the head and an
        // unequal one blames the layer bodies.
        for t in 0..k {
            self.hidden_probe("verify_hc", t, hidden.offset(t * h * 2), stream);
        }
        self.final_norm_apply(hidden, normed, k as u32, h as u32, eps, stream)?;
        self.lm_head_batched(normed, k as u32, self.buffers.logits(), stream)?;

        for t in 0..k {
            // AVAROK_LOGIT_PROBE=1: the verify side of the row-by-row A/B
            // against a serial decode of the same prefix. `lm_head_batched`
            // always writes BF16 here (the FP32-logits buffer is the
            // single-token decode path only).
            self.logit_probe(
                "verify_hc",
                t,
                self.buffers.logits().offset(t * vocab * bf16),
                false,
                stream,
            );
        }
        // ONE batched argmax + ONE readback, replacing K single-CTA scans each
        // followed by its own blocking 4-byte `copy_d2h`. Each of those drained
        // the stream inside the copy, so at K=3 this tail paid three full
        // drains — and the first of them absorbs the whole 48-layer verify
        // backlog, which is why it reads as the expensive call in a profile.
        //
        // `argmax_batch_dispatch` runs the identical per-row body (its own doc:
        // ties resolve the same way, byte-identical) and falls back to the loop
        // when the kernel set lacks the batched entry, so this is a transport
        // change only — the emitted tokens cannot move.
        //
        // NOT deleted outright even though the MTP path discards the result
        // (`verify_k3_step.rs` returns into `verify_mtp_wide::finish` before
        // reading it): `decode_verify_graphed_k3` is ALSO the DFlash path under
        // EP, where the values are read. The caller owns that choice; this end
        // just stops paying K drains for it.
        let out = self.argmax_batch_dispatch(self.buffers.logits(), k, stream)?;

        seq.tokens.extend_from_slice(tokens);
        seq.seq_len += k;
        Ok(out)
    }
}
