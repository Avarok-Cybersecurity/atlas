// SPDX-License-Identifier: AGPL-3.0-only

//! K-row speculative verify for models carrying an mHC highway.
//!
//! # Why this exists
//!
//! `decode_verify_dispatch` (verify_a.rs) verifies K tokens by running the
//! attention layers per-token through `decode()` and the SSM layers through
//! `decode_batched()`. Under an mHC highway that second call REFUSES:
//! `refuse_batched_under_hc` fires because the batched paths keep their own
//! residual bookkeeping and the highway replaces it, so running them would add
//! every block output to the residual twice. The scheduler turns that error into
//! `a.finished = true` — a SILENTLY TRUNCATED RESPONSE, not a fallback. So
//! speculation has been unavailable on this model class, for ANY proposer: the
//! same refusal sits under DFlash's batched verify (`verify_e.rs` routes the GDN
//! conv+WY body through `decode_verify_multi`).
//!
//! # The shape
//!
//! The only working multi-row mHC path is `prefill_inner_hc`, and BOTH layer
//! types have one (`qwen3_ssm/trait_prefill_hc.rs`,
//! `qwen3_attention/trait_impl/prefill_inner.rs:531`). `prefill()` dispatches to
//! it whenever `self.hc.is_some()`. So a K-row verify is expressible as a
//! MINI-PREFILL of the K candidate tokens at positions
//! `[seq_len, seq_len + K)`.
//!
//! Running EVERY layer through prefill (rather than mixing per-token attention
//! decode with K-row SSM prefill) is not a stylistic choice: the highway buffer
//! is laid out `[T, hc, H]`, so a 1-row attention path and a K-row SSM path
//! would disagree about what row a stream belongs to. Uniform K keeps one
//! layout.
//!
//! # MEASURED END TO END (2026-08-28) — RUNS, BUT WRONG AND SLOW
//!
//! With the proposer armed (`--speculative --num-drafts 1` +
//! `ATLAS_QWEN4EXP_MTP_VERIFY=1`), 4K ctx, greedy, vs a same-config baseline:
//! ```text
//!   baseline      19.8 tok/s  (50.5 ms/token)  correct output
//!   speculative,
//!     before      ~4.9 tok/s  (205 ms/token)   degenerate output
//!     after        8.3 tok/s  (120 ms/token)   degenerate output
//!   errors: 0
//! ```
//! The chain is COMPLETE — draft, verify, rollback and both carries run without
//! a single error, which no earlier revision managed. Two problems remain. The
//! COST one is now largely understood and 1.7x better (item 2); the CORRECTNESS
//! one is still open and its leading hypothesis has been disproved (item 1).
//!
//! 1. CORRECTNESS - STILL OPEN, and the leading hypothesis was TESTED AND
//!    DISPROVED. Four arms, same prompts, greedy, 4K ctx:
//!    ```text
//!      spec off (baseline)              "Red, blue, and green."   coherent
//!      spec on, rollback off            "Red light")..."          diverges ~tok 2-3
//!      spec on, rollback on             "Redaccion, 1."           diverges ~tok 2
//!      spec on, rollback on, old MoE    "Redaccion, ..."          diverges ~tok 2
//!      rollback errors: 0   panics: 0
//!    ```
//!    Read these carefully, because two plausible culprits are ELIMINATED:
//!
//!    * The missing rollback was the leading suspect - `rollback_verify_hc` was
//!      written but NOTHING CALLED IT, so a rejected draft left the aux carries
//!      un-restored. It is now wired (`Model::rollback_verify_rows`, called from
//!      the scheduler's K=2 reject branch) and it changes NOTHING: armed and
//!      unarmed diverge at the same point. It ships OFF
//!      (`ATLAS_QWEN4EXP_MTP_ROLLBACK=1` to arm) as unproven, not as harmful.
//!    * The small-M FFN substitution below is likewise exonerated - forcing the
//!      OLD grouped-MoE verify reproduces the identical corruption.
//!
//!    Note also that the "first ~12 tokens match the baseline" behaviour an
//!    earlier revision recorded DOES NOT REPRODUCE under this harness; every
//!    speculative arm diverges within 2-3 tokens. Treat the 12-token figure as
//!    prompt-specific and do not reason from it.
//!
//!    Divergence that early, with leaked raw special-token ids in the output
//!    (`| 100257`, `<|fim_prefix|>`), is a wrong-LOGITS signature rather than a
//!    slow state leak - the verify appears to return bad rows from nearly the
//!    first step, which no rewind can repair. Next suspects, in order: (a) the
//!    K-row logits the mini-prefill hands back - row indexing/aliasing into the
//!    logits buffer, the defect class this repo has hit repeatedly; (b) what
//!    `apply_aux_states` restores, PLE's rolling conv/history window especially,
//!    since unlike QSA's contiguous marks it cannot be rebuilt by truncation;
//!    (c) the scheduler's accepted-row bookkeeping vs what this verify advances.
//!    A row-by-row A/B of verify logits against a serial decode of the same
//!    tokens would settle (a) immediately and is the cheapest next experiment.
//!
//! 2. COST. Measured, gamma=1:
//!    ```text
//!      decode step           50.5 ms
//!      draft forward          2.6 ms   (shadow-on 53.1 vs baseline 50.5)
//!      verify (before)      ~395 ms  -> 205 ms/token end to end
//!      verify (after)                   120 ms/token end to end
//!    ```
//!    ★ THE DRAFT IS ESSENTIALLY FREE - 5% of a decode. The economics are
//!    entirely about verify. Break-even at ~91% accept needs
//!    `draft + verify < 95 ms`.
//!
//!    AN EARLIER REVISION OF THIS BLOCK CALLED THAT STRUCTURALLY BLOCKED, on
//!    the theory that the GDN prefill floor made a 2-row verify cost what a
//!    large chunk costs. PROFILING DISPROVED IT. Per-layer, per-verify-row:
//!    ```text
//!                  before    after
//!      moe        2700 us    191 us   (14x)
//!      gdn_block   860 us    862 us   (unchanged)
//!    ```
//!    The dominant term was never the GDN. It was the MoE: `forward_prefill`
//!    routes through the grouped GEMM, which streams every one of the 512
//!    experts' weights regardless of row count, so ONE row paid nearly what a
//!    28-row chunk paid (T=16 6.7-9.6 ms, T=28 8.5-12.3 ms -- 1.75x the rows
//!    for 1.2x the time). Substituting the single-token/K=2/K=3 MoE kernels at
//!    small row counts (`ATLAS_QWEN4EXP_HC_SMALL_M_FFN`, default on) cut it 14x.
//!
//!    NOTE the K=1 arm is the one that matters: `decode_verify_hc` splits a
//!    verify into row-0-then-drafts, so at gamma=1 BOTH calls arrive as a
//!    single row and the k2/k3 arms never fire.
//!
//!    WHERE IT STANDS: 120 ms/token vs a 50 ms decode -- speculation still does
//!    not pay, but it is now ~2.4x rather than ~4x, and the remaining cost has
//!    moved to the GDN: 36 layers x 862 us x 2 rows ~= 62 ms.
//!
//!    NEXT LEVER, and it is the same shape as the fix above: at T=1 a "prefill"
//!    row under the highway is just a decode step, so the hc decode body
//!    (`qwen3_ssm/trait_decode_hc.rs`) should serve it instead of the chunk
//!    scan. That is a 1-row substitution -- it does NOT require the batched
//!    multi-row GDN feature (#753 item B) that the earlier conclusion pinned
//!    this on. A batched K-row step remains the better endpoint, since two
//!    serial decodes (~101 ms) still exceed the ~92 ms budget on their own.
//!
//! Speculation therefore stays behind BOTH `--speculative` and
//! `ATLAS_QWEN4EXP_MTP_VERIFY=1`, and neither is a default.
//!
//! # What the caller still owes
//!
//! This advances sequence state by K rows and does NOT roll back. The caller
//! must `checkpoint_ssm_states` before and rewind on partial accept, exactly as
//! the non-hc verify path requires. Two rewinds are NOT yet implemented and are
//! why this stays behind a flag:
//!   * QSA `ingested` is monotone with a hard `ensure!(pos == st.ingested)` and
//!     has no rewind API;
//!   * PLE host history advances per row with a documented corruption class.
//!
//! # The rewind design (step 4), worked out but not yet wired
//!
//! The two carries need DIFFERENT mechanisms, which is why one blanket
//! "restore the snapshot" does not work:
//!
//! * **QSA is a mark rewind and is now implemented** —
//!   `QsaIndexer::rewind_seq_state`. `ingested`/`pooled` are contiguous marks
//!   and both device buffers are written forward from them, so moving the marks
//!   back is sufficient; stale bytes past the mark are overwritten by the next
//!   ingest. Cheap, no replay.
//! * **PLE needs a SNAPSHOT.** `PleSeqState::conv` is a rolling FP32 device
//!   convolution state and `history` is a fixed-length window whose oldest
//!   entries have already rolled off, so neither can be reconstructed by
//!   truncation. `snapshot_aux`/`restore_aux` already serialize both.
//!
//! ★ The placement is the subtle part. Restoring a PRE-verify snapshot leaves
//! the carries at `seq_len`, but a partial accept lands the sequence at
//! `seq_len + accepted` — one or more rows short. Taking the snapshot AFTER the
//! committed row 0 (the real sampled token) and before the DRAFT rows makes
//! restore land exactly right for the common γ=1 case: accept keeps everything,
//! reject restores to precisely "token_0 committed, draft discarded". That
//! argues for running verify as row-0-then-drafts rather than one K-row pass,
//! and is the open design decision for step 4.
//!
//! `snapshot_aux`/`restore_aux` are today wired only into the prefix-cache
//! path.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;
use crate::traits::SequenceState;

impl TransformerModel {
    /// True when K-row verify must take the mHC path.
    pub(super) fn verify_needs_hc_path(&self) -> bool {
        self.config.hc_mult > 0
    }

    /// Verify `tokens` by mini-prefill, SPLIT so a rejected draft can be rolled
    /// back exactly.
    ///
    /// Row 0 is the already-sampled real token and is always kept; rows 1.. are
    /// the drafts. The aux carries (QSA marks + PLE conv/history) are
    /// snapshotted BETWEEN the two, which is what makes `rollback_verify_hc`
    /// land on "token_0 committed, drafts discarded" rather than on pre-verify.
    /// Restoring a pre-verify snapshot would leave the carries a row SHORT of
    /// the sequence, which is a silent desync, not an error.
    pub(super) fn decode_verify_hc(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        let stream_d = self.gpu.default_stream();
        // Committed row first, on its own.
        let mut out = self.verify_hc_rows(&tokens[..1], seq, stream)?;
        if k == 1 {
            return Ok(out);
        }
        // Snapshot the carries with token_0 committed and no draft applied.
        // `collect_aux_states` covers BOTH: qwen3_attention's snapshot_aux
        // serializes the QSA carry, qwen3_ssm's serializes PLE's.
        let stash = self.collect_aux_states(seq, stream_d)?;
        *self
            .pending_verify_aux
            .lock()
            .map_err(|_| anyhow::anyhow!("verify aux stash poisoned"))? = Some(stash);

        out.extend(self.verify_hc_rows(&tokens[1..], seq, stream)?);
        Ok(out)
    }

    /// Undo `rows` draft rows after a rejected verify, restoring both aux
    /// carries to the snapshot taken after the committed row.
    ///
    /// The KV written for the discarded positions is left alone deliberately:
    /// it is past `seq_len` and the next step overwrites it.
    /// Restore ONLY the auxiliary carries stashed by `decode_verify_hc`, leaving
    /// `seq_len`/`tokens` to the caller.
    ///
    /// The scheduler's reject branch already owns the token/seq_len rewind
    /// (`seq_len -= 1` + `commit_accepted_prefix`); what it cannot do is rewind
    /// the QSA `ingested`/`pooled` marks a mini-prefill advanced. Splitting the
    /// aux half out lets the scheduler call exactly the missing piece instead of
    /// rewinding `seq_len` a second time.
    pub(super) fn restore_verify_aux(&self, seq: &mut SequenceState, stream: u64) -> Result<()> {
        let stash = self
            .pending_verify_aux
            .lock()
            .map_err(|_| anyhow::anyhow!("verify aux stash poisoned"))?
            .take();
        let Some(blobs) = stash else {
            anyhow::bail!(
                "restore_verify_aux with no stashed aux snapshot — decode_verify_hc \
                 must run first, or the carries cannot be rewound"
            );
        };
        self.apply_aux_states(seq, &blobs, stream)
    }

    pub(super) fn rollback_verify_hc(
        &self,
        seq: &mut SequenceState,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        self.restore_verify_aux(seq, stream)?;
        seq.seq_len = seq.seq_len.saturating_sub(rows);
        let keep = seq.tokens.len().saturating_sub(rows);
        seq.tokens.truncate(keep);
        Ok(())
    }

    /// One K-row mini-prefill. Advances sequence state by K rows.
    fn verify_hc_rows(
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
            let blk = kv_cache.alloc_block()?;
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
            profile: false,
            comm: self.comm_ref(),
            // Host-built metadata: capture is illegal here.
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            // PLE reads HOST ids for the rows it is about to process.
            host_token_ids: Some(tokens),
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        };

        // ── Every layer through its K-row mHC prefill path ──
        for (i, layer) in self.layers.iter().enumerate() {
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
        }
        drop(kv_cache);

        // ── K-row head: same tail as the non-hc verify ──
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        self.final_norm_apply(hidden, normed, k as u32, h as u32, eps, stream)?;
        self.lm_head_batched(normed, k as u32, self.buffers.logits(), stream)?;

        let mut out = Vec::with_capacity(k);
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
            let mut b = [0u8; 4];
            self.gpu.copy_d2h(out_ptr, &mut b)?;
            out.push(u32::from_le_bytes(b));
        }

        seq.tokens.extend_from_slice(tokens);
        seq.seq_len += k;
        Ok(out)
    }
}
