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

    /// Verify `tokens` (K rows) by mini-prefill. Returns the argmax token for
    /// each row, so the caller can compare row `i`'s output against draft
    /// `i + 1` exactly as on the non-hc path.
    pub(super) fn decode_verify_hc(
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
