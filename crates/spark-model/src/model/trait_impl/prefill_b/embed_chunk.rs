// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 1+1b: embed chunk tokens to hidden buffer + overlay vision-pad
//! positions with pre-computed vision encoder embeddings.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;

use super::super::super::types::TransformerModel;
use crate::layers::ops;

impl TransformerModel {
    pub(super) fn prefill_b_embed_chunk(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        stream: u64,
    ) -> Result<()> {
        // Single-stream entry point: write to the arena's hidden buffer at offset 0.
        let hidden = self.buffers.hidden_states();
        self.prefill_b_embed_chunk_at(tokens, chunk_start, chunk_len, hidden, stream)
    }

    /// Embed `chunk_len` tokens into `hidden_dst` starting at position 0
    /// of the destination, then apply embedding scale + vision-pad overlay.
    /// Used by both the single-stream entry point above (writing into the
    /// arena's `hidden_states()`) and by Q12 batched prefill (writing into
    /// per-stream offsets of a shared stacked-streams buffer).
    pub(in crate::model) fn prefill_b_embed_chunk_at(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        hidden_dst: spark_runtime::gpu::DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        // BF16 residual is the shipping config (2 bytes/element).
        let elem_bytes = 2usize;

        // ── 1. Embed chunk tokens → [chunk_len, H] contiguous at hidden_dst ──
        // Upload token IDs to device and do a single batched embed kernel launch
        // instead of chunk_len individual D2D copies.
        {
            let chunk_tokens = &tokens[chunk_start..chunk_start + chunk_len];
            // SAFETY: `chunk_tokens` is sliced on the line above with an END
            // bound of `chunk_start + chunk_len`, so its length IS `chunk_len`
            // (an out-of-range chunk panics in that slice index first) and the
            // byte length is `chunk_tokens.len() * size_of::<u32>()` over a live
            // `&[u32]`.
            let token_ids_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(chunk_tokens.as_ptr() as *const u8, chunk_len * 4)
            };
            let token_ids_dev = self.buffers.scratch(); // temporary, overwritten by MoE later
            self.gpu
                .copy_h2d_async(token_ids_bytes, token_ids_dev, stream)?;
            // Also stage this chunk's token IDs into the STABLE token_ids buffer
            // (scratch is reused by MoE routing). DeepSeek-V4 hash-MoE reads
            // `tid2eid[token_id]` per token in this same chunk order.
            self.gpu
                .copy_h2d_async(token_ids_bytes, self.buffers.token_ids(), stream)?;
            if self.has_ngram_embedding() {
                // THE chunked-prefill embed. n-gram hashes read behind the
                // chunk, so hand it the earlier tokens of the prompt as well.
                let cs = chunk_start.saturating_sub(self.ngram_lookbehind());
                self.embed_tokens_fused(
                    &tokens[cs..chunk_start + chunk_len],
                    chunk_len,
                    hidden_dst,
                    stream,
                )?;
            } else {
                ops::batched_embed(
                    self.gpu.as_ref(),
                    self.batched_embed_kernel,
                    token_ids_dev,
                    self.embed_tokens.weight,
                    hidden_dst,
                    chunk_len as u32,
                    h as u32,
                    stream,
                )?;
            }
            if std::env::var("ATLAS_DUMP_EMBED").ok().as_deref() == Some("1") {
                self.gpu.synchronize(stream)?;
                let offset = (chunk_len - 1) * h * 2;
                let mut buf = vec![0u8; h * 2];
                let _ = self.gpu.copy_d2h(hidden_dst.offset(offset), &mut buf);
                let v: Vec<f32> = buf
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                tracing::info!(
                    "ATLAS_EMBED post-batched_embed (chunk_start={}, last_tok_id={}): |x|={:.4} first5={:?}",
                    chunk_start,
                    tokens[chunk_start + chunk_len - 1],
                    n,
                    &v[..5]
                );
            }
            // Feature-2: overlay overridden vocab rows AFTER the gather, BEFORE
            // the embed scale (the override row is a raw embed row that must
            // also be scaled). `token_ids()` holds this chunk's ids (staged
            // above); uniform-active route (seq_slot NULL). No-op when no
            // overlay is installed.
            self.apply_embed_overlay(
                self.buffers.token_ids(),
                spark_runtime::gpu::DevicePtr(0),
                hidden_dst,
                chunk_len as u32,
                stream,
            )?;
            self.scale_embeddings(hidden_dst, chunk_len, stream)?;
            if std::env::var("ATLAS_DUMP_EMBED").ok().as_deref() == Some("1") {
                self.gpu.synchronize(stream)?;
                let offset = (chunk_len - 1) * h * 2;
                let mut buf = vec![0u8; h * 2];
                let _ = self.gpu.copy_d2h(hidden_dst.offset(offset), &mut buf);
                let v: Vec<f32> = buf
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                tracing::info!(
                    "ATLAS_EMBED post-scale_embeddings: |x|={:.4} first5={:?}",
                    n,
                    &v[..5]
                );
            }
        }

        // ── 1b. Overwrite image_pad token positions with vision encoder embeddings ──
        // Vision embeddings are pre-computed by prepare_vision_embed() and stored in
        // the VisionEncoder's buf_out buffer ([total_patches, out_hidden_size] BF16).
        {
            let pending = *self.vision_embed_patches.lock();
            // Log BEFORE the guard, and on every rank. Under TP the ranks each
            // embed the same tokens and all-reduce every layer, so a rank that
            // skips the splice keeps the raw pad-token embedding at exactly
            // the positions the other rank filled with the picture. Logging
            // only inside the guard cannot show that: the rank that never
            // splices is the one that stays silent.
            {
                let (ipad, vpad) = self.vision_pad_ids();
                let pads = tokens[chunk_start..chunk_start + chunk_len]
                    .iter()
                    .filter(|&&t| t == ipad || t == vpad)
                    .count();
                if pads > 0 {
                    tracing::info!(
                        "Vision splice: {pads} pad tokens in chunk, {pending} encoder rows available"
                    );
                }
            }
            if pending > 0
                && let Some(ve) = &self.vision_encoder
            {
                let chunk_tokens = &tokens[chunk_start..chunk_start + chunk_len];
                // EITHER pad token. Matching only the image one meant a
                // video's positions were skipped entirely — no encoder row was
                // copied over them, the hidden state kept the raw token
                // embedding, and the model described a featureless gray field
                // while every token count looked correct.
                let (image_pad, video_pad) = self.vision_pad_ids();
                // Co-dispatch: this request's slice starts at vision_row_base
                // in the shared packed buf_out (0 for the legacy single encode).
                let row_base = *self.vision_row_base.lock();
                let mut img_idx = 0usize; // pad-token count within the chunk
                for (i, &tok) in chunk_tokens.iter().enumerate() {
                    if tok == image_pad || tok == video_pad {
                        let src = ve
                            .scratch()
                            .buf_out
                            .offset((row_base + img_idx) * ve.out_hidden_size * 2);
                        let dst = hidden_dst.offset(i * h * elem_bytes);
                        self.gpu
                            .copy_d2d_async(src, dst, ve.out_hidden_size * 2, stream)?;
                        img_idx += 1;
                    }
                }
                // ATLAS_SPLICE_DUMP: the hidden chunk AFTER the overwrite.
                // The encoder dump proves what buf_out HOLDS; only this proves
                // what the language model actually RECEIVES — that every
                // encoder row reached a pad position, in order, at the right
                // magnitude relative to the text rows around it.
                if let Ok(path) = std::env::var("ATLAS_SPLICE_DUMP")
                    && !path.is_empty()
                {
                    self.gpu.synchronize(stream).ok();
                    let bytes = chunk_len * h * elem_bytes;
                    let mut host = vec![0u8; bytes];
                    if self.gpu.copy_d2h(hidden_dst, &mut host).is_ok() {
                        let _ = std::fs::write(&path, &host);
                        tracing::info!(
                            "ATLAS_SPLICE_DUMP: {chunk_len} x {h} ({elem_bytes} B/elem), \
                             {img_idx} pads spliced of {pending} encoder rows -> {path}"
                        );
                    }
                }
            }
        }

        Ok(())
    }
}
