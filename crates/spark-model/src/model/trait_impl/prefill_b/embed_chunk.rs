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
            if std::env::var("AVAROK_DUMP_EMBED").ok().as_deref() == Some("1") {
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
                    "AVAROK_EMBED post-batched_embed (chunk_start={}, last_tok_id={}): |x|={:.4} first5={:?}",
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
            if std::env::var("AVAROK_DUMP_EMBED").ok().as_deref() == Some("1") {
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
                    "AVAROK_EMBED post-scale_embeddings: |x|={:.4} first5={:?}",
                    n,
                    &v[..5]
                );
            }
        }

        // The vision splice is a SEPARATE method because it has to run from
        // TWO places, and for a long time it only ran from one. See
        // `prefill_b_splice_vision_at`.
        self.prefill_b_splice_vision_at(tokens, chunk_start, chunk_len, hidden_dst, stream)?;

        Ok(())
    }

    /// Overwrite this range's vision-pad rows with the encoder's patch
    /// embeddings. Split out of `prefill_b_embed_chunk_at` so the warm-prefix
    /// re-embed can apply it too.
    ///
    /// WHY THAT MATTERS: on a prefix-cache hit, `proc_range` RE-EMBEDS the
    /// uncached suffix into `hidden` at row 0 with a plain `batched_embed`,
    /// deliberately overwriting what phase 1 wrote. Phase 1 had already spliced
    /// the picture in at the pad rows of the FULL chunk; the re-embed then
    /// replaced those rows with the pads' raw token embeddings, and nothing put
    /// the picture back. The model processed a prompt whose image rows were
    /// literally the `<|image_pad|>` vocab vector, and answered fluently from
    /// the surrounding text — measured 2026-09-15: same prompt, temp 0, cold
    /// said "three chevron-like shapes ... light purple ... cyan" (correct) and
    /// warm said "a rectangular box with a lid ... briefcase". No test caught
    /// it because no test warmed a prefix and then asked about an image.
    ///
    /// The encoder row index is ABSOLUTE — seeded with the pads in
    /// `tokens[..chunk_start]` — so a range starting after earlier media still
    /// indexes its own patches. The caller must still never begin a range
    /// INSIDE a pad run, or that item's patches are split across two seeds.
    pub(in crate::model) fn prefill_b_splice_vision_at(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        hidden_dst: spark_runtime::gpu::DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let elem_bytes = 2usize;
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
            // ABSOLUTE pad count, not a per-range one. `img_idx` indexes the
            // encoder's packed output, which is ordered over the WHOLE prompt,
            // so it must be seeded with the pads that came BEFORE this range —
            // otherwise row 0 of buf_out is handed to whatever media happens to
            // start the range.
            //
            // Caught by video-fidelity's `video-before-image` leg: with the
            // range starting after a video's pads, the image's first pad took
            // encoder row 0 — the VIDEO's first patch — so the reading came
            // back [red, green, blue] (the clip) with the image's yellow
            // missing entirely. The same error applies to an image whose pad
            // run is split across two prefill chunks, where the second chunk
            // used to restart at row 0.
            //
            // chunk_start == 0 (the cold, full-chunk path) makes this 0, so
            // that path is byte-unchanged.
            let mut img_idx = super::upload_meta::mrope_pos::pad_rows_before(
                &tokens[..chunk_start],
                image_pad,
                video_pad,
            );
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
            // AVAROK_SPLICE_DUMP: the hidden chunk AFTER the overwrite.
            // The encoder dump proves what buf_out HOLDS; only this proves
            // what the language model actually RECEIVES — that every
            // encoder row reached a pad position, in order, at the right
            // magnitude relative to the text rows around it.
            if let Ok(path) = std::env::var("AVAROK_SPLICE_DUMP")
                && !path.is_empty()
            {
                self.gpu.synchronize(stream).ok();
                let bytes = chunk_len * h * elem_bytes;
                let mut host = vec![0u8; bytes];
                if self.gpu.copy_d2h(hidden_dst, &mut host).is_ok() {
                    let _ = std::fs::write(&path, &host);
                    tracing::info!(
                        "AVAROK_SPLICE_DUMP: start={chunk_start} rows={chunk_len} x {h} \
                             ({elem_bytes} B/elem), {img_idx} pads spliced of {pending} \
                             encoder rows -> {path}"
                    );
                }
            }
        }

        Ok(())
    }
}
