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
            if pending > 0
                && let Some(ve) = &self.vision_encoder
            {
                let chunk_tokens = &tokens[chunk_start..chunk_start + chunk_len];
                let pad_id = self
                    .config
                    .vision
                    .as_ref()
                    .map(|v| v.image_pad_token_id)
                    .filter(|v| *v != 0)
                    .unwrap_or(crate::layers::vision_encoder::IMAGE_PAD_TOKEN_ID);
                let mut img_idx = 0usize; // index into buf_out rows
                for (i, &tok) in chunk_tokens.iter().enumerate() {
                    if tok == pad_id {
                        let src = ve.buf_out.offset(img_idx * ve.out_hidden_size * 2);
                        let dst = hidden_dst.offset(i * h * elem_bytes);
                        self.gpu
                            .copy_d2d_async(src, dst, ve.out_hidden_size * 2, stream)?;
                        img_idx += 1;
                    }
                }
            }
        }

        Ok(())
    }
}
