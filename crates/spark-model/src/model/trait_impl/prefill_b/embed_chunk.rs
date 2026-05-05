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
        let h = self.config.hidden_size;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let hidden = self.buffers.hidden_states();

        // ── 1. Embed chunk tokens → [chunk_len, H] contiguous ──
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
            ops::batched_embed(
                self.gpu.as_ref(),
                self.batched_embed_kernel,
                token_ids_dev,
                self.embed_tokens.weight,
                hidden,
                chunk_len as u32,
                h as u32,
                stream,
            )?;
            self.scale_embeddings(hidden, chunk_len, stream)?;
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
                        let dst = hidden.offset(i * h * fp32);
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
