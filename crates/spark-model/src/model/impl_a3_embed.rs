// SPDX-License-Identifier: AGPL-3.0-only

//! Single-token embedding for the decode paths.
//!
//! `embed` REFUSES a bare token id on an n-gram model: 12 of the 13
//! contributions are hashed from the token's predecessors, so embedding
//! without context would silently drop most of the signal — a failure that
//! reads as fluent degenerate text, not as an error. Callers holding the
//! sequence use `embed_ctx`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::TransformerModel;

impl TransformerModel {
    pub(super) fn embed(&self, token: u32, output: DevicePtr, stream: u64) -> Result<()> {
        // A bare token id cannot produce a correct embedding for an n-gram
        // model: 12 of the 13 contributions are hashed from the token's
        // PREDECESSORS. Refuse loudly rather than emit a plausible-looking
        // embedding that is missing most of its signal — that failure mode
        // reads as fluent degenerate text, not as a bug. Callers that have
        // the sequence in hand should use `embed_ctx`.
        anyhow::ensure!(
            self.ngram_embed.is_none(),
            "embed(): this model fuses n-gram lookups into its input embedding,              so embedding a token without its preceding context would silently              drop 12/13 of the signal. This code path has not been given the              sequence's tokens yet — use embed_ctx()."
        );
        self.embed_row(token, output, stream)
    }

    /// The plain single-row gather, with no n-gram guard. Shared by `embed`
    /// and by `embed_ctx`'s non-n-gram arm.
    fn embed_row(&self, token: u32, output: DevicePtr, stream: u64) -> Result<()> {
        let h = self.config.hidden_size;
        let row_bytes = h * 2; // BF16 embedding row
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        self.gpu.copy_d2d_async(src, output, row_bytes, stream)?;
        // Scale embeddings (Gemma-4: sqrt(hidden_size))
        self.scale_embeddings(output, 1, stream)
    }

    /// Embed ONE token that is preceded by `history` in its sequence.
    /// `history` must NOT already include `token`.
    pub(super) fn embed_ctx(
        &self,
        history: &[u32],
        token: u32,
        output: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if self.ngram_embed.is_none() {
            return self.embed_row(token, output, stream);
        }
        let lb = self.ngram_lookbehind();
        let tail = &history[history.len().saturating_sub(lb)..];
        let mut ctx = Vec::with_capacity(tail.len() + 1);
        ctx.extend_from_slice(tail);
        ctx.push(token);
        self.embed_tokens_fused(&ctx, 1, output, stream)?;
        self.scale_embeddings(output, 1, stream)
    }
}
