// SPDX-License-Identifier: AGPL-3.0-only

//! The one place that decides how input tokens become hidden states.
//!
//! Most architectures gather a row of `embed_tokens` per token id. The
//! LongCat family instead FUSES that row with 12 hashed n-gram lookups whose
//! ids are a function of the token's PREDECESSORS, so an embedding site needs
//! the sequence's context — not just the id being embedded. Routing every
//! site through here keeps that distinction in a single place; a site that
//! bypasses it silently drops 12/13 of the embedding signal, which reads as a
//! model that produces fluent-looking degenerate text rather than as a crash.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::TransformerModel;
use crate::layers::ops;

impl TransformerModel {
    /// Embed `seq_len` tokens into `out` (`[seq_len, hidden]` BF16).
    ///
    /// `ctx_tokens` must end with the tokens being embedded and may be
    /// preceded by up to `neighbor_num - 1` earlier tokens of the same
    /// sequence — the n-gram hash reads backwards across that boundary. When
    /// no history is available (start of sequence) the reference's own
    /// shift-with-EOS-reset rule fills in, so passing just the new tokens is
    /// correct there and only there.
    pub(crate) fn embed_tokens_fused(
        &self,
        ctx_tokens: &[u32],
        seq_len: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if self.gpu.op_cache().first_n("diag:ngram_embed_arm", 3) {
            tracing::info!(
                "embed_tokens_fused: ngram={} ctx_len={} seq_len={}",
                self.ngram_embed.is_some(),
                ctx_tokens.len(),
                seq_len
            );
        }
        if let Some(ngram) = self.ngram_embed.as_ref() {
            let mut ng = ngram
                .lock()
                .map_err(|_| anyhow::anyhow!("ngram embedding mutex poisoned"))?;
            return ng.embed(ctx_tokens, seq_len, out, self.gpu.as_ref(), stream);
        }
        // Plain path: gather the last `seq_len` ids.
        let ids = &ctx_tokens[ctx_tokens.len() - seq_len..];
        let bytes: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        let ids_dev = self.buffers.scratch();
        self.gpu.copy_h2d_async(&bytes, ids_dev, stream)?;
        ops::batched_embed(
            self.gpu.as_ref(),
            self.batched_embed_kernel,
            ids_dev,
            self.embed_tokens.weight,
            out,
            seq_len as u32,
            self.config.hidden_size as u32,
            stream,
        )
    }

    /// How many earlier tokens the n-gram hash reads behind the first token
    /// being embedded. Zero when this model has no n-gram embedding, so
    /// callers can build the context slice unconditionally.
    pub(crate) fn ngram_lookbehind(&self) -> usize {
        match self.ngram_embed.as_ref() {
            Some(ng) => ng
                .lock()
                .map(|g| g.dims.neighbor_num.saturating_sub(1))
                .unwrap_or(0),
            None => 0,
        }
    }
}
