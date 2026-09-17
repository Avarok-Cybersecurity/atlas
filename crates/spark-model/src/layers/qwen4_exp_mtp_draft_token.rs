// SPDX-License-Identifier: AGPL-3.0-only

//! Drafting a token (batched and single, with and without grammar), split
//! out of `qwen4_exp_mtp.rs` to keep it under the 500-line cap.

use super::*;

impl Qwen4ExpMtpHead {
    /// Score `n` staged draft hiddens with ONE LM-head pass and ONE batched
    /// argmax, returning their token ids from a single D2H.
    ///
    /// The per-sequence `draft_token` streams the full-vocab NVFP4 head
    /// (~318 MB at this vocab) and drains the queue once PER DRAFT; at C=4,
    /// DRAFTS=2 that is 8 head reads and 8 drains a step for a weight that is
    /// the same for every sequence. Chunked at the batchm family's width.
    pub fn draft_tokens_batched(
        &self,
        n: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(
            (1..=BATCH_CAP).contains(&n),
            "draft_tokens_batched: n={n} (cap {BATCH_CAP})"
        );
        let vocab = ctx.config.vocab_size;
        let h = ctx.config.hidden_size;
        // NATIVE EXL3 FIRST, as the single-row `draft_token` does: under
        // `AVAROK_EXL3_NATIVE` there is no NVFP4 head to fall back to, and the
        // borrowed trellis head is the one the target samples from.
        if let Some(exl3) = self.lm_head_exl3.as_ref() {
            exl3.project_draft_rows(
                ctx.gpu,
                self.buf.batch_h_out,
                n,
                self.buf.batch_logits,
                stream,
            )?;
            return self.batched_argmax(n, vocab, ctx, stream);
        }
        let w = self.lm_head_nvfp4.as_ref().ok_or_else(|| {
            anyhow::anyhow!("draft_tokens_batched: no NVFP4 and no native-EXL3 lm_head")
        })?;
        let mut off = 0usize;
        while off < n {
            let take = (n - off).min(8);
            let k = self.w4a16_batchm.kernel(take as u32);
            anyhow::ensure!(
                k.0 != 0,
                "draft_tokens_batched: no batchm tier for {take} rows"
            );
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                k,
                self.buf.batch_h_out.offset(off * h * 2),
                w,
                self.buf.batch_logits.offset(off * vocab * 2),
                take as u32,
                vocab as u32,
                h as u32,
                stream,
            )?;
            off += take;
        }
        self.batched_argmax(n, vocab, ctx, stream)
    }

    /// ONE batched argmax over `[n, vocab]` draft logits and ONE D2H of the n
    /// token ids — shared by both head arms, so the NVFP4 and native-EXL3
    /// paths cannot drift in how they read a batch back.
    pub(super) fn batched_argmax(
        &self,
        n: usize,
        vocab: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>> {
        ops::argmax_bf16_batch(
            ctx.gpu,
            self.argmax_batch_k,
            self.buf.batch_logits,
            self.buf.batch_tok,
            vocab as u32,
            n as u32,
            vocab as u32,
            stream,
        )?;
        let mut b = vec![0u8; n * 4];
        ctx.gpu.copy_d2h(self.buf.batch_tok, &mut b)?;
        Ok(b.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Turn the draft's final hidden into a token id, entirely inside the
    /// draft's own arena.
    ///
    /// qwen4_exp sets `final_norm_identity`, so there is NO final norm here —
    /// the mHC head's own `hc_norm` plays that role. Applying one would be an
    /// uninvited extra RMS divide (a bug this model already shipped once).
    pub fn draft_token(&self, h_out: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<u32> {
        self.draft_token_with_grammar(h_out, ctx, stream, None)
    }

    /// Grammar state belongs to the scheduler. Only the first draft can use
    /// its current mask; later draft rows are checked by target verification.
    pub(in crate::layers) fn draft_token_with_grammar(
        &self,
        h_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let vocab = ctx.config.vocab_size as u32;
        let h = ctx.config.hidden_size as u32;
        let logits = self.arena.logits();
        // NATIVE EXL3 FIRST. Under `AVAROK_EXL3_NATIVE` there is no NVFP4 head
        // to fall back to, and the borrowed trellis head is the SAME head the
        // target samples from — which is the whole point of scoring a draft.
        // `project_draft` writes ONE row into the DRAFT's own arena using the
        // head's reserved scratch row, inside a section of the model-shared
        // `Exl3LaunchState`.
        if let Some(exl3) = self.lm_head_exl3.as_ref() {
            exl3.project_draft(ctx.gpu, h_out, logits, stream)?;
        } else {
            let w = self.lm_head_nvfp4.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "qwen4_exp MTP: no NVFP4 and no native-EXL3 lm_head for the draft head"
                )
            })?;
            // Single-warp GEMV per the model lever (bit-identical to the base
            // kernel, gemv_sw.rs): the draft LM head is the widest GEMV of the
            // step (vocab rows) and ran the base kernel alone.
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                ctx.levers.gemv_sw,
                h_out,
                w,
                logits,
                vocab,
                h,
                stream,
            )?;
        }
        if let Some(bitmask) = grammar_bitmask {
            return sampling::grammar_argmax(ctx.gpu, logits, vocab as usize, bitmask);
        }
        let out_ptr = self.arena.scratch();
        ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out_ptr, vocab, stream)?;
        let mut b = [0u8; 4];
        ctx.gpu.copy_d2h(out_ptr, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    /// Park the target's logits so the draft's `lm_head` cannot change what the
    /// model emits. MUST be paired with [`Self::restore_logits`].
    pub fn stash_logits(
        &self,
        gpu: &dyn GpuBackend,
        logits: DevicePtr,
        vocab: usize,
        stream: u64,
    ) -> Result<()> {
        gpu.copy_d2d_async(logits, self.buf.logits_stash, vocab * 2, stream)
    }

    /// Put the target's logits back after the draft has used the buffer.
    pub fn restore_logits(
        &self,
        gpu: &dyn GpuBackend,
        logits: DevicePtr,
        vocab: usize,
        stream: u64,
    ) -> Result<()> {
        gpu.copy_d2d_async(self.buf.logits_stash, logits, vocab * 2, stream)
    }

    /// Record a shadow observation and return the running accept rate.
    pub fn shadow_observe(&self, drafted: Option<u32>, actual: u32) {
        if let Some(d) = drafted {
            self.shadow_drafts.fetch_add(1, Ordering::Relaxed);
            if d == actual {
                self.shadow_hits.fetch_add(1, Ordering::Relaxed);
            }
            let n = self.shadow_drafts.load(Ordering::Relaxed);
            if n.is_multiple_of(16) {
                let hits = self.shadow_hits.load(Ordering::Relaxed);
                tracing::info!(
                    "qwen4_exp MTP shadow: {hits}/{n} drafts matched the target \
                     ({:.1}% accept). NO speculation is running — this measures \
                     whether the combiner reading is right.",
                    100.0 * hits as f64 / n as f64
                );
            }
        }
    }
}
