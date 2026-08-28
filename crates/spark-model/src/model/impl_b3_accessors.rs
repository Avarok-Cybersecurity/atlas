// SPDX-License-Identifier: AGPL-3.0-only

//! Post-construction proposer-wiring accessors for [`TransformerModel`].
//! Split out of `impl_b3.rs` (500-LoC cap) — borrow/install hooks only.

use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

use super::types::TransformerModel;
use crate::speculative::DraftProposer;
// `argmax_on_device` lives on the Model trait; the shadow step calls it.
use crate::traits::Model;

impl TransformerModel {
    /// Borrow the GPU backend for post-construction wiring (e.g. installing
    /// a DFlash proposer that needs to allocate paged KV caches against the
    /// same GPU the target uses).
    pub fn gpu_backend(&self) -> &dyn GpuBackend {
        self.gpu.as_ref()
    }

    /// Borrow the model config for post-construction wiring (e.g. building the
    /// DeepSeek-V4 MTP proposer, which needs `hidden_size` / `kv_lora_rank` /
    /// `qk_rope_head_dim` to size its private MLA KV cache).
    pub fn config_ref(&self) -> &ModelConfig {
        &self.config
    }

    /// Install a DFlash drafter as the active proposer, replacing whatever
    /// MTP proposer (if any) `TransformerModel::new` built. The target's
    /// hidden-state capture buffer is already allocated when the config's
    /// `dflash_capture_layers` is non-empty (factory.rs populates it before
    /// construction), so this method only swaps the proposer slot.
    ///
    /// Mutually exclusive with `--speculative` MTP at the CLI level
    /// (clap `conflicts_with`); this method does not enforce that — the
    /// caller is expected to have validated the flag combination already.
    pub fn set_dflash_proposer(&mut self, proposer: std::sync::Arc<dyn DraftProposer>) {
        if self.proposer.is_some() {
            tracing::info!("DFlash: replacing existing MTP proposer with BlockDiffusionDraftHead");
        }
        self.proposer = Some(proposer);
    }

    /// Take ownership of a loaded qwen4_exp MTP draft module.
    ///
    /// This is NOT decoration. `DevicePtr` has no `Drop`, so a module that is
    /// built and then dropped leaks its quantized MoE and attention buffers.
    /// Nothing reads the field yet: no proposer consumes it, so
    /// `has_proposer()` stays false and speculation stays off.
    pub fn set_qwen4_exp_mtp(
        &mut self,
        module: Box<crate::weight_loader::qwen4_exp::Qwen4ExpMtpModule>,
    ) {
        tracing::info!(
            "qwen4_exp MTP module installed on the served model (held so its \
             device buffers are not leaked; no proposer consumes it yet)"
        );
        self.qwen4_exp_mtp = Some(module);
    }

    /// Install the qwen4_exp MTP draft head for SHADOW measurement.
    ///
    /// The head owns the module, so this replaces `set_qwen4_exp_mtp` rather
    /// than accompanying it. Installing it does NOT enable speculation:
    /// `has_proposer()` is unaffected and nothing feeds a draft back into the
    /// sequence. It only lets the decode path ask "would this draft have been
    /// right?" and count.
    pub fn set_qwen4_exp_mtp_head(
        &mut self,
        module: crate::weight_loader::qwen4_exp::Qwen4ExpMtpModule,
        embed_tokens: crate::weight_map::DenseWeight,
        max_seq_len: usize,
    ) -> anyhow::Result<()> {
        // Built here rather than in the factory because the model owns `gpu`
        // by this point.
        let head = crate::layers::qwen4_exp_mtp::Qwen4ExpMtpHead::new(
            module,
            embed_tokens,
            // Share the target's NVFP4 vocab head (Copy pointers) so the draft
            // goes through the SAME quantization ladder the real token does —
            // a drafter scored against a different head measures the head, not
            // the draft.
            self.lm_head_nvfp4,
            &self.config,
            self.gpu.as_ref(),
            max_seq_len,
        )?;
        let state = head.alloc_state(self.gpu.as_ref())?;
        tracing::warn!(
            "qwen4_exp MTP SHADOW MODE is on: the draft head runs every decode \
             step and its accept rate is logged. Verified INERT — shadow-on \
             output is byte-identical to a shadow-off control — because the \
             draft runs in its own BufferArena. It still costs a FULL EXTRA \
             draft forward per token and produces NO speedup (nothing is fed \
             back), so do not benchmark with it on."
        );
        self.qwen4_exp_mtp_head = Some(head);
        self.qwen4_exp_mtp_state = Some(std::sync::Mutex::new(state));
        Ok(())
    }

    /// Run one shadow draft step: score the previous step's draft against the
    /// token the target just produced, then draft the next one.
    ///
    /// `target_streams` is the four-stream mHC highway for the row that
    /// produced `actual_token`. No-op unless shadow mode is installed.
    pub(super) fn qwen4_exp_mtp_shadow_step(
        &self,
        actual_token: u32,
        target_streams: spark_runtime::gpu::DevicePtr,
        position: usize,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) {
        let (Some(head), Some(state)) = (&self.qwen4_exp_mtp_head, &self.qwen4_exp_mtp_state)
        else {
            return;
        };
        use crate::layers::qwen4_exp_mtp::{ShadowStage, shadow_stage};
        let Ok(mut st) = state.lock() else { return };
        // Score the draft made LAST step against what the target actually just
        // emitted. This is the whole measurement.
        head.shadow_observe(st.pending_draft.take(), actual_token);

        // BISECTION stage 1: count only. If the target's output is DIRTY even
        // here, the corruption is the extra `argmax_on_device` itself (it writes
        // `buffers.scratch()` and runs on the DEFAULT stream, not this one) —
        // not the draft forward.
        if shadow_stage() == ShadowStage::Observe {
            return;
        }

        // Draft the NEXT token from (token just emitted, this row's highway).
        let h_out = self.mtp_hidden_save;
        if let Err(e) = head.draft_hidden(
            actual_token,
            target_streams,
            position,
            &mut st,
            h_out,
            ctx,
            stream,
        ) {
            // Shadow is diagnostic: never fail the real decode over it.
            tracing::warn!("qwen4_exp MTP shadow draft failed, disabling: {e:#}");
            return;
        }
        // BISECTION stage 3: stop before the draft's own LM head.
        if shadow_stage() < ShadowStage::Full {
            return;
        }

        // The draft's logits go into the DRAFT's arena, so nothing here can
        // touch the buffer the scheduler is about to sample from. The earlier
        // stash/restore of the target's logits existed only because the draft
        // borrowed the target's LM-head buffer; private-arena isolation removed
        // the need for it entirely.
        match head.draft_token(h_out, ctx, stream) {
            Ok(d) => st.pending_draft = Some(d),
            Err(e) => tracing::warn!("qwen4_exp MTP shadow draft head failed: {e:#}"),
        }
    }

    /// Install the fused n-gram input embedding (LongCat family). Once set,
    /// every embedding site routes through it instead of the plain
    /// `embed_tokens` gather.
    pub fn set_ngram_embedding(&mut self, ngram: crate::layers::ngram_embed::NgramEmbedding) {
        tracing::info!("set_ngram_embedding: installed on the served model");
        self.ngram_embed = Some(std::sync::Mutex::new(ngram));
    }

    /// True when this model fuses n-gram lookups into its input embedding.
    pub fn has_ngram_embedding(&self) -> bool {
        self.ngram_embed.is_some()
    }

    /// True when MLA prefill cannot honour a prefix-cache skip.
    ///
    /// `paged_mla`'s flash call is fed the K/V it just assembled — its own
    /// comment says "not from paged cache" — so it attends ONLY over the
    /// tokens being processed. For a full prompt that is correct, and it is
    /// how every MLA model has been exercised. With a SKIPPED prefix it is
    /// not: the cached tokens are simply absent from attention and the model
    /// answers from the tail of its prompt, fluently and wrongly.
    ///
    /// MLA keeps a COMPRESSED (latent) KV cache, so letting this path attend
    /// over history means absorbed attention against that cache, not a wider
    /// gather. Until that exists, decline the SKIP rather than the cache:
    /// prefix caching stays on and correct — block reuse and the decode path
    /// still benefit — and prefill pays full price.
    ///
    /// ATLAS_MLA_PREFIX_SKIP=1 opts back in once `paged_mla` attends the cache.
    pub(crate) fn mla_prefill_needs_full_recompute(&self) -> bool {
        if std::env::var("ATLAS_MLA_PREFIX_SKIP").as_deref() == Ok("1") {
            return false;
        }
        self.layers.iter().any(|l| l.uses_local_mla_prefill())
    }
}
