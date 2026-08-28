// SPDX-License-Identifier: AGPL-3.0-only

//! Post-construction proposer-wiring accessors for [`TransformerModel`].
//! Split out of `impl_b3.rs` (500-LoC cap) — borrow/install hooks only.

use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

use super::types::TransformerModel;
use crate::speculative::DraftProposer;

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
