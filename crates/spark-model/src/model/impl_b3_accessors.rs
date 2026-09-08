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
        // 🔴 ANOMALIES A59. `new()` sized `mtp_prefill_hidden` at `max_seq_len` because the
        // post-construction proposers (V4, GLM-5.3, DFlash) do not exist yet when it runs —
        // they need the model's owned GPU backend and its shared embed/lm_head. Now that one
        // is installed, ask it how many rows it can actually be handed and give back the rest.
        //
        // Keyed to the trait, not to a model name: a proposer that can follow the target to
        // the end of the served context returns `max_seq_len` (the default) and nothing
        // happens. GLM-5.3's drafter is a DSA block capped at `max_dsa_context`, so at
        // `--max-seq-len 524288` this returns 4.0 GiB of unreachable capture buffer that was
        // covered by no reserve at all (see `Glm5NextMtpHead::new`).
        //
        // Safe here and nowhere later: construction time, no sequence exists, so no capture is
        // in flight and no `mtp_prefill_capture_len` is live. Shrink only — a proposer must
        // never be able to GROW a buffer the capture epilogue already bounds-checks against.
        // 🪤 FREE the old buffer BEFORE allocating the small one. The obvious alloc-then-free
        // ordering holds both at once, and the peak it creates — 4.3 GB — is exactly the
        // pressure this is here to remove, on a box that has ~3 GB free at this point.
        let rows = proposer.prefill_hidden_rows(self.mtp_prefill_capacity);
        if !self.mtp_prefill_hidden.is_null() && rows < self.mtp_prefill_capacity {
            let was = self.mtp_prefill_capacity;
            let bytes = rows * self.config.hidden_size * 2;
            let old = std::mem::replace(
                &mut self.mtp_prefill_hidden,
                spark_runtime::gpu::DevicePtr::NULL,
            );
            self.mtp_prefill_capacity = 0;
            match self.gpu.free(old).and_then(|_| self.gpu.alloc(bytes)) {
                Ok(smaller) => {
                    self.mtp_prefill_hidden = smaller;
                    self.mtp_prefill_capacity = rows;
                    tracing::info!(
                        "MTP drafter context: capture buffer rightsized {was} -> {rows} rows \
                         ({:.0} -> {:.0} MB) — the proposer cannot be handed a position past \
                         {rows} (A59)",
                        (was * self.config.hidden_size * 2) as f64 / 1e6,
                        bytes as f64 / 1e6,
                    );
                }
                // NULL + capacity 0 is the feature's own "off" state: the capture epilogue
                // and the propose-site coverage check both gate on it, so drafter-prefill
                // disables and the serve keeps running at plain acceptance. Losing a
                // throughput feature beats failing a serve over an optimisation.
                Err(e) => tracing::warn!(
                    "MTP drafter context: rightsizing the capture buffer failed ({e:#}) — \
                     drafter prefill and carry are DISABLED for this serve"
                ),
            }
        }
        self.proposer = Some(proposer);
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
