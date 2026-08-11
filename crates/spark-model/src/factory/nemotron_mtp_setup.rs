// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron MTP proposer setup — the two `build_model` hooks for the
//! Nemotron-3.5 Lightning DeepSeek-style draft head, extracted from
//! `build.rs` for the 500-LoC file cap (pure code move + gating).
//!
//! Load happens PRE-construction (the module borrows `store` and `gpu`
//! before they move into the model); install happens POST-construction
//! (the proposer needs the model's owned GPU backend for its private KV
//! cache), mirroring the DeepSeek-V4 MTP flow in `build.rs` Step 6b.

use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use atlas_core::config::ModelConfig;

use crate::model::TransformerModel;
use crate::weight_loader::nemotron::{NemotronMtpModule, load_nemotron_mtp_module};
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// Load the Nemotron MTP module when this build wants it: nemotron_h family,
/// `--speculative`, rank 0 (the draft runs on rank 0 only — see the V4
/// comment in `build.rs`). Load failures disable MTP with an error log
/// rather than failing the whole model build.
pub(super) fn maybe_load_nemotron_mtp(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    use_speculative: bool,
) -> Option<NemotronMtpModule> {
    if !(config.model_type.starts_with("nemotron_h") && use_speculative && config.ep_rank == 0) {
        return None;
    }
    match load_nemotron_mtp_module(store, config, gpu) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!("Nemotron MTP module load FAILED: {e:#}");
            None
        }
    }
}

/// Build the `NemotronMtpHead` proposer from a loaded module and install it
/// on the model. No-op when no module was loaded; build failures disable
/// speculative decoding with a warning (never fatal — the target model is
/// fully usable without the drafter).
pub(super) fn maybe_install_nemotron_mtp(
    model: &mut TransformerModel,
    module: Option<NemotronMtpModule>,
    shared_embed: DenseWeight,
    draft_lm_head_nvfp4: Option<QuantizedWeight>,
    mtp_vocab_size: u32,
    max_seq_len: usize,
) {
    let Some(module) = module else { return };
    let Some(draft_head) = draft_lm_head_nvfp4 else {
        tracing::warn!(
            "Nemotron MTP module loaded but no NVFP4 LM head available — \
             speculative decoding disabled."
        );
        return;
    };
    match crate::layers::NemotronMtpHead::new(
        module,
        shared_embed,
        draft_head,
        model.config_ref(),
        model.gpu_backend(),
        mtp_vocab_size,
        max_seq_len,
    ) {
        Ok(head) => {
            model.set_dflash_proposer(std::sync::Arc::new(head));
            tracing::info!(
                "Nemotron MTP speculative decoding: ENABLED (1-step DeepSeek-style head)"
            );
        }
        Err(e) => tracing::warn!(
            "Failed to build Nemotron MTP proposer: {e:#}. Speculative decoding disabled."
        ),
    }
}
