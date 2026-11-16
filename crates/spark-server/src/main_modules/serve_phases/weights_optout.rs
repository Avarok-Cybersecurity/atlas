// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint tensors a serve's FLAGS make unreachable, so the loaders need
//! not upload them. The per-model opt-ins (`skip_mtp`, `binds_vision`) answer
//! "can this port ever read it"; these answer "will this serve".

use avarok_core::config::ModelConfig;

use crate::cli;

/// No speculative mode is on, so nothing reads the checkpoint's `mtp.*` block.
///
/// Scoped to the Qwen3.5 families, whose loaders bind the block only through
/// `load_mtp_weights`, which returns `None` when `mtp.fc.weight` is absent
/// (`weight_loader/qwen35.rs`, `qwen35_dense.rs`), and whose model builds no
/// MTP head without `--speculative` ("MTP weights available ... but
/// --speculative not set, skipping MTP head construction"). Every speculative
/// flag keeps the block, including the ones that do not draft with it
/// (`--self-speculative`, `--ngram-speculative`, `--dflash`): the saving is
/// for the plain serve, and none of those is one.
///
/// On Qwen3.5-4B the block is 0.225 GiB that stayed resident for the life of
/// a plain serve. No numerics change: the target forward never reads it.
///
/// Only on a target that declares the `memory_fit_consumer` row (ada): every
/// other target uploads the block exactly as before.
pub(crate) fn mtp_unused(args: &cli::ServeArgs, config: &ModelConfig) -> bool {
    mtp_unused_for(
        args,
        config,
        spark_model::layers::ops::target_defaults::resolved()
            .memory_fit_consumer
            .value,
    )
}

/// [`mtp_unused`] with the target row passed in.
pub(crate) fn mtp_unused_for(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    consumer_fit: bool,
) -> bool {
    let any_spec =
        args.speculative || args.self_speculative || args.ngram_speculative || args.dflash;
    consumer_fit && !any_spec && qwen35_family(config)
}

fn qwen35_family(config: &ModelConfig) -> bool {
    let family = config.model_type.to_lowercase().replace(['-', '.'], "_");
    matches!(
        family.as_str(),
        "qwen3_5_moe" | "qwen3_5" | "qwen35_moe" | "qwen35"
    )
}

#[cfg(test)]
#[path = "weights_optout_tests.rs"]
mod tests;
