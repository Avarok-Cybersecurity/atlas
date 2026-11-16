// SPDX-License-Identifier: AGPL-3.0-only

//! The checkpoint loaders' peak-memory multiplier, per model.
//!
//! Every safetensors loader refuses a load whose on-disk bytes times this
//! multiplier, plus `--oom-guard-mb`, exceed free device memory. `None` takes
//! the loaders' generic default: 1.3x, or 1.5x when the checkpoint carries FP8
//! tensors. That default models the model BUILD, not the upload (the fast
//! loader's staging is a bounded read/copy pipeline on the host): transposed
//! prefill copies, NVFP4 copies of a quantized checkpoint, and transient BF16
//! intermediates, all allocated while the store still holds the originals
//! (`spark_runtime::weights::loader`, the pre-flight comment).

use avarok_core::config::ModelConfig;

/// Peak over on-disk bytes for a plain BF16 dense Qwen3.5 checkpoint.
///
/// WHY not 1.3. The dense Qwen3.5 loader keeps a BF16 checkpoint's FFN as
/// BF16 (no NVFP4 copies unless `AVAROK_DENSE_FP8_KEEP_NVFP4` asks for them),
/// and where it does derive a copy (the attention NVFP4 weights, the load-time
/// BF16 -> FP8 FFN route) it frees the BF16 source inside the same layer, so
/// the build's transient is one layer wide. Measured on an RTX 4060 Ti 16 GB,
/// Qwen3.5-4B, 8.68 GiB on disk: the upload peaks at 8.80 GiB (1.014x), the
/// layer build adds 0.023 GiB with BF16 FFN and frees 2.04 GiB with the FP8
/// FFN, and the MTP head and its draft LM head add 0.43 GiB before the prune,
/// 1.066x at most. 1.10 keeps a margin over that. The generic 1.3x projected
/// 11.28 GiB, which with the default 4 GiB guard exceeded the card's
/// 14.84 GiB free and refused a load that peaks at 9.25 GiB.
pub(crate) const BF16_DENSE_QWEN35_PEAK: f64 = 1.10;

pub(crate) fn quant_multiplier(config: &ModelConfig) -> Option<f64> {
    quant_multiplier_for(
        config,
        spark_model::weight_loader::fp8_residency::keep_nvfp4_fallback(),
        spark_model::layers::ops::target_defaults::resolved()
            .memory_fit_consumer
            .value,
    )
}

/// [`quant_multiplier`] with its environment inputs passed in, so the table
/// is testable without a process-global cell. `consumer_fit` is the
/// `memory_fit_consumer` target row (ada only): every other target keeps the
/// generic default for this checkpoint class.
pub(crate) fn quant_multiplier_for(
    config: &ModelConfig,
    keep_nvfp4: bool,
    consumer_fit: bool,
) -> Option<f64> {
    if config.model_type == "glm5_next" {
        // GLM-5.3 at TP=2/EP=2 is 99.64 GB on-disk per rank against ~111.5 GB free. The
        // generic 1.3x default projects 129.54 GB and refuses a load that fits: the fast
        // loader's staging is a BOUNDED read/copy pipeline, not a fraction of the model, so
        // on a ~100 GB checkpoint the 1.3x is 30 GB of imaginary overhead.
        //
        // ⚠️ 1.10 is a HEADROOM ALLOWANCE, not a measurement: nothing has profiled this
        // load yet. It also does NOT cover what comes after: `Glm5NextWeightLoader` builds
        // TP-sliced copies of the non-expert tensors while the store still holds the
        // originals (the routed experts are bound zero-copy from the store's own pointers,
        // so they are not doubled). That is ~6 GB this pre-flight cannot see. If a load OOMs
        // shortly AFTER this gate passes, that is the cause, not this number.
        Some(1.10)
    } else if config.model_type == "minimax_m2" || config.model_type == "step3p7" {
        Some(1.02)
    } else if config
        .quantization_config
        .as_ref()
        .is_some_and(|qc| qc.quant_method == "fp8")
    {
        Some(1.05)
    } else if consumer_fit && bf16_dense_qwen35(config) && !keep_nvfp4 {
        // `AVAROK_DENSE_FP8_KEEP_NVFP4` builds the NVFP4 FFN and its transposed
        // twins beside a BF16 snapshot, which is the build overhead the generic
        // default exists for, so that hatch keeps it.
        Some(BF16_DENSE_QWEN35_PEAK)
    } else {
        None
    }
}

/// A dense Qwen3.5 checkpoint that declares no quantization: the pre-load
/// view of the dense loader's `Nvfp4Variant::Bf16Raw` arm (the variant itself
/// is detected from tensor names, which the pre-flight has not read yet). A
/// checkpoint that declares any `quantization_config` keeps the default.
fn bf16_dense_qwen35(config: &ModelConfig) -> bool {
    config.is_qwen35_dense() && config.quantization_config.is_none()
}

#[cfg(test)]
#[path = "weights_multiplier_tests.rs"]
mod tests;
