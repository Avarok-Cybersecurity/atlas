// SPDX-License-Identifier: AGPL-3.0-only

//! The per-model opt-ins in [`super`] that WITHHOLD checkpoint tensors.
//!
//! 🔴 Each one is an allow-list because withholding a tensor a loader does
//! read is invisible until the output is subtly wrong. A test that only
//! asserted the members would not notice the list turning into a default, so
//! these pin both directions.

use super::{skip_activation_scales, skip_mtp};
use avarok_core::config::ModelConfig;

fn cfg(model_type: &str) -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = model_type.to_string();
    c
}

/// `glm5_next` was added 2026-09-21: `nvidia/GLM-5.3-Flash-NVFP4` ships ~19k
/// `*.input_scale` scalars and the GLM port reads none of them — its routed
/// experts are `Nvfp4Proj` (packed / block scales / `weight_scale_2`) and its
/// dense MLP is host-dequantised, so nothing in it ever builds the
/// `QuantizedWeight` that carries an `input_scale` field.
#[test]
fn activation_scales_are_skipped_only_for_the_listed_models() {
    assert!(skip_activation_scales(&cfg("glm5_next")));
    assert!(skip_activation_scales(&cfg("qwen4_exp")));

    // 🪤 `step3p7` reads `input_scale` on its own loader path. It must never
    // join this list, and neither may anything unlisted.
    for keep in ["step3p7", "qwen3_5_moe", "minimax_m2", "kimi_k3", "llama"] {
        assert!(
            !skip_activation_scales(&cfg(keep)),
            "{keep} must keep its activation scales"
        );
    }
}

/// The two opt-ins are independent: adding a model to one must not enrol it in
/// the other. `glm5_next` DOES build an MTP head (`layers.45`), so skipping
/// `mtp.*` for it would be a different bug entirely.
#[test]
fn the_mtp_skip_list_is_unchanged_by_the_activation_scale_list() {
    assert!(skip_mtp(&cfg("qwen4_exp")));
    assert!(!skip_mtp(&cfg("glm5_next")));
}
