// SPDX-License-Identifier: AGPL-3.0-only

//! The per-model load multipliers in [`super`], in both directions: the
//! tighter BF16 dense Qwen3.5 figure applies to that checkpoint only, only on
//! a target that declares `memory_fit_consumer`, and the models that already
//! had a figure keep it everywhere.

use super::{BF16_DENSE_QWEN35_PEAK, quant_multiplier_for};
use avarok_core::config::{ModelConfig, QuantizationConfig};

fn cfg(model_type: &str, experts: usize, quant: Option<&str>) -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = model_type.to_string();
    c.num_experts = experts;
    c.quantization_config = quant.map(|m| QuantizationConfig {
        quant_method: m.to_string(),
        quant_algo: String::new(),
        format: String::new(),
        ignore_modules: Vec::new(),
    });
    c
}

/// On a target that declares the consumer fits (ada).
fn ada(c: &ModelConfig, keep_nvfp4: bool) -> Option<f64> {
    quant_multiplier_for(c, keep_nvfp4, true)
}

#[test]
fn a_bf16_dense_qwen35_checkpoint_gets_the_measured_peak_on_ada() {
    let c = cfg("qwen3_5", 0, None);
    assert_eq!(ada(&c, false), Some(BF16_DENSE_QWEN35_PEAK));
    // The 4B on the 16 GB card: 8.68 GiB on disk, 14.84 GiB free, the
    // default 4 GiB guard. The generic 1.3x refused it; this admits it.
    let (disk, free, guard) = (8.68_f64, 14.84_f64, 4.0_f64);
    assert!(disk * 1.3 + guard > free);
    assert!(disk * BF16_DENSE_QWEN35_PEAK + guard <= free);
    // Still above the measured peak (8.80 GiB upload, 9.25 GiB with MTP).
    assert!(disk * BF16_DENSE_QWEN35_PEAK > 9.25);
}

#[test]
fn every_other_target_keeps_the_generic_default() {
    // GB10, Hopper, B200, B300: `memory_fit_consumer = false`.
    assert_eq!(
        quant_multiplier_for(&cfg("qwen3_5", 0, None), false, false),
        None
    );
    // ...and the figures other models already had are not the row's.
    assert_eq!(
        quant_multiplier_for(&cfg("glm5_next", 0, None), false, false),
        Some(1.10)
    );
    assert_eq!(
        quant_multiplier_for(&cfg("qwen3_5", 0, Some("fp8")), false, false),
        Some(1.05)
    );
}

#[test]
fn the_keep_nvfp4_hatch_keeps_the_generic_default() {
    assert_eq!(ada(&cfg("qwen3_5", 0, None), true), None);
}

#[test]
fn quantized_moe_and_other_checkpoints_are_unchanged() {
    // A declared quantization keeps the default (or its own figure).
    assert_eq!(ada(&cfg("qwen3_5", 0, Some("modelopt")), false), None);
    assert_eq!(ada(&cfg("qwen3_5", 0, Some("fp8")), false), Some(1.05));
    // MoE and other families: no change.
    assert_eq!(ada(&cfg("qwen3_5", 256, None), false), None);
    assert_eq!(ada(&cfg("qwen3_next", 0, None), false), None);
    assert_eq!(ada(&cfg("llama", 0, None), false), None);
    assert_eq!(ada(&cfg("glm5_next", 0, None), false), Some(1.10));
    assert_eq!(ada(&cfg("minimax_m2", 0, None), false), Some(1.02));
}
