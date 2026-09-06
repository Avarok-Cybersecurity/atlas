// SPDX-License-Identifier: AGPL-3.0-only

//! The KV-completeness capability gate: does this model's prefill build
//! per-sequence state that KV blocks do not carry?
//!
//! Split out of `methods.rs` (the 500-LoC cap) rather than grown in place.

use crate::config::ModelConfig;

/// Model types whose prefill owns state outside KV.
const NOT_KV_COMPLETE: [&str; 2] = ["glm5_next", "glm5_next_text"];

#[test]
fn any_compressed_deepseek_v4_layer_is_not_kv_cache_complete() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "deepseek_v4".to_string();

    for ratios in [vec![4, 0, 0], vec![0, 4, 0], vec![0, 0, 128]] {
        config.compress_ratios = ratios;
        assert!(!config.kv_only_prefix_cache_is_safe());
    }
}

#[test]
fn glm_prompt_built_dsa_state_is_not_kv_cache_complete() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();

    for model_type in NOT_KV_COMPLETE {
        config.model_type = model_type.to_string();
        assert!(
            !config.kv_only_prefix_cache_is_safe(),
            "{model_type}: prefix cache must stay closed"
        );
    }
}

#[test]
fn kv_complete_models_can_use_the_prefix_cache() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    assert!(config.kv_only_prefix_cache_is_safe());

    config.model_type = "deepseek_v4".to_string();
    config.compress_ratios = vec![0; 3];
    assert!(config.kv_only_prefix_cache_is_safe());
}
