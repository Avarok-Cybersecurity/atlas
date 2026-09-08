// SPDX-License-Identifier: AGPL-3.0-only

//! The two KV-completeness capability gates, tested together because they
//! start from one question — does this model's prefill build per-sequence
//! state that KV blocks do not carry? — and DIVERGE on exactly one arm.
//!
//! The prefix cache is not KV-only: a radix node travels with a Marconi
//! snapshot (SSM pool slot + per-layer aux blobs), so a model whose non-KV
//! state rides that snapshot may use it. The `--swap-space-gb` spill image IS
//! KV-only (KV blocks + LinearAttention `SsmLayerState`, no aux record), so
//! the same model must still refuse it. GLM-5.3 is that model today. The
//! symmetry the old header promised ("one question, both gates") is broken
//! deliberately, via `non_kv_state_is_marconi_snapshottable`, and only there.
//!
//! Split out of `methods.rs` (the 500-LoC cap) rather than grown in place, and
//! held in one file so a new model type cannot be taught to one gate and
//! forgotten by the other.

use crate::config::ModelConfig;

/// Model types whose prefill owns state outside KV AND whose Marconi snapshot
/// carries all of it. Prefix cache open, swap-out closed.
const MARCONI_SNAPSHOTTABLE: [&str; 2] = ["glm5_next", "glm5_next_text"];

#[test]
fn any_compressed_deepseek_v4_layer_is_not_kv_cache_complete() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "deepseek_v4".to_string();

    for ratios in [vec![4, 0, 0], vec![0, 4, 0], vec![0, 0, 128]] {
        config.compress_ratios = ratios;
        // No aux carry exists for the compressor pool/ring, so BOTH stay closed.
        assert!(!config.kv_only_prefix_cache_is_safe());
        assert!(!config.kv_only_swap_out_is_safe());
    }
}

/// The one deliberate asymmetry. Flipping swap-out here would be the unsafe
/// chain: `resolve_swap_space_gb` → `KvSpillManager` → `spill_out_sequence` →
/// a KV+SsmLayerState image → `release_state` frees the DSA indexer → the
/// swap-in re-allocates it at zero rows behind a populated KV image.
#[test]
fn glm_prefix_cache_opens_via_marconi_but_swap_out_stays_closed() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();

    for model_type in MARCONI_SNAPSHOTTABLE {
        config.model_type = model_type.to_string();
        assert!(
            config.kv_only_prefix_cache_is_safe(),
            "{model_type}: KDA rides the SSM slot, DSA rides snapshot_aux — the radix \
             cache may be installed"
        );
        assert!(
            !config.kv_only_swap_out_is_safe(),
            "{model_type}: the spill image carries no aux record; swap-out must stay closed"
        );
    }
}

#[test]
fn kv_complete_models_keep_both_capabilities() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    assert!(config.kv_only_prefix_cache_is_safe());
    assert!(config.kv_only_swap_out_is_safe());

    config.model_type = "deepseek_v4".to_string();
    config.compress_ratios = vec![0; 3];
    assert!(config.kv_only_prefix_cache_is_safe());
    assert!(config.kv_only_swap_out_is_safe());
}
