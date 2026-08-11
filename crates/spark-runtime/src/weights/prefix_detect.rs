// SPDX-License-Identifier: AGPL-3.0-only

//! Weight-key prefix auto-detection for nested (multimodal) checkpoints.
//!
//! Split out of `weights.rs` for the 500-LoC cap — that file sat at exactly
//! 500 after the Q2 keep-packed additions, and the `check_oom_guard` export
//! fix re-tripped it; this move buys real headroom instead of shaving to the
//! line again.

use super::WeightStore;

/// Resolve the weight-key prefix a nested (multimodal) checkpoint uses.
///
/// Lives here rather than in the server because it depends only on the store
/// and the config, and BOTH the serve path and the integration harness need it:
/// a nested checkpoint stores `model.language_model.layers.0.…`, and a caller
/// that skips this step fails with "Weight 'model.layers.0.input_layernorm.
/// weight' not found in store" after a full weight load. Keeping one copy is
/// what stops the test from accepting a different set of checkpoints than
/// production does.
pub fn auto_detect_weight_prefix(
    store: &WeightStore,
    config: &mut atlas_core::config::ModelConfig,
) {
    if config.weight_prefix.is_empty() && config.nested_config {
        config.weight_prefix = if store.contains("language_model.model.embed_tokens.weight") {
            "language_model.model".to_string()
        } else if store.contains("model.language_model.embed_tokens.weight") {
            "model.language_model".to_string()
        } else {
            let scanned = store
                .names()
                .find(|k| k.contains(".layers.0."))
                .and_then(|k| k.split(".layers.0.").next())
                .map(|s| s.to_string());
            if let Some(ref prefix) = scanned {
                tracing::info!("Auto-detected weight prefix: '{prefix}'");
            }
            scanned.unwrap_or_else(|| "model".to_string())
        };
    }
    if !config.weight_prefix.is_empty() {
        tracing::info!("Weight prefix: {}", config.weight_prefix);
    }
}
