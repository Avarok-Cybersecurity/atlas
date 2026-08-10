// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 E2B (everything-to-BF16) weight loader: per-layer-embedding (PLE)
//! tables + the double-wide MLP geometry helper.
//!
//! E2B checkpoints (`hidden_size_per_layer_input > 0`) ship extra tensors
//! under the `model.language_model.` prefix:
//!   - `embed_tokens_per_layer.weight` [vocab, num_layers * per_layer_dim] —
//!     a per-layer embedding table whose row for token `t` at layer `i` is
//!     columns `[i*per_layer_dim, (i+1)*per_layer_dim)`.
//!   - `per_layer_model_projection.weight` [num_layers*256, hidden_size]
//!   - `per_layer_projection_norm.weight` [256]
//!   - per layer: `per_layer_input_gate.weight` [256, hidden_size],
//!     `per_layer_projection.weight` [hidden_size, 256],
//!     `post_per_layer_input_norm.weight` [hidden_size].
//!
//! This wave (W1.2.2-4) loads the BF16 tables and threads them onto the
//! model; wiring them into the layer forward is a later wave.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::weight_map::{DenseWeight, dense};

/// The three per-layer PLE weights attached to each E2B layer.
pub struct Gemma4PerLayerPleWeights {
    /// `layers.{i}.per_layer_input_gate.weight` — `[256, hidden_size]` Linear (no bias).
    pub input_gate: DenseWeight,
    /// `layers.{i}.per_layer_projection.weight` — `[hidden_size, 256]` Linear (no bias).
    pub projection: DenseWeight,
    /// `layers.{i}.post_per_layer_input_norm.weight` — `[hidden_size]` RMSNorm.
    pub post_norm: DenseWeight,
}

/// All model-level + per-layer PLE tables for a Gemma-4 E2B checkpoint.
pub struct Gemma4PleTables {
    /// One BF16 slice per layer of `embed_tokens_per_layer.weight`.
    /// Slice `i` = columns `[i*256, (i+1)*256)` of the full `[vocab, 8960]`
    /// table, stored as a base-pointer offset (`weight.offset(i*256*2)`)
    /// so each entry reads as a standalone `[vocab, 256]` table.
    pub embed_tokens_per_layer: Vec<DenseWeight>,
    /// `per_layer_model_projection.weight` — `[8960, hidden_size]` context projection.
    pub per_layer_model_projection: DenseWeight,
    /// `per_layer_projection_norm.weight` — `[256]` RMSNorm over the per-layer vector.
    pub per_layer_projection_norm: DenseWeight,
    /// Per-layer (all `num_hidden_layers`) input-gate / projection / norm.
    pub per_layer: Vec<Gemma4PerLayerPleWeights>,
}

/// MLP intermediate size for layer `i`.
///
/// E2B (`use_double_wide_mlp`) makes the KV-shared band — the last
/// `num_kv_shared_layers` layers — double-wide: gate/up/down project to
/// `2 * intermediate_size`. Layers outside the band (and all non-E2B
/// variants, where `use_double_wide_mlp` is false) keep `intermediate_size`.
pub(super) fn layer_intermediate_size(config: &ModelConfig, i: usize) -> usize {
    let band_start = config
        .num_hidden_layers
        .saturating_sub(config.num_kv_shared_layers);
    if config.use_double_wide_mlp && i >= band_start {
        config.intermediate_size * 2
    } else {
        config.intermediate_size
    }
}

/// Load all E2B PLE tables (model-level + per-layer) in one pass.
///
/// Returns `Ok(None)` when PLE is disabled (`hidden_size_per_layer_input == 0`,
/// all non-E2B Gemma-4 variants). BF16 `dense(...)` loads; no quantization.
pub(super) fn load_ple_tables_impl(
    store: &WeightStore,
    config: &ModelConfig,
    _gpu: &dyn GpuBackend,
) -> Result<Option<Gemma4PleTables>> {
    if config.hidden_size_per_layer_input == 0 {
        return Ok(None);
    }
    let prefix = &config.weight_prefix;
    let num_layers = config.num_hidden_layers;
    let per_layer = config.hidden_size_per_layer_input;
    let full_table = dense(store, &format!("{prefix}.embed_tokens_per_layer.weight"))?;
    let mut embed_tokens_per_layer = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        // BF16: 2 bytes/element; slice i starts at column i*per_layer.
        embed_tokens_per_layer.push(DenseWeight {
            weight: full_table.weight.offset(i * per_layer * 2),
        });
    }
    let per_layer_model_projection = dense(
        store,
        &format!("{prefix}.per_layer_model_projection.weight"),
    )?;
    let per_layer_projection_norm =
        dense(store, &format!("{prefix}.per_layer_projection_norm.weight"))?;
    let mut per_layer_weights = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let lp = config.layer_prefix(i);
        per_layer_weights.push(Gemma4PerLayerPleWeights {
            input_gate: dense(store, &format!("{lp}.per_layer_input_gate.weight"))?,
            projection: dense(store, &format!("{lp}.per_layer_projection.weight"))?,
            post_norm: dense(store, &format!("{lp}.post_per_layer_input_norm.weight"))?,
        });
    }
    tracing::info!(
        "Gemma-4 E2B: PLE enabled — {num_layers} per-layer embedding slices \
         ({per_layer}-dim) + per-layer input gates/projections loaded"
    );
    Ok(Some(Gemma4PleTables {
        embed_tokens_per_layer,
        per_layer_model_projection,
        per_layer_projection_norm,
        per_layer: per_layer_weights,
    }))
}

#[cfg(test)]
mod tests {
    use super::{layer_intermediate_size, load_ple_tables_impl};
    use atlas_core::config::parse_config;
    use spark_runtime::gpu::DevicePtr;
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    /// Gemma-4 E2B text config: 35 layers, hidden 1536, 256 per-layer input,
    /// 20 KV-shared layers, double-wide MLP.
    const E2B_JSON: &str = r#"{
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": 1536,
            "num_hidden_layers": 35,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 6144,
            "vocab_size": 262144,
            "hidden_size_per_layer_input": 256,
            "num_kv_shared_layers": 20,
            "use_double_wide_mlp": true,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-6
        }
    }"#;

    fn tensor(base: u64, shape: &[usize]) -> WeightTensor {
        WeightTensor {
            ptr: DevicePtr(base),
            shape: shape.to_vec(),
            dtype: WeightDtype::BF16,
        }
    }

    /// E2B geometry: L0-14 single-wide (6144), L15-34 (the KV-shared band,
    /// i >= 35-20) double-wide (12288).
    #[test]
    fn double_wide_band_geometry() {
        let cfg = parse_config(E2B_JSON).unwrap();
        assert_eq!(layer_intermediate_size(&cfg, 0), 6144);
        assert_eq!(layer_intermediate_size(&cfg, 14), 6144);
        assert_eq!(layer_intermediate_size(&cfg, 15), 12288);
        assert_eq!(layer_intermediate_size(&cfg, 34), 12288);
    }

    /// 26B/31B (no double-wide flag) must keep intermediate_size for every
    /// layer byte-for-byte (regression guard).
    #[test]
    fn non_e2b_geometry_is_unchanged() {
        let json = r#"{
            "model_type": "gemma4",
            "text_config": {
                "hidden_size": 5376,
                "num_hidden_layers": 12,
                "num_attention_heads": 32,
                "num_key_value_heads": 16,
                "head_dim": 256,
                "intermediate_size": 21504,
                "vocab_size": 262144,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6
            }
        }"#;
        let cfg = parse_config(json).unwrap();
        assert!(!cfg.use_double_wide_mlp);
        for i in 0..cfg.num_hidden_layers {
            assert_eq!(layer_intermediate_size(&cfg, i), 21504);
        }
    }

    /// PLE disabled (`hidden_size_per_layer_input == 0`) → load returns None,
    /// never errors, even against an empty store.
    #[test]
    fn ple_disabled_returns_none() {
        let json = r#"{
            "model_type": "gemma4",
            "text_config": {
                "hidden_size": 5376,
                "num_hidden_layers": 12,
                "num_attention_heads": 32,
                "num_key_value_heads": 16,
                "head_dim": 256,
                "intermediate_size": 21504,
                "vocab_size": 262144,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6
            }
        }"#;
        let cfg = parse_config(json).unwrap();
        assert_eq!(cfg.hidden_size_per_layer_input, 0);
        let store = WeightStore::empty();
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let tables = load_ple_tables_impl(&store, &cfg, &gpu).unwrap();
        assert!(tables.is_none());
    }

    /// Each PLE embedding slice must start at column `i*256` of the full
    /// 8960-wide BF16 table — i.e. byte offset `i*256*2` from the base.
    #[test]
    fn ple_slices_are_column_offsets() {
        let mut cfg = parse_config(E2B_JSON).unwrap();
        cfg.weight_prefix = "model.language_model".to_string();
        let n = cfg.num_hidden_layers;
        let mut weights = std::collections::HashMap::new();
        let p = |s: &str| format!("model.language_model.{s}");
        weights.insert(
            p("embed_tokens_per_layer.weight"),
            tensor(0x1000, &[262144, 8960]),
        );
        weights.insert(
            p("per_layer_model_projection.weight"),
            tensor(0x2000, &[8960, 1536]),
        );
        weights.insert(
            p("per_layer_projection_norm.weight"),
            tensor(0x3000, &[256]),
        );
        for i in 0..n {
            let lp = p(&format!("layers.{i}"));
            weights.insert(
                format!("{lp}.per_layer_input_gate.weight"),
                tensor(0x4000 + i as u64 * 8, &[256, 1536]),
            );
            weights.insert(
                format!("{lp}.per_layer_projection.weight"),
                tensor(0x5000 + i as u64 * 8, &[1536, 256]),
            );
            weights.insert(
                format!("{lp}.post_per_layer_input_norm.weight"),
                tensor(0x6000 + i as u64 * 8, &[1536]),
            );
        }
        let store = WeightStore::from_map(weights);
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let tables = load_ple_tables_impl(&store, &cfg, &gpu).unwrap().unwrap();
        assert_eq!(tables.embed_tokens_per_layer.len(), 35);
        for i in [0usize, 1, 14, 15, 34] {
            let expected = 0x1000 + (i * 256 * 2) as u64;
            assert_eq!(
                tables.embed_tokens_per_layer[i].weight.0, expected,
                "embed slice {i}"
            );
        }
        assert_eq!(tables.per_layer.len(), 35);
        assert_eq!(tables.per_layer_model_projection.weight.0, 0x2000);
        assert_eq!(tables.per_layer_projection_norm.weight.0, 0x3000);
        assert_eq!(tables.per_layer[15].input_gate.weight.0, 0x4000 + 15 * 8);
        assert_eq!(tables.per_layer[15].projection.weight.0, 0x5000 + 15 * 8);
        assert_eq!(tables.per_layer[15].post_norm.weight.0, 0x6000 + 15 * 8);
    }
}
