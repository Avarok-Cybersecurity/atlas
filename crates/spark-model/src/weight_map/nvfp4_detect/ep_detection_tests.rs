// SPDX-License-Identifier: AGPL-3.0-only
//! Detector tests, split out of `nvfp4_detect.rs` so that file stays under the
//! 500-line cap. Moved verbatim -- `use super::*` still resolves to
//! `nvfp4_detect`, so every assertion reads exactly what it read inline.
use super::*;
use avarok_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;

/// A store holding only the FP8 attention marker at a given layer, which is
/// what the detector sniffs for. Names are all the detector reads.
fn store_with(names: &[String]) -> WeightStore {
    use std::collections::HashMap;
    let map: HashMap<String, spark_runtime::weights::WeightTensor> = names
        .iter()
        .map(|n| {
            (
                n.clone(),
                spark_runtime::weights::WeightTensor {
                    ptr: spark_runtime::gpu::DevicePtr::NULL,
                    shape: vec![1],
                    dtype: spark_runtime::weights::WeightDtype::FP8E4M3,
                },
            )
        })
        .collect();
    WeightStore::from_map(map)
}

#[test]
fn alternate_layer0_fp8_dtype_is_detected_on_every_ep_rank() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.quantization_config = None;
    let store = store_with(&["model.language_model.layers.0.self_attn.q_proj.weight".to_string()]);

    cfg.ep_world_size = 2;
    for ep_rank in 0..2 {
        cfg.ep_rank = ep_rank;
        assert_eq!(
            detect_nvfp4_variant(&store, &cfg),
            Nvfp4Variant::Fp8Dequanted,
            "EP rank {ep_rank} must inspect the same layer-zero checkpoint marker"
        );
    }
}

#[test]
fn scale_inv_suffix_fallback_detects_an_unexpected_prefix() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.quantization_config = None;
    let store =
        store_with(&["third_party.transformer.blocks.17.attn.q.weight_scale_inv".to_string()]);
    assert_eq!(
        detect_nvfp4_variant(&store, &cfg),
        Nvfp4Variant::Fp8Dequanted
    );
}
