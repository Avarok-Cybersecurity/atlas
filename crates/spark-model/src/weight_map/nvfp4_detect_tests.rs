// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `nvfp4_detect.rs` (child module; split out for the
//! ≤500 LoC cap, same shape as `exl3_materialize_dense_tests.rs`). The two
//! test groups keep their own modules; `super::super` is `nvfp4_detect`.

mod ep_detection_tests {
    use super::super::*;
    use atlas_core::config::ModelConfig;
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
        let store =
            store_with(&["model.language_model.layers.0.self_attn.q_proj.weight".to_string()]);

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
}

mod key_variant_tests {
    use super::super::*;
    use spark_runtime::gpu::DevicePtr;
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    fn store_of(entries: &[(&str, WeightDtype)]) -> WeightStore {
        let map = entries
            .iter()
            .map(|(n, d)| {
                (
                    n.to_string(),
                    WeightTensor {
                        ptr: DevicePtr::NULL,
                        shape: vec![1],
                        dtype: *d,
                    },
                )
            })
            .collect();
        WeightStore::from_map(map)
    }

    /// The EXL3-materialized shared-expert triplet inside a Bf16Raw-declared
    /// store (the compute-sanitizer CUDA-700 case) must load as Standard —
    /// but ONLY under the native-MoE gate; with the gate off detection is
    /// byte-identical to before (the key stays on the declared variant).
    #[test]
    fn materialized_triplet_under_bf16raw_routes_to_standard_when_gated() {
        let p = "model.layers.3.mlp.shared_expert.gate_proj";
        let store = store_of(&[
            (&format!("{p}.weight"), WeightDtype::UInt8),
            (&format!("{p}.weight_scale"), WeightDtype::FP8E4M3),
            (&format!("{p}.weight_scale_2"), WeightDtype::FP32),
        ]);
        assert_eq!(
            resolve_key_variant(&store, p, Nvfp4Variant::Bf16Raw, true),
            Nvfp4Variant::Standard
        );
        assert_eq!(
            resolve_key_variant(&store, p, Nvfp4Variant::Bf16Raw, false),
            Nvfp4Variant::Bf16Raw
        );
        // Already Standard: nothing to override.
        assert_eq!(
            resolve_key_variant(&store, p, Nvfp4Variant::Standard, true),
            Nvfp4Variant::Standard
        );
    }

    /// Negative controls: a real BF16 dense key and a CompressedTensors key
    /// are never mistaken for the triplet, gate on or off.
    #[test]
    fn triplet_fallback_does_not_steal_other_layouts() {
        let p = "model.layers.3.linear_attn.out_proj";
        let dense = store_of(&[(&format!("{p}.weight"), WeightDtype::BF16)]);
        assert_eq!(
            resolve_key_variant(&dense, p, Nvfp4Variant::Standard, true),
            Nvfp4Variant::Bf16Raw
        );
        let ct = store_of(&[
            (&format!("{p}.weight_packed"), WeightDtype::UInt8),
            (&format!("{p}.weight_scale"), WeightDtype::FP8E4M3),
            (&format!("{p}.weight_global_scale"), WeightDtype::FP32),
        ]);
        assert_eq!(
            resolve_key_variant(&ct, p, Nvfp4Variant::CompressedTensors, true),
            Nvfp4Variant::CompressedTensors
        );
        // Per-row FP8 tail key in an NVFP4 checkpoint keeps its own fallback.
        let fp8 = store_of(&[
            (&format!("{p}.weight"), WeightDtype::FP8E4M3),
            (&format!("{p}.weight_scale"), WeightDtype::BF16),
        ]);
        assert_eq!(
            resolve_key_variant(&fp8, p, Nvfp4Variant::Standard, true),
            Nvfp4Variant::Fp8Dequanted
        );
    }
}
