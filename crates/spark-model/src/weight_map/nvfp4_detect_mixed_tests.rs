// SPDX-License-Identifier: AGPL-3.0-only

mod mixed_main_mtp_tests {
    use super::super::super::*;
    use atlas_core::config::{ModelConfig, QuantizationConfig};
    use spark_runtime::weights::WeightTensor;

    fn config() -> ModelConfig {
        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.quantization_config = Some(QuantizationConfig {
            quant_method: "modelopt".into(),
            quant_algo: "MIXED_PRECISION".into(),
            format: String::new(),
            ignore_modules: Vec::new(),
        });
        cfg
    }

    fn store(prefix: &str, dtype: WeightDtype, complete: bool, mtp: bool) -> WeightStore {
        let mut entries = vec![
            (format!("{prefix}.weight"), dtype),
            (format!("{prefix}.weight_scale"), WeightDtype::FP8E4M3),
        ];
        if complete {
            entries.push((format!("{prefix}.weight_scale_2"), WeightDtype::FP32));
        }
        if mtp {
            entries.push((
                "mtp.layers.0.mlp.experts.0.gate_proj.weight".into(),
                WeightDtype::FP8E4M3,
            ));
            entries.push((
                "mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_inv".into(),
                WeightDtype::FP32,
            ));
        }
        WeightStore::from_map(
            entries
                .into_iter()
                .map(|(name, dtype)| {
                    (
                        name,
                        WeightTensor {
                            ptr: DevicePtr::NULL,
                            shape: vec![1],
                            dtype,
                        },
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn main_nvfp4_detection_does_not_change_when_fp8_mtp_is_loaded() {
        let mut cfg = config();
        cfg.ep_world_size = 2;
        for rank in 0..2 {
            cfg.ep_rank = rank;
            let p = format!(
                "{}.mlp.experts.{}.gate_proj",
                cfg.layer_prefix(0),
                cfg.local_expert_range().0
            );
            for mtp in [false, true] {
                let weights = store(&p, WeightDtype::UInt8, true, mtp);
                assert_eq!(detect_nvfp4_variant(&weights, &cfg), Nvfp4Variant::Standard);
                if mtp {
                    assert_eq!(
                        resolve_key_variant(
                            &weights,
                            "mtp.layers.0.mlp.experts.0.gate_proj",
                            Nvfp4Variant::Standard,
                            false
                        ),
                        Nvfp4Variant::Fp8Dequanted
                    );
                }
            }
        }
    }

    #[test]
    fn mixed_precision_requires_positive_main_expert_evidence() {
        let cfg = config();
        let p = format!("{}.mlp.experts.0.gate_proj", cfg.layer_prefix(0));
        for weights in [
            store(&p, WeightDtype::UInt8, false, true),
            store(&p, WeightDtype::FP8E4M3, true, true),
            store("unrelated.expert", WeightDtype::UInt8, true, true),
        ] {
            assert_eq!(
                detect_nvfp4_variant(&weights, &cfg),
                Nvfp4Variant::Fp8Dequanted
            );
        }
    }

    #[test]
    fn explicit_scheme_still_precedes_main_expert_sniff() {
        let mut cfg = config();
        let p = format!("{}.mlp.experts.0.gate_proj", cfg.layer_prefix(0));
        let weights = store(&p, WeightDtype::UInt8, true, true);
        for (method, algo, expected) in [
            ("modelopt", "FP8", Nvfp4Variant::Fp8Dequanted),
            ("fp8", "", Nvfp4Variant::Fp8Dequanted),
            ("compressed-tensors", "", Nvfp4Variant::CompressedTensors),
        ] {
            let qc = cfg.quantization_config.as_mut().unwrap();
            qc.quant_method = method.into();
            qc.quant_algo = algo.into();
            assert_eq!(detect_nvfp4_variant(&weights, &cfg), expected);
        }
    }
}
