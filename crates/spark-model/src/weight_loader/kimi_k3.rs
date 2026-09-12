// SPDX-License-Identifier: AGPL-3.0-only

//! Kimi K3 weight loader — C1: BF16 0.40B twin bind; MXFP4 packed is S5.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::DenseWeight;

mod bf16;
mod classes;
mod dry_run;

pub use dry_run::{KimiK3DryRun, dry_run_index_json, dry_run_weight_map};

pub struct KimiK3WeightLoader;

impl KimiK3WeightLoader {
    pub fn dry_run(index_json: &str, language_model_only: bool) -> Result<KimiK3DryRun> {
        dry_run::dry_run_index_json(index_json, language_model_only)
    }
}

/// Packed expert MXFP4 is a later slice. Twin C1 is unpacked `.weight`.
pub fn refuse_mxfp4(store: &WeightStore) -> Result<()> {
    if store.names().any(|n| n.contains("weight_packed")) {
        bail!("S5 MXFP4 not this slice");
    }
    Ok(())
}

impl ModelWeightLoader for KimiK3WeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn binds_vision_encoder(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        refuse_mxfp4(store)?;
        bf16::load_layers(store, config, gpu)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        refuse_mxfp4(store)?;
        bf16::load_embedding(store, config, gpu)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        refuse_mxfp4(store)?;
        bf16::load_final_norm(store, config, gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        refuse_mxfp4(store)?;
        bf16::load_lm_head(store, config, gpu)
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::weight_loader::MtpWeights>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factory::loader_for_config;
    use atlas_core::config::parse_config;
    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::weights::{WeightDtype, WeightTensor};
    use std::collections::HashMap;

    #[test]
    fn loader_for_config_kimi_k3() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        for ty in ["kimi_k3", "kimi_linear", "Kimi-K3"] {
            config.model_type = ty.to_string();
            let loader = loader_for_config(&config).expect(ty);
            let store = WeightStore::empty();
            let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
            let err = match loader.load_layers(&store, &config, &gpu, &[]) {
                Ok(_) => panic!("{ty}: empty store must not bind"),
                Err(e) => e.to_string(),
            };
            assert!(
                err.contains("not found") || err.contains("S5 MXFP4"),
                "{ty}: {err}"
            );
            assert!(!err.contains("K3-WIP"), "{ty}: stale WIP bail: {err}");
        }
    }

    #[test]
    fn load_bails_on_mxfp4_packed() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "kimi_k3".into();
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let ptr = gpu.alloc(4).unwrap();
        let store = WeightStore::from_map(HashMap::from([(
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed".to_string(),
            WeightTensor {
                ptr,
                shape: vec![2, 2],
                dtype: WeightDtype::UInt8,
            },
        )]));
        let loader = KimiK3WeightLoader;
        let err = match loader.load_layers(&store, &config, &gpu, &[]) {
            Ok(_) => panic!("packed MXFP4 must not bind layers"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("S5 MXFP4 not this slice"), "{err}");
        let err = match loader.load_embedding(&store, &config, &gpu) {
            Ok(_) => panic!("packed MXFP4 must not bind embed"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("S5 MXFP4 not this slice"), "{err}");
    }

    #[test]
    fn load_bf16_twin_binds_layers() {
        const TWIN: &str = include_str!("../../../../docs/k3/fixtures/Kimi-K3-0.40B-config.json");
        let config = parse_config(TWIN).expect("0.40B twin");
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let graph = atlas_core::kimi_k3::K3Graph::from_config(&config);
        let mut map = HashMap::new();
        let mut put = |name: String| {
            let ptr = gpu.alloc(4).unwrap();
            map.insert(
                name,
                WeightTensor {
                    ptr,
                    shape: vec![2],
                    dtype: WeightDtype::BF16,
                },
            );
        };
        put(bf16::text_key(&config, "model.embed_tokens.weight"));
        put(bf16::text_key(&config, "model.norm.weight"));
        put(bf16::text_key(&config, "lm_head.weight"));
        for spec in &graph.layers {
            for k in bf16::layer_keys(
                &config,
                spec.index,
                spec.mixer,
                spec.mlp,
                config.num_experts,
            ) {
                put(k);
            }
        }
        let store = WeightStore::from_map(map);
        let loader = KimiK3WeightLoader;
        let layers = loader.load_layers(&store, &config, &gpu, &[]).unwrap();
        assert_eq!(layers.len(), 8);
        loader.load_embedding(&store, &config, &gpu).unwrap();
        loader.load_final_norm(&store, &config, &gpu).unwrap();
        loader.load_lm_head(&store, &config, &gpu).unwrap();
    }
}
