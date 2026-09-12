// SPDX-License-Identifier: AGPL-3.0-only

//! Kimi K3 weight loader — C0: name-map dry-run only.
//!
//! `load_*` bails until S1 wires the graph. Do not copy GDN/Mamba kernels.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::DenseWeight;

mod classes;
mod dry_run;

pub use dry_run::{KimiK3DryRun, dry_run_index_json, dry_run_weight_map, require_shard_count};

pub struct KimiK3WeightLoader;

const WIP: &str = "K3-WIP: graph not implemented; use dry_run";

impl KimiK3WeightLoader {
    pub fn dry_run(index_json: &str, language_model_only: bool) -> Result<KimiK3DryRun> {
        dry_run::dry_run_index_json(index_json, language_model_only)
    }
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
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        bail!(WIP)
    }

    fn load_embedding(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(WIP)
    }

    fn load_final_norm(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(WIP)
    }

    fn load_lm_head(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(WIP)
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

    #[test]
    fn loader_for_config_kimi_k3() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        for ty in ["kimi_k3", "kimi_linear", "Kimi-K3"] {
            config.model_type = ty.to_string();
            let loader = loader_for_config(&config).expect(ty);
            let store = WeightStore::empty();
            let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
            let err = match loader.load_layers(&store, &config, &gpu, &[]) {
                Ok(_) => panic!("{ty}: expected K3-WIP bail"),
                Err(e) => e.to_string(),
            };
            assert!(err.contains("K3-WIP"), "{ty}: {err}");
        }
    }
}
