// SPDX-License-Identifier: AGPL-3.0-only

//! NLLB / M2M-100 model-family marker loader.
//!
//! NLLB-200 checkpoints declare `model_type = "m2m_100"` and use an
//! encoder-decoder architecture with learned absolute positions and decoder
//! cross-attention. NLLB **is** served on the GPU, but through a dedicated
//! encoder-decoder runtime (`crate::model::nllb`, `NllbGpuModel`) that
//! `build_model` constructs *before* consulting this loader — see
//! `crate::factory::build`. This marker exists only because the generic
//! `ModelWeightLoader` table needs an entry for the model type; the generic
//! `TransformerLayer`/paged-KV pipeline is decoder-only and cannot express the
//! encoder self-attention + decoder cross-attention that NLLB requires, so
//! every generic entry point here fails fast. Reaching one means the dedicated
//! serve path was bypassed (a routing bug), not that NLLB is unsupported.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::weight_loader::ModelWeightLoader;
use crate::weight_map::{DenseWeight, MtpWeights};

pub struct NllbWeightLoader;

impl NllbWeightLoader {
    fn unsupported() -> anyhow::Error {
        anyhow::anyhow!(
            "NLLB / m2m_100 is served by the dedicated GPU encoder-decoder runtime (spark_model::model::nllb::NllbGpuModel), which build_model selects before this loader; the generic decoder-only ModelWeightLoader pipeline cannot serve it. Reaching this loader means the dedicated serve path was bypassed."
        )
    }
}

impl ModelWeightLoader for NllbWeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        bail!(Self::unsupported())
    }

    fn load_embedding(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(Self::unsupported())
    }

    fn load_final_norm(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(Self::unsupported())
    }

    fn load_lm_head(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        bail!(Self::unsupported())
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}
