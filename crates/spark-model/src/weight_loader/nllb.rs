// SPDX-License-Identifier: AGPL-3.0-only

//! NLLB / M2M-100 model-family marker loader.
//!
//! NLLB-200 checkpoints declare `model_type = "m2m_100"` and use an
//! encoder-decoder architecture with learned absolute positions and decoder
//! cross-attention. Atlas's executable model stack is currently decoder-only;
//! this loader is registered so the model type has a single, explicit
//! fail-fast path while the CUDA and Metal target metadata can resolve.

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
            "NLLB / m2m_100 checkpoints are recognized, but Atlas does not yet implement the encoder-decoder runtime required by facebook/nllb-200-3.3B (encoder self-attention + decoder cross-attention + learned absolute positions)"
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
