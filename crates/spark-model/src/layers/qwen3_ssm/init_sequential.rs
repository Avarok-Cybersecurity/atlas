// SPDX-License-Identifier: AGPL-3.0-only

//! Sequential-QKVZ constructor variant, split from `init.rs` for the
//! ≤500 LoC cap. Child module of `init` so it shares the same `super`
//! (the `qwen3_ssm` module) and field visibility.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::super::Qwen3SsmLayer;
use crate::layers::FfnComponent;
use crate::weight_map::{DenseWeight, QuantizedWeight, SsmWeights};

impl Qwen3SsmLayer {
    /// Construct an SSM layer where QKVZ projection output is already sequential.
    ///
    /// Used by Qwen3.5 where separate QKV and Z weights are concatenated at load
    /// time into `[Q|K|V|Z]` row order. The `deinterleave_qkvz` kernel is skipped
    /// and plain `w4a16_gemv` writes directly to the deinterleaved buffer.
    pub fn new_sequential(
        input_norm: DenseWeight,
        ssm: SsmWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        qkvz_nvfp4: Option<QuantizedWeight>,
        qkvz_nvfp4_t: Option<QuantizedWeight>,
        out_proj_nvfp4_t: Option<QuantizedWeight>,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let mut layer = Self::new(
            input_norm,
            ssm,
            post_attn_norm,
            ffn,
            qkvz_nvfp4,
            config,
            gpu,
        )?;
        layer.sequential_qkvz = true;
        layer.qkvz_nvfp4_t = qkvz_nvfp4_t;
        layer.out_proj_nvfp4_t = out_proj_nvfp4_t;
        Ok(layer)
    }
}
