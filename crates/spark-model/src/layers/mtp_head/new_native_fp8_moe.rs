// SPDX-License-Identifier: AGPL-3.0-only

//! The MTP drafter's native-FP8 MoE layer — split out of `new.rs` (500-line
//! cap) by exact piecewise copy.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::MtpHead;
use crate::layers::MoeLayer;
use crate::weight_map::{ExpertWeight, MoeWeights, MtpWeights};

impl MtpHead {
    /// The drafter's MoE on the checkpoint's FP8 tables, constructed exactly
    /// as `qwen35/load_layers.rs` builds a native-FP8 main layer: null NVFP4
    /// expert slots for the constructor's pointer tables, the BF16 router
    /// (`gate_nvfp4 = None` → `dense_gemv`/`dense_gemm`), then
    /// `set_fp8_experts`. Every FP8 decode arm reads `fp8_shared_expert`, so
    /// the NVFP4 shared slot stays null too. The loader's BF16 dequants are
    /// released afterwards — they were the same tensors, kept only for the
    /// NVFP4 re-quantization path this head does not take.
    pub(super) fn new_native_fp8_moe(
        weights: &mut MtpWeights,
        config: &avarok_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<MoeLayer> {
        let fp8 = weights
            .fp8_experts
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("new_native_fp8_moe: no FP8 tables"))?;
        anyhow::ensure!(
            fp8.experts.len() == config.num_experts,
            "MTP FP8 tables cover {} experts, config says {}",
            fp8.experts.len(),
            config.num_experts
        );
        let moe_weights = MoeWeights {
            gate: weights.moe_gate,
            shared_expert: ExpertWeight::null(),
            shared_expert_gate: weights.shared_expert_gate,
            experts: vec![ExpertWeight::null(); config.num_experts],
            router_pre_norm: None,
            correction_bias: None,
        };
        let mut moe = MoeLayer::new(moe_weights, config.num_experts, None, gpu, config)?;
        moe.set_fp8_experts(&fp8.experts, fp8.shared_expert, gpu)?;
        weights.release_bf16_expert_dequants(gpu)?;
        Ok(moe)
    }
}
