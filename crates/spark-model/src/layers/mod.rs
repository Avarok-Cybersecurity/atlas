// SPDX-License-Identifier: AGPL-3.0-only

pub mod deepseek_v4_mtp;
pub mod dense_ffn;
pub mod dflash_head;
pub mod ep_dispatch;
pub mod fp8_calibration;
pub mod moe;
pub mod mtp_head;
pub mod mtp_multi;
pub mod nemotron_mamba2;
pub mod nemotron_moe;
pub mod ops;
pub mod qwen3_attention;
pub mod qwen3_ssm;
pub mod vision_encoder;

pub use deepseek_v4_mtp::{DeepseekV4MtpHead, DeepseekV4MtpProposerState};
pub use dense_ffn::{DenseFfnLayer, FfnActivation};
pub use dflash_head::{
    BlockDiffusionDraftHead, DflashLayer, DflashProposerState, DflashQuantization,
};
pub use moe::MoeLayer;
pub use mtp_head::{MtpHead, MtpQuantization};
pub use nemotron_mamba2::NemotronMamba2Layer;
pub use nemotron_moe::NemotronMoeLayer;
pub use qwen3_attention::Qwen3AttentionLayer;
pub use qwen3_ssm::Qwen3SsmLayer;
pub use vision_encoder::{MergerLayer, ViTBlock, VisionEncoder};

use crate::layer::ForwardContext;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// Try to load an optional kernel, logging at debug level if it's not found.
/// Returns `KernelHandle(0)` (null) on failure — callers must check before use.
///
/// Debug (not warn) because misses are expected when a model doesn't use a
/// given feature: e.g. Qwen3-Coder-Next (GDN+attention) never calls MLA
/// kernels, but the layer builder still probes them. Warning on expected
/// misses drowned out genuine problems in startup logs.
pub fn try_kernel(gpu: &dyn GpuBackend, module: &str, func: &str) -> KernelHandle {
    match gpu.kernel(module, func) {
        Ok(h) => h,
        Err(_) => {
            tracing::debug!("Optional kernel '{module}::{func}' not loaded");
            KernelHandle(0)
        }
    }
}

/// FFN component: MoE (expert routing), dense SwiGLU, or None (standalone attention).
#[allow(clippy::large_enum_variant)]
pub enum FfnComponent {
    Moe(MoeLayer),
    Dense(DenseFfnLayer),
    /// No FFN — used by Nemotron-H standalone attention layers.
    None,
}

impl FfnComponent {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// ATLAS_FP32_ROUTING active for this FFN (MoE only; false otherwise).
    pub fn fp32_routing_active(&self) -> bool {
        match self {
            Self::Moe(m) => m.fp32_routing_active(),
            _ => false,
        }
    }

    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        match self {
            Self::Moe(m) => m.forward(input, ctx, stream),
            Self::Dense(d) => d.forward(input, ctx, stream),
            Self::None => Ok(input),
        }
    }

    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_k2(input, ctx, stream),
            Self::Dense(d) => d.forward_k2(input, ctx, stream),
            Self::None => Ok(()),
        }
    }

    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_k3(input, ctx, stream),
            Self::Dense(d) => d.forward_k3(input, ctx, stream),
            Self::None => Ok(()),
        }
    }

    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_prefill(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_prefill(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_batched(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }
}
