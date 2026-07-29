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

/// Minimum K at which the deep-K `w4a16_gemm_t_k64` (K_STEP_T=64) beats the
/// K_STEP_T=32 `w4a16_gemm_t`.
///
/// ★ 6144, not 4096. Measured with `w4a16_m17_bench` on the REAL decode shapes at
/// M=16 against the STREAM-measured 230 GB/s ceiling — `_k64` is the WORST tile
/// variant at K=5120 and the best only at K>=6144:
///
///   ssm_qkvz     N=16384 K=5120   _t 281.9us   _k64 341.6us   _m128 272.4us
///   attn qkv     N=14336 K=5120   _t 273.9us   _k64 328.5us   _m128 262.8us
///   ssm_out_proj N=5120  K=6144   _t 237.7us   _k64 163.3us   _m128 240.7us
///
/// The original 4096 threshold (this session) was derived from the ffn/out_proj
/// shapes and wrongly generalised to K=5120, sending 48 qkvz + 16 fused-qkv
/// launches per step to the slowest variant. Both variants accumulate K
/// sequentially, so moving between them is byte-identical.
///
/// `ATLAS_NO_W4A16_K64=1` restores the pre-session 8192 threshold.
pub(crate) fn w4a16_k64_min_k() -> u32 {
    static MIN_K: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MIN_K.get_or_init(|| {
        // Explicit override so an A/B can pin a previous threshold exactly.
        if let Some(n) = std::env::var("ATLAS_W4A16_K64_MIN_K")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
        {
            return n;
        }
        if std::env::var("ATLAS_NO_W4A16_K64").ok().as_deref() == Some("1") {
            8192
        } else {
            6144
        }
    })
}

pub use deepseek_v4_mtp::{DeepseekV4MtpHead, DeepseekV4MtpProposerState};
pub use dense_ffn::{DenseFfnLayer, DenseFfnWeights, FfnActivation};
pub use dflash_head::{
    BlockDiffusionDraftHead, DflashLayer, DflashProposerState, DflashQuantization,
};
pub use moe::MoeLayer;
pub use mtp_head::{MtpHead, MtpQuantization, mtp_drafter_prefill_enabled};
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
/// Resolve the N128/M64 tile GEMM, preferring the 3-deep weight-pipeline variant.
/// **ON by default**; `ATLAS_NO_TGEMM_PIPELINE3` (presence — `=0` is NOT "off")
/// falls back to the 2-stage parent. Falls back automatically on any target that
/// does not ship `_p3`.
///
/// Same mechanism as [`k64_kernel`]: the parent drains its cp.async group before
/// the dequant phase, which only a co-resident CTA can cover. This kernel's live
/// shapes — ssm_qkvz (128 CTAs) and the fused QKV (112) — sit in the exposed
/// band of the grid.x-vs-efficiency curve. Bit-identical.
pub fn tgemm_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    if std::env::var("ATLAS_NO_TGEMM_PIPELINE3").is_err() {
        let h = try_kernel(gpu, "w4a16", "w4a16_gemm_t_p3");
        if h.0 != 0 {
            return h;
        }
    }
    try_kernel(gpu, "w4a16", "w4a16_gemm_t")
}

/// Resolve the k64 deep-K tile GEMM, preferring the 3-deep weight-pipeline
/// variant. **ON by default**; `ATLAS_NO_K64_PIPELINE3` (presence — `=0` is NOT
/// "off") falls back to the 2-stage parent.
///
/// The parent issues one cp.async group then `wait_all`s it before the dequant
/// phase, so with a small grid there are ZERO outstanding loads across that
/// phase. The out_proj/o_proj shapes (N=5120, K=6144) launch 40 CTAs on 48 SMs —
/// exactly 1 CTA/SM — so nothing covers the drain, and they measure ~38% of
/// achievable while lm_head (1938 CTAs) reaches 83% on the identical loop.
/// `_p3` keeps step i+2's loads in flight across dequant(i+1). Bit-identical.
pub fn k64_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    let want_p3 = std::env::var("ATLAS_NO_K64_PIPELINE3").is_err();
    if want_p3 {
        let h = try_kernel(gpu, "w4a16", "w4a16_gemm_t_k64_p3");
        if h.0 != 0 {
            return Ok(h);
        }
    }
    gpu.kernel("w4a16", "w4a16_gemm_t_k64")
}

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

    /// True for a plain dense (SwiGLU) FFN. Wide-batch verify paths gate their
    /// `forward_prefill` fast path on this: batching reads dense weights once
    /// (big win at N=17), but on a 256-expert MoE the grouped-GEMM is a net
    /// loss at small batch (per-expert M~1 + sort/permute overhead), so MoE
    /// keeps its per-token loop.
    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
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

    /// Whether the K=m (m<=8) batched-GEMV verify FFN is available (dense
    /// only — MoE / missing batch4/batch8 kernel / non-NVFP4 weights →
    /// false). Lets callers gate branch entry BEFORE computing the pre-FFN
    /// norm, so there is no half-done fallthrough to `forward_prefill`.
    pub fn can_forward_km(&self, m: u32) -> bool {
        matches!(self, Self::Dense(d) if d.can_forward_km(m))
    }

    /// K=m (m=4..8) verify FFN via batched GEMV (dense only). Returns
    /// `false` when the path is unavailable (MoE / missing batchm kernel /
    /// non-NVFP4 weights) so the caller can fall back to `forward_prefill`.
    pub fn try_forward_km(
        &self,
        input: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        match self {
            Self::Dense(d) if d.can_forward_km(m) => {
                d.forward_km(input, m, ctx, stream)?;
                Ok(true)
            }
            _ => Ok(false),
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

    pub fn forward_token_major_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_token_major_decode(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_atomic_c4_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_atomic_c4_decode(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }
}
