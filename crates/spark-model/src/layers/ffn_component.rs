// SPDX-License-Identifier: AGPL-3.0-only

//! `FfnComponent`'s methods, split out of `layers/mod.rs` to keep it under
//! the 500-line cap.

use super::*;

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

    /// True when this MoE FFN can serve DECODE through the grouped read-once
    /// GEMM (forward_prefill) instead of the pairwise per-slot loop. The
    /// is_dense() comment above asserts grouped is "a net loss at small batch"
    /// on a 256-expert MoE, but that was never measured for decode CONCURRENCY
    /// (n=4) where the pairwise path re-reads ~14-20 distinct experts as 40
    /// per-slot CTAs. Native-NVFP4-routed only (forward_prefill's unconditional
    /// grouped path); dense/none are false.
    pub fn moe_grouped_decode_ok(&self) -> bool {
        match self {
            Self::Moe(m) => m.grouped_decode_ok(),
            _ => false,
        }
    }

    /// ATLAS_FP32_ROUTING active for this FFN (MoE only; false otherwise).
    pub fn fp32_routing_active(&self, levers: &ops::ModelLevers) -> bool {
        match self {
            Self::Moe(m) => m.fp32_routing_active(levers),
            _ => false,
        }
    }

    /// True when this FFN's routed experts are served natively from EXL3
    /// trellis (`ATLAS_EXL3_NATIVE_MOE=1`). Every mgemm in that arm is a
    /// COOPERATIVE launch — not CUDA-graph-capturable — so each layer kind's
    /// `decode_graph_unsupported` (and the verify-path `use_graphs` terms)
    /// must include this, exactly like the `lm_head_exl3` veto.
    pub fn exl3_native_moe(&self) -> bool {
        match self {
            Self::Moe(m) => m.exl3_native_active(),
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
        match self {
            Self::Dense(d) => d.can_forward_km(m),
            Self::Moe(moe) => moe.can_forward_km(m),
            Self::None => false,
        }
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
            Self::Moe(moe) if moe.can_forward_km(m) => {
                moe.forward_km(input, m, ctx, stream)?;
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
