// SPDX-License-Identifier: AGPL-3.0-only

//! Feature-1 MoE LoRA install on the GDN / linear-attention layer.
//!
//! Linear-attention layers carry NO attention LoRA (their projections are
//! rejected at classify), but the MoE FFN exists on every layer, so a real
//! all-layer MoE adapter installs its router + routed-expert deltas here too —
//! the same `MoeLayer::set_lora_weights` path the full-attention layer uses.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::Qwen3SsmLayer;
use crate::layers::FfnComponent;
use crate::layers::ops::lora_delta::{LoraKernels, LoraPair};
use crate::lora::ExpertLoraLayer;

impl Qwen3SsmLayer {
    /// Install this GDN layer's MoE router + routed-expert LoRA onto its
    /// `FfnComponent::Moe`. Hard-rejects (never silently drops) when the layer's
    /// FFN is not MoE — an expert/router delta on a dense-FFN GDN layer is a
    /// loader/adapter mismatch.
    pub fn set_moe_lora_weights(
        &mut self,
        router: Option<LoraPair>,
        experts: ExpertLoraLayer,
        kernels: LoraKernels,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        if let FfnComponent::Moe(m) = &mut self.ffn {
            return m.set_lora_weights(router, experts, kernels, gpu);
        }
        anyhow::bail!(
            "LoRA: router/expert deltas installed on a linear-attention layer whose \
             FFN is not MoE (loader/adapter mismatch)"
        )
    }
}

impl Qwen3SsmLayer {
    /// Install this linear-attention layer's DENSE-FFN LoRA onto its
    /// `FfnComponent::Dense`.
    ///
    /// The mirror of `set_moe_lora_weights` for dense-FFN hybrids. A
    /// linear-attention layer carries no attention projections, but on
    /// Qwen3.8-27B it does carry the SwiGLU FFN — all 64 layers do, only 16 of
    /// which are full attention — and real adapters for that architecture ship
    /// gate/up/down for every one of them. Rejecting those rejected three
    /// quarters of the adapter, and the old message could only suggest
    /// retraining with `layers_to_transform`.
    ///
    /// The component is the same `DenseFfnLayer` the full-attention layers
    /// hold, so the delta path, its pinned dispatch arms and its refusals are
    /// identical here — this only hands it the weights.
    ///
    /// Hard-rejects a non-dense FFN rather than dropping the pairs: a dense
    /// delta arriving at a MoE or absent FFN is a loader/adapter mismatch, and
    /// silently ignoring it would be an adapter that reports success and does
    /// nothing — the exact failure this whole change removes.
    pub fn set_ffn_lora_weights(
        &mut self,
        ffn: crate::layers::ops::lora_delta::LoraFfnWeights,
    ) -> Result<()> {
        match &mut self.ffn {
            FfnComponent::Dense(d) => d.set_lora_weights(ffn),
            FfnComponent::Moe(_) => anyhow::bail!(
                "LoRA: dense-FFN delta on a linear-attention layer whose FFN is MoE — \
                 routed-expert deltas belong on set_moe_lora_weights"
            ),
            FfnComponent::None => {
                anyhow::bail!("LoRA: dense-FFN delta on a linear-attention layer that has no FFN")
            }
        }
    }
}

impl Qwen3SsmLayer {
    /// Install this layer's GDN `out_proj` delta.
    ///
    /// Separate from `set_ffn_lora_weights`: that one targets the block's FFN,
    /// this one the linear-attention block's own output projection.
    pub fn set_out_proj_lora(&mut self, pair: LoraPair, kernels: LoraKernels) {
        self.lora_out_proj = Some((pair, kernels));
    }

    /// Install this layer's FUSED GDN input-projection deltas (either may be
    /// absent — an adapter can target qkv/z without a/b and vice versa).
    pub fn set_gdn_in_lora(
        &mut self,
        qkvz: Option<(LoraPair, LoraKernels)>,
        ba: Option<(LoraPair, LoraKernels)>,
    ) {
        self.lora_gdn_qkvz = qkvz;
        self.lora_gdn_ba = ba;
    }

    /// Fold the fused qkv+z delta into the deinterleaved `[m, qkvz]` buffer:
    /// `deinterleaved += scale * (normed @ A^T) @ B^T`.
    ///
    /// MUST run immediately after the qkvz projection on EVERY arm — before
    /// conv1d consumes rows 0..conv_dim and before the gated norm consumes the
    /// Z slice — or the recurrent state desyncs between arms. No-op without an
    /// adapter (base path byte-identical). GRAPH-SAFE (pure launches into
    /// fixed arena scratch).
    pub(super) fn apply_lora_gdn_qkvz(
        &self,
        ctx: &crate::layers::ForwardContext,
        normed: spark_runtime::gpu::DevicePtr,
        deinterleaved: spark_runtime::gpu::DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        let Some((ref pair, ref kernels)) = self.lora_gdn_qkvz else {
            return Ok(());
        };
        crate::layers::ops::lora_delta::apply_lora_delta(
            ctx.gpu,
            kernels,
            pair,
            normed,
            deinterleaved,
            m,
            ctx.buffers.lora_xa(),
            ctx.buffers.lora_delta(),
            stream,
        )
    }

    /// Compute the RAW (unscaled) fused b+a delta `[m, ssm_ba_size]` into the
    /// `ssm_ba` arena scratch (unused by the fused BA-gates path) and return
    /// `(delta_ptr, scale)` for the gates kernel to fold pre-transform.
    /// `(DevicePtr(0), 0.0)` without an adapter — the kernels treat a null
    /// pointer as "no delta" and stay byte-identical.
    pub(super) fn compute_lora_gdn_ba(
        &self,
        ctx: &crate::layers::ForwardContext,
        normed: spark_runtime::gpu::DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<(spark_runtime::gpu::DevicePtr, f32)> {
        if crate::layers::ops::lora_delta::lora_no_apply() {
            return Ok((spark_runtime::gpu::DevicePtr(0), 0.0));
        }
        let Some((ref pair, ref kernels)) = self.lora_gdn_ba else {
            return Ok((spark_runtime::gpu::DevicePtr(0), 0.0));
        };
        let delta = ctx.buffers.ssm_ba();
        crate::layers::ops::lora_delta::compute_lora_delta_raw(
            ctx.gpu,
            kernels,
            pair,
            normed,
            delta,
            m,
            ctx.buffers.lora_xa(),
            stream,
        )?;
        Ok((delta, pair.scale))
    }

    /// `out += scale * (normed_out @ A^T) @ B^T`.
    ///
    /// No-op without an adapter, so the base path stays byte-identical.
    ///
    /// The caller must invoke this AFTER any TP reduce: `out` is a partial
    /// row-parallel product until then, and a delta added to a partial would
    /// be summed once per rank.
    pub(super) fn apply_lora_out_proj(
        &self,
        ctx: &crate::layers::ForwardContext,
        normed_out: spark_runtime::gpu::DevicePtr,
        out: spark_runtime::gpu::DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        if crate::layers::ops::lora_delta::lora_no_ffn() {
            return Ok(());
        }
        let Some((ref pair, ref kernels)) = self.lora_out_proj else {
            return Ok(());
        };
        crate::layers::ops::lora_delta::apply_lora_delta(
            ctx.gpu,
            kernels,
            pair,
            normed_out,
            out,
            m,
            ctx.buffers.lora_xa(),
            ctx.buffers.lora_delta(),
            stream,
        )
    }
}
