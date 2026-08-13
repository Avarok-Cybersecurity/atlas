// SPDX-License-Identifier: AGPL-3.0-only

//! Device-side BF16 MoE for the MTP drafter.
//!
//! Production Qwen3.6 MTP is BF16 (`--mtp-quantization bf16`). The generic
//! per-expert loop D2Hs top-k indices and launches ~40 GEMVs with
//! token-dependent weight pointers — that slice is ~2.3 ms/step on GB10
//! and cannot live in a CUDA graph. These fused kernels read expert ids
//! from device memory (same recipe as `MoeLayer`'s BF16 decode path).

use super::{MtpHead, ProjectionWeight};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::DenseWeight;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// Kill switch `ATLAS_NO_MTP_BF16_MOE_FUSED`. ON unless the value is exactly
/// `"1"`. `=0` / empty / unset do **not** disable.
pub const DISABLE_ENV: &str = "ATLAS_NO_MTP_BF16_MOE_FUSED";

pub fn mtp_bf16_moe_fused_from(no_fused: Option<&str>) -> bool {
    no_fused != Some("1")
}

pub(super) struct MtpBf16MoeFused {
    gate_ptrs: DevicePtr,
    up_ptrs: DevicePtr,
    down_ptrs: DevicePtr,
    gate_up_k: KernelHandle,
    silu_down_k: KernelHandle,
}

impl MtpBf16MoeFused {
    pub(super) fn try_build(
        experts: &[(ProjectionWeight, ProjectionWeight, ProjectionWeight)],
        gpu: &dyn GpuBackend,
    ) -> Option<Self> {
        if !mtp_bf16_moe_fused_from(std::env::var(DISABLE_ENV).ok().as_deref()) {
            tracing::info!("MTP BF16 MoE fused OFF (ATLAS_NO_MTP_BF16_MOE_FUSED=1)");
            return None;
        }
        let gate_up_k = crate::layers::try_kernel(
            gpu,
            "moe_shared_expert_fused_bf16",
            "moe_expert_gate_up_shared_bf16",
        );
        let silu_down_k = crate::layers::try_kernel(
            gpu,
            "moe_shared_expert_fused_bf16",
            "moe_expert_silu_down_shared_bf16",
        );
        if gate_up_k.0 == 0 || silu_down_k.0 == 0 {
            tracing::warn!("MTP BF16 MoE fused kernels missing — keeping host expert loop");
            return None;
        }
        let mut gates = Vec::with_capacity(experts.len());
        let mut ups = Vec::with_capacity(experts.len());
        let mut downs = Vec::with_capacity(experts.len());
        for (g, u, d) in experts {
            gates.push(as_bf16(g)?);
            ups.push(as_bf16(u)?);
            downs.push(as_bf16(d)?);
        }
        let built = (|| -> Result<Self> {
            Ok(Self {
                gate_ptrs: crate::layers::moe::build_bf16_ptr_table(&gates, gpu)?,
                up_ptrs: crate::layers::moe::build_bf16_ptr_table(&ups, gpu)?,
                down_ptrs: crate::layers::moe::build_bf16_ptr_table(&downs, gpu)?,
                gate_up_k,
                silu_down_k,
            })
        })();
        match built {
            Ok(v) => {
                tracing::info!(
                    "MTP BF16 MoE fused: device-side expert dispatch ({} experts, graphable)",
                    experts.len()
                );
                Some(v)
            }
            Err(e) => {
                tracing::warn!("MTP BF16 MoE fused ptr-table build failed ({e:#}) — host loop");
                None
            }
        }
    }
}

fn as_bf16(proj: &ProjectionWeight) -> Option<DenseWeight> {
    match proj {
        ProjectionWeight::Bf16(w) => Some(*w),
        _ => None,
    }
}

impl MtpHead {
    /// Gate → top-k → fused BF16 expert FFN → blend. Zero D2H.
    pub(super) fn moe_forward_bf16_fused(
        &self,
        fused: &MtpBf16MoeFused,
        input: DevicePtr,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;

        let gate_logits = ctx.buffers.gate_logits();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k.unwrap(),
            input,
            &self.moe_gate,
            gate_logits,
            num_experts,
            h,
            stream,
        )?;

        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(top_k as usize * 4);
        ops::moe_topk_softmax(
            ctx.gpu,
            self.moe_topk_k.unwrap(),
            gate_logits,
            indices_dev,
            weights_dev,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            stream,
        )?;

        let (sh_gate, sh_up, sh_down) = self.moe_shared_generic.as_ref().unwrap();
        let sh_gate_w = as_bf16(sh_gate)
            .ok_or_else(|| anyhow::anyhow!("MTP fused MoE expected BF16 shared gate"))?;
        let sh_up_w = as_bf16(sh_up)
            .ok_or_else(|| anyhow::anyhow!("MTP fused MoE expected BF16 shared up"))?;
        let sh_down_w = as_bf16(sh_down)
            .ok_or_else(|| anyhow::anyhow!("MTP fused MoE expected BF16 shared down"))?;

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let shared_out = ctx.buffers.attn_output();

        ops::moe_expert_gate_up_shared_bf16(
            ctx.gpu,
            fused.gate_up_k,
            input,
            fused.gate_ptrs,
            expert_gate_out,
            fused.up_ptrs,
            expert_up_out,
            indices_dev,
            sh_gate_w.weight,
            shared_gate_scratch,
            sh_up_w.weight,
            shared_up_scratch,
            inter,
            h,
            top_k,
            stream,
        )?;
        ops::moe_expert_silu_down_shared_bf16(
            ctx.gpu,
            fused.silu_down_k,
            expert_gate_out,
            expert_up_out,
            fused.down_ptrs,
            expert_down_out,
            indices_dev,
            shared_gate_scratch,
            shared_up_scratch,
            sh_down_w.weight,
            shared_out,
            h,
            inter,
            top_k,
            stream,
        )?;

        let output = ctx.buffers.moe_output();
        ops::moe_weighted_sum_blend(
            ctx.gpu,
            self.moe_weighted_sum_blend_k.unwrap(),
            output,
            expert_down_out,
            weights_dev,
            shared_out,
            input,
            self.shared_expert_gate.weight,
            h,
            top_k,
            h,
            stream,
        )?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_kill_switch_is_exactly_one() {
        assert!(mtp_bf16_moe_fused_from(None), "unset → ON");
        assert!(mtp_bf16_moe_fused_from(Some("0")), "`=0` is NOT off");
        assert!(mtp_bf16_moe_fused_from(Some("")), "empty is NOT off");
        assert!(!mtp_bf16_moe_fused_from(Some("1")), "`=1` is the kill");
    }
}
