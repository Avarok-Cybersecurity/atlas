// SPDX-License-Identifier: AGPL-3.0-only

//! ATLAS_FP8_W8A8 sub-paths of `MoeLayer::forward_prefill_fp8`.
//!
//! Hoisted from `forward_prefill_fp8.rs` to keep that file under the 500
//! LoC cap. These methods carry the per-token-quant + W8A8 grouped-GEMM
//! variant (vLLM-equivalent) for the shared expert, the routed gate+up
//! grouped GEMM, and the routed down grouped GEMM. Each mirrors the
//! original inline `force_w8a8` block 1:1 — same kernel launch order,
//! same scratch alloc/free, same synchronize points.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::*;
use crate::weight_map::Fp8ExpertWeight;

impl MoeLayer {
    /// W8A8 shared-expert path (gate/up/down) for FP8 prefill.
    ///
    /// Writes the dense shared-expert down-projection into
    /// `ctx.buffers.attn_output()`. `n` = num_tokens, `h` = hidden_size,
    /// `shared_inter` = shared_expert_intermediate_size.
    pub(super) fn w8a8_shared_expert(
        &self,
        input: DevicePtr,
        sh: &Fp8ExpertWeight,
        n: u32,
        h: u32,
        shared_inter: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let shared_gate_out = ctx.buffers.ssm_deinterleaved();
        let shared_up_out = ctx.buffers.ssm_qkvz();
        let m_us: usize = n as usize;
        let a_fp8_bytes: usize = m_us * h as usize;
        let a_scale_bytes: usize = m_us * (h as usize / 128) * 4;
        let input_fp8 = ctx.gpu.alloc(a_fp8_bytes)?;
        let input_scale = ctx.gpu.alloc(a_scale_bytes)?;
        ops::per_token_group_quant_fp8(
            ctx.gpu,
            self.per_token_group_quant_fp8_k,
            input,
            input_fp8,
            input_scale,
            n,
            h,
            stream,
        )?;
        ops::fp8_gemm_t_blockscaled(
            ctx.gpu,
            self.fp8_gemm_t_blockscaled_k,
            input_fp8,
            input_scale,
            sh.gate_proj.weight,
            sh.gate_proj.row_scale,
            shared_gate_out,
            n,
            shared_inter,
            h,
            stream,
        )?;
        ops::fp8_gemm_t_blockscaled(
            ctx.gpu,
            self.fp8_gemm_t_blockscaled_k,
            input_fp8,
            input_scale,
            sh.up_proj.weight,
            sh.up_proj.row_scale,
            shared_up_out,
            n,
            shared_inter,
            h,
            stream,
        )?;
        ctx.gpu.synchronize(stream)?;
        ctx.gpu.free(input_fp8)?;
        ctx.gpu.free(input_scale)?;
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            shared_gate_out,
            shared_up_out,
            shared_gate_out,
            n * shared_inter,
            stream,
        )?;
        let shared_down_out = ctx.buffers.attn_output();
        // Quant the post-silu intermediate (K=shared_inter)
        let a2_bytes: usize = m_us * shared_inter as usize;
        let a2_scale_bytes: usize = m_us * (shared_inter as usize / 128) * 4;
        let down_in_fp8 = ctx.gpu.alloc(a2_bytes)?;
        let down_in_scale = ctx.gpu.alloc(a2_scale_bytes)?;
        ops::per_token_group_quant_fp8(
            ctx.gpu,
            self.per_token_group_quant_fp8_k,
            shared_gate_out,
            down_in_fp8,
            down_in_scale,
            n,
            shared_inter,
            stream,
        )?;
        ops::fp8_gemm_t_blockscaled(
            ctx.gpu,
            self.fp8_gemm_t_blockscaled_k,
            down_in_fp8,
            down_in_scale,
            sh.down_proj.weight,
            sh.down_proj.row_scale,
            shared_down_out,
            n,
            h,
            shared_inter,
            stream,
        )?;
        ctx.gpu.synchronize(stream)?;
        ctx.gpu.free(down_in_fp8)?;
        ctx.gpu.free(down_in_scale)?;
        Ok(())
    }

    /// W8A8 routed gate+up grouped GEMM. Quantizes `input` once and runs
    /// both projections, writing `expert_gate_out` / `expert_up_out`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn w8a8_routed_gate_up(
        &self,
        input: DevicePtr,
        gp: &Fp8ExpertPtrTable,
        up: &Fp8ExpertPtrTable,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        num_tokens: usize,
        num_experts: u32,
        inter: u32,
        h: u32,
        max_m_tiles: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Quant input [num_tokens, h] → input_fp8 + input_a_scale ONCE,
        // reuse for both gate and up.
        let m = num_tokens;
        let a_fp8_bytes = m * h as usize;
        let a_scale_bytes = m * (h as usize / 128) * 4;
        let input_fp8 = ctx.gpu.alloc(a_fp8_bytes)?;
        let input_a_scale = ctx.gpu.alloc(a_scale_bytes)?;
        ops::per_token_group_quant_fp8(
            ctx.gpu,
            self.per_token_group_quant_fp8_k,
            input,
            input_fp8,
            input_a_scale,
            m as u32,
            h,
            stream,
        )?;
        ops::moe_w8a8_grouped_gemm(
            ctx.gpu,
            self.moe_w8a8_grouped_gemm_k,
            input_fp8,
            input_a_scale,
            gp.weight_ptrs,
            gp.scale_ptrs,
            expert_gate_out,
            expert_offsets,
            sorted_token_ids,
            num_experts,
            inter,
            h,
            max_m_tiles,
            stream,
        )?;
        ops::moe_w8a8_grouped_gemm(
            ctx.gpu,
            self.moe_w8a8_grouped_gemm_k,
            input_fp8,
            input_a_scale,
            up.weight_ptrs,
            up.scale_ptrs,
            expert_up_out,
            expert_offsets,
            sorted_token_ids,
            num_experts,
            inter,
            h,
            max_m_tiles,
            stream,
        )?;
        ctx.gpu.synchronize(stream)?;
        ctx.gpu.free(input_fp8)?;
        ctx.gpu.free(input_a_scale)?;
        Ok(())
    }

    /// W8A8 routed down grouped GEMM. Applies SiLU+mul, quantizes the
    /// permuted intermediate, and writes `expert_down_out`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn w8a8_routed_down(
        &self,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        dp: &Fp8ExpertPtrTable,
        expert_offsets: DevicePtr,
        total_expanded: u32,
        num_experts: u32,
        inter: u32,
        h: u32,
        max_m_tiles: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            expert_gate_out,
            expert_up_out,
            expert_gate_out,
            total_expanded * inter,
            stream,
        )?;
        // Quant the permuted post-silu intermediate. Length is
        // total_expanded, K is `inter` (down_proj input dim).
        let m: usize = total_expanded as usize;
        let a_fp8_bytes: usize = m * inter as usize;
        let a_scale_bytes: usize = m * (inter as usize / 128) * 4;
        let down_in_fp8 = ctx.gpu.alloc(a_fp8_bytes)?;
        let down_in_scale = ctx.gpu.alloc(a_scale_bytes)?;
        ops::per_token_group_quant_fp8(
            ctx.gpu,
            self.per_token_group_quant_fp8_k,
            expert_gate_out,
            down_in_fp8,
            down_in_scale,
            m as u32,
            inter,
            stream,
        )?;
        ops::moe_w8a8_grouped_gemm(
            ctx.gpu,
            self.moe_w8a8_grouped_gemm_k,
            down_in_fp8,
            down_in_scale,
            dp.weight_ptrs,
            dp.scale_ptrs,
            expert_down_out,
            expert_offsets,
            spark_runtime::gpu::DevicePtr(0),
            num_experts,
            h,
            inter,
            max_m_tiles,
            stream,
        )?;
        ctx.gpu.synchronize(stream)?;
        ctx.gpu.free(down_in_fp8)?;
        ctx.gpu.free(down_in_scale)?;
        Ok(())
    }
}
