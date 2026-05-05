// SPDX-License-Identifier: AGPL-3.0-only

//! predequant_for_prefill, set_fp8_experts, router_input.

use super::*;

impl MoeLayer {
    /// Pre-dequant dense (non-expert) NVFP4 weights to FP8 for zero-overhead prefill.
    ///
    /// Only affects gate GEMM and shared expert GEMMs.  Expert weights stay NVFP4
    /// (they're bandwidth-bound so FP8 wouldn't help).
    pub fn predequant_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let h = config.hidden_size;
        let shared_inter = config.shared_expert_intermediate_size;
        let num_experts = config.num_experts;
        let predequant_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;

        // Pre-dequant gate weight: [num_experts, H] → FP8 [num_experts, H]
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            self.gate_fp8 =
                Some(nvfp4.predequant_to_fp8(gpu, predequant_k, num_experts, h, stream)?);
        }

        // Pre-dequant shared expert weights
        if !self.weights.shared_expert.gate_proj.is_null() && shared_inter > 0 {
            self.shared_gate_fp8 = Some(self.weights.shared_expert.gate_proj.predequant_to_fp8(
                gpu,
                predequant_k,
                shared_inter,
                h,
                stream,
            )?);
            self.shared_up_fp8 = Some(self.weights.shared_expert.up_proj.predequant_to_fp8(
                gpu,
                predequant_k,
                shared_inter,
                h,
                stream,
            )?);
            self.shared_down_fp8 = Some(self.weights.shared_expert.down_proj.predequant_to_fp8(
                gpu,
                predequant_k,
                h,
                shared_inter,
                stream,
            )?);
        }

        Ok(())
    }

    /// Set FP8 expert weights for native FP8 dispatch.
    ///
    /// Builds device-side pointer tables from FP8 expert weights so the
    /// fused FP8 MoE kernel can index by expert_id at dispatch time.
    /// Also stores the shared expert FP8 weights for direct pointer passing.
    pub fn set_fp8_experts(
        &mut self,
        experts: &[Fp8ExpertWeight],
        shared_expert: Fp8ExpertWeight,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        self.fp8_gate_weight_ptrs = Some(build_fp8_ptr_table(experts, |e| &e.gate_proj, gpu)?);
        self.fp8_up_weight_ptrs = Some(build_fp8_ptr_table(experts, |e| &e.up_proj, gpu)?);
        self.fp8_down_weight_ptrs = Some(build_fp8_ptr_table(experts, |e| &e.down_proj, gpu)?);
        self.fp8_shared_expert = Some(shared_expert);
        Ok(())
    }

    /// Apply the router pre-normalization (Gemma-4 only) and return the
    /// pointer that should be fed into the gate GEMV. If the MoE has no
    /// router_pre_norm weight, this is a no-op and returns `input` unchanged.
    ///
    /// HF Gemma4TextRouter computes:
    ///   router_input = rms_norm(x) * scale * hidden_size^(-0.5)
    /// We fused `scale * root_size` into a single BF16 weight at load time
    /// so the existing rms_norm kernel applies both steps in one pass.
    ///
    /// The normed output is written to `ctx.buffers.qkv_output()` which is
    /// free at MoE time (the attention block already consumed qkv_output).
    pub(super) fn router_input(
        &self,
        input: DevicePtr,
        num_tokens: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let Some(ref weight) = self.weights.router_pre_norm else {
            return Ok(input);
        };
        let eps = ctx.config.rms_norm_eps as f32;
        let normed = ctx.buffers.qkv_output();
        ops::rms_norm(
            ctx.gpu,
            self.pre_expert_norm_k,
            input,
            weight,
            normed,
            num_tokens,
            h,
            eps,
            stream,
        )?;
        Ok(normed)
    }
}
