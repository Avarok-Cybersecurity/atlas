// SPDX-License-Identifier: AGPL-3.0-only

//! Shared-expert phase of `MoeLayer::forward_prefill`.
//!
//! Hoisted from `forward_prefill.rs` to keep that file under the 500 LoC
//! cap. The single entry point [`MoeLayer::run_shared_expert_prefill`]
//! mirrors the original block 1:1 — same control flow, same kernel
//! launches, same buffer wiring.

use super::*;

impl MoeLayer {
    /// Shared-expert path of the prefill pipeline (gate + up GEMM → SiLU →
    /// down GEMM). Runs sequentially on the supplied `aux` stream when
    /// `use_overlap == false`; otherwise issues an event so the routed
    /// path can wait on completion.
    ///
    /// Skips entirely when `shared_inter == 0` (e.g. Qwen3-VL-30B has no
    /// shared expert). Launching kernels with N=0 returns
    /// CUDA_ERROR_INVALID_VALUE (grid.x=0).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_shared_expert_prefill(
        &self,
        input: DevicePtr,
        n: u32,
        h: u32,
        shared_inter: u32,
        aux: u64,
        stream: u64,
        use_overlap: bool,
        ctx: &ForwardContext,
    ) -> Result<()> {
        if shared_inter == 0 {
            return Ok(());
        }
        if use_overlap {
            // Ensure secondary stream sees `input` (produced by prior default-stream work)
            ctx.gpu.record_event(self.event_a, stream)?;
            ctx.gpu.stream_wait_event(aux, self.event_a)?;
        }

        // Shared gate + up GEMM on aux stream
        let shared_gate_out = ctx.buffers.ssm_deinterleaved();
        let shared_up_out = ctx.buffers.ssm_qkvz();
        if let (Some(sg_fp8), Some(su_fp8)) = (self.shared_gate_fp8, self.shared_up_fp8) {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                input,
                sg_fp8,
                shared_gate_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                input,
                su_fp8,
                shared_up_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
        } else if let (Some(sg), Some(su), Some(_sd)) =
            (&self.shared_gate_t, &self.shared_up_t, &self.shared_down_t)
        {
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t,
                input,
                sg,
                shared_gate_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t,
                input,
                su,
                shared_up_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                input,
                &self.weights.shared_expert.gate_proj,
                shared_gate_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                input,
                &self.weights.shared_expert.up_proj,
                shared_up_out,
                n,
                shared_inter,
                h,
                aux,
            )?;
        }

        // Shared activation (SiLU or GeGLU) + down GEMM on aux stream
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            shared_gate_out,
            shared_up_out,
            shared_gate_out,
            n * shared_inter,
            aux,
        )?;
        let shared_down_out = ctx.buffers.attn_output();
        if let Some(sd_fp8) = self.shared_down_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                shared_gate_out,
                sd_fp8,
                shared_down_out,
                n,
                h,
                shared_inter,
                aux,
            )?;
        } else if let Some(sd) = &self.shared_down_t {
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t,
                shared_gate_out,
                sd,
                shared_down_out,
                n,
                h,
                shared_inter,
                aux,
            )?;
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                shared_gate_out,
                &self.weights.shared_expert.down_proj,
                shared_down_out,
                n,
                h,
                shared_inter,
                aux,
            )?;
        }

        if use_overlap {
            ctx.gpu.record_event(self.event_b, aux)?;
        }
        Ok(())
    }
}
