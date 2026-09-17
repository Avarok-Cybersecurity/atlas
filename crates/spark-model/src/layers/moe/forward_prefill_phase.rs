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
    /// `AVAROK_MOE_SHARED_CUTLASS=1`: run the shared expert's three projections
    /// on the same native CUTLASS NVFP4 (W4A4) path the ROUTED experts already
    /// use, instead of `w4a16_gemm_n128`.
    ///
    /// Measured on qwen4_exp (7.8K chunk, 48 layers, TP=2 x EP=2): the routed
    /// experts do 18.5 TFLOP in 571 ms (32.4 TFLOP/s) while the shared expert
    /// does 3.70 TFLOP in the SAME 571 ms (6.5 TFLOP/s) — five times less
    /// arithmetic for the same wall clock, because W4A16 dequantises to BF16
    /// and gives up the FP4 tensor cores. The shared expert was 40.4% of the
    /// whole MoE block on that profile.
    ///
    /// W4A4 quantises the ACTIVATIONS, so this is not bit-exact — the same
    /// trade the routed experts already ship with under
    /// AVAROK_HOLO_MOE_GROUPED_CUTLASS. Opt-in until it has agentic receipts.
    fn shared_cutlass_enabled() -> bool {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| std::env::var("AVAROK_MOE_SHARED_CUTLASS").as_deref() == Ok("1"))
    }

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

        let shared_gate_out = ctx.buffers.ssm_deinterleaved();
        let shared_up_out = ctx.buffers.ssm_qkvz();
        let shared_down_out = ctx.buffers.attn_output();
        if self.run_bf16_shared_expert(
            input,
            n,
            h,
            shared_inter,
            shared_gate_out,
            shared_up_out,
            shared_down_out,
            ctx,
            aux,
        )? {
            if use_overlap {
                ctx.gpu.record_event(self.event_b, aux)?;
            }
            return Ok(());
        }

        // Shared gate + up GEMM on aux stream
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
            // CUTLASS NVFP4 arm. `shared_*_t` are already the transposed
            // layout `cutlass_nvfp4_proj` wants (built by
            // `transpose_for_gemm`), so this is a swap, not a repack. Gated on
            // n > 64 like the routed grouped GEMM: below that the pack and
            // launch cost more than the tensor cores save.
            if Self::shared_cutlass_enabled() && n > 64 {
                {
                    static SAID: std::sync::Once = std::sync::Once::new();
                    SAID.call_once(|| {
                        tracing::info!(
                            n,
                            shared_inter,
                            h,
                            "MoE shared expert: CUTLASS NVFP4 (W4A4)"
                        )
                    });
                }
                ops::cutlass_nvfp4_proj(ctx, input, sg, shared_gate_out, n, shared_inter, h, aux)?;
                ops::cutlass_nvfp4_proj(ctx, input, su, shared_up_out, n, shared_inter, h, aux)?;
            } else {
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
            }
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
            if Self::shared_cutlass_enabled() && n > 64 {
                ops::cutlass_nvfp4_proj(
                    ctx,
                    shared_gate_out,
                    sd,
                    shared_down_out,
                    n,
                    h,
                    shared_inter,
                    aux,
                )?;
            } else {
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
            }
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
