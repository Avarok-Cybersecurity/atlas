// SPDX-License-Identifier: AGPL-3.0-only

//! `MoeLayer::batched_gate_logits` — the routing-logits phase of `forward_batched`,
//! split out for the LoC budget. Computes the gate GEMM `[N, num_experts]`, applies
//! the router (`mlp.gate`) LoRA fold in place before top-k, and returns the gate
//! buffer plus the FP32-gate flags the caller's top-k + dispatch need.

use super::*;

impl MoeLayer {
    /// Compute the batched routing logits and fold the router LoRA delta in place.
    /// Returns `(gate_logits, fp32_gate, gate_elem)` — the expert-dispatch phase in
    /// `forward_batched` reads all three. `router_in` is fully internal here.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn batched_gate_logits(
        &self,
        input: DevicePtr,
        n: u32,
        h: u32,
        num_experts: u32,
        row_adapter_base: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, bool, usize)> {
        // FP32 gate path (ATLAS_FP32_GATE): keep the router GEMM accumulator in
        // FP32 through top-K so two experts whose logits differ by less than a
        // BF16 ULP no longer flip routing (the cross-compiler routing-cascade
        // trigger on gfx1151). Only the softmax-routed dense-gate path is
        // covered — the NVFP4 gate and the sigmoid+bias path keep BF16. Falls
        // back to BF16 if the f32 kernels are absent on this target.
        // ATLAS_FP32_ROUTING: the SSM-side norm already wrote an FP32 router_in
        // (residual_add_rms_norm_gatef32 → moe_router_in_f32); the gate GEMM
        // reads it at full precision via dense_gemm_f32in. Supersedes the
        // gate-only ATLAS_FP32_GATE (which keeps the BF16 router_in but f32 gate
        // accumulation). Either way the gate logits + top-K run in FP32.
        let fp32_routing = self.fp32_routing_active();
        let fp32_gate = fp32_routing
            || (self.gate_nvfp4.is_none()
                && self.correction_bias_dev.is_none()
                && self.dense_gemm_f32out.0 != 0
                && self.moe_topk_f32.0 != 0
                && std::env::var("ATLAS_FP32_GATE").as_deref() == Ok("1"));
        let gate_elem = if fp32_gate { 4usize } else { 2usize };

        // Gemma-4 router pre-norm (no-op for other models).
        let router_in = self.router_input(input, n, h, ctx, stream)?;
        // Gate GEMM: [N, H] × [H, num_experts] → [N, num_experts]
        let gate_logits = if fp32_gate {
            ctx.buffers.gate_logits_f32() // [N, num_experts] FP32
        } else {
            ctx.buffers.gate_logits() // [N, num_experts] BF16
        };
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                router_in,
                nvfp4,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        } else if fp32_routing {
            // FP32 router_in (from residual_add_rms_norm_gatef32) × bf16 gate.
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_f32in,
                ctx.buffers.moe_router_in_f32(),
                &self.weights.gate,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                if fp32_gate {
                    self.dense_gemm_f32out
                } else {
                    self.dense_gemm
                },
                router_in,
                &self.weights.gate,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        }
        // Routing-divergence diagnostic (no-op unless ATLAS_DUMP_EXPERT_IDS=1):
        // last-token gate logits, so the batched path can be compared to gb10
        // the same way the grouped paths are (HIP MoE routing-flip bisection).
        // The dump reads BF16; skip it on the FP32-gate path.
        if !fp32_gate {
            super::dump::dump_gate_logits(ctx.gpu, stream, gate_logits, n, num_experts)?;
        }

        // SOLID Incr-4 (router): fold the router (mlp.gate) LoRA delta onto the
        // whole-batch gate_logits IN PLACE, BEFORE top-k — one launch over all N
        // tokens (the gate GEMM already produced every row), route-agnostic via
        // the device per-row `row_adapter` map (base rows no-op). No-op when no
        // router adapter is installed; refuses the FP32-gate path (no BF16-ULP
        // oracle). Multi-adapter `Refuse` is bailed upstream, pre-capture. This is
        // the batched analogue of the prefill/single-seq `apply_router_lora_prefill`
        // call site, and bit-identical to the single-stream decode router fold.
        self.apply_router_lora_batched(
            router_in,
            gate_logits,
            n,
            row_adapter_base,
            fp32_gate,
            ctx,
            stream,
        )?;

        Ok((gate_logits, fp32_gate, gate_elem))
    }
}
