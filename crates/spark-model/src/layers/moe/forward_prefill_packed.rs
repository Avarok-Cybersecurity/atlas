// SPDX-License-Identifier: AGPL-3.0-only

//! Keep-packed GGUF Q4_K_M MoE prefill compute (Laguna-S-2.1).
//!
//! Drop-in replacement for [`MoeLayer::run_routed_grouped_gemm`] when the layer
//! holds keep-packed experts ([`MoeWeights::packed_experts`]). It reuses the
//! caller's routing (gate GEMM + top-k + `moe_sort_by_expert` → `expert_offsets`
//! / `sorted_token_ids`) and the caller's post-blend (`moe_unpermute_reduce_
//! indexed`); only the per-expert COMPUTE differs: native W4A8 `q4k_mmq` on the
//! packed Q4_K gate/up blocks (weights never dequant to BF16 — mirroring the
//! NVFP4 grouped path), and a native Q6_K MMQ for `down`. It writes
//! `expert_down_out` in the SAME sorted layout the blend expects, so the
//! surrounding forward_prefill body is unchanged.

use super::*;

impl MoeLayer {
    /// Routed-compute arm selection for `forward_prefill` (hoisted there from —
    /// 500-LoC cap): keep-packed GGUF experts take the native q4k_mmq arm below,
    /// everything else the NVFP4 grouped GEMM in forward_prefill_routed.rs.
    /// Same routing before, same post-blend after — only expert_down_out's
    /// producer differs.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_routed_compute(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        sorted_expert_ids: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        top_k: u32,
        num_tokens: usize,
        ne: usize,
        t0: &mut Option<std::time::Instant>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.weights.packed_experts.is_some() {
            self.run_routed_grouped_gemm_packed(
                expert_input,
                expert_offsets,
                sorted_token_ids,
                sorted_expert_ids,
                n,
                h,
                inter,
                num_experts,
                top_k,
                ctx,
                stream,
            )
        } else {
            self.run_routed_grouped_gemm(
                expert_input,
                expert_offsets,
                sorted_token_ids,
                n,
                h,
                inter,
                num_experts,
                top_k,
                num_tokens,
                ne,
                t0,
                ctx,
                stream,
            )
        }
    }

    /// Native keep-packed Q4_K/Q6_K routed compute. Writes routed expert outputs
    /// into `ctx.buffers.expert_down_out()` in sorted (permuted) order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_routed_grouped_gemm_packed(
        &self,
        expert_input: DevicePtr,      // [n, h] BF16 (normed MoE input)
        expert_offsets: DevicePtr,    // [ne+1] i32, device — sorted cumulative counts
        sorted_token_ids: DevicePtr,  // [n*top_k] i32, device
        sorted_expert_ids: DevicePtr, // [n*top_k] i32, device — expert per sorted slot
        n: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        top_k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let packed = self.weights.packed_experts.as_ref().ok_or_else(|| {
            anyhow::anyhow!("run_routed_grouped_gemm_packed: layer has no packed_experts")
        })?;
        let total_expanded = n * top_k;

        // Fused n=1 DECODE path: at small te (decode, n≈1) the grouped MMQ wastes
        // ~2000 CTAs (127/128 of each 128-row tensor-core tile idle). Two
        // output-tiled GEMV kernels that gather+dequant directly (no permute, no
        // q8_1 quantize, no tensor-core-tile waste) are ~18% faster (14→~16.5
        // tok/s) and stay CUDA-graph-legal. Host-known `n` (decode is always n=1
        // per graph capture) keeps the captured graph on ONE arm. Default-on;
        // kill-switch = `ctx.dispatch.packed_decode_fused` (resolved once at
        // model build — no env read here, so the captured branch is stable).
        const DECODE_FUSED_MAX_SLOTS: u32 = 32; // ~n<=2 at top_k=10
        // A resident gate/up expert LoRA cannot be folded on the fused arm: the
        // decode kernel fuses gate+up+silu and writes ONLY the post-silu product
        // (moe_q4k_decode_fused.cu) — there is no pre-silu gate/up buffer to fold
        // onto. Force the grouped arm (which materializes separate
        // expert_gate_out/expert_up_out) whenever a gate/up delta is INSTALLED.
        // Gate on RESIDENCY (self.lora routes), NOT the per-request Fold/Skip
        // route, so a resident adapter set keeps the captured decode graph on ONE
        // arm (matches the fused kernel's single-arm capture contract). Down-only
        // and router-only adapters keep the fast fused path (their folds run in
        // the caller — forward_prefill.rs router/down — unaffected).
        let gateup_adapter = self
            .lora
            .as_ref()
            .is_some_and(|l| l.gate_route.is_some() || l.up_route.is_some());
        if total_expanded <= DECODE_FUSED_MAX_SLOTS
            && !gateup_adapter
            && self.q4k_decode_gate_up_k.0 != 0
            && self.q4k_decode_down_k.0 != 0
            && ctx.dispatch.packed_decode_fused
        {
            let down_is_q6k = matches!(packed[0].down, crate::weight_map::QuantWeight::PackedQ6(_));
            let down_base = match &packed[0].down {
                crate::weight_map::QuantWeight::PackedQ4(w4) => w4.weight,
                crate::weight_map::QuantWeight::PackedQ6(w6) => w6.weight,
                other => anyhow::bail!("packed MoE down_proj: unexpected variant {other:?}"),
            };
            ops::moe_q4k_decode_fused(
                ctx.gpu,
                self.q4k_decode_gate_up_k,
                self.q4k_decode_down_k,
                expert_input,
                packed[0].gate.weight,
                packed[0].up.weight,
                down_base,
                sorted_token_ids,
                sorted_expert_ids,
                ctx.buffers.expert_gate_out(), // s_act staging
                ctx.buffers.expert_down_out(),
                h,
                inter,
                total_expanded,
                down_is_q6k,
                stream,
            )?;
            return Ok(());
        }

        // All scratch is persistent arena — NO per-call alloc/free, so the whole
        // arm is CUDA-graph-capture-legal (decode). `permuted` aliases
        // `expert_down_out`: the gathered activations are consumed (quantized to
        // q8) before the grouped down projection overwrites it with the real
        // output, so the two uses never overlap in time. `q8` is a dedicated
        // arena buffer sized for the whole sorted [k_max*top_k, h] tile.
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let permuted = expert_down_out;
        let q8 = ctx.buffers.moe_grouped_q8();

        // 1. Gather rows into expert-contiguous (sorted) order.
        ops::moe_permute_tokens(
            ctx.gpu,
            self.moe_permute_tokens_k,
            expert_input,
            permuted,
            sorted_token_ids,
            h,
            total_expanded,
            stream,
        )?;

        // 2. Gate/up: quantize the whole sorted activation buffer ONCE, then run
        // ONE fused device-side grouped GEMM (grid.z=num_experts) that reads
        // per-expert row ranges from `expert_offsets` on-device and writes
        // sorted [total_expanded, inter] gate AND up outputs. Weights are the
        // contiguous per-proj expert stacks — pass expert 0's base pointer.
        ops::quantize_act_q8_1(
            ctx.gpu,
            self.q4k_quant_act_k,
            permuted,
            q8,
            total_expanded,
            h,
            stream,
        )?;
        let gate_base = packed[0].gate.weight;
        let up_base = packed[0].up.weight;
        // FUSED gate+up: ONE grouped launch; each CTA computes both projections
        // (shared empty-expert early-return + ids setup → half the scheduled CTAs
        // vs two separate launches). Numerically identical to the two-call path.
        ops::q4k_grouped_gemm_gate_up(
            ctx.gpu,
            self.q4k_grouped_gate_up_nc_k,
            self.q4k_grouped_gate_up_wc_k,
            gate_base,
            up_base,
            q8,
            expert_offsets,
            expert_gate_out,
            expert_up_out,
            inter,
            h,
            num_experts,
            total_expanded,
            stream,
        )?;
        // GGUF LoRA parity: fold the routed-expert gate/up_proj LoRA deltas onto
        // the sorted expert_gate_out/expert_up_out BEFORE silu_mul consumes them
        // in place — exact mirror of the NVFP4 arm (forward_prefill_routed.rs).
        // x = token-major expert_input, gathered per sorted row. No-op unless a
        // gate/up delta is installed AND the request routes Fold. The delta is an
        // output-space fold, so it is BF16-ULP identical to the validated NVFP4
        // fold regardless of the Q4_K/Q6_K base quant. Router + expert-down folds
        // already fire in the caller, outside this packed early-return.
        self.apply_expert_lora_prefill_gateup(
            expert_gate_out,
            expert_up_out,
            expert_input,
            expert_offsets,
            sorted_token_ids,
            total_expanded,
            ctx,
            stream,
        )?;
        // SiLU(gate) * up over the whole sorted buffer (one launch).
        ops::silu_mul(
            ctx.gpu,
            self.moe_silu_mul,
            expert_gate_out,
            expert_up_out,
            expert_gate_out,
            total_expanded * inter,
            stream,
        )?;

        // 3. Down: one DEVICE-SIDE GROUPED GEMM over the post-silu buffer. Q4_K_M
        // mixes the down projection Q4_K vs Q6_K PER LAYER (all experts in a layer
        // share one ggml type — the GGUF stores down_exps as a single tensor), so
        // the whole layer takes one grouped launch of the matching type. Q6_K stays
        // packed (native Q6_K MMQ) — no BF16 dequant. Activations quantize once:
        // Q4_K wants the DS4 q8_1 layout, Q6_K wants D4.
        match &packed[0].down {
            crate::weight_map::QuantWeight::PackedQ4(w4) => {
                ops::quantize_act_q8_1(
                    ctx.gpu,
                    self.q4k_quant_act_k,
                    expert_gate_out, // [total_expanded, inter] post-silu
                    q8,
                    total_expanded,
                    inter,
                    stream,
                )?;
                ops::q4k_grouped_gemm(
                    ctx.gpu,
                    self.q4k_grouped_nc_k,
                    self.q4k_grouped_wc_k,
                    w4.weight,
                    q8,
                    expert_offsets,
                    expert_down_out,
                    h,
                    inter,
                    num_experts,
                    total_expanded,
                    stream,
                )?;
            }
            crate::weight_map::QuantWeight::PackedQ6(w6) => {
                ops::quantize_act_q8_1(
                    ctx.gpu,
                    self.q4k_quant_act_d4_k,
                    expert_gate_out, // [total_expanded, inter] post-silu
                    q8,
                    total_expanded,
                    inter,
                    stream,
                )?;
                ops::q4k_grouped_gemm(
                    ctx.gpu,
                    self.q6k_grouped_nc_k,
                    self.q6k_grouped_wc_k,
                    w6.weight,
                    q8,
                    expert_offsets,
                    expert_down_out,
                    h,
                    inter,
                    num_experts,
                    total_expanded,
                    stream,
                )?;
            }
            other => anyhow::bail!("packed MoE down_proj: unexpected variant {other:?}"),
        }

        Ok(())
    }
}
