// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward_fp8_grouped_decode — cross-row GROUPED FP8 MoE decode.
//!
//! Serves M decode rows (concurrent sequences, or an MTP verify's Σk rows)
//! through ONE routed+shared expert dispatch: rows are grouped by expert with
//! `moe_sort_by_expert`, and the grouped kernels (`fp8_moe_grouped.rs`) give
//! each expert one CTA row that streams its weights once for every row routed
//! to it. This replaces the per-token `forward()` loop that ran the MoE M
//! times per layer — the reason aggregate MoE throughput was flat above C=1
//! (Qwen3.6-35B-A3B-FP8: C=1 60.9 / C=2 69.9 / C=4 70.8 tok/s, 2026-09-22).
//!
//! NUMERICS versus the per-token loop it replaces: the grouped kernels copy
//! the single-token kernels' per-row arithmetic (lane partition, FMA-free
//! `a*w` grouping, 32-lane butterfly), so each row's gate/up/down/blend bytes
//! are identical given the same routing. The routing itself is the batched
//! `dense_gemm`/`w4a16_gemm` + `moe_topk_*_batched` pair the shipped K=2/3
//! verify paths (`forward_k2`/`forward_k3`) already use, which differs from
//! the per-token `dense_gemv` + `moe_topk_*` in FP32 summation order and can
//! flip a razor-margin top-k choice; the GPU oracle
//! (`examples/fp8_moe_grouped_decode_microtest.rs`) therefore feeds both legs
//! the same routing and requires bit-identical output.

use super::*;

/// Widest row count this path serves. Above it the sort scratch and the
/// per-expert pass loop still work, but nothing has been measured there and
/// `forward_prefill_fp8` exists for >64; keep the envelope explicit.
pub const FP8_GROUPED_DECODE_MAX_ROWS: usize = 64;

/// The grouped decode's four kernels, resolved optionally (a zero handle
/// declines the path in [`MoeLayer::fp8_grouped_decode_arena_ok`]).
pub(super) struct GroupedKernels {
    pub gate_up: KernelHandle,
    pub silu_down: KernelHandle,
    pub blend: KernelHandle,
    pub compact: KernelHandle,
}

impl GroupedKernels {
    /// Direct `try_kernel` calls, not a closure: `try_kernel` is
    /// `#[track_caller]`, so each lookup keeps its own line in the audit.
    pub(super) fn resolve(gpu: &dyn GpuBackend) -> Self {
        use super::super::try_kernel;
        const FUSED: &str = "moe_shared_expert_fused_fp8_grouped";
        Self {
            gate_up: try_kernel(gpu, FUSED, "moe_expert_gate_up_shared_fp8_grouped"),
            silu_down: try_kernel(gpu, FUSED, "moe_expert_silu_down_shared_fp8_grouped"),
            blend: try_kernel(
                gpu,
                "moe_fp8_grouped_blend",
                "moe_weighted_sum_blend_fp8_grouped",
            ),
            compact: try_kernel(gpu, FUSED, "moe_fp8_grouped_compact"),
        }
    }
}

/// Kill switch: PRESENCE of `AVAROK_NO_FP8_MOE_GROUPED_DECODE` (any value)
/// restores the per-token loop — the house convention for `AVAROK_NO_*`. It
/// covers both users: the MTP drafter's batched propose (on by default) and
/// the target model's multi-row decode, which additionally needs the opt-in
/// `ModelLevers::moe_fp8_grouped_decode_target` (`FfnComponent`'s gate).
fn fp8_grouped_decode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("AVAROK_NO_FP8_MOE_GROUPED_DECODE").is_none())
}

/// Pure shape admission for the grouped kernels — testable without a GPU.
///
/// * `m` in `2..=FP8_GROUPED_DECODE_MAX_ROWS` (1 row is the single-token path).
/// * `hidden % 16 == 0`: the gate/up loop consumes 16 K-elements per lane
///   iteration (two `uint4` activation loads), the blend 8.
/// * `inter % 8 == 0`: the silu/down loop consumes 8 per iteration (and its
///   float4 activation reads need 16-byte-aligned rows, which `% 8` gives).
/// * The silu/down pass keeps `GROUP_ROWS × inter` FP32 activations in dynamic
///   shared memory next to the 1 KB LUT, under the 48 KB no-opt-in limit.
pub fn fp8_grouped_decode_shape_ok(m: usize, hidden: u32, inter: u32) -> bool {
    const SMEM_NO_OPT_IN: usize = 48 * 1024;
    const LUT_BYTES: usize = 256 * 4;
    (2..=FP8_GROUPED_DECODE_MAX_ROWS).contains(&m)
        && hidden >= 16
        && hidden.is_multiple_of(16)
        && inter >= 8
        && inter.is_multiple_of(8)
        && ops::fp8_grouped_silu_down_smem_bytes(inter) + LUT_BYTES <= SMEM_NO_OPT_IN
}

/// Byte requirements on the arena buffers this path borrows, so a wider batch
/// than the arena was sized for is refused here rather than found by the
/// next buffer over. Pure; the layer predicate feeds it the accessors.
pub(crate) struct GroupedDecodeBufferNeed {
    pub scratch: usize,
    pub gate_logits: usize,
    pub expert_gate_out: usize,
    pub expert_down_out: usize,
    pub shared_inter: usize,
    pub row_hidden: usize,
}

pub(crate) fn grouped_decode_buffer_need(
    m: usize,
    hidden: usize,
    inter: usize,
    num_experts: usize,
    top_k: usize,
) -> GroupedDecodeBufferNeed {
    let te = m * top_k;
    GroupedDecodeBufferNeed {
        // indices [te] u32 + weights [te] f32
        scratch: 2 * te * 4,
        // max(gate logits [m, E] BF16, sort scratch: sorted_token_ids [te] +
        // sorted_expert_ids [te] + expert_offsets [E+1] + token_to_perm [te] +
        // active_experts [cap] + active_count [1])
        gate_logits: (m * num_experts * 2)
            .max(te * 4 * 3 + (num_experts + 1) * 4 + (te.min(num_experts) + 1) * 4),
        expert_gate_out: te * inter * 2,
        expert_down_out: te * hidden * 2,
        // shared gate/up scratch [m, inter] BF16
        shared_inter: m * inter * 2,
        // shared down out / moe_output [m, hidden] BF16
        row_hidden: m * hidden * 2,
    }
}

impl MoeLayer {
    /// The weight, kernel, shape and arena terms of
    /// [`Self::fp8_grouped_decode_ok`] — everything decidable from the layer,
    /// the config and the arena alone, so a width policy
    /// (`MtpHead::propose_batch_max`) can size a batch without a
    /// `ForwardContext`. NOT sufficient on its own: the kill switch, the
    /// FP32-routing lever and expert parallelism are context terms only the
    /// full predicate adds.
    pub fn fp8_grouped_decode_arena_ok(
        &self,
        m: usize,
        cfg: &avarok_core::config::ModelConfig,
        b: &spark_runtime::buffers::BufferArena,
    ) -> bool {
        let h = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let need =
            grouped_decode_buffer_need(m, h, inter, cfg.num_experts, cfg.num_experts_per_tok);
        fp8_grouped_decode_shape_ok(m, h as u32, inter as u32)
            && self.fp8_gate_weight_ptrs.is_some()
            && self.fp8_up_weight_ptrs.is_some()
            && self.fp8_down_weight_ptrs.is_some()
            && self.fp8_shared_expert.is_some()
            && self.moe_expert_gate_up_shared_fp8_grouped_k.0 != 0
            && self.moe_expert_silu_down_shared_fp8_grouped_k.0 != 0
            && self.moe_weighted_sum_blend_fp8_grouped_k.0 != 0
            && self.moe_fp8_grouped_compact_k.0 != 0
            && self.moe_sort_by_expert.0 != 0
            // No fold hooks here: a resident MoE adapter takes forward_batched.
            && self.lora.is_none()
            && self.pre_expert_norm.is_none()
            && self.tid2eid_dev.is_none()
            && !self.has_mixed_bf16_shared_expert()
            // LongCat zero-experts route through a wider router; not wired.
            && self.router_logits_n as usize == cfg.num_experts
            // sqrtsoftplus (DeepSeek-V4) has no proven batched top-k kernel.
            && !(self.correction_bias_dev.is_some() && cfg.scoring_func == "sqrtsoftplus")
            && b.scratch_bytes() >= need.scratch
            && b.gate_logits_bytes() >= need.gate_logits
            && b.expert_gate_out_bytes() >= need.expert_gate_out
            && b.expert_down_out_bytes() >= need.expert_down_out
            && b.logits_bytes() >= need.shared_inter
            && b.ssm_qkvz_bytes() >= need.shared_inter
            && b.attn_output_bytes() >= need.row_hidden
            && b.moe_output_bytes() >= need.row_hidden
    }

    /// True when `forward_fp8_grouped_decode` can serve `m` rows on this layer
    /// under `ctx`: [`Self::fp8_grouped_decode_arena_ok`] plus the kill switch
    /// and the context terms, so callers can branch on it before touching the
    /// pre-FFN norm.
    pub fn fp8_grouped_decode_ok(&self, m: usize, ctx: &ForwardContext) -> bool {
        let cfg = ctx.config;
        fp8_grouped_decode_enabled()
            && self.fp8_grouped_decode_arena_ok(m, cfg, ctx.buffers)
            && !self.fp32_routing_active(ctx.levers)
            && !(ctx.comm.is_some() && cfg.ep_world_size > 1)
    }

    /// Cross-row grouped FP8 MoE for `m` decode rows: `input` is `[m, H]` BF16,
    /// output lands in `moe_output()[0..m]`. Caller must have checked
    /// `fp8_grouped_decode_ok(m, ctx)`; this refuses loudly otherwise.
    pub fn forward_fp8_grouped_decode(
        &self,
        input: DevicePtr,
        m: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.fp8_grouped_decode_ok(m, ctx),
            "forward_fp8_grouped_decode: predicate false for m={m} (caller must gate on it)"
        );
        let (Some(gp), Some(up), Some(dp), Some(sh)) = (
            &self.fp8_gate_weight_ptrs,
            &self.fp8_up_weight_ptrs,
            &self.fp8_down_weight_ptrs,
            &self.fp8_shared_expert,
        ) else {
            unreachable!("fp8_grouped_decode_ok checked the FP8 tables");
        };
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = m as u32;
        let te = m * top_k as usize;

        if ctx.stats.once("log:moe_fp8_grouped_decode") {
            tracing::info!(
                "MoE FP8 grouped decode: cross-row batched routed+shared expert dispatch \
                 active (first use M={m}, top_k={top_k}, experts={num_experts}; one-time log)"
            );
        }

        // 1. Router: [m, H] x [H, E] -> gate_logits [m, E] (batched, as forward_k2/k3).
        let router_in = self.router_input(input, n, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();
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
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm,
                router_in,
                &self.weights.gate,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        }

        // 2. Batched top-k: indices [m*top_k] u32, weights [m*top_k] f32, slot-major.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(te * 4);
        if let Some(bias) = self.correction_bias_dev {
            ops::moe_topk_sigmoid_batched(
                ctx.gpu,
                self.moe_topk_sigmoid_batched_k,
                gate_logits,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                n,
                stream,
            )?;
        } else {
            ops::moe_topk_softmax_batched(
                ctx.gpu,
                self.moe_topk_batched,
                gate_logits,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                n,
                stream,
            )?;
        }

        // 3. Group slots by expert. gate_logits is free once top-k has read it
        //    (same stream), so it hosts the sort scratch exactly as the FP8
        //    prefill path lays it out.
        let ne = num_experts as usize;
        let sorted_token_ids = gate_logits;
        let sorted_expert_ids = gate_logits.offset(te * 4);
        let expert_offsets = gate_logits.offset(te * 4 * 2);
        let token_to_perm = gate_logits.offset(te * 4 * 2 + (ne + 1) * 4);
        ops::moe_sort_by_expert(
            ctx.gpu,
            self.moe_sort_by_expert,
            indices_dev,
            sorted_token_ids,
            sorted_expert_ids,
            expert_offsets,
            token_to_perm,
            te as u32,
            num_experts,
            top_k,
            stream,
        )?;

        // 4. Compact the active experts (fixed cap per M keeps the grids
        //    graph-shape-stable), then one grouped dispatch: gate+up,
        //    silu+down, blend.
        let cap = ops::fp8_grouped_active_cap(n, top_k, num_experts);
        let active_experts = token_to_perm.offset(te * 4);
        let active_count = active_experts.offset(cap as usize * 4);
        ops::moe_fp8_grouped_compact(
            ctx.gpu,
            self.moe_fp8_grouped_compact_k,
            expert_offsets,
            active_experts,
            active_count,
            num_experts,
            stream,
        )?;
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let shared_out = ctx.buffers.attn_output();
        let output = ctx.buffers.moe_output();

        ops::moe_expert_gate_up_shared_fp8_grouped(
            ctx.gpu,
            self.moe_expert_gate_up_shared_fp8_grouped_k,
            input,
            gp.weight_ptrs,
            gp.scale_ptrs,
            expert_gate_out,
            up.weight_ptrs,
            up.scale_ptrs,
            expert_up_out,
            expert_offsets,
            sorted_token_ids,
            active_experts,
            active_count,
            &sh.gate_proj,
            shared_gate_scratch,
            &sh.up_proj,
            shared_up_scratch,
            inter,
            h,
            cap,
            n,
            stream,
        )?;
        ops::moe_expert_silu_down_shared_fp8_grouped(
            ctx.gpu,
            self.moe_expert_silu_down_shared_fp8_grouped_k,
            expert_gate_out,
            expert_up_out,
            dp.weight_ptrs,
            dp.scale_ptrs,
            expert_down_out,
            expert_offsets,
            active_experts,
            active_count,
            shared_gate_scratch,
            shared_up_scratch,
            &sh.down_proj,
            shared_out,
            h,
            inter,
            cap,
            n,
            stream,
        )?;
        ops::moe_weighted_sum_blend_fp8_grouped(
            ctx.gpu,
            self.moe_weighted_sum_blend_fp8_grouped_k,
            output,
            expert_down_out,
            weights_dev,
            token_to_perm,
            shared_out,
            input,
            self.weights.shared_expert_gate.weight,
            h,
            top_k,
            h,
            n,
            stream,
        )
    }
}

#[cfg(test)]
#[path = "forward_fp8_grouped_decode_tests.rs"]
mod tests;
