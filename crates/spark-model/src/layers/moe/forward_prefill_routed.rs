// SPDX-License-Identifier: AGPL-3.0-only

//! Routed grouped-GEMM phase of `MoeLayer::forward_prefill`.
//!
//! Hoisted from `forward_prefill.rs` to keep that file under the 500 LoC
//! cap. The single entry point [`MoeLayer::run_routed_grouped_gemm`]
//! mirrors the original block 1:1 — same control flow, same kernel
//! launches, same buffer wiring. Covers steps 4-6 of the prefill
//! pipeline: grid sizing, grouped gate+up GEMM, SiLU, grouped down GEMM.

use super::*;

impl MoeLayer {
    /// Routed-expert grouped-GEMM path: upper-bound grid sizing → grouped
    /// gate+up GEMM → SiLU+mul → grouped down GEMM.
    ///
    /// Writes the routed expert outputs into `ctx.buffers.expert_down_out()`.
    /// `t0` carries the running profile timer so per-step timing output
    /// matches the original inline pipeline exactly.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_routed_grouped_gemm(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
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
        macro_rules! prof_step {
            ($label:expr) => {
                if let Some(t) = t0.take() {
                    ctx.gpu.synchronize(stream)?;
                    let elapsed = t.elapsed().as_micros();
                    tracing::info!("  MoE prefill [{}] N={}: {}µs", $label, num_tokens, elapsed);
                    *t0 = Some(std::time::Instant::now());
                }
            };
        }

        // 4. Upper-bound max_m_tiles — sized for the absolute worst case
        // (one expert eats all tokens) so the kernel never silently truncates
        // a heavily-loaded expert's rows. The previous `avg*2` heuristic was
        // wrong: real learned MoE routers concentrate experts ~7× the average
        // (observed on Qwen3.6-35B-A3B at chunk=4097: avg=129, max=929 for
        // expert 227 → kernel covered 320 rows but needed 929 → 609 rows
        // silently dropped → systematic ~-14% under-count in routed-MoE
        // output). The Poisson(avg) assumption in the old comment doesn't
        // hold for trained routers — they're sparse + concentrated.
        //
        // Cost: extra empty tiles for under-utilized experts; each early-
        // exits on `m_idx >= M_expert` so overhead is low vs the correctness
        // bug.
        //
        // Mirrors the FP8 path (see forward_prefill_fp8.rs).
        let avg_per_expert = (num_tokens * top_k as usize).div_ceil(ne);
        let max_m_tiles = (num_tokens * top_k as usize).div_ceil(64).max(1) as u32;
        super::dump::dump_expert_load(
            ctx.gpu,
            stream,
            expert_offsets,
            ne,
            num_tokens,
            avg_per_expert,
            max_m_tiles,
        );
        prof_step!("grid_setup");

        let total_expanded = n * top_k;

        // 5. Grouped gate+up GEMM — cp.async pipelined FP8-MMA K64 (transposed).
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        // Zero expert buffers unconditionally before the grouped GEMMs.
        // Even with worst-case `max_m_tiles` (above), some kernel paths only
        // write rows where `m_idx < M_expert` per expert — rows past the
        // expert's actual count keep stale data from the previous prefill
        // (or uninit memory on first prefill), which then propagates
        // through unpermute_reduce as spurious contributions. Previously
        // guarded by `ctx.comm.is_some()` (EP only), now unconditional.
        // Mirrors the FP8 path fix (commit 34626d3).
        {
            let gate_bytes = total_expanded as usize * inter as usize * 2;
            let up_bytes = gate_bytes;
            let down_bytes = total_expanded as usize * h as usize * 2;
            ctx.gpu
                .memset_async(expert_gate_out, 0, gate_bytes, stream)?;
            ctx.gpu.memset_async(expert_up_out, 0, up_bytes, stream)?;
            ctx.gpu
                .memset_async(ctx.buffers.expert_down_out(), 0, down_bytes, stream)?;
        }
        if max_m_tiles > 0 {
            if let (Some(gp), Some(up)) = (&self.gate_ptrs_t, &self.up_ptrs_t) {
                // Block D #3 dispatch: M=128 path needs the env var on AND
                // the kernel actually loaded (try_kernel returns 0 on
                // models that don't ship it). max_m_tiles_m128 = ceil(...
                // /128) instead of /64; reuse the same upper bound by
                // halving (each m128 tile covers 2 m64 tiles).
                let use_m128 = self.nvfp4_gate_up_m128 && self.moe_fused_gate_up_t_k64_m128.0 != 0;
                if use_m128 {
                    let max_m_tiles_m128 = max_m_tiles.div_ceil(2).max(1);
                    ops::moe_w4a16_fused_gate_up_k64_m128(
                        ctx.gpu,
                        self.moe_fused_gate_up_t_k64_m128,
                        expert_input,
                        gp.packed_ptrs,
                        gp.scale_ptrs,
                        gp.scale2_vals,
                        up.packed_ptrs,
                        up.scale_ptrs,
                        up.scale2_vals,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        sorted_token_ids,
                        num_experts,
                        inter,
                        h,
                        max_m_tiles_m128,
                        stream,
                    )?;
                } else {
                    ops::moe_w4a16_fused_gate_up_k64_n128(
                        ctx.gpu,
                        self.moe_fused_gate_up_t_k64,
                        expert_input,
                        gp.packed_ptrs,
                        gp.scale_ptrs,
                        gp.scale2_vals,
                        up.packed_ptrs,
                        up.scale_ptrs,
                        up.scale2_vals,
                        expert_gate_out,
                        expert_up_out,
                        expert_offsets,
                        sorted_token_ids,
                        num_experts,
                        inter,
                        h,
                        max_m_tiles,
                        stream,
                    )?;
                }
            } else {
                let (gp, up) = (&self.gate_ptrs, &self.up_ptrs);
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_input,
                    gp.packed_ptrs,
                    gp.scale_ptrs,
                    gp.scale2_vals,
                    expert_gate_out,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    max_m_tiles,
                    stream,
                )?;
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_input,
                    up.packed_ptrs,
                    up.scale_ptrs,
                    up.scale2_vals,
                    expert_up_out,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    max_m_tiles,
                    stream,
                )?;
            }
        }
        prof_step!("grouped_gate_up");

        // 6. Activation+mul for routed experts + grouped down GEMM (K64 pipelined).
        let expert_down_out = ctx.buffers.expert_down_out();
        if max_m_tiles > 0 {
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                total_expanded * inter,
                stream,
            )?;
            if let Some(dp) = &self.down_ptrs_t {
                ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                    ctx.gpu,
                    self.moe_grouped_gemm_t_k64,
                    expert_gate_out,
                    dp.packed_ptrs,
                    dp.scale_ptrs,
                    dp.scale2_vals,
                    expert_down_out,
                    expert_offsets,
                    DevicePtr(0),
                    num_experts,
                    h,
                    inter,
                    max_m_tiles,
                    stream,
                )?;
            } else {
                ops::moe_w4a16_grouped_gemm_ptrtable(
                    ctx.gpu,
                    self.moe_grouped_gemm,
                    expert_gate_out,
                    self.down_ptrs.packed_ptrs,
                    self.down_ptrs.scale_ptrs,
                    self.down_ptrs.scale2_vals,
                    expert_down_out,
                    expert_offsets,
                    DevicePtr(0),
                    num_experts,
                    h,
                    inter,
                    max_m_tiles,
                    stream,
                )?;
            }
        }
        prof_step!("grouped_silu_down");

        Ok(())
    }
}
