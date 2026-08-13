// SPDX-License-Identifier: AGPL-3.0-only

//! Unified-T (transposed NVFP4) K=2 verify branch, including the Laguna
//! mixed NVFP4-routed / BF16-shared handling. Split from `forward_k2.rs`
//! (500-LoC cap) as a child module so field access is unchanged.

use super::super::*;

impl MoeLayer {
    /// Phase 8a unified-layout NVFP4 batch=2 verify (MTP K=2). Hybrid
    /// mode skips this branch — small-N MTP verify wins on warp-
    /// reduction originals.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_k2_unified_t(
        &self,
        input: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        shared_gate_scratch: DevicePtr,
        shared_up_scratch: DevicePtr,
        shared_down_out: DevicePtr,
        output: DevicePtr,
        inter: u32,
        h: u32,
        top_k: u32,
        is_ep: bool,
        mixed_bf16_shared: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Phase 8a unified-layout NVFP4 batch=2 verify (MTP K=2). Hybrid
        // mode skips this branch — small-N MTP verify wins on warp-
        // reduction originals.
        let gate_t = self
            .gate_ptrs_t
            .as_ref()
            .expect("gate_ptrs_t under unified_t");
        let up_t = self.up_ptrs_t.as_ref().expect("up_ptrs_t under unified_t");
        let down_t = self
            .down_ptrs_t
            .as_ref()
            .expect("down_ptrs_t under unified_t");
        let null_qw = QuantizedWeight::null();
        // Mixed config: force the in-kernel shared expert off (NULL weights
        // → the kernel skips it) and compute it in BF16 below instead. The
        // NVFP4 shared_*_t tables are load-time placeholders whose values
        // would be numerically wrong for this checkpoint.
        let (sh_gate_t, sh_up_t, sh_down_t) = if mixed_bf16_shared {
            (&null_qw, &null_qw, &null_qw)
        } else {
            (
                self.shared_gate_t.as_ref().unwrap_or(&null_qw),
                self.shared_up_t.as_ref().unwrap_or(&null_qw),
                self.shared_down_t.as_ref().unwrap_or(&null_qw),
            )
        };
        ops::moe_expert_gate_up_shared_batch2_t(
            ctx.gpu,
            self.moe_expert_gate_up_shared_batch2_t_k,
            input,
            gate_t.packed_ptrs,
            gate_t.scale_ptrs,
            gate_t.scale2_vals,
            expert_gate_out,
            up_t.packed_ptrs,
            up_t.scale_ptrs,
            up_t.scale2_vals,
            expert_up_out,
            indices_dev,
            sh_gate_t,
            shared_gate_scratch,
            sh_up_t,
            shared_up_scratch,
            inter,
            h,
            top_k,
            stream,
        )?;
        ops::moe_expert_silu_down_shared_batch2_t(
            ctx.gpu,
            self.moe_expert_silu_down_shared_batch2_t_k,
            expert_gate_out,
            expert_up_out,
            down_t.packed_ptrs,
            down_t.scale_ptrs,
            down_t.scale2_vals,
            expert_down_out,
            indices_dev,
            shared_gate_scratch,
            shared_up_scratch,
            sh_down_t,
            shared_down_out,
            h,
            inter,
            top_k,
            stream,
        )?;
        // Mixed config: one batched BF16 shared-expert pass for both tokens
        // (3 GEMMs + silu_mul total, vs 4 launches per token in the
        // per-token fallback). Must run after silu_down_t, which owns the
        // shared scratch buffers for the non-mixed case.
        if mixed_bf16_shared {
            let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
            self.run_bf16_shared_expert(
                input,
                2,
                h,
                shared_inter,
                shared_gate_scratch,
                shared_up_scratch,
                shared_down_out,
                ctx,
                stream,
            )?;
        }
        // The _t branch previously returned without writing moe_output at
        // all — every sibling branch ends in this blend.
        let shared_for_blend = if is_ep && !shared_down_out.is_null() {
            ctx.gpu
                .memset_async(expert_gate_out, 0, 2 * h as usize * 2, stream)?;
            expert_gate_out
        } else {
            shared_down_out
        };
        ops::moe_weighted_sum_blend_batch2(
            ctx.gpu,
            self.moe_weighted_sum_blend_batch2,
            output,
            expert_down_out,
            weights_dev,
            shared_for_blend,
            input,
            self.weights.shared_expert_gate.weight,
            h,
            top_k,
            h,
            stream,
        )?;
        Ok(())
    }
}
