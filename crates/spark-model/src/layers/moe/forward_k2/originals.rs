// SPDX-License-Identifier: AGPL-3.0-only

//! Originals-layout NVFP4 K=2 verify branch, including the Laguna mixed
//! NVFP4-routed / BF16-shared handling. Split from `forward_k2.rs`
//! (500-LoC cap) as a child module so field access is unchanged.

use super::super::*;
use super::batch2_block_width;

impl MoeLayer {
    /// NVFP4 batch2 verify path over the ORIGINALS (non-transposed) layout.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_k2_originals(
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
        // NVFP4 batch2 path (originals layout)
        let null_shared = QuantizedWeight::null();
        let batch2_block = batch2_block_width(ctx.config.hidden_size);
        ops::moe_expert_gate_up_shared_batch2(
            ctx.gpu,
            self.moe_expert_gate_up_shared_batch2,
            input,
            self.gate_ptrs.packed_ptrs,
            self.gate_ptrs.scale_ptrs,
            self.gate_ptrs.scale2_vals,
            expert_gate_out,
            self.up_ptrs.packed_ptrs,
            self.up_ptrs.scale_ptrs,
            self.up_ptrs.scale2_vals,
            expert_up_out,
            indices_dev,
            if mixed_bf16_shared {
                &null_shared
            } else {
                &self.weights.shared_expert.gate_proj
            },
            shared_gate_scratch,
            if mixed_bf16_shared {
                &null_shared
            } else {
                &self.weights.shared_expert.up_proj
            },
            shared_up_scratch,
            inter,
            h,
            top_k,
            batch2_block,
            stream,
        )?;
        ops::moe_expert_silu_down_shared_batch2(
            ctx.gpu,
            self.moe_expert_silu_down_shared_batch2,
            expert_gate_out,
            expert_up_out,
            self.down_ptrs.packed_ptrs,
            self.down_ptrs.scale_ptrs,
            self.down_ptrs.scale2_vals,
            expert_down_out,
            indices_dev,
            shared_gate_scratch,
            shared_up_scratch,
            if mixed_bf16_shared {
                &null_shared
            } else {
                &self.weights.shared_expert.down_proj
            },
            shared_down_out,
            h,
            inter,
            top_k,
            batch2_block,
            stream,
        )?;
        // Mixed config: one batched BF16 shared-expert pass for both tokens,
        // replacing the placeholder the kernel was told (via NULL) to skip.
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
        // EP fix: after silu_down, expert_gate_out is free — use as zero buffer
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
