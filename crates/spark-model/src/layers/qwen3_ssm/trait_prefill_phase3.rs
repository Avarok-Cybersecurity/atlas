// SPDX-License-Identifier: AGPL-3.0-only

//! prefill_phase3 + alloc_state.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_phase3_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;

        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let value_dim = nv * vd;

        // ── 9. Gated RMS norm (batched: all chunk tokens × heads) ──
        // Read GDN output and Z from full-sequence buffers at token_offset.
        let gdn_out_chunk = gdn_bufs.output.offset(token_offset * value_dim * bf16);
        let z_chunk = gdn_bufs.z.offset(token_offset * value_dim * bf16);

        // Output buffer: reuse ssm_qkvz (same as monolithic prefill)
        let normed_out_buf = ctx.buffers.ssm_qkvz();
        ops::gated_rms_norm_prefill(
            ctx.gpu,
            self.gated_rms_norm_prefill_k,
            gdn_out_chunk,
            z_chunk,
            &self.ssm.norm,
            normed_out_buf,
            nv as u32,
            vd as u32,
            eps,
            k,
            value_dim as u32, // input_token_stride: GDN output is [N, value_dim] contiguous
            value_dim as u32, // gate_token_stride: Z buffer is [N, value_dim] contiguous
            stream,
        )?;

        // ── 10. Output projection GEMM: [N, 4096] × [4096, 2048] → [N, 2048] ──
        let out_proj_buf = ctx.buffers.moe_output();
        // Shared single-stream out_proj dispatch: CUTLASS-NVFP4 (from nvfp4_t or
        // the fp8-packed weight) first, then the tensor-core pipelined BF16
        // kernel for the dense fallback. The batched phase3 previously inlined a
        // chain whose dense branch used the scalar `dense_gemm` — nsys showed
        // that on co-dispatched requests while single-stream (which already used
        // this helper) ran NVFP4. Routing through it equalises the two paths.
        self.prefill_out_proj_dispatch(ctx, normed_out_buf, out_proj_buf, k, h, value_dim, stream)?;
        // GDN HeadParallel: reduce the row-parallel partial out_proj across TP
        // ranks (num_tokens × h BF16) before the residual add. No-op at tp=1.
        self.ssm_tp_all_reduce(out_proj_buf, num_tokens, ctx, stream)?;

        // ── 11. Batched residual + post-norm + MoE ──
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            out_proj_buf,
            &self.post_attn_norm,
            ctx.buffers.norm_output(),
            residual,
            num_tokens as u32,
            h as u32,
            eps,
            stream,
        )?;
        self.ffn
            .forward_prefill(ctx.buffers.norm_output(), num_tokens, ctx, stream)?;
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            ctx.buffers.moe_output(),
            (num_tokens * h) as u32,
            stream,
        )?;

        Ok(())
    }

    pub(super) fn alloc_state_inner(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let h_state = gpu.alloc(self.h_state_bytes)?;
        gpu.memset(h_state, 0, self.h_state_bytes)?;
        let conv_state = gpu.alloc(self.conv_state_bytes)?;
        gpu.memset(conv_state, 0, self.conv_state_bytes)?;
        Ok(Box::new(SsmLayerState {
            h_state,
            conv_state,
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: Vec::new(),
            conv_state_intermediates: Vec::new(),
        }))
    }
}
