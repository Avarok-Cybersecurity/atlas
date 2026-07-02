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
        let force_w8a8_op = matches!(std::env::var("ATLAS_FP8_W8A8").ok().as_deref(), Some("1"));
        if let Some(ref dense_out) = self.out_proj_dense {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed_out_buf,
                dense_out,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else if force_w8a8_op
            && let Some(ref fp8w) = self.out_proj_fp8w
            && self.per_token_group_quant_fp8_k.0 != 0
            && self.fp8_gemm_t_blockscaled_k.0 != 0
        {
            let m = k as usize;
            let k_dim = h;
            let a_fp8_bytes = m * k_dim;
            let a_scale_bytes = m * (k_dim / 128) * 4;
            let a_fp8_buf = ctx.gpu.alloc(a_fp8_bytes)?;
            let a_scale_buf = ctx.gpu.alloc(a_scale_bytes)?;
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                normed_out_buf,
                a_fp8_buf,
                a_scale_buf,
                k,
                k_dim as u32,
                stream,
            )?;
            ops::fp8_gemm_t_blockscaled(
                ctx.gpu,
                self.fp8_gemm_t_blockscaled_k,
                a_fp8_buf,
                a_scale_buf,
                fp8w.weight,
                fp8w.row_scale,
                out_proj_buf,
                k,
                value_dim as u32,
                h as u32,
                stream,
            )?;
            ctx.gpu.synchronize(stream)?;
            ctx.gpu.free(a_fp8_buf)?;
            ctx.gpu.free(a_scale_buf)?;
            Ok(())
        } else if let Some(fp8) = self.out_proj_fp8 {
            if k > 128 {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    normed_out_buf,
                    fp8,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    normed_out_buf,
                    fp8,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )
            }
        } else if let Some(ref nvfp4_t) = self.out_proj_nvfp4_t {
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t_k,
                normed_out_buf,
                nvfp4_t,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        }
        .map_err(|e| anyhow::anyhow!("ssm phase3: out_proj GEMM failed: {e}"))?;

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
