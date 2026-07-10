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
        // ATLAS_DFLASH_VERIFY_GDN_F32 prototype: when the GDN dispatch wrote
        // FP32 output (gdn_bufs.output_f32) this call, read it directly here
        // instead of the BF16-truncated `output` buffer.
        //
        // CORRECTNESS GUARD: `output_f32_written` is set by whichever GDN
        // dispatch branch ran for THIS call (prefill_gdn_wy16_inner for
        // total_len==16, prefill_gdn_full_inner's split4 branch for other
        // sizes reaching that rung of the ladder) — it is the direct signal
        // that output_f32 holds fresh data, not a total_len-based proxy.
        // A prior stopgap gated on `num_tokens != 16` alone, which made the
        // FP32 path a permanent no-op on production K=γ verify batches
        // (total_len=γ+1=16); this reads the real per-call outcome instead.
        let fp32 = 4usize;
        let use_f32_gdn = gdn_bufs.output_f32_written.get()
            && self.gated_rms_norm_prefill_f32_k.0 != 0
            && !gdn_bufs.output_f32.is_null()
            && std::env::var("ATLAS_DFLASH_VERIFY_GDN_F32").ok().as_deref() == Some("1");
        let z_chunk = gdn_bufs.z.offset(token_offset * value_dim * bf16);

        // Output buffer: reuse ssm_qkvz (same as monolithic prefill)
        let normed_out_buf = ctx.buffers.ssm_qkvz();
        if use_f32_gdn {
            let gdn_out_chunk_f32 = gdn_bufs.output_f32.offset(token_offset * value_dim * fp32);
            ops::gated_rms_norm_prefill(
                ctx.gpu,
                self.gated_rms_norm_prefill_f32_k,
                gdn_out_chunk_f32,
                z_chunk,
                &self.ssm.norm,
                normed_out_buf,
                nv as u32,
                vd as u32,
                eps,
                k,
                value_dim as u32, // input_token_stride: GDN FP32 output is [N, value_dim] contiguous
                value_dim as u32, // gate_token_stride: Z buffer is [N, value_dim] contiguous
                stream,
            )?;
        } else {
            let gdn_out_chunk = gdn_bufs.output.offset(token_offset * value_dim * bf16);
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
        }

        // ── 10. Output projection GEMM: [N, 4096] × [4096, 2048] → [N, 2048] ──
        let out_proj_buf = ctx.buffers.moe_output();
        // Serial decode (ssm_forward.rs) uses the dedicated w4a16_gemv kernel
        // with the untransposed `self.ssm.out_proj` weight on native-NVFP4
        // builds; the shared batched dispatch below would otherwise pick a
        // transposed-weight GEMM kernel family, diverging from serial decode
        // by ~2.26% rel_diff at matched shapes. For verify-window sized calls
        // (K=γ ≤ 17 tokens — normal prefill chunks are far larger, and a
        // per-row GEMV loop there would serialize the hot path) loop per-row
        // through the same w4a16_gemv kernel serial decode uses to keep the
        // two paths numerically aligned. FP8/dense builds keep the dispatch:
        // their serial arm uses w8a16_gemv/dense_gemv, so there is no
        // NVFP4-vs-GEMM mismatch to fix and `ssm.out_proj` may be NULL.
        // Escape hatch: ATLAS_DFLASH_VERIFY_OUTPROJ_FIX=0.
        let out_proj_fix = num_tokens <= 17
            && self.out_proj_fp8w.is_none()
            && self.out_proj_dense.is_none()
            && std::env::var("ATLAS_DFLASH_VERIFY_OUTPROJ_FIX")
                .ok()
                .as_deref()
                != Some("0");
        if out_proj_fix {
            for t in 0..num_tokens {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    normed_out_buf.offset(t * value_dim * bf16),
                    &self.ssm.out_proj,
                    out_proj_buf.offset(t * h * bf16),
                    h as u32,
                    value_dim as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("ssm phase3: out_proj GEMV fix failed (row {t}): {e}")
                })?;
            }
        } else {
            // Shared single-stream out_proj dispatch: CUTLASS-NVFP4 (from
            // nvfp4_t or the fp8-packed weight) first, then the tensor-core
            // pipelined BF16 kernel for the dense fallback.
            self.prefill_out_proj_dispatch(
                ctx,
                normed_out_buf,
                out_proj_buf,
                k,
                h,
                value_dim,
                stream,
            )?;
        }

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
