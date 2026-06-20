// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode (single-token).

use super::*;

impl Qwen3SsmLayer {
    pub(super) fn decode_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        // Never synchronize while a CUDA graph is being captured: a
        // cuStreamSynchronize during capture is illegal (CUDA error 900,
        // STREAM_CAPTURE_UNSUPPORTED) and bricks the context. The debug
        // dumps below are diagnostic-only, so suppress them under capture
        // (issue #106: RUST_LOG=trace + --speculative crashed here).
        let debug = tracing::enabled!(tracing::Level::DEBUG) && !ctx.graph_capture;
        let trace = false;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            1,
            h as u32,
            eps,
            stream,
        )?;
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "pre-norm", normed, 4);
        }

        let ssm_out = self.ssm_forward(normed, ssm_state, ctx, stream, trace)?;
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "ssm-out", ssm_out, 4);
        }

        // Profile: time SSM vs MoE separately
        if ctx.profile {
            use std::time::Instant;
            ctx.gpu.synchronize(stream)?;
            let t0 = Instant::now();

            let normed2 = ctx.buffers.norm_output();
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                ssm_out,
                &self.post_attn_norm,
                normed2,
                residual,
                1,
                h as u32,
                eps,
                stream,
            )?;
            let moe_out = self.ffn.forward(normed2, ctx, stream)?;
            ctx.gpu.synchronize(stream)?;
            let moe_us = t0.elapsed().as_micros();
            tracing::info!("  SSM-MoE: {:.1}ms", moe_us as f64 / 1000.0);

            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                h as u32,
                stream,
            )?;
            return Ok(());
        }

        let normed2 = ctx.buffers.norm_output();
        // ATLAS_FP32_ROUTING: emit the MoE-input norm in FP32 (router_in) so the
        // gate GEMM runs at full precision — removes the bf16-store rounding that
        // flips experts on gfx1151. bf16 normed2 + residual are unchanged.
        if self.ffn.fp32_routing_active() && self.residual_add_rms_norm_gatef32_k.0 != 0 {
            ops::residual_add_rms_norm_gatef32(
                ctx.gpu,
                self.residual_add_rms_norm_gatef32_k,
                hidden,
                ssm_out,
                &self.post_attn_norm,
                normed2,
                ctx.buffers.moe_router_in_f32(),
                residual,
                1,
                h as u32,
                eps,
                stream,
            )?;
        } else {
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                ssm_out,
                &self.post_attn_norm,
                normed2,
                residual,
                1,
                h as u32,
                eps,
                stream,
            )?;
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "post-ssm-residual", residual, 4);
            Self::debug_bf16(ctx.gpu, "post-ssm-hidden", hidden, 4);
            Self::debug_bf16(ctx.gpu, "moe-input-normed", normed2, 4);
        }

        let moe_out = self.ffn.forward(normed2, ctx, stream)?;
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "moe-output", moe_out, 8);
        }
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            moe_out,
            h as u32,
            stream,
        )?;
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "final-hidden", hidden, 4);
        }

        Ok(())
    }
}
