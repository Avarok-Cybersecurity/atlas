// SPDX-License-Identifier: AGPL-3.0-only

//! Single-token decode body for [`super::super::Qwen3AttentionLayer`],
//! split out of the trait impl for file-size budget. The trait impl
//! delegates 1:1 to [`Qwen3AttentionLayer::decode_inner`].

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use super::{diag_norm, gemma4_diag_enabled};
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    pub(super) fn decode_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        _state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // DeepSeek-V4: Manifold-Constrained Hyper-Connections (mHC).
        // When HC is enabled, the persistent multi-stream state lives in
        // `hc_streams`; `hidden` is used as a single-stream scratch buffer.
        if self.hc.is_some() {
            return self.decode_inner_hc(
                hidden,
                residual,
                _state,
                kv_cache,
                seq_len,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            );
        }

        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        // Disable diagnostics during CUDA graph capture — diag_norm does d2h
        // copy + sync which invalidates stream capture (status 901).
        let gemma4_diag =
            ctx.config.model_type == "gemma4" && gemma4_diag_enabled() && !ctx.graph_capture;
        // The residual stream is always BF16, so `hidden` is a BF16 buffer.
        let diag_hidden =
            |gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize, stream: u64, label: &str| {
                diag_norm(gpu, ptr, n, stream, label);
            };

        let normed = ctx.buffers.norm_output();
        if gemma4_diag {
            diag_hidden(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("L{:02} hidden_in", self.attn_layer_idx),
            );
        }
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
        if gemma4_diag {
            diag_norm(
                ctx.gpu,
                normed,
                h,
                stream,
                &format!("L{:02} normed", self.attn_layer_idx),
            );
        }

        let attn_out = self.attention_forward(
            normed,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_cache,
            ctx,
            stream,
        )?;
        // TP all-reduce on attn_out after o_proj (Megatron row-parallel
        // pattern). When tp_world_size==1 this is a no-op. The o_proj GEMM
        // produced this rank's partial output on the full hidden dim; the
        // reduction across TP ranks gives the full attention output ready
        // for the residual add. Decode path: 1 token × hidden BF16.
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            let bytes = h * 2; // 1 token × hidden × BF16
            comm.all_reduce_async(attn_out.0, bytes, stream)?;
        }
        if gemma4_diag {
            diag_norm(
                ctx.gpu,
                attn_out,
                h,
                stream,
                &format!("L{:02} attn_out", self.attn_layer_idx),
            );
        }

        // Gemma-4: post-attention norm (applied to attn output before residual add).
        // Weight pre-scaled by layer_scalar at load time: norm(attn) * (w * scalar).
        if let Some(ref post_norm) = self.post_attn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                attn_out,
                post_norm,
                attn_out,
                1,
                h as u32,
                eps,
                stream,
            )?;
            if gemma4_diag {
                diag_norm(
                    ctx.gpu,
                    attn_out,
                    h,
                    stream,
                    &format!("L{:02} post_attn_normed", self.attn_layer_idx),
                );
            }
        }

        // Standalone attention (Nemotron-H): no post-attn FFN
        if self.ffn.is_none() {
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                attn_out,
                h as u32,
                stream,
            )?;
            return Ok(());
        }

        // Profile: time attention vs MoE separately
        if ctx.profile {
            use std::time::Instant;
            ctx.gpu.synchronize(stream)?;
            let t0 = Instant::now();

            let normed2 = ctx.buffers.norm_output();
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                attn_out,
                &self.post_attn_norm,
                normed2,
                residual,
                1,
                h as u32,
                eps,
                stream,
            )?;
            let moe_out = self.ffn.forward(normed2, ctx, stream)?;

            // Gemma-4: post-FFN norm
            if let Some(ref post_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    moe_out,
                    post_norm,
                    moe_out,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }

            ctx.gpu.synchronize(stream)?;
            let moe_us = t0.elapsed().as_micros();
            tracing::info!("  Attn-MoE: {:.1}ms", moe_us as f64 / 1000.0);

            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                h as u32,
                stream,
            )?;
            // Gemma-4: hidden *= layer_scalar at end of layer
            if let Some(scalar) = self.layer_scalar {
                self.apply_layer_scalar(ctx.gpu, hidden, h, scalar, stream)?;
            }
            return Ok(());
        }

        let normed2 = ctx.buffers.norm_output();
        // ATLAS_FP32_ROUTING: attention layers also have an MoE FFN — emit the
        // MoE-input norm in FP32 so their gates route at full precision too.
        if self.ffn.fp32_routing_active() && self.residual_add_rms_norm_gatef32_k.0 != 0 {
            ops::residual_add_rms_norm_gatef32(
                ctx.gpu,
                self.residual_add_rms_norm_gatef32_k,
                hidden,
                attn_out,
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
                attn_out,
                &self.post_attn_norm,
                normed2,
                residual,
                1,
                h as u32,
                eps,
                stream,
            )?;
        }

        // Gemma-4 26B MoE dual FFN: run MoE FIRST (before dense FFN result is used)
        // to avoid buffer conflicts (MoE fused kernel uses attn_output internally).
        //
        // HF reference: combined = norm(norm1(mlp_out) + norm2(moe_out))
        //               hidden = residual + combined
        if let (Some(moe_ffn), Some(_pre_norm), Some(post_norm), Some(dense_norm)) = (
            &self.moe_ffn,
            &self.pre_moe_norm,
            &self.post_moe_out_norm,
            &self.post_dense_ffn_norm,
        ) {
            // 1. Run MoE on raw residual (before dense FFN output is touched).
            //    MoE writes result to moe_output buffer.
            let moe_out = moe_ffn.forward(hidden, ctx, stream)?;
            // post-MoE norm (in-place on moe_output)
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                moe_out,
                post_norm,
                moe_out,
                1,
                h as u32,
                eps,
                stream,
            )?;
            // Save normed MoE output — dense FFN will overwrite moe_output.
            // Use logits buffer (vocab_size * 2 bytes >> h * 2) — gate_logits is too small
            let moe_saved = ctx.buffers.logits();
            ctx.gpu.copy_d2d_async(moe_out, moe_saved, h * 2, stream)?;

            // 2. Dense FFN (writes to moe_output, overwriting MoE result)
            let dense_out = self.ffn.forward(normed2, ctx, stream)?;
            // post-dense norm (layernorm_1)
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                dense_out,
                dense_norm,
                dense_out,
                1,
                h as u32,
                eps,
                stream,
            )?;

            // 3. Combine: dense_normed + moe_normed → dense_out (in-place)
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                dense_out,
                moe_saved,
                h as u32,
                stream,
            )?;

            // 4. post_feedforward_layernorm on combined
            if let Some(ref combined_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    dense_out,
                    combined_norm,
                    dense_out,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }

            // 5. Residual add: hidden += combined
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                dense_out,
                h as u32,
                stream,
            )?;
        } else {
            // Non-MoE (31B dense)
            if gemma4_diag {
                diag_norm(
                    ctx.gpu,
                    normed2,
                    h,
                    stream,
                    &format!("L{:02} normed2", self.attn_layer_idx),
                );
            }
            let dense_out = self.ffn.forward(normed2, ctx, stream)?;
            if gemma4_diag {
                diag_norm(
                    ctx.gpu,
                    dense_out,
                    h,
                    stream,
                    &format!("L{:02} dense_out", self.attn_layer_idx),
                );
            }
            if let Some(ref post_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    dense_out,
                    post_norm,
                    dense_out,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
                if gemma4_diag {
                    diag_norm(
                        ctx.gpu,
                        dense_out,
                        h,
                        stream,
                        &format!("L{:02} post_ffn_normed", self.attn_layer_idx),
                    );
                }
            }
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                dense_out,
                h as u32,
                stream,
            )?;
        }

        if gemma4_diag {
            diag_hidden(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("L{:02} post_residual", self.attn_layer_idx),
            );
        }

        // Gemma-4: hidden *= layer_scalar at end of layer
        if let Some(scalar) = self.layer_scalar {
            self.apply_layer_scalar(ctx.gpu, hidden, h, scalar, stream)?;
            if gemma4_diag {
                diag_hidden(
                    ctx.gpu,
                    hidden,
                    h,
                    stream,
                    &format!(
                        "L{:02} post_layer_scalar(scalar={:.4})",
                        self.attn_layer_idx, scalar
                    ),
                );
            }
        }

        Ok(())
    }

    /// HC-enabled decode inner.  The persistent state is `hc_streams`
    /// ([1, hc_mult, H] BF16); `hidden` is used as a single-stream scratch.
    #[allow(clippy::too_many_arguments)]
    fn decode_inner_hc(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        _state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let hc = self.hc.as_ref().unwrap();
        let hc_mult = hc.hc_mult as u32;
        let is_first_layer = self.attn_layer_idx == 0;
        let is_last_layer = self.attn_layer_idx + 1 == ctx.config.num_hidden_layers;
        let hc_streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        let diag_all =
            std::env::var("ATLAS_DIAG_V4_ALL_LAYERS").is_ok_and(|v| v == "1" || v == "true");
        let diag_this = self.attn_layer_idx == 0 || diag_all;

        // 1. Expand single-stream embedding into hc_mult copies on first layer.
        if is_first_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                hc_streams,
                1,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        // ── Attention sublayer ──
        ops::hc_pre(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
            hc.attn.hc_fn,
            hc.attn.hc_scale,
            hc.attn.hc_base,
            hidden,
            post,
            comb,
            1,
            h as u32,
            hc_mult,
            hc.sinkhorn_iters as u32,
            eps,
            hc.hc_eps,
            stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("V4-decode L{} hc_pre-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                hc_mult as usize,
                stream,
                &format!("V4-decode L{} post-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-decode L{} comb-attn", self.attn_layer_idx),
            );
        }

        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            hidden,
            &self.input_norm,
            normed,
            1,
            h as u32,
            eps,
            stream,
        )?;

        let attn_out = self.attention_forward(
            normed,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_cache,
            ctx,
            stream,
        )?;

        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            let bytes = h * 2;
            comm.all_reduce_async(attn_out.0, bytes, stream)?;
        }

        if let Some(ref post_norm) = self.post_attn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                attn_out,
                post_norm,
                attn_out,
                1,
                h as u32,
                eps,
                stream,
            )?;
        }

        // Standalone attention (no FFN)
        if self.ffn.is_none() {
            ops::hc_post(
                ctx.gpu,
                self.hc_post_k,
                attn_out,
                hc_streams,
                post,
                comb,
                hc_streams,
                1,
                h as u32,
                hc_mult,
                stream,
            )?;
            if is_last_layer && let Some(ref head) = hc.head {
                ops::hc_head(
                    ctx.gpu,
                    self.hc_head_k,
                    hc_streams,
                    head.hc_fn,
                    head.hc_scale,
                    head.hc_base,
                    hidden,
                    1,
                    h as u32,
                    hc_mult,
                    eps,
                    hc.hc_eps,
                    stream,
                )?;
            }
            return Ok(());
        }

        // Expand attention output back into multi-stream state.
        ops::hc_post(
            ctx.gpu,
            self.hc_post_k,
            attn_out,
            hc_streams,
            post,
            comb,
            hc_streams,
            1,
            h as u32,
            hc_mult,
            stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-decode L{} hc_post-attn", self.attn_layer_idx),
            );
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                (hc_mult as usize) * (h),
                stream,
                &format!(
                    "V4-decode L{} hc_post-attn ALL_STREAMS",
                    self.attn_layer_idx
                ),
            );
        }

        // ── FFN sublayer ──
        ops::hc_pre(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
            hc.ffn.hc_fn,
            hc.ffn.hc_scale,
            hc.ffn.hc_base,
            hidden,
            post,
            comb,
            1,
            h as u32,
            hc_mult,
            hc.sinkhorn_iters as u32,
            eps,
            hc.hc_eps,
            stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("V4-decode L{} hc_pre-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                hc_mult as usize,
                stream,
                &format!("V4-decode L{} post-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-decode L{} comb-ffn", self.attn_layer_idx),
            );
        }

        let normed2 = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            hidden,
            &self.post_attn_norm,
            normed2,
            1,
            h as u32,
            eps,
            stream,
        )?;

        let ffn_out = self.ffn.forward(normed2, ctx, stream)?;

        if let Some(ref post_norm) = self.post_ffn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                ffn_out,
                post_norm,
                ffn_out,
                1,
                h as u32,
                eps,
                stream,
            )?;
        }

        if let Some(scalar) = self.layer_scalar {
            self.apply_layer_scalar(ctx.gpu, ffn_out, h, scalar, stream)?;
        }

        ops::hc_post(
            ctx.gpu,
            self.hc_post_k,
            ffn_out,
            hc_streams,
            post,
            comb,
            hc_streams,
            1,
            h as u32,
            hc_mult,
            stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-decode L{} hc_post-ffn", self.attn_layer_idx),
            );
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                (hc_mult as usize) * (h),
                stream,
                &format!("V4-decode L{} hc_post-ffn ALL_STREAMS", self.attn_layer_idx),
            );
        }

        if is_last_layer && let Some(ref head) = hc.head {
            ops::hc_head(
                ctx.gpu,
                self.hc_head_k,
                hc_streams,
                head.hc_fn,
                head.hc_scale,
                head.hc_base,
                hidden,
                1,
                h as u32,
                hc_mult,
                eps,
                hc.hc_eps,
                stream,
            )?;
            if diag_this {
                super::diag_norm(
                    ctx.gpu,
                    hidden,
                    h,
                    stream,
                    &format!("V4-decode L{} hc_head", self.attn_layer_idx),
                );
            }
        } else if is_last_layer {
            tracing::warn!(
                "V4-decode L{}: hc_head SKIPPED (no head weights)",
                self.attn_layer_idx
            );
        }

        Ok(())
    }
}
