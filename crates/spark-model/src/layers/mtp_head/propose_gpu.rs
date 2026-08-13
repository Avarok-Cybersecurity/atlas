// SPDX-License-Identifier: AGPL-3.0-only

//! Grammarless MTP propose GPU body (rms_norm through GPU argmax).
//!
//! Pointer-stable: embed already lives in `ssm_qkvz`, hidden in
//! `propose_in_hidden`, attention metadata already on device. Safe to
//! record inside a CUDA graph.

use super::propose_graph::ProposeKvView;
use super::{MtpHead, MtpQuantization, ProjectionWeight};
use crate::layer::ForwardContext;
use crate::layers::ops;
use anyhow::Result;

impl MtpHead {
    /// Embed/hidden through post-attn norm. Pointer-stable — graphable.
    pub(super) fn propose_gpu_pre_moe(
        &self,
        ctx: &ForwardContext<'_>,
        stream: u64,
        kv: &ProposeKvView,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let nq = ctx.config.num_attention_heads as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let hd = ctx.config.head_dim as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h as usize * 2;
        let embed_out = ctx.buffers.ssm_qkvz();
        let hidden_in = self.propose_in_hidden;

        let normed_embed = ctx.buffers.ssm_deinterleaved();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            embed_out,
            &self.pre_fc_norm_embedding,
            normed_embed,
            1,
            h,
            eps,
            stream,
        )?;
        let normed_hidden = ctx.buffers.ssm_gates();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden_in,
            &self.pre_fc_norm_hidden,
            normed_hidden,
            1,
            h,
            eps,
            stream,
        )?;

        let concat_out = ctx.buffers.ssm_ba();
        ops::bf16_concat(
            ctx.gpu,
            self.bf16_concat_k,
            normed_embed,
            normed_hidden,
            concat_out,
            h,
            stream,
        )?;

        let hidden = ctx.buffers.hidden_states();
        self.gemv(ctx.gpu, concat_out, &self.fc, hidden, h, h * 2, stream)?;

        let residual = ctx.buffers.residual();
        ctx.gpu
            .copy_d2d_async(hidden, residual, row_bytes, stream)?;

        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.input_layernorm,
            normed,
            1,
            h,
            eps,
            stream,
        )?;

        let q_out = ctx.buffers.qkv_output();
        let q_dim = nq * hd;
        let qg_dim = q_dim * 2;
        let qg_bytes = qg_dim as usize * 2;

        match self.quant {
            MtpQuantization::Nvfp4 => {
                if let ProjectionWeight::Nvfp4(ref w) = self.q_proj {
                    ops::w4a16_gemv_qg(
                        ctx.gpu,
                        self.w4a16_gemv_qg_k,
                        normed,
                        w,
                        q_out,
                        qg_dim,
                        h,
                        nq,
                        hd,
                        stream,
                    )?;
                }
            }
            MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                self.gemv(ctx.gpu, normed, &self.q_proj, q_out, qg_dim, h, stream)?;
                ops::deinterleave_qg(
                    ctx.gpu,
                    self.deinterleave_qg_k.unwrap(),
                    q_out,
                    1,
                    nq,
                    hd,
                    nq * hd * 2,
                    stream,
                )?;
            }
        }
        let gate_ptr = q_out.offset(q_dim as usize * 2);
        let k_out = q_out.offset(qg_bytes);
        let v_out = k_out.offset((nkv * hd) as usize * 2);

        match self.quant {
            MtpQuantization::Nvfp4 => {
                if let (ProjectionWeight::Nvfp4(kw), ProjectionWeight::Nvfp4(vw)) =
                    (&self.k_proj, &self.v_proj)
                {
                    ops::w4a16_gemv_dual(
                        ctx.gpu,
                        self.w4a16_gemv_dual_k,
                        normed,
                        kw,
                        k_out,
                        vw,
                        v_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                }
            }
            MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                self.gemv(ctx.gpu, normed, &self.k_proj, k_out, nkv * hd, h, stream)?;
                self.gemv(ctx.gpu, normed, &self.v_proj, v_out, nkv * hd, h, stream)?;
            }
        }

        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            q_out,
            &self.q_norm,
            q_out,
            nq,
            hd,
            eps,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            k_out,
            &self.k_norm,
            k_out,
            nkv,
            hd,
            eps,
            stream,
        )?;

        let meta_base = kv.meta_base;
        ops::rope(
            ctx.gpu,
            self.rope_k,
            q_out,
            k_out,
            meta_base,
            1,
            nq,
            nkv,
            hd,
            ctx.config.rotary_dim() as u32,
            ctx.config.rope_theta as f32,
            stream,
        )?;

        let kv_stride = nkv * hd;
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = 1.0f32 / (hd as f32).sqrt();
        if self.kv_bf16 {
            ops::reshape_and_cache(
                ctx.gpu,
                self.reshape_cache_k,
                k_out,
                v_out,
                kv.k_pool,
                kv.v_pool,
                meta_base.offset(8),
                1,
                nkv,
                hd,
                kv.block_size,
                kv_stride,
                kv_stride,
                kv.cache_stride,
                stream,
            )?;
            ops::paged_decode_attn_bf16(
                ctx.gpu,
                self.paged_decode_k,
                q_out,
                kv.k_pool,
                kv.v_pool,
                attn_out,
                meta_base.offset(256),
                meta_base.offset(16),
                kv.max_blocks,
                1,
                nq,
                nkv,
                hd,
                kv.block_size,
                inv_sqrt_d,
                nq * hd,
                0,
                stream,
            )?;
        } else {
            ops::reshape_and_cache_fp8(
                ctx.gpu,
                self.reshape_cache_k,
                k_out,
                v_out,
                kv.k_pool,
                kv.v_pool,
                meta_base.offset(8),
                1,
                nkv,
                hd,
                kv.block_size,
                1.0,
                1.0,
                kv_stride,
                kv_stride,
                kv.cache_stride,
                stream,
            )?;
            ops::paged_decode_attn_fp8(
                ctx.gpu,
                self.paged_decode_k,
                q_out,
                kv.k_pool,
                kv.v_pool,
                attn_out,
                meta_base.offset(256),
                meta_base.offset(16),
                kv.max_blocks,
                1,
                nq,
                nkv,
                hd,
                kv.block_size,
                inv_sqrt_d,
                1.0,
                1.0,
                nq * hd,
                kv.cache_stride,
                0,
                stream,
            )?;
        }

        ops::sigmoid_gate_mul(
            ctx.gpu,
            self.sigmoid_gate_mul_k,
            attn_out,
            gate_ptr,
            attn_out,
            nq * hd,
            stream,
        )?;

        let o_out = ctx.buffers.norm_output();
        self.gemv(ctx.gpu, attn_out, &self.o_proj, o_out, h, nq * hd, stream)?;

        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            o_out,
            &self.post_attn_layernorm,
            normed2,
            residual,
            1,
            h,
            eps,
            stream,
        )?;
        Ok(())
    }

    /// MoE / dense FFN. BF16/FP8 generic experts D2H indices and pick
    /// per-token weight pointers — must stay eager (not inside a graph).
    pub(super) fn propose_gpu_ffn(
        &self,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<spark_runtime::gpu::DevicePtr> {
        let normed2 = ctx.buffers.norm_output();
        if self.dense_ffn_generic.is_some() {
            self.dense_ffn_forward_generic(normed2, ctx, stream)
        } else {
            match self.quant {
                MtpQuantization::Nvfp4 => self
                    .moe_nvfp4
                    .as_ref()
                    .unwrap()
                    .forward(normed2, ctx, stream),
                MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                    self.moe_forward_generic(normed2, ctx, stream)
                }
            }
        }
    }

    /// Residual + final norm + NVFP4 lm_head + GPU argmax. Pointer-stable.
    pub(super) fn propose_gpu_post_moe(
        &self,
        ctx: &ForwardContext<'_>,
        stream: u64,
        ffn_out: spark_runtime::gpu::DevicePtr,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let hidden = ctx.buffers.hidden_states();
        ops::residual_add(ctx.gpu, self.residual_add_k, hidden, ffn_out, h, stream)?;

        let final_normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.norm,
            final_normed,
            1,
            h,
            eps,
            stream,
        )?;

        let v = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        // Base 64-thread GEMV, not the Batch 5 single-warp decode launcher.
        // SW is an occupancy win on small-N C=1 decode; this lm_head is
        // N≈1e5 × K=hidden=2048, where the single-warp K-walk is slower.
        // C=1 decode still uses SW — not this graphable propose slice.
        ops::w4a16_gemv(
            ctx.gpu,
            self.w4a16_gemv_k,
            final_normed,
            &self.lm_head_nvfp4,
            ctx.buffers.logits(),
            v,
            h,
            stream,
        )?;
        ops::argmax_bf16(
            ctx.gpu,
            self.argmax_k,
            ctx.buffers.logits(),
            ctx.buffers.scratch(),
            v,
            stream,
        )?;
        Ok(())
    }

    /// Single-token MTP GPU body ending at `argmax_bf16` into `scratch`.
    pub(super) fn propose_gpu_to_argmax(
        &self,
        ctx: &ForwardContext<'_>,
        stream: u64,
        kv: &ProposeKvView,
    ) -> Result<()> {
        self.propose_gpu_pre_moe(ctx, stream, kv)?;
        let ffn_out = self.propose_gpu_ffn(ctx, stream)?;
        self.propose_gpu_post_moe(ctx, stream, ffn_out)
    }
}

#[cfg(test)]
mod tests {
    /// NEGATIVE: the graphable MTP lm_head must not take Batch 5's SW GEMV.
    /// The single-warp K-walk is slower at K=hidden=2048; production propose
    /// (2.16 ms) was image-proven on the base 64-thread launch.
    ///
    /// PROVEN BY: restoring the SW decode launcher in `propose_gpu_post_moe`
    /// turns this red.
    #[test]
    fn propose_lm_head_does_not_dispatch_sw_gemv() {
        let src = include_str!("propose_gpu.rs");
        let prod = src.split("#[cfg(test)]").next().expect("prod before tests");
        assert!(
            prod.contains("ops::w4a16_gemv("),
            "lm_head must launch the base w4a16_gemv"
        );
        assert!(
            !prod.contains("ops::w4a16_decode_gemv("),
            "SW decode GEMV must not be on the propose lm_head path"
        );
    }
}
