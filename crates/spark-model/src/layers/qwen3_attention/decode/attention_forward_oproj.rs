// SPDX-License-Identifier: AGPL-3.0-only

//! O-projection (GEMV) branch of `attention_forward` decode path.
//! Picks one of: MLA NVFP4 wo, BF16 dense (Gemma-4), W8A16 (FP8 native),
//! or default w4a16. Extracted from `attention_forward.rs` to keep that
//! file under 500 LoC. (MLA decode actually returns through
//! `attention_forward_mla.rs`, but the standard chain still has its own
//! MLA fallback for layers that didn't take the absorbed path.)

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_forward_oproj(
        &self,
        attn_out: DevicePtr,
        nq: u32,
        hd: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let o_out = ctx.buffers.norm_output();
        if let Some(ref mla) = self.mla {
            if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    attn_out,
                    wo_nvfp4,
                    o_out,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    attn_out,
                    &mla.wo,
                    o_out,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(o_bf16) = self.o_dense_bf16.as_ref() {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                attn_out,
                o_bf16,
                o_out,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(fp8) = self.o_weight.as_ref().and_then(|w| w.as_fp8()) {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                attn_out,
                fp8.weight,
                fp8.row_scale,
                o_out,
                h,
                nq * hd,
                stream,
            )?;
        } else {
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                attn_out,
                &self.attn.o_proj,
                o_out,
                h,
                nq * hd,
                stream,
            )?;
        }
        // ── LoRA delta on o_proj (decode, m=1). attn_out is already
        // sigmoid-gated by the caller — exactly the tensor HF feeds o_proj.
        if let Some(ref lw) = self.lora
            && let Some(ref pair) = lw.o
        {
            debug_assert_eq!(pair.k_in, nq * hd);
            debug_assert_eq!(pair.n_out, h);
            // Request-scoped routing (see attention_forward.rs). Route this
            // request's O delta via the bgmv when a per-seq slot + route exist;
            // else the installed-active-pair path (byte-identical to pre-M2).
            let seq_slot = ctx
                .attn_metadata
                .map(|m| m.seq_slot)
                .unwrap_or(DevicePtr(0));
            if seq_slot.0 != 0
                && let Some(ref route) = lw.o_route
            {
                ops::lora_delta::apply_lora_bgmv(
                    ctx.gpu,
                    &lw.kernels,
                    route,
                    attn_out,
                    o_out,
                    seq_slot,
                    1,
                    pair.k_in,
                    pair.n_out,
                    ctx.buffers.lora_xa(),
                    stream,
                )?;
            } else {
                ops::lora_delta::apply_lora_delta(
                    ctx.gpu,
                    &lw.kernels,
                    pair,
                    attn_out,
                    o_out,
                    1,
                    ctx.buffers.lora_xa(),
                    ctx.buffers.lora_delta(),
                    stream,
                )?;
            }
        }
        Ok(o_out)
    }
}
