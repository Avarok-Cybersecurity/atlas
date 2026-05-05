// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 7: residual + post-norm + MoE/dense FFN.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    pub(super) fn ms_phase_ffn(&self, c: &MultiSeqCtx<'_>, o_out: DevicePtr) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            eps,
            bf16,
            hidden,
            residual,
            ..
        } = *c;

        if self.ffn.is_none() {
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                o_out,
                (n * h) as u32,
                stream,
            )?;
            return Ok(());
        }
        if n == 3 {
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                3,
                h as u32,
                eps,
                stream,
            )?;
            self.ffn.forward_k3(normed2, fwd, stream)?;
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (3 * h) as u32,
                stream,
            )?;
        } else if n == 2 {
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                2,
                h as u32,
                eps,
                stream,
            )?;
            self.ffn.forward_k2(normed2, fwd, stream)?;
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (2 * h) as u32,
                stream,
            )?;
        } else {
            // CONCURRENT-DECODE BUG (sibling of qwen3_ssm.rs:1102 fix):
            // the per-seq hidden/residual stride must match the actual
            // residual element size. When `use_fp32_residual()` is false
            // (BF16 hidden — GB10 default via HARDWARE.toml
            // ATLAS_HW_FP32_RESIDUAL=false), hardcoded `i * h * 4` would
            // over-stride into the wrong batch slot for i>=1.
            let residual_elem = if fwd.config.use_fp32_residual() {
                4usize
            } else {
                2usize
            };
            for i in 0..n {
                let hidden_i = hidden.offset(i * h * residual_elem);
                let o_out_i = o_out.offset(i * h * bf16); // BF16 attn output
                let residual_i = residual.offset(i * h * residual_elem);
                let normed2 = fwd.buffers.norm_output().offset(i * h * bf16);
                ops::residual_add_rms_norm(
                    fwd.gpu,
                    self.residual_add_rms_norm_k,
                    hidden_i,
                    o_out_i,
                    &self.post_attn_norm,
                    normed2,
                    residual_i,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
                let moe_out = self.ffn.forward(normed2, fwd, stream)?;
                ops::residual_add(
                    fwd.gpu,
                    self.residual_add_k,
                    hidden_i,
                    moe_out,
                    h as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
