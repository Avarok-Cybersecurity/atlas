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
        // MLA models (Mistral-Small-4) route the FFN through the
        // sequential per-token branch below, NOT the fused `forward_k2`
        // / `forward_k3` batched-MoE kernels. The batched-MoE K=2/K=3
        // path has a pre-existing crash for Mistral-Small-4's MoE config
        // (illegal address in `moe_expert_silu_down_shared_batch2`) — it
        // was never exercised because Mistral always ran at batch=1. The
        // sequential branch calls `FfnComponent::forward` (the proven
        // single-token MoE path used by `decode()`), processing each
        // sequence's normed input independently, so the batched MLA
        // attention path (issue #84) gets correct, isolated FFN output
        // without depending on the buggy batched-MoE kernels. Fixing the
        // batched-MoE kernel is tracked separately (out of #84 scope).
        let force_seq_ffn = self.mla.is_some();
        if n == 3 && !force_seq_ffn {
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
        } else if n == 2 && !force_seq_ffn {
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
            // Phase A: per-token residual_add + post-attention RMS norm.
            // Lays out `norm_output[0..n]` as a contiguous [N, h] MoE input.
            for i in 0..n {
                let hidden_i = hidden.offset(i * h * residual_elem);
                let o_out_i = o_out.offset(i * h * bf16); // BF16 attn output
                let residual_i = residual.offset(i * h * residual_elem);
                let normed2_i = fwd.buffers.norm_output().offset(i * h * bf16);
                ops::residual_add_rms_norm(
                    fwd.gpu,
                    self.residual_add_rms_norm_k,
                    hidden_i,
                    o_out_i,
                    &self.post_attn_norm,
                    normed2_i,
                    residual_i,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
            // Phase B+C: per-token MoE + residual. The generic grouped-GEMM
            // (forward_prefill) is a NET LOSS for this 256-expert MoE at
            // small batch — per-expert M ~1 and the sort/permute/ptr-table
            // overhead dominates (measured: attention block ~40ms vs ~20ms
            // per-token at N=4 on GB10). N=2/3 already take the fused
            // forward_k2/k3 branches above; this `else` only sees N>=4 (or
            // MLA, which must avoid the batched-MoE kernels anyway), so the
            // per-token path — identical to decode()'s MoE — is fastest here
            // until a true batched-EP MoE kernel exists. Mirrors the SSM
            // dispatch in qwen3_ssm/trait_decode_multi_seq.rs. Each forward()
            // writes moe_output[0]; consume it immediately before the next
            // iteration overwrites it.
            let normed_base = fwd.buffers.norm_output();
            for i in 0..n {
                let hidden_i = hidden.offset(i * h * residual_elem);
                let normed2_i = normed_base.offset(i * h * bf16);
                let moe_out = self.ffn.forward(normed2_i, fwd, stream)?;
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
