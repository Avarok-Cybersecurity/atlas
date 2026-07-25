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
        } else if n == 4 && !force_seq_ffn {
            // K=4 verify (num_drafts=3). Without this arm n=4 fell through to the
            // dense `forward_prefill` branch below, i.e. the prefill GEMM at M=4 --
            // 94% of the M-tile is padding, ~33 GB/s effective. `forward_k4` reads
            // each projection weight ONCE for all 4 rows via the batched GEMV
            // (~290 GB/s on these shapes), which is the same MMQ cliff the gb10
            // side hit and fixed. The attention-layer FFN was the ONLY site still
            // missing it -- lm_head (impl_a3), QKV (multi_seq/qkv) and the GDN
            // decode path all already route M<=4 to the batch4 kernels, which is
            // why K=4 measured net-negative here while gb10 measured it a win.
            //
            // `try_forward_k4` returns false when the path is unavailable (MoE /
            // missing batch4 kernel / non-NVFP4 weights); the fallback below is
            // byte-identical to the pre-existing behaviour for those configs.
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                4,
                h as u32,
                eps,
                stream,
            )?;
            if !self.ffn.try_forward_k4(normed2, fwd, stream)? {
                self.ffn.forward_prefill(normed2, 4, fwd, stream)?;
            }
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (4 * h) as u32,
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
        } else if !force_seq_ffn && self.ffn.is_dense() {
            // WIDE-VERIFY BATCHED DENSE FFN (DFlash γ=16, n=17). The dense FFN
            // (Qwen3.6-27B is dense) batches over all n rows via
            // `forward_prefill`, reading gate/up/down ONCE instead of the
            // per-token loop below that re-read the FFN weights n× — the
            // measured wide-γ verify bottleneck (~844ms → target ~150ms).
            // Direct mirror of the `forward_k3` branch above, with count=n.
            //
            // DENSE ONLY: on a 256-expert MoE the grouped-GEMM is a net loss at
            // small batch, so MoE (and MLA / force_seq) fall through to the
            // per-token loop below — no regression for 122b/35b-a3b.
            let normed2 = fwd.buffers.norm_output();
            ops::residual_add_rms_norm(
                fwd.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_norm,
                normed2,
                residual,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
            self.ffn.forward_prefill(normed2, n, fwd, stream)?;
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (n * h) as u32,
                stream,
            )?;
        } else {
            // force_seq_ffn (MLA / batched-MoE-unsafe): per-token sequential.
            // CONCURRENT-DECODE BUG (sibling of qwen3_ssm.rs:1102 fix):
            // the per-seq hidden/residual stride must match the residual
            // element size. The residual stream is always BF16, so the stride
            // is `i * h * 2`; a hardcoded `i * h * 4` would over-stride into
            // the wrong batch slot for i>=1.
            let residual_elem = 2usize;
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
            // Per-token MoE + residual (256-expert MoE: grouped-GEMM is a net
            // loss at small batch — per-expert M ~1, sort/permute overhead
            // dominates). Each forward() writes moe_output[0]; consume it
            // immediately before the next iteration overwrites it.
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
