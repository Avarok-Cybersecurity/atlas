// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 7: residual + post-norm + MoE/dense FFN.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// Kill-switch for the pairwise batched MoE decode path (`ATLAS_MOE_PAIRWISE_DECODE=0`).
fn pairwise_moe_decode_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_MOE_PAIRWISE_DECODE").as_deref() != Ok("0"))
}

/// Route batched decode MoE (n >= min) through the grouped read-once GEMM
/// (forward_prefill) instead of the pairwise per-slot loop. Default OFF.
/// Min default 2: one consistent grouped path for every batched decode size.
fn grouped_routed_decode_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_MOE_GROUPED_ROUTED_DECODE").as_deref() == Ok("1"))
}
fn grouped_routed_decode_min() -> usize {
    use std::sync::OnceLock;
    static M: OnceLock<usize> = OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("ATLAS_MOE_GROUPED_ROUTED_DECODE_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2)
    })
}

impl Qwen3AttentionLayer {
    pub(super) fn ms_phase_ffn(&self, c: &MultiSeqCtx<'_>, o_out: DevicePtr) -> Result<()> {
        // A model with a shortcut MoE (LongCat) has an architectural component
        // that ONLY the per-token branch below implements. Every other arm
        // would compute a structurally incomplete block and return no error —
        // so refuse instead of serving it. `force_seq_ffn` is true whenever
        // `mla.is_some()`, which is exactly the LongCat case, but assert it
        // rather than rely on that coupling holding forever.
        anyhow::ensure!(
            (self.shortcut_carry_out.is_none() && self.shortcut_carry_in.is_none())
                || self.mla.is_some(),
            "shortcut-MoE model reached the batched FFN ladder, which does not              implement the shortcut; only the per-token branch does"
        );
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
        // Grouped read-once MoE decode: when enabled and eligible, ALL batched
        // decode (n >= min, default 2) goes through the single grouped branch
        // below instead of the n==2/n==3/pairwise special cases. One consistent
        // path for every batch size.
        let use_grouped = !force_seq_ffn
            && n >= grouped_routed_decode_min()
            && grouped_routed_decode_enabled()
            && self.ffn.moe_grouped_decode_ok();
        if !use_grouped && n == 3 && !force_seq_ffn {
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
            self.ffn.forward_k3(normed2, 0, fwd, stream)?;
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (3 * h) as u32,
                stream,
            )?;
        } else if !use_grouped && n == 2 && !force_seq_ffn {
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
            self.ffn.forward_k2(normed2, 0, fwd, stream)?;
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (2 * h) as u32,
                stream,
            )?;
        } else if (4..=8).contains(&n) && !force_seq_ffn && self.ffn.can_forward_km(n as u32) {
            // MISSING K=4 ARM (2026-07-24): the ladder jumped from n==2/3
            // straight to the dense `forward_prefill` GEMM below, so K=4
            // verify ran the 16 attention layers' FFN through the MMQ/tile
            // prefill path (~156 GB/s cliff — the exact arm forward_km's
            // docstring quantifies at 54.8 ms/step vs ~31 ms batched on the
            // GDN stack, where it IS wired: trait_decode_batched.rs). Live
            // K=4 A/B showed verify ~1.41x K=3 cost from this alone. Mirror
            // of the n==3 arm with the M<=4 batched GEMV; n=5..8 (chain
            // verify K=5..8) rides the same arm via `w4a16_gemv_batch8`.
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
            let used = self.ffn.try_forward_km(normed2, n as u32, fwd, stream)?;
            debug_assert!(used, "can_forward_km checked at branch entry");
            let moe_out = fwd.buffers.moe_output();
            ops::residual_add(
                fwd.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (n * h) as u32,
                stream,
            )?;
        } else if !force_seq_ffn
            && (self.ffn.is_dense() || crate::layers::moe_grouped_decode_for(n))
        {
            // TASK-167 (gx10): mirror the SSM-side ATLAS_MOE_GROUPED_DECODE arm
            // for the attention layers' MoE — at large n the per-token loop
            // below re-reads each routed expert per token; forward_prefill
            // reads each distinct expert once (same body as the dense branch).
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
        } else if use_grouped {
            // GROUPED READ-ONCE MoE DECODE (A/B, default off). The pairwise
            // branch below issues 4*top_k per-slot CTAs at n=4, each re-reading
            // an expert weight for one token; forward_prefill sorts by expert
            // and reads each DISTINCT active expert ONCE (+ one batched BF16
            // shared pass). Byte-identical structure to the is_dense branch
            // above; only reachable for native-NVFP4-routed MoE with the flag.
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
        } else if !force_seq_ffn && n > 2 && n % 2 == 0 && pairwise_moe_decode_enabled() {
            // BATCHED MoE DECODE (n = 4/8 after padding). The per-token loop
            // below re-reads every routed expert weight once per token; the
            // fused batch2 kernels process a token PAIR in 5 launches. Walk the
            // batch two tokens at a time and consume moe_output before the next
            // pair overwrites it. Falls back inside forward_k2 for layouts that
            // have no fused batch2 path, which is still no worse than per-token
            // (the gate GEMM is batched there too).
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
            for pair in 0..(n / 2) {
                let off = pair * 2 * h;
                self.ffn
                    .forward_k2(normed2.offset(off * bf16), pair * 2, fwd, stream)?;
                ops::residual_add(
                    fwd.gpu,
                    self.residual_add_k,
                    hidden.offset(off * 2),
                    fwd.buffers.moe_output(),
                    (2 * h) as u32,
                    stream,
                )?;
            }
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
            if std::env::var("ATLAS_MOE_BATCHED_DECODE").ok().as_deref() == Some("1") {
                // Batched MoE decode over all N tokens (mirrors the SSM multi-seq
                // path): the routed per-token expert kernels run under one call so
                // the Feature-1 LoRA fold (which the per-token `forward` refuses
                // for num_seqs>1) applies via `forward_batched`'s per-row map.
                self.ffn.forward_batched(normed_base, n, fwd, stream)?;
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
                for i in 0..n {
                    let hidden_i = hidden.offset(i * h * residual_elem);
                    let normed2_i = normed_base.offset(i * h * bf16);
                    // LongCat shortcut MoE (producer). MUST run before the
                    // dense FFN below, which reuses `moe_output`. This is the
                    // BATCHED mirror of the single-token path in
                    // `decode_inner`: without it, batched decode silently drops
                    // the block's entire 256-expert shortcut contribution for
                    // every sequence — attention still reads the right tokens,
                    // so the topic survives while the distribution does not,
                    // which reads as words fragmenting mid-answer rather than
                    // as anything crashing.
                    if let (Some(moe_ffn), Some((carry, cap))) =
                        (&self.moe_ffn, self.shortcut_carry_out)
                    {
                        anyhow::ensure!(
                            n <= cap,
                            "shortcut carry capacity {cap} < decode batch {n}"
                        );
                        let sc_out = moe_ffn.forward(normed2_i, i, fwd, stream)?;
                        if let crate::layers::FfnComponent::Moe(m) = moe_ffn {
                            m.apply_zero_expert(sc_out, normed2_i, 1, fwd, stream)?;
                        }
                        fwd.gpu.copy_d2d_async(
                            sc_out,
                            carry.offset(i * h * bf16),
                            h * bf16,
                            stream,
                        )?;
                    }
                    let moe_out = self.ffn.forward(normed2_i, i, fwd, stream)?;
                    ops::residual_add(
                        fwd.gpu,
                        self.residual_add_k,
                        hidden_i,
                        moe_out,
                        h as u32,
                        stream,
                    )?;
                    // LongCat shortcut carry (consumer): the paired previous
                    // sublayer's stashed MoE output, this sequence's row.
                    if let Some((carry, _cap)) = self.shortcut_carry_in {
                        ops::residual_add(
                            fwd.gpu,
                            self.residual_add_k,
                            hidden_i,
                            carry.offset(i * h * bf16),
                            h as u32,
                            stream,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}
