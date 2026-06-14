// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_multi_seq.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    /// Multi-sequence decode for SSM (gated-delta-net) layers.
    ///
    /// The SSM mixer (conv1d + GDN recurrence + in/out projections) carries
    /// independent per-sequence recurrent state, so it runs in a per-seq loop
    /// using the SAME single-token kernels as `decode()` (proven correct). The
    /// MoE sublayer is stateless and shared across sequences, so it is hoisted
    /// OUT of the loop and run ONCE as a batched grouped-GEMM over all N
    /// tokens — the same `forward_prefill` path the prefill scheduler and the
    /// attention layers' multi-seq path already use.
    ///
    /// This supersedes the earlier "delegate every sequence to the full
    /// single-token `decode()`" fallback, which ran N separate single-token
    /// MoE forwards (N × top_k expert GEMVs + N per-token all_reduces under
    /// EP). Phase B collapses those to one grouped gate+up+down GEMM and one
    /// batched all_reduce.
    ///
    /// Buffer safety (the old bug #6): each per-seq mixer writes its MoE input
    /// to `norm_output[i]` — a distinct per-seq offset. `ssm_forward` never
    /// touches `norm_output` (verified: 0 references) and its returned
    /// `ssm_out` (in `moe_output[0]`) is consumed by the same iteration's
    /// `residual_add_rms_norm` before the next iteration runs, so nothing
    /// needs to survive across sequences and no aliasing is possible.
    /// `forward_prefill` then reads the assembled `norm_output[0..n]` and
    /// writes `moe_output[0..n]`.
    pub(super) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_seqs;

        // Per-seq hidden/residual stride: the residual stream is always
        // BF16 (2 bytes), so hardcode the per-seq stride.
        let residual_elem = 2usize;

        // ── Phase A: per-sequence SSM mixer ──
        // Pre-norm, SSM mixer (recurrent, per-seq state), post-attn-norm.
        // Lays out `norm_output[0..n]` as the contiguous [N, h] BF16 MoE
        // input. Identical kernel sequence to `decode()`'s mixer; only the
        // MoE is deferred to Phase B.
        for i in 0..n {
            let hidden_i = hidden.offset(i * h * residual_elem);
            let residual_i = residual.offset(i * h * residual_elem);
            let normed_i = ctx.buffers.norm_output().offset(i * h * bf16);

            let ssm_state = states[i]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;

            // normed_i = rms_norm(hidden_i); residual_i = hidden_i
            ops::rms_norm_residual(
                ctx.gpu,
                self.rms_norm_residual_k,
                hidden_i,
                &self.input_norm,
                normed_i,
                residual_i,
                1,
                h as u32,
                eps,
                stream,
            )?;

            // SSM mixer: consumes normed_i, returns ssm_out (in moe_output[0]).
            let ssm_out = self.ssm_forward(normed_i, ssm_state, ctx, stream, false)?;

            // hidden_i += ssm_out; normed_i = rms_norm(hidden_i); residual_i = hidden_i
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden_i,
                ssm_out,
                &self.post_attn_norm,
                normed_i,
                residual_i,
                1,
                h as u32,
                eps,
                stream,
            )?;
        }

        // ── Phase B+C: MoE + residual, dispatched by batch size ──
        // Measured on GB10 (qwen3.5-122b, 256-expert MoE, EP=2):
        //   N=2/3: the FUSED batch-2/3 expert kernels (forward_k2/k3) win —
        //          SSM step 44->36.5ms at N=2 (one batched all_reduce, no
        //          per-token launch overhead).
        //   N>=4:  the generic grouped-GEMM (forward_prefill) is a NET LOSS
        //          here — per-expert M ~1, and the expert sort/permute/ptr-
        //          table overhead (paid once per layer, x36 SSM layers)
        //          dominates (SSM step ~88ms per-token vs ~140ms grouped).
        //          So fall back to the per-token MoE loop, identical to
        //          decode()'s MoE — the fastest option at these sizes until
        //          a true batched-EP MoE kernel exists.
        // Mirrors the attention layers' forward_k2/k3 dispatch
        // (qwen3_attention/.../multi_seq/ffn.rs); diverges only in declining
        // forward_prefill at N>=4, which that path uses but which loses for
        // the 36-layer SSM stack.
        let normed_base = ctx.buffers.norm_output();
        match n {
            2 | 3 => {
                if n == 2 {
                    self.ffn.forward_k2(normed_base, ctx, stream)?;
                } else {
                    self.ffn.forward_k3(normed_base, ctx, stream)?;
                }
                // Batched output lives in moe_output[0..n].
                for i in 0..n {
                    let hidden_i = hidden.offset(i * h * residual_elem);
                    let moe_out_i = ctx.buffers.moe_output().offset(i * h * bf16);
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden_i,
                        moe_out_i,
                        h as u32,
                        stream,
                    )?;
                }
            }
            _ => {
                // Per-token MoE: each seq's forward() writes moe_output[0];
                // consume it immediately with a per-seq residual add before
                // the next iteration overwrites it.
                for i in 0..n {
                    let hidden_i = hidden.offset(i * h * residual_elem);
                    let normed_i = normed_base.offset(i * h * bf16);
                    let moe_out = self.ffn.forward(normed_i, ctx, stream)?;
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden_i,
                        moe_out,
                        h as u32,
                        stream,
                    )?;
                }
            }
        }

        Ok(())
    }
}
