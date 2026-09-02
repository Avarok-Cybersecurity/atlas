// SPDX-License-Identifier: AGPL-3.0-only

//! Batched (N-sequence) decode for the mHC-highway GDN layer — the piece
//! `refuse_batched_under_hc` existed to guard until it was built (#753
//! item B, milestone 2: "real concurrent batches, not queued interleaving").
//!
//! Mirrors the attention side's `decode_multi_seq_inner_hc`
//! (`qwen3_attention/trait_impl/multi_seq/mod.rs`, built for DeepSeek-V4):
//! the highway REPLACES the layer's own residual bookkeeping, so the
//! non-hc path's fused `rms_norm_residual` / `residual_add_rms_norm` steps
//! must not run. Structure per layer:
//!
//!   hc_expand [N]           (first model layer only)
//!   PLE per-seq mini-loop   (each row against its own PleSeqState)
//!   hc_pre  [N]  -> mixed rows in `hidden`
//!   ssm_forward per seq     (recurrence is inherently per-seq; outputs
//!                            staged into norm_output rows — ssm_forward
//!                            reuses moe_output[0] every call)
//!   hc_post [N]  <- norm_output rows
//!   hc_pre  [N]  -> mixed rows in norm_output (the MoE input rows)
//!   MoE: forward_k2 at N=2, one n-row forward_token_major_decode above
//!        that (rows in moe_output); N=1 and the `hc_batched_moe_decode`
//!        kill switch keep the per-row loop staged into `hidden` rows
//!        (the mixed GDN inputs are dead by then)
//!   hc_post [N]
//!
//! The batched-GEMM projection lever (`try_decode_multi_seq_ssm_batched`)
//! is deliberately NOT engaged yet: it folds the residual into its
//! epilogues. Amortizing the QKVZ/out_proj weight reads across N rows
//! under the highway is the next increment.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3SsmLayer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::hc_ffn_plan::{HcFfnPlan, hc_ffn_plan};
use crate::layers::ops;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn decode_multi_seq_inner_hc<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        seq_lens: &[usize],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_seqs;
        let bf16 = 2usize;
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode_multi_seq_inner_hc without mHC weights"))?;
        let hc_mult = hc.hc_mult as u32;
        let streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        let normed = ctx.buffers.norm_output();
        let moe_out = ctx.buffers.moe_output();

        if hc.is_first_model_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                streams,
                n as u32,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        // ── PLE: per-seq rows against per-seq state ──
        // The highway rows are FP32 [n, hc*h]; each sequence's injection
        // reads ITS token id and ITS conv/history carry. `decode()` runs
        // this via prestage; here the forward does its own host half — the
        // per-seq token comes from `ctx.host_token_ids` via the override.
        if let Some(ple) = self.ple.as_ref() {
            let host = ctx.host_token_ids.ok_or_else(|| {
                anyhow::anyhow!("hc multi-seq decode: PLE needs host_token_ids threaded")
            })?;
            for (i, state) in states.iter_mut().enumerate().take(n) {
                // Padding rows (seq_len 0 — the mixed step pads n_decode up
                // to the batch-kernel width) carry a dummy state with no PLE
                // carry and no host id; their highway rows are discarded, so
                // the injection is skipped, NOT zero-filled.
                if seq_lens.get(i).copied() == Some(0) {
                    continue;
                }
                anyhow::ensure!(
                    host.len() > i,
                    "hc multi-seq decode: {} host ids for real seq row {i}",
                    host.len()
                );
                let ssm = state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
                let st = ssm.ple.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("PLE multi-seq decode before prefill: no seq state {i}")
                })?;
                ple.forward_row(
                    st,
                    streams.offset(i * (hc_mult as usize) * h * 4),
                    &host[i..i + 1],
                    ctx,
                    stream,
                )?;
            }
        }

        // ── GDN sublayer ──
        // hc_pre writes the mixed+normed rows straight into norm_output —
        // both the batched-GEMM core and the per-seq fallback read there.
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            &hc.attn,
            hc,
            normed,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        // Batched-GEMM core (QKVZ/out_proj weights read ONCE for all n rows —
        // the C=N scaling lever on bandwidth-bound LPDDR5X); out rows land in
        // moe_output[0..n). `hidden`/`residual` are unused in hc mode.
        let gdn_rows = if self.try_decode_multi_seq_ssm_batched(
            hidden,
            DevicePtr::NULL,
            n,
            states,
            true,
            ctx,
            stream,
        )? {
            moe_out
        } else {
            // Fallback: per-seq recurrence reading the mixed rows; each
            // output (ssm_forward reuses moe_output[0]) staged into `hidden`
            // rows, which are dead in hc mode.
            for (i, state) in states.iter_mut().enumerate().take(n) {
                let ssm_state = state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
                let ssm_out =
                    self.ssm_forward(normed.offset(i * h * bf16), ssm_state, ctx, stream, false)?;
                ctx.gpu
                    .copy_d2d_async(ssm_out, hidden.offset(i * h * bf16), h * bf16, stream)?;
            }
            hidden
        };
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            gdn_rows,
            streams,
            post,
            comb,
            streams,
            n as u32,
            h as u32,
            stream,
        )?;

        // ── MoE sublayer ──
        // hc_pre writes the mixed rows straight into norm_output — the
        // batched expert kernels' input convention.
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            &hc.ffn,
            hc,
            normed,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        let plan = hc_ffn_plan(
            n,
            !self.ffn.is_none(),
            ctx.config.hc_batched_moe_decode,
            true,
        );
        let moe_rows = match plan {
            // No FFN on this layer: the mixed rows ARE the output, exactly as
            // the per-row `FfnComponent::forward` (returns its input) left them.
            HcFfnPlan::Passthrough => normed,
            // n == 2 keeps the shipping fused pair kernel — byte-identical.
            HcFfnPlan::FusedK2 => {
                self.ffn.forward_k2(normed, ctx, stream)?;
                moe_out
            }
            // ONE n-row dispatch for the whole batch (#753 item B follow-up).
            // `forward_token_major_decode` delegates per arm — EXL3 native,
            // NVFP4 token-major, LoRA/FP8/BF16/T-layout via forward_batched,
            // dense FFN via forward_batched — and issues exactly one EP
            // all-reduce over all n rows. The n == 3 `forward_k3` arm is gone:
            // `padded_batch_n` (traits/model.rs) maps 3 to 4, so it was dead,
            // and this arm is correct for every n regardless.
            HcFfnPlan::Batched => {
                self.ffn
                    .forward_token_major_decode(normed, n, ctx, stream)?;
                moe_out
            }
            HcFfnPlan::PerRow => {
                // Per-row loop; stage into `hidden` rows (the mixed GDN
                // inputs there are dead once the recurrence loop above ran).
                for i in 0..n {
                    let out = self.ffn.forward(normed.offset(i * h * bf16), ctx, stream)?;
                    ctx.gpu
                        .copy_d2d_async(out, hidden.offset(i * h * bf16), h * bf16, stream)?;
                }
                hidden
            }
        };
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            moe_rows,
            streams,
            post,
            comb,
            streams,
            n as u32,
            h as u32,
            stream,
        )?;
        Ok(())
    }
}
