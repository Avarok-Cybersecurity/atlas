// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-sequence batched decode for `NemotronMoeLayer` (Milestone A).
//!
//! The default `decode_multi_seq` loop runs the single-token `decode_inner`
//! per sequence: N gate GEMVs, N shared-expert passes and N×top_k expert
//! GEMVs re-stream the same weights N times per layer per step. This
//! override runs the PREFILL body over the N decode rows instead — batched
//! gate GEMM, one shared-expert UP pass, sorted grouped-GEMM expert dispatch
//! — so every weight matrix is read once. Math is identical to
//! `decode_inner` (same weights, same precision); only the kernels differ
//! (grouped GEMM + elementwise relu² vs per-token `moe_expert_gemv` + fused
//! down), a tiny FP-reordering delta that prefill already exhibits for these
//! exact weights.
//!
//! Pad rows (`num_seqs = padded_n` includes them): zeroed hidden → normed=0
//! → sigmoid routing picks arbitrary experts → a few junk expanded rows in
//! the grouped GEMM. Harmless (their hidden rows are never sampled) and
//! cheap (≤ (padded_n−n)·top_k extra rows). Do not skip them — the batched
//! kernels need contiguous rows.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::NemotronMoeLayer;
use super::prefill_sorted::SortedPrefillCtx;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl NemotronMoeLayer {
    /// Whether the batched decode path can run: gates on exactly what the
    /// sorted path launches (batched routing, expert sort, grouped GEMM,
    /// unpermute-reduce, elementwise relu² for the shared expert). NOT
    /// prefill's `has_batched` trio — `nemotron_moe_up/down_prefill` are not
    /// used by the sorted path.
    pub(super) fn can_batch_decode(&self) -> bool {
        self.topk_sigmoid_batched_k.0 != 0
            && self.moe_sort_k.0 != 0
            && self.moe_grouped_gemm_k.0 != 0
            && self.moe_unpermute_reduce_k.0 != 0
            && self.moe_relu2_elementwise_k.0 != 0
    }

    /// Batched N-sequence MoE decode: the `prefill()` body over `num_seqs`
    /// decode rows, always taking the sorted grouped-GEMM dispatch
    /// (`can_batch_decode` checked by the caller). The Super-120B latent
    /// variant works for free — the sorted path branches on
    /// `moe_latent_size` internally.
    pub(super) fn decode_multi_seq_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let inter = self.moe_inter as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = self.top_k as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let scale = ctx.config.routed_scaling_factor as f32;
        let n = num_seqs as u32;

        // Routing indices/weights live at scratch()[0..2*n*top_k*4). The
        // batch attention metadata parked at scratch()+32768 by
        // `upload_batch_metadata_fixed` is read EVERY layer of this same
        // step by the attention layers — it must not be clobbered. Worst
        // case here is 128 rows × top_k≈6 → 6 KB, but assert the invariant.
        // The MIXED path (decode_b.rs) parks the prefill chunk's
        // positions/slots in scratch too; its offset is sized to clear this
        // region for every padded_n (`trait_impl/mixed_layout.rs`).
        debug_assert!(
            2 * num_seqs * self.top_k * 4 <= 32768,
            "MoE batched-decode routing scratch ({} rows × top_k {}) would \
             clobber the batch attention metadata at scratch()+32768",
            num_seqs,
            self.top_k,
        );

        // 1. Batched RMS norm: [N, H] → normed[N, H] + residual save.
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )?;

        // 2. Batched gate GEMM: [N, H] → gate_logits[N, E].
        let gate_logits = ctx.buffers.gate_logits();
        self.dense_gemm_prefill(
            ctx.gpu,
            normed,
            &self.weights.gate,
            gate_logits,
            n,
            num_experts,
            h as u32,
            stream,
        )?;

        // 3. Shared expert UP for all N rows, weights read once (arm
        //    selection — native FP8 / pd-FP8 / transposed NVFP4 / W4A16 —
        //    lives in prefill_shared_up.rs).
        let shared_up_out_base = ctx.buffers.ssm_qkvz();
        self.prefill_shared_up(normed, shared_up_out_base, n, h, shared_inter, ctx, stream)?;

        // 4. LatentMoE only (Super 120B): fc1 GEMM [N, H] → [N, L] into
        //    attn_output (safe here: the mamba2 layers' y buffer is dead by
        //    the time this layer runs on the ordered stream).
        let latent = self.moe_latent_size as u32;
        let latent_base = if latent > 0 {
            let latent_buf = ctx.buffers.attn_output();
            if let Some(w_fp8) = self.fc1_pd_fp8 {
                ops::fp8_gemm_m128_mfast(
                    ctx.gpu,
                    self.fp8_gemm_m128_k,
                    normed,
                    w_fp8,
                    latent_buf,
                    n,
                    latent,
                    h as u32,
                    stream,
                )?;
            } else {
                let fc1 = self.weights.fc1_latent_proj.as_ref().unwrap();
                self.dense_gemm_prefill(
                    ctx.gpu, normed, fc1, latent_buf, n, latent, h as u32, stream,
                )?;
            }
            Some(latent_buf)
        } else {
            None
        };

        // 5. Sorted grouped-GEMM dispatch: batched sigmoid routing → expert
        //    sort → grouped UP (+relu²) → grouped DOWN → unpermute-reduce →
        //    shared relu² + shared DOWN (into ssm_deinterleaved, NOT
        //    attn_output — the documented mamba2 y-buffer hazard) → two
        //    residual adds over N*h.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(num_seqs * self.top_k * 4);
        let p = SortedPrefillCtx {
            n,
            num_tokens: num_seqs,
            h,
            inter,
            shared_inter,
            num_experts,
            top_k,
            scale,
            latent,
            gate_logits,
            indices_dev,
            weights_dev,
            normed,
            hidden,
            latent_base,
            shared_up_out_base,
        };
        self.prefill_sorted_path(&p, ctx, stream)
    }
}
