// SPDX-License-Identifier: AGPL-3.0-only

//! The GDN prefill under an mHC highway (Qwen3.8-Flash-Next).
//!
//! Not a wrapper around `prefill_inner`, for a structural reason.
//! `prefill_inner` FUSES its residual bookkeeping into its norms:
//!
//! ```text
//! rms_norm_residual(hidden, input_norm) -> normed, residual      # step 1
//!   prefill_block(normed)               -> out_proj_buf          # steps 2-10
//! residual_add_rms_norm(hidden, out_proj_buf, post_attn_norm)    # step 11
//! ffn.forward_prefill(norm_output)                               # step 12
//! residual_add(hidden, moe_output)                               # step 13
//! ```
//!
//! Under mHC the highway IS the residual, and the block output has to reach
//! it through `hc_post` — scaled per stream by the injection vector that
//! `hc_pre` emitted. Steps 1, 11 and 13 would each add it a second time. So
//! this path replaces them rather than running them, and the shape below
//! mirrors `qwen3_attention`'s `prefill_inner_hc` exactly:
//!
//! ```text
//! hc_expand(hidden -> streams)                    # MODEL layer 0 only
//! hc_pre(streams, attn_site) -> hidden, inj       # `hidden` is scratch here
//!   prefill_block(hidden)    -> out_proj_buf
//! hc_post(out_proj_buf, streams, inj) -> streams
//! hc_pre(streams, ffn_site)  -> hidden, inj
//!   ffn.forward_prefill(hidden) -> moe_output
//! hc_post(moe_output, streams, inj) -> streams
//! ```
//!
//! No `input_norm`, no `post_attn_norm`, no `residual_add`: on this model
//! those slots hold ones-filled placeholders because the checkpoint has no
//! per-layer norms at all — `hc_norm` inside `hc_pre` is the norm.
//!
//! ## Why `hc_expand` lives here
//!
//! It used to fire on `attn_layer_idx == 0`. With `layer_types` interleaving
//! 3:1, model layers 0-2 are GDN and attention layer 0 is model layer 3 — so
//! the highway was being seeded three layers late, on top of whatever the
//! buffer held. The seeding layer and the collapsing layer are different
//! concrete types on this model, and this is the seeding one.
//!
//! There is no `hc_head` here: the LAST model layer (47) is full attention,
//! so the final collapse stays on the attention path.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_inner_hc(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        seq_len_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("prefill_inner_hc without mHC weights"))?;
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens as u32;

        // Same counter the non-HC path bumps: one increment per SSM layer per
        // prefill, so `ATLAS_GDN_DUMP` still attributes an intermediate to the
        // right layer.
        let ssm_layer_idx =
            super::debug::SSM_LAYER_CALL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Stage profiler, the prefill twin of ATLAS_QWEN4EXP_DECODE_PROF:
        // serialize at stage seams and log per-stage µs for the first ~8
        // chunks (48 layer calls each). Prefill sits at ~110-130 tok/s vs a
        // 300-600 llama.cpp reference; the ranked suspects (GDN chunk path,
        // MoE grouped GEMM, PLE host work) live at different seams here.
        static PROF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static PROF_LEFT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(400);
        let prof = *PROF
            .get_or_init(|| std::env::var("ATLAS_QWEN4EXP_PREFILL_PROF").as_deref() == Ok("1"))
            && PROF_LEFT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed) > 0;
        let mut t = if prof {
            ctx.gpu.synchronize(stream).ok();
            Some(std::time::Instant::now())
        } else {
            None
        };
        macro_rules! stage {
            ($name:expr) => {
                if let Some(t0) = t.as_mut() {
                    ctx.gpu.synchronize(stream).ok();
                    tracing::info!(
                        "hc-prefill L{ssm_layer_idx} T={num_tokens} [{}]: {}us",
                        $name,
                        t0.elapsed().as_micros()
                    );
                    *t0 = std::time::Instant::now();
                }
            };
        }

        // ATLAS_FP32_ROUTING has the SSM's fused `residual_add_rms_norm_gatef32`
        // populate `moe_router_in_f32` for the gate GEMM to read at full
        // precision. This path does not run that kernel — `hc_norm` inside
        // `hc_pre` is the norm — so the buffer would hold the PREVIOUS layer's
        // activations and the router would route on them. Latent today (the
        // flag is off by default) and silent if it ever is not.
        anyhow::ensure!(
            !self.ffn.fp32_routing_active(),
            "qwen3_ssm mHC: ATLAS_FP32_ROUTING needs the fused gate-f32 norm, \
             which the highway path replaces. The router would read a stale \
             moe_router_in_f32. Unset it."
        );

        // Mixed steps park the prefill chunk's highway rows above the
        // decode rows (ctx.hc_row_offset = padded decode count; 0 elsewhere).
        let streams = ctx
            .buffers
            .hc_streams()
            .offset(ctx.hc_row_offset * hc.hc_mult * h * 4);
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();

        if hc.is_first_model_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                streams,
                n,
                h as u32,
                hc.hc_mult as u32,
                stream,
            )?;
        }

        // PLE injects into the highway BEFORE this layer's own
        // hyper-connection — the reference's
        // `hidden_states = hidden_states + self.ple(...)` sits above
        // `attn_hyper_connection` in `Qwen4ExpTextDecoderLayer.forward`.
        // `fresh` is a prefill starting at position 0: a new sequence, so the
        // conv state and the token history both reset.
        if let Some(ple) = self.ple.as_ref() {
            let ssm = state
                .as_any_mut()
                .downcast_mut::<crate::layer::SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("PLE host layer state is not SsmLayerState"))?;
            if ssm.ple.is_none() {
                ssm.ple = Some(ple.new_seq_state(ctx.gpu)?);
            }
            let st = ssm.ple.as_mut().expect("just created");
            ple.forward(st, streams, num_tokens, seq_len_start == 0, ctx, stream)?;
        }
        stage!("ple");

        // ── GDN sublayer ──
        // `hidden` is scratch from here on: the highway carries the state
        // between layers, exactly as on the attention path.
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            &hc.attn,
            hc,
            hidden,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n,
            h as u32,
            eps,
            stream,
        )?;
        stage!("hc_pre_attn");
        let hc_dim = hc.hc_mult * h;
        crate::layers::ple::dump::tap_highway(
            ctx.gpu,
            streams,
            ssm_layer_idx,
            "in",
            num_tokens,
            hc_dim,
            stream,
        );
        // Tap hc_pre's BOTH outputs. `L00_post_gdn` diverges at cosine 0.80
        // with 2.6x the reference magnitude, and the sublayer is exactly
        // hc_pre -> block -> hc_post, so splitting it three ways is the whole
        // remaining search.
        crate::layers::ple::dump::tap_bf16(
            ctx.gpu,
            hidden,
            ssm_layer_idx,
            "hc_pre_mixed",
            num_tokens * h,
            stream,
        );
        crate::layers::ple::dump::tap_f32(
            ctx.gpu,
            post,
            ssm_layer_idx,
            "hc_pre_inj",
            num_tokens * hc.hc_mult,
            stream,
        );
        let out_proj_buf =
            self.prefill_block(hidden, num_tokens, state, ssm_layer_idx, ctx, stream)?;
        stage!("gdn_block");
        crate::layers::ple::dump::tap_bf16(
            ctx.gpu,
            out_proj_buf,
            ssm_layer_idx,
            "block_out",
            num_tokens * h,
            stream,
        );
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            out_proj_buf,
            streams,
            post,
            comb,
            streams,
            n,
            h as u32,
            stream,
        )?;

        stage!("hc_post_attn");
        // Tapped BEFORE the MoE on purpose: reproducing this point in the
        // reference needs only the GDN projections, not 512 experts.
        crate::layers::ple::dump::tap_highway(
            ctx.gpu,
            streams,
            ssm_layer_idx,
            "post_gdn",
            num_tokens,
            hc_dim,
            stream,
        );

        // ── MoE sublayer ──
        // `prefill_block` returned `ctx.buffers.moe_output()`, which the FFN
        // is about to overwrite — safe only because the `hc_post` above has
        // already consumed it into the highway. Keep that order.
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            &hc.ffn,
            hc,
            hidden,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n,
            h as u32,
            eps,
            stream,
        )?;
        stage!("hc_pre_ffn");
        // Small-M FFN: at K=2/K=3 rows this body is a speculative VERIFY, not a
        // real prefill. forward_prefill routes the MoE through the grouped GEMM,
        // which streams every expert's weights and so costs ~9ms/layer almost
        // independently of row count (measured: T=16 6.7-9.6ms, T=28 8.5-12.3ms
        // — 1.75x the rows for 1.2x the time). The fused K=2/K=3 kernels do the
        // same rows in 5 launches at decode cost. Both write moe_output(), so
        // this is a kernel-shape substitution, not a math change.
        // ATLAS_QWEN4EXP_HC_SMALL_M_FFN=0 restores the grouped path for A/B.
        let small_m = {
            static SMALL_M: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *SMALL_M
                .get_or_init(|| std::env::var("ATLAS_QWEN4EXP_HC_SMALL_M_FFN").as_deref() != Ok("0"))
        };
        // NOTE the K=1 arm carries the weight here: `decode_verify_hc` splits a
        // verify into row-0-then-drafts, so at the default gamma=1 BOTH calls
        // arrive as ONE row and the k2/k3 arms never fire.
        match num_tokens {
            1 if small_m => {
                let out = self.ffn.forward(hidden, ctx, stream)?;
                anyhow::ensure!(
                    out == ctx.buffers.moe_output(),
                    "small-M FFN: single-token MoE returned a buffer other than \
                     moe_output(), which the hc post-site reads unconditionally"
                );
            }
            2 if small_m => self.ffn.forward_k2(hidden, ctx, stream)?,
            3 if small_m => self.ffn.forward_k3(hidden, ctx, stream)?,
            _ => self.ffn.forward_prefill(hidden, num_tokens, ctx, stream)?,
        }
        stage!("moe");
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            ctx.buffers.moe_output(),
            streams,
            post,
            comb,
            streams,
            n,
            h as u32,
            stream,
        )?;
        crate::layers::ple::dump::tap_highway(
            ctx.gpu,
            streams,
            ssm_layer_idx,
            "post_moe",
            num_tokens,
            hc_dim,
            stream,
        );
        stage!("hc_post_ffn");

        Ok(())
    }
}
