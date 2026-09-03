// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_batched.

use super::*;

/// ATLAS_K4_DIAG=1 phase checkpoint (see verify_c2.rs). Synchronizes the
/// stream after a named phase of the batched GDN decode so an illegal access
/// is attributed to the exact op. No-op (and no env read past the first call)
/// unless the diagnostic env is set. Only legal in eager mode — verify_c2
/// disables CUDA-graph capture whenever the env is set, and this checkpoint
/// is only reachable from that eager path.
fn k4_diag_checkpoint(ctx: &ForwardContext, phase: &str, stream: u64) -> Result<()> {
    let on = ctx.levers.k4_diag;
    if on
        && !ctx.graph_capture
        && let Err(e) = ctx.gpu.synchronize(stream)
    {
        anyhow::bail!("K4_DIAG: CUDA error after GDN phase `{phase}`: {e:#}");
    }
    Ok(())
}

/// Presence kill switch for the single-launch fused BA projection + GDN gates
/// (`ATLAS_NO_BATCHED_BA_GATES` restores the per-token GEMV + `compute_gdn_gates`
/// pair). Presence, not value — `=0` is NOT "off".
fn batched_ba_gates_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_NO_BATCHED_BA_GATES").is_err())
}

/// Presence kill switch for the single-launch gated RMS norm
/// (`ATLAS_NO_BATCHED_GDN_NORM` restores the per-token loop). The batched kernel
/// is bit-identical, so this switch exists only to isolate the change in an A/B.
fn batched_norm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_NO_BATCHED_GDN_NORM").is_err())
}

/// Row count above which the batched decode/verify GDN projections stop taking
/// the FP8 PREFILL arm (`fp8_gemm_n128` on the single-scale FP8 copy) and read
/// the NVFP4 twin through a tile GEMM instead. SSOT for both projections —
/// QKVZ and out_proj make the same trade on the same rows, and the threshold
/// has the same derivation for both.
///
/// DERIVED, not tuned: every weight-streaming NVFP4 GEMV in this dispatcher
/// (`w4a16_gemv_batch2/3/4`, and `w4a16_gemv_batchm` on the batch8 handle)
/// caps at M=8, so 8 is the last row count those arms can serve. Above it the
/// chain reaches the tile GEMMs, and that is exactly where the choice of
/// weight copy starts to cost bandwidth.
pub(super) const VERIFY_TGEMM_MIN_TOKENS: usize = 8;

/// Presence kill switch for the NVFP4 QKVZ arm of the batched decode/verify
/// projection — `ATLAS_NO_QKVZ_NVFP4_DECODE` restores the FP8-prefill-copy
/// dispatch verbatim. PRESENCE, not value, per house convention (`=0` is NOT
/// "off"). Read ONCE: this site runs under CUDA-graph capture.
fn qkvz_nvfp4_decode_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("ATLAS_NO_QKVZ_NVFP4_DECODE").is_ok())
}

/// Should the batched decode/verify QKVZ projection read the NVFP4 transposed
/// twin instead of the single-scale FP8 prefill copy?
///
/// PURE (SBIO): every input is a parameter — row count, which weight copies the
/// layer actually holds, and the already-resolved kill switch — so the dispatch
/// decision is decidable without a GPU, a layer, or a process-global env read.
///
/// `has_fp8_prefill` is a precondition, not a preference: this predicate exists
/// only to divert the FP8 arm. When that copy is absent the chain's own NVFP4
/// arms already run and must keep running unchanged.
///
/// `has_tile_gemm` is `deep_k_gemm(K).0 != 0` — the terminal handle
/// `ms_proj_gemm` falls back to. Launching a 0 handle is the NULL-dispatch
/// class this dispatcher has already been bitten by three times, so the arm
/// declines rather than assuming a target carries the kernel.
pub(super) fn qkvz_verify_nvfp4_wins(
    num_tokens: usize,
    has_fp8_prefill: bool,
    has_nvfp4_t: bool,
    has_tile_gemm: bool,
    kill_switch: bool,
) -> bool {
    !kill_switch
        && has_fp8_prefill
        && has_nvfp4_t
        && has_tile_gemm
        && num_tokens > VERIFY_TGEMM_MIN_TOKENS
}

/// GDN-state routing for [`Qwen3SsmLayer::decode_batched_inner`].
///
/// `Single`: today's path — `num_tokens` rows belong to ONE sequence, whose
/// conv/GDN state advances through all of them (K-token MTP verify, DFlash).
///
/// `Multi`: batched MTP verify — `num_tokens = Σ ks` seq-major rows, RAGGED
/// per sequence since D-Cut (`ks[i]` rows for sequence i, uniform being the
/// special case); projections/FFN batch across all rows, while the stateful
/// conv/GDN body runs per-sequence against `states[i]` with row-offset buffer
/// bases (per-sequence math byte-identical to `Single` at `num_tokens =
/// ks[i]`; only base addresses move).
pub(super) enum GdnStates<'a, 'b> {
    Single(&'a mut dyn LayerState),
    Multi {
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ks: &'a [usize],
        /// This layer's slice of the model-staged WY pointer tables
        /// (`crate::layer::VERIFY_WY_LAYER_STRIDE_BYTES`; NULL → no
        /// single-launch WY batch, per-sequence loop only).
        wy_tables: DevicePtr,
    },
}

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_batched_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gdn: GdnStates<'_, '_>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize; // bytes per BF16

        // ── 1. RMS norm + residual for K tokens ──
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            k,
            h as u32,
            eps,
            stream,
        )?;

        k4_diag_checkpoint(ctx, "1:rms_norm_residual", stream)?;

        // ── 2-9. The residual-free GDN block. ──
        // Everything between the two residual sites depends only on the
        // normed rows and the per-sequence GDN state — `hidden`/`residual`
        // appear nowhere in it. That is what lets the mHC twin
        // (`decode_batched_inner_hc`) reuse it verbatim under a different
        // bracket.
        let out_proj_buf = self.decode_batched_block(normed, num_tokens, gdn, ctx, stream)?;

        // ── 10. Batched residual + post-norm, then MoE + residual ──
        // residual_add_rms_norm supports multi-token (grid.x = num_tokens)
        let normed2_base = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            out_proj_buf,
            &self.post_attn_norm,
            normed2_base,
            residual,
            num_tokens as u32,
            h as u32,
            eps,
            stream,
        )?;
        if num_tokens == 3 {
            // Fused K=3 MoE: 5 kernel launches instead of 15
            self.ffn.forward_k3(normed2_base, ctx, stream)?;
            let moe_out = ctx.buffers.moe_output();
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (3 * h) as u32,
                stream,
            )?;
        } else if num_tokens == 2 {
            // Fused K=2 MoE: 5 kernel launches instead of 10
            self.ffn.forward_k2(normed2_base, ctx, stream)?;
            // Batched residual add for 2 tokens (flat element-wise, 2*h elements)
            let moe_out = ctx.buffers.moe_output();
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (2 * h) as u32,
                stream,
            )?;
        } else if (4..=8).contains(&num_tokens)
            && self
                .ffn
                .try_forward_km(normed2_base, num_tokens as u32, ctx, stream)
                .inspect_err(|e| tracing::error!("ffn.try_forward_km: {e:#}"))
                .unwrap_or(false)
        {
            // K=4..8 verify FFN via batched GEMV (batch4 M<=4, batch8
            // M=5..8): one weight read per projection for all rows at
            // near-peak stream bandwidth. nsys (2026-07-18): the
            // forward_prefill MMQ arm below cost 54.8 ms/verify-step across
            // the 64-layer dense FFN stack at M=4 vs the ~31 ms
            // weight-traffic floor this path hits. Falls through to
            // forward_prefill when unavailable (MoE / missing kernel).
            let moe_out = ctx.buffers.moe_output();
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (num_tokens * h) as u32,
                stream,
            )?;
        } else if self.ffn.is_dense() {
            // WIDE-VERIFY BATCHED DENSE FFN (DFlash γ=16, num_tokens=17). This
            // is the MAJORITY layer type (GDN/SSM) on the hybrid 27B, so its
            // per-token FFN loop was the dominant remaining verify cost after
            // the attention layers were batched. normed2_base is already
            // [num_tokens, h] (batched residual_add_rms_norm above), so
            // forward_prefill reads gate/up/down ONCE for all tokens.
            //
            // DENSE ONLY: the per-token `else` below is retained for 256-expert
            // MoE, where grouped-GEMM is a net loss at small batch (per-expert
            // M~1 + sort/permute overhead across the 36-layer SSM stack).
            k4_diag_checkpoint(ctx, "10a:residual_add_rms_norm", stream)?;
            self.ffn
                .forward_prefill(normed2_base, num_tokens, ctx, stream)?;
            k4_diag_checkpoint(ctx, "10b:ffn_forward_prefill", stream)?;
            let moe_out = ctx.buffers.moe_output();
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (num_tokens * h) as u32,
                stream,
            )?;
        } else {
            // Per-token MoE fallback for K!=2 (256-expert MoE).
            // CONCURRENT-DECODE BUG (sibling of decode_multi_seq fix at line 1102):
            // hardcoded `t * h * 4` over-strides for BF16 hidden (GB10 default).
            let residual_elem = 2usize;
            for t in 0..(num_tokens as u32) {
                let normed2 = normed2_base.offset(t as usize * h * bf16);
                let moe_out = self.ffn.forward(normed2, ctx, stream)?;
                let hidden_t = hidden.offset(t as usize * h * residual_elem);
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden_t,
                    moe_out,
                    h as u32,
                    stream,
                )?;
            }
        }

        Ok(())
    }

    /// Steps 2-9 of the batched GDN decode: the QKVZ projection, the BA
    /// projection and GDN gates, the conv1d/L2-norm/GDN scan (which is what
    /// writes the per-row `h_state_intermediates` /
    /// `conv_state_intermediates` that `commit_accepted_prefix` rewinds
    /// from), the gated RMS norm and the output projection.
    ///
    /// RESIDUAL-FREE BY CONSTRUCTION. `hidden` and `residual` are not
    /// parameters because nothing here touches them: the caller owns both
    /// residual sites (`rms_norm_residual` before, `residual_add_rms_norm` +
    /// `residual_add` after) and an mHC caller replaces them with
    /// `hc_pre_site` / `hc_post_site` instead.
    ///
    /// Returns the `[num_tokens, H]` block output — `ctx.buffers.moe_output()`,
    /// which the FFN is about to overwrite, so the caller must consume it
    /// before dispatching the FFN.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_batched_block(
        &self,
        normed: DevicePtr,
        num_tokens: usize,
        gdn: GdnStates<'_, '_>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize; // bytes per BF16
        let fp32 = 4usize; // bytes per FP32

        // Whether THIS pass must run its rows as sequential decode rows —
        // SSOT for the projections, the conv+GDN arm and the phase-8 norm
        // skip. Resolved once: it is a pure function of process flags and the
        // pass's own contract, so it is CUDA-graph-stable.
        let row_exact =
            super::verify_row_exact_leg(ctx.gdn_exact_replay, super::RowExactLeg::Proj);

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd; // 2048
        let value_dim = nv * vd; // 4096
        let conv_dim = key_dim * 2 + value_dim; // 8192
        let qk_ch = (key_dim * 2) as u32; // Q+K channels for fused L2 norm
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size(); // 12288

        // ── 2+3. QKVZ projection (+ deinterleave if needed) ──
        // For sequential_qkvz (Qwen3.5): write directly to deinterleaved buffer.
        // For interleaved (80B): write to qkvz_out, then deinterleave per token.
        let deinterleaved = ctx.buffers.ssm_deinterleaved(); // [K, 12288] BF16
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        // Native-FP8 build (e.g. Qwen3.6-35B-A3B-FP8): the dense and NVFP4
        // QKVZ slots are NULL — the block-scaled FP8 weight (`qkvz_fp8w`) is
        // the ONLY live copy. The K=2/K=3 MTP-verify batched pass must
        // dispatch through it; falling to `dense_gemv` below dereferences the
        // NULL slot (CUDA_ERROR_ILLEGAL_ADDRESS on the first graphed K=2
        // verify — 2026-07-02 flagship gate). Mirrors the M<=4 dispatch in
        // trait_decode_multi_seq/ssm_batched.rs: one weight pass via
        // `w8a16_gemv_batch4`, per-token `w8a16_gemv` when it isn't linked.
        if let Some(ref g) = self.exl3_gdn {
            // Native EXL3 (ATLAS_EXL3_NATIVE_DENSE=1): the in_proj pair
            // writes all K rows of the sequential [Q|K|V|Z] arena (row stride
            // qkvz_size, Z block at +conv_dim) through the strided egress —
            // the layout the conv/GDN sites below already consume. The packed
            // pair is the ONLY live QKVZ weight on this layer (install-time
            // check), so no arm below is shadowed. Never under graph capture
            // (exl3_graph_veto; the funnel refuses loudly).
            //
            // ROW-EXACT: `exl3_gemv` picks its kernel INSTANCE from the row
            // count — `exl3_gemv_k{K}_cb{CB}_m0_*` at m == 1 and the `_m1_`
            // instance at 2..=8 (ops/exl3_matmul.rs, `let mmode = if m == 1
            // { 0 } else { 1 }`). The two are different compiled bodies, so a
            // verify row processed inside a K-row launch is NOT the bits the
            // M=1 decode GEMV produced for it. When the pass declares
            // `gdn_exact_replay` the funnel is called once per row at m = 1
            // instead: the identical call `ssm_forward` makes
            // (`self.exl3_in_proj(g, ctx, normed, deinterleaved, 1, stream)`),
            // at this row's bases. Costs K weight passes instead of one over
            // the packed pair.
            if row_exact {
                for t in 0..num_tokens {
                    self.exl3_in_proj(
                        g,
                        ctx,
                        normed.offset(t * h * bf16),
                        proj_dst.offset(t * qkvz_size * bf16),
                        1,
                        stream,
                    )?;
                }
            } else {
                self.exl3_in_proj(g, ctx, normed, proj_dst, num_tokens, stream)?;
            }
        } else if let Some(ref q2) = self.qkvz_q2 {
            // Tier-1c keep-packed Q2_0: per-token 2-bit fused-qkvz GEMV. Bonsai
            // (dense qwen35) has no MTP, so this batched path is only reached
            // under multi-token verify — a per-token loop is bit-identical to
            // the M=1 decode GEMV and needs no batched packed kernel.
            for t in 0..num_tokens {
                ops::q2_0_gemv_vec(
                    ctx.gpu,
                    self.q2_0_gemv_k,
                    normed.offset(t * h * bf16),
                    q2,
                    proj_dst.offset(t * qkvz_size * bf16),
                    stream,
                )?;
            }
        // 2..=4: the K=4 verify (num_drafts=3) hits this same NULL-slot
        // hazard — on native-FP8-GDN checkpoints (e.g. nvidia/Qwen3.6-27B-
        // NVFP4, whose GDN layers ship FP8) `in_proj_qkvz`/`qkvz_nvfp4*` are
        // NULL and `qkvz_fp8w` is the only live weight. The old `== 2 || == 3`
        // guard let num_tokens=4 fall through to `dense_gemm` on the NULL
        // dense slot → CUDA_ERROR_ILLEGAL_ADDRESS on the first K=4 verify
        // (localized via ATLAS_K4_DIAG, 2026-07-18). `w8a16_gemv_batch4`
        // is built for M<=4 (see w8a16_gemv_batch4.cu), so widening the
        // guard is sufficient; the per-token `w8a16_gemv` fallback already
        // loops over num_tokens.
        } else if (2..=4).contains(&num_tokens)
            && let Some(ref fp8) = self.qkvz_fp8w
        {
            if self.w8a16_gemv_batch4_k.0 != 0 {
                ops::w8a16_gemv_batch4(
                    ctx.gpu,
                    self.w8a16_gemv_batch4_k,
                    normed,
                    fp8.weight,
                    fp8.row_scale,
                    proj_dst,
                    num_tokens as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                for t in 0..num_tokens {
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        normed.offset(t * h * bf16),
                        fp8.weight,
                        fp8.row_scale,
                        proj_dst.offset(t * qkvz_size * bf16),
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
            }
        } else if (5..=8).contains(&num_tokens)
            && self.w4a16_batchm.kernel(num_tokens as u32).0 != 0
            && let Some(ref nvfp4) = self.qkvz_nvfp4
        {
            // Chain-verify K=5..8: keep the NVFP4 QKVZ on the weight-streaming
            // batched GEMV (the narrowest tier covering these rows) instead of
            // falling through to the tile GEMMs below (the M>4 projection
            // cliff). FP8 checkpoints fall through unchanged (fp8_gemm arm
            // below).
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                self.w4a16_batchm.kernel(num_tokens as u32),
                normed,
                nvfp4,
                proj_dst,
                num_tokens as u32,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if num_tokens > 4
            && (self.w8a16_gemm_pipelined_k.0 != 0 || self.w8a16_gemm_k.0 != 0)
            && let Some(ref fp8) = self.qkvz_fp8w
        {
            // Batched MTP verify at R = Σ ks > 4 on native-FP8-GDN checkpoints
            // (e.g. nvidia/Qwen3.6-27B-NVFP4): `qkvz_fp8w` is the ONLY live
            // QKVZ weight — the dense, NVFP4 and single-scale-FP8 slots are
            // all NULL/None (qwen35_dense.rs native-FP8 GDN arm) — and the
            // fp8w arm above stops at 4 because `w8a16_gemv_batch4` is an
            // M<=4 kernel. Without this arm the dispatch fell through to
            // `dense_gemm` on the NULL dense slot: CUDA_ERROR_ILLEGAL_ADDRESS
            // on the FIRST n>=2 batched verify (ks=[4,3] ⇒ R=7), a sticky
            // context loss that 503s the whole serve. Third instance of this
            // NULL-slot dispatch-gap class (see the 2..=4 arm's history);
            // route through the SAME block-scaled W8A16 GEMM the SSM prefill
            // path uses (`trait_prefill_proj.rs` — pipelined twin preferred,
            // bit-identical to `w8a16_gemm`).
            if self.w8a16_gemm_pipelined_k.0 != 0 {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    normed,
                    fp8.weight,
                    fp8.row_scale,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed,
                    fp8.weight,
                    fp8.row_scale,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if num_tokens == 4 {
            if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                ops::w4a16_gemv_batchm(
                    ctx.gpu,
                    self.w4a16_batchm.kernel(num_tokens as u32),
                    normed,
                    nvfp4,
                    proj_dst,
                    num_tokens as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else if let Some(ref fp8w) = self.qkvz_fp8w {
                ops::w8a16_gemv_batch4(
                    ctx.gpu,
                    self.w8a16_gemv_batch4_k,
                    normed,
                    fp8w.weight,
                    fp8w.row_scale,
                    proj_dst,
                    num_tokens as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                for t in 0..4u32 {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed.offset(t as usize * h * bf16),
                        &self.ssm.in_proj_qkvz,
                        proj_dst.offset(t as usize * qkvz_size * bf16),
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
            }
        } else if num_tokens == 3 {
            if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                ops::w4a16_gemv_batch3(
                    ctx.gpu,
                    self.w4a16_gemv_batch3_k,
                    normed,
                    nvfp4,
                    proj_dst,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                for t in 0..3u32 {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed.offset(t as usize * h * bf16),
                        &self.ssm.in_proj_qkvz,
                        proj_dst.offset(t as usize * qkvz_size * bf16),
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
            }
        } else if num_tokens == 2 {
            if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                ops::w4a16_gemv_batch2(
                    ctx.gpu,
                    self.w4a16_gemv_batch2_k,
                    normed,
                    nvfp4,
                    proj_dst,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                // Batched M=2: one pass over in_proj_qkvz for both verify
                // tokens instead of two M=1 reads of the full projection
                // weight. Bit-identical to the two dense_gemv calls it
                // replaces (same per-row accumulation order); the dominant
                // per-verify-step weight-bandwidth term across the 48 GDN
                // layers on FP8 checkpoints (in_proj dequanted to BF16).
                ops::dense_gemv_batch2(
                    ctx.gpu,
                    self.dense_gemv_batch2_k,
                    normed,
                    &self.ssm.in_proj_qkvz,
                    proj_dst,
                    qkvz_size as u32,
                    h as u32,
                    qkvz_size as u32,
                    stream,
                )?;
            }
        } else if qkvz_verify_nvfp4_wins(
            num_tokens,
            self.qkvz_fp8.is_some(),
            self.qkvz_nvfp4_t.is_some(),
            self.deep_k_gemm(h as u32).0 != 0,
            qkvz_nvfp4_decode_off(),
        ) && let Some(ref nvfp4_t) = self.qkvz_nvfp4_t
        {
            // M > 8 (batched K-token verify, R = Σ ks rows; DFlash wide
            // verify): the single-scale FP8 PREFILL arm below (`qkvz_fp8`,
            // built by `bf16_to_fp8` at load) reads 1 byte per element where
            // the NVFP4 twin reads 0.5625 — on THIS matrix
            // (N=16384, K=5120) across the 48 GDN layers that is
            // +1.762 GB/step, +77.8%, on a path already at the LPDDR5X wall.
            // nsys, binary b508679e4, unsloth/Qwen3.8-27B-NVFP4 at C=8:
            // `fp8_fp8_gemm_ldmab` 42.58 ms/step = 26.3% of the entire step
            // over 48 calls, achieving 94.6 GB/s against the 203 GB/s the
            // shape admits — its 128-row M tile is ~75% padding at the M=32
            // a C=8 step produces. This is ~90% of the measured 1.56x
            // GEMM-time regression from C=1 to C=8.
            //
            // IDENTICAL defect and identical fix to the out_proj arm below;
            // that half was found first only because attention `o_proj` was
            // already on the tile GEMM and fingered it. Route to the NVFP4
            // transposed twin through `ms_proj_gemm` — the SAME call the
            // multi-seq decode QKVZ makes on this same weight
            // (`trait_decode_multi_seq/ssm_batched.rs`), which also picks the
            // 128-row M tile once N fills the machine and the batch is wide
            // enough that the 64-row tile would re-stream the weight.
            //
            // Numerics: `qkvz_fp8` and `qkvz_nvfp4_t` are BOTH derived from
            // the one BF16 `qkvz_dense` at load (`qwen35_dense.rs`:
            // `bf16_to_fp8` and `quantize_to_nvfp4`), so this changes which
            // lossy copy is read, not what the weight means. NVFP4 is the
            // copy every OTHER decode arm here already reads (batch2/3/4/8
            // above, multi-seq decode, single-token decode), so this makes
            // batched verify agree with the decode it is verifying — the FP8
            // arm was the odd one out. BF16 activations (vs the FP8 arm's
            // e4m3 downcast) against a 4-bit weight: not byte-identical.
            // Kill switch ATLAS_NO_QKVZ_NVFP4_DECODE (PRESENCE check).
            self.ms_proj_gemm(
                ctx.gpu,
                normed,
                nvfp4_t,
                proj_dst,
                num_tokens as u32,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if let Some(fp8) = self.qkvz_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                normed,
                fp8,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if let Some(ref nvfp4_t) = self.qkvz_nvfp4_t {
            // Prefer the 8-warp pipelined v2 (fast even at small M — the wide
            // DFlash verify runs M=17, where plain m128 padding loses to n128,
            // but v2 wins; it's the same kernel the dense FFN prefill uses).
            // Else m128 for large-M prefill; else n128.
            if self.w4a16_gemm_t_m128_v2_k.0 != 0 {
                ops::w4a16_gemm_n128_m128_v2(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_v2_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else if k > 128 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                nvfp4,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else {
            // Fail fast: launching `dense_gemm` on a NULL weight is a device
            // ILLEGAL_ADDRESS that destroys the CUDA context for every live
            // request (sticky 700). A checkpoint whose QKVZ ships only in a
            // form this dispatch has no arm for must error per-request, not
            // kill the serve.
            anyhow::ensure!(
                !self.ssm.in_proj_qkvz.weight.is_null(),
                "batched GDN QKVZ dispatch: no usable weight for num_tokens={num_tokens} \
                 (dense slot NULL; fp8w={}, nvfp4={}, gemm kernels pipelined/base: {:#x}/{:#x})",
                self.qkvz_fp8w.is_some(),
                self.qkvz_nvfp4.is_some(),
                self.w8a16_gemm_pipelined_k.0,
                self.w8a16_gemm_k.0,
            );
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &self.ssm.in_proj_qkvz,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        }
        if !self.sequential_qkvz {
            for t in 0..(num_tokens as u32) {
                let src = proj_dst.offset(t as usize * qkvz_size * bf16);
                let dst = deinterleaved.offset(t as usize * qkvz_size * bf16);
                ops::deinterleave_qkvz(
                    ctx.gpu,
                    self.deinterleave_k,
                    src,
                    dst,
                    1,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )?;
            }
        }

        k4_diag_checkpoint(ctx, "2+3:qkvz_proj+deinterleave", stream)?;

        // ── 4. BA projection + GDN gates per token ──
        // BA output: ssm_ba buffer; gates: ssm_gates buffer [K, nv*2] FP32
        // Layout per token: [gate(nv), beta(nv)] → stride = 2*nv FP32 elements.
        // Must match gdn_decode_chunk2's gb_stride parameter.
        let gates_buf = ctx.buffers.ssm_gates(); // [K, gate(nv) + beta(nv)] FP32
        let gate_beta_stride = nv * 2 * fp32; // bytes per token in gates buffer
        let ba_size = ctx.config.ssm_ba_size(); // 64
        if row_exact {
            // ROW-EXACT: `dense_gemv_ba_gates` — the FUSED single-row kernel
            // `ssm_forward` calls — once per row, at this row's bases. The
            // token-parallel `dense_gemm_ba_gates_prefill` below is
            // arithmetically line-for-line the same reduction, but the
            // row-exact arm exists to keep the WHOLE block on the decode
            // kernels rather than to argue each one individually.
            for t in 0..num_tokens {
                ops::dense_gemv_ba_gates(
                    ctx.gpu,
                    self.ba_gates_k,
                    normed.offset(t * h * bf16),
                    &self.ssm.in_proj_ba,
                    self.ssm.a_log.weight,
                    self.ssm.dt_bias.weight,
                    gates_buf.offset(t * gate_beta_stride),
                    gates_buf.offset(t * gate_beta_stride + nv * fp32),
                    ba_size as u32,
                    h as u32,
                    vpg as u32,
                    stream,
                )?;
            }
        } else if batched_ba_gates_enabled() {
            // ONE fused launch for all rows instead of `num_tokens` × (dense
            // GEMV + compute_gdn_gates). The per-token loop re-read the whole
            // [64, hidden] BA weight once PER ROW PER LAYER — 655 KB × 32 rows ×
            // 48 GDN layers = ~1.0 GB/step of redundant traffic at R=32, plus
            // 2R launches per layer (3072/step) that the verify CUDA graph then
            // has to carry as nodes.
            //
            // `dense_gemm_ba_gates_prefill` is the token-parallel twin of the
            // decode-path `dense_gemv_ba_gates` (identical uint4 K-reduction,
            // identical warp+smem tree, identical gate/beta transforms) and is
            // ALREADY the shipped form on both the prefill path and the
            // multi-seq batched-recurrent decode path. It writes the same
            // [gate(nv) | beta(nv)] interleaved row layout at `gate_stride`
            // FP32 elements per row that `gates_buf` expects.
            //
            // NOT bit-identical to the loop it replaces: fusing removes the
            // BF16 round-trip through the `ssm_ba` staging buffer, so the gate
            // and beta transforms see the FP32 projection result directly.
            // Kill switch below restores the split form.
            ops::dense_gemm_ba_gates_prefill(
                ctx.gpu,
                self.ba_gates_prefill_k,
                normed,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gates_buf,
                num_tokens as u32,
                ba_size as u32,
                h as u32,
                h as u32,
                (nv * 2) as u32,
                nv as u32,
                vpg as u32,
                stream,
            )?;
        } else {
            for t in 0..(num_tokens as u32) {
                let normed_t = normed.offset(t as usize * h * bf16);
                let ba_out = ctx.buffers.ssm_ba().offset(t as usize * ba_size * bf16);
                // Dense GEMV for BA projection (small: 64 outputs)
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed_t,
                    &self.ssm.in_proj_ba,
                    ba_out,
                    ba_size as u32,
                    h as u32,
                    stream,
                )?;
                // Apply gate transforms
                let gate_t = gates_buf.offset(t as usize * gate_beta_stride);
                let beta_t = gates_buf.offset(t as usize * gate_beta_stride + nv * fp32);
                ops::compute_gdn_gates(
                    ctx.gpu,
                    self.compute_gdn_gates_k,
                    ba_out,
                    self.ssm.a_log.weight,
                    self.ssm.dt_bias.weight,
                    gate_t,
                    beta_t,
                    1,
                    nv as u32,
                    nk as u32,
                    vpg as u32,
                    ba_size as u32,
                    stream,
                )?;
            }
        }

        k4_diag_checkpoint(ctx, "4:ba_proj+gates", stream)?;

        // ── 5-7. Conv1d + L2 norm + GDN per token (with intermediate checkpoints) ──
        // Reuse ssm_qkvz buffer for conv output (safe: deinterleave is done)
        let conv_out_buf = ctx.buffers.ssm_qkvz();
        let gdn_out_buf = ctx.buffers.attn_output();
        // POOL PITCH, not the FP32 width: every consumer of
        // `ConvGdnArgs::h_bytes` uses it to stride or byte-copy pool h
        // regions, all of which narrow under the f16-SIZED pool. Identical to
        // `h_state_bytes` on an FP32-sized pool, so this is a no-op there.
        let h_bytes = self.h_slot_stride_bytes();
        let conv_bytes = self.conv_state_bytes;

        match gdn {
            GdnStates::Single(state) => {
                let ssm_state = state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
                // Intermediates are pre-allocated from the pool (fixed GPU addresses for
                // CUDA graph stability). Verify they exist BEFORE we index into them — a
                // bare `debug_assert!` is a no-op in release and produces an opaque
                // out-of-bounds panic instead of an actionable error (see #bugs
                // m0t0chan EP=2 2026-04-05). Most-common cause: EP=2 worker started
                // without `--speculative --mtp-quantization` to mirror the head.
                if ssm_state.h_state_intermediates.len() + 1 < num_tokens
                    || ssm_state.conv_state_intermediates.len() < num_tokens
                {
                    anyhow::bail!(
                        "SSM MTP intermediate buffers not allocated (need K-1 h + K conv; \
                         h_state_intermediates.len()={}, \
                         conv_state_intermediates.len()={}, num_tokens={}). \
                         If this is an EP=2 worker, the head node is sending MTP verify commands \
                         but the worker was started without `--speculative` (and matching \
                         `--mtp-quantization`/`--num-drafts`). Add those flags to the worker invocation.",
                        ssm_state.h_state_intermediates.len(),
                        ssm_state.conv_state_intermediates.len(),
                        num_tokens,
                    );
                }

                let args = super::trait_decode_batched_conv_gdn::ConvGdnArgs {
                    num_tokens,
                    deinterleaved,
                    gates_buf,
                    conv_out_buf,
                    gdn_out_buf,
                    normed_out: conv_out_buf, // row0 == 0: bases coincide
                    h_bytes,
                    conv_bytes,
                    qkvz_size,
                    conv_dim,
                    key_dim,
                    value_dim,
                    d_conv,
                    qk_ch,
                    nk,
                    nv,
                    kd,
                    vd,
                    bf16,
                    fp32,
                    stream,
                };
                self.decode_batched_conv_gdn(ssm_state, ctx, &args)?;
            }
            GdnStates::Multi {
                states,
                ks,
                wy_tables,
            } => {
                // Batched MTP verify. Fast path: cross-sequence batched
                // conv+WY — ONE `gdn_verify_fused_conv_kn_batched` launch +
                // ONE table-form `gdn_decode_wy4` launch for the whole batch
                // (trait_decode_batched_conv_gdn_multi.rs; preconditions
                // checked on the actual state pointers, kill switch
                // ATLAS_NO_VERIFY_GDN_BATCH). Fallback: the SAME per-sequence
                // conv+GDN body, one call per sequence with row-offset buffer
                // bases. Strides match decode_batched_conv_gdn's consumers
                // exactly: deinterleaved rows at qkvz_size*bf16, conv_out at
                // conv_dim*bf16, gates at nv*2*fp32, gdn_out at value_dim*bf16
                // per token.
                anyhow::ensure!(
                    !states.is_empty()
                        && ks.len() == states.len()
                        && num_tokens == ks.iter().sum::<usize>(),
                    "decode_batched_inner Multi: num_tokens {} != Σ ks {:?} (n {})",
                    num_tokens,
                    ks,
                    states.len(),
                );
                // Row offset of each sequence (ragged; `i*k` when uniform).
                let mut off: Vec<usize> = Vec::with_capacity(states.len());
                let mut acc = 0usize;
                for &k in ks.iter() {
                    off.push(acc);
                    acc += k;
                }
                // Sequences are ordered deepest-first by the scheduler, so
                // equal depths form CONTIGUOUS runs. One batched conv+WY
                // attempt per run keeps the two-launch fast path alive under
                // ragged depths (uniform ⇒ exactly one run = today's call).
                let mut g0 = 0usize;
                while g0 < states.len() {
                    let kk = ks[g0];
                    let mut g1 = g0 + 1;
                    while g1 < states.len() && ks[g1] == kk {
                        g1 += 1;
                    }
                    let row0 = off[g0];
                    let run_args = super::trait_decode_batched_conv_gdn::ConvGdnArgs {
                        num_tokens: kk,
                        deinterleaved: deinterleaved.offset(row0 * qkvz_size * bf16),
                        gates_buf: gates_buf.offset(row0 * nv * 2 * fp32),
                        conv_out_buf: conv_out_buf.offset(row0 * conv_dim * bf16),
                        gdn_out_buf: gdn_out_buf.offset(row0 * value_dim * bf16),
                        // Normed rows stride value_dim from ROW 0 of the
                        // phase-8/9 buffer — a conv_dim-scaled offset here
                        // would land the exact arm's output between rows.
                        normed_out: conv_out_buf.offset(row0 * value_dim * bf16),
                        h_bytes,
                        conv_bytes,
                        qkvz_size,
                        conv_dim,
                        key_dim,
                        value_dim,
                        d_conv,
                        qk_ch,
                        nk,
                        nv,
                        kd,
                        vd,
                        bf16,
                        fp32,
                        stream,
                    };
                    // Table entries are per sequence in batch order, so a run
                    // slices them by its own start index (8 B per u64 entry).
                    let run_tables = if wy_tables.is_null() {
                        wy_tables
                    } else {
                        wy_tables.offset(g0 * 8)
                    };
                    let batched = self.decode_batched_conv_gdn_multi(
                        &mut states[g0..g1],
                        run_tables,
                        ctx,
                        &run_args,
                    )?;
                    if !batched {
                        for i in g0..g1 {
                            let ssm_state = states[i]
                                .as_any_mut()
                                .downcast_mut::<SsmLayerState>()
                                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
                            if ssm_state.h_state_intermediates.len() + 1 < kk
                                || ssm_state.conv_state_intermediates.len() < kk
                            {
                                anyhow::bail!(
                                    "SSM MTP intermediate buffers not allocated for batched \
                                 verify (seq {i}: h={}, conv={}, k={kk})",
                                    ssm_state.h_state_intermediates.len(),
                                    ssm_state.conv_state_intermediates.len(),
                                );
                            }
                            let r = off[i];
                            let args_i = super::trait_decode_batched_conv_gdn::ConvGdnArgs {
                                num_tokens: kk,
                                deinterleaved: deinterleaved.offset(r * qkvz_size * bf16),
                                gates_buf: gates_buf.offset(r * nv * 2 * fp32),
                                conv_out_buf: conv_out_buf.offset(r * conv_dim * bf16),
                                gdn_out_buf: gdn_out_buf.offset(r * value_dim * bf16),
                                normed_out: conv_out_buf.offset(r * value_dim * bf16),
                                h_bytes,
                                conv_bytes,
                                qkvz_size,
                                conv_dim,
                                key_dim,
                                value_dim,
                                d_conv,
                                qk_ch,
                                nk,
                                nv,
                                kd,
                                vd,
                                bf16,
                                fp32,
                                stream,
                            };
                            self.decode_batched_conv_gdn(ssm_state, ctx, &args_i)?;
                        }
                    }
                    g0 = g1;
                }
            }
        }

        k4_diag_checkpoint(ctx, "5-7:conv1d+l2norm+gdn_wy", stream)?;

        // ── 8. Gated RMS norm per token (Z gate at [Q|K|V] offset) ──
        let normed_out_buf = conv_out_buf;
        let z_offset = key_dim * 2 + value_dim; // == conv_dim
        if super::verify_row_exact_leg(ctx.gdn_exact_replay, super::RowExactLeg::ConvGdn) {
            // Issue #435 exact arm (phase 5-7 above): the norm is already
            // applied inside the per-token chain — the SAME fused/unfused arm
            // sequential decode uses — and the normed rows are already in
            // `normed_out_buf` at value_dim BF16 stride. Running any norm
            // here would re-normalize final output with stale gdn_out data.
        } else if num_tokens == 2 && self.fused_verify_k2_enabled() {
            // STAGE 1: single-launch gated-RMS-norm for BOTH positions (cos==1.0).
            ops::gdn_verify_fused_norm_k2(
                ctx.gpu,
                self.gdn_verify_fused_norm_k2_k,
                gdn_out_buf,
                deinterleaved,
                &self.ssm.norm,
                normed_out_buf,
                nv as u32,
                vd as u32,
                eps,
                qkvz_size as u32, // deint position stride (BF16 elems)
                z_offset as u32,  // Z offset within a position
                value_dim as u32, // gdn/out position stride
                stream,
            )?;
        } else if batched_norm_enabled() {
            // ONE launch for all (head, row) pairs instead of `num_tokens` of
            // them. `gated_rms_norm_prefill` is `gated_rms_norm` with the token
            // index moved to blockIdx.y and the two row strides passed
            // explicitly — the reduction, the register cache and the quad loop
            // are line-for-line the same, so this is BIT-IDENTICAL, not an
            // approximation. The fused K=2 arm above already covers
            // `num_tokens == 2`; the batched verify (R = n*k, 8..32 rows) never
            // reached it and was paying R launches per GDN layer, i.e. 1536
            // launches/step at R=32 across 48 layers.
            ops::gated_rms_norm_prefill(
                ctx.gpu,
                self.gated_rms_norm_prefill_k,
                gdn_out_buf,
                deinterleaved.offset(z_offset * bf16),
                &self.ssm.norm,
                normed_out_buf,
                nv as u32,
                vd as u32,
                eps,
                num_tokens as u32,
                value_dim as u32,
                qkvz_size as u32,
                stream,
            )?;
        } else {
            for t in 0..(num_tokens as u32) {
                let gdn_t = gdn_out_buf.offset(t as usize * value_dim * bf16);
                let z_t = deinterleaved.offset(t as usize * qkvz_size * bf16 + z_offset * bf16);
                let normed_t = normed_out_buf.offset(t as usize * value_dim * bf16);
                ops::gated_rms_norm(
                    ctx.gpu,
                    self.gated_rms_norm_k,
                    gdn_t,
                    z_t,
                    &self.ssm.norm,
                    normed_t,
                    nv as u32,
                    vd as u32,
                    vd as u32,
                    eps,
                    vd as u32,
                    stream,
                )?;
            }
        }

        k4_diag_checkpoint(ctx, "8:gated_rms_norm", stream)?;

        // ── 9. Output projection → [K, H] ──
        let out_proj_buf = ctx.buffers.moe_output(); // [K, H] BF16
        if let Some(ref g) = self.exl3_gdn {
            // Native EXL3 (ATLAS_EXL3_NATIVE_DENSE=1): the packed trellis is
            // the ONLY live out_proj weight (all other slots null, enforced
            // at install). C is the contiguous [K, h] block; the GEMV tier
            // covers K <= 8, the GEMM tier the wider chain verifies.
            //
            // ROW-EXACT: same m-instance split as the QKVZ funnel above —
            // one m = 1 call per row, which is the call `ssm_forward` makes.
            if row_exact {
                for t in 0..num_tokens {
                    self.exl3_out_proj(
                        g,
                        ctx,
                        normed_out_buf.offset(t * value_dim * bf16),
                        out_proj_buf.offset(t * h * bf16),
                        1,
                        stream,
                    )?;
                }
            } else {
                self.exl3_out_proj(g, ctx, normed_out_buf, out_proj_buf, num_tokens, stream)?;
            }
        } else if let Some(ref dense_out) = self.out_proj_dense {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed_out_buf,
                dense_out,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if (2..=4).contains(&num_tokens)
            && let Some(ref fp8) = self.out_proj_fp8w
        {
            // 2..=4: same K=4 NULL-slot hazard as the QKVZ dispatch above —
            // on native-FP8-GDN checkpoints `ssm.out_proj` is NULL and
            // `out_proj_fp8w` is the only live weight; the old guard sent
            // num_tokens=4 to `w4a16_gemm` on the NULL slot.
            // Native-FP8 build: `ssm.out_proj` is a NULL QuantizedWeight —
            // the block-scaled FP8 copy (`out_proj_fp8w`) is the only live
            // weight. Same NULL-deref hazard as the QKVZ dispatch above.
            if self.w8a16_gemv_batch4_k.0 != 0 {
                ops::w8a16_gemv_batch4(
                    ctx.gpu,
                    self.w8a16_gemv_batch4_k,
                    normed_out_buf,
                    fp8.weight,
                    fp8.row_scale,
                    out_proj_buf,
                    num_tokens as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                for t in 0..num_tokens {
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        normed_out_buf.offset(t * value_dim * bf16),
                        fp8.weight,
                        fp8.row_scale,
                        out_proj_buf.offset(t * h * bf16),
                        h as u32,
                        value_dim as u32,
                        stream,
                    )?;
                }
            }
        } else if (4..=8).contains(&num_tokens)
            && !self.ssm.out_proj.weight.is_null()
            && self.w4a16_batchm_kernel(num_tokens).0 != 0
        {
            // NVFP4 out_proj at M=4..8 (K=4 verify + K=5..8 chain verify):
            // previously fell through to the w4a16 tile GEMMs below (M>3
            // cliff — there was no ==4 arm at all on the NVFP4 side); the
            // batchm GEMV streams the weight once for all rows.
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                self.w4a16_batchm_kernel(num_tokens),
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                num_tokens as u32,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if num_tokens > 4
            && (self.w8a16_gemm_pipelined_k.0 != 0 || self.w8a16_gemm_k.0 != 0)
            && let Some(ref fp8) = self.out_proj_fp8w
        {
            // Batched MTP verify at R = Σ ks > 4 on native-FP8-GDN
            // checkpoints: `out_proj_fp8w` is the ONLY live out_proj weight
            // (dense/NVFP4 slots NULL — qwen35_dense.rs native-FP8 GDN arm)
            // and the fp8w arm above stops at 4. Without this arm the
            // dispatch fell through to `w4a16_gemm` on the null
            // `ssm.out_proj` — the out_proj half of the same
            // CUDA_ERROR_ILLEGAL_ADDRESS the QKVZ dispatch above hits first.
            // Same block-scaled W8A16 GEMM pair as the prefill path.
            if self.w8a16_gemm_pipelined_k.0 != 0 {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    normed_out_buf,
                    fp8.weight,
                    fp8.row_scale,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed_out_buf,
                    fp8.weight,
                    fp8.row_scale,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        } else if num_tokens > VERIFY_TGEMM_MIN_TOKENS
            && let Some(ref nvfp4_t) = self.out_proj_nvfp4_t
            && {
                // Kill switch, PRESENCE check per house convention (`=0` is NOT
                // off), cached once.
                static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                !*OFF.get_or_init(|| std::env::var("ATLAS_NO_VERIFY_OUTPROJ_TGEMM").is_ok())
            }
        {
            // M > 8 (batched K=4 verify, R = n*4 rows; DFlash wide verify):
            // the pre-dequanted FP8 prefill arm below reads 2x the weight
            // bytes of NVFP4 — bandwidth-bound at this M, measured 379 us
            // vs 182 us per call at M=16 N=5120 (fp8_fp8_gemm_ldmab vs
            // w4a16_gemm_t_k64_p3), ~9.5 ms/step across 48 GDN layers at
            // C=4. Route to the NVFP4 transposed-twin tile GEMM — the SAME
            // kernel+handle the multi-seq decode out_proj uses
            // (`deep_k_gemm`). BF16 activations (vs the FP8 arm's e4m3
            // downcast) — equal-or-better numerics, not byte-identical.
            // Kill switch ATLAS_NO_VERIFY_OUTPROJ_TGEMM (PRESENCE check).
            //
            // 2026-08-17: the call now goes through `ms_proj_gemm` rather than
            // `deep_k_gemm` directly. `ms_proj_gemm` IS `deep_k_gemm` plus the
            // narrow-N twin test this arm was skipping: at THIS shape
            // (N=h=5120, K=value_dim=6144) the wide tile launches
            // ceil(5120/128)=40 CTAs on a 48-SM device, which is precisely the
            // `k64_n64_wins` case — bit-identical output, measured 1.42x on
            // the replay bench (see `layers::K64_N64_MAX_WIDE_CTAS`) and
            // 100 GB/s -> 151 GB/s in the C=8 profile, ~2.9 ms/step. Attention
            // `o_proj` already applies the same test to the same shape
            // (`qwen3_attention/trait_impl/multi_seq/qkv.rs`), and the
            // multi-seq decode out_proj already makes this exact
            // `ms_proj_gemm` call on this exact weight — so this is one call
            // site rejoining the other three, not a new rule.
            self.ms_proj_gemm(
                ctx.gpu,
                normed_out_buf,
                nvfp4_t,
                out_proj_buf,
                num_tokens as u32,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if num_tokens == 3 {
            ops::w4a16_gemv_batch3(
                ctx.gpu,
                self.w4a16_gemv_batch3_k,
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if num_tokens == 2 {
            ops::w4a16_gemv_batch2(
                ctx.gpu,
                self.w4a16_gemv_batch2_k,
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if let Some(fp8) = self.out_proj_fp8 {
            if k > 128 {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    normed_out_buf,
                    fp8,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    normed_out_buf,
                    fp8,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4_t) = self.out_proj_nvfp4_t {
            if self.w4a16_gemm_t_m128_v2_k.0 != 0 {
                // 8-warp pipelined v2 (fast at M=17 wide verify; FFN's kernel).
                ops::w4a16_gemm_n128_m128_v2(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_v2_k,
                    normed_out_buf,
                    nvfp4_t,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed_out_buf,
                    nvfp4_t,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        } else {
            // Fail fast: a null out_proj here is the same sticky-700 context
            // killer as the QKVZ dense arm above — refuse per-request instead.
            anyhow::ensure!(
                !self.ssm.out_proj.weight.is_null(),
                "batched GDN out_proj dispatch: no usable weight for num_tokens={num_tokens} \
                 (quant slot NULL; fp8w={}, dense={}, gemm kernels pipelined/base: {:#x}/{:#x})",
                self.out_proj_fp8w.is_some(),
                self.out_proj_dense.is_some(),
                self.w8a16_gemm_pipelined_k.0,
                self.w8a16_gemm_k.0,
            );
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        }

        // GDN HeadParallel: reduce the row-parallel partial out_proj across TP
        // ranks (num_tokens × h BF16) before the residual add. No-op at tp=1.
        self.ssm_tp_all_reduce(out_proj_buf, normed_out_buf, num_tokens, ctx, stream)?;

        k4_diag_checkpoint(ctx, "9:out_proj", stream)?;

        Ok(out_proj_buf)
    }
}
