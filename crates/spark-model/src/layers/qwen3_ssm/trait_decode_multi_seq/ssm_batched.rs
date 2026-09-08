// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_multi_seq — batched-projection SSM mixer.

use super::super::*;

/// Tensor-core mixer projections for wide decode batches. **ON by default at
/// n>=9**; `ATLAS_SSM_TC_PROJ=&lt;n&gt;` moves the threshold, `=0` disables.
///
/// WHY: the mixer's qkvz/out_proj run through `w4a16_gemv_batchm`, a SCALAR-FMA
/// kernel. It reads the weights once for all n rows, but its arithmetic scales
/// with n, so it stops being weight-bound at about M=3 and measures 2.3-3.0
/// TFLOP/s at M=16. `w4a16_gemm_t` is an M64/N128 FP8-MMA tile GEMM on the
/// SAME weights and is roughly flat in M. This is the same class of fix as the
/// wide dense-FFN arm (+30% at C=16) applied to the other half of the mixer.
///
/// Costs nothing to try: `qkvz_nvfp4_t` / `out_proj_nvfp4_t` are transposed
/// NVFP4 twins ALREADY built at load (init.rs) and already used by the SSM
/// PREFILL path, so there is no repack, no new buffer and no extra VRAM.
///
/// MEASURED 2 reps/cell, coherence identical:
///   GEMV (old):   C=8 57.8 | C=16 79.6
///   TC n>=9:      C=8 57.6 | C=16 **86.9  (+9.2%)**
///   TC n>=5:      C=8 54.9 (**regresses**) | C=16 86.4
/// So the mixer's crossover is 9 — NOT the FFN's 5. Different shapes, different
/// crossover; do not assume one transfers to the other.
///
/// ACCURACY DEBT: `w4a16_gemm_t` dequants the FP4 weight to E4M3 and converts
/// the BF16 activations to E4M3 (W4A8) where the GEMV path is W4A16, so this
/// CAN move a greedy token. It is the production SSM PREFILL path for these
/// exact two weights, and the coherence smoke is identical, but a BFCL gate is
/// owed before this merges. Read ONCE — this site runs under graph capture.
fn ssm_tc_proj_min_n() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(
        || match std::env::var("ATLAS_SSM_TC_PROJ").ok().as_deref() {
            None => Some(9),
            Some("0") => None,
            Some("1") => Some(9),
            Some(v) => v.parse::<usize>().ok().filter(|&x| x >= 2),
        },
    )
}

impl Qwen3SsmLayer {
    /// Batched-projection SSM mixer for N concurrent decode sequences.
    ///
    /// Returns `Ok(false)` (caller falls back to the per-seq loop) unless the
    /// layer is in the GB10 Holo serving config: sequential-QKVZ dense/NVFP4
    /// weights + FP32 conv/GDN recurrent kernels. When eligible, the big QKVZ
    /// and out_proj projections run as a single `[N, ...]` GEMM each (weights
    /// read ONCE, not N times — the dominant bandwidth cost on LPDDR5X), while
    /// the recurrent inner (BA/gates → conv1d → GDN → gated-norm) stays a
    /// per-seq loop using the SAME single-token kernels as `ssm_forward`, so
    /// the recurrence is byte-identical to the proven path. The per-seq states
    /// are read straight from each `SsmLayerState`, so no contiguous-slot
    /// assumption is required.
    #[allow(clippy::too_many_arguments)]
    /// `hc` (#753 item B): input rows arrive pre-mixed in `norm_output`
    /// (hc_pre) and the out_proj rows stay in `moe_output` for the caller's
    /// hc_post — both norm/residual steps skip; `hidden`/`residual` unused.
    pub(super) fn try_decode_multi_seq_ssm_batched<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        n: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        hc: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let use_f32_conv = self.conv1d_l2norm_f32_k.0 != 0;
        let use_f32_gdn = self.gdn_f32_k.0 != 0 && self.gated_rms_norm_f32_k.0 != 0;
        // QKVZ via dense-BF16/cuBLASLt, FP8 w8a16 GEMM, or NVFP4 batchm GEMV
        // (M<=16; tile-GEMM twins `*_nvfp4_t` lift the cap for wide batches
        // when both twins exist and the TC threshold admits n). Any arm
        // amortizes the QKVZ/out_proj weight read across the n seqs;
        // interleaved-QKVZ layouts take the proven per-seq loop.
        let tc_wide_ok = self.qkvz_nvfp4_t.is_some()
            && self.out_proj_nvfp4_t.is_some()
            && ssm_tc_proj_min_n().is_some_and(|min| n >= min);
        // qkvz_nvfp4.is_none() covers TWO builds: the FP8 build (qkvz_fp8w
        // Some -> batched w8a16 GEMM, needs w8a16_gemm_k) and the pure
        // native-BF16 build (both None -> the batched `dense_gemm` fallback
        // below, which uses dense_gemm_k — loaded with an unconditional `?`
        // in init.rs:147, so it is always non-zero here). Gate each on the
        // kernel it actually dispatches so the BF16 build engages the batched
        // fast path instead of silently dropping to the per-seq loop. The FP8
        // sub-case (qkvz_fp8w Some) is byte-identical to the old gate.
        // Native EXL3 (packed in_proj pair, any n) is admitted explicitly.
        let qkvz_ok = self.exl3_gdn.is_some()
            || (self.qkvz_nvfp4.is_none()
                && ((self.qkvz_fp8w.is_some() && self.w8a16_gemm_k.0 != 0)
                    || (self.qkvz_fp8w.is_none() && self.dense_gemm_k.0 != 0)))
            || (self.qkvz_nvfp4.is_some()
                && ((self.w4a16_batchm.has_base() && n <= 16) || tc_wide_ok));
        let out_ok = self.out_proj_fp8w.is_some()
            || self.out_proj_dense.is_some()
            || self.exl3_gdn.is_some()
            || self.qkvz_nvfp4.is_some();
        // Tier-1c keep-packed Q2_0 has no batched packed GEMM; decline so the
        // per-seq fallback (`ssm_forward` → `q2_0_gemv_vec`) handles it.
        if n < 2
            || !self.sequential_qkvz
            || !use_f32_conv
            || !use_f32_gdn
            || !qkvz_ok
            || !out_ok
            || self.qkvz_q2.is_some()
        {
            // Say WHY, once (`ssm_batched_log.rs`).
            if n >= 2 {
                let flags = [use_f32_conv, use_f32_gdn, qkvz_ok, out_ok];
                self.log_batched_proj_declined(n, flags, tc_wide_ok);
            }
            return Ok(false);
        }
        {
            static ON: std::sync::Once = std::sync::Once::new();
            ON.call_once(|| {
                tracing::info!("SSM batched projections ACTIVE — QKVZ/out_proj read once per step");
            });
        }

        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        let eps = ctx.config.rms_norm_eps as f32;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = (key_dim * 2 + value_dim) as u32;
        let qk_channels = (key_dim * 2) as u32;
        let d_conv = ctx.config.linear_conv_kernel_dim as u32;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size() as u32;

        let normed_base = ctx.buffers.norm_output();
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        // normed_out[0..n] (post gated-norm, [N, value_dim] BF16) parks in the
        // QKVZ scratch — free here because QKVZ projects into `deinterleaved`
        // and the FP32 conv path uses `ssm_conv_out_f32`, not `ssm_qkvz`.
        let normed_out_base = ctx.buffers.ssm_qkvz();
        let ssm_out_base = ctx.buffers.moe_output();
        let detail_profile = std::env::var("ATLAS_SSM_DETAIL_PROFILE").ok().as_deref() == Some("1")
            && !ctx.graph_capture;
        let mut detail_parts: Vec<(&'static str, u128)> = Vec::new();
        let mut detail_t0 = if detail_profile {
            ctx.gpu.synchronize(stream).ok();
            Some(std::time::Instant::now())
        } else {
            None
        };
        macro_rules! detail_step {
            ($label:expr) => {
                if let Some(t0) = detail_t0.take() {
                    ctx.gpu.synchronize(stream).ok();
                    detail_parts.push(($label, t0.elapsed().as_micros()));
                    detail_t0 = Some(std::time::Instant::now());
                }
            };
            ($label:expr, final) => {
                if let Some(t0) = detail_t0.take() {
                    ctx.gpu.synchronize(stream).ok();
                    detail_parts.push(($label, t0.elapsed().as_micros()));
                }
            };
        }

        // ── 1. Batched input RMS norm (hc: hc_pre already normed; skip) ──
        if !hc {
            ops::rms_norm_residual(
                ctx.gpu,
                self.rms_norm_residual_k,
                hidden,
                &self.input_norm,
                normed_base,
                residual,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
        }
        detail_step!("input_norm");

        // ── 2. Batched QKVZ projection: ONE [N,h]→[N,qkvz] GEMM (weights ×1) ──
        // FP8 (w8a16) when the decode overlay is installed, else BF16 dense.
        // Prefer the pipelined (cp.async) w8a16 kernel — bit-identical, ~4.6×
        // faster than the base w8a16_gemm, which nsys showed as 44.6% of the
        // C>1 decode step. `.0 == 0` → fall back to the base kernel.
        let w8a16_pipe = self.w8a16_gemm_pipelined_k.0 != 0;
        // Weight-streaming block-scaled GEMV for batched decode: avoids the
        // pipelined kernel's M->128 MMA pad (issue-bound). batch4 (M<=4) for the
        // common path, batch16 (M<=16) for high-concurrency C=8/16. Bit-identical
        // per row to w8a16_gemv. Disable with ATLAS_SSM_GEMV_BATCH4=0.
        let gemv_batch_k = if n <= 4 {
            self.w8a16_gemv_batch4_k
        } else {
            self.w8a16_gemv_batch16_k
        };
        let use_batch4 = gemv_batch_k.0 != 0
            && n <= 16
            && std::env::var("ATLAS_SSM_GEMV_BATCH4").ok().as_deref() != Some("0");
        // FP4 sibling: the narrow w4a16_gemv batch{4..8} family (M<=8), else
        // batch16 (M<=16). Single NVFP4 weight pass for the QKVZ + out_proj
        // GEMVs (amortizes the weight read). The narrow tiers size acc/smem —
        // and, because the row loop is unrolled, the CODE — to the real row
        // count instead of batch16's 16; 0-handle → batch16 as before.
        let narrow = self.w4a16_batchm.kernel(n as u32);
        let fp4_gemv_batch_k = if narrow.0 != 0 {
            narrow
        } else {
            self.w4a16_gemv_batch16_k
        };
        if let Some(ref g) = self.exl3_gdn {
            // Native EXL3: the in_proj pair writes all n [Q|K|V|Z] rows of
            // `deinterleaved` (row stride qkvz_size) via the strided egress —
            // the only live QKVZ weight on this layer (install-time check).
            self.exl3_in_proj(g, ctx, normed_base, deinterleaved, n, stream)?;
        } else if let Some(ref fp8) = self.qkvz_fp8w {
            if use_batch4 {
                ops::w8a16_gemv_batch4(
                    ctx.gpu,
                    gemv_batch_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else if w8a16_pipe {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
            match (ssm_tc_proj_min_n(), self.qkvz_nvfp4_t.as_ref()) {
                (Some(min_n), Some(nvfp4_t)) if n >= min_n => {
                    // Tile GEMM on the transposed twin — the same call the SSM
                    // prefill path makes on this same weight. `ms_proj_gemm`
                    // picks the 128-row M-tile at wide batches so the weight
                    // is streamed once instead of ceil(n/64) times.
                    self.ms_proj_gemm(
                        ctx.gpu,
                        normed_base,
                        nvfp4_t,
                        deinterleaved,
                        n as u32,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
                // FP4 batched QKVZ: ONE NVFP4 weight pass for all n seqs
                // (sequential layout writes the deinterleaved buffer directly).
                _ => {
                    // w4a16_gemv_batch16 is a MAX_M=16 template: at M>16 it
                    // silently computes rows 0..15 and never writes rows 16..
                    // — garbage, not a crash. The eligibility gate makes this
                    // arm unreachable at n>16 today; fail fast if that drifts.
                    anyhow::ensure!(
                        n <= 16,
                        "SSM batchm QKVZ GEMV caps at M=16 (n={n}); tile-GEMM twins required"
                    );
                    ops::w4a16_gemv_batchm(
                        ctx.gpu,
                        fp4_gemv_batch_k,
                        normed_base,
                        nvfp4,
                        deinterleaved,
                        n as u32,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?
                }
            }
        } else {
            // BF16-kept GDN build: scalar `dense_gemm` costs ~1.03 ms/layer
            // at n=2 (measured) — cuBLASLt tensor-cores it (381 us).
            ops::cublas_bf16_proj_dense(
                normed_base,
                self.ssm.in_proj_qkvz.weight,
                deinterleaved,
                n as u32,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        }
        detail_step!("qkvz");

        // ── 3. Recurrent inner ──
        // Default: per-seq, byte-identical to ssm_forward. Experimental path:
        // use existing batch dimensions for BA/gates, conv, GDN, and gated norm
        // when the SSM pool states are contiguous slots [0..n).
        self.decode_ms_ssm_recurrent(
            states,
            n,
            normed_base,
            deinterleaved,
            normed_out_base,
            qkvz_size,
            key_dim,
            value_dim,
            conv_dim,
            qk_channels,
            d_conv,
            nk,
            nv,
            kd,
            vd,
            vpg,
            ba_size,
            h,
            bf16,
            eps,
            detail_profile,
            &mut detail_parts,
            &mut detail_t0,
            ctx,
            stream,
        )?;
        detail_step!("recurrent_total_tail");

        // ── 4. Batched out_proj: ONE [N,value_dim]→[N,h] GEMM (weights ×1) ──
        // Native EXL3 when the packed trellis is installed (the ONLY live
        // out_proj weight then), FP8 (w8a16) when the decode overlay is
        // installed, else BF16 dense.
        if let Some(ref g) = self.exl3_gdn {
            self.exl3_out_proj(g, ctx, normed_out_base, ssm_out_base, n, stream)?;
        } else if let Some(ref fp8) = self.out_proj_fp8w {
            if use_batch4 {
                ops::w8a16_gemv_batch4(
                    ctx.gpu,
                    gemv_batch_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else if w8a16_pipe {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        } else if let Some(ref out_proj_dense) = self.out_proj_dense {
            // Same cuBLASLt swap as the QKVZ arm (513 -> 194 us at n=2).
            ops::cublas_bf16_proj_dense(
                normed_out_base,
                out_proj_dense.weight,
                ssm_out_base,
                n as u32,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if self.qkvz_nvfp4.is_some() {
            match (ssm_tc_proj_min_n(), self.out_proj_nvfp4_t.as_ref()) {
                (Some(min_n), Some(nvfp4_t)) if n >= min_n => {
                    // Tile GEMM on the transposed twin — mirrors the SSM
                    // prefill out_proj call on this same weight. `ms_proj_gemm`
                    // picks the 128-row M-tile at wide batches so the weight
                    // is streamed once instead of ceil(n/64) times.
                    self.ms_proj_gemm(
                        ctx.gpu,
                        normed_out_base,
                        nvfp4_t,
                        ssm_out_base,
                        n as u32,
                        h as u32,
                        value_dim as u32,
                        stream,
                    )?;
                }
                // FP4 batched out_proj: ONE NVFP4 weight pass for all n seqs.
                // (qkvz_nvfp4.is_some() ⇒ the NVFP4 SSM build, where
                // ssm.out_proj is the NVFP4 weight the per-seq path uses.)
                _ => {
                    // Same MAX_M=16 template as the QKVZ arm — silent row
                    // truncation above 16. Unreachable at n>16 today; fail
                    // fast if the eligibility gate drifts.
                    anyhow::ensure!(
                        n <= 16,
                        "SSM batchm out_proj GEMV caps at M=16 (n={n}); tile-GEMM twins required"
                    );
                    ops::w4a16_gemv_batchm(
                        ctx.gpu,
                        fp4_gemv_batch_k,
                        normed_out_base,
                        &self.ssm.out_proj,
                        ssm_out_base,
                        n as u32,
                        h as u32,
                        value_dim as u32,
                        stream,
                    )?
                }
            }
        }
        detail_step!("out_proj");

        // GDN HeadParallel: reduce the row-parallel partial out_proj across TP
        // ranks (n × h BF16) before the residual add. No-op at tp=1.
        self.ssm_tp_all_reduce(ssm_out_base, normed_out_base, n, ctx, stream)?;

        // ── 5. Residual + post-norm (hc: caller hc_posts moe_output; skip) ──
        if !hc {
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                ssm_out_base,
                &self.post_attn_norm,
                normed_base,
                residual,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
        }
        detail_step!("post_norm", final);
        if detail_profile {
            let summary = detail_parts
                .iter()
                .map(|(label, us)| format!("{label}={us}us"))
                .collect::<Vec<_>>()
                .join(" ");
            tracing::info!("ATLAS_SSM_DETAIL n={n}: {summary}");
        }

        Ok(true)
    }
}
