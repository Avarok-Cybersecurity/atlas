// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_multi_seq.

use super::*;

mod hc;
mod ssm_batched;
mod ssm_batched_recurrent;

/// Batched dense FFN for wide SSM decode batches: **ON by default**, disabled
/// only by `ATLAS_NO_SSM_FFN_PREFILL=1`.
///
/// Strict `== "1"` on an `ATLAS_NO_*` name rather than a presence check —
/// presence-checked flags in this codebase are ENABLED by `=0`, a trap that has
/// burned it before. Read once; the dispatch site is per-layer per-step.
/// Smallest batch that takes the batched dense FFN. Default 5, MEASURED.
///
/// Tunable because the +30% measured at C=16 is LARGER than eliminating the
/// double weight read alone predicts (~18%), which implies the tile GEMM also
/// beats the batch-8 GEMV per pass. If that holds, the crossover is below 9 and
/// C=4/C=8 have headroom too — `ATLAS_SSM_FFN_PREFILL_MIN_N` exists to find it
/// by measurement rather than assertion. It was: 2 reps/cell, coherence held —
///   MIN_N=9: C=4 37.7 | C=8 53.4
///   MIN_N=5: C=4 37.8 | C=8 **57.8**  (+8% at C=8, C=4 untouched)
///   MIN_N=4: C=4 36.2 | C=8 57.8      (C=4 regresses — GEMV still wins at 4)
/// So the tile GEMM overtakes the batch-8 GEMV at n=5, not n=9: eliminating the
/// double weight read was only part of the win, the kernel is simply better per
/// pass. 5 is the default; 4 is a measured regression; 9 was the conservative
/// first guess.
fn ssm_ffn_prefill_min_n() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_SSM_FFN_PREFILL_MIN_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v| v >= 2)
            .unwrap_or(5)
    })
}

fn ssm_ffn_prefill_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_NO_SSM_FFN_PREFILL").as_deref() != Ok("1"))
}

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
        let ssm_ms_profile = std::env::var("ATLAS_SSM_MS_PROFILE").ok().as_deref() == Some("1")
            && !ctx.graph_capture;
        let phase_a_t0 = if ssm_ms_profile {
            ctx.gpu.synchronize(stream).ok();
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Per-seq hidden/residual stride: the residual stream is always
        // BF16 (2 bytes), so hardcode the per-seq stride.
        let residual_elem = 2usize;

        // ── Phase A: SSM mixer ──
        // Pre-norm, SSM mixer (recurrent, per-seq state), post-attn-norm.
        // Lays out `norm_output[0..n]` as the contiguous [N, h] BF16 MoE
        // input. The MoE is deferred to Phase B.
        //
        // Fast path (batched projections): when the layer uses the
        // sequential-QKVZ dense/NVFP4 weights with the FP32 conv+GDN
        // recurrent kernels (the GB10 Holo serving config), the big
        // QKVZ and out_proj GEMMs are batched into a single [N, ...] GEMM
        // each — reading the ~50 MB QKVZ / out_proj weights ONCE instead
        // of N times. On bandwidth-bound LPDDR5X this is the dominant
        // decode cost, so it is the lever that makes C=N decode scale.
        // The recurrent inner (BA/gates, conv1d, GDN, gated-norm) stays a
        // per-seq loop with byte-identical kernels to `decode()`/`ssm_forward`.
        if !self
            .try_decode_multi_seq_ssm_batched(hidden, residual, n, states, false, ctx, stream)?
        {
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
        }
        let phase_a_us = if let Some(t0) = phase_a_t0 {
            ctx.gpu.synchronize(stream).ok();
            t0.elapsed().as_micros()
        } else {
            0
        };
        let phase_b_t0 = if ssm_ms_profile {
            Some(std::time::Instant::now())
        } else {
            None
        };

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
                    self.ffn.forward_k2(normed_base, 0, ctx, stream)?;
                } else {
                    self.ffn.forward_k3(normed_base, 0, ctx, stream)?;
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
            // WIDE BATCH (n>8): one batched dense FFN, weights read ONCE.
            //
            // MEASURED 2026-07-27 (ATLAS_SSM_MS_PROFILE, 9168 samples/n): the
            // chunked arm below costs 751 / 1023 / 2022 us per layer at
            // n=4/8/16 — n=16 is 1.98x n=8, and FFN-per-sequence is FLAT from
            // 8 to 16 (127.9 -> 126.3 us). That is the signature of the m<=8
            // GEMV cap: above 8 the ~7.2 GB of FFN weights are streamed TWICE
            // per step. A batched FFN is weight-bandwidth-bound, so n=16 should
            // cost about what n=8 does. Worth ~1000us x 48 layers = ~48 ms of a
            // 264.9 ms step.
            //
            // forward_prefill reads gate/up/down once for all n rows — the
            // ATTENTION layers already take exactly this branch for dense
            // models (multi_seq/ffn.rs, "WIDE-VERIFY BATCHED DENSE FFN"). This
            // is that arm's SSM-side twin.
            //
            // n<=8 deliberately KEEPS the chunked GEMV: the recorded crossover
            // has GEMV winning at M=4, and a single chunk reads the weights
            // once anyway, so there is nothing to gain there.
            //
            // DENSE ONLY: on a 256-expert MoE the grouped-GEMM is a net loss at
            // small batch (see the per-token arm below), so MoE keeps the loop.
            n if n >= ssm_ffn_prefill_min_n()
                && self.ffn.is_dense()
                && ssm_ffn_prefill_enabled() =>
            {
                self.ffn.forward_prefill(normed_base, n, ctx, stream)?;
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden,
                    ctx.buffers.moe_output(),
                    (n * h) as u32,
                    stream,
                )?;
            }
            4.. if self.ffn.can_forward_km(8.min(n) as u32) => {
                // MISSING n=4..8 ARM (2026-07-26) — the SSM-side twin of the
                // attention ladder's 2026-07-24 K=4 gap (multi_seq/ffn.rs:100).
                // The ladder fell from n==2/3 straight to the per-token loop,
                // so C=4..8 decode streamed the full dense FFN weights n
                // TIMES per layer. MS_PROFILE at C=4: 2632us/layer FFN vs
                // 577us mixer = 63% of the step, 126ms of ~200ms across the
                // 48-layer SSM stack. forward_km reads each projection ONCE
                // for all n rows (batched GEMV, ~290 GB/s vs the loop's
                // per-seq streams). Dense-only by construction:
                // can_forward_km is false for MoE, which keeps the loop.
                // n > 8 is processed in chunks of 8 (the batchm GEMV family
                // caps at m=8): weights are read ceil(n/8) times instead of n.
                // moe_output[0..m] is reused per chunk, so each chunk's
                // residual add runs before the next chunk overwrites it.
                let mut done = 0usize;
                while done < n {
                    let m = (n - done).min(8);
                    let normed_c = normed_base.offset(done * h * bf16);
                    let used = self.ffn.try_forward_km(normed_c, m as u32, ctx, stream)?;
                    debug_assert!(used, "can_forward_km checked at branch entry");
                    let hidden_c = hidden.offset(done * h * residual_elem);
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden_c,
                        ctx.buffers.moe_output(),
                        (m * h) as u32,
                        stream,
                    )?;
                    done += m;
                }
            }
            _ => {
                // Per-token MoE: each seq's forward() writes moe_output[0];
                // consume it immediately with a per-seq residual add before
                // the next iteration overwrites it.
                //
                // WIDTH-DEPENDENT, and the two measurements below disagree only
                // because they were taken at different n — read both before
                // concluding anything about the grouped arm:
                //
                //   n=4    grouped-GEMM (forward_prefill) is SLOWER on Holo,
                //          c4 31 vs 56 tok/s — the expert sort/permute fixed
                //          overhead per layer dominates at small N.
                //   n>=16  grouped-GEMM WINS: SSM-side flag alone measured
                //          C=32 172.7 -> 216.2 tok/s (+25%), and #415's
                //          attention-side extension adds +7.9% at C=32 /
                //          +9.7% at C=64 on Qwen3.6-35B-A3B-NVFP4 (paired
                //          gsm8k n=200 gate: strict 0.960 vs 0.900 baseline).
                //
                // So the flag below is not a loss waiting to be removed; it is
                // a narrow-N loss and a wide-N win, currently opt-in at every
                // width. Flipping the default for n>=16 is the open question
                // (#415 raised it); no recipe sets the variable today, so the
                // wide-N win is measured but not realised in any shipped
                // config.
                //
                // The launch-overhead fix at small N remains CUDA graphs for
                // n>=2, not MoE batching (graphs capture these per-token
                // launches for free).
                if crate::layers::moe_grouped_decode_for(n) {
                    // Grouped-GEMM MoE over all N tokens (each expert read once).
                    // Only sensible under CUDA graphs, where the sort/permute
                    // launch overhead that made this a loss is captured for free.
                    self.ffn.forward_prefill(normed_base, n, ctx, stream)?;
                    let moe_out = ctx.buffers.moe_output();
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden,
                        moe_out,
                        (n * h) as u32,
                        stream,
                    )?;
                } else if n == 4
                    && std::env::var("ATLAS_MOE_ATOMIC_C4_DECODE").ok().as_deref() == Some("1")
                {
                    // Purpose-built C=4 routed MoE decode: batched routing,
                    // token-major gate/up, FP32 atomicAdd routed down
                    // accumulation, then BF16 finalize/blend.
                    self.ffn
                        .forward_atomic_c4_decode(normed_base, n, ctx, stream)?;
                    let moe_out = ctx.buffers.moe_output();
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden,
                        moe_out,
                        (n * h) as u32,
                        stream,
                    )?;
                } else if std::env::var("ATLAS_MOE_LEGACY_PERTOKEN_DECODE")
                    .ok()
                    .as_deref()
                    != Some("1")
                {
                    // Token-major N-token MoE decode — DEFAULT for n>=4. Packs
                    // (token, expert) into blockIdx.y so all N tokens' experts run
                    // in ~3 batched kernel launches/layer instead of the legacy
                    // per-token loop's ~24 serial launches (which also aliased
                    // scratch → forced cross-token serialization) + a wsum_blend
                    // starved to 8 blocks. Measured +7-19% concurrent decode.
                    // LoRA (SOLID Incr-4): with a resident MoE adapter,
                    // `forward_token_major_decode` DELEGATES to the per-row
                    // `forward_batched` folds before any GPU work (presence
                    // gate, like forward_k2/k3); the `.is_err()` fallback below
                    // remains for non-NVFP4 weights / genuine errors only.
                    // Opt out fully with ATLAS_MOE_LEGACY_PERTOKEN_DECODE=1.
                    if self
                        .ffn
                        .forward_token_major_decode(normed_base, n, ctx, stream)
                        .is_err()
                    {
                        self.ffn.forward_batched(normed_base, n, ctx, stream)?;
                    }
                    let moe_out = ctx.buffers.moe_output();
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden,
                        moe_out,
                        (n * h) as u32,
                        stream,
                    )?;
                } else if std::env::var("ATLAS_MOE_BATCHED_DECODE").ok().as_deref() == Some("1") {
                    // Batched gate GEMM over all N tokens, but keep the proven
                    // per-token expert kernels. This avoids the grouped path's
                    // sort/GEMM overhead while testing whether reading router
                    // weights once helps C=4 decode.
                    self.ffn.forward_batched(normed_base, n, ctx, stream)?;
                    let moe_out = ctx.buffers.moe_output();
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden,
                        moe_out,
                        (n * h) as u32,
                        stream,
                    )?;
                } else {
                    for i in 0..n {
                        let hidden_i = hidden.offset(i * h * residual_elem);
                        let normed_i = normed_base.offset(i * h * bf16);
                        let moe_out = self.ffn.forward(normed_i, i, ctx, stream)?;
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
        }
        if let Some(t0) = phase_b_t0 {
            ctx.gpu.synchronize(stream).ok();
            tracing::info!(
                "ATLAS_SSM_MS_PROFILE n={n}: mixer={}us moe_residual={}us",
                phase_a_us,
                t0.elapsed().as_micros(),
            );
        }

        Ok(())
    }
}
