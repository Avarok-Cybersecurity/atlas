// SPDX-License-Identifier: AGPL-3.0-only

//! The small-M mHC FFN arm, split out of `trait_decode_batched_hc.rs` to
//! keep it under the 500-line cap.

use super::*;

impl Qwen3SsmLayer {
    pub(in crate::layers::qwen3_ssm) fn hc_small_m_ffn(
        &self,
        rows: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let small_m = {
            static SMALL_M: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *SMALL_M.get_or_init(|| {
                std::env::var("ATLAS_QWEN4EXP_HC_SMALL_M_FFN").as_deref() != Ok("0")
            })
        };
        // ROW-EXACT (`ForwardContext::gdn_exact_replay`, kill switch
        // ATLAS_NO_VERIFY_ROW_EXACT): every arm below dispatches on ROW COUNT
        // — `forward_k2` fuses both rows' expert GEMVs into 5 launches where
        // `forward` runs 5 per row — and row-count-shaped MoE arms round
        // differently from the single-row one (#459). A verify whose row 0
        // re-processes an already-committed token has to reproduce that
        // token's serial `decode()`, and decode's MoE is `ffn.forward` at ONE
        // row. So under the row-exact contract the FFN runs one row at a time.
        //
        // DESCENDING, and that is load-bearing: `forward` always writes ROW 0
        // of `moe_output()`, so each row's result has to be moved aside before
        // the next call overwrites it. The FFN is stateless across rows, so
        // visiting them K-1..0 lets every row but 0 be copied out immediately
        // and leaves row 0's result in the place it already belongs — no
        // staging buffer, no lost row.
        if crate::layers::qwen3_ssm::verify_row_exact_leg(
            ctx.gdn_exact_replay,
            crate::layers::qwen3_ssm::RowExactLeg::Ffn,
        ) {
            let h = ctx.config.hidden_size;
            let moe_out = ctx.buffers.moe_output();
            for t in (0..num_tokens).rev() {
                let out = self.ffn.forward(rows.offset(t * h * 2), ctx, stream)?;
                anyhow::ensure!(
                    out == moe_out,
                    "row-exact FFN: single-token MoE returned a buffer other than \
                     moe_output(), which the hc post-site reads unconditionally"
                );
                if t > 0 {
                    ctx.gpu
                        .copy_d2d_async(moe_out, moe_out.offset(t * h * 2), h * 2, stream)?;
                }
            }
            return Ok(());
        }
        use super::super::hc_ffn_dispatch::{HcFfnDispatch, hc_ffn_dispatch};
        let km_available = self.ffn.can_forward_km(num_tokens as u32);
        {
            // Engagement, once per width band. `can_forward_km() == false`
            // routes 4..=8 rows to `Prefill` — the per-token expert loop over
            // all 512 experts that made MTP-3 read 14.9 tok/s (#1060) — and
            // that fallback is SILENT. A K=3 verify is 4 rows at C=1, so a
            // DRAFTS=3 measurement is uninterpretable without knowing which
            // arm ran.
            static SAID: std::sync::Once = std::sync::Once::new();
            if (4..=16).contains(&num_tokens) {
                SAID.call_once(|| {
                    tracing::info!(
                        num_tokens,
                        km_available,
                        "hc small-M FFN at {num_tokens} rows: {}",
                        if km_available {
                            "Km batched arm"
                        } else {
                            "PREFILL FALLBACK (512-expert loop)"
                        }
                    );
                });
            }
        }
        match hc_ffn_dispatch(
            num_tokens,
            small_m,
            ctx.gdn_exact_replay,
            self.ffn.exl3_native_moe(),
            km_available,
        ) {
            HcFfnDispatch::Single => {
                let out = self.ffn.forward(rows, ctx, stream)?;
                anyhow::ensure!(
                    out == ctx.buffers.moe_output(),
                    "small-M FFN: single-token MoE returned a buffer other than \
                     moe_output(), which the hc post-site reads unconditionally"
                );
            }
            HcFfnDispatch::K2 => self.ffn.forward_k2(rows, ctx, stream)?,
            HcFfnDispatch::K3 => self.ffn.forward_k3(rows, ctx, stream)?,
            HcFfnDispatch::Km => {
                let ran = self
                    .ffn
                    .try_forward_km(rows, num_tokens as u32, ctx, stream)?;
                anyhow::ensure!(ran, "K=m FFN arm reported available and then declined");
            }
            HcFfnDispatch::NativeBatched => match &self.ffn {
                FfnComponent::Moe(moe) => moe.forward_batched(rows, num_tokens, ctx, stream)?,
                _ => anyhow::bail!("native EXL3 replay requires a MoE FFN"),
            },
            HcFfnDispatch::Prefill => {
                // No fused arm at this width. Decompose into fused chunks
                // rather than paying the grouped GEMM, which streams all 512
                // experts regardless of row count. See the module note.
                let chunked = {
                    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    *ON.get_or_init(|| std::env::var("ATLAS_HC_FFN_CHUNKED").as_deref() != Ok("0"))
                };
                // ★ CAPPED at verify widths. This decomposition was measured
                // (73cb95b43) against the grouped GEMM at 1..24 rows, where the
                // fused ladder is ~105 us/row and the grouped GEMM streams all
                // 512 experts for a fixed ~0.5-2 ms regardless of row count —
                // so at verify widths the ladder wins (DRAFTS=3 C=1 15.24 ->
                // 28.10). But `hc_small_m_ffn` is ALSO the entry the hc PREFILL
                // path calls with its whole chunk, and ~105 us/row FLAT at 2052
                // rows is ~215 ms per GDN layer against the grouped GEMM's
                // 27 ms at that width. Every profiled prefill chunk on this
                // model showed the one-shot "CHUNKED into fused arms" line at
                // num_tokens=2052, and ~150 ms per GDN layer that no phase
                // timer could attribute — 684 fused launches + 684 D2D copies
                // per layer per chunk, x36 layers = most of every 7.3 s chunk.
                // Prefill 8K/11K was 231-261/267 tok/s with it; see the commit
                // that added this cap for the number without it.
                //
                // The cap is a ROW COUNT because that is the axis the curve was
                // measured on: nothing narrower than the widest batched verify
                // (11 rows at C=4 K=2, ~15 at K=3, VERIFY_ROW_CAP is 96) should
                // change, and no prefill chunk (hundreds to thousands of rows)
                // should ever land here. See `hc_ffn_chunk_max_rows` for why 64.
                if chunked
                    && small_m
                    && num_tokens > 3
                    && num_tokens <= Self::hc_ffn_chunk_max_rows()
                {
                    let h = ctx.config.hidden_size;
                    let bf16 = 2usize;
                    // Widths, seq-major: 3s then the 1-or-2 remainder.
                    let mut widths: Vec<usize> = Vec::new();
                    let mut left = num_tokens;
                    while left > 0 {
                        let w = left.min(3);
                        widths.push(w);
                        left -= w;
                    }
                    {
                        static SAID: std::sync::Once = std::sync::Once::new();
                        SAID.call_once(|| {
                            tracing::info!(
                                num_tokens,
                                "hc small-M FFN: CHUNKED into fused arms instead of the \
                                 grouped GEMM (ATLAS_HC_FFN_CHUNKED=0 restores it)"
                            );
                        });
                    }
                    // DESCENDING offsets: each chunk is copied to its place
                    // before the next one overwrites moe_output[0, k).
                    let mut off = num_tokens;
                    for &w in widths.iter().rev() {
                        off -= w;
                        let src = rows.offset(off * h * bf16);
                        match w {
                            1 => {
                                let out = self.ffn.forward(src, ctx, stream)?;
                                anyhow::ensure!(
                                    out == ctx.buffers.moe_output(),
                                    "chunked small-M FFN: single-token MoE returned a \
                                     buffer other than moe_output()"
                                );
                            }
                            2 => self.ffn.forward_k2(src, ctx, stream)?,
                            _ => self.ffn.forward_k3(src, ctx, stream)?,
                        }
                        if off > 0 {
                            ctx.gpu.copy_d2d_async(
                                ctx.buffers.moe_output(),
                                ctx.buffers.moe_output().offset(off * h * bf16),
                                w * h * bf16,
                                stream,
                            )?;
                        }
                    }
                } else {
                    self.ffn.forward_prefill(rows, num_tokens, ctx, stream)?;
                }
            }
        }
        Ok(())
    }
}
