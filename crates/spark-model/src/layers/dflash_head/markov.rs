// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DSpark sequential Markov fixup over the DFlash block logits.
//!
//! After the parallel backbone + lm_head GEMM write `[γ, vocab]` logits,
//! sample left to right instead of argmaxing all rows at once:
//!
//! ```text
//!   logits[i] += markov_w2 @ markov_w1[prev_i]     (low-rank bigram bias)
//!   draft[i]   = argmax(logits[i])
//! ```
//!
//! where `prev_0 = last_token` and `prev_i = draft[i-1]` for i ≥ 1 — the
//! greedy prev-token chain from the reference (`dspark.py::VanillaMarkov`,
//! adapted from DeepSeek's DeepSpec `markov_head.py`; the checkpoint's own
//! modeling file applies the bias at every position with the teacher-forced
//! previous token, which at greedy inference is exactly this chain).
//!
//! Graph-capture safety (this loop runs inside the captured tail subgraph):
//! every read is device-side. `draft_tokens_dev[0]` holds `last_token` at
//! tail entry — forward_block.rs step 2 H2Ds the `[last_token, MASK, …]`
//! token-id row into it on EVERY propose, before the captured region — and
//! row 0's bias is computed from that slot *before* row 0's argmax
//! overwrites it. Rows i ≥ 1 read the argmax output of row i-1, written
//! earlier on the same stream. No host round-trips, no per-call pointers
//! baked into the capture.
//!
//! Cost per row: one `[1,rank]` gather + one `[vocab,rank]` GEMV + one
//! `[vocab]` residual add + the argmax that was already there. The GEMV is
//! ~rank/hidden (256/5120 ≈ 1/20th) of the lm_head GEMM the block already
//! pays per row, so the fixup is launch-bound, not FLOP-bound.

use anyhow::Result;

use super::BlockDiffusionDraftHead;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl BlockDiffusionDraftHead {
    /// True when the sequential Markov fixup should run. A drafter without
    /// the head (plain DFlash) — or an operator override — degrades to the
    /// original batched argmax path bit-for-bit.
    pub(super) fn markov_active(&self) -> bool {
        self.markov_rank > 0
            && self.markov_w1.is_some()
            && self.markov_w2.is_some()
            && self.scratch.markov_embed.0 != 0
            && std::env::var("ATLAS_DSPARK_MARKOV").ok().as_deref() != Some("0")
    }

    /// True when the confidence head should also run inside the sequential
    /// chain: weights present, scratch allocated, and a positive threshold
    /// configured via `ATLAS_DSPARK_CONF_TAU` (0/unset = off, matching the
    /// reference's `threshold <= 0.0 → full block`).
    pub(super) fn confidence_active(&self) -> bool {
        self.confidence_proj.is_some() && self.scratch.conf_out.0 != 0 && Self::conf_tau() > 0.0
    }

    /// `ATLAS_DSPARK_CONF_TAU` parsed once per call site. Sigmoid-space
    /// threshold; the confident prefix ends at the first row whose predicted
    /// acceptance probability falls below it.
    pub(super) fn conf_tau() -> f32 {
        std::env::var("ATLAS_DSPARK_CONF_TAU")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0)
    }

    /// Sequential Markov-biased argmax over the γ block logits. Replaces the
    /// tail's per-row argmax loop when [`Self::markov_active`] is true.
    ///
    /// `norm_noise`: base pointer of the γ post-final-norm hidden rows (the
    /// same buffer the lm_head GEMM consumed) — the confidence head's hidden
    /// feature per the reference (`proposal_hidden_states` = backbone output
    /// after `norm`).
    ///
    /// Bias-vs-anchor convention: the reference applies the bigram bias at
    /// every position, so row 0 (the anchor row, prev = `last_token`) is
    /// biased by default. `ATLAS_DSPARK_ANCHOR_BIAS=0` exempts row 0
    /// (Lightning's official DSpark convention) for A/B measurement.
    pub(super) fn markov_argmax_block(
        &self,
        ctx: &ForwardContext,
        norm_noise: spark_runtime::gpu::DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let bf16 = 2usize;
        let vocab = self.vocab_size as u32;
        let rank = self.markov_rank as u32;
        let w1 = self
            .markov_w1
            .as_ref()
            .expect("markov_active() checked markov_w1");
        let w2 = self
            .markov_w2
            .as_ref()
            .expect("markov_active() checked markov_w2");
        let anchor_bias = std::env::var("ATLAS_DSPARK_ANCHOR_BIAS").ok().as_deref() != Some("0");
        let conf_on = self.confidence_active();
        let bf16u = 2usize;

        for i in 0..self.gamma {
            let logits_row = self.scratch.logits.offset(i * self.vocab_size * bf16);
            let token_slot = self.scratch.draft_tokens_dev.offset(i * 4);
            let biased = i > 0 || anchor_bias;
            if biased {
                // prev_0 lives in slot 0 itself (still `last_token` — this
                // read happens before row 0's argmax overwrites the slot);
                // prev_i for i ≥ 1 is row i-1's argmax output.
                let prev_slot = self
                    .scratch
                    .draft_tokens_dev
                    .offset(i.saturating_sub(1) * 4);
                ops::batched_embed(
                    gpu,
                    self.kernels.batched_embed,
                    prev_slot,
                    w1.weight,
                    self.scratch.markov_embed,
                    1,
                    rank,
                    stream,
                )?;
                ops::dense_gemv(
                    gpu,
                    self.kernels.dense_gemv,
                    self.scratch.markov_embed,
                    w2,
                    self.scratch.markov_bias,
                    vocab,
                    rank,
                    stream,
                )?;
                ops::residual_add(
                    gpu,
                    self.kernels.residual_add,
                    logits_row,
                    self.scratch.markov_bias,
                    vocab,
                    stream,
                )?;
                // ── Confidence head (AcceptRatePredictor) for this row ──
                // conf[i] = W · [hidden_i ‖ markov_embed_i] + b, computed as
                // two GEMV slices over the contiguous [1, hidden+rank] weight
                // (hidden cols first — dspark.py builds conf_input_dim as
                // hidden_size then += markov_rank). The prev-token embed is
                // the one just gathered for the Markov bias (the reference
                // uses the same prev chain for both heads). markov_bias[0]
                // is free as a 1-element staging slot here — its vocab-wide
                // content was consumed by the residual_add above.
                if conf_on {
                    let (w, b) = (
                        self.confidence_proj.as_ref().expect("confidence_active"),
                        self.confidence_bias.as_ref().expect("confidence_active"),
                    );
                    let conf_i = self.scratch.conf_out.offset(i * bf16u);
                    let hidden_row = norm_noise.offset(i * self.hidden_size * bf16u);
                    ops::dense_gemv(
                        gpu,
                        self.kernels.dense_gemv,
                        hidden_row,
                        w,
                        conf_i,
                        1,
                        self.hidden_size as u32,
                        stream,
                    )?;
                    if self.confidence_with_markov {
                        let w_embed = crate::weight_map::DenseWeight {
                            weight: w.weight.offset(self.hidden_size * bf16u),
                        };
                        ops::dense_gemv(
                            gpu,
                            self.kernels.dense_gemv,
                            self.scratch.markov_embed,
                            &w_embed,
                            self.scratch.markov_bias,
                            1,
                            rank,
                            stream,
                        )?;
                        ops::residual_add(
                            gpu,
                            self.kernels.residual_add,
                            conf_i,
                            self.scratch.markov_bias,
                            1,
                            stream,
                        )?;
                    }
                    ops::residual_add(gpu, self.kernels.residual_add, conf_i, b.weight, 1, stream)?;
                }
            }
            ops::argmax_bf16(
                gpu,
                self.kernels.argmax,
                logits_row,
                token_slot,
                vocab,
                stream,
            )?;
        }
        Ok(())
    }
}
