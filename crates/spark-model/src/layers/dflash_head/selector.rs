// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash2 candidate-path selector — host-side v0.
//!
//! Reference: z-lab/dflash `CandidateSelector.select` (and llama.cpp
//! PR #27342's port). Per draft row t (Atlas rows 1..γ — row 0 is the
//! anchor echo the propose path drops):
//!
//!   score_k = logit[t, cand_k]
//!           + dot(pred_codebook[prev] ⊙ proj(h_t), succ_codebook[cand_k])
//!   pick_t  = cand_{argmax score}, prev := pick_t
//!
//! over the top-`selector_top_k` logits of the row, with `prev` seeded by
//! the anchor (`last_token` — the reference's `block_output_ids[:, 0]`).
//!
//! Runs AFTER the propose tail (including a CUDA-graph replay: the tail
//! writes `scratch.logits` / the norm buffer every propose, and the
//! draft-token D2H in `forward_block` has already drained the stream), so
//! no graph suppression is needed. Cost at γ=8 / vocab 248320: one 4 MB
//! logits D2H + ~10 M host MACs per propose — the v0 trade for zero new
//! device code; a GPU top-k is the follow-up if the accept win holds.
//!
//! Greedy only (matches Atlas's raw-argmax DFlash verify basis). The
//! reference's temperature>0 path draws from selector scores with
//! rejection sampling — not ported yet.
//!
//! Kill switch: `ATLAS_NO_DFLASH2_SELECTOR` (presence) restores the plain
//! per-row argmax for A/B.

use anyhow::{Context, Result};
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::BlockDiffusionDraftHead;

fn selector_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("ATLAS_NO_DFLASH2_SELECTOR").is_some())
}

#[inline]
fn b2f(u: u16) -> f32 {
    bf16::from_bits(u).to_f32()
}

impl BlockDiffusionDraftHead {
    /// Replace the per-row argmax drafts with the selector's chain walk.
    /// `argmax_drafts` is the tail's picks (row 0 kept verbatim — anchor
    /// echo); `norm_noise` is the post-final-norm hidden `[γ, H]` BF16 the
    /// lm_head consumed. Any failure falls back to the argmax drafts.
    pub(super) fn dflash2_selector_pick(
        &self,
        gpu: &dyn GpuBackend,
        argmax_drafts: &[u32],
        last_token: u32,
        norm_noise: DevicePtr,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let Some(sel) = &self.dflash2_selector else {
            return Ok(argmax_drafts.to_vec());
        };
        if selector_off() || argmax_drafts.len() < 2 {
            return Ok(argmax_drafts.to_vec());
        }
        let gamma = self.gamma;
        let vocab = self.vocab_size;
        let h = self.hidden_size;
        let rank = sel.rank;
        let top_k = sel.top_k.max(1);

        // The draft-token D2H just above this call event-synced the stream,
        // so both buffers are quiescent.
        let mut logits = vec![0u8; gamma * vocab * 2];
        gpu.copy_d2h(self.scratch.logits, &mut logits)
            .context("DFlash2 selector: logits D2H")?;
        let mut hidden = vec![0u8; gamma * h * 2];
        gpu.copy_d2h(norm_noise, &mut hidden)
            .context("DFlash2 selector: hidden D2H")?;

        let hp = &sel.hidden_projection; // [rank, h] bf16
        let pred = &sel.predecessor_codebook; // [vocab, rank] bf16
        let succ = &sel.successor_codebook; // [vocab, rank] bf16

        let mut out = argmax_drafts.to_vec();
        let mut prev = last_token as usize;
        for t in 1..gamma.min(argmax_drafts.len()) {
            let row = &logits[t * vocab * 2..(t + 1) * vocab * 2];

            // Top-k logits of the row (small fixed k: insertion into a
            // sorted-by-min scratch beats a full sort of 248k).
            let mut cand: Vec<(f32, usize)> = Vec::with_capacity(top_k + 1);
            let mut floor = f32::NEG_INFINITY;
            for (id, ch) in row.chunks_exact(2).enumerate() {
                let v = b2f(u16::from_le_bytes([ch[0], ch[1]]));
                if v > floor || cand.len() < top_k {
                    cand.push((v, id));
                    cand.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    cand.truncate(top_k);
                    floor = cand.last().map(|c| c.0).unwrap_or(f32::NEG_INFINITY);
                }
            }

            // g = proj(h_t): [rank] = hidden_projection · h_t.
            let ht = &hidden[t * h * 2..(t + 1) * h * 2];
            let htf: Vec<f32> = ht
                .chunks_exact(2)
                .map(|c| b2f(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let mut g = vec![0f32; rank];
            for (r, gr) in g.iter_mut().enumerate() {
                let wrow = &hp[r * h..(r + 1) * h];
                let mut acc = 0f32;
                for (w, a) in wrow.iter().zip(&htf) {
                    acc += b2f(*w) * a;
                }
                *gr = acc;
            }

            // e = pred[prev] ⊙ g, then score each candidate.
            let prow = &pred[prev * rank..(prev + 1) * rank];
            let e: Vec<f32> = prow.iter().zip(&g).map(|(p, gv)| b2f(*p) * gv).collect();
            let mut best = (f32::NEG_INFINITY, cand[0].1);
            for &(unary, id) in &cand {
                let srow = &succ[id * rank..(id + 1) * rank];
                let mut dot = 0f32;
                for (s, ev) in srow.iter().zip(&e) {
                    dot += b2f(*s) * ev;
                }
                let score = unary + dot;
                if score > best.0 {
                    best = (score, id);
                }
            }
            out[t] = best.1 as u32;
            prev = best.1;
        }
        Ok(out)
    }
}
