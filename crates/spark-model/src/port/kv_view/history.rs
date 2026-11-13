// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the bounded view's selection history (port lane L32), pure
//! host functions over per-block score vectors:
//!
//! - [`propagate`]: one simultaneous diffusion step of a round's scores, a
//!   share `alpha / 4` of each block's positive score to each of its two
//!   neighbors and `alpha / 2` split over its sampled attention edges to
//!   earlier blocks, the rest staying in place (kvmem's formula). Total mass
//!   is conserved; negative scores never spread. `AVAROK_KV_VIEW_GRAPH_ALPHA`
//!   (0 = off, the identity).
//! - [`History`]: decayed relevance, `H = lambda * H_prev + min(r, 0) + hit *
//!   max(r, 0)` with `r` the round's score minus the round mean and `hit` a
//!   score in the top 90% of the round. A preview ranks the current
//!   selection; it is committed once the conversation continues from that
//!   round (an accepted round). Vectors are immutable `Arc`s so forks of one
//!   conversation share them. `AVAROK_KV_VIEW_HISTORY_LAMBDA` (0 = off: the
//!   ranking is the round's scores).
//!
//! The attention-edge sampler (a feedback kernel recording the top past
//! destinations of sampled rows) is not built in this sprint, so the edge list
//! is empty and the edge share stays in place; see `KVMEM-VIEW-NOTES.md`.

use std::sync::Arc;

/// Fraction of a round's blocks counted as hits (the top 90%).
pub const HIT_FRACTION: f64 = 0.9;

/// One sampled attention edge: block `src` attended block `dst` with weight
/// `w` (only `dst < src` edges are used: no self or future edges).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    pub src: u32,
    pub dst: u32,
    pub w: f32,
}

/// One diffusion step with weight `alpha` (0 returns the input unchanged).
pub fn propagate(scores: &[f32], edges: &[Edge], alpha: f32) -> Vec<f32> {
    let n = scores.len();
    if alpha <= 0.0 || n == 0 {
        return scores.to_vec();
    }
    let a = f64::from(alpha);
    let mut out: Vec<f64> = scores.iter().map(|&s| f64::from(s)).collect();
    // Edge weights per source, normalized over its valid edges.
    let mut edge_sum = vec![0f64; n];
    for e in edges {
        if e.dst < e.src && (e.src as usize) < n && e.w > 0.0 {
            edge_sum[e.src as usize] += f64::from(e.w);
        }
    }
    for (b, &s) in scores.iter().enumerate() {
        let pos = f64::from(s.max(0.0));
        if pos == 0.0 {
            continue;
        }
        let neighbor = a / 4.0 * pos;
        if b > 0 {
            out[b - 1] += neighbor;
            out[b] -= neighbor;
        }
        if b + 1 < n {
            out[b + 1] += neighbor;
            out[b] -= neighbor;
        }
        if edge_sum[b] > 0.0 {
            let share = a / 2.0 * pos;
            for e in edges
                .iter()
                .filter(|e| e.src as usize == b && e.dst < e.src && e.w > 0.0)
            {
                let moved = share * f64::from(e.w) / edge_sum[b];
                out[e.dst as usize] += moved;
                out[b] -= moved;
            }
        }
    }
    out.into_iter().map(|v| v as f32).collect()
}

/// The score at or above which a block counts as a hit (the round's top
/// [`HIT_FRACTION`]). Deterministic for equal inputs.
pub fn hit_threshold(scores: &[f32]) -> f32 {
    if scores.is_empty() {
        return f32::INFINITY;
    }
    let mut v: Vec<f32> = scores.to_vec();
    // Integer form of HIT_FRACTION (9 of 10), so no float rounding moves the cut.
    let hits = (9 * v.len()).div_ceil(10);
    let k = (v.len() - hits).min(v.len() - 1);
    let (_, kth, _) = v.select_nth_unstable_by(k, f32::total_cmp);
    *kth
}

/// A conversation's committed relevance history (one value per block).
#[derive(Clone, Debug, PartialEq)]
pub struct History {
    pub committed: Arc<[f32]>,
    /// Rounds committed so far.
    pub rounds: u32,
}

impl Default for History {
    fn default() -> Self {
        Self {
            committed: Arc::from(Vec::new()),
            rounds: 0,
        }
    }
}

impl History {
    /// The ranking of this round: `lambda * H_prev + min(r, 0) + hit * max(r, 0)`
    /// over `scores` (blocks past the committed length start at 0). With
    /// `lambda == 0` the ranking is `scores` itself.
    pub fn preview(&self, scores: &[f32], lambda: f32) -> Arc<[f32]> {
        if lambda <= 0.0 || scores.is_empty() {
            return Arc::from(scores);
        }
        let mean = scores.iter().map(|&s| f64::from(s)).sum::<f64>() / scores.len() as f64;
        let thr = hit_threshold(scores);
        let lam = f64::from(lambda);
        scores
            .iter()
            .enumerate()
            .map(|(b, &s)| {
                let prev = self.committed.get(b).map_or(0.0, |&h| f64::from(h));
                let r = f64::from(s) - mean;
                let hit = if s >= thr { 1.0 } else { 0.0 };
                (lam * prev + r.min(0.0) + hit * r.max(0.0)) as f32
            })
            .collect()
    }

    /// The history after an accepted round whose preview was `preview`.
    pub fn commit(&self, preview: Arc<[f32]>) -> Self {
        Self {
            committed: preview,
            rounds: self.rounds + 1,
        }
    }
}
