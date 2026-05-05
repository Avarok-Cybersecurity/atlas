// SPDX-License-Identifier: AGPL-3.0-only

//! Lookback-Lens guided decoding (C.4, 2026-04-25).
//!
//! Reference: arXiv:2407.07071. Per-head attention-mass ratio
//! classifier — the model's per-head attention is sharply biased
//! toward the tool-result tokens when it's actually using them, vs
//! diffusely scattered when it's hallucinating beyond what the tool
//! returned. A small linear classifier over per-head ratios reranks
//! top-k candidates on tool-result turns.
//!
//! Reported gain: -10% contextual hallucinations on summarisation /
//! QA tasks. Most relevant to Atlas's tool-result paths where the
//! model fabricates beyond what the tool actually said (e.g.
//! invented file contents).
//!
//! ## Scope
//!
//! Atlas's attention kernel currently doesn't expose per-head
//! attention sums. FlashInfer can return logsumexp (LSE) cheaply,
//! and the per-head attention mass over a span is derivable from
//! that — but that path needs kernel cooperation.
//!
//! This module ships:
//!   - The classifier weights container (loaded from offline
//!     calibration similar to halluc_probe).
//!   - The ratio-feature builder (given per-head attention sums
//!     over a "lookback" span and over the rest, compute the
//!     normalised ratio per head).
//!   - The rerank logic (apply classifier to top-k candidates,
//!     pick the one with highest grounded score).
//!
//! Production wiring requires:
//!   1. Attention kernel exposes per-head attention mass over a
//!      designated token span (the most recent tool_result block).
//!   2. Offline classifier training on grounded vs hallucinated
//!      generation traces.
//!   3. Server wires the rerank into the sample path on tool-result
//!      turns only (gate by detecting recent tool_result message in
//!      the prompt).

/// Per-head attention sums over the lookback window AND over the
/// rest of the prompt. `lookback_sums[h]` is the total attention
/// mass head h placed on lookback-window tokens for the current
/// generation step. `rest_sums[h]` is the mass on everything else.
/// Both are post-softmax probabilities (per-head, summing to 1
/// across all attended positions).
#[derive(Debug, Clone)]
pub struct AttentionSums {
    pub lookback_sums: Vec<f32>,
    pub rest_sums: Vec<f32>,
}

impl AttentionSums {
    /// Build the per-head ratio feature vector: `ratio[h] =`
    /// lookback / (lookback + rest). Bounded in [0, 1]; closer to
    /// 1 means the head is "looking back" at the tool result.
    pub fn ratios(&self) -> Vec<f32> {
        if self.lookback_sums.len() != self.rest_sums.len() {
            return Vec::new();
        }
        self.lookback_sums
            .iter()
            .zip(self.rest_sums.iter())
            .map(|(lb, rest)| {
                let denom = lb + rest;
                if denom > 1e-10 { lb / denom } else { 0.0 }
            })
            .collect()
    }

    pub fn num_heads(&self) -> usize {
        self.lookback_sums.len()
    }
}

/// Linear classifier on per-head ratios → grounded probability.
/// Loaded from offline calibration (similar to `halluc_probe::LinearProbe`).
#[derive(Debug, Clone)]
pub struct GroundedClassifier {
    weights: Vec<f32>,
    bias: f32,
}

impl GroundedClassifier {
    pub fn new(weights: Vec<f32>, bias: f32) -> Option<Self> {
        if weights.is_empty() {
            return None;
        }
        Some(Self { weights, bias })
    }

    /// Returns grounded probability in [0, 1] given the per-head
    /// ratio feature vector.
    pub fn grounded_prob(&self, ratios: &[f32]) -> f32 {
        if ratios.len() != self.weights.len() {
            return 0.5; // uninformative fallback
        }
        let dot: f32 = ratios
            .iter()
            .zip(self.weights.iter())
            .map(|(r, w)| r * w)
            .sum();
        let z = dot + self.bias;
        1.0 / (1.0 + (-z).exp())
    }
}

/// Rerank a set of top-k candidate tokens given each candidate's
/// attention sums (one per candidate). Returns the index of the
/// candidate with the HIGHEST grounded probability — caller picks
/// `candidates[best_idx]` instead of the model's argmax when the
/// LM's argmax has substantially lower grounded prob.
///
/// Returns `None` when the candidate list is empty or the
/// classifier disagrees by less than `min_gap` (no rerank).
pub fn rerank(
    candidates: &[u32],
    per_candidate_sums: &[AttentionSums],
    classifier: &GroundedClassifier,
    min_gap: f32,
) -> Option<usize> {
    if candidates.is_empty() || per_candidate_sums.is_empty() {
        return None;
    }
    if candidates.len() != per_candidate_sums.len() {
        return None;
    }
    let scores: Vec<f32> = per_candidate_sums
        .iter()
        .map(|s| classifier.grounded_prob(&s.ratios()))
        .collect();
    let mut best = 0usize;
    let mut best_score = scores[0];
    for (i, &s) in scores.iter().enumerate() {
        if s > best_score {
            best_score = s;
            best = i;
        }
    }
    // Argmax candidate is index 0 (caller's convention).
    if best == 0 {
        return None;
    }
    let baseline = scores[0];
    if best_score - baseline < min_gap {
        return None;
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratios_in_unit_interval() {
        let s = AttentionSums {
            lookback_sums: vec![0.3, 0.7, 0.0],
            rest_sums: vec![0.7, 0.3, 1.0],
        };
        let r = s.ratios();
        assert_eq!(r.len(), 3);
        assert!((r[0] - 0.3).abs() < 1e-5);
        assert!((r[1] - 0.7).abs() < 1e-5);
        assert!((r[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn ratios_returns_empty_on_dim_mismatch() {
        let s = AttentionSums {
            lookback_sums: vec![0.5, 0.5],
            rest_sums: vec![0.5],
        };
        assert!(s.ratios().is_empty());
    }

    #[test]
    fn grounded_prob_sigmoid_range() {
        let c = GroundedClassifier::new(vec![1.0, 1.0, 1.0], 0.0).unwrap();
        let p = c.grounded_prob(&[0.5, 0.5, 0.5]);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn rerank_picks_best_when_above_threshold() {
        let c = GroundedClassifier::new(vec![10.0], 0.0).unwrap();
        let s_lo = AttentionSums {
            lookback_sums: vec![0.1],
            rest_sums: vec![0.9],
        };
        let s_hi = AttentionSums {
            lookback_sums: vec![0.9],
            rest_sums: vec![0.1],
        };
        let candidates = vec![100u32, 200];
        // candidates[1] has high grounded score; should be preferred.
        let picked = rerank(&candidates, &[s_lo, s_hi], &c, 0.1);
        assert_eq!(picked, Some(1));
    }

    #[test]
    fn rerank_returns_none_when_argmax_already_best() {
        let c = GroundedClassifier::new(vec![10.0], 0.0).unwrap();
        let s_hi = AttentionSums {
            lookback_sums: vec![0.9],
            rest_sums: vec![0.1],
        };
        let s_lo = AttentionSums {
            lookback_sums: vec![0.1],
            rest_sums: vec![0.9],
        };
        let picked = rerank(&[100, 200], &[s_hi, s_lo], &c, 0.1);
        assert!(picked.is_none(), "no rerank needed when argmax is grounded");
    }
}
