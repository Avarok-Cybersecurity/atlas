// SPDX-License-Identifier: AGPL-3.0-only

//! Hallucination-probe side-signal primitive (C.1, 2026-04-25).
//!
//! Reference: Real-Time Hallucinated Entities (Nanda et al.,
//! arXiv:2509.03531, Sep 2025) — per-token linear probe on the last-
//! layer residual stream classifies whether the model is fabricating
//! an entity. AUC 0.90 on 70B class. Reference impl:
//! `github.com/obalcells/hallucination_probes`.
//!
//! ## Scope
//!
//! Atlas's residual stream lives on the GPU; exposing it for a
//! per-token linear probe needs (a) the kernel to dump the last-
//! layer hidden state to host every step, OR (b) the probe weights
//! cooperate with a fused GPU kernel. Either is a kernel change.
//!
//! This module ships the **runtime classifier interface** plus a
//! production-ready linear-probe evaluator. The kernel cooperation
//! to extract the hidden state is documented as a future activation.
//!
//! Per-token AUC ≈ 0.90 is high but not perfect; ship as a
//! READ-ONLY side signal (`x-atlas-confidence` SSE channel) that
//! clients can use for display or down-stream gating, NOT as a
//! generation-time mask. False positives at decode would harm
//! quality; false positives in a confidence display do not.

/// Linear probe weights produced by offline training on a labeled
/// hallucinated/grounded dataset. Length must equal the model's
/// hidden dim (e.g. 2048 for Qwen3.5).
#[derive(Debug, Clone)]
pub struct LinearProbe {
    weights: Vec<f32>,
    bias: f32,
}

impl LinearProbe {
    /// Build from offline-trained weights. Returns `None` if
    /// `weights` is empty.
    pub fn new(weights: Vec<f32>, bias: f32) -> Option<Self> {
        if weights.is_empty() {
            return None;
        }
        Some(Self { weights, bias })
    }

    /// Raw logit = w · h + b. Hidden vector length must match
    /// weights length.
    pub fn logit(&self, hidden: &[f32]) -> f32 {
        if hidden.len() != self.weights.len() {
            return self.bias;
        }
        let dot: f32 = hidden
            .iter()
            .zip(self.weights.iter())
            .map(|(h, w)| h * w)
            .sum();
        dot + self.bias
    }

    /// Hallucination probability via sigmoid(logit). Caller
    /// thresholds at e.g. 0.5 for binary classification or streams
    /// the float as a confidence signal.
    pub fn probability(&self, hidden: &[f32]) -> f32 {
        let z = self.logit(hidden);
        1.0 / (1.0 + (-z).exp())
    }

    /// Hidden dim that this probe expects.
    pub fn hidden_dim(&self) -> usize {
        self.weights.len()
    }
}

/// Per-token confidence record streamed to clients on the
/// `x-atlas-confidence` SSE channel.
#[derive(Debug, Clone)]
pub struct ConfidenceSample {
    pub token_id: u32,
    /// Hallucination probability in [0, 1]. Higher = more likely
    /// fabricated.
    pub p_halluc: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_weights_rejected() {
        let p = LinearProbe::new(Vec::new(), 0.0);
        assert!(p.is_none());
    }

    #[test]
    fn logit_computes_dot_plus_bias() {
        let p = LinearProbe::new(vec![1.0, 2.0, -1.0], 0.5).unwrap();
        // h · w = 1·1 + 0·2 + 1·(-1) = 0; +0.5 bias → 0.5
        let z = p.logit(&[1.0, 0.0, 1.0]);
        assert!((z - 0.5).abs() < 1e-5);
    }

    #[test]
    fn dim_mismatch_returns_bias_only() {
        let p = LinearProbe::new(vec![1.0, 2.0], 0.7).unwrap();
        let z = p.logit(&[1.0, 2.0, 3.0]);
        assert!((z - 0.7).abs() < 1e-5);
    }

    #[test]
    fn probability_in_unit_interval() {
        let p = LinearProbe::new(vec![1.0, -1.0, 2.0], 0.0).unwrap();
        let pr = p.probability(&[0.5, 0.3, 0.7]);
        assert!((0.0..=1.0).contains(&pr));
    }

    #[test]
    fn extreme_logit_saturates() {
        let p = LinearProbe::new(vec![100.0, 100.0], 0.0).unwrap();
        let pr_high = p.probability(&[1.0, 1.0]);
        assert!(pr_high > 0.99);
        let pr_low = p.probability(&[-1.0, -1.0]);
        assert!(pr_low < 0.01);
    }

    #[test]
    fn hidden_dim_reports_weights_length() {
        let p = LinearProbe::new(vec![0.0; 2048], 0.0).unwrap();
        assert_eq!(p.hidden_dim(), 2048);
    }
}
