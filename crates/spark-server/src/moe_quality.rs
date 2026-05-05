// SPDX-License-Identifier: AGPL-3.0-only

//! MoE quality interventions: LASER routing rebalance + MoE-Spec
//! verification-time expert budgeting (B.5, 2026-04-25).
//!
//! References:
//!  - LASER: arXiv:2510.03293 — score-distribution-aware top-k
//!    expansion. Sharply skewed routing layers stay top-k; flat
//!    layers expand to least-loaded experts. Reduces expert
//!    imbalance ~48% at <0.02 accuracy loss on Mixtral / DeepSeek-MoE.
//!  - MoE-Spec: arXiv:2602.16052 — verification-time expert budgeting:
//!    cap experts loaded per layer to top contributors, drop the
//!    long tail.
//!
//! ## Scope
//!
//! Atlas's actual MoE forward kernel lives in `atlas-kernels`, not
//! here. This module ships the **policy primitives** that decide:
//!
//!   1. Given a router's softmax distribution and the configured
//!      top-k, should we expand to a wider set of experts? (LASER)
//!   2. Given a list of expert indices and their per-token weights,
//!      what's the minimal subset whose combined weight covers
//!      `coverage` of the total? (MoE-Spec budgeting)
//!
//! Production wiring requires the kernel to call these helpers
//! before the expert dispatch and respect the returned subsets.

/// Compute Shannon entropy of a router distribution. Used by LASER
/// to decide whether the layer is "sharply skewed" (low entropy →
/// stay top-k) or "flat" (high entropy → expand the candidate pool).
///
/// `scores` is the softmax output (probabilities summing to ~1).
pub fn router_entropy(scores: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in scores {
        if p > 1e-10 {
            h -= p * p.ln();
        }
    }
    h
}

/// LASER top-k expansion decision. Returns the number of experts to
/// route to:
///   - When entropy is below `entropy_threshold`, return the original
///     top-k (sharply skewed → stay tight).
///   - When entropy is above the threshold, return `top_k_expanded`
///     (flat → widen the pool to balance load).
///
/// Recommended threshold (per LASER paper): 0.7 * log(num_experts).
pub fn laser_top_k(
    router_h: f32,
    num_experts: usize,
    base_top_k: usize,
    expanded_top_k: usize,
) -> usize {
    let h_max = (num_experts as f32).ln();
    let threshold = 0.7 * h_max;
    if router_h >= threshold {
        expanded_top_k.min(num_experts)
    } else {
        base_top_k.min(num_experts)
    }
}

/// MoE-Spec expert budgeting: from a list of (expert_id, weight)
/// pairs, return the smallest prefix (sorted by descending weight)
/// whose cumulative weight covers `coverage` (e.g., 0.95 = top
/// experts that together account for ≥95% of the routing mass).
///
/// Drops the long tail of low-weight experts at MTP verify time —
/// reduces the per-step expert-load count without measurably hurting
/// accept rate.
pub fn budget_experts(weights: &[(u32, f32)], coverage: f32) -> Vec<(u32, f32)> {
    if weights.is_empty() {
        return Vec::new();
    }
    let coverage = coverage.clamp(0.0, 1.0);
    let total: f32 = weights.iter().map(|(_, w)| *w).sum();
    if total <= 0.0 {
        return Vec::new();
    }
    let target = total * coverage;
    let mut sorted: Vec<(u32, f32)> = weights.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0f32;
    let mut out: Vec<(u32, f32)> = Vec::with_capacity(sorted.len());
    for (eid, w) in sorted {
        out.push((eid, w));
        cum += w;
        if cum >= target {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_uniform_equals_log_n() {
        let n = 8;
        let p = 1.0 / n as f32;
        let probs = vec![p; n];
        let h = router_entropy(&probs);
        assert!((h - (n as f32).ln()).abs() < 1e-3);
    }

    #[test]
    fn entropy_one_hot_is_zero() {
        let probs = vec![1.0, 0.0, 0.0, 0.0];
        let h = router_entropy(&probs);
        assert!(h < 1e-3);
    }

    #[test]
    fn laser_stays_topk_on_sharp_distribution() {
        // Sharply skewed: most mass on one expert.
        let probs = vec![0.95, 0.03, 0.01, 0.01];
        let h = router_entropy(&probs);
        let k = laser_top_k(h, 4, 2, 4);
        assert_eq!(k, 2, "sharp distribution → stay top-2");
    }

    #[test]
    fn laser_expands_on_flat_distribution() {
        // Near-uniform → entropy near max.
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let h = router_entropy(&probs);
        let k = laser_top_k(h, 4, 2, 4);
        assert_eq!(k, 4, "flat distribution → expand to 4");
    }

    #[test]
    fn budget_experts_covers_target_mass() {
        let weights = vec![
            (1u32, 0.4),
            (2, 0.3),
            (3, 0.2),
            (4, 0.05),
            (5, 0.03),
            (6, 0.02),
        ];
        let kept = budget_experts(&weights, 0.9);
        // Top-3 cover 0.9 mass — kept length should be ≤ 3.
        assert!(kept.len() <= 3, "got {} experts: {:?}", kept.len(), kept);
        let cum: f32 = kept.iter().map(|(_, w)| *w).sum();
        assert!(cum >= 0.9 - 1e-3);
    }

    #[test]
    fn budget_experts_full_coverage_returns_all() {
        let weights = vec![(1u32, 0.5), (2, 0.3), (3, 0.2)];
        let kept = budget_experts(&weights, 1.0);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn budget_experts_empty_input_returns_empty() {
        let kept = budget_experts(&[], 0.95);
        assert!(kept.is_empty());
    }
}
