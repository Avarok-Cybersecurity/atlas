// SPDX-License-Identifier: AGPL-3.0-only

//! Retrieval-head identification + KV-preservation primitive
//! (B.2, 2026-04-25).
//!
//! References:
//!  - DuoAttention / RazorAttention (arXiv:2407.15891)
//!  - Park et al., "Characterizing SSM and Hybrid LM Performance with
//!    Long Context" (arXiv:2507.12442) — shows ~7-8% of attention
//!    heads in hybrid models carry the recall capacity.
//!
//! In Qwen3.5-35B-A3B (30 GDN + 10 attention layers), only the 10
//! full-attention layers can do exact recall — and within those, a
//! small fraction of heads do the lifting. Identifying those heads
//! offline and exempting them from any KV compression / eviction is
//! the published path to >25K-context recall preservation in hybrids.
//!
//! ## Scope
//!
//! Ships the **per-layer-per-head retrieval-flag table** plus a
//! lookup function. The actual identification is done offline via a
//! calibration script (TBD: `scripts/calibrate_retrieval_heads.rs`)
//! that runs needle-in-haystack probes and measures per-head KL-
//! divergence between the full and ablated runs.
//!
//! Production wiring requires:
//!   1. Offline calibration → produces `retrieval_heads.bincode`
//!   2. Server loads at startup
//!   3. KV cache compression / eviction consults `is_retrieval_head`
//!      and skips compression for those (head, layer) pairs.

use std::collections::HashSet;

/// Identifies a specific (layer, head) pair in the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HeadId {
    pub layer: u16,
    pub head: u16,
}

/// Set of heads marked as retrieval-critical, loaded from offline
/// calibration. Lookup is O(1) via HashSet.
#[derive(Debug, Clone, Default)]
pub struct RetrievalHeadSet {
    heads: HashSet<HeadId>,
    /// Per-layer count for fast "any retrieval head in this layer?"
    /// queries during compression.
    layer_counts: Vec<u16>,
}

impl RetrievalHeadSet {
    /// Empty set — every head is treated as compressible. Default
    /// behaviour when no calibration has been done yet.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build from a calibration result. `num_layers` sizes the
    /// per-layer count array; pass 40 for Qwen3.5.
    pub fn new(heads: impl IntoIterator<Item = HeadId>, num_layers: usize) -> Self {
        let heads: HashSet<HeadId> = heads.into_iter().collect();
        let mut layer_counts = vec![0u16; num_layers];
        for h in &heads {
            if (h.layer as usize) < num_layers {
                layer_counts[h.layer as usize] += 1;
            }
        }
        Self {
            heads,
            layer_counts,
        }
    }

    /// True iff (layer, head) was identified as retrieval-critical.
    /// KV compression / eviction MUST skip this head.
    pub fn is_retrieval_head(&self, layer: u16, head: u16) -> bool {
        self.heads.contains(&HeadId { layer, head })
    }

    /// Number of retrieval heads in `layer`. Useful for planning
    /// per-layer compression budgets.
    pub fn count_in_layer(&self, layer: u16) -> u16 {
        self.layer_counts.get(layer as usize).copied().unwrap_or(0)
    }

    /// Total retrieval heads identified across all layers.
    pub fn total(&self) -> usize {
        self.heads.len()
    }

    /// True iff calibration has been done (set is non-empty).
    pub fn is_calibrated(&self) -> bool {
        !self.heads.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_set_is_not_calibrated() {
        let s = RetrievalHeadSet::empty();
        assert!(!s.is_calibrated());
        assert_eq!(s.total(), 0);
        assert!(!s.is_retrieval_head(0, 0));
    }

    #[test]
    fn lookup_after_construction() {
        let heads = vec![
            HeadId { layer: 5, head: 3 },
            HeadId { layer: 10, head: 7 },
            HeadId {
                layer: 10,
                head: 11,
            },
        ];
        let s = RetrievalHeadSet::new(heads, 40);
        assert!(s.is_calibrated());
        assert!(s.is_retrieval_head(5, 3));
        assert!(s.is_retrieval_head(10, 7));
        assert!(!s.is_retrieval_head(5, 4));
        assert_eq!(s.count_in_layer(10), 2);
        assert_eq!(s.count_in_layer(5), 1);
        assert_eq!(s.count_in_layer(0), 0);
        assert_eq!(s.total(), 3);
    }

    #[test]
    fn out_of_range_layer_returns_zero() {
        let heads = vec![HeadId { layer: 5, head: 3 }];
        let s = RetrievalHeadSet::new(heads, 40);
        assert_eq!(s.count_in_layer(100), 0, "out-of-range layer is safe");
    }
}
