// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the picker ([`pick`]) that turns one sequence's view-block scores into
//! the blocks its bounded view keeps, the diagnostic recall mass ([`recall_mass`]) and the
//! block-to-page expansion ([`expand_to_pages`]). Pure host functions, deterministic for
//! equal inputs.
//!
//! Priority, highest first, each class taken until the budget runs out:
//! 1. the last block (the partially filled tail, always the newest);
//! 2. the `recent_blocks` newest blocks, newest first;
//! 3. the `sink_blocks` oldest blocks, oldest first;
//! 4. the mandatory blocks (the query span's blocks), newest first;
//! 5. every other block by descending score, ties toward the newer block (NaN ranks last).
//!
//! The output is the kept block indices in ascending (chronological) order and never holds
//! more than `budget_blocks` entries. A budget of at least `n_blocks` keeps every block,
//! which is the identity view.

/// What the picker keeps besides the scored blocks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PickCfg {
    /// Most view blocks the view may hold.
    pub budget_blocks: usize,
    /// Oldest blocks always kept (attention sinks).
    pub sink_blocks: usize,
    /// Newest blocks always kept (the tail block counts as one of them).
    pub recent_blocks: usize,
    /// Block indices always kept (the query span). Out-of-range entries are ignored.
    pub mandatory: Vec<u32>,
}

impl PickCfg {
    /// Token-denominated limits to block counts: the budget rounds DOWN (the view never
    /// exceeds `budget_tokens`), sink and recent round UP (they cover at least what was
    /// asked).
    pub fn from_tokens(
        budget_tokens: usize,
        sink_tokens: usize,
        recent_tokens: usize,
        block_tokens: usize,
        mandatory: Vec<u32>,
    ) -> Self {
        let bt = block_tokens.max(1);
        Self {
            budget_blocks: budget_tokens / bt,
            sink_blocks: sink_tokens.div_ceil(bt),
            recent_blocks: recent_tokens.div_ceil(bt),
            mandatory,
        }
    }

    /// Mandatory blocks covering the token span `[start, end)` (the query span).
    pub fn span_blocks(start: usize, end: usize, block_tokens: usize) -> Vec<u32> {
        let bt = block_tokens.max(1);
        if end <= start {
            return Vec::new();
        }
        (start / bt..end.div_ceil(bt))
            .filter_map(|b| u32::try_from(b).ok())
            .collect()
    }
}

/// The blocks kept so far and the budget left.
struct Selection {
    kept: Vec<bool>,
    left: usize,
}

impl Selection {
    fn take(&mut self, b: usize) {
        if self.left > 0 && b < self.kept.len() && !self.kept[b] {
            self.kept[b] = true;
            self.left -= 1;
        }
    }
}

/// Order key of a score: NaN ranks below every number.
fn rank(s: f32) -> f32 {
    if s.is_nan() { f32::NEG_INFINITY } else { s }
}

/// The kept view blocks, ascending. `scores` holds at least `n_blocks` entries (extra
/// entries are ignored; missing ones rank as NaN).
pub fn pick(scores: &[f32], n_blocks: usize, cfg: &PickCfg) -> Vec<u32> {
    if n_blocks == 0 || cfg.budget_blocks == 0 {
        return Vec::new();
    }
    if cfg.budget_blocks >= n_blocks {
        return (0..n_blocks)
            .filter_map(|b| u32::try_from(b).ok())
            .collect();
    }
    let mut sel = Selection {
        kept: vec![false; n_blocks],
        left: cfg.budget_blocks,
    };
    sel.take(n_blocks - 1);
    let recent = cfg.recent_blocks.min(n_blocks);
    for b in (n_blocks - recent..n_blocks).rev() {
        sel.take(b);
    }
    for b in 0..cfg.sink_blocks.min(n_blocks) {
        sel.take(b);
    }
    let mut mandatory: Vec<usize> = cfg.mandatory.iter().map(|&b| b as usize).collect();
    mandatory.sort_unstable_by(|a, b| b.cmp(a));
    mandatory.dedup();
    for b in mandatory {
        sel.take(b);
    }
    let mut order: Vec<usize> = (0..n_blocks).filter(|&b| !sel.kept[b]).collect();
    order.sort_by(|&a, &b| {
        let (sa, sb) = (
            rank(scores.get(a).copied().unwrap_or(f32::NAN)),
            rank(scores.get(b).copied().unwrap_or(f32::NAN)),
        );
        sb.total_cmp(&sa).then(b.cmp(&a))
    });
    for b in order {
        if sel.left == 0 {
            break;
        }
        sel.take(b);
    }
    sel.kept
        .iter()
        .enumerate()
        .filter(|&(_, &k)| k)
        .filter_map(|(b, _)| u32::try_from(b).ok())
        .collect()
}

/// Fraction of the (non-negative) score mass inside `picked`. NaN and negative scores
/// count as 0; 1.0 when there is no mass at all (nothing could be lost).
pub fn recall_mass(scores: &[f32], picked: &[u32]) -> f64 {
    let w = |s: f32| {
        if s.is_nan() || s < 0.0 {
            0.0
        } else {
            f64::from(s)
        }
    };
    let total: f64 = scores.iter().map(|&s| w(s)).sum();
    if total <= 0.0 {
        return 1.0;
    }
    let mut seen = vec![false; scores.len()];
    let mut kept = 0.0f64;
    for &b in picked {
        let b = b as usize;
        if b < scores.len() && !seen[b] {
            seen[b] = true;
            kept += w(scores[b]);
        }
    }
    (kept / total).min(1.0)
}

/// The page indices (into the sequence's page list) of the picked view blocks, ascending:
/// block `b` covers pages `b * pages_per_block .. min((b + 1) * pages_per_block, n_pages)`.
pub fn expand_to_pages(picked: &[u32], pages_per_block: usize, n_pages: usize) -> Vec<usize> {
    let mut blocks: Vec<usize> = picked.iter().map(|&b| b as usize).collect();
    blocks.sort_unstable();
    blocks.dedup();
    let ppb = pages_per_block.max(1);
    blocks
        .into_iter()
        .flat_map(|b| (b * ppb).min(n_pages)..((b + 1) * ppb).min(n_pages))
        .collect()
}
