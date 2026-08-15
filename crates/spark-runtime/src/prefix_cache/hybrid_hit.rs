// SPDX-License-Identifier: AGPL-3.0-only

//! Hybrid/GDN prefix-cache hit definition.
//!
//! KV radix blocks and the SSM/GDN snapshot index are separate objects.
//! A logged prefix-cache hit on a hybrid model is valid only when both
//! exist at the same token count. A KV-only match is a miss: the serve
//! path cannot reuse those blocks without a restorable recurrent state,
//! and calling that a hit then recomputing all KV is a lie.

use super::PrefixMatch;

impl PrefixMatch {
    /// Token count at which this match is a hybrid hit, or `None` (miss).
    ///
    /// A hit requires KV blocks **and** a restorable SSM snapshot (resident
    /// or spilled) at the same length. Snapshots deeper than the KV match
    /// cannot restore — that is the #353 block-alignment edge (match lands
    /// one block short of the saved snapshot) and is a miss, not a hit.
    /// A shallower block-aligned snapshot is a hit at that shallower length;
    /// the caller must trim KV to that length so the two cannot diverge.
    ///
    /// `block_size == 0` fails closed (`None`).
    pub fn paired_ssm_tokens(&self, block_size: usize) -> Option<usize> {
        if block_size == 0 || self.matched_tokens == 0 || self.matched_blocks.is_empty() {
            return None;
        }
        let snap = if self.ssm_snapshot.is_some() {
            self.ssm_snapshot_tokens
        } else if self.ssm_snapshot_tier_key.is_some() {
            self.ssm_snapshot_tier_tokens
        } else {
            0
        };
        if snap == 0 || snap > self.matched_tokens {
            return None;
        }
        if snap == self.matched_tokens {
            return Some(snap);
        }
        if snap.is_multiple_of(block_size) {
            return Some(snap);
        }
        None
    }

    /// Truncate KV fields so `matched_tokens == paired`. Snapshot fields
    /// already describe `paired` when [`Self::paired_ssm_tokens`] returned it.
    pub fn trim_to_paired_len(&mut self, paired: usize, block_size: usize) {
        let n_blocks = paired.div_ceil(block_size);
        self.matched_blocks.truncate(n_blocks);
        if !self.matched_disk_block_ids.is_empty() {
            self.matched_disk_block_ids.truncate(n_blocks);
        }
        self.matched_tokens = paired;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv_only(tokens: usize, block_size: usize) -> PrefixMatch {
        PrefixMatch {
            matched_blocks: vec![1; tokens / block_size],
            matched_disk_block_ids: Vec::new(),
            matched_tokens: tokens,
            ssm_snapshot: None,
            ssm_snapshot_tokens: 0,
            ssm_snapshot_tier_key: None,
            ssm_snapshot_tier_tokens: 0,
            ssm_snapshot_is_tail: false,
        }
    }

    fn paired(tokens: usize, block_size: usize, slot: usize) -> PrefixMatch {
        let mut m = kv_only(tokens, block_size);
        m.ssm_snapshot = Some(slot);
        m.ssm_snapshot_tokens = tokens;
        m
    }

    /// (a) KV hit without SSM is a miss — the old serve path logged a hit
    /// then recomputed all KV. This classification is the product invariant.
    #[test]
    fn kv_without_ssm_is_a_miss() {
        let m = kv_only(64, 16);
        assert_eq!(m.paired_ssm_tokens(16), None);
        assert_eq!(
            m.matched_tokens, 64,
            "classifier must not mutate the raw match"
        );
    }

    /// (b) Paired snapshot at the KV length is a hit — restore that length,
    /// do not recompute those KV blocks.
    #[test]
    fn paired_snapshot_is_a_hit_at_matched_length() {
        let m = paired(64, 16, 7);
        assert_eq!(m.paired_ssm_tokens(16), Some(64));
    }

    #[test]
    fn block_alignment_edge_is_a_miss() {
        // KV floored to 32; snapshot saved at 48 (one block past the match).
        // lookup_tiered already drops token_count > matched; if a caller
        // still sees this shape, it is a miss, not a lying hit.
        let mut m = kv_only(32, 16);
        m.ssm_snapshot = Some(9);
        m.ssm_snapshot_tokens = 48;
        assert_eq!(m.paired_ssm_tokens(16), None);
    }

    #[test]
    fn shallower_block_aligned_snapshot_trims_to_that_length() {
        let mut m = kv_only(64, 16);
        m.ssm_snapshot = Some(3);
        m.ssm_snapshot_tokens = 32;
        assert_eq!(m.paired_ssm_tokens(16), Some(32));
        m.trim_to_paired_len(32, 16);
        assert_eq!(m.matched_tokens, 32);
        assert_eq!(m.matched_blocks.len(), 2);
        assert_eq!(m.ssm_snapshot_tokens, 32);
    }

    #[test]
    fn non_aligned_shallower_snapshot_is_a_miss() {
        let mut m = kv_only(64, 16);
        m.ssm_snapshot = Some(3);
        m.ssm_snapshot_tokens = 40;
        assert_eq!(m.paired_ssm_tokens(16), None);
    }

    #[test]
    fn spilled_anchor_pairs_at_tier_depth() {
        let mut m = kv_only(64, 16);
        m.ssm_snapshot_tier_key = Some(0xabc);
        m.ssm_snapshot_tier_tokens = 64;
        assert_eq!(m.paired_ssm_tokens(16), Some(64));
    }

    #[test]
    fn zero_block_size_fails_closed() {
        let m = paired(16, 16, 1);
        assert_eq!(m.paired_ssm_tokens(0), None);
    }

    #[test]
    fn empty_match_is_a_miss() {
        assert_eq!(PrefixMatch::empty().paired_ssm_tokens(16), None);
    }
}
