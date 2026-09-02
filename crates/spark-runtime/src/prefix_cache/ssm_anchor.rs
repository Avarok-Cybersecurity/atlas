// SPDX-License-Identifier: AGPL-3.0-only

//! [`SsmAnchor`] — the Marconi SSM-snapshot half of a [`PrefixMatch`],
//! addressable independently of the KV radix walk.
//!
//! `PrefixCache::lookup` returns the DEEPEST snapshot at or below the matched
//! prefix. When the restore site declines that anchor — the exact full-prompt
//! leaf is bypassed by default (`prefill_b/prefix_lookup.rs`), and a finish
//! leaf never carries the stashed hidden the exact shortcut needs — the
//! caller re-anchors on the deepest snapshot STRICTLY below the prompt via
//! `PrefixCache::lookup_ssm_anchor` instead of falling to a full recompute.
//! Split out of `prefix_cache.rs` to keep it under the repo's 500-LoC cap.

use super::PrefixMatch;

/// An SSM snapshot anchor: where a matched Marconi state lives and how many
/// prompt tokens it covers. Mirrors the `ssm_*` fields of [`PrefixMatch`]
/// exactly (same resident-vs-tier encoding), so it can be read out of and
/// written back into a match without any other field changing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SsmAnchor {
    /// Resident HBM snapshot slot (`PrefixMatch::ssm_snapshot`).
    pub snapshot: Option<usize>,
    /// Depth of `snapshot` (`PrefixMatch::ssm_snapshot_tokens`).
    pub snapshot_tokens: usize,
    /// Spill-tier key when the anchor is not resident
    /// (`PrefixMatch::ssm_snapshot_tier_key`).
    pub tier_key: Option<u64>,
    /// Depth of `tier_key` (`PrefixMatch::ssm_snapshot_tier_tokens`).
    pub tier_tokens: usize,
    /// Whether the anchor is a TAIL snapshot (`PrefixMatch::ssm_snapshot_is_tail`).
    pub is_tail: bool,
}

impl SsmAnchor {
    /// No anchor at all (the `PrefixMatch::empty()` encoding).
    pub const NONE: SsmAnchor = SsmAnchor {
        snapshot: None,
        snapshot_tokens: 0,
        tier_key: None,
        tier_tokens: 0,
        is_tail: false,
    };

    /// Whether a snapshot (resident or spilled) was found.
    pub fn is_some(&self) -> bool {
        self.snapshot.is_some() || self.tier_key.is_some()
    }

    /// Token depth of the anchor, folding the resident-vs-tier encoding the
    /// same way `ssm_fault_in::eff_ssm_snapshot` does: a resident hit reports
    /// `snapshot_tokens`, a spilled one `tier_tokens`, no anchor `0`.
    pub fn depth(&self) -> usize {
        if self.snapshot.is_some() {
            self.snapshot_tokens
        } else if self.tier_key.is_some() {
            self.tier_tokens
        } else {
            0
        }
    }
}

impl PrefixMatch {
    /// The SSM snapshot half of this match.
    pub fn ssm_anchor(&self) -> SsmAnchor {
        SsmAnchor {
            snapshot: self.ssm_snapshot,
            snapshot_tokens: self.ssm_snapshot_tokens,
            tier_key: self.ssm_snapshot_tier_key,
            tier_tokens: self.ssm_snapshot_tier_tokens,
            is_tail: self.ssm_snapshot_is_tail,
        }
    }

    /// Replace the SSM snapshot half of this match; the KV half (matched
    /// blocks / tokens / disk ids) is untouched.
    pub fn set_ssm_anchor(&mut self, anchor: SsmAnchor) {
        self.ssm_snapshot = anchor.snapshot;
        self.ssm_snapshot_tokens = anchor.snapshot_tokens;
        self.ssm_snapshot_tier_key = anchor.tier_key;
        self.ssm_snapshot_tier_tokens = anchor.tier_tokens;
        self.ssm_snapshot_is_tail = anchor.is_tail;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_depth_folds_resident_vs_tier() {
        assert_eq!(SsmAnchor::NONE.depth(), 0);
        assert!(!SsmAnchor::NONE.is_some());
        let resident = SsmAnchor {
            snapshot: Some(7),
            snapshot_tokens: 3264,
            ..SsmAnchor::NONE
        };
        assert_eq!(resident.depth(), 3264);
        let spilled = SsmAnchor {
            tier_key: Some(0xabc),
            tier_tokens: 3200,
            ..SsmAnchor::NONE
        };
        assert!(spilled.is_some());
        assert_eq!(spilled.depth(), 3200);
    }

    #[test]
    fn set_anchor_round_trips_and_leaves_kv_half_alone() {
        let mut m = PrefixMatch::empty();
        m.matched_tokens = 3286;
        m.matched_blocks = vec![1, 2, 3];
        let a = SsmAnchor {
            snapshot: Some(3),
            snapshot_tokens: 3264,
            is_tail: true,
            ..SsmAnchor::NONE
        };
        m.set_ssm_anchor(a);
        assert_eq!(m.ssm_anchor(), a);
        assert_eq!(m.ssm_snapshot, Some(3));
        assert_eq!(m.ssm_snapshot_tokens, 3264);
        assert!(m.ssm_snapshot_is_tail);
        assert_eq!(m.matched_tokens, 3286);
        assert_eq!(m.matched_blocks, vec![1, 2, 3]);
        m.set_ssm_anchor(SsmAnchor::NONE);
        assert_eq!(m.ssm_anchor(), SsmAnchor::NONE);
        assert_eq!(m.matched_tokens, 3286);
    }
}
