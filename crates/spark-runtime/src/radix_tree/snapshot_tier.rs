// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 1b spill tier — resident vs spilled state machine (`ATLAS_SSM_TIER`).
//! Split from `snapshot.rs` (file-size cap); same `SsmSnapshotIndex` impl.

use super::hash_token_prefix;
use super::snapshot::{SnapLoc, SnapMatch, SsmSnapshotIndex};

impl SsmSnapshotIndex {
    /// Spill victim selection (tier engaged). Marks the victim spilled, returns
    /// `(freed_slot, key)`. Entry stays in the index for `lookup_tiered` fault-in.
    pub(super) fn evict_to_tier(&mut self) -> Option<(usize, u64)> {
        if self.entries.is_empty() {
            return None;
        }
        let tail_protect = self.tail_lease_active();
        let idx = self.session_aware_victim(tail_protect, /*skip_tiered*/ true)?;
        let e = &mut self.entries[idx];
        e.tiered = true;
        let freed_slot = e.snapshot_id;
        let key = e.prefix_hash;
        self.stats.tier_spills += 1;
        self.evictions_since_lookup = self.evictions_since_lookup.saturating_add(1);
        Some((freed_slot, key))
    }

    /// **Tier-aware lookup** (used in place of `lookup` when the tier is on).
    /// Returns the deepest matching anchor and where its state lives, so the
    /// caller either restores from HBM or faults in from the tier. Feeds the
    /// same Phase-0 telemetry as `lookup`, plus `tier_hits`.
    pub(super) fn lookup_tiered(
        &mut self,
        tokens: &[u32],
        matched_tokens: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> Option<SnapMatch> {
        if session_hash != 0 {
            self.last_lookup_session = session_hash;
            self.evictions_since_lookup = 0;
        }
        // Deepest matching prefix across BOTH resident and spilled entries.
        let mut best: Option<usize> = None;
        let mut best_depth = 0usize;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.token_count > matched_tokens {
                continue;
            }
            // Session gate applies ONLY to is_tail entries. A tail's SSM state
            // bleeds past its advertised token_count (a partial block beyond the
            // exact prefix), so it is byte-safe only within the same non-zero
            // session. Exact snapshots and is_tail_sibling early captures are a
            // pure function of the verified token prefix — safe cross-session,
            // exactly like the KV radix (which shares blocks on token match with
            // no session gate). The old blanket gate rejected every otherwise-
            // valid non-tail anchor from a prior turn because session_hash =
            // hash(first min(len,1024) prompt tokens) is UNSTABLE across turns of
            // a growing conversation — the warm-TTFT-climb root cause.
            if entry.is_tail && (session_hash == 0 || entry.session_hash != session_hash) {
                continue;
            }
            if hash_token_prefix(tokens, entry.token_count, adapter_id) != entry.prefix_hash {
                continue;
            }
            if best.is_none() || entry.token_count > best_depth {
                best = Some(i);
                best_depth = entry.token_count;
            }
        }
        self.stats.lookups += 1;
        let result = if let Some(i) = best {
            self.access_counter += 1;
            let ac = self.access_counter;
            let e = &mut self.entries[i];
            e.last_access = ac;
            let tiered = e.tiered;
            let depth = e.token_count;
            let is_tail = e.is_tail;
            let loc = if tiered {
                SnapLoc::Tier(e.prefix_hash)
            } else {
                SnapLoc::Hbm(e.snapshot_id)
            };
            self.stats.hits += 1;
            self.stats.anchor_depth_sum += depth as u64;
            self.stats.recompute_tokens_on_hit += matched_tokens.saturating_sub(depth) as u64;
            if tiered {
                self.stats.tier_hits += 1;
            }
            Some(SnapMatch {
                token_count: depth,
                loc,
                is_tail,
            })
        } else {
            self.stats.recompute_tokens_on_miss += matched_tokens as u64;
            None
        };
        self.log_stats_if_due();
        result
    }

    /// **Promote** a spilled entry back to HBM after the caller faulted its
    /// bytes into `new_slot`. Flips `tiered → false` and re-homes `snapshot_id`.
    /// Returns `false` if `prefix_hash` is unknown (entry evicted meanwhile).
    pub(super) fn promote(&mut self, prefix_hash: u64, new_slot: usize) -> bool {
        for e in &mut self.entries {
            if e.prefix_hash == prefix_hash {
                e.tiered = false;
                e.snapshot_id = new_slot;
                self.stats.tier_fault_ins += 1;
                return true;
            }
        }
        false
    }
}
