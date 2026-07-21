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
        //
        // READ THIS BEFORE CONCLUDING THE SELECTOR IS LEAVING DEPTH ON THE TABLE
        // (2026-07-21, dgx1): a serve log records a snapshot's POSITION but not
        // its CONTENT, so grepping saves and hits and asking "did a save at a
        // deeper position exist?" scores ~100% of warm hits as mis-selected —
        // the previous turn's own end-of-prompt leaf and finish leaf ALWAYS sit
        // between this turn's restore point and its prompt end. They cover the
        // model's generated continuation, which the next prompt does not
        // reproduce (re-rendered history + the chat template's generation-only
        // suffix), so their prefix hash does not match and they are correctly
        // rejected below. Classified in place with `ATLAS_SNAP_LOOKUP_DBG`, the
        // measured count of selectable-but-not-selected entries is ZERO, and the
        // selected depth is exactly `matched - bs` every time (the tail-checkpoint
        // split in prefill_b.rs puts an anchor there for precisely this purpose).
        // Use the dump, not the log positions.
        let mut best: Option<usize> = None;
        let mut best_depth = 0usize;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.token_count > matched_tokens {
                continue;
            }
            if session_hash != 0 && entry.session_hash != 0 && entry.session_hash != session_hash {
                continue;
            }
            // TAIL snapshots bleed past the exact prefix — byte-safe ONLY for
            // the same non-zero session (ported from `lookup`; the serving
            // path previously had NO is_tail gate, so a sessionless lookup
            // could restore another session's tail — the exact cross-request
            // corruption the session-gate exists to prevent).
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
            })
        } else {
            self.stats.recompute_tokens_on_miss += matched_tokens as u64;
            None
        };
        if std::env::var_os("ATLAS_SNAP_LOOKUP_DBG").is_some() {
            self.log_lookup_candidates(
                tokens,
                matched_tokens,
                session_hash,
                adapter_id,
                result.as_ref().map(|m| m.token_count),
            );
        }
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
