// SPDX-License-Identifier: AGPL-3.0-only

//! Aggregate SSM-snapshot cache telemetry (Phase 0). Split from `snapshot.rs`
//! (file-size cap); summarised via `SsmSnapshotIndex::log_stats_if_due` when
//! `ATLAS_SSM_SNAP_STATS` is set. All counters are aggregate and off the hot
//! path's critical decisions — they only observe.

#[derive(Default, Clone, Copy)]
pub(super) struct SnapshotStats {
    /// Snapshots registered (new prefix inserted, not an overwrite).
    pub saves: u64,
    /// Prefix lookups attempted.
    pub lookups: u64,
    /// Lookups that restored *some* anchor (deep or shallow).
    pub hits: u64,
    /// Σ restored-anchor depth over hits — mean anchor = this / hits.
    pub anchor_depth_sum: u64,
    /// Σ (matched_tokens − anchor_depth) over hits: the SSM tokens that still
    /// had to be recomputed because the deep tail was not resident. This is the
    /// #278 metric ("mean recompute 4438→262 tok") and Phase 1's target.
    pub recompute_tokens_on_hit: u64,
    /// Σ matched_tokens over *misses* (no anchor at all → full recompute).
    pub recompute_tokens_on_miss: u64,
    /// Snapshot slots freed by `evict_lru` — a DROP (state discarded) on the
    /// default path; Phase 1 spills instead via `evict_to_tier`.
    pub evictions: u64,
    /// Phase 1b: entries moved HBM→Tier by `evict_to_tier` (spills, not drops).
    pub tier_spills: u64,
    /// Phase 1b: lookups whose deepest anchor was found in the tier (would have
    /// been a recompute pre-spill) — the converted loss.
    pub tier_hits: u64,
    /// Phase 1b: tier entries faulted back into HBM via `promote`.
    pub tier_fault_ins: u64,
}
