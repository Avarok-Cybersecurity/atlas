// SPDX-License-Identifier: AGPL-3.0-only

//! SSM snapshot LRU index — independent of the token-radix structure.
//!
//! Snapshots are keyed by (session_hash, token_count, prefix_hash) so the
//! same prompt across requests can hit a cached SSM state without going
//! through the radix tree.

use super::hash_token_prefix;

pub(super) struct SnapshotEntry {
    snapshot_id: usize,
    session_hash: u64,
    token_count: usize,
    prefix_hash: u64,
    last_access: u64,
    /// Cumulative hits over the entry's lifetime — combined with
    /// `last_access` in eviction to approximate the forecast-based
    /// policy from the Marconi paper §4 (B.4, 2026-04-25). Hot
    /// prefixes (high hit count) survive longer than cold ones at
    /// the same age.
    hit_count: u32,
    /// Phase 1b — spill-not-drop location. `false` = resident in HBM at
    /// `snapshot_id`. `true` = spilled to the byte tier; `snapshot_id` is stale
    /// and the state is addressed by `prefix_hash` (the tier key). Always
    /// `false` when `ATLAS_SSM_TIER` is off, so the default path is unchanged.
    tiered: bool,
}

/// Where a matched snapshot's state currently lives (Phase 1b).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SnapLoc {
    /// Resident in the HBM snapshot pool at this slot — restore directly.
    Hbm(usize),
    /// Spilled to the byte tier — fault in by this key (the prefix hash) into a
    /// fresh HBM slot, then `promote`.
    Tier(u64),
}

/// A tier-aware snapshot match (Phase 1b): the deepest anchor for a prefix plus
/// where its state currently lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SnapMatch {
    pub token_count: usize,
    pub loc: SnapLoc,
}

pub(super) struct SsmSnapshotIndex {
    pub(super) entries: Vec<SnapshotEntry>,
    pub(super) access_counter: u64,
    /// Session of the most recent `lookup` — the live conversation. Its
    /// DEEPEST snapshot is the one its next warm turn will restore from, so
    /// `evict_lru` protects it (ATLAS_SSM_TAIL_PROTECT, ported from #278).
    /// Tracks the running tip: recomputed each eviction from `token_count`,
    /// never a pinned slot. Complements the session-aware victim ranking
    /// below — session-aware protects the live session vs *dormant* ones;
    /// tail-protect protects the deep tail *within* the live session (the
    /// single/dominant-conversation case session-freshness can't see).
    last_lookup_session: u64,
    /// Phase-0 measurement counters (ATLAS_SSM_SNAP_STATS). All aggregate,
    /// off the hot path's critical decisions — they only observe. The residual
    /// `recompute_tokens_on_hit` after tail-protect + a large pool is exactly
    /// what Phase 1 (spill-not-drop) converts from recompute → fault-in.
    stats: SnapshotStats,
}

/// Aggregate SSM-snapshot cache telemetry (Phase 0). Summarised via
/// `log_stats_if_due` when `ATLAS_SSM_SNAP_STATS` is set.
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

impl SsmSnapshotIndex {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            access_counter: 0,
            last_lookup_session: 0,
            stats: SnapshotStats::default(),
        }
    }

    pub(super) fn insert(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Option<usize> {
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash {
                let old = entry.snapshot_id;
                entry.snapshot_id = snapshot_id;
                entry.session_hash = session_hash;
                entry.token_count = token_count;
                // A fresh HBM save re-homes the prefix: it is resident again.
                entry.tiered = false;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return Some(old);
            }
        }
        self.access_counter += 1;
        self.stats.saves += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            hit_count: 0,
            tiered: false,
        });
        None
    }

    /// Find deepest snapshot matching session within matched_tokens range.
    /// Task #24: `adapter_id` is folded into the recomputed prefix hash so a
    /// snapshot registered under one adapter never matches another's lookup.
    ///
    /// Resident-only (skips tiered entries). The serving path uses the
    /// tier-aware `lookup_tiered`; this is retained as the reference for the
    /// pre-tier contract and exercised by focused unit tests.
    #[allow(dead_code)]
    pub(super) fn lookup(
        &mut self,
        tokens: &[u32],
        matched_tokens: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> Option<(usize, usize)> {
        // Track the live conversation so eviction can protect its deep tail
        // (ATLAS_SSM_TAIL_PROTECT).
        if session_hash != 0 {
            self.last_lookup_session = session_hash;
        }
        let mut best: Option<(usize, usize)> = None; // (snapshot_id, token_count)
        for entry in &mut self.entries {
            // Tiered entries hold no HBM slot — the non-tier `lookup` must never
            // hand back a spilled entry's stale slot. Tier-aware callers use
            // `lookup_tiered`. (No entry is ever tiered when ATLAS_SSM_TIER off.)
            if entry.tiered {
                continue;
            }
            if entry.token_count > matched_tokens {
                continue;
            }
            if session_hash != 0 && entry.session_hash != 0 && entry.session_hash != session_hash {
                continue;
            }
            let h = hash_token_prefix(tokens, entry.token_count, adapter_id);
            if h != entry.prefix_hash {
                continue;
            }
            if best.is_none() || entry.token_count > best.unwrap().1 {
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                entry.hit_count = entry.hit_count.saturating_add(1);
                best = Some((entry.snapshot_id, entry.token_count));
            }
        }
        // Phase-0 telemetry: hit-rate + recompute distance. `recompute` is the
        // SSM prefix that still had to be re-run because no deeper anchor was
        // resident — the exact loss tail-protect shrinks and Phase 1 removes.
        self.stats.lookups += 1;
        match best {
            Some((_, anchor)) => {
                self.stats.hits += 1;
                self.stats.anchor_depth_sum += anchor as u64;
                self.stats.recompute_tokens_on_hit += matched_tokens.saturating_sub(anchor) as u64;
            }
            None => {
                self.stats.recompute_tokens_on_miss += matched_tokens as u64;
            }
        }
        if std::env::var("ATLAS_SNAP_LOOKUP_DBG").is_ok() {
            let mut cands: Vec<usize> = self.entries.iter().map(|e| e.token_count).collect();
            cands.sort_unstable();
            tracing::info!(
                "snap-lookup: matched={matched_tokens} selected={:?} n_entries={} token_counts={:?}",
                best.map(|b| b.1),
                self.entries.len(),
                cands,
            );
        }
        self.log_stats_if_due();
        best
    }

    /// Emit an aggregate SSM-snapshot cache summary every 64 lookups when
    /// `ATLAS_SSM_SNAP_STATS` is set. Off-by-default and read-only, so it never
    /// perturbs serving; the line is the Phase-0 measurement surface (hit-rate,
    /// mean restore anchor, mean recompute tok/turn — the #278 metrics).
    fn log_stats_if_due(&self) {
        if !self.stats.lookups.is_multiple_of(64)
            || std::env::var_os("ATLAS_SSM_SNAP_STATS").is_none()
        {
            return;
        }
        let s = &self.stats;
        let hit_rate = s.hits as f64 / s.lookups.max(1) as f64;
        let mean_anchor = s.anchor_depth_sum as f64 / s.hits.max(1) as f64;
        let mean_recompute_hit = s.recompute_tokens_on_hit as f64 / s.hits.max(1) as f64;
        let tiered = self.entries.iter().filter(|e| e.tiered).count();
        tracing::info!(
            "ssm-snap-stats: lookups={} hits={} hit_rate={:.2} saves={} evictions(drops)={} \
             mean_anchor={:.0}tok mean_recompute_on_hit={:.0}tok recompute_on_miss={}tok \
             resident={} tiered={} tier_spills={} tier_hits={} tier_fault_ins={}",
            s.lookups,
            s.hits,
            hit_rate,
            s.saves,
            s.evictions,
            mean_anchor,
            mean_recompute_hit,
            s.recompute_tokens_on_miss,
            self.entries.len() - tiered,
            tiered,
            s.tier_spills,
            s.tier_hits,
            s.tier_fault_ins,
        );
    }

    pub(super) fn evict_lru(&mut self) -> Option<usize> {
        if self.entries.is_empty() {
            return None;
        }
        // Per-entry forecast score (B.4, Marconi paper §4): old AND cold first.
        // last_access * (1 + hit_count) — recent/hot survive. #155 fixed the
        // inverted (÷) form that evicted just-selected snapshots.
        let escore = |e: &SnapshotEntry| e.last_access.saturating_mul(1 + e.hit_count as u64);

        // SESSION-AWARE eviction (default ON; ATLAS_SNAP_EVICT_LEGACY=1 → old
        // per-entry policy). The agentic workload interleaves ~20 multi-turn
        // conversations; per-entry LRU evicts the active conversation's OWN deep
        // checkpoints whenever it goes briefly dormant (its unique deep snapshots
        // have hit_count=0 and a stale last_access vs another conversation's fresh
        // ones), so its next warm turn full-recomputes the SSM state (TTFT 1s→50s).
        // Fix: evict from the STALEST conversation first — rank by the session's
        // freshness (max last_access over its entries), so the active conversation's
        // ENTIRE deep checkpoint chain stays resident until every other (completed/
        // dormant) conversation is gone. Within the victim session, drop its lowest
        // forecast-score entry. This is "prefix caching like llama" for SSM state:
        // the live conversation never re-recomputes what it already computed.
        // Selecting a different victim is correctness-safe — restore re-validates
        // (session_hash + prefix_hash) before using any snapshot; eviction only
        // frees a slot.
        if std::env::var_os("ATLAS_SNAP_EVICT_LEGACY").is_none() {
            let tail_protect = self.last_lookup_session != 0
                && std::env::var_os("ATLAS_SSM_TAIL_PROTECT").is_some();
            // Skip tiered entries — they hold no HBM slot to free.
            let victim_idx = self.session_aware_victim(tail_protect, true)?;
            let entry = self.entries.swap_remove(victim_idx);
            self.stats.evictions += 1;
            return Some(entry.snapshot_id);
        }

        let mut victim_idx = None;
        let mut victim_score = u64::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.tiered {
                continue; // no HBM slot to free
            }
            let score = escore(entry);
            if score < victim_score {
                victim_score = score;
                victim_idx = Some(i);
            }
        }
        let entry = self.entries.swap_remove(victim_idx?);
        self.stats.evictions += 1;
        Some(entry.snapshot_id)
    }

    /// Pure victim selection for the session-aware eviction policy (default
    /// path). Returns the index into `self.entries` to drop. Split out so it
    /// is unit-testable without mutating process env.
    ///
    /// Ranking: evict the STALEST conversation first (by session freshness =
    /// max `last_access` over its entries), then the lowest per-entry forecast
    /// score within it — the live conversation's whole deep chain stays
    /// resident until every dormant conversation is gone.
    ///
    /// `tail_protect` (ATLAS_SSM_TAIL_PROTECT, ported from #278): additionally
    /// exempt the live conversation's DEEPEST snapshot — its next warm turn
    /// restores from it. Session-aware ranking alone only protects the live
    /// session vs *dormant* ones; within a single/dominant session it falls
    /// through to lowest-escore = the just-aged deep tail (hit_count=0), which
    /// the hot token-8192 anchor out-scores, so warm restore falls back to 8192
    /// and recomputes thousands of SSM tokens (~4400 tok, ~7.6s TTFT/turn).
    /// Exactly ONE entry is exempted, scoped to the active session, so any pool
    /// \>=2 always has a victim and never deadlocks. Correctness-safe: restore
    /// re-validates session_hash+prefix_hash+depth, so changing the victim
    /// cannot cause a wrong-position restore.
    ///
    /// `skip_tiered` (Phase 1b): ignore entries already spilled to the byte tier
    /// — they hold no HBM slot, so they are neither a valid drop victim (freeing
    /// their stale `snapshot_id` would double-free) nor a spill candidate.
    /// Returns `None` when no eligible entry exists (all spilled / empty).
    fn session_aware_victim(&self, tail_protect: bool, skip_tiered: bool) -> Option<usize> {
        let escore = |e: &SnapshotEntry| e.last_access.saturating_mul(1 + e.hit_count as u64);
        let eligible = |e: &SnapshotEntry| !(skip_tiered && e.tiered);

        // session freshness = max last_access among that session's eligible entries.
        let mut session_fresh: std::collections::HashMap<u64, u64> =
            std::collections::HashMap::with_capacity(self.entries.len());
        for e in self.entries.iter().filter(|e| eligible(e)) {
            let f = session_fresh.entry(e.session_hash).or_insert(0);
            if e.last_access > *f {
                *f = e.last_access;
            }
        }
        let n_eligible = self.entries.iter().filter(|e| eligible(e)).count();
        let protected_idx: Option<usize> = if tail_protect {
            self.entries
                .iter()
                .enumerate()
                .filter(|(_, e)| eligible(e) && e.session_hash == self.last_lookup_session)
                .max_by_key(|(_, e)| e.token_count)
                .map(|(i, _)| i)
        } else {
            None
        };
        // (stalest session first, then lowest entry score within it). Protected
        // bites only when >1 eligible entry remains, so a single-entry pool
        // (even if it is the protected tail) still yields a victim → no deadlock.
        let mut victim: Option<(usize, (u64, u64))> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if !eligible(e) {
                continue;
            }
            if Some(i) == protected_idx && n_eligible > 1 {
                continue; // never evict the live session's deepest tail
            }
            let sf = *session_fresh.get(&e.session_hash).unwrap_or(&0);
            let key = (sf, escore(e));
            if victim.is_none_or(|(_, vk)| key < vk) {
                victim = Some((i, key));
            }
        }
        victim.map(|(i, _)| i)
    }

    // ─────────────────────────── Phase 1b: spill tier ───────────────────────
    // Location state machine over the same `entries`: an entry is either
    // resident (`tiered == false`, state at `snapshot_id`) or spilled
    // (`tiered == true`, state at `prefix_hash` in the byte tier). Only reached
    // when the caller has ATLAS_SSM_TIER on; the default path never tiers.
    // Consumed via the `PrefixCache` trait (`evict_snapshot_to_tier`,
    // `promote_snapshot`) and `RadixTree::lookup`'s tier-aware sub-lookup.

    /// **Spill victim selection** (replaces `evict_lru`'s drop when the tier is
    /// engaged). Pick the same session-aware/tail-protected victim as the drop
    /// path but among HBM-resident entries only, mark it spilled, and return
    /// `(freed_slot, key)` so the caller moves its bytes to the tier and reuses
    /// the slot. The entry STAYS in the index (findable by `lookup_tiered`), so
    /// a later warm turn faults it back in instead of recomputing.
    /// `None` ⇒ nothing HBM-resident to free (all already spilled / empty).
    pub(super) fn evict_to_tier(&mut self) -> Option<(usize, u64)> {
        if self.entries.is_empty() {
            return None;
        }
        let tail_protect =
            self.last_lookup_session != 0 && std::env::var_os("ATLAS_SSM_TAIL_PROTECT").is_some();
        let idx = self.session_aware_victim(tail_protect, /*skip_tiered*/ true)?;
        let e = &mut self.entries[idx];
        e.tiered = true;
        let freed_slot = e.snapshot_id;
        let key = e.prefix_hash;
        self.stats.tier_spills += 1;
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
        }
        // Deepest matching prefix across BOTH resident and spilled entries.
        let mut best: Option<usize> = None;
        let mut best_depth = 0usize;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.token_count > matched_tokens {
                continue;
            }
            if session_hash != 0 && entry.session_hash != 0 && entry.session_hash != session_hash {
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
            e.hit_count = e.hit_count.saturating_add(1);
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

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
#[path = "tests/snapshot_index.rs"]
mod tests;
