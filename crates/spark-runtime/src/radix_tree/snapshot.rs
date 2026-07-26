// SPDX-License-Identifier: AGPL-3.0-only

//! SSM snapshot LRU index — independent of the token-radix structure.
//!
//! Snapshots are keyed by (session_hash, token_count, prefix_hash) so the
//! same prompt across requests can hit a cached SSM state without going
//! through the radix tree.

use super::hash_token_prefix;
use super::snapshot_stats::SnapshotStats;

pub(super) struct SnapshotEntry {
    pub(super) snapshot_id: usize,
    pub(super) session_hash: u64,
    pub(super) token_count: usize,
    pub(super) prefix_hash: u64,
    pub(super) last_access: u64,
    /// Phase 1b — spill-not-drop location. `false` = resident in HBM at
    /// `snapshot_id`. `true` = spilled to the byte tier; `snapshot_id` is stale
    /// and the state is addressed by `prefix_hash` (the tier key). Always
    /// `false` when `ATLAS_SSM_TIER` is off, so the default path is unchanged.
    pub(super) tiered: bool,
    /// True for the per-session TAIL snapshot (the restore point the next turn's
    /// block-floored `matched_tokens` looks up). Exactly one is kept per session.
    pub(super) is_tail: bool,
    /// True for the tail's EARLY sibling (the mid-chunk capture at `tb - bs`).
    /// Serves warm turns whose block-floored match lands one block below the
    /// tail (measured 2/7 of restores on the eviction rig). Exact-prefix keyed
    /// — NOT session-gated in lookups (safe cross-session, unlike `is_tail`) —
    /// but leased alongside the tail. At most one per session (swept together
    /// with the tail by `insert_tail`).
    pub(super) is_tail_sibling: bool,
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
    /// Session of the most recent `lookup`/`lookup_tiered` — the live
    /// conversation. Eviction leases that session's `is_tail` entry (the
    /// TRUE restore point the next warm turn's block-floored
    /// `matched_tokens` looks up) — NOT its deepest entry: the deepest is
    /// the finish leaf written at end-of-turn, which sits ABOVE
    /// `matched_tokens` and is unusable for restore (#278's original
    /// deepest-entry semantics protected exactly the wrong slot; measured
    /// 2026-07-20 eviction rig: 0 Marconi hits with it on OR off).
    /// Complements the session-aware victim ranking — session-aware
    /// protects the live session vs *dormant* ones; the tail lease protects
    /// the restore point *within* the live session under same-session or
    /// sessionless-churn pressure.
    pub(super) last_lookup_session: u64,
    /// Evictions since the last session-latching lookup. The lease is a
    /// LEASE, not a pin: if the leased session never looks up again (it
    /// ended; subsequent cold traffic never reaches lookup at
    /// matched_tokens == 0), the lease lapses after
    /// [`tail_lease_ttl`] evictions so a dead session's tail cannot
    /// squat a slot indefinitely.
    pub(super) evictions_since_lookup: u32,
    /// Phase-0 measurement counters (ATLAS_SSM_SNAP_STATS). All aggregate,
    /// off the hot path's critical decisions — they only observe. The residual
    /// `recompute_tokens_on_hit` after tail-protect + a large pool is exactly
    /// what Phase 1 (spill-not-drop) converts from recompute → fault-in.
    pub(super) stats: SnapshotStats,
}

/// Tail-lease kill switch. Default ON; `ATLAS_SSM_TAIL_PROTECT=0` (or `off`)
/// disables. Backward compatible with the old opt-in scripts that set `=1`.
///
/// **INERT IN THE SHIPPING (MLPerf-edge) CONFIG — it protects nothing there.**
/// The lease only ever shields an entry with `is_tail == true`, and the sole
/// production writer of that flag is `insert_tail_snapshot`, called only from
/// `finalize_midchunk_capture`, which is unreachable when
/// `ATLAS_SSM_TAIL_MIDCHUNK=0` — which the frozen MLPerf-edge config sets.
/// Verified 2026-07-21 by call-graph audit (the 2026-07-20 eviction rig
/// likewise measured 0 lease hits). Do not read a `ATLAS_SSM_TAIL_PROTECT=1`
/// in a launch script as evidence that tail protection is doing work; check
/// `ATLAS_SSM_TAIL_MIDCHUNK` first. Behaviour here is deliberately unchanged —
/// this note is a warning to the next reader, not a defect report.
fn tail_lease_enabled() -> bool {
    !matches!(
        std::env::var("ATLAS_SSM_TAIL_PROTECT").as_deref(),
        Ok("0") | Ok("off")
    )
}

/// Evictions a leased tail survives without its session looking up again.
/// Derivation: the 2026-07-20 eviction rig measured ~18 evictions between a
/// deep session's turns at 8 slots with 6 churn requests/turn — 64 is a >3x
/// margin there, while production pools (128–256 slots) evict rarely enough
/// that the TTL almost never binds. Override: ATLAS_SSM_TAIL_LEASE_TTL.
///
/// Same caveat as [`tail_lease_enabled`]: with `ATLAS_SSM_TAIL_MIDCHUNK=0` no
/// entry is ever marked `is_tail`, so this TTL governs an empty set and
/// `ATLAS_SSM_TAIL_LEASE_TTL=128` in a launch script changes nothing.
fn tail_lease_ttl() -> u32 {
    std::env::var("ATLAS_SSM_TAIL_LEASE_TTL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
}

/// Marconi Eq-2 depth weight (staged, INERT by default). Within a session,
/// the eviction rank becomes `S(e) = norm(recency) + α·norm(token_count)`
/// (MLSys'25 arXiv:2411.19379: `S(n) = recency(n) + α·flop_efficiency(n)`;
/// with uniform snapshot slots, FLOPs-saved/byte degenerates to the token
/// depth the snapshot lets a warm turn skip — and losing an SSM snapshot
/// forfeits the whole KV prefix hit, so value ∝ depth). α = 0 (default) is
/// exactly today's pure-LRU ordering (min-max normalization is monotonic);
/// clamped to [0, 8] so a runaway env value cannot make depth
/// recency-insensitive (a depth-pinned analog of the 07-10 hit-pinning).
/// Default flip requires its own measured A/B: ATLAS_SNAP_EVICT_ALPHA.
fn snap_evict_alpha() -> f64 {
    std::env::var("ATLAS_SNAP_EVICT_ALPHA")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|a| a.clamp(0.0, 8.0))
        .unwrap_or(0.0)
}

impl SsmSnapshotIndex {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            access_counter: 0,
            last_lookup_session: 0,
            evictions_since_lookup: 0,
            stats: SnapshotStats::default(),
        }
    }

    /// Whether the live session's tail lease is currently in force.
    pub(super) fn tail_lease_active(&self) -> bool {
        self.last_lookup_session != 0
            && tail_lease_enabled()
            && self.evictions_since_lookup < tail_lease_ttl()
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
        // Track the live conversation so eviction can lease its is_tail
        // restore point; a fresh lookup renews the lease.
        if session_hash != 0 {
            self.last_lookup_session = session_hash;
            self.evictions_since_lookup = 0;
        }
        // Side-effect-free scan: only the WINNER gets its recency bumped,
        // below. Bumping every improving candidate kept shallow early-prefix
        // entries eternally fresh (each deep lookup walks the improving chain
        // through them), pinning them in the pool while the tail checkpoints
        // the next warm turn actually needs were evicted — the measured
        // frozen-anchor / 18.6k-token SSM replay pathology (2026-07-10,
        // re-landed after #317's re-cut reverted it).
        let mut best: Option<(usize, usize)> = None; // (snapshot_id, token_count)
        let mut best_idx: Option<usize> = None;
        for (i, entry) in self.entries.iter().enumerate() {
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
            // TAIL snapshots bleed past the exact prefix — byte-safe ONLY for the
            // same non-zero session. Cross-request reuse corrupts SSM state.
            if entry.is_tail && (session_hash == 0 || entry.session_hash != session_hash) {
                continue;
            }
            let h = hash_token_prefix(tokens, entry.token_count, adapter_id);
            if h != entry.prefix_hash {
                continue;
            }
            if best.is_none() || entry.token_count > best.unwrap().1 {
                best = Some((entry.snapshot_id, entry.token_count));
                best_idx = Some(i);
            }
        }
        if let Some(i) = best_idx {
            self.access_counter += 1;
            self.entries[i].last_access = self.access_counter;
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
    pub(super) fn log_stats_if_due(&self) {
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
        // Pure recency (LRU). The former forecast score
        // `last_access * (1 + hit_count)` multiplied a monotonic timestamp by
        // hit count, so any once-hit old entry outscored every fresh save;
        // once the cold entries drained, each new save's evict victim was the
        // PREVIOUS fresh save — the live turn's tail checkpoint died ms before
        // the next turn's lookup needed it, freezing the restore anchor
        // (measured 2026-07-10: anchor pinned at token 9056 for 29 turns,
        // 12.7k-token SSM replay, 40s TTFT tail). Re-landed after #317's
        // re-cut restored the old score.
        let escore = |e: &SnapshotEntry| e.last_access;

        // SESSION-AWARE eviction (default ON; ATLAS_SNAP_EVICT_LEGACY=1 → old per-entry).
        if std::env::var_os("ATLAS_SNAP_EVICT_LEGACY").is_none() {
            let tail_protect = self.tail_lease_active();
            // Skip tiered entries (no HBM slot to free).
            let victim_idx = self.session_aware_victim(tail_protect, true)?;
            let entry = self.entries.swap_remove(victim_idx);
            self.stats.evictions += 1;
            self.evictions_since_lookup = self.evictions_since_lookup.saturating_add(1);
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

    /// Pure victim selection for session-aware eviction. Split out for unit tests.
    /// Ranking: stalest session first, then oldest entry within it.
    /// `tail_protect`: lease the live session's `is_tail` restore point — the
    /// snapshot the next warm turn's block-floored `matched_tokens` actually
    /// looks up, NOT the deepest entry (that is the end-of-turn finish leaf,
    /// which sits above `matched_tokens` and never restores; #278 semantics
    /// protected it and bought nothing — 2026-07-20 rig: 0 hits either way).
    /// insert_tail's supersede sweep plus insert()'s is_tail-clearing keep
    /// the leased set at <=1 entry, so pools >=2 never deadlock.
    /// Correctness-safe (restore re-validates).
    /// `skip_tiered`: ignore spilled entries (no HBM slot). Returns `None` if none eligible.
    pub(super) fn session_aware_victim(
        &self,
        tail_protect: bool,
        skip_tiered: bool,
    ) -> Option<usize> {
        self.session_aware_victim_with_alpha(tail_protect, skip_tiered, snap_evict_alpha())
    }

    /// α-parameterized core of [`Self::session_aware_victim`] — split so unit
    /// tests can exercise α without mutating process-global env (a data race
    /// under the parallel test runner).
    pub(super) fn session_aware_victim_with_alpha(
        &self,
        tail_protect: bool,
        skip_tiered: bool,
        alpha: f64,
    ) -> Option<usize> {
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
        let leased = |e: &SnapshotEntry| {
            tail_protect
                && (e.is_tail || e.is_tail_sibling)
                && e.session_hash != 0
                && e.session_hash == self.last_lookup_session
        };
        // The lease bites only while an UNLEASED eligible candidate exists, so
        // a pool of only-leased entries (or a single entry) still yields a
        // victim — no deadlock, and `save`/reclaim always make progress.
        let n_unleased = self
            .entries
            .iter()
            .filter(|e| eligible(e) && !leased(e))
            .count();
        // Within-session rank: S(e) = norm(recency) + α·norm(depth) — Marconi
        // Eq 2 (see snap_evict_alpha). At the default α=0 this is exactly
        // pure-LRU ordering. Min-max normalization over the ELIGIBLE pool per
        // pass keeps scores bounded and relative (no per-entry accumulation —
        // the 07-10 fossil vector cannot re-enter here); a degenerate range
        // (max == min) normalizes to 0 rather than dividing by zero.
        let (mut min_a, mut max_a, mut min_t, mut max_t) = (u64::MAX, 0u64, usize::MAX, 0usize);
        for e in self.entries.iter().filter(|e| eligible(e)) {
            min_a = min_a.min(e.last_access);
            max_a = max_a.max(e.last_access);
            min_t = min_t.min(e.token_count);
            max_t = max_t.max(e.token_count);
        }
        let norm = |x: u64, min: u64, max: u64| {
            if max > min {
                (x - min) as f64 / (max - min) as f64
            } else {
                0.0
            }
        };
        let score = |e: &SnapshotEntry| {
            norm(e.last_access, min_a, max_a)
                + alpha * norm(e.token_count as u64, min_t as u64, max_t as u64)
        };
        // (stalest session first, then lowest S within it). The lease bites
        // only when >1 eligible entry remains, so a single-entry pool (even
        // if it is the leased tail) still yields a victim → no deadlock.
        let mut victim: Option<(usize, u64, f64)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if !eligible(e) {
                continue;
            }
            if leased(e) && n_unleased >= 1 {
                continue; // never evict the live session's restore points
            }
            let sf = *session_fresh.get(&e.session_hash).unwrap_or(&0);
            let s = score(e);
            let better = match victim {
                None => true,
                Some((_, vsf, vs)) => sf < vsf || (sf == vsf && s.total_cmp(&vs).is_lt()),
            };
            if better {
                victim = Some((i, sf, s));
            }
        }
        victim.map(|(i, _, _)| i)
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
#[path = "tests/snapshot_index.rs"]
mod tests;
