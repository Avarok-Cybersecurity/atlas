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
}

pub(super) struct SsmSnapshotIndex {
    pub(super) entries: Vec<SnapshotEntry>,
    pub(super) access_counter: u64,
    /// Session of the most recent `lookup` — the live conversation. Its
    /// DEEPEST snapshot is the one its next warm turn will restore from, so
    /// `evict_lru` protects it (ATLAS_SSM_TAIL_PROTECT). Tracks the running
    /// tip: recomputed each eviction from `token_count`, never a pinned slot.
    last_lookup_session: u64,
}

impl SsmSnapshotIndex {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            access_counter: 0,
            last_lookup_session: 0,
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
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return Some(old);
            }
        }
        self.access_counter += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            hit_count: 0,
        });
        None
    }

    /// Find deepest snapshot matching session within matched_tokens range.
    pub(super) fn lookup(
        &mut self,
        tokens: &[u32],
        matched_tokens: usize,
        session_hash: u64,
    ) -> Option<(usize, usize)> {
        // Track the live conversation so eviction can protect its deep tail.
        if session_hash != 0 {
            self.last_lookup_session = session_hash;
        }
        let mut best: Option<(usize, usize)> = None; // (snapshot_id, token_count)
        for entry in &mut self.entries {
            if entry.token_count > matched_tokens {
                continue;
            }
            if session_hash != 0 && entry.session_hash != 0 && entry.session_hash != session_hash {
                continue;
            }
            let h = hash_token_prefix(tokens, entry.token_count);
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
        best
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
            // session freshness = max last_access among that session's entries.
            let mut session_fresh: std::collections::HashMap<u64, u64> =
                std::collections::HashMap::with_capacity(self.entries.len());
            for e in &self.entries {
                let f = session_fresh.entry(e.session_hash).or_insert(0);
                if e.last_access > *f {
                    *f = e.last_access;
                }
            }
            // ATLAS_SSM_TAIL_PROTECT: exempt the live conversation's DEEPEST
            // snapshot from eviction — its next warm turn restores from it.
            // Without this the just-saved deep tail (hit_count=0, low escore) is
            // evicted before the hot token-8192 anchor (self-reinforced hit_count),
            // so warm restore falls back to 8192 and recomputes thousands of SSM
            // tokens (measured 50-75% of restores, mean ~4400 tok, ~7.6s TTFT/turn).
            // Recomputed each call (follows the deepening tip, never a pinned slot);
            // exactly ONE entry, scoped to the active session, so any pool >=2 has a
            // victim and never deadlocks. Correctness-safe: restore re-validates
            // session_hash+prefix_hash+depth, so changing the victim cannot drift.
            let protected_idx: Option<usize> = if self.last_lookup_session != 0
                && std::env::var_os("ATLAS_SSM_TAIL_PROTECT").is_some()
            {
                self.entries
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.session_hash == self.last_lookup_session)
                    .max_by_key(|(_, e)| e.token_count)
                    .map(|(i, _)| i)
            } else {
                None
            };
            let n = self.entries.len();
            let mut victim_idx = protected_idx.map_or(0, |p| if p == 0 && n > 1 { 1 } else { 0 });
            // (stalest session first, then lowest entry score within it)
            let mut victim_key = (u64::MAX, u64::MAX);
            for (i, e) in self.entries.iter().enumerate() {
                if Some(i) == protected_idx && n > 1 {
                    continue; // never evict the live session's deepest tail
                }
                let sf = *session_fresh.get(&e.session_hash).unwrap_or(&0);
                let key = (sf, escore(e));
                if key < victim_key {
                    victim_key = key;
                    victim_idx = i;
                }
            }
            let entry = self.entries.swap_remove(victim_idx);
            return Some(entry.snapshot_id);
        }

        let mut victim_idx = 0;
        let mut victim_score = u64::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            let score = escore(entry);
            if score < victim_score {
                victim_score = score;
                victim_idx = i;
            }
        }
        let entry = self.entries.swap_remove(victim_idx);
        Some(entry.snapshot_id)
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}
