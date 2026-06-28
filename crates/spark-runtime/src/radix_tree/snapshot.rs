// SPDX-License-Identifier: AGPL-3.0-only

//! SSM snapshot LRU index — independent of the token-radix structure.
//!
//! Snapshots are keyed by (session_hash, token_count, prefix_hash) so the
//! same prompt across requests can hit a cached SSM state without going
//! through the radix tree.

use std::sync::OnceLock;

use super::hash_token_prefix;

/// SBR M1 two-tier eviction policy. When enabled, [`SsmSnapshotIndex::evict_lru`]
/// evicts NON-resumable (one-shot / untracked) snapshots before any resumable
/// multi-turn conversation's snapshot — protecting live conversations' SSM
/// checkpoints from transient one-shot churn (the cause of the "replay grows as
/// slots fill" TTFT blowup, 1s→21s). When every entry is resumable it degrades
/// identically to the baseline recency·hit forecast (Marconi §4) — provably
/// do-no-harm.
///
/// Enabled by default; set `ATLAS_SBR_TAIL_PIN=0` to restore the pure
/// forecast-LRU baseline (for A/B benchmarking against the fix).
fn tail_pin_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = std::env::var("ATLAS_SBR_TAIL_PIN").map(|v| v != "0").unwrap_or(true);
        tracing::info!(
            "SBR snapshot two-tier eviction (protect resumable convs): {}",
            if on { "ENABLED" } else { "disabled (baseline)" },
        );
        on
    })
}

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
}

impl SsmSnapshotIndex {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            access_counter: 0,
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
        // Forecast-based policy (B.4, 2026-04-25, Marconi paper §4):
        // evict the entry with the lowest last_access * (1 + hit_count)
        // — old AND cold first. Pure LRU (`last_access` only) discarded
        // hot prefixes that just happened to be re-accessed less
        // recently than a one-shot entry; weighting by hit_count keeps
        // recurrent prefixes (system prompts, tool descriptions in
        // agentic sessions) resident longer.
        //
        // #155: the original formula DIVIDED by (1 + hit_count), which
        // inverts the intent — frequently-hit snapshots scored LOWEST
        // and were evicted first at pool saturation (measured: a
        // just-selected snapshot evicted 7s later while ~50
        // never-accessed entries survived → selected=None mid-session
        // → full-conversation SSM recompute on the next warm hit).
        //
        // SBR M1 tail-pin: forecast score keeps HOT (shared-prefix) snapshots
        // resident, but that is orthogonal to REPLAY DISTANCE — a conversation's
        // deep per-turn checkpoint has hit_count≈0 and gets evicted before the
        // hot system-prompt prefix, stranding a resuming turn far from its tail
        // (replay = depth − shallow_prefix → 21s). We therefore exclude each
        // session's DEEPEST snapshot (its "tail" = the global section we keep
        // resident) from eviction, falling back to evicting the lowest-score
        // tail only if every entry is a tail (pool too small for the live set).
        // Pin only RESUMABLE sessions' deepest. A session is resumable if any
        // of its entries has been hit (hit_count>0) — i.e. a real multi-turn
        // conversation that has actually been resumed. One-shot sessions
        // (e.g. distractor traffic that never returns) are NOT pinned, so they
        // age out under the forecast score instead of wastefully holding slots
        // and evicting a live conversation's useful intermediate checkpoints.
        // (Measured 2026-06-27: naive "pin every session's deepest" REGRESSED
        // a main conversation 4.0s→8.8s by pinning ~96 dead one-shot tails.)
        // SBR M1 — TWO-TIER eviction (protect resumable conversations from
        // transient one-shot churn). A session is RESUMABLE iff some entry has
        // hit_count>0 (a real multi-turn conversation that was actually resumed
        // at least once). The stranding pathology is the forecast evicting a live
        // conversation's deep SSM checkpoint to make room for transient one-shot
        // traffic, leaving the next resume to replay from a far-shallow survivor
        // (1s→21s). Fix: evict NON-resumable (one-shot / untracked) entries BEFORE
        // any resumable conversation's entry; only when no non-resumable entry
        // exists do we fall through to the pure recency·hit forecast over all
        // entries.
        //
        // This is the right discriminator (NOT count/budget heuristics, which are
        // unreliable from the index's local view): it protects exactly what a
        // resuming conversation needs (its own checkpoints) and is PROVABLY
        // do-no-harm when every entry is resumable (balanced multi-conversation
        // round-robin) — the non-resumable pool is empty, so it is identical to
        // the baseline forecast. Earlier "pin the top-K deepest" variants
        // regressed that regime 5.9s→7.7s by displacing live convs' working sets.
        let pin = tail_pin_enabled();
        let mut resumable: std::collections::HashSet<u64> = std::collections::HashSet::new();
        if pin {
            for e in &self.entries {
                if e.session_hash != 0 && e.hit_count > 0 {
                    resumable.insert(e.session_hash);
                }
            }
        }
        let protected = |e: &SnapshotEntry| -> bool {
            pin && e.session_hash != 0 && resumable.contains(&e.session_hash)
        };

        // Tier 1: lowest-score among NON-protected (one-shot / untracked).
        // Tier 2 fallback: lowest-score among ALL (pure forecast) when every
        // entry belongs to a resumable conversation.
        let mut tier1_idx: Option<usize> = None;
        let mut tier1_score = u64::MAX;
        let mut all_idx = 0;
        let mut all_score = u64::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            // Saturating math: both factors fit u64 comfortably
            // (access_counter is monotonic per-process, hit_count u32).
            let score = entry.last_access.saturating_mul(1 + entry.hit_count as u64);
            if score < all_score {
                all_score = score;
                all_idx = i;
            }
            if !protected(entry) && score < tier1_score {
                tier1_score = score;
                tier1_idx = Some(i);
            }
        }
        let idx = tier1_idx.unwrap_or(all_idx);
        let entry = self.entries.swap_remove(idx);
        Some(entry.snapshot_id)
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}
