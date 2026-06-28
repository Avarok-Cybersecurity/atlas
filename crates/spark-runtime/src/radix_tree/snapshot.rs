// SPDX-License-Identifier: AGPL-3.0-only

//! SSM snapshot LRU index — independent of the token-radix structure.
//!
//! Snapshots are keyed by (session_hash, token_count, prefix_hash) so the
//! same prompt across requests can hit a cached SSM state without going
//! through the radix tree.

use std::sync::OnceLock;

use super::hash_token_prefix;

/// SBR M1 tail-pin eviction policy — OPT-IN, OFF by default. When enabled,
/// [`SsmSnapshotIndex::evict_lru`] pins the top-`tail_pin_k` DEEPEST snapshots of
/// each RESUMABLE session (one resumed at least once), so a resuming deep
/// conversation finds a near-tail SSM anchor instead of replaying from a far
/// shallow survivor (the "replay grows as slots fill" blowup, 1s→21s). Measured
/// 8× warm-resume speedup on the deep-conversation agentic regime.
///
/// DEFAULT OFF: it wins the single/few-deep-conversation regime but REGRESSES
/// balanced many-conversation serving ~30% (the recency·hit forecast is already
/// near-optimal there). Set `ATLAS_SBR_TAIL_PIN=1` for deep-agentic workloads.
fn tail_pin_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = std::env::var("ATLAS_SBR_TAIL_PIN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        tracing::info!(
            "SBR snapshot tail-pin eviction: {} (opt-in, top-K={})",
            if on { "ENABLED" } else { "off (baseline forecast)" },
            tail_pin_k()
        );
        on
    })
}

/// Number of a resumable session's DEEPEST snapshots to pin. The single deepest
/// overshoots the next match point (the leaf includes generated tokens, so the
/// block-aligned match lands just below it), so K≥2 is needed; K=8 spans the
/// overshoot and flattened warm-resume TTFT in measurement (mean 1.18s vs K=4
/// 3.37s vs baseline 9.53s). Override with `ATLAS_SBR_TAIL_PIN_K`.
fn tail_pin_k() -> usize {
    static K: OnceLock<usize> = OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("ATLAS_SBR_TAIL_PIN_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&k| k >= 1)
            .unwrap_or(8)
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
        self.evict_lru_inner(tail_pin_enabled(), tail_pin_k())
    }

    /// Eviction victim selection (split out so the policy is unit-testable
    /// without the env-gated `OnceLock`).
    ///
    /// Baseline forecast (B.4, 2026-04-25, Marconi paper §4): evict the entry
    /// with the lowest `last_access * (1 + hit_count)` — old AND cold first.
    /// hit_count weighting keeps recurrent prefixes (system prompts, tool
    /// descriptions) resident (#155 fixed the inverted divide-by form).
    ///
    /// SBR M1 tail-pin (`pin == true`, OPT-IN — see [`tail_pin_enabled`]): the
    /// forecast is orthogonal to REPLAY DISTANCE — a conversation's deep
    /// per-turn checkpoints have hit_count≈0 (never the resume anchor) and get
    /// evicted before the hot shallow prefix, stranding the next resume far from
    /// its tail (replay = depth − shallow_prefix → the 1s→21s blowup). When
    /// enabled we PIN the top-`pin_k` DEEPEST snapshots of each RESUMABLE session
    /// (one with any hit_count>0) so a near-tail anchor survives, evicting only
    /// NON-pinned entries (falling back to the global forecast only if every
    /// entry is pinned). Measured (deep conv idle under one-shot pressure):
    /// baseline 9.53s → 1.18s (8×, replay 11–984 tok).
    ///
    /// SCOPE / honesty: this wins the single/few-deep-conversation agentic
    /// regime but REGRESSES balanced many-conversation round-robin ~30% (the
    /// recency·hit forecast is already near-optimal there and pinning fights it),
    /// so it is OFF by default. `pin == false` is exactly the baseline forecast.
    pub(super) fn evict_lru_inner(&mut self, pin: bool, pin_k: usize) -> Option<usize> {
        if self.entries.is_empty() {
            return None;
        }

        // Pin the top-`pin_k` deepest snapshots of each RESUMABLE session.
        let mut pinned = std::collections::HashSet::new();
        if pin {
            let mut by_session: std::collections::HashMap<u64, Vec<(usize, usize)>> =
                std::collections::HashMap::new();
            let mut resumable: std::collections::HashSet<u64> = std::collections::HashSet::new();
            for (i, e) in self.entries.iter().enumerate() {
                if e.session_hash == 0 {
                    continue;
                }
                if e.hit_count > 0 {
                    resumable.insert(e.session_hash);
                }
                by_session
                    .entry(e.session_hash)
                    .or_default()
                    .push((e.token_count, i));
            }
            for (s, mut v) in by_session {
                if !resumable.contains(&s) {
                    continue;
                }
                v.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // deepest first
                for &(_, idx) in v.iter().take(pin_k) {
                    pinned.insert(idx);
                }
            }
        }

        // Evict the lowest-forecast NON-pinned entry; fall back to the global
        // lowest (pure forecast) only when every entry is pinned.
        let mut victim_idx: Option<usize> = None;
        let mut victim_score = u64::MAX;
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
            if !pinned.contains(&i) && score < victim_score {
                victim_score = score;
                victim_idx = Some(i);
            }
        }
        let idx = victim_idx.unwrap_or(all_idx);
        let entry = self.entries.swap_remove(idx);
        Some(entry.snapshot_id)
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}
