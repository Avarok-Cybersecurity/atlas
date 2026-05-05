// SPDX-License-Identifier: AGPL-3.0-only

//! Session-scoped SSM state manager.
//!
//! SSM recurrent state (h_state, conv_state) depends on ALL tokens seen so far,
//! not just the prefix. When two sessions share a system prompt prefix, the radix
//! tree's KV cache is safely reusable (position-dependent), but SSM snapshots are
//! NOT (sequence-dependent). This manager gates snapshot restore/save by session,
//! preventing cross-session state contamination.
//!
//! Sessions are identified by hashing the first N tokens of the prompt (covering
//! system prompt + first user message). The manager tracks which SSM snapshot pool
//! slots belong to which session and evicts the least-recently-used session when
//! the pool is exhausted.

use std::collections::HashMap;
use std::time::Instant;

/// Tracks SSM snapshot ownership per session with LRU eviction.
pub struct SessionSsmManager {
    sessions: HashMap<u64, SessionState>,
    /// TTL in seconds — sessions not accessed within this window are evictable.
    ttl_secs: u64,
}

/// Per-session state tracking.
struct SessionState {
    /// SSM snapshot pool slot IDs owned by this session.
    snapshot_slots: Vec<usize>,
    /// Last time this session was accessed (save or restore).
    last_access: Instant,
}

impl SessionSsmManager {
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            sessions: HashMap::new(),
            ttl_secs,
        }
    }

    /// Register a snapshot slot as belonging to a session.
    pub fn save_snapshot(&mut self, session_hash: u64, slot_id: usize) {
        let entry = self
            .sessions
            .entry(session_hash)
            .or_insert_with(|| SessionState {
                snapshot_slots: Vec::new(),
                last_access: Instant::now(),
            });
        entry.last_access = Instant::now();
        if !entry.snapshot_slots.contains(&slot_id) {
            entry.snapshot_slots.push(slot_id);
        }
    }

    /// Check if a snapshot slot belongs to the given session.
    /// Returns true and updates last_access if owned.
    pub fn owns_snapshot(&mut self, session_hash: u64, slot_id: usize) -> bool {
        if let Some(state) = self.sessions.get_mut(&session_hash)
            && state.snapshot_slots.contains(&slot_id)
        {
            state.last_access = Instant::now();
            return true;
        }
        false
    }

    /// Evict the least-recently-used session and return its freed snapshot slot IDs.
    /// Called when the SSM snapshot pool is exhausted and a new save is needed.
    /// Returns None if there are no sessions to evict.
    pub fn evict_lru(&mut self) -> Option<Vec<usize>> {
        if self.sessions.is_empty() {
            return None;
        }
        let lru_hash = *self
            .sessions
            .iter()
            .min_by_key(|(_, state)| state.last_access)?
            .0;
        self.evict_session(lru_hash)
    }

    /// Evict a specific session and return its freed snapshot slot IDs.
    pub fn evict_session(&mut self, session_hash: u64) -> Option<Vec<usize>> {
        let state = self.sessions.remove(&session_hash)?;
        if !state.snapshot_slots.is_empty() {
            tracing::info!(
                "Session {session_hash:#x} evicted: freed {} SSM snapshot slot(s)",
                state.snapshot_slots.len(),
            );
        }
        Some(state.snapshot_slots)
    }

    /// Evict all sessions that haven't been accessed within the TTL window.
    /// Returns all freed snapshot slot IDs.
    pub fn evict_expired(&mut self) -> Vec<usize> {
        let now = Instant::now();
        let ttl = std::time::Duration::from_secs(self.ttl_secs);
        let mut freed = Vec::new();
        self.sessions.retain(|hash, state| {
            if now.duration_since(state.last_access) > ttl {
                tracing::info!(
                    "Session {hash:#x} expired ({} slots freed)",
                    state.snapshot_slots.len(),
                );
                freed.extend_from_slice(&state.snapshot_slots);
                false
            } else {
                true
            }
        });
        freed
    }

    /// Number of active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Total snapshot slots across all sessions.
    pub fn total_slots(&self) -> usize {
        self.sessions.values().map(|s| s.snapshot_slots.len()).sum()
    }
}

/// Compute a session hash from tokenized prompt.
/// Hashes up to 1024 tokens — covers the full system prompt + first user
/// message for most clients (Claude Code ~7k, OpenCode ~17k system prompts).
/// Using more tokens reduces false session collisions where two different
/// conversations share a short prefix.
pub fn compute_session_hash(prompt_tokens: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let n = prompt_tokens.len().min(1024);
    for &tok in &prompt_tokens[..n] {
        tok.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_and_owns() {
        let mut mgr = SessionSsmManager::new(600);
        mgr.save_snapshot(0xAABB, 5);
        mgr.save_snapshot(0xAABB, 7);
        assert!(mgr.owns_snapshot(0xAABB, 5));
        assert!(mgr.owns_snapshot(0xAABB, 7));
        assert!(!mgr.owns_snapshot(0xAABB, 9));
        assert!(!mgr.owns_snapshot(0xCCDD, 5));
    }

    #[test]
    fn test_evict_lru() {
        let mut mgr = SessionSsmManager::new(600);
        mgr.save_snapshot(0x1111, 1);
        // Advance time for session 0x1111
        mgr.sessions.get_mut(&0x1111).unwrap().last_access =
            Instant::now() - std::time::Duration::from_secs(100);
        mgr.save_snapshot(0x2222, 2);
        // 0x1111 is older → should be evicted
        let freed = mgr.evict_lru().unwrap();
        assert_eq!(freed, vec![1]);
        assert_eq!(mgr.session_count(), 1);
        assert!(mgr.owns_snapshot(0x2222, 2));
    }

    #[test]
    fn test_evict_expired() {
        let mut mgr = SessionSsmManager::new(5); // 5 second TTL
        mgr.save_snapshot(0xAAAA, 10);
        mgr.sessions.get_mut(&0xAAAA).unwrap().last_access =
            Instant::now() - std::time::Duration::from_secs(10);
        mgr.save_snapshot(0xBBBB, 20);
        let freed = mgr.evict_expired();
        assert_eq!(freed, vec![10]);
        assert_eq!(mgr.session_count(), 1);
    }

    #[test]
    fn test_duplicate_slot_save() {
        let mut mgr = SessionSsmManager::new(600);
        mgr.save_snapshot(0x1234, 3);
        mgr.save_snapshot(0x1234, 3); // duplicate
        mgr.save_snapshot(0x1234, 4);
        let freed = mgr.evict_session(0x1234).unwrap();
        assert_eq!(freed, vec![3, 4]); // no duplicates
    }

    #[test]
    fn test_session_hash_stability() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let h1 = compute_session_hash(&tokens);
        let h2 = compute_session_hash(&tokens);
        assert_eq!(h1, h2);

        let different = vec![1u32, 2, 3, 4, 6];
        let h3 = compute_session_hash(&different);
        assert_ne!(h1, h3);
    }
}
