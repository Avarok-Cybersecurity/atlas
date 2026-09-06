// SPDX-License-Identifier: AGPL-3.0-only

//! Lease, sibling, and α-score tests for the SSM snapshot index. Split from
//! `snapshot_index.rs` (file-size cap). Uses that module's helpers.

use super::super::*;
use super::{entry, index, tail_entry};

/// The lease shields the live session's `is_tail` restore point — even when
/// it is the OLDEST entry in the pool — so the warm-turn restore survives
/// same-session save pressure.
#[test]
fn tail_lease_protects_live_session_is_tail() {
    let idx = index(vec![entry(7, 1, 8192, 100), tail_entry(9, 1, 6064, 50)], 1);
    let v = idx.session_aware_victim(true, false).unwrap();
    // Victim must NOT be the leased tail (id 9) despite its age.
    assert_eq!(idx.entries[v].snapshot_id, 7);
}

/// Direct inversion of #278's off-target semantics: the live session's
/// DEEPEST entry (the end-of-turn finish leaf, NOT a tail) is a normal
/// eviction candidate, while the tail is leased even when it is the pool's
/// natural LRU victim. The old policy protected exactly this entry — which
/// sits above `matched_tokens` and never restores — while the true restore
/// point died (2026-07-20 rig: 0 hits with old protection on or off).
///
/// The pool holds THREE entries on purpose. With the tail as the only other
/// entry the lease leaves exactly one candidate, so "no lease at all" and
/// "lease the tail" pick the same victim and the oracle cannot tell them
/// apart; the fresh shallow entry keeps both the lease and the within-session
/// recency order observable in the one answer.
#[test]
fn finish_leaf_not_protected() {
    let idx = index(
        vec![
            entry(6, 1, 100, 95),       // freshest, shallow, NOT a tail
            entry(7, 1, 6080, 50),      // finish leaf: deepest, NOT a tail
            tail_entry(9, 1, 6064, 40), // true restore point: OLDEST, leased
        ],
        1,
    );
    let v = idx.session_aware_victim(true, false).unwrap();
    assert_eq!(
        idx.entries[v].snapshot_id, 7,
        "the deepest non-tail entry must be the victim: the tail is leased \
         despite being the oldest, and recency still orders the unleased pair"
    );
}

/// The lease only shields the LIVE conversation's tail; a dormant session's
/// tail is still evictable (correct — session-aware ranking evicts the
/// stalest conversation first).
#[test]
fn dormant_session_tail_evictable() {
    // session 2 is live; session 1 is dormant (older last_access).
    let idx = index(
        vec![
            tail_entry(1, 1, 20000, 10), // dormant tail — should die first
            entry(2, 2, 4000, 90),       // live shallow
            tail_entry(3, 2, 12000, 95), // live tail — leased
        ],
        2,
    );
    let v = idx.session_aware_victim(true, false).unwrap();
    assert_eq!(
        idx.entries[v].snapshot_id, 1,
        "stalest (dormant) session evicted first"
    );
}

/// A pool of exactly one entry must still yield that entry as victim even
/// when it is the leased tail — otherwise `save` can never reclaim and the
/// cache deadlocks.
#[test]
fn single_leased_entry_still_evictable() {
    let idx = index(vec![tail_entry(5, 1, 16000, 50)], 1);
    let v = idx.session_aware_victim(true, false).unwrap();
    assert_eq!(idx.entries[v].snapshot_id, 5);
}

/// The lease LAPSES: if the leased session never looks up again (it ended;
/// cold churn never reaches lookup at matched_tokens == 0), its tail becomes
/// a normal candidate after the TTL so a dead session cannot squat a slot.
#[test]
fn lease_expires_without_live_lookups() {
    let mut idx = index(vec![tail_entry(9, 1, 6064, 50), entry(7, 2, 4000, 100)], 1);
    // Lease fresh: the newer non-tail entry is the victim... but session 2 is
    // FRESHER than session 1, so session-staleness picks session 1 first and
    // the lease deflects to... there is only the tail in session 1, so the
    // victim is the session-2 entry.
    assert_eq!(idx.evict_lru(), Some(7), "leased tail survives while fresh");
    idx.entries.push(entry(8, 2, 4000, 101));
    // Simulate TTL expiry: many evictions with no lookup from session 1.
    idx.evictions_since_lookup = super::tail_lease_ttl();
    assert_eq!(
        idx.evict_lru(),
        Some(9),
        "expired lease: the dead session's tail is evictable (stalest session)"
    );
}

/// Overwriting a prefix via plain `insert` clears `is_tail` — otherwise the
/// overwrite re-homes another session's tail (new session_hash, stale
/// is_tail=true), breaching the <=1-leased-entry invariant and leaking tail
/// semantics onto a non-tail save.
#[test]
fn insert_overwrite_clears_is_tail() {
    let mut idx = SsmSnapshotIndex::new();
    let displaced = idx.insert_tail(0xAB, /*slot*/ 1, /*session*/ 7, /*tok*/ 500);
    assert!(displaced.is_empty());
    assert!(idx.entries[0].is_tail);
    // A plain save re-homes the same prefix for a DIFFERENT session.
    let old = idx.insert(0xAB, /*slot*/ 2, /*session*/ 8, /*tok*/ 500);
    assert_eq!(old, Some(1));
    assert!(
        !idx.entries[0].is_tail,
        "plain insert must clear is_tail on overwrite"
    );
}

/// The serving-path lookup (`lookup_tiered`) must session-gate `is_tail`
/// entries exactly like the reference `lookup`: a tail is byte-safe ONLY for
/// the same non-zero session. (The gate was previously missing here.)
#[test]
fn lookup_tiered_tail_session_gate() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..64).collect();
    let ph = super::hash_token_prefix(&toks, 64, 0);
    idx.insert_tail(ph, /*slot*/ 4, /*session*/ 7, /*tok*/ 64);
    // Same session: hit.
    assert!(idx.lookup_tiered(&toks, 64, 7, 0).is_some());
    // Different session: the tail must NOT bleed.
    assert!(idx.lookup_tiered(&toks, 64, 8, 0).is_none());
    // Sessionless lookup: must NOT bleed either.
    assert!(idx.lookup_tiered(&toks, 64, 0, 0).is_none());
}

// ─────────────── Marconi α-score (staged, inert at default α=0) ───────────────

/// At the default α=0 the within-session rank is EXACTLY pure-LRU: the
/// oldest entry of the stalest session is the victim, depth ignored.
#[test]
fn alpha_zero_is_pure_lru_ordering() {
    // Same session: deep-but-old vs shallow-but-fresh. α=0 → oldest dies.
    let idx = index(vec![entry(1, 7, 20000, 10), entry(2, 7, 100, 90)], 7);
    let v = idx.session_aware_victim(false, false).unwrap();
    assert_eq!(idx.entries[v].snapshot_id, 1, "α=0 ranks purely by recency");
}

/// With α set, depth outweighs a modest recency gap WITHIN a session (a
/// deep tail outscores a shallow fresh save), while session staleness stays
/// the PRIMARY key — a fresher session's shallow entry still outlives a
/// staler session entirely. α is injected (never via env — process-global
/// mutation races the parallel test runner).
#[test]
fn alpha_prefers_depth_within_session_staleness_still_primary() {
    // One session: deep old entry vs shallow slightly-fresher entry.
    let idx = index(vec![entry(1, 7, 20000, 50), entry(2, 7, 100, 60)], 7);
    let v = idx
        .session_aware_victim_with_alpha(false, false, 2.0)
        .unwrap();
    assert_eq!(
        idx.entries[v].snapshot_id, 2,
        "α=2: the shallow entry is the victim despite being fresher"
    );
    // Two sessions: staleness first regardless of α — the stale session's
    // DEEP entry still dies before the fresh session's shallow one.
    let idx2 = index(vec![entry(1, 1, 20000, 10), entry(2, 2, 100, 90)], 2);
    let v2 = idx2
        .session_aware_victim_with_alpha(false, false, 2.0)
        .unwrap();
    assert_eq!(
        idx2.entries[v2].snapshot_id, 1,
        "session staleness remains the primary key at any α"
    );
}

// ───────────────────── tail EARLY sibling (tb - bs) lease ─────────────────────

/// A sibling entry (`is_tail_sibling`) of the live session is leased exactly
/// like the tail — it serves warm turns whose block-floored match lands one
/// block below `tb` (2/7 of restores on the eviction rig).
#[test]
fn sibling_leased_with_live_session() {
    let mut idx = index(vec![entry(7, 1, 8192, 100)], 1);
    idx.insert_tail_sibling(0xE1, /*slot*/ 9, /*session*/ 1, /*tok*/ 6048);
    // The sibling is the OLDER... actually newer here; make a non-leased older
    // competitor the natural LRU victim and assert the sibling never loses
    // even when we age it below the competitor.
    idx.entries
        .iter_mut()
        .find(|e| e.snapshot_id == 9)
        .unwrap()
        .last_access = 10; // oldest in pool
    let v = idx.session_aware_victim(true, false).unwrap();
    assert_eq!(idx.entries[v].snapshot_id, 7, "sibling must be leased");
}

/// `insert_tail`'s supersede sweep removes the session's previous tail AND
/// sibling together, so the leased set stays bounded at <=2 per session.
#[test]
fn new_tail_sweeps_old_tail_and_sibling() {
    let mut idx = SsmSnapshotIndex::new();
    idx.insert_tail(0xA1, 1, /*session*/ 7, 500);
    idx.insert_tail_sibling(0xA2, 2, 7, 484);
    // Next turn: a fresh tail supersedes both.
    let displaced = idx.insert_tail(0xB1, 3, 7, 1000);
    let mut d = displaced.clone();
    d.sort_unstable();
    assert_eq!(d, vec![1, 2], "old tail AND sibling displaced together");
    assert_eq!(idx.len(), 1);
}

/// A pool consisting ONLY of leased entries (tail + sibling) must still
/// yield a victim — the lease binds only while an unleased candidate exists.
///
/// The premise is asserted, not assumed: "some victim came back" is trivially
/// true whenever the lease is INACTIVE, so without the `tail_lease_active`
/// check this test passes even with the kill switch's polarity inverted (the
/// lease off for everyone) — it would then be measuring nothing.
#[test]
fn all_leased_pool_still_evicts() {
    let mut idx = SsmSnapshotIndex::new();
    idx.insert_tail(0xA1, 1, 7, 500);
    idx.insert_tail_sibling(0xA2, 2, 7, 484);
    // Latch the lease onto session 7.
    let toks: Vec<u32> = (0..8).collect();
    let _ = idx.lookup_tiered(&toks, 0, 7, 0);
    assert!(
        idx.tail_lease_active(),
        "the lease must actually be in force"
    );
    assert!(
        idx.entries
            .iter()
            .all(|e| (e.is_tail || e.is_tail_sibling) && e.session_hash == 7),
        "every entry must be a leased restore point of the live session"
    );
    assert_eq!(
        idx.evict_lru(),
        Some(1),
        "only-leased pool must still yield a victim, oldest-first"
    );
    assert_eq!(idx.len(), 1, "and the reclaim must actually have happened");
}

/// Sibling entries are exact-prefix keyed and carry NO session gate of any
/// kind: same-session, sessionless AND different-session lookers may all use
/// them (a tail would refuse the latter two). A sibling stores state at
/// EXACTLY `token_count`, so it is a pure function of the verified token
/// prefix — content-addressed, exactly like the KV radix, which shares blocks
/// on token match with no session gate at all.
///
/// There is NO "general session filter" behind non-tail entries in
/// `lookup_tiered` — an earlier version of this comment claimed one excluded
/// different-nonzero-session lookers, and the third assertion below is the
/// executable refutation. That is the intended contract (see the `is_tail`
/// INVARIANT on `SnapshotEntry`), not a hole: it is what un-broke warm TTFT
/// when `session_hash` proved unstable across turns of a growing prompt.
#[test]
fn sibling_not_session_gated_in_lookup() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..64).collect();
    let ph = super::hash_token_prefix(&toks, 64, 0);
    idx.insert_tail_sibling(ph, /*slot*/ 4, /*session*/ 7, /*tok*/ 64);
    // Same session: hit.
    assert!(idx.lookup_tiered(&toks, 64, 7, 0).is_some());
    // Sessionless looker: hit (an is_tail entry would be refused here).
    assert!(
        idx.lookup_tiered(&toks, 64, 0, 0).is_some(),
        "sibling must not carry the is_tail session gate"
    );
    // A DIFFERENT non-zero session: hit as well — the partition a re-added
    // blanket gate would silently break while the two above stayed green.
    assert!(
        idx.lookup_tiered(&toks, 64, 8, 0).is_some(),
        "a sibling is content-addressed: another session may restore from it"
    );
}

/// Plain `insert` overwrite clears the sibling flag, mirroring the is_tail
/// clearing (same cross-session rehoming breach vector).
#[test]
fn insert_overwrite_clears_sibling() {
    let mut idx = SsmSnapshotIndex::new();
    idx.insert_tail_sibling(0xC1, 1, 7, 500);
    assert!(idx.entries[0].is_tail_sibling);
    idx.insert(0xC1, 2, 8, 500);
    assert!(!idx.entries[0].is_tail_sibling);
}
