// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Build an entry with an explicit forecast profile. `snapshot_id` doubles
/// as a stable identity we assert on (independent of Vec index).
fn entry(
    snapshot_id: usize,
    session_hash: u64,
    token_count: usize,
    last_access: u64,
    hit_count: u32,
) -> SnapshotEntry {
    SnapshotEntry {
        snapshot_id,
        session_hash,
        token_count,
        prefix_hash: snapshot_id as u64, // unique, irrelevant to victim choice
        last_access,
        hit_count,
        tiered: false,
    }
}

fn index(entries: Vec<SnapshotEntry>, live: u64) -> SsmSnapshotIndex {
    SsmSnapshotIndex {
        entries,
        access_counter: 1000,
        last_lookup_session: live,
        stats: SnapshotStats::default(),
    }
}

/// The deep-tail eviction inversion (#278 root cause), reproduced against
/// the session-aware policy: within a SINGLE live session the hot 8192
/// anchor (self-reinforced hit_count) out-scores the just-aged deep tail
/// (hit_count=0), so without tail-protect the tail is the victim.
#[test]
fn deep_tail_evicted_without_tail_protect() {
    let idx = index(
        vec![
            entry(
                /*id*/ 7, /*sess*/ 1, /*tok*/ 8192, /*last*/ 100,
                /*hits*/ 10,
            ),
            entry(
                /*id*/ 9, /*sess*/ 1, /*tok*/ 16000, /*last*/ 50,
                /*hits*/ 0,
            ),
        ],
        1,
    );
    // Victim is the deep tail (id 9) — the pathology.
    let v = idx.session_aware_victim(false, false).unwrap();
    assert_eq!(idx.entries[v].snapshot_id, 9);
}

/// With tail-protect the live session's DEEPEST snapshot is exempt, so the
/// hot anchor is evicted instead and the warm-turn restore anchor survives.
#[test]
fn deep_tail_survives_with_tail_protect() {
    let idx = index(
        vec![entry(7, 1, 8192, 100, 10), entry(9, 1, 16000, 50, 0)],
        1,
    );
    let v = idx.session_aware_victim(true, false).unwrap();
    // Victim must NOT be the protected deep tail (id 9); it is the anchor.
    assert_eq!(idx.entries[v].snapshot_id, 7);
}

/// Tail-protect only shields the LIVE conversation's tail; a dormant
/// session's deep tail is still evictable (correct — session-aware ranking
/// evicts the stalest conversation first).
#[test]
fn dormant_session_tail_not_protected() {
    // session 2 is live; session 1 is dormant (older last_access).
    let idx = index(
        vec![
            entry(1, 1, 20000, 10, 0), // dormant deep tail — should die first
            entry(2, 2, 4000, 90, 0),  // live shallow
            entry(3, 2, 12000, 95, 0), // live deep tail — protected
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
/// when it is the protected tail — otherwise `save` can never reclaim and
/// the cache deadlocks.
#[test]
fn single_protected_entry_still_evictable() {
    let idx = index(vec![entry(5, 1, 16000, 50, 0)], 1);
    let v = idx.session_aware_victim(true, false).unwrap();
    assert_eq!(idx.entries[v].snapshot_id, 5);
}

/// `lookup` records the live session so a later eviction protects the right
/// conversation's tail.
#[test]
fn lookup_tracks_live_session() {
    let mut idx = SsmSnapshotIndex::new();
    assert_eq!(idx.last_lookup_session, 0);
    // No matching entries — lookup returns None but must still latch the session.
    let _ = idx.lookup(&[1, 2, 3], 3, /*session*/ 42, /*adapter*/ 0);
    assert_eq!(idx.last_lookup_session, 42);
}

/// Telemetry: a miss records full-recompute; a hit records the residual
/// distance between the match point and the restored anchor.
#[test]
fn stats_track_hits_and_recompute() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..100).collect();

    // Cold miss over 100 matched tokens → full recompute counted.
    assert!(
        idx.lookup(&toks, 100, /*session*/ 7, /*adapter*/ 0)
            .is_none()
    );
    // Register an anchor at depth 40 (hash must line up with lookup's recompute).
    let ph = super::hash_token_prefix(&toks, 40, 0);
    assert!(
        idx.insert(
            ph, /*snap*/ 3, /*session*/ 7, /*token_count*/ 40
        )
        .is_none()
    );
    // Warm turn: match 100 tokens, restore the depth-40 anchor → 60 recompute.
    let hit = idx.lookup(&toks, 100, 7, 0);
    assert_eq!(hit, Some((3, 40)));

    let s = idx.stats; // child module: private field is in scope
    assert_eq!(s.lookups, 2);
    assert_eq!(s.hits, 1);
    assert_eq!(s.saves, 1);
    assert_eq!(s.anchor_depth_sum, 40);
    assert_eq!(s.recompute_tokens_on_hit, 60, "matched(100) - anchor(40)");
    assert_eq!(
        s.recompute_tokens_on_miss, 100,
        "cold miss = full recompute"
    );
}

// ───────────────────────── Phase 1b state machine ───────────────────────

/// `evict_to_tier` keeps the entry (findable) but flips it to spilled and
/// frees its HBM slot — the core spill-not-drop transition.
#[test]
fn evict_to_tier_spills_not_removes() {
    // id 3 = hot 8192 anchor (escore 1100); id 9 = cold deep tail (escore 50).
    let mut idx = index(
        vec![entry(3, 1, 8192, 100, 10), entry(9, 1, 16000, 50, 0)],
        1,
    );
    let before = idx.len();
    let (freed_slot, key) = idx.evict_to_tier().expect("a resident victim exists");
    // No tail-protect (env off) → the coldest entry (deep tail id 9) is the
    // victim — the #278 pathology, but harmless here because we SPILL it
    // (faultable back in) rather than drop it.
    assert_eq!(freed_slot, 9);
    assert_eq!(key, 9, "key is the victim's prefix_hash");
    assert_eq!(idx.len(), before, "entry kept, not removed");
    assert_eq!(idx.stats.tier_spills, 1);
    // The spilled entry holds no HBM slot: the drop path must skip it and
    // free only the still-resident entry (id 3).
    assert_eq!(idx.evict_lru(), Some(3));
}

/// A spilled entry is invisible to the non-tier `lookup` (never hands back a
/// stale slot) but is found by `lookup_tiered` as `Tier(key)`.
#[test]
fn spilled_entry_lookup_semantics() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..50).collect();
    let ph = super::hash_token_prefix(&toks, 50, 0);
    idx.insert(ph, /*slot*/ 4, /*session*/ 7, /*tok*/ 50);
    // Spill it.
    let (freed, key) = idx.evict_to_tier().unwrap();
    assert_eq!((freed, key), (4, ph));

    // Non-tier lookup ignores the spilled entry → miss (safe recompute).
    assert!(idx.lookup(&toks, 50, 7, 0).is_none());
    // Tier-aware lookup finds it as Tier(key).
    let m = idx.lookup_tiered(&toks, 50, 7, 0).expect("tiered hit");
    assert_eq!(m.token_count, 50);
    assert_eq!(m.loc, SnapLoc::Tier(ph));
    assert_eq!(idx.stats.tier_hits, 1);
}

/// After fault-in, `promote` re-homes the entry to a fresh HBM slot and it
/// is resident again (visible to both lookups as `Hbm`).
#[test]
fn promote_rehomes_to_hbm() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..30).collect();
    let ph = super::hash_token_prefix(&toks, 30, 0);
    idx.insert(ph, 1, 7, 30);
    idx.evict_to_tier().unwrap();

    assert!(idx.promote(ph, /*new_slot*/ 12));
    assert_eq!(idx.stats.tier_fault_ins, 1);
    // Resident again: non-tier lookup now returns the new slot.
    assert_eq!(idx.lookup(&toks, 30, 7, 0), Some((12, 30)));
    let m = idx.lookup_tiered(&toks, 30, 7, 0).unwrap();
    assert_eq!(m.loc, SnapLoc::Hbm(12));
}

/// `evict_to_tier` returns None when every entry is already spilled — the
/// caller must not spin (there is no HBM slot left to free).
#[test]
fn evict_to_tier_none_when_all_spilled() {
    let mut idx = SsmSnapshotIndex::new();
    idx.insert(10, 0, 7, 5);
    idx.insert(20, 1, 7, 6);
    assert!(idx.evict_to_tier().is_some());
    assert!(idx.evict_to_tier().is_some());
    assert_eq!(idx.evict_to_tier(), None, "nothing resident left to spill");
    assert_eq!(idx.evict_lru(), None, "nothing resident left to drop");
}

/// Re-`insert` of a spilled prefix (a fresh HBM save) re-homes it to
/// resident — it must not stay marked tiered.
#[test]
fn reinsert_unspills() {
    let mut idx = SsmSnapshotIndex::new();
    idx.insert(0xAA, 1, 7, 40);
    idx.evict_to_tier().unwrap();
    // Fresh save of the same prefix into slot 5 re-homes it to resident.
    idx.insert(0xAA, 5, 7, 40);
    // The entry is resident again at slot 5; the drop path can free it.
    assert_eq!(idx.evict_lru(), Some(5));
}
