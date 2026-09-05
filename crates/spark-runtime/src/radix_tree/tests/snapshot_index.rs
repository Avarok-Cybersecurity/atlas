// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::prefix_cache::TierEvict;

#[path = "snapshot_lease.rs"]
mod lease;

#[path = "snapshot_insert_tier.rs"]
mod insert_tier;

/// Build an entry with an explicit recency profile. `snapshot_id` doubles
/// as a stable identity we assert on (independent of Vec index).
fn entry(
    snapshot_id: usize,
    session_hash: u64,
    token_count: usize,
    last_access: u64,
) -> SnapshotEntry {
    SnapshotEntry {
        snapshot_id,
        session_hash,
        token_count,
        prefix_hash: snapshot_id as u64, // unique, irrelevant to victim choice
        last_access,
        tiered: false,
        is_tail: false,
        is_tail_sibling: false,
    }
}

/// An `is_tail` restore-point entry (the per-session mid-chunk tail).
fn tail_entry(
    snapshot_id: usize,
    session_hash: u64,
    token_count: usize,
    last_access: u64,
) -> SnapshotEntry {
    SnapshotEntry {
        is_tail: true,
        ..entry(snapshot_id, session_hash, token_count, last_access)
    }
}

fn index(entries: Vec<SnapshotEntry>, live: u64) -> SsmSnapshotIndex {
    SsmSnapshotIndex {
        entries,
        access_counter: 1000,
        last_lookup_session: live,
        evictions_since_lookup: 0,
        stats: SnapshotStats::default(),
    }
}

/// Pure-LRU within a session once the lease is DISARMED: the older entry is
/// the victim even when it is the live session's deep `is_tail` restore point.
/// id 9 has to be a tail — with a pool of plain entries `tail_protect` is
/// inert and the check says nothing about the flag it is named after.
#[test]
fn deep_tail_evicted_without_tail_protect() {
    let idx = index(
        vec![
            entry(
                /*id*/ 7, /*sess*/ 1, /*tok*/ 8192, /*last*/ 100,
            ),
            tail_entry(
                /*id*/ 9, /*sess*/ 1, /*tok*/ 16000, /*last*/ 50,
            ),
        ],
        1,
    );
    // Victim is the OLDER entry (id 9) under pure LRU.
    let v = idx.session_aware_victim(false, false).unwrap();
    assert_eq!(idx.entries[v].snapshot_id, 9);
    // Control on the SAME pool: armed, the lease spares that tail. Without
    // this pair, a `leased` predicate that ignores `tail_protect` entirely is
    // indistinguishable from one that honours it.
    let armed = idx.session_aware_victim(true, false).unwrap();
    assert_eq!(idx.entries[armed].snapshot_id, 7);
}

// ─────────── 07-10 fossil-pinning regressions (re-landed, #317 revert) ──────────

/// Eviction must ignore hit HISTORY: an entry selected many times but not
/// recently loses to entries touched after it. Under the reverted
/// `last_access * (1 + hit_count)` score, A's 5 hits made it unbeatable
/// (escore 5*6=... vs fresh saves) and each new save evicted the previous
/// fresh save — the frozen-anchor pathology. Exercises the SERVING path
/// (`lookup_tiered`), not the dead `lookup`.
#[test]
fn eviction_ignores_hit_history() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..100).collect();
    let ph = super::hash_token_prefix(&toks, 40, 0);
    idx.insert(ph, /*slot*/ 1, /*session*/ 7, /*tok*/ 40);
    // A never-hit entry saved AFTER the anchor: fresher on save order alone.
    let ph50 = super::hash_token_prefix(&toks, 50, 0);
    idx.insert(ph50, /*slot*/ 4, /*session*/ 7, /*tok*/ 50);
    // Hit the anchor 5 times. `matched_tokens = 40` keeps the depth-50 entry
    // out of range, so only slot 1 is touched.
    for _ in 0..5 {
        assert!(idx.lookup_tiered(&toks, 40, 7, 0).is_some());
    }
    // Those hits must have MOVED recency: slot 1 is now fresher than the
    // never-hit slot 4, so slot 4 dies first. Without this the five lookups
    // above are decorative — the assertion below holds whether or not a hit
    // bumps `last_access` at all.
    assert_eq!(idx.evict_lru(), Some(4), "a hit must refresh recency");
    // Two fresh saves AFTER the last hit — strictly more recent.
    let ph80 = super::hash_token_prefix(&toks, 80, 0);
    let ph90 = super::hash_token_prefix(&toks, 90, 0);
    idx.insert(ph80, 2, 7, 80);
    idx.insert(ph90, 3, 7, 90);
    // The victim must be the OLDEST entry (the much-hit anchor), never a
    // fresh save. The old hit-weighted score inverted this.
    assert_eq!(idx.evict_lru(), Some(1), "hit history must not pin fossils");
}

/// The lookup scan must be side-effect-free for LOSING candidates: only the
/// winner's recency moves. (The pre-fix scan bumped every improving
/// candidate, keeping shallow early-prefix entries eternally fresh.)
#[test]
fn lookup_bumps_winner_only() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..100).collect();
    let ph40 = super::hash_token_prefix(&toks, 40, 0);
    let ph80 = super::hash_token_prefix(&toks, 80, 0);
    idx.insert(ph40, 1, 7, 40); // shallow — the improving-chain fossil
    idx.insert(ph80, 2, 7, 80); // deep — the winner
    let shallow_before = idx
        .entries
        .iter()
        .find(|e| e.snapshot_id == 1)
        .unwrap()
        .last_access;
    // Deep lookup walks past the shallow candidate to select the deep one.
    let m = idx.lookup_tiered(&toks, 100, 7, 0).expect("hit");
    assert_eq!(m.token_count, 80, "deep entry wins");
    let shallow_after = idx
        .entries
        .iter()
        .find(|e| e.snapshot_id == 1)
        .unwrap()
        .last_access;
    assert_eq!(
        shallow_before, shallow_after,
        "losing candidate's recency must not move"
    );
    // Same contract on the reference (non-tier) lookup.
    let m2 = idx.lookup(&toks, 100, 7, 0).expect("hit");
    assert_eq!(m2.1, 80);
    let shallow_final = idx
        .entries
        .iter()
        .find(|e| e.snapshot_id == 1)
        .unwrap()
        .last_access;
    assert_eq!(shallow_before, shallow_final);
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
    // A sessionless (single-turn) lookup must not steal the latch.
    let _ = idx.lookup(&[1, 2, 3], 3, /*session*/ 0, /*adapter*/ 0);
    assert_eq!(idx.last_lookup_session, 42, "session 0 must not latch");
    // The SERVING path latches too, and renews the lease by clearing the
    // eviction counter. `lookup` is the retained reference implementation;
    // production arms the tail lease from `lookup_tiered`, so asserting only
    // on `lookup` leaves the latch that actually runs unprotected.
    idx.evictions_since_lookup = 7;
    let _ = idx.lookup_tiered(&[1, 2, 3], 3, /*session*/ 99, /*adapter*/ 0);
    assert_eq!(idx.last_lookup_session, 99);
    assert_eq!(
        idx.evictions_since_lookup, 0,
        "a live lookup renews the lease"
    );
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
    // id 3 = fresh 8192 anchor (recency 100); id 9 = cold deep tail (recency 50).
    let mut idx = index(vec![entry(3, 1, 8192, 100), entry(9, 1, 16000, 50)], 1);
    let before = idx.len();
    let TierEvict::Spill {
        slot: freed_slot,
        key,
        ..
    } = idx
        .evict_to_tier(/*min_tokens*/ 0)
        .expect("a resident victim exists")
    else {
        panic!("an ungated evict must SPILL, not drop");
    };
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

/// The SPILL-side cost gate. A victim shallower than `min_tokens` must be
/// DROPPED — entry removed, no tier key, `tier_spills` untouched — not left
/// marked `tiered` with no bytes behind it, which would make every warm turn
/// pay a blob-sized `store.get` to discover a miss.
#[test]
fn shallow_victim_is_dropped_not_spilled() {
    // Single 100-token entry, gate at 1024 → too shallow to repay a spill.
    let mut idx = index(vec![entry(9, 1, 100, 50)], 1);
    let before = idx.len();
    let ev = idx
        .evict_to_tier(/*min_tokens*/ 1024)
        .expect("a victim exists");
    assert_eq!(
        ev,
        TierEvict::Drop {
            slot: 9,
            depth: 100
        }
    );
    assert_eq!(idx.len(), before - 1, "entry REMOVED, not kept findable");
    assert_eq!(idx.stats.tier_spills, 0, "a dropped victim is not a spill");
    assert_eq!(idx.stats.evictions, 1, "it is a plain eviction");
    // Nothing left to evict — and nothing tiered was left behind.
    assert_eq!(idx.evict_to_tier(1024), None);
}

/// Same victim, deep enough: spilled and still findable.
#[test]
fn deep_victim_is_spilled_under_the_gate() {
    let mut idx = index(vec![entry(9, 1, 16000, 50)], 1);
    let ev = idx
        .evict_to_tier(/*min_tokens*/ 1024)
        .expect("a victim exists");
    assert_eq!(
        ev,
        TierEvict::Spill {
            slot: 9,
            key: 9,
            depth: 16000
        }
    );
    assert_eq!(idx.len(), 1, "entry kept, findable for fault-in");
    assert_eq!(idx.stats.tier_spills, 1);
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
    let TierEvict::Spill {
        slot: freed, key, ..
    } = idx.evict_to_tier(0).unwrap()
    else {
        panic!("an ungated evict must SPILL, not drop");
    };
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
    idx.evict_to_tier(0).unwrap();

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
    assert!(idx.evict_to_tier(0).is_some());
    assert!(idx.evict_to_tier(0).is_some());
    assert_eq!(idx.evict_to_tier(0), None, "nothing resident left to spill");
    assert_eq!(idx.evict_lru(), None, "nothing resident left to drop");
}

/// Re-`insert` of a spilled prefix (a fresh HBM save) re-homes it to
/// resident — it must not stay marked tiered.
#[test]
fn reinsert_unspills() {
    let mut idx = SsmSnapshotIndex::new();
    idx.insert(0xAA, 1, 7, 40);
    idx.evict_to_tier(0).unwrap();
    // Fresh save of the same prefix into slot 5 re-homes it to resident.
    idx.insert(0xAA, 5, 7, 40);
    // The entry is resident again at slot 5; the drop path can free it.
    assert_eq!(idx.evict_lru(), Some(5));
}

// ── SsmSnapshotIndex tests ──

#[test]
fn test_snapshot_index_insert_lookup_roundtrip() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..32).collect();
    let prefix_hash = super::hash_token_prefix(&tokens, 32, 0);

    assert!(idx.insert(prefix_hash, 42, 100, 32).is_none());
    let result = idx.lookup(&tokens, 32, 100, 0);
    assert_eq!(result, Some((42, 32)));
}

#[test]
fn test_snapshot_index_lru_eviction() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens_a: Vec<u32> = (0..16).collect();
    let tokens_b: Vec<u32> = (100..116).collect();
    let ha = super::hash_token_prefix(&tokens_a, 16, 0);
    let hb = super::hash_token_prefix(&tokens_b, 16, 0);

    idx.insert(ha, 1, 0, 16); // older
    idx.insert(hb, 2, 0, 16); // newer

    // LRU eviction should evict snapshot 1 (older)
    let evicted = idx.evict_lru();
    assert_eq!(evicted, Some(1));
    assert_eq!(idx.len(), 1);

    // Only snapshot 2 remains
    let evicted = idx.evict_lru();
    assert_eq!(evicted, Some(2));
    assert_eq!(idx.len(), 0);

    // Empty
    assert_eq!(idx.evict_lru(), None);
}

#[test]
fn test_snapshot_index_session_isolation() {
    // Encodes the POST-2026-07-16 session contract, which is deliberately
    // asymmetric:
    //
    //  * PLAIN (non-tail) entries are content-addressed — a pure function of
    //    the verified token prefix — and are therefore SAFE and INTENDED to be
    //    shared across sessions. (The pre-fix version of this test asserted
    //    they were session-gated; that was the OLD semantics, and gating them
    //    would resurrect the cold-recompute cost the prefix cache exists to
    //    avoid.)
    //  * TAIL entries capture state that bleeds past their advertised
    //    token_count (mid-chunk capture), so they are gated to their OWN
    //    non-zero session — reusing another session's tail is exactly the
    //    cross-request corruption that garbled BFCL tool calls (77.31 vs 84.54
    //    normalized) before the fix in `lookup`.
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..16).collect();
    let prefix_hash = super::hash_token_prefix(&tokens, 16, 0);

    // Plain insert for session 100: cross-session lookup MUST match — the
    // entry is content-addressed and session-free by design.
    idx.insert(prefix_hash, 42, 100, 16);
    assert_eq!(idx.lookup(&tokens, 16, 200, 0), Some((42, 16)));
    assert_eq!(idx.lookup(&tokens, 16, 100, 0), Some((42, 16)));
    // Session-less lookups (single-turn requests hash to 0) also match plain
    // entries.
    assert_eq!(idx.lookup(&tokens, 16, 0, 0), Some((42, 16)));

    // Re-home the same prefix as session 100's TAIL: now the session gate
    // applies. Session 200 and session-less lookups must fall through to a
    // recompute rather than restore another session's tail.
    idx.insert_tail(prefix_hash, 43, 100, 16);
    assert_eq!(idx.lookup(&tokens, 16, 200, 0), None);
    assert_eq!(idx.lookup(&tokens, 16, 0, 0), None);
    // The owning session still restores its own tail.
    let result = idx.lookup(&tokens, 16, 100, 0);
    assert_eq!(result, Some((43, 16)));
}

#[test]
fn test_snapshot_index_overwrite_existing() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..16).collect();
    let prefix_hash = super::hash_token_prefix(&tokens, 16, 0);

    // Insert first
    assert!(idx.insert(prefix_hash, 5, 0, 16).is_none());
    assert_eq!(idx.len(), 1);

    // Overwrite same prefix_hash — returns old snapshot_id
    let old = idx.insert(prefix_hash, 8, 0, 16);
    assert_eq!(old, Some(5));
    assert_eq!(idx.len(), 1); // still 1 entry, not 2

    // Lookup returns new value
    let result = idx.lookup(&tokens, 16, 0, 0);
    assert_eq!(result, Some((8, 16)));
}

#[test]
fn test_snapshot_index_deepest_match() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..64).collect();

    // Snapshot at token 16
    let h16 = super::hash_token_prefix(&tokens, 16, 0);
    idx.insert(h16, 10, 0, 16);

    // Snapshot at token 32
    let h32 = super::hash_token_prefix(&tokens, 32, 0);
    idx.insert(h32, 20, 0, 32);

    // Lookup with 48 matched tokens — deepest snapshot at 32 wins
    let result = idx.lookup(&tokens, 48, 0, 0);
    assert_eq!(result, Some((20, 32)));

    // Lookup with 20 matched tokens — only snapshot at 16 qualifies
    let result = idx.lookup(&tokens, 20, 0, 0);
    assert_eq!(result, Some((10, 16)));
}
