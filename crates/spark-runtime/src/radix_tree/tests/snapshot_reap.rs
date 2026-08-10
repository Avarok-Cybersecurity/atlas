// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for tier-key REAPING (`forget_tiered` / `forget_snapshot_tier_key`) —
//! retiring an index entry whose tier blob is gone, so a capped disk cannot
//! thrash. Split out of `snapshot.rs` to keep both files under the 500-LoC cap.

use super::super::*;
use crate::prefix_cache::PrefixCache;

/// Phase 1b end-to-end trait surface: a snapshot's location transitions
/// resident → spilled → resident through the `PrefixCache` API, and
/// `lookup` reports each state correctly (the serving path's contract).
#[test]
fn test_spill_tier_lookup_transitions() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..64).collect();
    // Register a leaf snapshot (slot 99) for this prefix, session 7.
    tree.insert_with_snapshot(&tokens, &[10, 20, 30, 40], &[], 16, 99, 7, 0, 0);

    // Resident: lookup returns the HBM slot, no tier key.
    let m = tree.lookup(&tokens, 16, 7, 0);
    assert_eq!(m.ssm_snapshot, Some(99));
    assert_eq!(m.ssm_snapshot_tokens, 64);
    assert_eq!(m.ssm_snapshot_tier_key, None);

    // Spill (gate off): evict_to_tier KEEPS the entry, reports slot + key.
    let ev = tree.evict_snapshot_to_tier(0).expect("resident victim");
    let (freed, key) = match ev {
        crate::prefix_cache::TierEvict::Spill { slot, key, .. } => (slot, key),
        other => panic!("an ungated evict must SPILL, not drop: {other:?}"),
    };
    assert_eq!(freed, 99, "the resident slot is freed for reuse");

    // Spilled: lookup now reports the anchor as tiered — ssm_snapshot is None
    // (no HBM slot) and the tier key is present, so the caller faults it in.
    let m = tree.lookup(&tokens, 16, 7, 0);
    assert_eq!(m.ssm_snapshot, None, "no resident slot while spilled");
    assert_eq!(m.ssm_snapshot_tier_key, Some(key));
    assert_eq!(m.ssm_snapshot_tier_tokens, 64);

    // Fault-in completes into slot 123; promote re-homes the entry to HBM.
    assert!(tree.promote_snapshot(key, 123));

    // Resident again at the new slot.
    let m = tree.lookup(&tokens, 16, 7, 0);
    assert_eq!(m.ssm_snapshot, Some(123));
    assert_eq!(m.ssm_snapshot_tier_key, None);

    // REAP branch — the third transition, spilled → gone. Spill once more,
    // then retire the entry the way a fault-in miss does when the disk cap has
    // dropped its blob. The anchor must stop advertising the dead key AND offer
    // no resident slot, i.e. the prefix degrades to a plain recompute instead
    // of re-offering the key (and re-spilling a live victim) every warm turn.
    let ev = tree.evict_snapshot_to_tier(0).expect("resident victim");
    let key = match ev {
        crate::prefix_cache::TierEvict::Spill { key, .. } => key,
        other => panic!("an ungated evict must SPILL, not drop: {other:?}"),
    };
    assert!(
        tree.forget_snapshot_tier_key(key),
        "a tiered entry is reapable"
    );
    let m = tree.lookup(&tokens, 16, 7, 0);
    assert_eq!(
        m.ssm_snapshot_tier_key, None,
        "the dead key is not re-offered"
    );
    assert_eq!(m.ssm_snapshot, None, "and it has no resident slot either");
    tree.release(&tokens, 16, 0);
}

/// A caches-without-a-tier default: `evict_snapshot_to_tier` / `promote_snapshot`
/// / `forget_snapshot_tier_key` no-op on the trait default (NoPrefixCaching),
/// so a non-radix cache is safe.
#[test]
fn test_no_tier_default_impl() {
    use crate::prefix_cache::NoPrefixCaching;
    let c = NoPrefixCaching;
    assert_eq!(c.evict_snapshot_to_tier(/*min_tokens*/ 0), None);
    assert!(!c.promote_snapshot(123, 0));
    assert!(!c.forget_snapshot_tier_key(123));
}

/// Index-layer helper: one entry for `tokens`, spilled to the tier, returning
/// the index and its tier key. Mirrors what `evict_to_tier` leaves behind.
fn spilled_index(tokens: &[u32], slot: usize) -> (SsmSnapshotIndex, u64) {
    let mut idx = SsmSnapshotIndex::new();
    let ph = hash_token_prefix(tokens, tokens.len(), 0);
    idx.insert(ph, slot, /*sess*/ 7, tokens.len());
    let ev = idx
        .evict_to_tier(/*min_tokens*/ 0)
        .expect("resident victim");
    match ev {
        crate::prefix_cache::TierEvict::Spill { key, .. } => (idx, key),
        other => panic!("an ungated evict must SPILL: {other:?}"),
    }
}

/// A tiered entry is reapable, and the reap is what stops the thrash: after it
/// the prefix has no anchor at all, so the next warm turn plainly recomputes
/// instead of spilling a live snapshot to chase a dead key.
#[test]
fn forget_tiered_removes_a_tiered_entry() {
    let tokens: Vec<u32> = (0..32).collect();
    let (mut idx, key) = spilled_index(&tokens, 42);
    assert!(idx.forget_tiered(key));
    assert_eq!(idx.len(), 0);
    assert_eq!(idx.lookup_tiered(&tokens, 32, 7, 0), None);
}

/// **The slot-leak guard, and the promote-then-reap race.** A RESIDENT entry's
/// `snapshot_id` is a live pool slot that only its owner may free; `forget` has
/// no caller to hand it back to (unlike `evict_lru`), so removing one would
/// leak the slot for the process's lifetime. Reaping a key that was promoted
/// back to HBM between the miss and the forget must therefore be a clean no-op.
#[test]
fn forget_tiered_refuses_a_resident_entry() {
    let tokens: Vec<u32> = (0..32).collect();
    let (mut idx, key) = spilled_index(&tokens, 42);
    // The race: a concurrent fault-in promoted it back to HBM at slot 123.
    assert!(idx.promote(key, 123));

    assert!(!idx.forget_tiered(key), "a resident entry must survive");
    assert_eq!(idx.len(), 1);
    let m = idx
        .lookup_tiered(&tokens, 32, 7, 0)
        .expect("still findable");
    assert_eq!(
        m.loc,
        super::super::snapshot::SnapLoc::Hbm(123),
        "its live slot must be untouched — reaping it would leak the slot"
    );
}

/// An unknown key is `false`, and so is a second reap of the same key — the
/// caller gates `store.remove` on this return, so a duplicate reap cannot
/// delete anything.
///
/// NOTE what this does NOT buy: if the key were re-spilled between the two
/// reaps the entry would be TIERED again and the second reap WOULD remove it
/// (blob lost -> recompute; no corruption, no slot leak). The guard against
/// that is the single-threaded scheduler, not idempotence. Only a
/// generation-stamped tier key would close it, and that is not needed while
/// the SSM path stays single-threaded.
#[test]
fn forget_tiered_unknown_key_is_false() {
    let tokens: Vec<u32> = (0..32).collect();
    let (mut idx, key) = spilled_index(&tokens, 42);
    assert!(!idx.forget_tiered(0xDEAD_BEEF), "unknown key");
    assert!(idx.forget_tiered(key));
    assert!(!idx.forget_tiered(key), "second reap is a no-op");
}

/// A reap frees no slot and applies no pressure to the live session's restore
/// point, so it must not be counted as an eviction — counting it would shorten
/// the tail lease for no reason. It gets its own counter instead.
#[test]
fn forget_tiered_does_not_count_as_an_eviction() {
    let tokens: Vec<u32> = (0..32).collect();
    let (mut idx, key) = spilled_index(&tokens, 42);
    let evictions = idx.stats.evictions;
    let since_lookup = idx.evictions_since_lookup;

    assert!(idx.forget_tiered(key));
    assert_eq!(idx.stats.evictions, evictions, "a reap is not an eviction");
    assert_eq!(idx.evictions_since_lookup, since_lookup);
    assert_eq!(idx.stats.tier_reaps, 1, "it has its own counter");
}
