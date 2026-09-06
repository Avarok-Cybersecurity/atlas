// SPDX-License-Identifier: AGPL-3.0-only

//! The insert paths crossed with the spill tier. Split from
//! `snapshot_index.rs` (file-size cap); uses that module's helpers.
//!
//! # What these pin
//!
//! A `tiered` entry's `snapshot_id` is STALE. `evict_to_tier` hands that slot
//! back once (`TierEvict::Spill { slot, .. }`) and the model frees it; the
//! entry lives on in the index purely as a fault-in record. Every consumer
//! honours that — `lookup` skips tiered entries, and both victim scans pass
//! `skip_tiered = true` — except, until this change, the three insert paths,
//! which handed the stale id back to be freed a SECOND time.
//!
//! `SsmSnapshotPool::free` is `free_slots.lock().push(slot)` with no
//! membership check (unlike `ssm_pool::release_slot`, which at least
//! `debug_assert!`s), so the double-free is silent in release: the pool then
//! hands one slot to two sequences and they share a GDN/conv state buffer.
//! Wrong output, no error — the exact failure `ssm_pool::claim_specific`
//! exists to prevent on the KV side.
//!
//! Reachability is ordinary warm traffic, not a corner: spill a prefix, then
//! checkpoint that same prefix again on a later turn. Agentic sessions with a
//! stable system prompt do this every turn, which is precisely the workload
//! the tier is built for.

use super::super::*;
use super::index;
use crate::prefix_cache::TierEvict;

/// A spilled entry, ready to be overwritten by a fresh save of the same
/// prefix. Returns `(index, prefix_hash, the slot the spill already freed)`.
fn spilled(prefix_hash: u64, slot: usize) -> (SsmSnapshotIndex, u64, usize) {
    let mut idx = index(
        vec![SnapshotEntry {
            snapshot_id: slot,
            session_hash: 7,
            token_count: 16000,
            prefix_hash,
            last_access: 50,
            tiered: false,
            is_tail: false,
            is_tail_sibling: false,
        }],
        7,
    );
    let TierEvict::Spill { slot: freed, .. } = idx.evict_to_tier(0).expect("a victim exists")
    else {
        panic!("an ungated evict must SPILL, not drop");
    };
    assert_eq!(freed, slot, "the spill freed exactly this slot");
    (idx, prefix_hash, freed)
}

/// Plain `insert` over a spilled entry must hand back NOTHING. Before the
/// fix it returned `Some(freed)` — the id the spill had already released.
#[test]
fn insert_over_a_spilled_entry_hands_back_no_slot() {
    let (mut idx, ph, freed) = spilled(0xA1, 4);
    let displaced = idx.insert(
        ph, /*new slot*/ 11, /*session*/ 7, /*tok*/ 16000,
    );
    assert_eq!(
        displaced, None,
        "slot {freed} was already freed at spill time; returning it again \
         double-frees it into the snapshot pool"
    );
}

/// `insert_tail` over a spilled entry: same double-free, via `displaced`.
#[test]
fn insert_tail_over_a_spilled_entry_hands_back_no_slot() {
    let (mut idx, ph, freed) = spilled(0xB2, 5);
    let displaced = idx.insert_tail(
        ph, /*new slot*/ 12, /*session*/ 7, /*tok*/ 16000,
    );
    assert!(
        displaced.is_empty(),
        "expected no slot to free, got {displaced:?} (slot {freed} was freed at spill time)"
    );
}

/// `insert_tail` was also the one overwrite arm that never cleared `tiered`.
/// An entry left `tiered` while holding a LIVE slot is skipped by `lookup`
/// and by both victim scans, so that slot is reachable by nothing and
/// freeable by nothing — a permanent leak of a scarce pool slot, on top of
/// `lookup_tiered` faulting in bytes that no longer describe the entry.
#[test]
fn insert_tail_rehomes_a_spilled_entry_to_hbm() {
    let (mut idx, ph, _) = spilled(0xC3, 6);
    idx.insert_tail(
        ph, /*new slot*/ 13, /*session*/ 7, /*tok*/ 16000,
    );

    assert!(
        !idx.entries[0].tiered,
        "a fresh HBM save re-homes the prefix; leaving it `tiered` strands slot 13"
    );
    // Reachable again: the plain (non-tier) victim scan can now free it.
    assert_eq!(
        idx.evict_lru(),
        Some(13),
        "a re-homed entry must be evictable — otherwise its slot leaks for the process lifetime"
    );
}

/// `insert_tail_sibling` — the third path, same contract, AND the same
/// re-home obligation `insert_tail` carries (see the test above).
///
/// The re-home half was the suite's blind spot: deleting `entry.tiered =
/// false` from this arm survived the WHOLE 86-check `radix_tree` registry.
/// `reinsert_unspills` covers the plain-`insert` arm and
/// `insert_tail_rehomes_a_spilled_entry_to_hbm` covers the tail arm; nothing
/// covered this one, so a sibling save over a spilled prefix could strand its
/// fresh slot — skipped by `lookup` and by both victim scans, hence reachable
/// by nothing and freeable by nothing — for the process lifetime.
#[test]
fn insert_tail_sibling_over_a_spilled_entry_hands_back_no_slot() {
    let (mut idx, ph, freed) = spilled(0xD4, 8);
    let displaced = idx.insert_tail_sibling(
        ph, /*new slot*/ 14, /*session*/ 7, /*tok*/ 16000,
    );
    assert_eq!(
        displaced, None,
        "slot {freed} was already freed at spill time"
    );
    assert!(
        !idx.entries[0].tiered,
        "a fresh HBM save re-homes the prefix; leaving it `tiered` strands slot 14"
    );
    assert_eq!(
        idx.evict_lru(),
        Some(14),
        "a re-homed sibling must be evictable — otherwise its slot leaks"
    );
}

/// ★ The SUPERSEDE SWEEP — the site my first six tests did not reach.
///
/// `insert_tail` has TWO displacement sites: the sweep that clears this
/// session's previous tail/sibling, and the overwrite of a matching
/// `prefix_hash`. `spilled()` builds a NON-tail entry, so every earlier test
/// falls straight past the sweep into the overwrite. Reverting only the sweep
/// line left all six green — verified, not assumed.
///
/// The sweep is reachable in the DEFAULT configuration: `ATLAS_SSM_TAIL_MIDCHUNK`
/// is default-ON, so `is_tail` entries exist; `session_aware_victim` will spill
/// a tail once its lease lapses or its session goes dormant; and the next
/// `finalize_midchunk_capture` for that session runs the sweep — i.e. every
/// turn of every multi-turn conversation.
#[test]
fn the_tail_supersede_sweep_does_not_hand_back_a_spilled_tail() {
    // A TAIL entry for session 7, then spilled.
    let mut idx = index(
        vec![SnapshotEntry {
            snapshot_id: 41,
            session_hash: 7,
            token_count: 16000,
            prefix_hash: 0xF1,
            last_access: 50,
            tiered: false,
            is_tail: true,
            is_tail_sibling: false,
        }],
        7,
    );
    let TierEvict::Spill { slot: freed, .. } = idx.evict_to_tier(0).expect("a victim exists")
    else {
        panic!("an ungated evict must SPILL, not drop");
    };
    assert_eq!(freed, 41);
    assert!(idx.entries[0].tiered, "the victim is now tiered");

    // A NEW tail for the same session at a DIFFERENT prefix: the supersede
    // sweep removes the old (tiered) one. Its slot was freed at spill time.
    let displaced = idx.insert_tail(
        /*ph*/ 0xF2, /*slot*/ 42, /*session*/ 7, /*tok*/ 17000,
    );
    assert!(
        displaced.is_empty(),
        "the sweep must not hand back slot {freed} — it was freed at spill time, and by now \
         the pool has handed it to a live fault-in target; got {displaced:?}"
    );
}

/// The guard is scoped to TIERED entries only: a resident entry being
/// overwritten must still hand its live slot back, or that slot leaks.
/// Without this, "return None" would be a trivially-passing fix that traded
/// a double-free for a leak.
#[test]
fn insert_over_a_resident_entry_still_hands_back_its_slot() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..40).collect();
    let ph = super::super::hash_token_prefix(&toks, 40, 0);
    idx.insert(ph, /*slot*/ 21, /*session*/ 7, /*tok*/ 40);

    assert_eq!(
        idx.insert(ph, /*slot*/ 22, /*session*/ 7, /*tok*/ 40),
        Some(21),
        "the displaced LIVE slot must still be returned for the caller to free"
    );
}

/// Same scoping check on the tail sweep: superseding a session's live tail
/// must still yield its slot.
#[test]
fn insert_tail_still_displaces_a_resident_tail() {
    let mut idx = SsmSnapshotIndex::new();
    idx.insert_tail(
        /*ph*/ 0xE5, /*slot*/ 31, /*session*/ 7, /*tok*/ 100,
    );
    // A different prefix, same session → the supersede sweep removes the old
    // tail and must hand back its (resident) slot.
    let displaced = idx.insert_tail(
        /*ph*/ 0xE6, /*slot*/ 32, /*session*/ 7, /*tok*/ 200,
    );
    assert_eq!(displaced, vec![31]);
}
