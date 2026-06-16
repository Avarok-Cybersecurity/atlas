// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::Arc;

use super::SsmStatePool;

/// RAII owner of a claimed SSM pool slot.
///
/// Stored on the owning [`crate::traits::SequenceState`]. While `idx` is
/// `Some(i)`, this guard is responsible for returning slot `i` to the pool's
/// free list. The slot is released on `Drop` UNLESS the explicit teardown path
/// has already neutralized the guard via [`take`](Self::take) (normal
/// `free_sequence`) or transferred it via [`migrate`](Self::migrate)
/// (slot-migration in `compact_sequence`). This makes the release happen
/// EXACTLY once on every exit path:
///   - normal finish / error / cancel / swap-out → `free_sequence` calls
///     `take()` then releases explicitly (one push);
///   - slot-migration → `compact_sequence` releases the OLD slot explicitly and
///     calls `migrate(new)` so the guard tracks the NEW slot;
///   - abort/early-return/panic where `free_sequence` is never reached →
///     `Drop` releases the still-`Some` slot (one push).
///
/// Because the explicit sites `take()` the idx before pushing, and `Drop` only
/// pushes when the idx is still `Some`, the same slot index is never pushed
/// twice — no double-release / `free_slots` corruption. `free_slots` is a
/// `parking_lot::Mutex`; the scheduler is single-threaded, so claim and release
/// never race, but the mutex keeps the EP-worker path sound regardless.
pub(crate) struct SlotGuard {
    pool: Arc<SsmStatePool>,
    idx: Option<usize>,
}

impl SlotGuard {
    /// Construct a guard from its raw parts. Used by
    /// [`SsmStatePool::claim_guarded`] which owns the claim/release contract.
    pub(super) fn from_parts(pool: Arc<SsmStatePool>, idx: Option<usize>) -> Self {
        Self { pool, idx }
    }

    /// A guard that owns no slot (released/migrated, or a placeholder for the
    /// reserved-dummy / sentinel paths). Holds an `Arc` to the pool but its
    /// `Drop` is a no-op while `idx` is `None`.
    pub(crate) fn empty(pool: Arc<SsmStatePool>) -> Self {
        Self { pool, idx: None }
    }

    /// The currently-owned claimable slot index, if any.
    #[inline]
    pub(crate) fn idx(&self) -> Option<usize> {
        self.idx
    }

    /// Neutralize the guard, returning the owned slot index (if any) WITHOUT
    /// releasing it. The caller becomes responsible for releasing exactly once
    /// (the explicit `free_sequence` path). After this the guard's `Drop` is a
    /// no-op, so there is no double-release.
    #[inline]
    pub(crate) fn take(&mut self) -> Option<usize> {
        self.idx.take()
    }

    /// Slot-migration: the guard's OLD slot has already been released by the
    /// caller (`compact_sequence`); point the guard at the NEW slot it now
    /// owns. Asserts the old slot was already taken so a stale idx cannot be
    /// silently leaked or double-released.
    #[inline]
    pub(crate) fn migrate(&mut self, new_idx: usize) {
        debug_assert!(
            self.idx.is_none(),
            "SlotGuard::migrate called before the old slot was released/taken"
        );
        self.idx = Some(new_idx);
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        if let Some(idx) = self.idx.take() {
            // Reached only when the sequence exited WITHOUT the explicit
            // teardown path neutralizing the guard (abort, early-return after
            // an owned `ActiveSeq` move, panic/unwind). Returns the slot to the
            // free list so the pool cannot leak itself into exhaustion.
            tracing::debug!("SlotGuard::drop releasing un-freed SSM slot {idx}");
            self.pool.release_slot(idx);
        }
    }
}

#[cfg(test)]
mod slot_guard_tests {
    use super::*;
    use crate::model::ssm_pool::SsmStatePool;
    use parking_lot::Mutex;

    /// Build a bare pool that touches ONLY the CPU-side slot bookkeeping
    /// (`free_slots`/`max_slots`). All GPU pointer vectors are empty; the guard
    /// path and `claim_slot`/`release_slot` never dereference them, so no GPU is
    /// required to validate the exactly-once release invariant.
    fn bare_pool(max_slots: usize) -> Arc<SsmStatePool> {
        Arc::new(SsmStatePool {
            h_state_pools: Vec::new(),
            conv_state_pools: Vec::new(),
            h_intermediate_pools: Vec::new(),
            conv_intermediate_pools: Vec::new(),
            h_checkpoint_pools: Vec::new(),
            conv_checkpoint_pools: Vec::new(),
            h_bytes: 0,
            conv_bytes: 0,
            max_slots,
            num_ssm_layers: 0,
            has_mtp: false,
            num_intermediates: 0,
            free_slots: Mutex::new((0..max_slots).rev().collect()),
        })
    }

    fn free_count(pool: &SsmStatePool) -> usize {
        pool.free_slots.lock().len()
    }

    #[test]
    fn guard_releases_on_drop() {
        let pool = bare_pool(2);
        let claimed;
        {
            let g = pool.claim_guarded().unwrap();
            // free_slots is `(0..max).rev()`, so `pop()` returns the LOWEST
            // index first (0) — matching the original `claim_slot` behavior.
            claimed = g.idx().expect("guard owns a slot");
            assert_eq!(claimed, 0);
            assert_eq!(free_count(&pool), 1);
        } // guard dropped (abort/panic surrogate) → slot returned
        assert_eq!(
            free_count(&pool),
            2,
            "drop must return the slot exactly once"
        );
        // The released slot is back in the free list (no phantom indices).
        assert!(pool.free_slots.lock().contains(&claimed));
    }

    #[test]
    fn take_neutralizes_drop_no_double_release() {
        let pool = bare_pool(2);
        let mut g = pool.claim_guarded().unwrap();
        let idx = g.take().expect("guard owns a slot");
        // Explicit teardown releases exactly once...
        pool.release_slot(idx);
        assert_eq!(free_count(&pool), 2);
        drop(g); // ...and the now-empty guard's Drop is a no-op (no double push)
        assert_eq!(
            free_count(&pool),
            2,
            "take() must make Drop a no-op (no double-release)"
        );
    }

    #[test]
    fn migration_releases_old_once_then_owns_new() {
        // Two live sequences so the migration target is a genuinely-claimed
        // slot (as in production), not one still sitting in the free list.
        let pool = bare_pool(2); // {0,1}
        let mut survivor = pool.claim_guarded().unwrap(); // owns 0 (pop → 0)
        let donor = pool.claim_guarded().unwrap(); // owns 1
        assert_eq!(free_count(&pool), 0);
        let donor_slot = donor.idx().unwrap();

        // Simulate compact_sequence(survivor, donor_slot): release survivor's
        // OLD slot and migrate it onto the donor's slot.
        let old = survivor.take().unwrap();
        pool.release_slot(old); // survivor's old slot released once
        // donor is being torn down; disown its slot WITHOUT releasing (survivor
        // takes it over). Mirrors detach_slot_for_reuse.
        let mut donor = donor;
        let _ = donor.take();
        drop(donor); // empty guard → no release
        survivor.migrate(donor_slot);
        assert_eq!(survivor.idx(), Some(donor_slot));

        // Free the survivor later: releases donor_slot exactly once.
        let final_idx = survivor.take().unwrap();
        pool.release_slot(final_idx);
        drop(survivor);

        let free = pool.free_slots.lock();
        let mut sorted = free.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1], "both slots free exactly once, no dupes");
    }

    #[test]
    fn retire_with_migration_is_double_free_free() {
        // Full scenario: retired R owns slot i; survivor S owns slot j; S is
        // compacted onto i; R is then disowned (detach_slot_for_reuse) and
        // freed. Then S is freed. Every slot must be released exactly once.
        let pool = bare_pool(2); // slots {0,1}
        let mut r = pool.claim_guarded().unwrap(); // R owns 1
        let mut s = pool.claim_guarded().unwrap(); // S owns 0
        assert_eq!(free_count(&pool), 0);
        let r_slot = r.idx().unwrap();
        let s_slot = s.idx().unwrap();

        // compact_sequence(S, r_slot): release S's old slot, migrate to R's slot.
        let old = s.take().unwrap();
        assert_eq!(old, s_slot);
        pool.release_slot(old); // j released once
        s.migrate(r_slot); // S now owns i

        // detach_slot_for_reuse(R): take WITHOUT release (S owns it now).
        let _ = r.take();
        drop(r); // R's guard is empty → no release of i

        // free_sequence(S) later: release i exactly once.
        let i = s.take().unwrap();
        pool.release_slot(i);
        drop(s);

        let free = pool.free_slots.lock();
        let mut sorted = free.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1], "both slots free, exactly once each");
    }
}
