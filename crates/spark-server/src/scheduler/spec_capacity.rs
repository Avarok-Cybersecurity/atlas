// SPDX-License-Identifier: AGPL-3.0-only

//! Tiered verify-pool DISPATCH clamp (2026-08-16).
//!
//! The MTP verify state pools size each slot's per-token H intermediates to
//! the deepest draft count the STATIC ladder can hand a sequence occupying
//! it (`ssm_reserve::verify_slot_h_intermediates` — slots 0..8 keep the full
//! `--num-drafts` under the default `4:3,8:3,16:1,32:1` ladder, slots 8..
//! are sized for one draft). Two dispatchers can exceed that static bound:
//!
//! * a TRANSIENT contiguity break (LIFO free-list claim after churn) can
//!   park a sequence on a high slot while `n_active` is small, where the
//!   ladder offers a deeper K than the slot holds;
//! * `adaptive_rung` may LIFT n in 9..=16 to 2 drafts on tool-shaped accept
//!   stats, above the static rung the sizing derives from.
//!
//! Spec dispatch is all-or-nothing, so the invariant is: the step's draft
//! count must respect the MINIMUM capacity across the slots of the
//! currently-active sequences — a sequence in a K=2-sized slot must never
//! receive K=4 drafts. Capacities come from the model's ACTUAL pool
//! geometry (`Model::mtp_slot_draft_capacity`), not a re-derivation, so
//! sizing and dispatch cannot disagree; `ATLAS_MTP_POOL_FULL_WIDTH`
//! restores uniform full-K pools, which makes every capacity `num_drafts`
//! and this clamp vacuous. Consequence worth knowing: under the tiered
//! default the adaptive 16:2 lift is clamped back to K=2 whenever any
//! active sequence sits in a capacity-1 slot — i.e. at every n >= 9 under
//! contiguity — so re-enabling the lift requires the kill switch (or a
//! deeper explicit `ATLAS_MTP_K_LADDER`, which widens the tier with it).

/// Clamp a step's draft count to the minimum verify-slot capacity across
/// the active sequences. `usize::MAX` entries (no SSM verify pools) are
/// no-ops; an empty iterator leaves `drafts` unchanged.
pub(crate) fn clamp_drafts_to_slot_capacity(
    drafts: usize,
    slot_capacities: impl IntoIterator<Item = usize>,
) -> usize {
    slot_capacities
        .into_iter()
        .fold(drafts, |acc, cap| acc.min(cap))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_low_capacity_slot_bounds_the_whole_step() {
        // The named invariant: a sequence in a K=2-sized slot (capacity 1)
        // must never receive K=4 drafts — and dispatch is all-or-nothing,
        // so the whole step drops to its depth.
        assert_eq!(clamp_drafts_to_slot_capacity(3, [3, 3, 1]), 1);
        // Transient churn shape: one straggler on a high slot at small n.
        assert_eq!(clamp_drafts_to_slot_capacity(3, [1]), 1);
        // The adaptive-rung lift (2 drafts at n in 9..=16) is clamped by a
        // capacity-1 slot in the batch.
        assert_eq!(clamp_drafts_to_slot_capacity(2, [3, 1, 3]), 1);
    }

    #[test]
    fn full_capacity_slots_do_not_clamp() {
        assert_eq!(clamp_drafts_to_slot_capacity(3, [3, 3, 3]), 3);
        // Uniform / full-width pools report usize::MAX per slot.
        assert_eq!(clamp_drafts_to_slot_capacity(3, [usize::MAX; 4]), 3);
        // Pure-attention models have no active-slot constraint at all.
        assert_eq!(clamp_drafts_to_slot_capacity(2, std::iter::empty()), 2);
    }
}
