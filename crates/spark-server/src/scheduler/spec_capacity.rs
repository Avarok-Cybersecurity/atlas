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
//! sizing and dispatch cannot disagree; `AVAROK_MTP_POOL_FULL_WIDTH`
//! restores uniform full-K pools, which makes every capacity `num_drafts`
//! and this clamp vacuous. Consequence worth knowing: under the tiered
//! default the adaptive 16:2 lift is clamped back to K=2 whenever any
//! active sequence sits in a capacity-1 slot — i.e. at every n >= 9 under
//! contiguity — so re-enabling the lift requires the kill switch (or a
//! deeper explicit `AVAROK_MTP_K_LADDER`, which widens the tier with it).

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

/// The verify arm the SERIAL (per-sequence) path runs for one sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SerialArm {
    DFlash,
    K4,
    K3,
    K2,
}

/// Serial-path twin of the batched path's surplus truncation (G25).
///
/// A verify re-proposes at the width it was handed, so a sequence can carry
/// more pending drafts than THIS step's clamped width `step_drafts` (the
/// ladder rung after [`clamp_drafts_to_slot_capacity`]). The batched path
/// truncates that surplus; the serial path used to dispatch on the pending
/// count against the serve-wide `--num-drafts`, so `AVAROK_MTP_K_LADDER=1:1`
/// with `--num-drafts 3` alternated K=2 -> re-propose 3 -> K=4 verify on a
/// slot sized for one draft ("SSM MTP intermediate buffers not allocated
/// (h_state_intermediates.len()=1 ... num_tokens=4)"), and the failed graph
/// capture poisoned every later request. Returns how many pending drafts to
/// keep and the arm to run; `step_drafts` is also the re-propose width the
/// caller hands that arm. DFlash γ-blocks keep their own width: the caller
/// passes the serve-wide count there and nothing is truncated.
pub(crate) fn serial_verify_plan(
    pending: usize,
    step_drafts: usize,
    dflash: bool,
) -> (usize, SerialArm) {
    let keep = if dflash {
        pending
    } else {
        pending.min(step_drafts)
    };
    let arm = if keep >= 4 {
        SerialArm::DFlash
    } else if step_drafts >= 3 && keep >= 3 {
        SerialArm::K4
    } else if step_drafts >= 2 && keep >= 2 {
        SerialArm::K3
    } else {
        SerialArm::K2
    };
    (keep, arm)
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

    #[test]
    fn a_ladder_below_num_drafts_never_reaches_the_k4_verify_at_n1() {
        // G25, the arm-k1 serve: `AVAROK_MTP_K_LADDER=1:1,...`, `--num-drafts
        // 3`, one sequence. Slot 0 is sized for the ladder's one draft, so the
        // step width is 1; the previous K=2 verify re-proposed three drafts.
        let step = clamp_drafts_to_slot_capacity(1, [1]);
        assert_eq!(serial_verify_plan(3, step, false), (1, SerialArm::K2));
        assert_eq!(serial_verify_plan(2, step, false), (1, SerialArm::K2));
        // A two-draft rung keeps two of three and runs K=3.
        assert_eq!(serial_verify_plan(3, 2, false), (2, SerialArm::K3));
    }

    #[test]
    fn a_full_width_step_dispatches_as_before() {
        assert_eq!(serial_verify_plan(3, 3, false), (3, SerialArm::K4));
        assert_eq!(serial_verify_plan(2, 3, false), (2, SerialArm::K3));
        assert_eq!(serial_verify_plan(1, 3, false), (1, SerialArm::K2));
        // A grammar-truncated sequence carries fewer drafts than the width.
        assert_eq!(serial_verify_plan(0, 3, false), (0, SerialArm::K2));
        // DFlash γ-blocks are never truncated to an MTP width.
        assert_eq!(serial_verify_plan(16, 16, true), (16, SerialArm::DFlash));
        assert_eq!(serial_verify_plan(3, 3, true), (3, SerialArm::K4));
    }
}
