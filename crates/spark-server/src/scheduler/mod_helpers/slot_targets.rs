// SPDX-License-Identifier: AGPL-3.0-only

//! Pure target selection for SSM pool-slot compaction. Split from
//! `mod_helpers.rs` (500-LoC cap) so the rule is unit-testable without a
//! `Model`, a GPU or an `ActiveSeq`.
//!
//! ## Why this is a separate, tested rule (#1002, H100 round 13)
//!
//! `compact_survivors_into_range` packs the DECODING sequences into
//! contiguous pool slots `[0..n)` so the batched GDN recurrence keeps its
//! "slots are consecutive in slice order" precondition. It used to derive
//! the free targets from the survivors alone:
//!
//! ```text
//! free_targets = (0..n) \ { a.seq.slot_idx : a in survivors }
//! ```
//!
//! That set is only correct while the decoding sequences are the ONLY
//! owners of pool slots. They are not. A sequence claims its slot at
//! admission (`alloc_sequence_state`), so every stream sitting in the
//! scheduler's `prefilling` queue — mid chunked prefill, not yet decoding —
//! owns one too, and those slots are invisible to the formula above.
//!
//! Round 13 cell V made that live. `--prefill-varlen-batch` makes
//! `phase_start_prefills` DEFER chunk 0 into `prefilling` so co-arriving
//! prompts can batch (`want_varlen_defer`), which is the first configuration
//! that parks fourteen slot-owning streams there while two decode. The
//! per-tick compaction then read `n = 2`, saw the survivor at slot 7 as
//! out-of-range, and migrated it onto slot 1 — owned by a prefilling stream.
//! The serve log caught the migration in the act, 30 ms apart:
//!
//! ```text
//! Captured CUDA graph for batch size 2 (n=2, slots=Some([0, 7]))
//! Captured CUDA graph for batch size 2 (n=2, slots=Some([0, 1]))
//! ```
//!
//! Two live sequences then shared one GDN `h_state`/`conv_state`, which is
//! cross-stream state bleed: cell V logged 24 content-loop-watchdog fires, 2
//! fuzzy-repetition stops and 5 SimHash stops against ZERO on cells A, D, T1
//! and T2, and 6/16 probe responses were cut at 49 tokens. Worse, the
//! collision outlives the burst — both owners later release the same index,
//! double-pushing it onto the pool free list, so `claim_slot` hands it to two
//! fresh sequences for the rest of the serve. That is why the 16-way probe
//! four minutes later still decoded with `slots=[0, 0, 0, 1, 1, 2, 2, 3, ...]`
//! even though varlen never engaged for it.
//!
//! The rule therefore takes the slots held by NON-survivors as an explicit
//! input, and a survivor with no legal target simply stays where it is:
//! non-contiguous slots cost the batched-recurrence fast path (the model logs
//! `SSM batched recurrent DECLINED`), which is a measured ~28% on one block —
//! a price worth paying against shared recurrent state.

/// Plan `(survivor index, target slot)` migrations that pack survivors into
/// contiguous pool slots `[0..n)`.
///
/// * `occupied[i]` — survivor `i`'s current pool slot, in survivor order.
/// * `reserved` — pool slots owned by sequences that are NOT survivors
///   (streams in the scheduler's `prefilling` queue). NEVER a target.
///
/// Only survivors whose slot is `>= n` move, and only onto a slot in
/// `[0..n)` that no survivor occupies and no non-survivor reserves. When no
/// such slot exists the survivor is omitted from the plan and keeps its own.
/// Every planned target is distinct, and no target is ever a slot that is
/// still owned by anything.
pub(crate) fn plan_slot_compaction(occupied: &[usize], reserved: &[usize]) -> Vec<(usize, usize)> {
    let n = occupied.len();
    // Ascending, then popped from the back — byte-identical target choice to
    // the pre-#1002 `(0..n).filter(..).collect::<Vec<_>>().pop()` for the case
    // that formula already got right (nothing reserved).
    let mut free_targets: Vec<usize> = (0..n)
        .filter(|s| !occupied.contains(s) && !reserved.contains(s))
        .collect();
    let mut plan = Vec::new();
    for (i, &slot) in occupied.iter().enumerate() {
        if slot < n {
            continue;
        }
        match free_targets.pop() {
            Some(target) => plan.push((i, target)),
            // Not an error: with slot-owning streams parked in `prefilling`
            // there genuinely may be no free slot below `n`. Staying put is
            // the correct outcome — see the module doc.
            None => break,
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::plan_slot_compaction;

    #[test]
    fn packs_out_of_range_survivors_when_nothing_is_reserved() {
        // The pre-existing behaviour this must not change: two survivors at
        // slots 0 and 7, nothing else owns a slot, so 7 migrates to 1.
        assert_eq!(plan_slot_compaction(&[0, 7], &[]), vec![(1, 1)]);
    }

    #[test]
    fn in_range_survivors_do_not_move() {
        assert_eq!(plan_slot_compaction(&[0, 1, 2], &[]), Vec::new());
    }

    #[test]
    fn a_reserved_slot_is_never_a_target() {
        // THE round-13 cell-V shape (#1002). Two streams finished prefill and
        // decode at slots 0 and 7; the other fourteen are still in
        // `prefilling` under `--prefill-varlen-batch` and own slots 1..=6 and
        // 8..=15. `n == 2`, so the only in-range candidate is slot 1 — and a
        // prefilling stream owns it. The survivor must stay at slot 7.
        //
        // Before this rule the plan was `[(1, 1)]`, and the two sequences at
        // slot 1 shared one GDN h_state: 24 content-loop-watchdog fires, 6/16
        // responses cut at 49 tokens, then a double-release that poisoned the
        // pool free list for the rest of the serve.
        let reserved: Vec<usize> = (1..=6).chain(8..=15).collect();
        assert_eq!(plan_slot_compaction(&[0, 7], &reserved), Vec::new());
    }

    #[test]
    fn packs_into_whatever_is_genuinely_free() {
        // Three survivors at 5, 1, 9; slot 0 reserved by a prefilling stream,
        // slot 2 free. Only ONE target exists, so exactly one survivor moves
        // and the other keeps its slot rather than colliding.
        let plan = plan_slot_compaction(&[5, 1, 9], &[0]);
        assert_eq!(plan, vec![(0, 2)]);
    }

    #[test]
    fn targets_are_distinct_and_never_occupied_or_reserved() {
        let occupied = [11usize, 3, 12, 1, 13];
        let reserved = [0usize, 7];
        let plan = plan_slot_compaction(&occupied, &reserved);
        let n = occupied.len();
        let mut seen = Vec::new();
        for &(i, target) in &plan {
            assert!(occupied[i] >= n, "an in-range survivor was moved");
            assert!(target < n, "target {target} outside [0..{n})");
            assert!(!reserved.contains(&target), "target {target} is reserved");
            assert!(
                !occupied.contains(&target),
                "target {target} is held by another survivor"
            );
            assert!(!seen.contains(&target), "target {target} planned twice");
            seen.push(target);
        }
    }

    #[test]
    fn empty_survivor_set_plans_nothing() {
        assert_eq!(plan_slot_compaction(&[], &[]), Vec::new());
        assert_eq!(plan_slot_compaction(&[], &[0, 1, 2]), Vec::new());
    }
}
