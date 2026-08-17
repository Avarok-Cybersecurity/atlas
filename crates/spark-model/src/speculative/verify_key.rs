// SPDX-License-Identifier: AGPL-3.0-only

//! Batched-verify graph key: the canonical depth→slot assignment and the key
//! bytes derived from it. ONE ordering rule, shared by the scheduler that
//! dispatches the batch (`mtp_dcut::plan`, `mtp_step`) and the model that
//! builds the CUDA-graph cache key (`verify_e2::verify_batched_graph_key`).
//!
//! # The measured defect (nsys + A/B on dgx2, binary `b508679e4`)
//!
//! The batched-verify graph cache keys on the per-row `(ssm slot, depth k)`
//! pairs in batch order, because a capture bakes each row's pool state
//! addresses AND the depth-run launch structure. D-Cut re-ranks WHICH
//! sequence gets WHICH depth every step, so the key space was the set of
//! ARRANGEMENTS of the step's depth multiset over the batch's slots:
//!
//! ```text
//! n=8, the three multisets actually observed:
//!   8!/(5!·2!·1!) + 8!/(4!·4!) + 8!/(6!·2!) = 168 + 70 + 28 = 266 keys
//! ```
//!
//! against `VERIFY_BATCHED_GRAPH_CAP` = 32. Measured key counts: n=2 → 2,
//! n=4 → 10, **n=8 → 160-253**, n=16 → 1 (D-Cut is off above width 8). nsys
//! at C=8: 149 captures in 167 steps (89% of steps), `cuGraphInstantiate` +
//! `cuGraphExecDestroy` + `cuGraphDestroy` = 23.2 ms/step ≈ 20% of the step;
//! GPU busy 96.3% → 77.2%. A/B at C=8: control 78.89 tok/s vs 84.35 with
//! D-Cut off (+6.9%, key count 253 → 1) — and that leg also LOSES the row
//! pruning, so the thrash alone costs more than 6.9%.
//!
//! # The fix: canonical depth→slot assignment
//!
//! D-Cut's ranking chooses HOW MANY drafts survive at each depth (the
//! multiset) — that is where its row saving comes from. It also chooses WHO
//! gets them, and that half is what multiplies the key space. So the multiset
//! stays confidence-chosen and the ARRANGEMENT becomes a pure function of the
//! batch: depths descending are paired with slots ascending. The key is then
//! determined by (slot set, depth multiset) alone — at n=8 the 266 observed
//! arrangements collapse to the 3 multisets that produced them (worst case
//! over all reachable shapes: multisets of size 8 over depths {2,3,4} =
//! C(10,2) = 45, versus 3^8 = 6561 arrangements).
//!
//! ★ The two orderings RECONCILE instead of fighting. The dispatch needs
//! depths descending (equal depths must form contiguous runs — the batched
//! conv+WY fast path launches once per run,
//! `trait_decode_batched_conv_gdn_multi.rs`) and the SSM batched arms need
//! slots ascending and consecutive in batch order (`ssm_batched_recurrent.rs`,
//! `decode_step.rs`, `mtp_step.rs`). Under the confidence-chosen arrangement
//! those two demands are in direct conflict: a ragged batch sorted
//! deepest-first scrambles the slot order, so each depth run gets an
//! arbitrary SUBSET of the pool slots and the consecutive-slot precondition
//! fails. Pairing depths-descending with slots-ascending makes the two orders
//! THE SAME order, and each depth run then owns a consecutive slot block.
//!
//! Correctness: which sequence gets which depth is a pure PERFORMANCE choice.
//! Every batchable sequence enters the step with exactly `ladder_nd` drafts
//! (`mtp_step` truncates the surplus), each assigned depth is in
//! `1..=ladder_nd` drafts, and a verify of a shorter draft prefix is the same
//! math on fewer rows. Σ rows is unchanged, so the row budget and chunking
//! are unchanged. What is NOT free to change is the pairing between a batch
//! POSITION and the slot whose pointers the graph baked there — hence one
//! ordering rule, used by both the dispatch and the key.
//!
//! Kill switch `ATLAS_NO_CANONICAL_VERIFY_KEY` (PRESENCE — house convention,
//! `=0` is NOT off) restores the pre-canonical behaviour: each sequence keeps
//! its own confidence-chosen depth and the batch is sorted deepest-first,
//! ssm-slot second.

/// Canonical assignment ON unless `ATLAS_NO_CANONICAL_VERIFY_KEY` is present.
/// Read once per process.
pub fn canonical_verify_key_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_CANONICAL_VERIFY_KEY").is_none())
}

/// Dispatch ORDER for one verify batch — the permutation only.
///
/// `slots[i]` / `ks[i]` describe batch member `i` in the caller's arbitrary
/// order (`ks[i]` = that member's ROW count, drafts+1). `order[p]` is the
/// input index dispatched at position `p`.
///
/// * `canonical = true` — sort by ssm slot ASCENDING. `ks` is unread: under
///   the canonical assignment the depths are already descending along that
///   order, so slot order IS depth order and there is nothing to trade off.
/// * `canonical = false` — the kill-switch path: deepest first, ssm slot
///   second (today's behaviour, where the two demands genuinely conflict).
///
/// Ties break on input index, so the result is a deterministic function of
/// the inputs — a graph key must never depend on sort instability. Callers
/// that build the batch in ascending active-sequence index therefore agree
/// on the order of slot-less (`usize::MAX`) members.
///
/// Idempotent under `canonical = true`, so a chunked caller may re-apply it
/// to a contiguous sub-range of an already-ordered batch.
pub fn verify_batch_permutation(slots: &[usize], ks: &[usize], canonical: bool) -> Vec<usize> {
    debug_assert_eq!(
        slots.len(),
        ks.len(),
        "verify_batch_permutation: slots/ks mismatch"
    );
    let n = slots.len().min(ks.len());
    let mut order: Vec<usize> = (0..n).collect();
    if canonical {
        order.sort_by_key(|&i| (slots[i], i));
    } else {
        order.sort_by_key(|&i| (std::cmp::Reverse(ks[i]), slots[i], i));
    }
    order
}

/// Order one verify batch AND assign its depths — the planner's entry point
/// (`mtp_dcut::plan`), the one place a sequence's verify depth is decided.
///
/// Returns `(order, depths)` where `order` is [`verify_batch_permutation`]
/// and `depths[p]` is the row count position `p` verifies.
///
/// * `canonical = true` — the depth MULTISET is re-paired onto the ordered
///   batch, deepest onto the lowest slot. `depths[p]` is therefore NOT
///   generally `ks[order[p]]`; the multiset is preserved exactly, only the
///   pairing is re-made. Both dispatch invariants then hold by construction:
///   slots non-decreasing in `p` (the SSM consecutive-slot precondition) and
///   depths non-increasing in `p` (the contiguous depth-run precondition).
/// * `canonical = false` — each member keeps its own depth.
///
/// Because a caller must TRUNCATE each sequence's drafts to the depth it was
/// assigned, this must be called exactly once per batch; downstream stages
/// that only need the batch in dispatch order use
/// [`verify_batch_permutation`], which cannot disturb an assignment.
pub fn verify_batch_order(
    slots: &[usize],
    ks: &[usize],
    canonical: bool,
) -> (Vec<usize>, Vec<usize>) {
    let order = verify_batch_permutation(slots, ks, canonical);
    let depths: Vec<usize> = if canonical {
        let mut d: Vec<usize> = ks[..order.len()].to_vec();
        d.sort_unstable_by(|a, b| b.cmp(a));
        d
    } else {
        order.iter().map(|&i| ks[i]).collect()
    };
    (order, depths)
}

/// The batched-verify CUDA-graph cache key for one batch: the `(ssm slot,
/// row count)` pairs in DISPATCH order, then a wy-tables-present sentinel.
///
/// Every SSM pointer a capture bakes (h/conv state, rollback intermediates,
/// WY table contents) is a pure function of the pair at that batch position,
/// and the depth-run launch structure is a pure function of the depth
/// sequence — so the key must carry both, in order. All other captured
/// addresses (hidden/logits/scratch/meta) are fixed buffers refreshed
/// pre-replay. The sentinel keeps a table-less capture from ever replaying a
/// table-full step or vice versa.
///
/// Pairs arrive in the order [`verify_batch_order`] produced, so with
/// canonicalization on the key is a pure function of (slot set, depth
/// multiset, sentinel) — the whole point of this module.
pub fn verify_graph_key(pairs: &[(u32, u32)], wy_tables_null: bool) -> Vec<u32> {
    let mut key: Vec<u32> = Vec::with_capacity(2 * pairs.len() + 1);
    for &(slot, k) in pairs {
        key.push(slot);
        key.push(k);
    }
    key.push(u32::MAX - u32::from(wy_tables_null));
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the key the dispatch would produce for a batch given as
    /// `(slot, k)` pairs in the scheduler's arbitrary pre-sort order.
    fn key_for(batch: &[(usize, usize)], canonical: bool) -> Vec<u32> {
        let slots: Vec<usize> = batch.iter().map(|&(s, _)| s).collect();
        let ks: Vec<usize> = batch.iter().map(|&(_, k)| k).collect();
        let (order, depths) = verify_batch_order(&slots, &ks, canonical);
        let pairs: Vec<(u32, u32)> = order
            .iter()
            .zip(&depths)
            .map(|(&i, &k)| (slots[i] as u32, k as u32))
            .collect();
        verify_graph_key(&pairs, false)
    }

    /// THE DEFECT, pinned. The same depth multiset spread over the same slots
    /// in different arrangements — exactly what D-Cut re-ranking produces
    /// step to step — must collapse onto ONE key.
    #[test]
    fn same_multiset_different_arrangement_is_one_key() {
        // Multiset {4,4,3,3} over slots {0,1,2,3}: 4!/(2!·2!) = 6 arrangements.
        let arrangements: [[(usize, usize); 4]; 6] = [
            [(0, 4), (1, 4), (2, 3), (3, 3)],
            [(0, 4), (1, 3), (2, 4), (3, 3)],
            [(0, 4), (1, 3), (2, 3), (3, 4)],
            [(0, 3), (1, 4), (2, 4), (3, 3)],
            [(0, 3), (1, 4), (2, 3), (3, 4)],
            [(0, 3), (1, 3), (2, 4), (3, 4)],
        ];
        let keys: std::collections::HashSet<Vec<u32>> =
            arrangements.iter().map(|a| key_for(a, true)).collect();
        assert_eq!(keys.len(), 1, "6 arrangements must collapse to 1 key");
        // And the ONE key is the canonical pairing: slots ascending, depths
        // descending.
        assert_eq!(
            keys.into_iter().next().unwrap(),
            vec![0, 4, 1, 4, 2, 3, 3, 3, u32::MAX]
        );
    }

    /// The n=8 arithmetic from the module docs, executed: the three depth
    /// multisets the profile observed produce 266 keys pre-canonicalization
    /// and 3 after. Generated exhaustively so the count is computed, not
    /// asserted from a comment.
    #[test]
    fn n8_key_space_collapses_266_to_3() {
        // (count of depth 4, count of 3, count of 2) for each observed shape.
        let multisets = [(5usize, 2usize, 1usize), (4, 4, 0), (6, 0, 2)];
        let mut legacy: std::collections::HashSet<Vec<u32>> = Default::default();
        let mut canon: std::collections::HashSet<Vec<u32>> = Default::default();
        for &(c4, c3, c2) in &multisets {
            let mut depths: Vec<usize> = Vec::new();
            depths.extend(std::iter::repeat_n(4usize, c4));
            depths.extend(std::iter::repeat_n(3usize, c3));
            depths.extend(std::iter::repeat_n(2usize, c2));
            assert_eq!(depths.len(), 8);
            // Every distinct arrangement of this multiset over slots 0..8.
            let mut perm: Vec<usize> = (0..8).collect();
            permute(&mut perm, 0, &mut |p| {
                let batch: Vec<(usize, usize)> = (0..8).map(|s| (s, depths[p[s]])).collect();
                legacy.insert(key_for(&batch, false));
                canon.insert(key_for(&batch, true));
            });
        }
        assert_eq!(legacy.len(), 266, "pre-canonical arrangement count");
        assert_eq!(canon.len(), 3, "one key per depth multiset");
    }

    /// Every permutation of `v`, in place.
    fn permute(v: &mut Vec<usize>, i: usize, f: &mut impl FnMut(&[usize])) {
        if i == v.len() {
            f(v);
            return;
        }
        for j in i..v.len() {
            v.swap(i, j);
            permute(v, i + 1, f);
            v.swap(i, j);
        }
    }

    /// Different multisets must NOT collide — a graph baked for one depth
    /// shape replaying against another is the state-poisoning failure mode.
    #[test]
    fn different_multisets_have_different_keys() {
        let a = key_for(&[(0, 4), (1, 4), (2, 3), (3, 3)], true);
        let b = key_for(&[(0, 4), (1, 3), (2, 3), (3, 3)], true);
        let c = key_for(&[(0, 4), (1, 4), (2, 4), (3, 3)], true);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    /// A different SLOT SET at the same depth multiset must also differ: the
    /// graph bakes those pool addresses.
    #[test]
    fn different_slot_sets_have_different_keys() {
        let a = key_for(&[(0, 4), (1, 3)], true);
        let b = key_for(&[(0, 4), (2, 3)], true);
        assert_ne!(a, b);
    }

    /// The sentinel keeps a table-less capture from replaying a table-full
    /// step.
    #[test]
    fn wy_table_presence_splits_the_key() {
        let pairs = [(0u32, 4u32), (1, 3)];
        assert_ne!(
            verify_graph_key(&pairs, true),
            verify_graph_key(&pairs, false)
        );
    }

    /// The invariants the SSM batched arms depend on, asserted explicitly on
    /// the canonical order: slots ASCENDING in batch order (the
    /// consecutive-slot precondition of `ssm_batched_recurrent.rs` and
    /// `trait_decode_batched_conv_gdn_multi.rs`) and depths NON-INCREASING
    /// (equal depths form contiguous runs — `decode_batched_inner`'s run
    /// loop).
    #[test]
    fn canonical_order_holds_both_dispatch_invariants() {
        // Deliberately hostile input: slot order and depth order disagree.
        let slots = [7usize, 2, 5, 0, 3];
        let ks = [2usize, 4, 2, 3, 4];
        let (order, depths) = verify_batch_order(&slots, &ks, true);
        let placed: Vec<usize> = order.iter().map(|&i| slots[i]).collect();
        assert!(
            placed.windows(2).all(|w| w[0] < w[1]),
            "slots must be ascending in batch order, got {placed:?}"
        );
        assert!(
            depths.windows(2).all(|w| w[0] >= w[1]),
            "depths must be non-increasing, got {depths:?}"
        );
        // The multiset — and therefore Σ rows, the row budget and chunking —
        // is preserved exactly.
        let mut before = ks.to_vec();
        before.sort_unstable();
        let mut after = depths.clone();
        after.sort_unstable();
        assert_eq!(before, after);
        assert_eq!(placed, vec![0, 2, 3, 5, 7]);
        assert_eq!(depths, vec![4, 4, 3, 2, 2]);
    }

    /// Contiguous slots stay contiguous per depth RUN — the precondition the
    /// two-launch conv+WY fast path actually checks.
    #[test]
    fn each_depth_run_owns_a_consecutive_slot_block() {
        let slots = [0usize, 1, 2, 3, 4, 5, 6, 7];
        let ks = [2usize, 4, 3, 4, 2, 3, 4, 3];
        let (order, depths) = verify_batch_order(&slots, &ks, true);
        let placed: Vec<usize> = order.iter().map(|&i| slots[i]).collect();
        let mut g0 = 0usize;
        while g0 < depths.len() {
            let mut g1 = g0 + 1;
            while g1 < depths.len() && depths[g1] == depths[g0] {
                g1 += 1;
            }
            assert!(
                placed[g0..g1].windows(2).all(|w| w[1] == w[0] + 1),
                "run {g0}..{g1} (k={}) must be consecutive slots, got {:?}",
                depths[g0],
                &placed[g0..g1]
            );
            g0 = g1;
        }
    }

    /// The permutation-only entry point must NEVER re-pair depths — a
    /// downstream stage re-assigning them could hand a sequence a depth
    /// deeper than the drafts the planner truncated it to.
    #[test]
    fn permutation_leaves_depths_attached_to_their_member() {
        // An arrangement the canonical planner would never emit (depths
        // ascending along the slots), to prove the permutation does not
        // "repair" it.
        let slots = [7usize, 2, 5, 0];
        let ks = [4usize, 2, 3, 2];
        for canonical in [true, false] {
            let order = verify_batch_permutation(&slots, &ks, canonical);
            let (_, assigned) = verify_batch_order(&slots, &ks, canonical);
            let carried: Vec<usize> = order.iter().map(|&i| ks[i]).collect();
            if canonical {
                // The planner's entry point DOES re-pair; the permutation
                // does not. That difference is the point of the split.
                assert_eq!(carried, vec![2, 2, 3, 4]);
                assert_eq!(assigned, vec![4, 3, 2, 2]);
            } else {
                assert_eq!(carried, assigned);
            }
        }
    }

    /// Idempotence: a chunked caller re-applies the ordering per chunk.
    #[test]
    fn canonical_order_is_idempotent() {
        let slots = [7usize, 2, 5, 0, 3];
        let ks = [2usize, 4, 2, 3, 4];
        let (o1, d1) = verify_batch_order(&slots, &ks, true);
        let s1: Vec<usize> = o1.iter().map(|&i| slots[i]).collect();
        let (o2, d2) = verify_batch_order(&s1, &d1, true);
        assert_eq!(o2, (0..s1.len()).collect::<Vec<_>>());
        assert_eq!(d2, d1);
    }

    /// Kill switch: the legacy ordering keeps each sequence's own depth, so
    /// the same multiset in different arrangements yields DIFFERENT keys —
    /// today's behaviour, restored exactly.
    #[test]
    fn kill_switch_restores_the_arrangement_keyed_behaviour() {
        let a = key_for(&[(0, 4), (1, 4), (2, 3), (3, 3)], false);
        let b = key_for(&[(0, 3), (1, 3), (2, 4), (3, 4)], false);
        assert_ne!(a, b, "legacy keys must still separate arrangements");
        // Legacy order is deepest-first, slot second — each member keeps its
        // OWN depth.
        assert_eq!(b, vec![2, 4, 3, 4, 0, 3, 1, 3, u32::MAX]);
    }

    /// Uniform depths (D-Cut off, above `dcut_width_cap`, or ratio 1.0):
    /// both arms reduce to "sort by slot", so the whole D-Cut-off regime is
    /// byte-identical under either setting.
    #[test]
    fn uniform_depths_are_identical_under_both_arms() {
        let slots = [3usize, 1, 2, 0];
        let ks = [3usize; 4];
        let canon = verify_batch_order(&slots, &ks, true);
        let legacy = verify_batch_order(&slots, &ks, false);
        assert_eq!(canon, legacy);
        assert_eq!(
            canon.0.iter().map(|&i| slots[i]).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    /// Env-independent as long as the test process does not set the kill
    /// switch (CI does not) — same pattern as `dcut_width_cap`'s default pin.
    #[test]
    fn canonical_is_on_by_default() {
        assert!(canonical_verify_key_enabled());
    }

    /// Degenerate widths must not panic: the key path runs for every batch
    /// the scheduler forms.
    #[test]
    fn empty_and_single_batches_are_well_formed() {
        assert_eq!(verify_batch_order(&[], &[], true), (vec![], vec![]));
        assert_eq!(verify_batch_order(&[5], &[3], true), (vec![0], vec![3]));
        assert_eq!(verify_graph_key(&[], false), vec![u32::MAX]);
    }
}
