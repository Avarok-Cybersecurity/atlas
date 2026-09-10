// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for [`super`] (`verify_key`). A sibling file via `#[path]` — the
//! `ssm_reserve.rs`/`ssm_reserve_tests.rs` idiom — so `verify_key.rs` stays
//! under the 500-line cap; module position (child of `verify_key`) is
//! unchanged, so `super::*` paths are untouched.
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
    verify_graph_key(&pairs, false, false)
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
        verify_graph_key(&pairs, true, false),
        verify_graph_key(&pairs, false, false)
    );
}

/// The write-on-accept bit splits the key too: the captured K=4 GDN launch
/// is a different kernel under it, so a graph captured for one request must
/// never replay for the other. All four sentinel states are distinct.
#[test]
fn write_on_accept_splits_the_key() {
    let pairs = [(0u32, 4u32), (1, 4)];
    let keys = [
        verify_graph_key(&pairs, false, false),
        verify_graph_key(&pairs, true, false),
        verify_graph_key(&pairs, false, true),
        verify_graph_key(&pairs, true, true),
    ];
    for i in 0..keys.len() {
        for j in 0..keys.len() {
            assert_eq!(keys[i] == keys[j], i == j, "{i} vs {j}");
        }
    }
    // The pairs are untouched; only the sentinel moves.
    assert_eq!(keys[0][..4], keys[2][..4]);
}

/// Canonical assignment puts slots in ascending order and depths in
/// non-increasing order. Pool fragmentation can still leave slot gaps; the
/// batched GDN implementation checks actual pointer adjacency and declines
/// its fast path when a depth run is not consecutive.
#[test]
fn canonical_order_is_slot_ascending_and_depth_descending() {
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

/// Degenerate widths must not panic: the key path runs for every batch
/// the scheduler forms.
#[test]
fn empty_and_single_batches_are_well_formed() {
    assert_eq!(verify_batch_order(&[], &[], true), (vec![], vec![]));
    assert_eq!(verify_batch_order(&[5], &[3], true), (vec![0], vec![3]));
    assert_eq!(verify_graph_key(&[], false, false), vec![u32::MAX]);
}

#[test]
fn mismatched_batch_vectors_fail_closed() {
    for (slots, ks) in [(&[1, 2][..], &[4][..]), (&[1][..], &[4, 3][..])] {
        let result = std::panic::catch_unwind(|| verify_batch_permutation(slots, ks, true));
        assert!(result.is_err(), "slots={slots:?}, ks={ks:?}");
    }
}

// ───────────────────────── the width gate ─────────────────────────

/// Build the key through THE GATE with the threshold and kill switch
/// injected — the same composition production uses
/// ([`canonical_assignment`] → [`canonical_assignment_at`]), so these tests
/// exercise the shipped policy rather than a re-derivation of it.
fn key_through_gate(batch: &[(usize, usize)], min_width: usize, kill_clear: bool) -> Vec<u32> {
    key_for(
        batch,
        canonical_assignment_at(batch.len(), min_width, kill_clear),
    )
}

/// The boundary sits exactly at the constant: n=7 legacy, n=8 canonical.
/// n=8 is the rung the A/B measured the +3.9% at, and n<8 the rungs it
/// measured -2.4% / -3.7% at.
#[test]
fn the_gate_boundary_is_the_constant() {
    for n in 0..CANONICAL_KEY_MIN_WIDTH {
        assert!(!canonical_assignment(n), "n={n} must take the legacy arm");
    }
    for n in CANONICAL_KEY_MIN_WIDTH..=64 {
        assert!(canonical_assignment(n), "n={n} must take the canonical arm");
    }
}

/// ★ THE POINT OF THE GATE. Below the threshold the ASSIGNMENT and the KEY
/// BYTES are byte-identical to the pre-canonical (pre-#552) behaviour, for
/// every batch shape reachable at those widths.
///
/// "Pre-canonical" is reproduced here from first principles rather than from
/// the `canonical = false` arm: a STABLE sort by `(Reverse(k), slot)` with
/// each member keeping its own depth — literally the
/// `refs.sort_by_key(|(a, k)| (Reverse(*k), slot))` both call sites ran
/// before #552.
#[test]
fn below_the_threshold_is_byte_identical_to_pre_canonical() {
    // Every ragged shape D-Cut can emit at n in 1..8: rows per sequence are
    // in 2..=4 (`mtp_dcut` v1 keeps >= 1 draft, ladder_nd <= 3), over slot
    // sets that are neither sorted nor contiguous.
    let slot_sets: [&[usize]; 4] = [&[0], &[3, 1], &[5, 0, 2, 9], &[7, 2, 5, 0, 3, 11, 4]];
    for slots in slot_sets {
        let n = slots.len();
        assert!(n < CANONICAL_KEY_MIN_WIDTH);
        for shape in 0..4usize.pow(n as u32) {
            let ks: Vec<usize> = (0..n).map(|i| 2 + (shape >> (2 * i)) % 3).collect();
            // Pre-#552: stable sort, deepest first then slot, depth stays
            // with its member.
            let mut pre: Vec<(usize, usize)> = slots.iter().copied().zip(ks.clone()).collect();
            pre.sort_by_key(|&(slot, k)| (std::cmp::Reverse(k), slot));
            let pre_key: Vec<u32> = pre
                .iter()
                .flat_map(|&(s, k)| [s as u32, k as u32])
                .chain(std::iter::once(u32::MAX))
                .collect();

            let (order, depths) = verify_batch_order(slots, &ks, canonical_assignment(n));
            let got: Vec<(usize, usize)> = order
                .iter()
                .zip(&depths)
                .map(|(&i, &k)| (slots[i], k))
                .collect();
            assert_eq!(got, pre, "assignment drifted at n={n} shape={shape}");

            let batch: Vec<(usize, usize)> = slots.iter().copied().zip(ks).collect();
            assert_eq!(
                key_for(&batch, canonical_assignment(n)),
                pre_key,
                "key bytes drifted at n={n} shape={shape}"
            );
        }
    }
}

/// At the threshold the gate selects the canonical arm, so #552's collapse
/// still happens where it was measured to pay: the same multiset in any
/// arrangement is ONE key at n=8.
#[test]
fn at_the_threshold_the_gate_selects_the_canonical_arm() {
    let a: Vec<(usize, usize)> = (0..8).map(|s| (s, if s < 5 { 4 } else { 3 })).collect();
    let b: Vec<(usize, usize)> = (0..8).map(|s| (s, if s < 3 { 3 } else { 4 })).collect();
    assert_eq!(a.len(), CANONICAL_KEY_MIN_WIDTH);
    let ka = key_for(&a, canonical_assignment(a.len()));
    assert_eq!(ka, key_for(&b, canonical_assignment(b.len())));
    assert_eq!(ka, key_for(&a, true));
    // The same two arrangements are TWO keys below the gate.
    assert_ne!(key_for(&a, false), key_for(&b, false));
}

/// `ATLAS_CANONICAL_KEY_MIN_WIDTH` parsing, as a pure function of the raw
/// value — the sweep knob must not silently ignore what it is handed.
#[test]
fn min_width_override_parses() {
    let os = |v: &str| Some(std::ffi::OsString::from(v));
    assert_eq!(min_width_from_env(None), CANONICAL_KEY_MIN_WIDTH);
    assert_eq!(min_width_from_env(os("4")), 4);
    assert_eq!(min_width_from_env(os("0")), 0);
    assert_eq!(min_width_from_env(os(" 16 ")), 16);
    // Unparseable is NOT silently "off" — it falls back to the measured
    // default, the same contract `dcut_width_cap` has.
    assert_eq!(min_width_from_env(os("")), CANONICAL_KEY_MIN_WIDTH);
    assert_eq!(min_width_from_env(os("-1")), CANONICAL_KEY_MIN_WIDTH);
    assert_eq!(min_width_from_env(os("eight")), CANONICAL_KEY_MIN_WIDTH);
}

/// The override MOVES the boundary, end to end in key bytes: at
/// `ATLAS_CANONICAL_KEY_MIN_WIDTH=4` the n=4 batch that is legacy-keyed by
/// default becomes canonical, and at `=16` the n=8 batch that is canonical
/// by default falls back to legacy.
#[test]
fn min_width_override_moves_the_boundary() {
    let n4 = [(0usize, 4usize), (1, 3), (2, 4), (3, 2)];
    let n8: Vec<(usize, usize)> = (0..8).map(|s| (s, if s < 5 { 4 } else { 3 })).collect();
    let lowered = min_width_from_env(Some(std::ffi::OsString::from("4")));
    let raised = min_width_from_env(Some(std::ffi::OsString::from("16")));

    assert_eq!(key_through_gate(&n4, lowered, true), key_for(&n4, true));
    assert_eq!(
        key_through_gate(&n4, CANONICAL_KEY_MIN_WIDTH, true),
        key_for(&n4, false)
    );
    assert_eq!(key_through_gate(&n8, raised, true), key_for(&n8, false));
    assert_eq!(
        key_through_gate(&n8, CANONICAL_KEY_MIN_WIDTH, true),
        key_for(&n8, true)
    );
    // `=0` is the "canonical everywhere" sweep point, i.e. plain #552.
    let all = min_width_from_env(Some(std::ffi::OsString::from("0")));
    assert_eq!(key_through_gate(&n4, all, true), key_for(&n4, true));
}

/// The kill switch disables the assignment ENTIRELY — it dominates both the
/// width and the override, so no sweep value can resurrect it.
#[test]
fn kill_switch_dominates_width_and_override() {
    let n8: Vec<(usize, usize)> = (0..8).map(|s| (s, if s < 5 { 4 } else { 3 })).collect();
    for min_width in [0usize, 1, 4, CANONICAL_KEY_MIN_WIDTH, 64] {
        for n in [0usize, 1, 2, 4, 8, 16, 128] {
            assert!(
                !canonical_assignment_at(n, min_width, false),
                "kill switch must dominate at n={n} min_width={min_width}"
            );
        }
        assert_eq!(key_through_gate(&n8, min_width, false), key_for(&n8, false));
    }
}
