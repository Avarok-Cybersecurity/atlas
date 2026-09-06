// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for [`super`] (`mtp_dcut`). A sibling file via `#[path]` — the
//! `ssm_reserve.rs`/`ssm_reserve_tests.rs` idiom — so `mtp_dcut.rs` stays
//! under the 500-line cap; module position (child of `mtp_dcut`) is
//! unchanged, so `super::*` paths are untouched.
use super::*;

#[test]
fn ratio_one_retains_full_depth() {
    let c: Vec<Vec<f32>> = vec![vec![-0.1, -2.0, -3.0]; 4];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    assert_eq!(select(&refs, 3, 32, 1.0), vec![3, 3, 3, 3]);
}

#[test]
fn zero_ratio_keeps_the_mandatory_first_draft() {
    let c: Vec<Vec<f32>> = vec![vec![-0.1, -0.2, -0.3]; 4];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    assert_eq!(select(&refs, 3, 32, 0.0), vec![1, 1, 1, 1]);
}

#[test]
fn budget_spent_on_the_confident_sequence() {
    // seq 0 is confident throughout, seq 1 collapses after its first draft.
    let c = [vec![-0.01f32, -0.02, -0.03], vec![-3.0f32, -4.0, -5.0]];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    // 2 sequences x 2 prunable depths = 4 candidates; ratio 0.5 keeps 2,
    // both of which belong to seq 0.
    assert_eq!(select(&refs, 3, 32, 0.5), vec![3, 1]);
}

#[test]
fn retained_set_is_always_a_prefix() {
    let c = [vec![-0.1f32, -9.0, -0.001], vec![-0.2f32, -0.2, -0.2]];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    let r = select(&refs, 3, 32, 0.5);
    // Depth-3 of seq 0 has a WORSE survival score than depth-2 despite its
    // own high confidence, because survival is the prefix product.
    assert!(r[0] <= 2, "prefix product must dominate the local value");
    assert!(r.iter().all(|&k| (1..=3).contains(&k)));
}

#[test]
fn row_budget_is_never_exceeded() {
    let c: Vec<Vec<f32>> = vec![vec![-0.001, -0.001, -0.001]; 8];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    // 8 sequences, budget 24 rows: 16 committed, 8 spare -> 8 extra depths.
    let r = select(&refs, 3, 24, 1.0);
    let rows: usize = r.iter().map(|k| k + 1).sum();
    assert!(rows <= 24, "rows={rows}");
}

#[test]
fn missing_confidences_are_never_pruned() {
    let empty: Vec<f32> = Vec::new();
    let c = [vec![-5.0f32, -5.0, -5.0], empty];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    assert_eq!(select(&refs, 3, 32, 0.5), vec![1, 3]);
}

#[test]
fn chunk_ranges_reproduce_the_uniform_caps() {
    // Every default-ladder-reachable shape is a SINGLE chunk — true under
    // the old 64-row budget too, so the 96 widening is default-inert.
    assert_eq!(chunk_ranges(&[4; 8]), vec![(0, 8)]);
    assert_eq!(chunk_ranges(&[3; 8]), vec![(0, 8)]);
    // The 16:2 default rung: [3; 16] = 48 rows, one chunk.
    assert_eq!(chunk_ranges(&[3; 16]), vec![(0, 16)]);
    assert_eq!(chunk_ranges(&[2; 16]), vec![(0, 16)]);
    // The 32:1 rung: one chunk up to n=32 (R = 64).
    assert_eq!(chunk_ranges(&[2; 17]), vec![(0, 17)]);
    assert_eq!(chunk_ranges(&[2; 32]), vec![(0, 32)]);
}

#[test]
fn chunk_ranges_seq_cap_derives_from_the_row_budget() {
    // Depth above n=8 (env-ladder / ragged-D-Cut shapes) is no longer
    // serialized into 8-wide chunks. Two bounds apply, and the chunk is the
    // MIN of them: the row budget (VERIFY_ROW_BUDGET = 160, giving 53 seqs
    // at rows=3, 40 at rows=4, 80 at rows=2) and the verify stash width
    // (VERIFY_WY_TABLE_SEQS = 32). Both are read from their SSOT below, so
    // this test states the boundary arithmetic and not frozen values.
    const W: usize = spark_model::layer::VERIFY_WY_TABLE_SEQS; // 32
    // rows=3: the row budget would allow 53, the stash allows 32.
    assert_eq!(chunk_ranges(&[3; 21]), vec![(0, 21)]);
    assert_eq!(chunk_ranges(&[3; W]), vec![(0, W)]);
    // Past the stash the chunker SPLITS rather than emitting a chunk the
    // model refuses (it previously returned a single (0,33) / (0,53)).
    assert_eq!(chunk_ranges(&[3; 33]), vec![(0, W), (W, 33)]);
    assert_eq!(chunk_ranges(&[3; 53]), vec![(0, W), (W, 53)]);
    // rows=4: row budget 40, stash 32.
    assert_eq!(chunk_ranges(&[4; 9]), vec![(0, 9)]);
    assert_eq!(chunk_ranges(&[4; 40]), vec![(0, W), (W, 40)]);
    // rows=2: row budget 80, stash 32.
    assert_eq!(chunk_ranges(&[2; 80]), vec![(0, W), (W, 64), (64, 80)]);
    // rows=3 with 10 seqs (the old (0,8),(8,10) split): one chunk now.
    assert_eq!(chunk_ranges(&[3; 10]), vec![(0, 10)]);
    // The row budget still binds where it is TIGHTER than the stash: at
    // rows=8 (the DFlash uniform K=γ+1 shape) 160/8 = 20 < 32, so 20 wins.
    assert_eq!(chunk_ranges(&[8; 20]), vec![(0, 20)]);
    assert_eq!(chunk_ranges(&[8; 21]), vec![(0, 20), (20, 21)]);
}

#[test]
fn chunk_ranges_respect_the_row_budget_when_ragged() {
    // Deepest-first, mixed depths: rows must never exceed the budget per
    // chunk.
    let ks = vec![4, 4, 4, 4, 4, 3, 3, 2, 2, 2];
    for (lo, hi) in chunk_ranges(&ks) {
        let rows: usize = ks[lo..hi].iter().sum();
        assert!(rows <= VERIFY_ROW_BUDGET, "rows={rows}");
        assert!(hi > lo);
    }
}

// Env-independent as long as the test process does not set
// ATLAS_MTP_DCUT_MAX_SEQS (CI does not) — same pattern as the ladder
// default-shape test.
#[test]
fn dcut_width_cap_default_is_the_measured_win_regime() {
    // 8 = the C=8 regime where ratio 0.75 measured +2.6%; pruning at the
    // 16:2 rung's n=16 measured -9% (fixer r2 leg D), so `plan` must
    // return the uniform shape for any wider batch.
    assert_eq!(dcut_width_cap(), 8);
}

/// The seam `plan` builds on: the canonical assignment consumes exactly
/// the multiset `select` produced, so Σ rows — and therefore the row
/// budget, `chunk_ranges` and the `record` telemetry — is untouched,
/// while the batch comes out slot-ascending / depth-descending. `plan`
/// itself needs live `ActiveSeq`s (GPU-backed `SequenceState`); this
/// pins the arithmetic it delegates.
#[test]
fn canonical_assignment_preserves_the_selected_row_total() {
    use spark_model::speculative::verify_key::verify_batch_order;
    // A confidence spread that prunes raggedly: seq 0 collapses, seq 3 is
    // confident throughout.
    let c = [
        vec![-0.01f32, -6.0, -7.0],
        vec![-0.01f32, -0.02, -4.0],
        vec![-0.01f32, -5.0, -6.0],
        vec![-0.01f32, -0.02, -0.03],
    ];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    let retained = select(&refs, 3, VERIFY_ROW_BUDGET, 0.5);
    let ks: Vec<usize> = retained.iter().map(|r| r + 1).collect();
    assert!(ks.iter().any(|&k| k != ks[0]), "the case must be ragged");
    // Pool slots deliberately out of batch order, as they are after
    // sequences finish and the pool fragments.
    let slots = [5usize, 0, 7, 2];
    let (order, depths) = verify_batch_order(&slots, &ks, true);
    assert_eq!(
        depths.iter().sum::<usize>(),
        ks.iter().sum::<usize>(),
        "Σ rows must survive the re-pairing — the row budget depends on it"
    );
    let placed: Vec<usize> = order.iter().map(|&i| slots[i]).collect();
    assert!(placed.windows(2).all(|w| w[0] < w[1]), "{placed:?}");
    assert!(depths.windows(2).all(|w| w[0] >= w[1]), "{depths:?}");
    // Every assigned depth stays inside the audited 2..=4 row envelope
    // (`can_batch_verify`, `gdn_decode_wy{2,3,4}`).
    assert!(depths.iter().all(|k| (2..=4).contains(k)));
    // Chunking sees the same shape it always did.
    assert_eq!(chunk_ranges(&depths), vec![(0, 4)]);
}

#[test]
fn ratio_snaps_to_a_bucket() {
    // Pure snapping arithmetic, no env: 0.6 is closest to 0.5.
    let nearest = |raw: f32| {
        *BUCKETS
            .iter()
            .min_by(|a, b| (*a - raw).abs().partial_cmp(&(*b - raw).abs()).unwrap())
            .unwrap()
    };
    assert_eq!(nearest(0.6), 0.5);
    assert_eq!(nearest(0.9), 1.0);
    assert_eq!(nearest(0.1), 0.25);
}

/// The C=2 rung PRUNES — so "skip the planner below a width threshold and
/// keep the uniform shape" is NOT a hoist, it is a behaviour change.
///
/// At C=2 the ladder gives `ladder_nd = 3` (rows = 4), which puts the batch
/// inside D-Cut's engagement envelope (`ladder_nd >= 2`, `n <= 8`). With the
/// default ratio 0.75 the 2x2 = 4 prunable positions keep 3, so exactly one
/// sequence loses its deepest draft and the plan is the RAGGED `{3, 2}`
/// retained multiset — verify rows `{4, 3}`, R = 7 instead of the uniform 8.
/// Anything that bypasses the planner at n = 2 therefore verifies one MORE
/// row per step than the shipped path, and produces a different graph key.
#[test]
fn width_two_is_inside_the_pruning_envelope_and_is_not_uniform() {
    // Both sequences confident, seq 1 slightly less so at its deepest draft.
    let c = [vec![-0.01f32, -0.02, -0.03], vec![-0.01f32, -0.02, -0.40]];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    let retained = select(&refs, 3, VERIFY_ROW_BUDGET, 0.75);
    assert_eq!(
        retained,
        vec![3, 2],
        "ratio 0.75 drops exactly one position"
    );
    let rows: Vec<usize> = retained.iter().map(|r| r + 1).collect();
    assert_ne!(rows, vec![4, 4], "the n=2 plan is NOT the uniform shape");
    assert_eq!(rows.iter().sum::<usize>(), 7);
    // ...and the row saving is real: one fewer verify row than uniform.
    assert_eq!(chunk_ranges(&rows), vec![(0, 2)], "still a single chunk");
}

/// The width gate reverts the ASSIGNMENT below
/// `verify_key::CANONICAL_KEY_MIN_WIDTH`, NOT the pruning. Composed exactly
/// as `plan` composes it (`select` → gate → `verify_batch_order`), because
/// `plan` itself needs GPU-backed `ActiveSeq`s.
///
/// At n=2 the ragged `{4, 3}` row multiset — D-Cut's whole row saving, R=7
/// against the uniform 8 — survives untouched, and each sequence keeps the
/// depth its own confidence earned. Under the canonical arm the deepest row
/// count would move to the LOWEST slot instead, which at this width buys a
/// key-space collapse from 2 keys to 1 and measured -2.4% at C=2.
#[test]
fn below_the_gate_the_pairing_is_legacy_but_the_pruning_is_kept() {
    use spark_model::speculative::verify_key::{canonical_assignment, verify_batch_order};
    let c = [vec![-0.01f32, -0.02, -0.03], vec![-0.01f32, -0.02, -0.40]];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    let ks: Vec<usize> = select(&refs, 3, VERIFY_ROW_BUDGET, 0.75)
        .iter()
        .map(|r| r + 1)
        .collect();
    assert_eq!(ks, vec![4, 3]);
    // Seq 0 (the deeper one) sits on the HIGHER pool slot, so the two arms
    // disagree — the case the gate is actually deciding.
    let slots = [5usize, 0];
    assert!(!canonical_assignment(slots.len()));

    let (order, depths) = verify_batch_order(&slots, &ks, canonical_assignment(slots.len()));
    let paired: Vec<(usize, usize)> = order.iter().map(|&i| (slots[i], ks[i])).collect();
    assert_eq!(
        depths.iter().sum::<usize>(),
        7,
        "the row saving must survive the gate — this is not a D-Cut kill switch"
    );
    assert_eq!(
        paired,
        vec![(5, 4), (0, 3)],
        "each sequence keeps the depth its confidence earned"
    );
    assert_eq!(depths, vec![4, 3]);
    assert_eq!(chunk_ranges(&depths), vec![(0, 2)]);

    // The canonical arm re-pairs: deepest onto the lowest slot.
    let (c_order, c_depths) = verify_batch_order(&slots, &ks, true);
    let c_paired: Vec<(usize, usize)> = c_order
        .iter()
        .zip(&c_depths)
        .map(|(&i, &k)| (slots[i], k))
        .collect();
    assert_eq!(c_paired, vec![(0, 4), (5, 3)]);
    assert_ne!(c_paired, paired);
    assert_eq!(c_depths.iter().sum::<usize>(), 7);
}

// ── Width bound: the chunker's OTHER cap ───────────────────────────────────
//
// `chunk_ranges` derives its per-chunk sequence cap from the ROW budget
// alone (`VERIFY_ROW_BUDGET / ks[lo]`), which at the shallow rungs is 40
// (rows=4), 53 (rows=3) and 80 (rows=2) sequences. But `can_batch_verify`
// enforces a SECOND, independent bound the chunker never consulted:
// `(2..=VERIFY_WY_TABLE_SEQS).contains(&n)`, i.e. n <= 32, because the
// batched verify's hidden stash has exactly 32 slots.
//
// A chunk wider than 32 is therefore refused WHOLESALE by the
// `chunk.len() >= 2 && model.can_batch_verify(chunk_ks)` gate in
// `mtp_step`, and every sequence in it falls back to the per-seq verify
// loop — one weight sweep PER SEQUENCE instead of one per chunk. That is
// the exact "stale cap silently serializes the batch" artifact class the
// module doc says was closed when the cap stopped being a hardcoded 8; it
// was closed for ROWS and left open for WIDTH.
#[test]
fn chunk_ranges_never_exceed_the_verify_width_bound() {
    // The hard bound `can_batch_verify` enforces, read from the model crate
    // rather than restated, so this test tracks the stash if it is resized.
    const WIDTH_CAP: usize = spark_model::layer::VERIFY_WY_TABLE_SEQS;
    // Every rung depth, at widths that span the row-derived caps (40/53/80).
    for rows in 2..=4usize {
        for n in 2..=96usize {
            let ks = vec![rows; n];
            for (lo, hi) in chunk_ranges(&ks) {
                assert!(
                    hi - lo <= WIDTH_CAP,
                    "rows={rows} n={n}: chunk ({lo},{hi}) is {} sequences wide, \
                     above the {WIDTH_CAP}-slot verify stash — can_batch_verify \
                     refuses it and the whole chunk serializes",
                    hi - lo
                );
            }
        }
    }
}

// Ragged (D-Cut) shapes must obey the width bound too: the cap is taken
// from `ks[lo]`, the chunk's DEEPEST row count, so a chunk that starts deep
// and continues shallow gets the deep cap while admitting shallow rows.
#[test]
fn ragged_chunks_also_respect_the_width_bound() {
    // Deepest-first, as the caller guarantees: 4 deep rows then a long
    // shallow tail. seq_cap is 160/4 = 40, so the shallow tail is admitted
    // well past the 32-slot stash.
    let mut ks = vec![4usize; 4];
    ks.extend(std::iter::repeat_n(2usize, 60));
    for (lo, hi) in chunk_ranges(&ks) {
        assert!(
            hi - lo <= spark_model::layer::VERIFY_WY_TABLE_SEQS,
            "ragged chunk ({lo},{hi}) is {} wide",
            hi - lo
        );
        // The row budget must still hold — the width clamp adds a bound, it
        // does not relax the existing one.
        let r: usize = ks[lo..hi].iter().sum();
        assert!(r <= VERIFY_ROW_BUDGET, "rows={r}");
        assert!(hi > lo, "empty range");
    }
}
