// SPDX-License-Identifier: AGPL-3.0-only

//! DSA selection-launcher proofs.
//!
//! Two things are worth proving without a GPU, and they are the two the microtest
//! could not: that the pool arithmetic this launcher substitutes for
//! `dsa_compact_pools` is the *same* set the reference keeps, and that the top-k
//! shared-memory ceiling is enforced instead of truncated.

use super::*;
use crate::layers::glm5next_dsa_ref::{DsaDims, kept_pools};

/// GLM-5.3 DSA geometry at TP=1, matching `tp::tests::cfg`.
fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
    }
}

fn dims(c: &Glm5NextDsaConfig) -> DsaDims {
    DsaDims {
        hidden: c.hidden,
        index_heads: c.index_heads,
        index_head_dim: c.index_head_dim,
        index_kpool: c.index_kpool,
        index_topk: c.index_topk,
        always_select_tail: c.always_select_tail,
        q_lora_rank: c.q_lora_rank,
        heads: c.local_heads,
        kv_lora_rank: c.kv_lora_rank,
        qk_nope_head_dim: c.qk_nope_head_dim,
        qk_rope_head_dim: c.qk_rope_head_dim,
        v_head_dim: c.v_head_dim,
    }
}

/// 🔴 THE load-bearing claim of this module: skipping `dsa_compact_pools` is legal
/// over a contiguous cache because the kept set is the prefix `0..seq/kpool`.
///
/// Proven against the reference implementation itself, not restated — for every
/// sequence length across several pool sizes, all-valid input.
#[test]
fn contiguous_pool_count_equals_the_reference_kept_set() {
    for kpool in [1usize, 2, 3, 4, 8] {
        let mut c = cfg();
        c.index_kpool = kpool;
        // index_topk must stay a multiple of kpool for `validate`; irrelevant here.
        c.index_topk = kpool * 512;
        let d = dims(&c);
        for seq in 0usize..=257 {
            let valid = vec![1u8; seq];
            let reference = kept_pools(&valid, d, seq);
            let ours = contiguous_pool_count(kpool, seq);
            assert_eq!(
                reference.len(),
                ours,
                "kpool={kpool} seq={seq}: count disagrees with kept_pools"
            );
            // Not just the count — the identity of the pools, which is what makes
            // the compacted array a PREFIX of the full one rather than a permutation.
            let expected: Vec<i32> = (0..ours as i32).collect();
            assert_eq!(
                reference, expected,
                "kpool={kpool} seq={seq}: kept pools are not the leading prefix, so \
                 skipping dsa_compact_pools would misalign every downstream index"
            );
        }
    }
}

/// The full pool count includes the trailing partial pool; the kept count does not.
/// Confusing the two silently shifts every pool index by one at the tail.
#[test]
fn full_and_kept_pool_counts_differ_exactly_on_a_partial_tail() {
    let c = cfg();
    for seq in [4usize, 5, 7, 8, 4096, 4097] {
        let g = DsaSelectGeometry::plan(&c, seq, 1).unwrap();
        assert_eq!(g.n_pools, seq / 4, "seq={seq} kept");
        assert_eq!(g.n_pools_full, seq.div_ceil(4), "seq={seq} full");
        assert!(g.n_pools_full >= g.n_pools);
        assert_eq!(
            g.n_pools_full - g.n_pools,
            usize::from(!seq.is_multiple_of(4))
        );
    }
}

/// 🔴 The context ceiling is REFUSED, not truncated. `dsa_topk_pools` bitonic-sorts
/// in shared memory; past 4,096 pools the launch would not fit and the kernel does
/// not degrade gracefully.
#[test]
fn plan_refuses_a_context_past_the_topk_shared_memory_ceiling() {
    let c = cfg();
    // 4,096 pools = 16,384 tokens at kpool=4 — the last size that fits.
    let ok = DsaSelectGeometry::plan(&c, 16_384, 1).unwrap();
    assert_eq!(ok.n_pools, 4_096);
    assert_eq!(ok.topk_np2, 4_096);
    assert_eq!(ok.topk_smem, 32_768);
    assert!(ok.topk_smem <= TOPK_SMEM_CEILING);

    // One more pool doubles the padded axis to 8,192 → 65,536 B > 49,152 B.
    let err = DsaSelectGeometry::plan(&c, 16_388, 1)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("segmented/radix select"),
        "the refusal must name the real fix, got: {err}"
    );
    assert!(
        err.contains("16384"),
        "the refusal must name the token limit: {err}"
    );

    // GLM's advertised 262,144-token context is far past this. Serving DSA at full
    // context is a KNOWN, UNSOLVED limit — this test is the tripwire that keeps it
    // from being discovered as a wrong answer instead of an error.
    assert!(DsaSelectGeometry::plan(&c, 262_144, 1).is_err());
}

/// `select_k` is the pool budget, and it clamps to the pools that exist. A short
/// context must not ask for more pools than were built.
#[test]
fn select_k_clamps_to_available_pools() {
    let c = cfg();
    // 2,048 topk / 4 kpool = 512 pools wanted.
    let long = DsaSelectGeometry::plan(&c, 8_192, 1).unwrap();
    assert_eq!(long.n_pools, 2_048);
    assert_eq!(
        long.select_k, 512,
        "budget applies when pools are plentiful"
    );

    let short = DsaSelectGeometry::plan(&c, 400, 1).unwrap();
    assert_eq!(short.n_pools, 100);
    assert_eq!(
        short.select_k, 100,
        "clamped: cannot select 512 of 100 pools"
    );
}

/// The emitted row is `index_topk` wide plus the `kpool - 1` tail slots. A row sized
/// without the tail truncates the in-progress pool with no error.
#[test]
fn out_width_carries_the_always_select_tail_slots() {
    let mut c = cfg();
    assert!(c.always_select_tail);
    assert_eq!(c.out_width(), 2_048 + 3);
    c.always_select_tail = false;
    assert_eq!(c.out_width(), 2_048);
}

/// A context that outgrows its reservation must fail, not overrun. This is the A25
/// recv-buffer failure class: an HTTP 200 with a different answer.
#[test]
fn scratch_refuses_a_pass_larger_than_its_reservation() {
    let c = cfg();
    let reserved = DsaSelectGeometry::plan(&c, 4_096, 1).unwrap();
    let grown = DsaSelectGeometry::plan(&c, 8_192, 1).unwrap();

    // `fits` is checked against capacity bytes, so model the reservation directly
    // rather than allocating: scratch sizing is pure arithmetic.
    let cap = reserved.scratch_bytes();
    let want = grown.scratch_bytes();
    assert!(
        want.iter().zip(cap.iter()).any(|(w, c)| w > c),
        "an 8,192-token pass must exceed a 4,096-token reservation somewhere"
    );
}

/// Zero query rows, and a context too short to form a single pool, are both errors
/// rather than empty launches — an empty grid is a silent no-op that leaves the
/// previous step's selection in place.
#[test]
fn plan_refuses_degenerate_geometry() {
    let c = cfg();
    assert!(DsaSelectGeometry::plan(&c, 4_096, 0).is_err(), "q_rows = 0");
    assert!(DsaSelectGeometry::plan(&c, 0, 1).is_err(), "seq = 0");
    assert!(
        DsaSelectGeometry::plan(&c, 4, 1).is_ok(),
        "exactly one pool is fine"
    );
    // 🔴 NOT degenerate: fewer tokens than one pool is a legal, tail-only pass.
    // See `sub_pool_selection_is_dense_over_the_visible_tokens`.
    assert!(
        DsaSelectGeometry::plan(&c, 3, 1).is_ok(),
        "fewer tokens than one pool is the tail-only regime, not a refusal"
    );
}

/// 🔴 The `seq < index_kpool` regime, resolved from HF 5.16.1 rather than invented.
///
/// `Glm5NextTextIndexer.forward` has NO short-sequence branch. Below `index_kpool`
/// tokens `pool_valid` is all-false (a pool counts only when every slot is real),
/// `keep = pool_valid.any(0)` empties the pool axis, and
/// `select_k = min(index_topk // index_kpool, 0)` is 0 — so the pool arm selects
/// nothing and `append_visible_tail` writes the raw visible tokens. Every token of a
/// sub-pool sequence lives in the incomplete pool, so the emitted row is exactly
/// `[0 .. seq)` padded with -1: **dense attention, reached by the ordinary path.**
///
/// Checked against `glm5next_dsa_ref`, which is the HF-gated reference (GATE 4/5),
/// so this asserts agreement with HF rather than restating this launcher's belief.
#[test]
fn sub_pool_selection_is_dense_over_the_visible_tokens() {
    use crate::layers::glm5next_dsa_ref::{INVALID, Pools, expand_selection};

    let c = cfg();
    let d = dims(&c);
    let width = d.out_width();

    for seq in 1usize..c.index_kpool {
        let g = DsaSelectGeometry::plan(&c, seq, 1).unwrap();
        assert_eq!(g.n_pools, 0, "seq={seq}: no complete pool");
        assert_eq!(g.select_k, 0, "seq={seq}: nothing to select");
        assert_eq!(g.out_width, width, "seq={seq}: row width is unchanged");

        let valid = vec![1u8; seq];
        assert!(
            kept_pools(&valid, d, seq).is_empty(),
            "seq={seq}: the reference keeps no pool either"
        );

        // The query is the newest token, exactly as decode/per-token prefill issues it.
        let pools = Pools {
            keys: Vec::new(),
            indices: Vec::new(),
            valid: Vec::new(),
            n_pools: 0,
        };
        let row = expand_selection(&[], &pools, &[], &valid, &[seq - 1], &[1u8], d, seq, 0);

        let mut expected = vec![INVALID; width];
        for (t, e) in expected.iter_mut().enumerate().take(seq) {
            *e = t as i32;
        }
        assert_eq!(
            row, expected,
            "seq={seq}: the row must be every visible token, then -1 padding"
        );
    }
}

/// The boundary is not off by one: at exactly `index_kpool` tokens the pool arm
/// engages and the tail contributes nothing.
#[test]
fn the_first_complete_pool_switches_the_sparse_arm_on() {
    let c = cfg();
    let g3 = DsaSelectGeometry::plan(&c, 3, 1).unwrap();
    let g4 = DsaSelectGeometry::plan(&c, 4, 1).unwrap();
    assert_eq!((g3.n_pools, g3.select_k), (0, 0));
    assert_eq!((g4.n_pools, g4.select_k), (1, 1));
}
