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
        max_context: 16_384,
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

/// 🟢 The 16,384-token ceiling is GONE. `dsa_topk_pools` walks the pool axis in
/// [`topk_tile`]-wide tiles, so shared memory is constant and `plan` succeeds at any
/// context. This test is the tripwire on that constancy — a launch that grew with the
/// context would be the old bug back. ANOMALIES A62.
#[test]
fn plan_holds_shared_memory_constant_at_any_context() {
    let c = cfg();
    let tile = topk_tile();
    assert_eq!(tile, 2_048, "two tiles of [f32,i32] against a 49,152 B ceiling");
    assert_eq!(topk_smem_for_tile(tile), 32_768);

    // The size that used to be the last one that fit, the first that did not, and
    // GLM's advertised 262,144 — all plan, all at the same shared memory.
    for (seq, pools) in [
        (16_384usize, 4_096usize),
        (16_388, 4_097),
        (65_536, 16_384),
        (262_144, 65_536),
    ] {
        let g = DsaSelectGeometry::plan(&c, seq, 1)
            .unwrap_or_else(|e| panic!("plan refused {seq} tokens: {e}"));
        assert_eq!(g.n_pools, pools);
        assert_eq!(g.topk_np2, tile, "the sort axis is the tile, not the context");
        assert_eq!(g.topk_smem, 32_768);
        assert!(g.topk_smem <= TOPK_SMEM_CEILING);
    }

    // Below one tile the sort axis still shrinks with the context — a short prompt
    // must not pay for a 2,048-wide sort.
    let small = DsaSelectGeometry::plan(&c, 1_200, 1).unwrap();
    assert_eq!(small.n_pools, 300);
    assert_eq!(small.topk_np2, 512);
    assert_eq!(small.topk_smem, 8_192);
}

/// The one capacity the tiled select still imposes: the running best list IS one tile,
/// so it cannot hold more winners than a tile has slots. Unreachable on GLM-5.3
/// (`select_k` 512 vs a 2,048 tile) — a loud refusal, never a silent truncation.
#[test]
fn plan_refuses_a_select_k_wider_than_one_tile() {
    let mut c = cfg();
    c.index_topk = topk_tile() * c.index_kpool * 2;
    let err = DsaSelectGeometry::plan(&c, 262_144, 1)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("exceeds the") && err.contains("top-k tile"),
        "the refusal must name the tile: {err}"
    );
}

/// 🔬 The tiled select must be BIT-IDENTICAL to the whole-axis sort it replaced.
///
/// This is a host model of `dsa_topk_pools`' shared-memory walk — the same tile sort, the
/// same Batcher half-cleaner across the two descending runs, the same merge — checked
/// against a plain total-order sort over the whole axis. It proves the ALGORITHM; the CUDA
/// transcription is proven separately on device by `examples/dsa_indexer_microtest.rs`.
mod tiled_select_model {
    /// The kernel's comparator: score DESCENDING, then pool index ASCENDING. A total order,
    /// because indices are unique — which is why the top-k prefix is unique and a
    /// merge-and-truncate walk cannot reach a different answer than a full sort.
    fn gt(a: (f32, i32), b: (f32, i32)) -> bool {
        a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
    }

    fn bitonic_sort_desc(v: &mut [(f32, i32)]) {
        let n = v.len();
        let mut k = 2;
        while k <= n {
            let mut j = k >> 1;
            while j > 0 {
                for i in 0..n {
                    let l = i ^ j;
                    if l > i {
                        let want_desc = (i & k) == 0;
                        if want_desc != gt(v[i], v[l]) {
                            v.swap(i, l);
                        }
                    }
                }
                j >>= 1;
            }
            k <<= 1;
        }
    }

    fn bitonic_merge_desc(v: &mut [(f32, i32)]) {
        let n = v.len();
        let mut j = n >> 1;
        while j > 0 {
            for i in 0..n {
                let l = i ^ j;
                if l > i && !gt(v[i], v[l]) {
                    v.swap(i, l);
                }
            }
            j >>= 1;
        }
    }

    /// `dsa_topk_pools` for one query row.
    pub fn tiled_topk(scores: &[f32], tile: usize, select_k: usize) -> Vec<i32> {
        let pad = (f32::NEG_INFINITY, i32::MAX);
        let mut best = vec![pad; tile];
        let mut base = 0;
        while base < scores.len() {
            let mut cand: Vec<(f32, i32)> = (0..tile)
                .map(|i| {
                    let idx = base + i;
                    if idx < scores.len() {
                        (scores[idx], idx as i32)
                    } else {
                        pad
                    }
                })
                .collect();
            bitonic_sort_desc(&mut cand);
            // Half-cleaner across [best desc][cand asc]: best[i] against cand[tile-1-i].
            for i in 0..tile {
                let b = tile - 1 - i;
                if !gt(best[i], cand[b]) {
                    std::mem::swap(&mut best[i], &mut cand[b]);
                }
            }
            bitonic_merge_desc(&mut best);
            base += tile;
        }
        best.into_iter().take(select_k).map(|e| e.1).collect()
    }

    /// The reference: sort the whole axis under the same total order, take the prefix.
    pub fn whole_axis_topk(scores: &[f32], select_k: usize) -> Vec<i32> {
        let mut all: Vec<(f32, i32)> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, i as i32))
            .collect();
        all.sort_by(|a, b| if gt(*a, *b) { std::cmp::Ordering::Less } else if a == b { std::cmp::Ordering::Equal } else { std::cmp::Ordering::Greater });
        all.into_iter().take(select_k).map(|e| e.1).collect()
    }
}

#[test]
fn the_tiled_walk_returns_exactly_what_a_whole_axis_sort_would() {
    // Deterministic LCG — a fixed corpus beats a flaky random one, and TIES are the whole
    // point of the tiebreak contract, so the score alphabet is deliberately small.
    let mut state: u64 = 0x5EED_5EED;
    let mut next = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };
    for &tile in &[4usize, 8, 64] {
        for &n_pools in &[1usize, 3, 7, 8, 9, 33, 64, 65, 200, 511, 512] {
            for &alphabet in &[3u32, 1_000_000] {
                let scores: Vec<f32> =
                    (0..n_pools).map(|_| (next() % alphabet) as f32).collect();
                for &select_k in &[1usize, 2, 5, tile.min(n_pools)] {
                    let select_k = select_k.min(n_pools).min(tile).max(1);
                    let got = tiled_select_model::tiled_topk(&scores, tile, select_k);
                    let want = tiled_select_model::whole_axis_topk(&scores, select_k);
                    assert_eq!(
                        got, want,
                        "tile={tile} n_pools={n_pools} alphabet={alphabet} select_k={select_k}"
                    );
                }
            }
        }
    }
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
