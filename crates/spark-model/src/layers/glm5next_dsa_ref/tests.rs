// SPDX-License-Identifier: AGPL-3.0-only
//! Structural invariants of the kpool indexer, provable without a GPU or a checkpoint.
//!
//! Numeric agreement with HF 5.16.1 is proven separately, in `examples/dsa_indexer_microtest.rs`,
//! which has the real layer packet. What is tested here is the class of defect that numeric
//! comparison is bad at catching: uninitialised destinations, sentinel handling, and pooling
//! rules that still produce well-formed output when wrong.

use super::*;

fn dims() -> DsaDims {
    DsaDims {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 4, // small so the fixtures are readable; the rules are dim-independent
        index_kpool: 4,
        index_topk: 16,
        always_select_tail: true,
        q_lora_rank: 1536,
        heads: 64,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
    }
}

fn pools_for(seq: usize, valid: &[u8]) -> Pools {
    let d = dims().index_head_dim;
    let k: Vec<f32> = (0..seq * d).map(|i| i as f32 * 0.01).collect();
    let gate: Vec<f32> = (0..seq * d).map(|i| (i % 7) as f32 * 0.1).collect();
    let ape = vec![0.0f32; dims().index_kpool * d];
    pool_states(&k, &gate, valid, &ape, dims(), seq)
}

/// 🪤 A trailing partial pool is NOT a pool. Seven tokens give ONE pool, not two — the second
/// would be `[4,5,6,out-of-range]`, and `pool_valid` requires every slot to be valid.
#[test]
fn a_partial_trailing_pool_is_never_valid() {
    let p = pools_for(7, &[1; 7]);
    // 🔴 And it is not merely marked invalid — HF's `keep = pool_valid.any(0)` DROPS it from the
    // pool axis, so seven tokens yield ONE pool, and `select_k` shrinks with it.
    assert_eq!(
        p.n_pools, 1,
        "the incomplete pool is dropped, not just masked"
    );
    assert_eq!(p.valid, vec![1]);
    assert_eq!(&p.indices[0..4], &[0, 1, 2, 3]);
    assert_eq!(kept_pools(&[1; 7], dims(), 7), vec![0]);
}

/// 🔴 The selection budget is DATA-DEPENDENT, because it is derived from the compacted pool
/// count. Same `index_topk`, different sequence, different `select_k` — and left padding
/// shrinks it further by pushing the last pool past the end.
#[test]
fn select_k_follows_the_compacted_pool_count() {
    let d = dims();
    assert_eq!(pools_for(7, &[1; 7]).n_pools, 1);
    assert_eq!(d.select_k(pools_for(7, &[1; 7]).n_pools), 1);
    assert_eq!(pools_for(8, &[1; 8]).n_pools, 2);
    assert_eq!(d.select_k(pools_for(8, &[1; 8]).n_pools), 2);
    // 8 slots, 2 of them left padding -> 6 real tokens -> ONE complete pool.
    let valid = [0u8, 0, 1, 1, 1, 1, 1, 1];
    assert_eq!(pools_for(8, &valid).n_pools, 1, "padding costs a pool");
    assert_eq!(kept_pools(&valid, d, 8), vec![0]);
}

/// 🪤 Pooling starts at the first VALID token. With left padding `[P,P,A,B,C,D]`, pool 0 must be
/// `[A,B,C,D]` — i.e. raw indices 2..=5 — not `[P,P,A,B]`.
#[test]
fn pooling_starts_at_the_first_valid_token_not_slot_zero() {
    let valid = [0u8, 0, 1, 1, 1, 1];
    let p = pools_for(6, &valid);
    assert_eq!(
        &p.indices[0..4],
        &[2, 3, 4, 5],
        "left padding is skipped, not pooled"
    );
    assert_eq!(p.valid[0], 1);
}

/// A fully invalid pool must produce a zero key, not NaN. torch's softmax over an all `-inf`
/// row is NaN and HF calls `nan_to_num`; the reference has to do the same or the score for that
/// pool becomes NaN and poisons the top-k comparison.
#[test]
fn a_fully_invalid_pool_yields_zero_not_nan() {
    let p = pools_for(8, &[0; 8]);
    assert!(
        p.keys.iter().all(|x| x.is_finite() && *x == 0.0),
        "keys must be 0, saw {:?}",
        &p.keys[..4]
    );
    assert!(p.valid.iter().all(|v| *v == 0));
}

/// 🔴 The vLLM day-0 scar, structurally. When the valid pool count falls below the budget the
/// remaining top-k slots must be the sentinel — never uninitialised, never a clamped index.
///
/// The destination is pre-filled with `INVALID` and every row is written for its full width, so
/// "uninitialised tail" is not a state this function can produce.
#[test]
fn insufficient_valid_pools_leave_the_sentinel_never_garbage() {
    let d = dims();
    let seq = 7; // -> 1 valid pool, budget is index_topk/kpool = 4
    let valid_keys = vec![1u8; seq];
    let pools = pools_for(seq, &valid_keys);
    let select_k = d.select_k(pools.n_pools);
    assert_eq!(
        select_k, 1,
        "one complete pool survives compaction, so the budget is one"
    );

    let q_rows = 1;
    let valid_candidates: Vec<u8> = (0..pools.n_pools)
        .map(|p| {
            let end = pools.indices[p * d.index_kpool + d.index_kpool - 1];
            (pools.valid[p] != 0 && end >= 0 && visible(&valid_keys, 6, end as usize)) as u8
        })
        .collect();
    assert_eq!(valid_candidates, vec![1]);

    let scores = vec![0.5f32];
    let selected = topk_pools(&scores, &valid_candidates, pools.n_pools, q_rows, select_k);
    let out = expand_selection(
        &selected,
        &pools,
        &valid_candidates,
        &valid_keys,
        &[6],
        &[1],
        d,
        seq,
        select_k,
    );
    assert_eq!(out.len(), d.out_width());

    // Exactly the one valid pool's 4 tokens, plus the 3-token tail [4,5,6].
    let real: Vec<i32> = out.iter().copied().filter(|x| *x != INVALID).collect();
    assert_eq!(real, vec![0, 1, 2, 3, 4, 5, 6]);
    // Everything else is the sentinel — and nothing is out of range.
    assert!(
        out.iter()
            .all(|x| *x == INVALID || (*x >= 0 && (*x as usize) < seq))
    );
    assert_eq!(selected, vec![0]);
    // Tokens 4, 5 and 6 are NOT in any pool — they survive only because the TAIL re-admits
    // them, once each. That is the whole point of `index_kpool_always_select_tail`.
    let tail_tokens: Vec<i32> = out.iter().copied().filter(|x| *x >= 4).collect();
    assert_eq!(
        tail_tokens,
        vec![4, 5, 6],
        "the incomplete pool's tokens come back via the tail"
    );
}

/// A padded QUERY row selects nothing at all, and still has a fully written row.
#[test]
fn a_padded_query_row_is_all_sentinel_and_fully_written() {
    let d = dims();
    let seq = 8;
    let valid_keys = vec![1u8; seq];
    let pools = pools_for(seq, &valid_keys);
    let select_k = d.select_k(pools.n_pools);
    let vc = vec![1u8; pools.n_pools];
    let selected = topk_pools(&vec![1.0; pools.n_pools], &vc, pools.n_pools, 1, select_k);
    let out = expand_selection(
        &selected,
        &pools,
        &vc,
        &valid_keys,
        &[7],
        &[0],
        d,
        seq,
        select_k,
    );
    assert_eq!(out.len(), d.out_width());
    assert!(out.iter().all(|x| *x == INVALID));
}

/// 🔴 Ties must resolve deterministically: score descending, then SMALLER pool index.
/// `torch.topk`'s tie order is implementation-defined, so Atlas pins its own.
#[test]
fn ties_resolve_to_the_smaller_pool_index() {
    let scores = vec![1.0f32, 1.0, 1.0, 0.5];
    let vc = vec![1u8; 4];
    let sel = topk_pools(&scores, &vc, 4, 1, 2);
    assert_eq!(sel, vec![0, 1], "equal scores -> ascending index");
    // And the order is stable regardless of how the scores were laid out.
    let sel2 = topk_pools(&[0.5f32, 1.0, 1.0, 1.0], &vc, 4, 1, 3);
    assert_eq!(sel2, vec![1, 2, 3]);
}

/// The mask collapses duplicates, matching HF's `scatter_add(...).ne(0)`. A token listed twice
/// must be attended once, or its value is double-counted in the softmax numerator.
#[test]
fn mask_construction_collapses_duplicates_and_drops_out_of_range() {
    let topk = vec![2i32, 2, 5, INVALID, 99, -7];
    let mask = topk_to_mask(&topk, 1, 6, 8);
    assert_eq!(mask, vec![0, 0, 1, 0, 0, 1, 0, 0]);
    assert_eq!(mask.iter().map(|x| *x as u32).sum::<u32>(), 2);
}

/// 🪤 `k_norm` is a LayerNorm: it subtracts the mean and adds a bias. An RMSNorm in its place
/// produces a same-shaped, same-dtype, plausible result — so this asserts they DIFFER.
#[test]
fn k_norm_is_layernorm_not_rmsnorm() {
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let w = vec![1.0f32; 4];
    let b = vec![0.25f32, -0.5, 0.75, 0.0];
    let ln = layer_norm(&x, &w, &b, 4, 1e-6);
    let rms = rms_norm(&x, &w, 4, 1e-6);
    let spread = ln
        .iter()
        .zip(&rms)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        spread > 0.5,
        "LayerNorm and RMSNorm must be distinguishable, saw {spread}"
    );
    // Mean-centred before the affine: with a zero bias the row sums to ~0.
    let ln0 = layer_norm(&x, &w, &[0.0; 4], 4, 1e-6);
    assert!(ln0.iter().sum::<f32>().abs() < 1e-5);
}

/// NoPE is a structural property, not a tuning knob: the rope section is zero-width, so
/// `qk_head_dim == qk_nope_head_dim` and `expand_kv` must refuse anything else.
#[test]
fn nope_geometry_is_explicit() {
    let d = dims();
    assert!(d.is_nope());
    assert_eq!(d.qk_head_dim(), d.qk_nope_head_dim);
    assert_eq!(d.out_width(), d.index_topk + d.index_kpool - 1);
    // select_k is capped by the pools that exist, not by the budget alone.
    assert_eq!(d.select_k(2), 2);
    assert_eq!(d.select_k(1000), d.index_topk / d.index_kpool);
}
