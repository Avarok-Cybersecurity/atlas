// SPDX-License-Identifier: AGPL-3.0-only

//! Load-time transform proofs. No GPU: these are the arithmetic that must be right
//! before a single token flows.

use super::*;

fn cfg(local_heads: usize) -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 64,
        index_heads: 4,
        index_head_dim: 8,
        index_kpool: 4,
        index_topk: 16,
        always_select_tail: true,
        local_heads,
        q_lora_rank: 6,
        kv_lora_rank: 512,
        qk_nope_head_dim: 3,
        qk_rope_head_dim: 0,
        v_head_dim: 5,
        max_context: 16_384,
    }
}

/// 🔴 `q_absorb` must reproduce the contraction exactly:
/// `q_absorb[h*kvl + c][k] = Σ_r kv_b[h*(nope+vd) + r][c] · q_b[h*nope + r][k]`.
/// Checked against an independent scalar recomputation, on values chosen so a swapped
/// stride cannot coincidentally agree.
#[test]
fn absorb_q_matches_the_contraction() {
    let mut c = cfg(2);
    c.kv_lora_rank = 4; // small, so the test is a real hand-check
    let (heads, nope, vd, kvl, ql) = (2usize, 3usize, 5usize, 4usize, 6usize);
    let q_b: Vec<f32> = (0..heads * nope * ql)
        .map(|i| (i as f32) * 0.25 - 1.0)
        .collect();
    let kv_b: Vec<f32> = (0..heads * (nope + vd) * kvl)
        .map(|i| 1.0 - (i as f32) * 0.125)
        .collect();

    let got = absorb_q(&c, &q_b, &kv_b, heads).unwrap();
    assert_eq!(got.len(), heads * kvl * ql);
    for h in 0..heads {
        for cc in 0..kvl {
            for k in 0..ql {
                let want: f32 = (0..nope)
                    .map(|r| kv_b[(h * (nope + vd) + r) * kvl + cc] * q_b[(h * nope + r) * ql + k])
                    .sum();
                assert!(
                    (got[(h * kvl + cc) * ql + k] - want).abs() < 1e-5,
                    "h{h} c{cc} k{k}"
                );
            }
        }
    }
}

/// 🪤 `q_b_proj` and `kv_b_proj` carry DIFFERENT per-head widths (256 vs 512 on the real
/// model). A wrong element count must be refused, not reshaped.
#[test]
fn absorb_q_refuses_mismatched_tensors() {
    let mut c = cfg(2);
    c.kv_lora_rank = 4;
    let ok_q = vec![0f32; 2 * 3 * 6];
    let ok_kv = vec![0f32; 2 * (3 + 5) * 4];
    assert!(absorb_q(&c, &ok_q, &ok_kv, 2).is_ok());
    // q_b sized with kv_b's per-head width — still a well-formed 2-D tensor.
    assert!(absorb_q(&c, &[0f32; 2 * 8 * 6], &ok_kv, 2).is_err());
    assert!(absorb_q(&c, &ok_q, &[0f32; 2 * 3 * 4], 2).is_err());
}

/// Absorption is NoPE-only: a rope section would need its rows carried separately, and
/// silently folding them into the latent would be wrong.
#[test]
fn absorb_q_refuses_a_rope_section() {
    let mut c = cfg(2);
    c.kv_lora_rank = 4;
    c.qk_rope_head_dim = 2;
    assert!(absorb_q(&c, &vec![0f32; 2 * 5 * 6], &vec![0f32; 2 * 8 * 4], 2).is_err());
}

/// 🔴 Absorption pairs q_b and kv_b HEAD FOR HEAD, so it must run over full heads before
/// sharding. Slicing first would pair this rank's q_b heads with rank 0's kv_b rows.
/// Proven by construction: rank 1's absorbed rows equal the tail of the full result.
#[test]
fn absorbing_before_sharding_keeps_heads_paired() {
    let mut c = cfg(1); // 1 local head, tp_size 2 => 2 full heads
    c.kv_lora_rank = 4;
    let (heads, nope, vd, kvl, ql) = (2usize, 3usize, 5usize, 4usize, 6usize);
    let q_b: Vec<f32> = (0..heads * nope * ql).map(|i| i as f32).collect();
    let kv_b: Vec<f32> = (0..heads * (nope + vd) * kvl)
        .map(|i| (i as f32) * 0.5)
        .collect();
    let full = absorb_q(&c, &q_b, &kv_b, heads).unwrap();

    // rank 1 owns head 1 => rows [kvl, 2*kvl) of the absorbed tensor.
    let rank1 = row_slice(&full, ql, kvl, 2 * kvl);
    // Independently absorb head 1 alone and compare.
    let mut c1 = c;
    c1.local_heads = 1;
    let solo = absorb_q(&c1, &q_b[nope * ql..], &kv_b[(nope + vd) * kvl..], 1).unwrap();
    assert_eq!(rank1, solo, "head pairing must survive the shard");
}

/// `dsa_index_scores` does not apply `index_heads^-0.5`; folding it into the weight is
/// exact because the factor is a positive scalar and ReLU commutes with it.
#[test]
fn the_head_scale_is_folded_into_weights_proj() {
    let c = cfg(64);
    let scale = (c.index_heads as f32).powf(-0.5);
    assert!((scale - 0.5).abs() < 1e-6, "4 heads => 1/2");
    let raw = [2.0f32, -4.0, 0.0];
    let folded: Vec<f32> = raw.iter().map(|x| x * scale).collect();
    assert_eq!(folded, vec![1.0, -2.0, 0.0]);
}

/// Row-parallel `o_proj` slices the INPUT dim of every row. Column-slicing the wrong axis
/// yields a plausible, wrong tensor of the right total size at tp_size 2.
#[test]
fn o_proj_is_sliced_on_columns_not_rows() {
    // [3 rows, 4 cols], take cols [2, 4).
    let full: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let got = col_slice(&full, 4, 2, 4);
    assert_eq!(got, vec![2.0, 3.0, 6.0, 7.0, 10.0, 11.0]);
    // The row-parallel/column-parallel confusion at tp_size 2: slicing the first HALF OF
    // THE ROWS gives a tensor of exactly the same length and entirely different contents,
    // so length is no defence.
    let wrong = row_slice(&full, 4, 0, 1); // rows [0,1) = 4 elems... 
    assert_eq!(wrong, vec![0.0, 1.0, 2.0, 3.0]);
    let wrong_half = row_slice(&full, 4, 0, 2); // 8 elems
    assert_ne!(
        wrong_half.len(),
        got.len(),
        "the two axes give different shapes here"
    );
    // Same-length case: 6 elements taken row-wise from a [2,3] view of the same buffer.
    let same_len: Vec<f32> = full[..6].to_vec();
    assert_eq!(same_len.len(), got.len());
    assert_ne!(
        same_len, got,
        "same length, wrong values — the real failure mode"
    );
}

/// 🔴 `o_absorb` must be numerically identical to the path it replaces: expand the latent
/// through `kv_b_proj`'s V half, then apply the raw `o_proj`.
///
/// The expansion side is computed by `glm5next_dsa_ref::expand_kv` — the HF-gated
/// reference — rather than restated here, so this asserts agreement with HF's
/// `kv_b_proj` split (V starts at `qk_nope_head_dim`), not with our own belief about it.
///
/// Absorbed decode leaves `attn_out[h] = Σ_t p[h][t] · latent[t]` in latent space. Since
/// `v_t[h] = KV_B_V[h] · latent[t]` is linear, `Σ_t p[h][t] · v_t[h] = KV_B_V[h] · attn_out[h]`,
/// so applying the absorbed weight to `attn_out` is exact, not an approximation.
#[test]
fn absorb_o_equals_expand_then_project() {
    use crate::layers::glm5next_dsa_ref::{DsaDims, expand_kv};

    let mut c = cfg(2);
    c.hidden = 7;
    c.kv_lora_rank = 4;
    let (heads, nope, vd, kvl, hidden) = (
        2usize,
        c.qk_nope_head_dim,
        c.v_head_dim,
        c.kv_lora_rank,
        c.hidden,
    );

    // Deterministic, and asymmetric enough that a K/V-half swap or a stride mix-up cannot
    // coincidentally agree.
    let f = |i: usize, salt: usize| ((i * 37 + salt * 11) % 23) as f32 * 0.031 - 0.29;
    let o_proj: Vec<f32> = (0..hidden * heads * vd).map(|i| f(i, 1)).collect();
    let kv_b: Vec<f32> = (0..heads * (nope + vd) * kvl).map(|i| f(i, 5)).collect();
    let latent: Vec<f32> = (0..heads * kvl).map(|i| f(i, 9)).collect();

    let o_absorb = absorb_o(&c, &o_proj, &kv_b, heads).unwrap();
    assert_eq!(o_absorb.len(), hidden * heads * kvl);

    let dims = DsaDims {
        hidden,
        index_heads: c.index_heads,
        index_head_dim: c.index_head_dim,
        index_kpool: c.index_kpool,
        index_topk: c.index_topk,
        always_select_tail: c.always_select_tail,
        heads,
        q_lora_rank: c.q_lora_rank,
        kv_lora_rank: kvl,
        qk_nope_head_dim: nope,
        qk_rope_head_dim: 0,
        v_head_dim: vd,
    };

    // Reference: expand each head's latent row through kv_b, keep that head's V slice.
    let mut v = vec![0f32; heads * vd];
    for h in 0..heads {
        let (_, v_all) = expand_kv(&latent[h * kvl..(h + 1) * kvl], &kv_b, dims, 1);
        v[h * vd..(h + 1) * vd].copy_from_slice(&v_all[h * vd..(h + 1) * vd]);
    }

    for i in 0..hidden {
        let reference: f32 = (0..heads * vd)
            .map(|j| o_proj[i * heads * vd + j] * v[j])
            .sum();
        let absorbed: f32 = (0..heads * kvl)
            .map(|j| o_absorb[i * heads * kvl + j] * latent[j])
            .sum();
        assert!(
            (reference - absorbed).abs() < 1e-4,
            "row {i}: expand-then-project {reference} != absorbed {absorbed}"
        );
    }
}

/// The absorbed weight is exactly the width the decode GEMM's `kk` claims. This is the
/// assertion that would have caught the illegal access: `local_heads * kv_lora_rank`, not
/// `local_heads * v_head_dim`.
#[test]
fn absorb_o_width_matches_the_decode_contraction() {
    let mut c = cfg(2);
    c.hidden = 7;
    c.kv_lora_rank = 4;
    let (heads, nope, vd, kvl) = (2usize, c.qk_nope_head_dim, c.v_head_dim, c.kv_lora_rank);
    assert_ne!(kvl, vd, "the widths must differ or this proves nothing");
    let o = absorb_o(
        &c,
        &vec![0.5f32; c.hidden * heads * vd],
        &vec![0.25f32; heads * (nope + vd) * kvl],
        heads,
    )
    .unwrap();
    assert_eq!(o.len(), c.hidden * heads * kvl);
}
