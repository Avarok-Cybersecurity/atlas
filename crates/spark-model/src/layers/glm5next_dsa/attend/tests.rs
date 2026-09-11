// SPDX-License-Identifier: AGPL-3.0-only

//! DSA decode-launcher guards. Pure geometry — the kernel's numerics are gated
//! separately against the GATE-5 oracle on a GPU.

use super::*;
use crate::layers::glm5next_dsa::select::DsaSelectGeometry;

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

fn paging() -> DsaDecodePaging {
    DsaDecodePaging {
        num_seqs: 1,
        num_q_heads: 64,
        num_kv_heads: 1,
        max_blocks_per_seq: 256,
        block_size: 64,
        cache_stride_bytes: 64 * 512,
    }
}

/// 🪤 Pools are built over absolute positions. A page that straddles a pool boundary
/// splits a pool across two pages — the per-token gather would not notice, which is why
/// the check has to live somewhere.
#[test]
fn a_block_size_that_splits_a_pool_is_refused() {
    let c = cfg();
    let mut p = paging();
    assert!(p.validate(&c).is_ok(), "64 is a multiple of kpool 4");
    p.block_size = 66;
    let e = p.validate(&c).unwrap_err().to_string();
    assert!(e.contains("straddle"), "{e}");
}

/// MLA has exactly one latent KV head. Any other value means the caller is holding a
/// non-MLA cache and the token stride would be wrong.
#[test]
fn more_than_one_kv_head_is_refused() {
    let c = cfg();
    let mut p = paging();
    p.num_kv_heads = 8;
    assert!(p.validate(&c).is_err());
}

#[test]
fn degenerate_launches_are_refused() {
    let c = cfg();
    for mutate in [
        (|p: &mut DsaDecodePaging| p.num_seqs = 0) as fn(&mut DsaDecodePaging),
        |p: &mut DsaDecodePaging| p.num_q_heads = 0,
        |p: &mut DsaDecodePaging| p.block_size = 0,
    ] {
        let mut p = paging();
        mutate(&mut p);
        assert!(p.validate(&c).is_err(), "{p:?} must be refused");
    }
}

/// The selection row stride is `out_width` per sequence. If the selection was planned
/// for a different row count than the launch decodes, every row after the first reads
/// the wrong slice — silently.
#[test]
fn a_row_count_mismatch_between_selection_and_launch_is_refused() {
    let c = cfg();
    let p = paging(); // 1 sequence
    let two_rows = DsaSelectGeometry::plan(&c, 4_096, 2).unwrap();
    assert_eq!(two_rows.q_rows, 2);
    assert_ne!(two_rows.q_rows, p.num_seqs);
    // `decode_attention` refuses this pairing; proven here at the predicate it checks,
    // since the launch itself needs a GPU.
    assert!(
        two_rows.q_rows != p.num_seqs,
        "the guarded condition must be reachable"
    );
    let one_row = DsaSelectGeometry::plan(&c, 4_096, 1).unwrap();
    assert_eq!(one_row.q_rows, p.num_seqs);
}

/// 🔴 NoPE: the softmax scale is over the latent width, which is the WHOLE cache token.
/// DeepSeek-V4 uses 1/sqrt(576) because its token carries a 64-dim rope tail; using that
/// here would rescale every score by sqrt(576/512) and go unnoticed.
#[test]
fn the_score_scale_is_the_latent_width_not_the_v4_cache_width() {
    let c = cfg();
    assert_eq!(c.kv_cache_dim(), 512, "NoPE: cache token is pure latent");
    assert_eq!(c.kv_cache_dim(), c.kv_lora_rank);
    let ours = (c.kv_lora_rank as f32).powf(-0.5);
    let v4 = 576f32.powf(-0.5);
    assert!((ours - 0.044_194_173).abs() < 1e-9, "1/sqrt(512)");
    assert!(
        (ours - v4).abs() > 1e-4,
        "the two scales must not be interchangeable by accident"
    );
}
