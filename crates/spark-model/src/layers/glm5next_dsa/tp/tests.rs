// SPDX-License-Identifier: AGPL-3.0-only

//! DSA TP shard-plan proofs. No GPU, no checkpoint — pure row arithmetic, pinned
//! against the measured layer-3 tensor sizes of
//! `LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3`.

use super::*;

/// GLM-5.3 DSA geometry at TP=1 (`local_heads == full_heads`).
fn cfg(local_heads: usize) -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
    }
}

/// `attn_DSA` = 2,748,117,504 B over 11 text DSA layers.
const DSA_LAYER_BYTES: usize = 249_828_864;

#[test]
fn full_shapes_match_the_reference_checkpoint() {
    let p = DsaTpPlan::new(0, 1, &cfg(64)).unwrap();
    let expect: &[(&str, usize)] = &[
        ("q_a_proj", 12_582_912),
        ("q_a_layernorm", 3_072),
        ("q_b_proj", 50_331_648),
        ("kv_a_proj_with_mqa", 4_194_304),
        ("kv_a_layernorm", 1_024),
        ("kv_b_proj", 33_554_432),
        ("o_proj", 134_217_728),
        ("indexer.wq_b", 12_582_912),
        ("indexer.wk", 1_048_576),
        ("indexer.k_norm.weight", 256),
        ("indexer.k_norm.bias", 256),
        ("indexer.weights_proj", 262_144),
        ("indexer.index_kpool_compress_gate", 1_048_576),
        ("indexer.index_kpool_compress_ape", 1_024),
    ];
    for (name, bytes) in expect {
        let t = p.get(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(t.full_bytes(), *bytes, "{name} full size");
    }
    assert_eq!(p.tensors.len(), expect.len(), "tensor count drifted");
    assert_eq!(p.full_bytes(), DSA_LAYER_BYTES, "per-layer DSA bytes");
    // Reconciles with the checkpoint accounting: 11 text DSA layers, and layer 45
    // (MTP) carries one more of exactly the same size.
    assert_eq!(p.full_bytes() * 11, 2_748_117_504);
    assert_eq!(p.full_bytes(), 249_828_864);
}

#[test]
fn tp1_is_inert() {
    let p = DsaTpPlan::new(0, 1, &cfg(64)).unwrap();
    assert!(!p.needs_output_all_reduce());
    for t in &p.tensors {
        assert_eq!(t.local_rows, t.full_rows, "{} rows", t.name);
        assert_eq!(t.local_row_elems, t.full_row_elems, "{} row elems", t.name);
        assert_eq!(t.src_row_offset, 0, "{} row off", t.name);
        assert_eq!(t.src_col_offset, 0, "{} col off", t.name);
    }
    assert_eq!(p.local_bytes(), p.full_bytes());
}

/// THE gate: at TP=2 every sharded tensor partitions exactly, and everything else
/// replicates.
#[test]
fn tp2_partitions_sharded_tensors_and_replicates_the_rest() {
    let (r0, r1) = (
        DsaTpPlan::new(0, 2, &cfg(32)).unwrap(),
        DsaTpPlan::new(1, 2, &cfg(32)).unwrap(),
    );
    assert_eq!(r0.full_heads, 64);
    assert_eq!(r0.local_heads, 32);
    assert!(r0.needs_output_all_reduce());

    for (a, b) in r0.tensors.iter().zip(r1.tensors.iter()) {
        assert_eq!(a.name, b.name);
        match a.kind {
            DsaShard::Replicated => {
                assert_eq!(a.local_bytes(), a.full_bytes(), "{} replicates", a.name);
                assert_eq!(b.local_bytes(), b.full_bytes(), "{} replicates", b.name);
            }
            DsaShard::HeadRows => {
                assert_eq!(a.src_row_offset, 0);
                assert_eq!(b.src_row_offset, a.local_rows, "{} rank1 start", a.name);
                assert_eq!(a.local_rows + b.local_rows, a.full_rows, "{}", a.name);
                assert_eq!(a.local_bytes() + b.local_bytes(), a.full_bytes());
            }
            DsaShard::HeadCols => {
                assert_eq!(a.local_rows, a.full_rows);
                assert_eq!(a.src_col_offset, 0);
                assert_eq!(b.src_col_offset, a.local_row_elems);
                assert_eq!(a.local_row_elems + b.local_row_elems, a.full_row_elems);
            }
        }
    }
}

/// 🔴 The whole indexer replicates. If this ever flips to sharded, the two ranks
/// can select DIFFERENT tokens — a wrong answer with no crash. See the module docs
/// for the bandwidth arithmetic that decided it.
#[test]
fn the_entire_indexer_replicates() {
    let p = DsaTpPlan::new(1, 2, &cfg(32)).unwrap();
    let idx: Vec<_> = p
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("indexer."))
        .collect();
    assert_eq!(idx.len(), 7, "indexer tensor count");
    for t in idx {
        assert_eq!(
            t.kind,
            DsaShard::Replicated,
            "{} must replicate — a sharded indexer makes ranks select different tokens",
            t.name
        );
        assert_eq!(t.local_bytes(), t.full_bytes());
    }
}

/// 🪤 The shared MQA latent projection has no head axis and must replicate.
#[test]
fn latent_kv_projection_replicates() {
    let p = DsaTpPlan::new(1, 2, &cfg(32)).unwrap();
    let kva = p.get("kv_a_proj_with_mqa").unwrap();
    assert_eq!(kva.kind, DsaShard::Replicated);
    // NoPE: the latent is exactly kv_lora_rank wide, 512 not 576.
    assert_eq!(kva.full_rows, 512);
    for n in ["q_a_proj", "q_a_layernorm", "kv_a_layernorm"] {
        assert_eq!(p.get(n).unwrap().kind, DsaShard::Replicated, "{n}");
    }
}

/// 🪤 `q_b_proj` and `kv_b_proj` shard by head at DIFFERENT per-head widths.
#[test]
fn q_b_and_kv_b_shard_at_their_own_per_head_widths() {
    let c = cfg(32);
    let p = DsaTpPlan::new(1, 2, &c).unwrap();
    let qb = p.get("q_b_proj").unwrap();
    let kvb = p.get("kv_b_proj").unwrap();

    // q_b: qk_head_dim = nope + rope = 256 + 0.
    assert_eq!(qb.full_rows, 64 * 256);
    assert_eq!(qb.local_rows, 32 * 256);
    assert_eq!(qb.src_row_offset, 32 * 256);
    // kv_b: nope + v_dim = 256 + 256 = 512 per head — twice q_b's stride.
    assert_eq!(kvb.full_rows, 64 * 512);
    assert_eq!(kvb.local_rows, 32 * 512);
    assert_eq!(kvb.src_row_offset, 32 * 512);
    assert_ne!(
        qb.local_rows, kvb.local_rows,
        "different per-head widths; reusing one stride for the other mixes heads"
    );
}

#[test]
fn o_proj_is_row_parallel() {
    let p = DsaTpPlan::new(1, 2, &cfg(32)).unwrap();
    let o = p.get("o_proj").unwrap();
    assert_eq!(o.kind, DsaShard::HeadCols);
    assert_eq!(o.full_rows, 4096, "hidden is never sharded");
    assert_eq!(o.full_row_elems, 64 * 256);
    assert_eq!(o.local_row_elems, 32 * 256);
    assert_eq!(o.src_col_offset, 32 * 256);
}

/// NoPE geometry: the latent cache is 512, not DeepSeek-V4-Flash's 576.
#[test]
fn nope_kv_cache_dim_is_kv_lora_exactly() {
    let c = cfg(64);
    assert!(c.is_nope());
    assert_eq!(c.qk_rope_head_dim, 0);
    assert_eq!(c.kv_cache_dim(), 512);
    assert_eq!(c.qk_head_dim(), 256);
}

/// Selection budget and emitted row width.
#[test]
fn select_budget_and_out_width() {
    let c = cfg(64);
    // 2048 / 4 = 512 pools, capped by how many pools exist.
    assert_eq!(c.select_k(100_000), 512);
    assert_eq!(c.select_k(7), 7, "budget is capped by the pool count");
    // always_select_tail widens the row by kpool - 1.
    assert_eq!(c.out_width(), 2048 + 3);
}

#[test]
fn config_rejects_a_topk_that_is_not_a_multiple_of_kpool() {
    let mut c = cfg(64);
    c.index_topk = 2047;
    let e = c.validate().unwrap_err();
    assert!(e.to_string().contains("multiple"), "unexpected: {e}");
}

#[test]
fn config_rejects_a_kpool_beyond_the_kernel_bound() {
    let mut c = cfg(64);
    c.index_kpool = 128;
    c.index_topk = 128 * 16;
    let e = c.validate().unwrap_err();
    assert!(e.to_string().contains("64-slot"), "unexpected: {e}");
}

#[test]
fn tp_rank_must_be_in_range() {
    assert!(DsaTpPlan::new(2, 2, &cfg(32)).is_err());
}

/// 🔴 The latent width is tied to `KV_LORA_DIM` in the MLA decode kernels. A
/// checkpoint whose latent differs must be refused at config time, not read at
/// the wrong width.
#[test]
fn a_latent_the_kernel_cannot_read_is_refused() {
    let mut c = cfg(64);
    assert_eq!(c.kv_lora_rank, super::super::KERNEL_KV_LORA_DIM);
    c.kv_lora_rank = 576; // DeepSeek-V4-Flash's latent+rope width
    let e = c.validate().unwrap_err();
    assert!(e.to_string().contains("KV_LORA_DIM"), "unexpected: {e}");
    assert!(e.to_string().contains("wrong width"), "unexpected: {e}");
}
