// SPDX-License-Identifier: AGPL-3.0-only

//! The #915 second pass, pinned at the numbers the H100 actually produced.
//!
//! Source: `h100-round6-report.md` (Qwen/Qwen3.8-27B-FP8 on one 80 GB H100,
//! native-FP8 profile, the residency fix in) and the preflight sweep in
//! `h100-round2-report.md`. Every constant below is a line from one of those
//! logs, not a number chosen to make an assertion pass:
//!
//! | term | round-6 value | source line |
//! |---|---|---|
//! | total / util | 79.2 GB x 90% = 71.3 GB budget | `KV cache: …` |
//! | weights | 28.75 GB | `Weights: 28.75 GB, …` |
//! | derived | 4.24 GB | `native FP8 dense residency: …` |
//! | buffer arena | 3,658 MB | `Preflight reserve: … buffer_arena=3658 MB` |
//! | reserve @bs16 | 25,107 MB (ring 18.94 GB) | `Preflight reserve: inference=…` |
//! | reserve @bs32 | 46,923 MB (ring 37.88 GB) | round-2 probe |
//! | reserve @bs64 | 51,771 MB (ring 4 = 37.88 GB) | round-2 probe |
//!
//! The verdicts these produce are the ones the operator had to reach by hand:
//! bs=16 boots at the full depth 8, bs=32 needs depth 4 (which round 6 had to
//! pass as `--ssm-decode-ring-slots 4`), bs=64 needs depth 1.

use super::super::decode_ring::{fit_ring, slot_bytes};
use super::*;

use atlas_core::config::{LayerType, QuantizationConfig};
use clap::Parser as _;

const GIB_F: f64 = 1024.0 * 1024.0 * 1024.0;
const MIB: usize = 1024 * 1024;

/// 48 GDN layers x (h + conv) = 151.5 MiB, the same constant
/// `decode_ring_tests.rs` derives from the 27B's GDN dims.
const PER_SEQ_BLOB: usize = 48 * ((48 * 128 * 128 * 4) + ((16 * 128 * 2 + 48 * 128) * 4 * 4));

fn gib(x: f64) -> usize {
    (x * GIB_F) as usize
}

fn args(batch: usize) -> cli::ServeArgs {
    cli::ServeArgs::parse_from([
        "spark",
        "Qwen/Qwen3.8-27B-FP8",
        "--max-batch-size",
        &batch.to_string(),
        "--max-seq-len",
        "24576",
        "--kv-cache-dtype",
        "fp8",
    ])
}

/// The same config fixture `predicted_residency_tests.rs` uses, so the KV
/// floor and the derived prediction are priced off one model.
fn qwen38_27b() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen3_5".to_string();
    c.num_experts = 0;
    c.num_experts_per_tok = 0;
    c.moe_intermediate_size = 0;
    c.hidden_size = 5120;
    c.intermediate_size = 17408;
    c.num_hidden_layers = 64;
    c.num_attention_heads = 24;
    c.num_key_value_heads = 4;
    c.head_dim = 256;
    c.attn_gated = true;
    c.linear_num_key_heads = 16;
    c.linear_key_head_dim = 128;
    c.linear_num_value_heads = 48;
    c.linear_value_head_dim = 128;
    c.full_attention_interval = 4;
    c.layer_types = (0..64)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    c.quantization_config = Some(QuantizationConfig {
        quant_method: "fp8".to_string(),
        quant_algo: String::new(),
        format: String::new(),
        ignore_modules: Vec::new(),
    });
    c
}

/// Round 6's pre-KV terms, with `fixed` recovered from the logged reserve by
/// subtracting the depth-8 ring — which is how the reserve is composed
/// (`fixed_reserve + slots * slot_bytes`), so this is arithmetic on the log,
/// not a fitted parameter.
fn round6_headroom(batch: usize, reserve_mib: usize, util: f64) -> Headroom {
    let ring8 = 8 * batch * PER_SEQ_BLOB;
    let fixed = reserve_mib * MIB - ring8;
    let budget = (gib(79.2) as f64 * util) as usize;
    let weights = gib(28.75);
    // The predictor's own answer for this config — see
    // `predicted_residency_tests::the_27b_prediction_matches_the_round6_residency_summary`.
    let derived = 4_242_882_560usize;
    let arena = 3658 * MIB;
    let a = args(batch);
    Headroom {
        budget,
        weights,
        derived,
        arena,
        fixed,
        kv_floor: kv_floor_bytes(&a, &qwen38_27b(), KvCacheDtype::Fp8),
        headroom: budget.saturating_sub(weights + derived + arena + fixed),
    }
}

fn fitted(batch: usize, reserve_mib: usize, util: f64) -> usize {
    let a = args(batch);
    let y = Yardstick::PostLoad(round6_headroom(batch, reserve_mib, util));
    fit_ring(
        &a,
        8,
        slot_bytes(&a, PER_SEQ_BLOB),
        PER_SEQ_BLOB,
        // `reserve_without_ring` is ignored in post-load mode — the ring
        // competes with the KV floor inside the headroom, not with the rest
        // of the reserve inside free memory.
        usize::MAX,
        0,
        &y,
        false,
    )
    .slots
}

/// Bytes per cached token come from the KV cache's own helper, so the floor
/// and the real pool cannot be priced differently. 16 attention layers x
/// (K + V) x 4 kv heads x 128... x 1 byte of FP8 = 32 KiB/token on the 27B,
/// which makes the floor an exact power of two per batch slot.
#[test]
fn the_kv_floor_is_the_cache_s_own_bytes_per_token() {
    let c = qwen38_27b();
    assert_eq!(c.num_attention_layers(), 16);
    assert_eq!(
        kv_bytes_per_token(&args(16), &c, KvCacheDtype::Fp8),
        16 * 2 * 4 * 256,
    );
    assert_eq!(kv_floor_tokens(), DEFAULT_KV_FLOOR_TOKENS);
    // 16 seqs x 4096 tokens x 32 KiB = exactly 2 GiB.
    assert_eq!(kv_floor_bytes(&args(16), &c, KvCacheDtype::Fp8), 2 << 30);
    assert_eq!(kv_floor_bytes(&args(32), &c, KvCacheDtype::Fp8), 4 << 30);
    assert_eq!(kv_floor_bytes(&args(64), &c, KvCacheDtype::Fp8), 8 << 30);
    // BF16 KV is twice the bytes, so twice the floor — the floor tracks the
    // dtype the cache will really be built with, not a constant.
    assert_eq!(
        kv_floor_bytes(&args(16), &c, KvCacheDtype::Bf16),
        2 * kv_floor_bytes(&args(16), &c, KvCacheDtype::Fp8),
    );
}

/// A shorter `--max-seq-len` than the floor caps it: reserving room for 4096
/// tokens a sequence can never reach would be a floor on a serve that cannot
/// use it.
#[test]
fn the_floor_never_exceeds_max_seq_len() {
    let a = cli::ServeArgs::parse_from([
        "spark",
        "Qwen/Qwen3.8-27B-FP8",
        "--max-batch-size",
        "16",
        "--max-seq-len",
        "1024",
    ]);
    assert_eq!(
        kv_floor_bytes(&a, &qwen38_27b(), KvCacheDtype::Fp8),
        16 * 1024 * 16 * 2 * 4 * 256,
    );
}

/// Round 6, `--max-batch-size 16`: the full depth-8 ring (18.94 GB) fits
/// beside the KV floor inside ~29 GB of headroom, so nothing is shrunk — and
/// the serve did boot, with 13.6 GB of real KV.
#[test]
fn round6_batch16_keeps_the_full_ring() {
    let h = round6_headroom(16, 25_107, 0.90);
    assert!(
        h.headroom > 8 * 16 * PER_SEQ_BLOB + h.kv_floor,
        "headroom {:.2} GB must cover ring 18.94 + floor 2.00",
        h.headroom as f64 / GIB_F,
    );
    assert_eq!(fitted(16, 25_107, 0.90), 8);
}

/// Round 6, `--max-batch-size 32`: the depth-8 ring wants 37.88 GB of a ~27
/// GB headroom, so the ladder drops to 4 — 18.94 GB — which is EXACTLY the
/// depth the operator had to pass by hand (`--ssm-decode-ring-slots 4`) to
/// get that boot, and which performed identically to bs=16 at C=16.
///
/// This is the case the first pass got wrong: it compared the reserve against
/// 78.6 GB of pre-load free memory, kept depth 8, and the serve was refused
/// four minutes later by the KV budget stage.
#[test]
fn round6_batch32_fits_the_ring_the_operator_had_to_pin_by_hand() {
    assert_eq!(fitted(32, 46_923, 0.90), 4);
    let a = args(32);
    let h = round6_headroom(32, 46_923, 0.90);
    assert!(
        4 * slot_bytes(&a, PER_SEQ_BLOB) + h.kv_floor <= h.headroom,
        "the chosen depth must actually fit",
    );
    assert!(
        8 * slot_bytes(&a, PER_SEQ_BLOB) + h.kv_floor > h.headroom,
        "and depth 8 must not — otherwise this test proves nothing",
    );
}

/// `--max-batch-size 64`: both the ring and the floor scale with the batch,
/// so the ladder has to go further down than the batch-32 case.
#[test]
fn batch64_falls_to_a_single_anchor() {
    assert_eq!(fitted(64, 51_771 + (4 * 64 * PER_SEQ_BLOB) / MIB, 0.90), 1);
}

/// A tighter `--gpu-memory-utilization` shrinks the budget and therefore the
/// ring, which is the whole point of fitting against the budget rather than
/// against free memory: at 0.84 the same bs=32 serve drops another rung.
#[test]
fn a_tighter_utilization_costs_rollback_depth_not_concurrency() {
    assert_eq!(fitted(16, 25_107, 0.84), 8);
    assert_eq!(fitted(32, 46_923, 0.84), 2);
    assert!(
        fitted(32, 46_923, 0.84) < fitted(32, 46_923, 0.90),
        "a smaller budget must never buy MORE ring",
    );
}

/// The shrink WARN names the yardstick, quotes the formula at both depths,
/// and still points at the flag that pins one — #915's third bullet.
#[test]
fn the_shrink_warning_carries_the_formula_and_the_yardstick() {
    let a = args(32);
    let y = Yardstick::PostLoad(round6_headroom(32, 46_923, 0.90));
    let fit = fit_ring(
        &a,
        8,
        slot_bytes(&a, PER_SEQ_BLOB),
        PER_SEQ_BLOB,
        usize::MAX,
        0,
        &y,
        false,
    );
    let w = fit.warning.expect("a shrink must be logged, never silent");
    assert!(
        w.contains("ring: 4 slots x 32 seqs x 151.5 MB/seq = 18.94 GB"),
        "{w}"
    );
    assert!(w.contains("(was 37.88 GB)"), "{w}");
    assert!(w.contains("ring + KV floor"), "{w}");
    assert!(
        w.contains("Sized from the predicted post-load KV headroom"),
        "{w}"
    );
    assert!(w.contains("--ssm-decode-ring-slots"), "{w}");
    // And the INFO line spells out every term of the subtraction.
    for term in [
        "budget",
        "weights",
        "derived",
        "arena",
        "fixed reserve",
        "headroom",
        "ring(4)",
        "KV floor",
    ] {
        assert!(
            fit.decision.contains(term),
            "{term} missing: {}",
            fit.decision
        );
    }
}

/// An explicit `--ssm-decode-ring-slots N` pins the depth on the post-load
/// yardstick exactly as it does on the pre-load one: the operator gets that
/// depth or a refusal, never a silent third answer.
#[test]
fn an_explicit_depth_is_not_fitted_against_the_headroom_either() {
    let a = args(32);
    let y = Yardstick::PostLoad(round6_headroom(32, 46_923, 0.90));
    let fit = fit_ring(
        &a,
        8,
        slot_bytes(&a, PER_SEQ_BLOB),
        PER_SEQ_BLOB,
        usize::MAX,
        0,
        &y,
        true,
    );
    assert_eq!(fit.slots, 8);
    assert!(fit.warning.is_none());
}

/// When even depth 0 cannot clear the KV floor the fit goes to 0 and says so,
/// rather than refusing: the headroom is an ESTIMATE, and the stage that
/// MEASURES (the KV budget) is the only one allowed to refuse a boot over it.
#[test]
fn an_unmeetable_floor_shrinks_to_zero_and_defers_to_the_kv_stage() {
    let a = args(32);
    let mut h = round6_headroom(32, 46_923, 0.90);
    h.headroom = h.kv_floor / 2;
    let fit = fit_ring(
        &a,
        8,
        slot_bytes(&a, PER_SEQ_BLOB),
        PER_SEQ_BLOB,
        usize::MAX,
        0,
        &Yardstick::PostLoad(h),
        false,
    );
    assert_eq!(fit.slots, 0);
    let w = fit.warning.expect("this must not be silent");
    assert!(w.contains("the KV budget stage will decide"), "{w}");
}

/// A model whose post-load residency cannot be predicted keeps the first
/// pass's behaviour — and the log says which yardstick was used and why, so
/// "the fitter approved this" is never confused with "the fitter could not
/// look".
#[test]
fn an_unpredictable_route_falls_back_to_pre_load_free_memory_and_says_so() {
    let a = args(32);
    let mut plain = qwen38_27b();
    plain.quantization_config = None;
    let y = post_load_yardstick(
        &a,
        &plain,
        &PostLoadInputs {
            total_mem: gib(79.2),
            model_dir: std::path::Path::new("/nonexistent-checkpoint"),
            kv_dtype: KvCacheDtype::Fp8,
            w8a8_prefill_kernels: true,
        },
        gib(8.0),
        gib(3.5),
    );
    // The exact reason depends on which gate declines FIRST, and one of them
    // (`ATLAS_DENSE_FP8`) is process environment this test must not pin — the
    // reason STRINGS are pinned in
    // `spark_model::weight_loader::predicted_residency`'s own tests. What
    // matters here is that an unpredictable route never reaches the post-load
    // arm, and that it always carries a reason for the log.
    match y {
        Yardstick::PreLoadFree(why) => assert!(!why.is_empty(), "the fallback must say why"),
        Yardstick::PostLoad(_) => panic!("must not predict a route it cannot see"),
    }
}

/// A checkpoint directory that cannot be sized is also a fallback, not a
/// zero: fitting against `budget - 0 - derived - …` would hand the ring the
/// whole card.
#[test]
fn a_missing_checkpoint_is_a_fallback_not_a_zero_weight_estimate() {
    assert_eq!(
        checkpoint_bytes(std::path::Path::new("/nonexistent-ckpt")),
        None
    );
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("a-00001-of-00002.safetensors"),
        vec![7u8; 2048],
    )
    .unwrap();
    std::fs::write(
        dir.path().join("a-00002-of-00002.safetensors"),
        vec![7u8; 1024],
    )
    .unwrap();
    std::fs::write(dir.path().join("README.md"), b"not a weight").unwrap();
    assert_eq!(checkpoint_bytes(dir.path()), Some(3072));
}

/// The index is read for its DISTINCT shards: a 1,606-tensor weight map over
/// 66 shards must not count the shards 1,606 times.
#[test]
fn an_index_counts_each_shard_once() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("s1.safetensors"), vec![0u8; 1000]).unwrap();
    std::fs::write(dir.path().join("s2.safetensors"), vec![0u8; 500]).unwrap();
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        br#"{"metadata":{"total_size":9999},"weight_map":{
            "a":"s1.safetensors","b":"s1.safetensors","c":"s2.safetensors"}}"#,
    )
    .unwrap();
    assert_eq!(checkpoint_bytes(dir.path()), Some(1500));
}

/// The ONLY test in this binary that touches the publication cell, so it
/// cannot seal the depth for anyone else: `autofit` must publish what it
/// fitted, or `TransformerModel::new` allocates 8 slots against a reserve
/// that funded 4.
#[test]
fn the_fitted_depth_is_published_for_the_allocation_side() {
    let a = args(32);
    let y = Yardstick::PostLoad(round6_headroom(32, 46_923, 0.90));
    assert_eq!(
        spark_model::ssm_reserve::published_decode_ring_slots(),
        None
    );
    let fit = super::super::decode_ring::autofit(
        &a,
        8,
        slot_bytes(&a, PER_SEQ_BLOB),
        PER_SEQ_BLOB,
        usize::MAX,
        0,
        &y,
    );
    assert_eq!(fit.slots, 4);
    assert_eq!(
        spark_model::ssm_reserve::published_decode_ring_slots(),
        Some(4),
        "the reserve and the allocation must read one cell",
    );
}
