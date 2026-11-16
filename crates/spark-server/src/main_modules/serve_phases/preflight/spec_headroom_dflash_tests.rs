// SPDX-License-Identifier: AGPL-3.0-only

//! The DFlash arm of the speculative CUDA headroom, pinned at the
//! Qwen3.5-4B + `z-lab/Qwen3.5-4B-DFlash` numbers, and the proof that every
//! target that does not declare `memory_fit_consumer` keeps exactly the 4 GiB
//! it always reserved.

use super::{
    BASE_HEADROOM, DFLASH_CTX_CAP, DFLASH_MARGIN, DFLASH_TAP_CEILING, FLAT_SPEC_HEADROOM,
    cuda_headroom_for, dflash_ctx_accumulator_bytes, dflash_pool_dummy_bytes,
};
use crate::cli;
use avarok_core::config::{LayerType, ModelConfig};
use clap::Parser as _;

const MIB: usize = 1024 * 1024;

fn args(extra: &[&str]) -> cli::ServeArgs {
    // 8,192 tokens unless the case states its own (clap refuses a repeat).
    let mut v = vec!["spark", "Qwen/Qwen3.5-4B"];
    if !extra.contains(&"--max-seq-len") {
        v.extend_from_slice(&["--max-seq-len", "8192"]);
    }
    v.extend_from_slice(extra);
    cli::ServeArgs::parse_from(v)
}

/// Qwen3.5-4B / holo-3.1-4b as the preflight sees it: hidden 2560, 32 layers
/// in the 3:1 GDN/full pattern (24 GDN), GDN heads 16 key / 32 value of
/// width 128, conv kernel 4. Per GDN layer: h = 32 x 128 x 128 x 4 B = 2 MiB,
/// conv = 8,192 x 4 x 4 B = 128 KiB. Per sequence: 48 MiB h + 3 MiB conv.
fn qwen35_4b() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen3_5".to_string();
    c.num_experts = 0;
    c.hidden_size = 2560;
    c.num_hidden_layers = 32;
    c.layer_types = (0..32)
        .map(|i| {
            if i % 4 == 3 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    c.linear_num_key_heads = 16;
    c.linear_key_head_dim = 128;
    c.linear_num_value_heads = 32;
    c.linear_value_head_dim = 128;
    c.linear_conv_kernel_dim = 4;
    c.mamba_num_heads = 0;
    c.mamba_head_dim = 0;
    c
}

fn spec_on(a: &cli::ServeArgs) -> bool {
    a.speculative || a.self_speculative || a.ngram_speculative || a.dflash
}

/// A target that declares `memory_fit_consumer` (ada), no escape hatch.
fn ada(a: &cli::ServeArgs, c: &ModelConfig) -> (usize, String) {
    cuda_headroom_for(a, c, spec_on(a), false, true)
}

#[test]
fn the_4b_config_has_the_sizes_the_arithmetic_below_assumes() {
    let c = qwen35_4b();
    assert_eq!(c.num_ssm_layers(), 24);
    assert_eq!(c.ssm_h_state_bytes(), 2 * MIB);
    assert_eq!(c.ssm_conv_state_bytes(), 128 * 1024);
}

/// The unchanged half of the lane's rule: GB10, Hopper, B200 and B300 do not
/// declare `memory_fit_consumer`, so a DFlash serve there reserves the same
/// 4,294,967,296 bytes before and after this change, at any batch, context or
/// gamma.
#[test]
fn a_target_without_the_row_keeps_exactly_4_gib_for_dflash() {
    assert_eq!(FLAT_SPEC_HEADROOM, 4_294_967_296);
    let c = qwen35_4b();
    for extra in [
        &["--dflash"][..],
        &["--dflash", "--max-batch-size", "1"][..],
        &["--dflash", "--max-batch-size", "32", "--dflash-gamma", "8"][..],
        &["--dflash", "--max-seq-len", "262144"][..],
    ] {
        let a = args(extra);
        let (bytes, why) = cuda_headroom_for(&a, &c, true, false, false);
        assert_eq!(bytes, 4_294_967_296, "{extra:?}");
        assert!(why.contains("memory_fit_consumer"), "{why}");
    }
    // The escape hatch, on a target that has the row, is the same number.
    let a = args(&["--dflash", "--max-batch-size", "1"]);
    let (bytes, why) = cuda_headroom_for(&a, &c, true, true, true);
    assert_eq!(bytes, 4_294_967_296);
    assert!(why.contains("AVAROK_SPEC_HEADROOM_FULL"), "{why}");
}

/// Pinned at the 4B with one stream: base 512 + dummy 921 + accumulators 320
/// + margin 1,024 = 2,777 MiB, 1,319 MiB under the flat term.
#[test]
fn the_4b_dflash_serve_at_one_stream_reserves_2777_mib_not_4_gib() {
    let a = args(&["--dflash", "--max-batch-size", "1"]);
    let c = qwen35_4b();
    // Dummy: one base blob (51 MiB) + one verify slot at gamma 16 unknown:
    // 16 x 48 MiB h + 17 x 3 MiB conv + 51 MiB checkpoint = 870 MiB.
    assert_eq!(dflash_pool_dummy_bytes(&a, &c) / MIB, 51 + 870);
    // 1 sequence x 8,192 tokens x 8 taps x 2,560 x 2 B.
    assert_eq!(dflash_ctx_accumulator_bytes(&a, &c), 8192 * 8 * 2560 * 2);
    assert_eq!(dflash_ctx_accumulator_bytes(&a, &c) / MIB, 320);
    let (bytes, why) = ada(&a, &c);
    assert_eq!(bytes / MIB, 512 + 921 + 320 + 1024);
    assert_eq!(bytes / MIB, 2777);
    assert_eq!((FLAT_SPEC_HEADROOM - bytes) / MIB, 1319);
    assert!(why.starts_with("DFlash drafter: base 512"), "{why}");
    assert!(!why.contains("clamped"), "{why}");
}

#[test]
fn an_explicit_gamma_sizes_the_dummy_slot_from_that_gamma() {
    // Block 16 serves at gamma 15 (K = 16): 15 x 48 + 16 x 3 + 51 + 51 MiB.
    let a = args(&["--dflash", "--dflash-gamma", "15", "--max-batch-size", "1"]);
    let c = qwen35_4b();
    assert_eq!(
        dflash_pool_dummy_bytes(&a, &c) / MIB,
        51 + 15 * 48 + 16 * 3 + 51
    );
    assert_eq!(dflash_pool_dummy_bytes(&a, &c) / MIB, 870);
    // A narrower block costs less.
    let narrow = args(&["--dflash", "--dflash-gamma", "7", "--max-batch-size", "1"]);
    assert!(dflash_pool_dummy_bytes(&narrow, &c) < dflash_pool_dummy_bytes(&a, &c));
}

/// At the default `--max-batch-size 8` and 8,192 tokens the accumulators are
/// 2.5 GiB, the sum passes 4 GiB, and the clamp makes pricing a no-op rather
/// than a larger reserve.
#[test]
fn the_default_batch_clamps_to_the_flat_term() {
    let a = args(&["--dflash"]);
    let c = qwen35_4b();
    assert_eq!(dflash_ctx_accumulator_bytes(&a, &c) / MIB, 8 * 320);
    let (bytes, why) = ada(&a, &c);
    assert_eq!(bytes, FLAT_SPEC_HEADROOM);
    assert!(why.contains("clamped to the flat 4096 MB"), "{why}");
}

#[test]
fn the_window_is_capped_at_the_heads_ctx_cap() {
    assert_eq!(DFLASH_CTX_CAP, 16384);
    let long = args(&[
        "--dflash",
        "--max-batch-size",
        "1",
        "--max-seq-len",
        "131072",
    ]);
    let c = qwen35_4b();
    assert_eq!(
        dflash_ctx_accumulator_bytes(&long, &c),
        DFLASH_CTX_CAP * DFLASH_TAP_CEILING * 2560 * 2
    );
    assert_eq!(dflash_ctx_accumulator_bytes(&long, &c) / MIB, 640);
}

#[test]
fn pricing_never_exceeds_the_flat_term_and_never_drops_below_its_floor() {
    let c = qwen35_4b();
    for batch in ["1", "2", "4", "8", "32"] {
        for seq in ["2048", "8192", "32768", "262144"] {
            for gamma in [None, Some("7"), Some("15")] {
                let mut v = vec!["--dflash", "--max-batch-size", batch, "--max-seq-len", seq];
                if let Some(g) = gamma {
                    v.extend_from_slice(&["--dflash-gamma", g]);
                }
                let a = args(&v);
                let (bytes, _) = ada(&a, &c);
                assert!(bytes <= FLAT_SPEC_HEADROOM, "{v:?}");
                assert!(bytes >= BASE_HEADROOM + DFLASH_MARGIN, "{v:?}");
            }
        }
    }
}

#[test]
fn more_streams_and_more_context_cost_more() {
    let c = qwen35_4b();
    let one = ada(
        &args(&["--dflash", "--max-batch-size", "1", "--max-seq-len", "2048"]),
        &c,
    )
    .0;
    let more_ctx = ada(
        &args(&["--dflash", "--max-batch-size", "1", "--max-seq-len", "8192"]),
        &c,
    )
    .0;
    let more_seqs = ada(
        &args(&["--dflash", "--max-batch-size", "2", "--max-seq-len", "2048"]),
        &c,
    )
    .0;
    assert!(more_ctx > one);
    assert!(more_seqs > one);
}

/// Only a Qwen-shaped DFlash serve is priced: another family, or DFlash
/// beside another speculative method, keeps the flat term.
#[test]
fn other_families_and_combined_methods_keep_the_flat_term() {
    let a = args(&["--dflash", "--max-batch-size", "1"]);
    for t in ["glm5_next", "deepseek_v4", "qwen3_next", "minimax_m2"] {
        let mut other = qwen35_4b();
        other.model_type = t.to_string();
        assert_eq!(ada(&a, &other).0, FLAT_SPEC_HEADROOM, "{t}");
    }
    // (`--dflash --speculative` is refused by clap, so the combination to
    // guard is DFlash beside the n-gram method.)
    let both = args(&["--dflash", "--ngram-speculative", "--max-batch-size", "1"]);
    assert_eq!(ada(&both, &qwen35_4b()).0, FLAT_SPEC_HEADROOM);
}

/// The MTP rows are unchanged by the DFlash arm.
#[test]
fn the_mtp_serve_is_untouched() {
    let a = args(&["--speculative"]);
    let (bytes, why) = ada(&a, &qwen35_4b());
    assert_eq!(bytes / MIB, 512 + 40 + 512);
    assert!(why.contains("Qwen MTP head"), "{why}");
}
