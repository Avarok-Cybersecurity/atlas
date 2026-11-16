// SPDX-License-Identifier: AGPL-3.0-only

//! The speculative CUDA headroom, pinned at the Qwen3.5-4B numbers from the
//! RTX 4060 Ti logs, and the drafters and targets that keep the flat 4 GiB.

use super::{BASE_HEADROOM, FLAT_SPEC_HEADROOM, MTP_MARGIN, capture_bytes, cuda_headroom_for};
use crate::cli;
use avarok_core::config::ModelConfig;
use clap::Parser as _;

const MIB: usize = 1024 * 1024;

fn args(extra: &[&str]) -> cli::ServeArgs {
    args_at("8192", extra)
}

fn args_at(max_seq_len: &str, extra: &[&str]) -> cli::ServeArgs {
    let mut v = vec!["spark", "Qwen/Qwen3.5-4B", "--max-seq-len", max_seq_len];
    v.extend_from_slice(extra);
    cli::ServeArgs::parse_from(v)
}

fn qwen35_4b() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen3_5".to_string();
    c.num_experts = 0;
    c.hidden_size = 2560;
    c
}

fn spec_on(a: &cli::ServeArgs) -> bool {
    a.speculative || a.self_speculative || a.ngram_speculative || a.dflash
}

/// On a target that declares `memory_fit_consumer` (ada), no escape hatch.
fn ada(a: &cli::ServeArgs, c: &ModelConfig) -> (usize, String) {
    cuda_headroom_for(a, c, spec_on(a), false, true)
}

#[test]
fn a_plain_serve_keeps_the_base_headroom_on_every_target() {
    let a = args(&[]);
    for consumer_fit in [true, false] {
        assert_eq!(
            cuda_headroom_for(&a, &qwen35_4b(), false, false, consumer_fit).0,
            BASE_HEADROOM
        );
    }
}

#[test]
fn the_4b_mtp_serve_reserves_its_capture_plus_a_margin_not_4_gib() {
    let a = args(&["--speculative"]);
    let (bytes, why) = ada(&a, &qwen35_4b());
    // "MTP drafter context: allocating 42 MB prompt-hidden capture
    // (8192 x 2560 BF16)" in 4b-accfp8spec8k.log: 40 MiB.
    assert_eq!(capture_bytes(&a, &qwen35_4b()), 8192 * 2560 * 2);
    assert_eq!(bytes, BASE_HEADROOM + 40 * MIB + MTP_MARGIN);
    assert_eq!(bytes / MIB, 1064);
    assert!(why.contains("Qwen MTP head"), "{why}");
    // The inference reserve that log printed was 4,510 MB with the flat
    // 4,096 MB term; it drops by 3,032 MB.
    assert_eq!((FLAT_SPEC_HEADROOM - bytes) / MIB, 3032);
}

#[test]
fn a_target_without_the_row_keeps_the_flat_headroom() {
    // GB10, Hopper, B200, B300: the same Qwen MTP serve reserves 4 GiB.
    let a = args(&["--speculative"]);
    let (bytes, why) = cuda_headroom_for(&a, &qwen35_4b(), true, false, false);
    assert_eq!(bytes, FLAT_SPEC_HEADROOM);
    assert!(why.contains("memory_fit_consumer"), "{why}");
}

#[test]
fn the_capture_scales_with_the_context_it_must_hold() {
    let short = args_at("4096", &["--speculative"]);
    let long = args_at("65536", &["--speculative"]);
    let c = qwen35_4b();
    assert_eq!(capture_bytes(&short, &c), 4096 * 2560 * 2);
    assert_eq!(
        ada(&long, &c).0,
        BASE_HEADROOM + 65536 * 2560 * 2 + MTP_MARGIN
    );
}

#[test]
fn every_drafter_that_is_not_priced_keeps_the_flat_headroom() {
    let c = qwen35_4b();
    // `--dflash` left this list in the port sprint (lane L14): on ada it is
    // priced and clamped to the flat term, see `spec_headroom_dflash_tests.rs`.
    for flags in [&["--self-speculative"][..], &["--ngram-speculative"][..]] {
        let a = args(flags);
        assert_eq!(ada(&a, &c).0, FLAT_SPEC_HEADROOM, "{flags:?}");
    }
    for t in ["glm5_next", "deepseek_v4", "qwen3_next", "minimax_m2"] {
        let mut other = qwen35_4b();
        other.model_type = t.to_string();
        let a = args(&["--speculative"]);
        assert_eq!(ada(&a, &other).0, FLAT_SPEC_HEADROOM, "{t}");
    }
}

#[test]
fn the_escape_hatch_restores_the_flat_headroom() {
    let a = args(&["--speculative"]);
    let (bytes, why) = cuda_headroom_for(&a, &qwen35_4b(), true, true, true);
    assert_eq!(bytes, FLAT_SPEC_HEADROOM);
    assert!(why.contains("AVAROK_SPEC_HEADROOM_FULL"));
}
