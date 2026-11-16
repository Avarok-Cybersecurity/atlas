// SPDX-License-Identifier: AGPL-3.0-only

//! Both directions of the flag-driven skips in [`super`], and the target row
//! that confines them to ada.

use super::mtp_unused_for;
use crate::cli;
use avarok_core::config::ModelConfig;
use clap::Parser as _;

fn args(extra: &[&str]) -> cli::ServeArgs {
    let mut v = vec!["spark", "Qwen/Qwen3.5-4B"];
    v.extend_from_slice(extra);
    cli::ServeArgs::parse_from(v)
}

fn cfg(model_type: &str) -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = model_type.to_string();
    c
}

/// On a target that declares `memory_fit_consumer` (ada).
fn ada(a: &cli::ServeArgs, c: &ModelConfig) -> bool {
    mtp_unused_for(a, c, true)
}

#[test]
fn a_plain_qwen35_serve_does_not_upload_the_mtp_block_on_ada() {
    assert!(ada(&args(&[]), &cfg("qwen3_5")));
    assert!(ada(&args(&[]), &cfg("qwen3_5_moe")));
}

#[test]
fn every_other_target_uploads_the_mtp_block_as_before() {
    // GB10, Hopper, B200, B300: `memory_fit_consumer = false`.
    assert!(!mtp_unused_for(&args(&[]), &cfg("qwen3_5"), false));
    assert!(!mtp_unused_for(&args(&[]), &cfg("qwen3_5_moe"), false));
}

#[test]
fn every_speculative_mode_keeps_the_mtp_block() {
    for flag in [
        "--speculative",
        "--self-speculative",
        "--ngram-speculative",
        "--dflash",
    ] {
        assert!(!ada(&args(&[flag]), &cfg("qwen3_5")), "{flag}");
    }
}

#[test]
fn other_families_keep_their_mtp_tensors() {
    // Their MTP blocks bind through other paths (GLM `layers.N`, DeepSeek
    // `mtp.0.*` loaded for verification), so the rule does not reach them.
    for t in ["deepseek_v4", "glm5_next", "qwen3_next", "minimax_m2"] {
        assert!(!ada(&args(&[]), &cfg(t)), "{t}");
    }
}
