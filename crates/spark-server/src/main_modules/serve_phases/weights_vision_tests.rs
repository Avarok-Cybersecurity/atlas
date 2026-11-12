// SPDX-License-Identifier: AGPL-3.0-only

//! `--no-vision` is an explicit opt-out: off unless the operator passes it,
//! and the tower prefixes it skips are the ones the build-time reclaim frees.

use crate::cli;
use clap::Parser as _;

fn args(extra: &[&str]) -> cli::ServeArgs {
    let mut v = vec!["spark", "Qwen/Qwen3.5-4B"];
    v.extend_from_slice(extra);
    cli::ServeArgs::parse_from(v)
}

#[test]
fn the_default_serve_keeps_the_vision_tower() {
    assert!(!args(&[]).no_vision);
}

#[test]
fn no_vision_is_a_plain_switch() {
    assert!(args(&["--no-vision"]).no_vision);
    // It composes with the vision bound flag rather than conflicting with it.
    let a = args(&["--no-vision", "--vision-max-pixels", "1048576"]);
    assert!(a.no_vision);
    assert_eq!(a.vision_max_pixels, 1_048_576);
}

#[test]
fn the_skipped_prefixes_cover_the_4b_tower() {
    // Qwen3.5-4B ships its tower as `model.visual.*` (0.621 GiB); the loader's
    // skip rule and the build-time reclaim share this predicate.
    assert!(spark_runtime::fast_weights::is_vision_tensor(
        "model.visual.patch_embed.proj.weight"
    ));
    assert!(!spark_runtime::fast_weights::is_vision_tensor(
        "model.language_model.layers.0.mlp.gate_proj.weight"
    ));
}
