// SPDX-License-Identifier: AGPL-3.0-only

//! Pure tests for the W4A16/NVFP4 split-SiLU-down route lines.
//!
//! What these DO NOT cover: the `let split_silu = ...` decision itself lives
//! inline in `dense_ffn.rs::forward` and depends on live kernel handles
//! (`self.w4a16_gemv`, `self.act_mul`) plus `self.lora` — exercising it a la
//! `dense_ffn_m16_tc_tests.rs::run_tiled` needs a `MockGpuBackend`-backed
//! `DenseFfnLayer`. What IS covered is the reporting layer this file exists
//! for: the log-once latch fires exactly once, fires on its own key without
//! touching the other arm's key, and the text names the lever, the default
//! kernel and the off switch.

use super::{
    W4A16_DOWN_FUSED_SILU_KEY, W4A16_DOWN_FUSED_SILU_MSG, W4A16_DOWN_SPLIT_SILU_KEY,
    W4A16_DOWN_SPLIT_SILU_MSG, log_w4a16_down_fused_silu_route, log_w4a16_down_split_silu_route,
};
use crate::layers::ops::ModelStats;

#[test]
fn split_silu_route_fires_exactly_once_per_model() {
    let stats = ModelStats::new();
    log_w4a16_down_split_silu_route(&stats);
    log_w4a16_down_split_silu_route(&stats);
    assert!(
        !stats.once(W4A16_DOWN_SPLIT_SILU_KEY),
        "two calls must consume the latch exactly once"
    );
}

#[test]
fn fused_silu_route_fires_exactly_once_per_model() {
    let stats = ModelStats::new();
    log_w4a16_down_fused_silu_route(&stats);
    log_w4a16_down_fused_silu_route(&stats);
    assert!(!stats.once(W4A16_DOWN_FUSED_SILU_KEY));
}

#[test]
fn each_arm_fires_its_own_key_only() {
    let stats = ModelStats::new();
    log_w4a16_down_split_silu_route(&stats);
    assert!(
        stats.once(W4A16_DOWN_FUSED_SILU_KEY),
        "the fused-arm key must still be unconsumed after only the split-SiLU \
         route fired"
    );

    let stats = ModelStats::new();
    log_w4a16_down_fused_silu_route(&stats);
    assert!(
        stats.once(W4A16_DOWN_SPLIT_SILU_KEY),
        "the split-SiLU key must still be unconsumed after only the fused \
         route fired"
    );
}

#[test]
fn split_silu_message_names_the_lever_kernel_and_off_switch() {
    let msg = W4A16_DOWN_SPLIT_SILU_MSG;
    assert!(
        msg.contains("ATLAS_NO_DECODE_SPLIT_SILU"),
        "must name the lever"
    );
    assert!(msg.contains("w4a16_gemv"), "must name the default kernel");
    assert!(
        msg.contains("restores the fused kernel"),
        "must name the off switch's effect"
    );
    assert!(
        msg.contains("NOT bit-identical"),
        "must state the numerics caveat"
    );
}

#[test]
fn fused_silu_message_names_the_lever_and_default_it_departs_from() {
    let msg = W4A16_DOWN_FUSED_SILU_MSG;
    assert!(
        msg.contains("ATLAS_NO_DECODE_SPLIT_SILU"),
        "must name the lever"
    );
    assert!(
        msg.contains("w4a16_gemv_silu_input"),
        "must name the fused kernel actually running"
    );
    assert!(
        msg.contains("split-SiLU + w4a16_gemv default"),
        "must say which path this is NOT, so an operator reading either \
         line alone still learns both names"
    );
}

#[test]
fn the_two_arms_are_textually_distinguishable() {
    // A weak but load-bearing property: if these two strings ever collapsed
    // to the same text, the whole point of a per-arm route line is gone.
    assert_ne!(W4A16_DOWN_SPLIT_SILU_MSG, W4A16_DOWN_FUSED_SILU_MSG);
    assert_ne!(W4A16_DOWN_SPLIT_SILU_KEY, W4A16_DOWN_FUSED_SILU_KEY);
}
