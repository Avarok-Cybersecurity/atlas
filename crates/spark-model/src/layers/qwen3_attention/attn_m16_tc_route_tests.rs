// SPDX-License-Identifier: AGPL-3.0-only

//! Pure tests for the `ATLAS_ATTN_M16_TC` route lines (#927, H100 round 9
//! cell W).
//!
//! What these DO NOT cover: the `tc` arm-selection boolean itself
//! (`self.m16_tc && self.w8a16_gemm_m16{,_strided}_k.0 != 0 && ...`) lives
//! inline in `qkv_fp8_batch.rs` / `attn/o_proj.rs` and needs a real
//! `Qwen3AttentionLayer` with resolved kernel handles plus a GPU-backed
//! `MultiSeqCtx` to exercise end to end — there is no CPU mock for the
//! multi-seq decode path the way `dense_ffn_m16_tc_tests.rs` has one for the
//! FFN. What IS covered, and is what the round-9 report's complaint was
//! actually about, is the reporting layer: does the log-once latch fire
//! exactly once, does it fire on its own key without touching the sibling
//! tier's key, and does the text name the lever, the kernel and the off
//! switch.

use super::{
    O_PROJ_M16_TC_ROUTE_KEY, O_PROJ_M16_TC_ROUTE_MSG, QKV_M16_TC_ROUTE_KEY, QKV_M16_TC_ROUTE_MSG,
    log_o_proj_m16_tc_route, log_qkv_m16_tc_route,
};
use crate::layers::ops::ModelStats;

#[test]
fn qkv_route_fires_exactly_once_per_model() {
    let stats = ModelStats::new();
    log_qkv_m16_tc_route(&stats);
    log_qkv_m16_tc_route(&stats);
    log_qkv_m16_tc_route(&stats);
    // The latch is already consumed by the calls above, so a direct probe of
    // the same key now returns false — the standard way this crate's tests
    // pin a `stats.once` key (see `model_stats.rs`'s own
    // `a_keyed_latch_fires_once_per_key_per_model`).
    assert!(
        !stats.once(QKV_M16_TC_ROUTE_KEY),
        "three calls must consume the latch exactly once, not three times \
         or zero times"
    );
}

#[test]
fn o_proj_route_fires_exactly_once_per_model() {
    let stats = ModelStats::new();
    log_o_proj_m16_tc_route(&stats);
    log_o_proj_m16_tc_route(&stats);
    assert!(!stats.once(O_PROJ_M16_TC_ROUTE_KEY));
}

#[test]
fn each_tier_fires_its_own_arm_only() {
    // Firing the Q/K/V route must not consume the o_proj key, and vice versa
    // — the whole point of the round-9 fix is that an operator can tell the
    // two tiers apart, which requires the keys to be independent.
    let stats = ModelStats::new();
    log_qkv_m16_tc_route(&stats);
    assert!(
        stats.once(O_PROJ_M16_TC_ROUTE_KEY),
        "the o_proj key must still be unconsumed after only the q/k/v route fired"
    );

    let stats = ModelStats::new();
    log_o_proj_m16_tc_route(&stats);
    assert!(
        stats.once(QKV_M16_TC_ROUTE_KEY),
        "the q/k/v key must still be unconsumed after only the o_proj route fired"
    );
}

#[test]
fn qkv_message_names_the_lever_kernel_band_and_off_switch() {
    let msg = QKV_M16_TC_ROUTE_MSG;
    assert!(msg.contains("ATLAS_ATTN_M16_TC"), "must name the lever");
    assert!(
        msg.contains("w8a16_gemm_m16_strided"),
        "must name the kernel this tier actually launches"
    );
    assert!(msg.contains("5..=16"), "must name the row band");
    assert!(
        msg.contains("w8a16_gemv_batch16_strided"),
        "must name the bit-exact kernel it displaces"
    );
    assert!(
        msg.contains("REASSOCIATED"),
        "must state the numeric consequence"
    );
    assert!(
        msg.contains("Unset it to restore the bit-exact tier"),
        "must name the off switch"
    );
    assert!(
        msg.contains("#927") && msg.contains("cell W"),
        "must cite the receipt"
    );
}

#[test]
fn o_proj_message_names_the_lever_kernel_band_and_off_switch() {
    let msg = O_PROJ_M16_TC_ROUTE_MSG;
    assert!(msg.contains("ATLAS_ATTN_M16_TC"), "must name the lever");
    assert!(
        msg.contains("w8a16_gemm_m16") && !msg.contains("w8a16_gemm_m16_strided"),
        "must name the CONTIGUOUS kernel, not the strided one the q/k/v tier uses"
    );
    assert!(msg.contains("5..=16"), "must name the row band");
    assert!(
        msg.contains("w8a16_gemv_batch16") && !msg.contains("w8a16_gemv_batch16_strided"),
        "must name the CONTIGUOUS bit-exact kernel it displaces"
    );
    assert!(
        msg.contains("REASSOCIATED"),
        "must state the numeric consequence"
    );
    assert!(
        msg.contains("Unset it to restore the bit-exact tier"),
        "must name the off switch"
    );
}
