// SPDX-License-Identifier: AGPL-3.0-only

//! The `ATLAS_*_M16_TC` LEVER GRAMMAR (#927, split in round 6) and the CTA-tile
//! choice it carries — the pure resolver only. Which arm a row count then takes
//! is `dense_ffn_m16_tc_tests.rs`; the numerics are the GPU oracle's.

use super::{m16_tc_kernel, resolve_m16_tc_levers};
use crate::layers::ops::{W8A16_GEMM_M16_N_TILE, W8A16_GEMM_M16_N_TILE_WIDE};
use spark_runtime::gpu::KernelHandle;

/// Distinct per-arm handles; the mock hands the same placeholder to every
/// kernel name, so the handle is the only thing that separates the two tiles.
const M16TC_K: u64 = 0x167C;
const M16TC_N64_K: u64 = 0x1640;

// ── The lever grammar ──────────────────────────────────────────────────────
//
// Round 6 (1xH100, 2026-09-11, bs16) measured the attention tiers at -21.7% and
// the dense-FFN arm at +13.7% in the SAME serve, net +5.2%. One lever could
// only ship both or neither, so there are now three. These drive the pure
// resolver, never the environment: the production accessor is a process-global
// `OnceLock` and a test that set the variables would leak into every other test
// in this binary.

#[test]
fn no_lever_set_leaves_every_tier_off() {
    let l = resolve_m16_tc_levers(false, false, false, None);
    assert!(!l.ffn, "the FFN arm must default OFF");
    assert!(!l.attn, "the attention tiers must default OFF");
    assert_eq!(l.ffn_n_tile, W8A16_GEMM_M16_N_TILE, "default tile is 32");
}

/// THE POINT OF THE SPLIT: `ATLAS_FFN_M16_TC` must no longer reach attention.
#[test]
fn the_ffn_lever_reaches_the_ffn_arm_only() {
    let l = resolve_m16_tc_levers(true, false, false, None);
    assert!(l.ffn);
    assert!(
        !l.attn,
        "ATLAS_FFN_M16_TC must leave the attention tiers alone"
    );
}

/// ...and its mirror, which is the recipe that buys round 6's -21.7% attention
/// win without its +13.7% FFN loss.
#[test]
fn the_attn_lever_reaches_the_attention_tiers_only() {
    let l = resolve_m16_tc_levers(false, true, false, None);
    assert!(l.attn);
    assert!(!l.ffn, "ATLAS_ATTN_M16_TC must leave the dense FFN alone");
}

/// The umbrella reproduces what the single old lever did — round 6's serve J.
#[test]
fn the_umbrella_sets_both_families() {
    let l = resolve_m16_tc_levers(false, false, true, None);
    assert!(l.ffn && l.attn, "ATLAS_M16_TC is both");
}

/// The umbrella is an OR, not an override: it cannot turn a family off.
#[test]
fn the_umbrella_composes_with_the_narrow_levers() {
    for (ffn, attn) in [(true, false), (false, true), (true, true)] {
        let l = resolve_m16_tc_levers(ffn, attn, true, None);
        assert!(
            l.ffn && l.attn,
            "ffn={ffn} attn={attn} with the umbrella set"
        );
    }
}

#[test]
fn the_n_tile_lever_selects_the_wide_instantiation() {
    assert_eq!(
        resolve_m16_tc_levers(true, false, false, Some("64")).ffn_n_tile,
        W8A16_GEMM_M16_N_TILE_WIDE
    );
    // Anything else keeps the tile that has a receipt, including an explicit
    // 32, an empty value and a typo — the tile is a perf knob and the route log
    // says which one ran, so a bad value must not fail a boot.
    for raw in ["32", "", "128", "yes", "6 4"] {
        assert_eq!(
            resolve_m16_tc_levers(true, false, false, Some(raw)).ffn_n_tile,
            W8A16_GEMM_M16_N_TILE,
            "ATLAS_FFN_M16_TC_NTILE={raw:?} must fall back to 32"
        );
    }
}

/// The tile lever picks the ENTRY POINT, and a shadow that lacks the wide one
/// falls back rather than launching a zero handle.
#[test]
fn the_wide_tile_falls_back_when_its_entry_point_is_absent() {
    let (_, kernel, tile) = m16_tc_kernel(
        W8A16_GEMM_M16_N_TILE_WIDE,
        KernelHandle(M16TC_K),
        KernelHandle(M16TC_N64_K),
    );
    assert_eq!(kernel.0, M16TC_N64_K);
    assert_eq!(tile, W8A16_GEMM_M16_N_TILE_WIDE);

    let (_, kernel, tile) = m16_tc_kernel(
        W8A16_GEMM_M16_N_TILE_WIDE,
        KernelHandle(M16TC_K),
        KernelHandle(0),
    );
    assert_eq!(
        kernel.0, M16TC_K,
        "no n64 entry point => the 32-wide kernel"
    );
    assert_eq!(tile, W8A16_GEMM_M16_N_TILE);

    let (_, kernel, tile) = m16_tc_kernel(
        W8A16_GEMM_M16_N_TILE,
        KernelHandle(M16TC_K),
        KernelHandle(M16TC_N64_K),
    );
    assert_eq!(
        kernel.0, M16TC_K,
        "the default tile never reaches the wide arm"
    );
    assert_eq!(tile, W8A16_GEMM_M16_N_TILE);
}
