// SPDX-License-Identifier: AGPL-3.0-only

//! The generated half of `tests/target_defaults.rs`, split from it at the
//! repo's 500-line ceiling.
//!
//! Split when #927's `attn_qkv_fused` / #928's `gdn_spine_vsplit` rows and
//! #928 round 16's `fp8_act_quant_hopper` row met in a merge: each branch left
//! the parent a line or two under the cap and the merge crossed it. A `#[path]`
//! child under `support/` rather than a sibling in `tests/`, so cargo does not
//! auto-discover it as a second test binary with no fixtures — the same seam
//! `hopper_27b.rs` uses for `support/inherited.rs`.
//!
//! The DATA half stays in the parent. This file asks the other question: that
//! the constant the build script EMITS, and the constant this binary was
//! BUILT with, both match the declaration a reviewer read.

use crate::build_defaults::{literal, read_defaults};
use crate::{declared, kernels_root};

/// The emitted `const` must be the initialiser `lib.rs` `include!`s — a
/// `TargetDefaults` literal naming every field. Checked as text because the
/// generator's output is compiled by a LATER rustc invocation, so a missing
/// field would surface as an unrelated error in `atlas-kernels` rather than
/// here.
#[test]
fn the_generated_constant_names_every_field() {
    let generated = literal(&declared("hopper"));
    assert!(generated.contains("pub const TARGET_DEFAULTS: TargetDefaults = TargetDefaults {"));
    for field in [
        "hw: \"hopper\"",
        "cublas_gemm_scope: \"ffn,ssm,attn\"",
        "ffn_batch16_tier: false",
        "ffn_m16_tc: false",
        "attn_m16_tc: true",
        "attn_ncol_gemv: false",
        "lm_head_m16_tc: true",
        "lm_head_batchm_max: 16",
        "ssm_batched_recurrent: true",
        "gdn_decode_hopper: false",
        "gdn_decode_strided_hopper: true",
        "gdn_prefill_tc: true",
        "ssm_ba_gates_hopper: true",
        "fp8_act_quant_hopper: true",
        "ffn_gateup_fused: true",
        "attn_qkv_fused: true",
        "decode_split_silu: true",
        "ssm_decode_ring_slots: \"auto\"",
        "gdn_spine_vsplit: 1",
    ] {
        assert!(
            generated.contains(field),
            "generated constant is missing `{field}`:\n{generated}"
        );
    }
}

/// The constant this BINARY was built with is the one its own hardware tree
/// declares. The join between the generator and the runtime: without it, a
/// build script that wrote the wrong tree's table would pass every test above.
#[test]
fn the_baked_constant_matches_its_own_hardware_tree() {
    let baked = atlas_kernels::TARGET_DEFAULTS;
    // `ATLAS_SKIP_BUILD` (every CPU gate) with no `ATLAS_TARGET_HW` bakes the
    // default tree. Whatever tree it is, its declaration must round-trip.
    let declared = read_defaults(&kernels_root(), baked.hw);
    assert_eq!(baked.hw, declared.hw);
    assert_eq!(baked.cublas_gemm_scope, declared.cublas_gemm_scope);
    assert_eq!(baked.ffn_batch16_tier, declared.ffn_batch16_tier);
    assert_eq!(baked.ffn_m16_tc, declared.ffn_m16_tc);
    assert_eq!(baked.attn_m16_tc, declared.attn_m16_tc);
    assert_eq!(baked.attn_ncol_gemv, declared.attn_ncol_gemv);
    assert_eq!(baked.lm_head_m16_tc, declared.lm_head_m16_tc);
    assert_eq!(baked.lm_head_batchm_max, declared.lm_head_batchm_max);
    assert_eq!(baked.ssm_batched_recurrent, declared.ssm_batched_recurrent);
    assert_eq!(baked.gdn_decode_hopper, declared.gdn_decode_hopper);
    assert_eq!(
        baked.gdn_decode_strided_hopper,
        declared.gdn_decode_strided_hopper
    );
    assert_eq!(baked.gdn_prefill_tc, declared.gdn_prefill_tc);
    assert_eq!(baked.gdn_spine_vsplit, declared.gdn_spine_vsplit);
    assert_eq!(baked.ssm_ba_gates_hopper, declared.ssm_ba_gates_hopper);
    assert_eq!(baked.fp8_act_quant_hopper, declared.fp8_act_quant_hopper);
    assert_eq!(baked.ffn_gateup_fused, declared.ffn_gateup_fused);
    assert_eq!(baked.attn_qkv_fused, declared.attn_qkv_fused);
    assert_eq!(baked.decode_split_silu, declared.decode_split_silu);
    assert_eq!(baked.ssm_decode_ring_slots, declared.ssm_decode_ring_slots);
}
