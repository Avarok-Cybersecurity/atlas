// SPDX-License-Identifier: AGPL-3.0-only

//! The `[defaults]` tables that are actually CHECKED IN, parsed with the real
//! build-script parser.
//!
//! Companion to `spark-model`'s `target_defaults_tests`, and deliberately a
//! different question. That file grades the RESOLVER against tables spelled
//! out in Rust. This one grades the DATA: that `kernels/hopper/HARDWARE.toml`
//! really declares the round-9 recipe, that `kernels/gb10/HARDWARE.toml`
//! really declares today's behaviour, and that the file the build reads is the
//! file a reviewer read.
//!
//! An integration test rather than a `#[cfg(test)]` module inside the build
//! script, because cargo never runs a build script's own unit tests — the same
//! reason `tests/kernel_build_flags.rs` and `tests/kernel_target_arch.rs`
//! exist. It compiles `build_defaults.rs` directly, so there is no second
//! parser to drift.

#[path = "../build_defaults.rs"]
mod build_defaults;

use build_defaults::{Defaults, baseline, literal, parse_defaults, read_defaults};

use std::path::PathBuf;

fn kernels_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/atlas-kernels is two levels below the workspace root")
        .join("kernels")
}

fn declared(hw: &str) -> Defaults {
    read_defaults(&kernels_root(), hw)
}

// ── the data ──

/// THE DELIVERABLE, as data. Every value here was a line in an H100 launch
/// script outside this repository before the 2026-09-11 maintainer review
/// ("every Hopper/GB10 divergence is expressed as an env lever set by an H100
/// recipe living outside this repo"). An H100 serve with an empty environment
/// resolves to exactly this.
#[test]
fn hopper_declares_the_round_nine_recipe() {
    let d = declared("hopper");
    assert_eq!(d.hw, "hopper");
    assert_eq!(
        d.cublas_gemm_scope, "ffn,ssm,attn",
        "scoped, not `all`: `head` is parsed but has no dispatch consumer, and \
         `ATLAS_CUBLAS_GEMM=1` arming every family is what cost 10.3 GiB of \
         unledgered SSM dequant (dispatch_config::CublasScope)"
    );
    assert!(d.attn_m16_tc, "round 6: -21.7% on the attention phase");
    assert!(d.lm_head_m16_tc, "+4% on the serve");
    assert_eq!(d.lm_head_batchm_max, 16);
    assert!(d.ssm_batched_recurrent, "+6%, md5-identical output");
    assert!(d.decode_split_silu);
    // The measured losses and the unmeasured tiers stay off. A default is a
    // claim about a measurement.
    assert!(!d.ffn_m16_tc, "round 6: +13.7% on the SSM-layer FFN");
    assert!(
        !d.ffn_batch16_tier,
        "the cuBLASLt FFN arm owns these widths once `cublas_gemm_scope` arms it"
    );
    assert!(!d.attn_ncol_gemv, "no H100 serving receipt");
    assert!(
        !d.gdn_decode_hopper,
        "round 12: the GDN decode twins are bit-identical and SLOWER here — \
         0.83x at contiguous n=1, +6.8% per C=1 nsys step, -0.4% on the serve \
         A/B. The kernel stays in [kernels] overrides; only the default moved"
    );
    // The one row round 13 ADDED to the recipe, and the largest measured win of
    // the campaign: cell T1 against cell A on the same binary, C=1 TTFT
    // 269.1 -> 162.4 ms and 889.3 -> 491.5 ms, C=16 aggregate +21.5%/+31.4%,
    // coherency 4/4, determinism 8/8 x 3.
    assert!(
        d.gdn_prefill_tc,
        "round 13: the tensor-core GDN prefill family is Hopper's default — \
         -39.6%/-44.7% on C=1 TTFT, +21.5%/+31.4% on C=16 aggregate"
    );
    // The row round 14 adds. BIT-IDENTICAL to its parent by construction, so
    // it is on without an accuracy receipt and its worst case is a null; the
    // cost it attacks is nsys round 13's 26 881.8 us = 5.85% of the 4593-token
    // prefill at 96 reads of every token's activation row, one per BA output.
    assert!(
        d.ssm_ba_gates_hopper,
        "the BA-gates twin is bit-identical to its parent and Hopper-only; it \
         is on because it cannot change output and off it re-reads every \
         activation row 96 times (`SSM-BA-GATES-ATTRIBUTION.md`)"
    );
    assert_eq!(d.ssm_decode_ring_slots, "auto");
}

/// THE REGRESSION GATE, as data: `kernels/gb10` declares EXACTLY the baseline,
/// i.e. the literals every resolver hardcoded before the table existed. A GB10
/// serve with an empty environment is unchanged by this whole change, and the
/// way to keep it that way is for this assertion to be an equality against
/// [`baseline`] rather than a list somebody has to remember to update.
#[test]
fn gb10_declares_the_baseline_apart_from_the_measured_w8a8_ceiling() {
    let d = declared("gb10");

    // The one intended divergence, pinned by value so it cannot drift
    // silently in either direction. gate/up is WIDENING (N=17408 > K=5120),
    // down is NARROWING; the crossovers differ by ~6x, which is why there are
    // two rows. Served receipt, spark-256a 2026-09-11, Qwen3.6-27B-FP8 M=949,
    // n=5/leg, complete separation: W8A8 3343.3 ms vs W8A16 2560.4 ms.
    assert_eq!(d.w8a8_prefill_max_m_widening, 64);
    assert_eq!(d.w8a8_prefill_max_m_narrowing, 384);
    assert_eq!(baseline("gb10").w8a8_prefill_max_m_widening, u32::MAX);
    assert_eq!(baseline("gb10").w8a8_prefill_max_m_narrowing, u32::MAX);

    // ...and EVERYTHING ELSE still restates the pre-existing hardcoded
    // defaults. Asserted as an equality against `baseline` rather than a list
    // somebody has to remember to update: normalising only the two fields
    // above keeps a third divergence from slipping in unnoticed.
    let normalised = Defaults {
        w8a8_prefill_max_m_widening: u32::MAX,
        w8a8_prefill_max_m_narrowing: u32::MAX,
        ..d
    };
    assert_eq!(
        normalised,
        baseline("gb10"),
        "apart from the W8A8 prefill ceiling, kernels/gb10/HARDWARE.toml \
         [defaults] must restate the pre-existing hardcoded defaults and \
         nothing else — it exists to SAY what GB10 serves with"
    );
}

/// B200 has no serving receipt of any kind, so it declares the conservative
/// table and NOT Hopper's. Copying a recipe across because both cards are
/// datacentre parts is the reasoning this whole mechanism replaces.
#[test]
fn b200_declares_the_conservative_table_not_hoppers() {
    let d = declared("b200");
    assert_eq!(d, baseline("b200"));
    assert_ne!(
        d.cublas_gemm_scope,
        declared("hopper").cublas_gemm_scope,
        "B200 must not inherit Hopper's measured recipe by resemblance"
    );
    assert!(
        !d.gdn_prefill_tc && declared("hopper").gdn_prefill_tc,
        "the GDN prefill family is ON for Hopper on a Hopper receipt (round 13) \
         and OFF here for want of one — the same rule, stated on the row that \
         most recently moved"
    );
    assert!(
        !d.ssm_ba_gates_hopper && declared("hopper").ssm_ba_gates_hopper,
        "the BA-gates twin is Hopper-only source; B200's common/ does not link \
         it, so the row is inert here and must read false"
    );
}

/// The targets that declare NO `[defaults]` table are unaffected: they resolve
/// to the baseline, which is what their resolvers did before. Named
/// explicitly so adding a hardware tree makes someone decide.
#[test]
fn the_silent_targets_resolve_to_the_baseline() {
    for hw in ["metal", "strix", "strix-hip"] {
        assert_eq!(
            declared(hw),
            baseline(hw),
            "kernels/{hw}/HARDWARE.toml declares no [defaults] and must be \
             byte-for-byte unaffected"
        );
    }
}

/// A HOPPER-ONLY kernel's lever still gets a row in every table that declares
/// one. `gdn_decode_hopper` is the second such row (`gdn_prefill_tc` was the
/// first): the twins live only in `kernels/hopper/common`, gb10 never compiles
/// them, and b200 does only because its `common/` symlinks Hopper's. A row
/// present in one target's table and missing from another's is how a lever
/// comes to mean two things in one repository — `parse_defaults` would read
/// the absence as agreement with the baseline, silently, which is the exact
/// failure this table was built to end.
#[test]
fn a_hopper_only_lever_is_still_declared_by_every_table() {
    for hw in ["hopper", "gb10", "b200"] {
        let raw = std::fs::read_to_string(kernels_root().join(hw).join("HARDWARE.toml"))
            .unwrap_or_else(|e| panic!("kernels/{hw}/HARDWARE.toml: {e}"));
        for lever in [
            "gdn_decode_hopper",
            "gdn_prefill_tc",
            // #928. The BA-gates twin is the third hopper-only boolean, and
            // gb10 and b200 declare the row false rather than omitting it.
            "ssm_ba_gates_hopper",
            // #917. GB10 caps, hopper and b200 declare u32::MAX. The row is
            // mandatory everywhere for the same reason as the two above: an
            // absent cap and a deliberate no-cap must not look identical.
            "w8a8_prefill_max_m_widening",
            "w8a8_prefill_max_m_narrowing",
        ] {
            assert!(
                raw.contains(&format!("\n{lever} = ")),
                "kernels/{hw}/HARDWARE.toml [defaults] must declare `{lever}` \
                 explicitly, not inherit it from the baseline"
            );
        }
        assert!(!declared(hw).gdn_decode_hopper, "no target ships them on");
    }
}

// ── the parser ──

/// A target declares only what it DIFFERS on; every absent key falls through
/// to the baseline. Without this a new lever would silently change every
/// target that had not been updated yet.
#[test]
fn absent_keys_fall_through_to_the_baseline() {
    let toml: toml::Value = "[defaults]\nlm_head_batchm_max = 16\n".parse().unwrap();
    let d = parse_defaults("fictional", &toml);
    assert_eq!(d.lm_head_batchm_max, 16);
    assert_eq!(
        Defaults {
            lm_head_batchm_max: baseline("fictional").lm_head_batchm_max,
            ..d
        },
        baseline("fictional"),
        "one declared key must move one field"
    );
}

/// A MISTYPED lever name must fail the build, not read as agreement with the
/// baseline. This is the failure mode a per-target default table cannot have:
/// it would re-create, inside the fix, exactly the silent divergence the
/// review objected to.
#[test]
#[should_panic(expected = "has no key `attn_m16_tcc`")]
fn an_unknown_lever_name_fails_the_build() {
    let toml: toml::Value = "[defaults]\nattn_m16_tcc = true\n".parse().unwrap();
    let _ = parse_defaults("fictional", &toml);
}

/// …and so must a value of the wrong TYPE, naming the key.
#[test]
#[should_panic(expected = "[defaults] attn_m16_tc must be a bool")]
fn a_mistyped_value_fails_the_build_naming_the_key() {
    let toml: toml::Value = "[defaults]\nattn_m16_tc = \"yes\"\n".parse().unwrap();
    let _ = parse_defaults("fictional", &toml);
}

/// A tree with no HARDWARE.toml at all resolves to the baseline rather than
/// panicking: the generator runs on the `ATLAS_SKIP_BUILD` path, where
/// `kernels/` may not be present (a vendored crate, a docs build). The normal
/// build still panics on a bad HARDWARE.toml, in `resolve_targets`.
#[test]
fn a_missing_hardware_toml_resolves_to_the_baseline() {
    let nowhere = kernels_root().join("no-such-hardware-tree-for-tests");
    assert_eq!(
        read_defaults(&nowhere, "gb10"),
        baseline("gb10"),
        "a missing tree must not fail a skip build"
    );
}

// ── the generated constant ──

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
        "gdn_prefill_tc: true",
        "ssm_ba_gates_hopper: true",
        "decode_split_silu: true",
        "ssm_decode_ring_slots: \"auto\"",
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
    assert_eq!(baked.gdn_prefill_tc, declared.gdn_prefill_tc);
    assert_eq!(baked.ssm_ba_gates_hopper, declared.ssm_ba_gates_hopper);
    assert_eq!(baked.decode_split_silu, declared.decode_split_silu);
    assert_eq!(baked.ssm_decode_ring_slots, declared.ssm_decode_ring_slots);
}
