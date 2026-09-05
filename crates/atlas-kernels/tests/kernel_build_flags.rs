// SPDX-License-Identifier: AGPL-3.0-only

//! The nvcc flag list a target is compiled with: three declaration layers,
//! one merge rule, pinned.
//!
//! `build.rs` used to merge exactly two layers inline —
//! `kernels/<hw>/common/KERNEL.toml` then the model's quant-dir KERNEL.toml,
//! "common's then the model's appended, deduped, model last". A third,
//! LEAST-specific layer now sits under them: `kernels/<hw>/HARDWARE.toml`
//! `[build] extra_nvcc_flags`, for facts about the ARCHITECTURE rather than
//! the model — a define that switches off a datapath the ISA does not have.
//!
//! Without it the only place to say "this architecture has no warp-level
//! block-scaled MMA" is a per-model KERNEL.toml, and on `kernels/hopper` and
//! `kernels/b200` every KERNEL.toml is a symlink into `kernels/gb10`. Saying
//! it there means forking the file — for each model, and again for each model
//! added later — which is the drift `tests/inherited_targets.rs` exists to
//! forbid.
//!
//! An integration test rather than a `#[cfg(test)]` module inside the build
//! script, because cargo never runs a build script's own unit tests — the same
//! reason `tests/kernel_target_arch.rs` and `tests/kernel_shadow_detector.rs`
//! exist.

#[path = "../build_flags.rs"]
mod build_flags;

use build_flags::{flag_key, hardware_extra_flags, merge_extra_flags};

use std::path::{Path, PathBuf};

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

// ── the merge rule ──

/// ORACLE: `build.rs::resolve_targets`, whose comment states the rule —
/// "common parses first as the base; the model toml appends flags (deduped,
/// model last)". HARDWARE.toml is prepended under both, because it is the
/// least specific of the three: a model that needs a different value for the
/// same flag must be able to state it and win.
#[test]
fn the_layers_merge_least_specific_first() {
    assert_eq!(
        merge_extra_flags(
            &s(&["-DATLAS_NO_WARP_BLOCKSCALE_MMA"]),
            &s(&["--fmad=false", "-DTQ_PLUS_SIGNS"]),
            &s(&["--fmad=false"]),
        ),
        s(&[
            "-DATLAS_NO_WARP_BLOCKSCALE_MMA",
            "--fmad=false",
            "-DTQ_PLUS_SIGNS"
        ]),
    );
}

/// A repeated flag appears ONCE, at the position its FIRST declaring layer
/// gave it. Deduping is not cosmetic: `-D` defines repeated with the same
/// spelling are harmless, but a duplicated `--fmad=false` in a command line
/// that a receipt is meant to reproduce makes two builds look different when
/// they are not.
#[test]
fn a_flag_declared_twice_is_emitted_once() {
    let merged = merge_extra_flags(
        &s(&["--fmad=false"]),
        &s(&["--fmad=false", "-DTQ_PLUS_SIGNS"]),
        &s(&["--fmad=false"]),
    );
    assert_eq!(merged, s(&["--fmad=false", "-DTQ_PLUS_SIGNS"]));
    assert_eq!(
        merged.iter().filter(|f| *f == "--fmad=false").count(),
        1,
        "a flag declared by every layer must still be passed once"
    );
}

/// The pre-existing two-layer behaviour is unchanged when no hardware layer is
/// declared — every NVIDIA target except `hopper` and `b200` is in this case,
/// and gb10's compile line must not move.
#[test]
fn no_hardware_layer_leaves_the_old_two_layer_result() {
    assert_eq!(
        merge_extra_flags(
            &[],
            &s(&["--fmad=false", "-DTQ_PLUS_SIGNS"]),
            &s(&["--fmad=false"]),
        ),
        s(&["--fmad=false", "-DTQ_PLUS_SIGNS"]),
    );
}

// ── the per-vendor key ──

/// ORACLE: `build_parse::parse_kernel_toml`, which reads `extra_metal_flags`
/// for Apple and `extra_nvcc_flags` otherwise. The hardware layer must use the
/// SAME key per vendor or a Metal HARDWARE.toml would hand `--fmad=false` to
/// `xcrun metal`, which rejects it.
#[test]
fn the_flag_key_follows_the_vendor() {
    assert_eq!(flag_key("nvidia"), "extra_nvcc_flags");
    assert_eq!(flag_key("amd"), "extra_nvcc_flags");
    assert_eq!(flag_key("apple"), "extra_metal_flags");
    assert_eq!(flag_key("metal"), "extra_metal_flags");
}

/// A HARDWARE.toml that declares the OTHER vendor's key contributes nothing,
/// rather than leaking nvcc flags into a Metal build.
#[test]
fn the_wrong_vendors_key_is_not_read() {
    let toml: toml::Value = toml::from_str("[build]\nextra_nvcc_flags = [\"-DX\"]\n").unwrap();
    assert_eq!(hardware_extra_flags(&toml, "nvidia"), s(&["-DX"]));
    assert!(hardware_extra_flags(&toml, "apple").is_empty());
}

/// No `[build]` table at all is the common case (gb10, strix, metal): empty,
/// not a panic.
#[test]
fn a_hardware_toml_with_no_build_table_declares_no_flags() {
    let toml: toml::Value =
        toml::from_str("[hardware]\nname = \"gb10\"\narch = \"sm_121f\"\n").unwrap();
    assert!(hardware_extra_flags(&toml, "nvidia").is_empty());
}

/// A non-string entry is a typo that would otherwise reach nvcc as nothing at
/// all. Fail the build, loudly, naming the file — the same posture
/// `parse_expected_absent` takes.
#[test]
#[should_panic(expected = "extra_nvcc_flags")]
fn a_non_string_flag_is_refused() {
    let toml: toml::Value = toml::from_str("[build]\nextra_nvcc_flags = [1]\n").unwrap();
    let _ = hardware_extra_flags(&toml, "nvidia");
}

// ── the real tree ──

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/atlas-kernels is two levels below the workspace root")
        .join("kernels")
}

fn hardware_toml(hw: &str) -> toml::Value {
    let path = kernels_root().join(hw).join("HARDWARE.toml");
    toml::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()))
}

/// gb10 declares NO hardware-level flags, and must not: this whole PR turns on
/// gb10's compile line being byte-identical before and after.
#[test]
fn gb10_declares_no_hardware_level_flags() {
    assert!(
        hardware_extra_flags(&hardware_toml("gb10"), "nvidia").is_empty(),
        "kernels/gb10/HARDWARE.toml must add no flags — the guarded W4A4 path \
         is compiled IN on GB10 and its PTX may not move"
    );
}
