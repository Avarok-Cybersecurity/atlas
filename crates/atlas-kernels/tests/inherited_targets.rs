// SPDX-License-Identifier: AGPL-3.0-only

//! The hardware targets that INHERIT `kernels/gb10`'s kernels instead of
//! shipping their own — `kernels/hopper` (H100/H200, sm_90a) and
//! `kernels/b200` (B200/GB200, sm_100a) — pinned against the REAL tree.
//!
//! Same posture as `target_resolution.rs`: `src/*_tests.rs` prove the rules on
//! fixtures, these prove the DATA that is actually checked in.
//!
//! Neither target ships a kernel of its own. Every source they compile is a
//! relative symlink into `kernels/gb10`, which makes `gb10` the ORACLE for
//! this whole file: an inherited kernel set is correct exactly when it is
//! gb10's kernel set, reachable. A symlink that dangles, or a gb10 file that
//! gained no counterpart, is a kernel that silently vanishes from that
//! hardware's build — the shadow-drift failure class documented in `build.rs`,
//! arriving through a different door.
//!
//! `cargo test` runs GPU-free with `ATLAS_SKIP_BUILD=1`, where `build.rs`
//! returns before target resolution ever happens, so without this file nothing
//! in CI looks at either tree at all.
//!
//! PARAMETRISED over [`INHERITED`] rather than duplicated per hardware set:
//! the second target arrived by copying the first, and two copies of an
//! assertion drift the moment one is updated. Each test names the hardware set
//! in its failure message, so a red still says which tree is wrong.
//!
//! This binary covers the tree AS MIRRORED — HARDWARE.toml, `common/`, the P0
//! MODEL.tomls, and what `build.rs` would resolve. The `ATLAS_NO_WARP_
//! BLOCKSCALE_MMA` opt-out and the `[expected_absent]` pins that answer it are
//! `inherited_targets_w4a4.rs`; both read [`INHERITED`] from
//! `support/inherited.rs`, so neither can be told about a target the other has
//! not heard of.

#[path = "support/inherited.rs"]
mod inherited;
#[path = "support/mirror.rs"]
mod mirror;

use inherited::{INHERITED, gb10_dir, hardware_toml, hw_dir};
use mirror::mirror_faults;

use std::path::PathBuf;
// ── (a) HARDWARE.toml ──

/// ORACLE: `build.rs::resolve_targets`, which reads exactly `hardware.arch`
/// (required, string) and `hardware.vendor` (steers `resolve_compute_target`
/// and the per-vendor KERNEL.toml flag key). `name` mirrors gb10's file.
///
/// The arch numbers are the point of the test. Hopper is SM 9.0 — `sm_100` is
/// Blackwell datacentre, not Hopper — and B200 is SM 10.0, which is neither
/// consumer Blackwell (`sm_120`) nor GB10 (`sm_121`) nor Blackwell Ultra
/// (`sm_103`). Getting one of those wrong produces a target that compiles and
/// cannot load.
#[test]
fn every_inherited_hardware_toml_declares_its_own_nvidia_arch() {
    for t in INHERITED {
        let toml = hardware_toml(t.hw);
        let hw = toml
            .get("hardware")
            .unwrap_or_else(|| panic!("kernels/{}: no [hardware] table", t.hw));
        assert_eq!(
            hw.get("name").and_then(|v| v.as_str()),
            Some(t.hw),
            "kernels/{}: [hardware].name must match the directory",
            t.hw
        );
        assert_eq!(
            hw.get("vendor").and_then(|v| v.as_str()),
            Some("nvidia"),
            "kernels/{}: vendor picks the compiler in \
             build_target::resolve_compute_target",
            t.hw
        );
        assert_eq!(
            hw.get("arch").and_then(|v| v.as_str()),
            Some(t.arch),
            "kernels/{}: arch is what reaches nvcc -arch=",
            t.hw
        );
        assert_eq!(
            hw.get("compute_capability").and_then(|v| v.as_str()),
            Some(t.cc),
            "kernels/{}: compute_capability is read by tests/target_hints.rs",
            t.hw
        );
    }
}

/// The keys gb10's HARDWARE.toml carries, each inherited target carries too.
/// The `memory_*` keys have no reader in the tree, so this pins the documented
/// shape rather than a behaviour, exactly as strix's file does. The strix
/// precedent is that an unread key must at least be honest: two were deleted
/// there for having no reader, and these survive as roofline/documentation
/// input.
///
/// `compute_capability` is no longer among them — `tests/target_hints.rs`
/// reads it and holds `atlas_core::arch::target_hint` to it, so the value has
/// to be right.
#[test]
fn every_inherited_hardware_toml_carries_the_same_key_set_as_gb10() {
    let gb10_path = gb10_dir().join("HARDWARE.toml");
    let gb10: toml::Value =
        toml::from_str(&std::fs::read_to_string(&gb10_path).expect("gb10 HARDWARE.toml"))
            .expect("valid TOML");
    let keys = |v: &toml::Value| -> std::collections::BTreeSet<String> {
        v.get("hardware")
            .and_then(|h| h.as_table())
            .expect("[hardware] table")
            .keys()
            .cloned()
            .collect()
    };
    for t in INHERITED {
        assert_eq!(
            keys(&hardware_toml(t.hw)),
            keys(&gb10),
            "kernels/{}/HARDWARE.toml must declare the same keys as the other \
             NVIDIA targets",
            t.hw
        );
    }
}

// ── (b) common/ — inherited from gb10 by relative symlink ──

/// ORACLE: `kernels/gb10/common`. Each inherited `common/` is that directory,
/// reachable — all 181 entries (171 `.cu`, 9 `.cuh` headers the `.cu` files
/// `#include`, and `KERNEL.toml`, which `build.rs` merges as the base layer of
/// every target's flags and `[modules]` overrides).
///
/// Unlike strix's curated 99, nothing is left out: these are NVIDIA targets
/// compiled by the same nvcc, so a file gb10 compiles is a file they must
/// compile, and a subset here would be an undocumented kernel drop.
#[test]
fn every_inherited_common_mirrors_every_gb10_common_file() {
    for t in INHERITED {
        let faults = mirror_faults(&hw_dir(t.hw).join("common"), &gb10_dir().join("common"));
        assert!(
            faults.is_empty(),
            "kernels/{}/common has drifted from kernels/gb10/common:\n  {}",
            t.hw,
            faults.join("\n  ")
        );
    }
}

/// The header files matter as much as the sources: `common/*.cuh` is what the
/// macro-declared kernels `#include`, and a missing one fails the compile of a
/// `.cu` that is itself present. Counted explicitly so a mirror that silently
/// became `.cu`-only is not read as "no faults".
#[test]
fn every_inherited_common_carries_the_headers_and_the_kernel_toml() {
    for t in INHERITED {
        let dir = hw_dir(t.hw).join("common");
        let mut by_ext = std::collections::BTreeMap::<String, usize>::new();
        for entry in std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .flatten()
        {
            let path = entry.path();
            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default();
            *by_ext.entry(ext).or_default() += 1;
        }
        assert!(
            dir.join("KERNEL.toml").exists(),
            "kernels/{}: KERNEL.toml is the flag base",
            t.hw
        );
        assert!(
            by_ext.get("cuh").copied().unwrap_or(0) > 0,
            "kernels/{}: no .cuh headers mirrored: {by_ext:?}",
            t.hw
        );
        assert!(
            by_ext.get("cu").copied().unwrap_or(0) > 100,
            "kernels/{}: suspiciously few .cu sources mirrored: {by_ext:?}",
            t.hw
        );
    }
}

// ── (c) the P0 model targets ──

/// Model directories under `kernels/<hw>`, by the rule `build.rs` uses to
/// expand `ATLAS_TARGET_MODEL=*`: a subdirectory that has a MODEL.toml.
fn model_dirs(hw: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(hw_dir(hw))
        .unwrap_or_else(|e| panic!("kernels/{hw}: {e}"))
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            e.path().join("MODEL.toml").exists().then_some(name)
        })
        .collect();
    names.sort();
    names
}

/// ORACLE: the model lists in [`INHERITED`], the campaign's declared set. A wildcard build
/// (`ATLAS_TARGET_MODEL=*`) compiles exactly the directories that carry a
/// MODEL.toml, so this set IS what an image for that hardware would serve.
#[test]
fn the_p0_model_targets_are_the_ones_declared() {
    for t in INHERITED {
        assert_eq!(model_dirs(t.hw), t.models, "kernels/{}", t.hw);
    }
}

/// Each MODEL.toml is a REAL file (behaviour and sampling are a per-target
/// decision, not something to inherit by link) whose `[model].name` matches
/// its directory — the invariant `resolve.rs` reports targets by.
#[test]
fn every_model_toml_is_a_real_file_naming_its_own_directory() {
    for t in INHERITED {
        for model in t.models {
            let path = hw_dir(t.hw).join(model).join("MODEL.toml");
            assert!(
                std::fs::read_link(&path).is_err(),
                "{}/{model}/MODEL.toml is a symlink; per-target behaviour must be \
                 editable for this hardware without moving gb10",
                t.hw
            );
            let toml: toml::Value = toml::from_str(&std::fs::read_to_string(&path).unwrap())
                .unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()));
            assert_eq!(
                toml.get("model")
                    .and_then(|m| m.get("name"))
                    .and_then(|v| v.as_str()),
                Some(*model),
                "kernels/{}/{model}",
                t.hw
            );
        }
    }
}

/// The copied MODEL.toml must say where it came from and what has NOT been
/// done. `[expected_absent]` is harvested per hardware (`spark serve
/// --check-kernels` on the real device) and these tables were harvested on
/// GB10 — carrying them over silently would let a kernel that is genuinely
/// missing on this hardware read as an expected absence.
#[test]
fn every_model_toml_records_its_inherited_provenance() {
    for t in INHERITED {
        for model in t.models {
            let path = hw_dir(t.hw).join(model).join("MODEL.toml");
            let text = std::fs::read_to_string(&path).unwrap();
            let head: String = text.lines().take(6).collect::<Vec<_>>().join("\n");
            assert!(
                head.contains(t.provenance),
                "{}/{model}/MODEL.toml does not open with the inheritance note \
                 {:?}:\n{head}",
                t.hw,
                t.provenance
            );
            assert!(
                head.contains("--check-kernels"),
                "{}/{model}/MODEL.toml does not say expected_absent is unharvested \
                 on this hardware",
                t.hw
            );
        }
    }
}

/// ORACLE: `kernels/gb10/<model>/nvfp4`. Same mirror rule as `common/`, per
/// model: every file individually symlinked, so a future hardware-tuned kernel
/// replaces ONE link instead of forking the whole directory.
///
/// An `nvfp4` dir even where the hardware has no NVFP4 datapath (Hopper): the
/// runtime's weight-format gate is that an nvfp4-built bundle also serves FP8
/// and BF16 checkpoints, so FP8 checkpoints run through these kernels. An
/// `fp8/` quant dir would be a second name for the same files.
#[test]
fn every_model_nvfp4_dir_mirrors_gb10() {
    for t in INHERITED {
        for model in t.models {
            // The 3.8 target retains its GB10 alias: only 3.6 owns a source
            // tree. Creating an nvfp4 directory here would fork that source.
            if t.hw == "hopper" && *model == "qwen3.8-27b" {
                assert!(!hw_dir(t.hw).join(model).join("nvfp4").exists());
                continue;
            }
            let faults = mirror_faults(
                &hw_dir(t.hw).join(model).join("nvfp4"),
                &gb10_dir().join(model).join("nvfp4"),
            );
            assert!(
                faults.is_empty(),
                "kernels/{}/{model}/nvfp4 has drifted from gb10's:\n  {}",
                t.hw,
                faults.join("\n  ")
            );
        }
    }
}

// ── (d) what build.rs would resolve ──

/// Mirror of `build.rs::resolve_targets` for the GPU-free runner, which never
/// reaches it: `ATLAS_SKIP_BUILD=1` returns from `main()` before target
/// resolution runs, so a hardware set can be structurally broken and every
/// existing test still passes.
///
/// Reproduces the parts that decide WHAT gets compiled — HARDWARE.toml `arch`,
/// the `MODEL.toml`-carrying subdirectory scan, the `[model] kernel_source`
/// redirect, and the default `nvfp4` quant — and returns `(model, quant, arch,
/// kernel dir)` per target.
fn resolved_targets(hw: &str, quant: &str) -> Vec<(String, String, String, PathBuf)> {
    let dir = hw_dir(hw);
    let arch = hardware_toml(hw)["hardware"]["arch"]
        .as_str()
        .unwrap()
        .to_string();
    model_dirs(hw)
        .into_iter()
        .map(|model| {
            let model_dir = dir.join(&model);
            let toml: toml::Value =
                toml::from_str(&std::fs::read_to_string(model_dir.join("MODEL.toml")).unwrap())
                    .unwrap();
            let src = toml
                .get("model")
                .and_then(|m| m.get("kernel_source"))
                .and_then(|v| v.as_str())
                .map(|s| dir.join(s))
                .unwrap_or(model_dir);
            let kernel_dir = src.join(quant);
            (model, quant.to_string(), arch.clone(), kernel_dir)
        })
        .collect()
}

/// A wildcard build of each inherited target resolves its declared targets, all
/// at that hardware's arch, each with a kernel directory that exists. This is
/// the assertion the real build would make on a CUDA host, made where CI can
/// actually run it.
#[test]
fn a_wildcard_build_resolves_declared_targets_at_the_declared_arch() {
    for t in INHERITED {
        let targets = resolved_targets(t.hw, "nvfp4");
        let names: Vec<&str> = targets.iter().map(|(m, ..)| m.as_str()).collect();
        assert_eq!(names, t.models, "kernels/{}", t.hw);
        for (model, quant, arch, kernel_dir) in &targets {
            assert_eq!(arch, t.arch, "{}/{model}", t.hw);
            assert_eq!(quant, "nvfp4", "{}/{model}", t.hw);
            assert!(
                kernel_dir.is_dir(),
                "{}/{model}: no kernel directory at {}",
                t.hw,
                kernel_dir.display()
            );
        }
    }
}

/// The first paid Hopper cell retains the GB10 3.8 -> 3.6 redirect within
/// Hopper. Every other target owns its source tree. An accidental cross-hardware
/// redirect must not bypass the inherited mirror and its hardware flags.
#[test]
fn inherited_redirects_resolve_only_to_the_declared_same_hardware_source() {
    for t in INHERITED {
        let targets = resolved_targets(t.hw, "nvfp4");
        assert!(!targets.is_empty(), "no {} targets resolved at all", t.hw);
        for (model, _, _, kernel_dir) in targets {
            let source = if t.hw == "hopper" && model == "qwen3.8-27b" {
                "qwen3.6-27b"
            } else {
                &model
            };
            assert_eq!(
                kernel_dir,
                hw_dir(t.hw).join(source).join("nvfp4"),
                "{}/{model}: kernel_source redirect changes where kernels come from",
                t.hw
            );
        }
    }
}
