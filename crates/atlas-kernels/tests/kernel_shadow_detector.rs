// SPDX-License-Identifier: AGPL-3.0-only

//! Shadow-drift detector pin.
//!
//! `build.rs` computes, per target, which `common/` kernels a model's
//! shadowing files DROP, and bakes the list into the binary so the startup
//! kernel audit can call a dropped kernel a build defect rather than a benign
//! absence. That machinery only works if the entry-point resolver can see the
//! entry points.
//!
//! It could not. It scanned for a literal `__global__`, and twenty-one
//! `kernels/gb10/common/*.cu` files declare their kernels through
//! `#define KERNEL_NAME` + `#include`, so they resolved to the EMPTY SET and
//! every comparison against them reported "drops nothing". That is the same
//! silence that hid the 27B's four missing multi-sequence GDN decode kernels
//! until 2026-07-26 — and because a build script's own `#[cfg(test)]` modules
//! are never run by `cargo test`, no test existed that could have caught it.
//!
//! So the resolver lives in `build_shadow.rs` and this integration test
//! compiles the same file. `cargo test` runs on a GPU-free runner with
//! `ATLAS_SKIP_BUILD=1`, where `build.rs` returns before the detector ever
//! runs — CI would otherwise have no coverage of this at all.

#[path = "../build_shadow.rs"]
mod build_shadow;

use build_shadow::{entry_points, shadowed_missing_symbols};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Hardware set -> kernel source extension. Mirrors `HW_SOURCE_EXT` in
/// `scripts/check_kernel_shadows.py` and `source_extension()` in
/// `build_target.rs`.
const HW_SOURCE_EXT: &[(&str, &str)] = &[
    ("b200", "cu"),
    ("gb10", "cu"),
    ("hopper", "cu"),
    ("metal", "metal"),
    ("strix", "cu"),
    ("strix-hip", "cu"),
];

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/atlas-kernels is two levels below the workspace root")
        .join("kernels")
}

/// Every `<file stem>` -> path in one directory with the given extension.
fn sources(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == ext))
        .collect();
    out.sort();
    out
}

/// THE REGRESSION GUARD. A file that binds its kernel name with a macro and
/// delegates the body to a header contains no literal `__global__` — a text
/// scan returns nothing for it. Both of these entry points must be resolved:
/// the header declares `KERNEL_NAME` and, through a token-pasting concat
/// macro, `KERNEL_NAME##_64`.
#[test]
fn macro_declared_kernels_resolve_through_define_and_include() {
    let file = kernels_root().join("gb10/common/inferspark_prefill_paged_fp8.cu");
    assert!(file.is_file(), "fixture moved: {}", file.display());
    let text = std::fs::read_to_string(&file).unwrap();
    assert!(
        !text.contains("__global__"),
        "{} now contains a literal __global__, so it no longer exercises the \
         macro path this test exists to pin — point the test at another \
         `#define KERNEL_NAME` file",
        file.display()
    );

    let found = entry_points(&file);
    let want: BTreeSet<String> = [
        "inferspark_prefill_paged_fp8",
        "inferspark_prefill_paged_fp8_64",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(
        found, want,
        "entry-point resolution regressed for a macro-declared kernel file"
    );
}

/// A shadow that keeps the primary kernel but drops the `_64` variant is drift,
/// and it is invisible to a `__global__` grep from either side: `common/` has
/// no literal `__global__` at all, so the grep's `common - model` difference is
/// empty regardless of what the shadow does.
#[test]
fn a_shadow_dropping_a_macro_declared_kernel_is_reported() {
    let dir = std::env::temp_dir().join(format!(
        "atlas-shadow-detector-{}-{}",
        std::process::id(),
        line!()
    ));
    let common = dir.join("common");
    let model = dir.join("model");
    std::fs::create_dir_all(&common).unwrap();
    std::fs::create_dir_all(&model).unwrap();

    std::fs::write(
        common.join("compute.cuh"),
        r#"
#define _CONCAT(a, b) a##b
#define CONCAT(a, b) _CONCAT(a, b)
extern "C" __global__ void KERNEL_NAME(const float* x) {}
extern "C" __global__ __launch_bounds__(128, 2) void CONCAT(KERNEL_NAME, _64)(const float* x) {}
"#,
    )
    .unwrap();
    std::fs::write(
        common.join("attn.cu"),
        "#define KERNEL_NAME atlas_attn\n#include \"compute.cuh\"\n",
    )
    .unwrap();
    // The fork kept the primary kernel and never picked up the BR=64 variant.
    std::fs::write(
        model.join("attn.cu"),
        "extern \"C\" __global__ void atlas_attn(const float* x) {}\n",
    )
    .unwrap();

    let dropped = shadowed_missing_symbols(&common.join("attn.cu"), &model.join("attn.cu"));
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        dropped,
        vec!["atlas_attn_64".to_string()],
        "the detector did not see the dropped macro-declared kernel — this is \
         exactly the shape a text scan for `__global__` reports as clean"
    );
}

/// No `common/` source may resolve to zero entry points. A file that declares
/// nothing is not necessarily wrong, but it is indistinguishable from a file
/// the resolver failed to understand — and "resolved nothing" is what made the
/// macro blind spot look like thoroughness for as long as it did. If a genuinely
/// kernel-free source lands in `common/`, this test is the place to say so
/// explicitly rather than letting the silence spread.
#[test]
fn every_common_source_declares_at_least_one_entry_point() {
    let root = kernels_root();
    let mut checked = 0usize;
    let mut empty = Vec::new();
    for (hw, ext) in HW_SOURCE_EXT {
        let mut hardware_checked = 0usize;
        for file in sources(&root.join(hw).join("common"), ext) {
            checked += 1;
            hardware_checked += 1;
            if entry_points(&file).is_empty() {
                empty.push(file.strip_prefix(&root).unwrap().display().to_string());
            }
        }
        assert!(
            hardware_checked > 0,
            "no {ext} common sources found for {hw}: wrong extension or root?"
        );
    }
    assert!(
        checked > 100,
        "only {checked} common sources found — wrong root?"
    );
    assert!(
        empty.is_empty(),
        "{} common source(s) resolve to no kernel entry point:\n  {}",
        empty.len(),
        empty.join("\n  ")
    );
}

/// One KERNEL.toml table, as `{ key: [values] }` or `{ key: "value" }`.
fn kernel_toml(dir: &Path) -> toml::Value {
    std::fs::read_to_string(dir.join("KERNEL.toml"))
        .ok()
        .and_then(|t| toml::from_str(&t).ok())
        .unwrap_or_else(|| toml::Value::Table(toml::map::Map::new()))
}

/// `[modules]` file-stem -> module-name overrides.
fn modules(v: &toml::Value, out: &mut BTreeMap<String, String>) {
    let Some(t) = v.get("modules").and_then(|m| m.as_table()) else {
        return;
    };
    for (stem, name) in t {
        if let Some(name) = name.as_str() {
            out.insert(stem.clone(), name.to_string());
        }
    }
}

/// `[shadow_exempt]` module -> kernels a shadow may omit.
fn shadow_exempt(v: &toml::Value, out: &mut BTreeSet<(String, String)>) {
    let Some(t) = v.get("shadow_exempt").and_then(|m| m.as_table()) else {
        return;
    };
    for (module, kernels) in t {
        for k in kernels.as_array().into_iter().flatten() {
            if let Some(k) = k.as_str() {
                out.insert((module.clone(), k.to_string()));
            }
        }
    }
}

/// The CPU-side gate: no model shadow may drop a kernel its `common/` namesake
/// declares, unless a KERNEL.toml `[shadow_exempt]` entry says so with a reason.
///
/// `build.rs` computes this same difference on a GPU box and bakes it into the
/// binary — but CI builds with `ATLAS_SKIP_BUILD=1`, which returns long before
/// the detector runs, so without this the drift class is caught only by whoever
/// happens to build kernels next.
///
/// `build.rs` is the SSOT for how `[modules]` and `[shadow_exempt]` are merged
/// (common first, then the model dir); this reads the same declarations from
/// the same files rather than restating their contents here, so an exemption
/// added or withdrawn there moves this test with it.
#[test]
fn no_model_shadow_drops_a_common_kernel() {
    let root = kernels_root();
    let mut pairs = 0usize;
    let mut drift = Vec::new();
    for (hw, ext) in HW_SOURCE_EXT {
        let hw_dir = root.join(hw);
        let common = hw_dir.join("common");
        if !common.is_dir() {
            continue;
        }
        let common_toml = kernel_toml(&common);
        let Ok(models) = std::fs::read_dir(&hw_dir) else {
            continue;
        };
        let mut model_dirs: Vec<PathBuf> = models
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir() && p.file_name().is_some_and(|n| n != "common"))
            .collect();
        model_dirs.sort();
        for model_dir in model_dirs {
            let Ok(quants) = std::fs::read_dir(&model_dir) else {
                continue;
            };
            let mut quant_dirs: Vec<PathBuf> = quants
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect();
            quant_dirs.sort();
            for quant_dir in quant_dirs {
                let model_toml = kernel_toml(&quant_dir);
                let mut module_of = BTreeMap::new();
                modules(&common_toml, &mut module_of);
                modules(&model_toml, &mut module_of);
                let mut exempt = BTreeSet::new();
                shadow_exempt(&common_toml, &mut exempt);
                shadow_exempt(&model_toml, &mut exempt);

                for shadow in sources(&quant_dir, ext) {
                    let namesake = common.join(shadow.file_name().unwrap());
                    if !namesake.is_file() {
                        continue; // model-only kernel, shadows nothing
                    }
                    pairs += 1;
                    let stem = shadow.file_stem().unwrap().to_string_lossy().to_string();
                    let module = module_of.get(&stem).cloned().unwrap_or(stem);
                    for kernel in shadowed_missing_symbols(&namesake, &shadow) {
                        if exempt.contains(&(module.clone(), kernel.clone())) {
                            continue;
                        }
                        drift.push(format!(
                            "{} drops {module}::{kernel}",
                            shadow.strip_prefix(&root).unwrap().display()
                        ));
                    }
                }
            }
        }
    }
    assert!(pairs > 0, "no shadow/common pairs found — wrong root?");
    assert!(
        drift.is_empty(),
        "{} shadowing file(s) drop a kernel their common/ namesake declares:\n  {}",
        drift.len(),
        drift.join("\n  ")
    );
}

/// Every hardware set in the tree is in `HW_SOURCE_EXT`.
///
/// The list is hand-maintained on both sides of the kernel-structure gate
/// (here and in `scripts/check_kernel_shadows.py`), and an unlisted hardware
/// set is not partially checked — it is not checked at all, silently, which is
/// the same shape as the macro blind spot this file was written for. A
/// directory with a HARDWARE.toml is a build target by `build.rs`'s own rule
/// (`resolve_targets` accepts any `kernels/<hw>` that has one), so that is the
/// oracle for what must be listed.
#[test]
fn every_hardware_set_in_the_tree_is_guarded() {
    let root = kernels_root();
    let mut unguarded: Vec<String> = std::fs::read_dir(&root)
        .expect("kernels/")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("HARDWARE.toml").is_file())
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .filter(|hw| !HW_SOURCE_EXT.iter().any(|(known, _)| known == hw))
        .collect();
    unguarded.sort();
    assert!(
        unguarded.is_empty(),
        "hardware set(s) not in HW_SOURCE_EXT, so no shadow or entry-point check \
         ever looks at them: {unguarded:?} — add them here AND in \
         scripts/check_kernel_shadows.py"
    );
}
